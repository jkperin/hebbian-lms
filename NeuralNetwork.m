classdef NeuralNetwork < handle
    %% Neural network for pattern classification problems
    properties
        numHiddenLayers % number of hidden layers
        numNeuronsHL % number of neurons in each hidden layer
        numNeuronsOut % number of neurons in the output layer
        W % weights for the entire network
        b % bias weights for the entire network
        S % output each hidden layer
        Y % output each hidden layer
        gamma = 0.3 % Gamma parameter of Hebbian-LMS neuron
        outputFcn='softmax' % function of output layer neurons: {'linear', 'sigmoid', 'softmax'} 
        hiddenFcn='sigmoid' % function of hidden layers neurons: {'sigmoid', 'rectifier'}
        dataPartitioning = [0.7 0 0.3] % how much is allocated for training, validation, and testing
    end   
    
    properties (Hidden)
        fHL % handle for output function of hidden layer neurons
        dfHL % handle for first derivative of output function of hidden layer neurons
        fO % handle for output function of output layer neurons
        errorO % handle for error function of output layer neurons
    end
    
    properties (Constant, Hidden)
        minTol = 1e-9;
        maxNcycles = 1e3;
    end
      
    methods
        function obj = NeuralNetwork(numHiddenLayers, numNeuronsHL, numNeuronsOut)
            %% Constructor: 
            % - numHiddenLayers: number of hidden layers
            % - numNeuronsHL: number of neurons in hidden layers
            % - numNeuronsOut: number of neurons in output layer
            obj.numHiddenLayers = numHiddenLayers;
            obj.numNeuronsHL = numNeuronsHL;
            obj.numNeuronsOut = numNeuronsOut;
                        
            obj.reset(); % initialize all variables
            
            obj.set_functions();
        end
        
        function [train, valid, test] = train(self, X, D, algorithm, adaptationConstant, varargin)
            %% Train neural network with inputs |X|, desired response |D|, and algorithm |algorithm|
            % Inputs:
            % - X: input matrix. Each column of X is a input vector
            % - D: desired response matrix. Each column of D is a desired
            % output vector
            % - algorithm: either 'backpropagation', 'hebbian-lms', or
            % 'hebbian-lms-modified'
            % - adaptationConstant: adaptation constant
            % - varargin{1}: has two meanings. If varargin < 1, varargin is
            % interpreted as the desired training error rate. If varargin
            % >= 1, varargin is interpreted as the number of training
            % cycles i.e., how many times we go through the data (X, D).
            % - varargin{2} (optional, default = true): whether to plot
            % learning curve at the end of trainig
            
            self.set_functions(); % set hidden and output layers functions
            
            % Partition the data in training, validation, and testing sets
            [train, valid, test] = self.partition_data(X, D); % Partitioning is not random
            
            if varargin{1} < 1 % interpret varargin as minimum error rate for training
%                 min_error = varargin{1};
                condition = @(x, n) x > min_error && n < self.maxNcycles;
%                 condition = @(x, n) n < self.maxNcycles;
                Ncycles = Inf;
            else % interpret varargin as maximum number of training cycles
                Ncycles = varargin{1}; 
                condition = @(x, n) n < Ncycles;
            end

            if length(adaptationConstant) == 1
                adaptationConstant = adaptationConstant*ones(1, self.numHiddenLayers+1);
            end

            % Select training algorithm
            switch lower(algorithm)
                case 'backpropagation'
                    trainingAlg = @(net, X, D) backpropagation(net, X, D, adaptationConstant);
                case 'hebbian-lms'
                    trainingAlg = @(net, X, D) hebbian_lms(net, X, D, adaptationConstant);
                case 'modified-hebbian-lms'
                    trainingAlg = @(net, X, D) hebbian_lms_modified(net, X, D, adaptationConstant);
                otherwise
                    error('NeuralNetwork: invalid training algorithm')
            end
            
            % Train
            train.error = ones(1, min(self.maxNcycles, Ncycles));
            valid.error = ones(1, min(self.maxNcycles, Ncycles));
            test.error = ones(1, min(self.maxNcycles, Ncycles));
            n = 1;
            while condition(train.error(n), n) 
                for k = 1:size(train.X, 2) % go through the data
                    trainingAlg(self, train.X(:, k), train.d(:, k)) 
                end

                train.error(n+1) = self.test(train.X, train.d);
                fprintf('- Training cycle #%d\n', n)
                fprintf('Training error = %G\n', train.error(n+1))
                if self.dataPartitioning(2) ~= 0
                    valid.error(n+1) = self.test(valid.X, valid.d);
                    fprintf('Validation error = %G\n', valid.error(n+1))  
                end
                if self.dataPartitioning(3) ~= 0
                    test.error(n+1) = self.test(test.X, test.d);
                    fprintf('Testing error = %G\n', test.error(n+1))  
                end                        
                n = n + 1;
            end
            
            % Plot learning curve
            if (length(varargin) == 1 && n > 2) || (length(varargin) == 2 && varargin{2} && n > 2) % whether to plot learning curve
                figure(1), hold on, box on
                hplot = plot(1:n-1, 100*train.error(2:n), 'LineWidth', 2, 'DisplayName', sprintf('%s: Training error', algorithm));
                if self.dataPartitioning(2) ~= 0
                    plot(1:n-1, 100*valid.error(2:n), ':', 'Color', get(hplot, 'Color'), 'LineWidth', 2, 'DisplayName', sprintf('%s: Validation error', algorithm))
                end
                if self.dataPartitioning(3) ~= 0
                    plot(1:n-1, 100*test.error(2:n), '--', 'Color', get(hplot, 'Color'), 'LineWidth', 2, 'DisplayName', sprintf('%s: Testing error', algorithm))
                end
                xlabel('Training cycles')
                ylabel('Error rate %')
                legend('-dynamiclegend')
                set(gca, 'FontSize', 12)
                drawnow
            end
        end

        function [train, valid, test] = train_sequentially(self, X, D, Ncycles)
            %% Train neural network with inputs |X|, desired response |D|, and algorithm |algorithm|
            % Inputs:
            % - X: input matrix. Each column of X is a input vector
            % - D: desired response matrix. Each column of D is a desired
            % output vector
            % - algorithm: either 'backpropagation', 'hebbian-lms', or
            % 'hebbian-lms-modified'
            % - varargin{1}: has two meanings. If varargin < 1, varargin is
            % interpreted as the desired training error rate. If varargin
            % >= 1, varargin is interpreted as the number of training
            % cycles i.e., how many times we go through the data (X, D).
            % - varargin{2} (optional, default = true): whether to plot
            % learning curve at the end of trainig
            
            self.set_functions(); % set hidden and output layers functions
            
            % Partition the data in training, validation, and testing sets
            [train, valid, test] = self.partition_data(X, D); % Partitioning is not random
              
            disp('first layer')
            for k = 1:Ncycles
                for pattern = 1:size(train.X, 2)
                    self.forward(train.X(:, pattern)); % calculate responses for input X

                    % Updates first layer
                    Xin = [train.X(:, pattern); zeros(self.numNeuronsHL-size(X, 1), 1)];
                    delta = (self.Y{1} - self.gamma*self.S{1});
                    self.W{1} = self.W{1} + 2*self.mu*delta*Xin.';
                    self.b{1} = self.b{1} + 2*self.mu*delta;   
                end
            end
            disp('hidden layers')
            for layer = 2:self.numHiddenLayers % train hidden layers
                for k = 1:Ncycles
                    for pattern = 1:size(train.X, 2)
                        self.forward(train.X(:, pattern)); % calculate responses for input X

                        % Updates first layer
                        delta = (self.Y{layer} - self.gamma*self.S{layer});
                        self.W{layer} = self.W{layer} + 2*self.mu*delta*self.Y{layer-1}.';
                        self.b{layer} = self.b{layer} + 2*self.mu*delta;       
                    end
                end
            end
                
            disp('train output layer')
            for k = 1:Ncycles
                for pattern = 1:size(train.X, 2)
                    self.forward(train.X(:, pattern)); % calculate responses for input X

                    % Updates output layer
                    delta = self.errorO(self.S{end}, train.d(:, pattern)); 
                    self.W{end} = self.W{end} + 2*self.mu*delta*self.Y{end-1}.';
                    self.b{end} = self.b{end} + 2*self.mu*delta;       

                    train.error(k) = self.test(train.X, train.d);
                    test.error(k) = self.test(test.X, test.d);

                    fprintf('- Training cycle #%d\n', k)
                    fprintf('Training error = %G\n', train.error(k))    
                    test.error(k) = self.test(test.X, test.d);
                    fprintf('Testing error = %G\n', test.error(k))  
                end
            end
            
            % Plot learning curve
            figure(1), hold on, box on
            hplot = plot(1:n-1, 100*train.error(2:n), 'LineWidth', 2, 'DisplayName', sprintf('%s: Training error', algorithm));
            if self.dataPartitioning(2) ~= 0
                plot(1:n-1, 100*valid.error(2:n), ':', 'Color', get(hplot, 'Color'), 'LineWidth', 2, 'DisplayName', sprintf('%s: Validation error', algorithm))
            end
            if self.dataPartitioning(3) ~= 0
                plot(1:n-1, 100*test.error(2:n), '--', 'Color', get(hplot, 'Color'), 'LineWidth', 2, 'DisplayName', sprintf('%s: Testing error', algorithm))
            end
            xlabel('Training cycles')
            ylabel('Error rate %')
            legend('-dynamiclegend')
            set(gca, 'FontSize', 12)
            drawnow
        end
        
        
        function hebbian_lms_modified(self, X, D, mu)
            %% Hebbian-LMS with modified gradient estimate
            % Inputs:
            % - X: input vector
            % - D: desired response
            % - mu: adaptation constant vector. mu(i) is the adaptation
            % constant of the ith layer.
            self.forward(X); % calculate responses for input X
            
            % Updates first layer
            X = [X; zeros(self.numNeuronsHL-size(X, 1), 1)];
            delta = -(self.Y{1} - self.gamma*self.S{1}).*(self.dfHL(self.S{1})  - self.gamma);
            self.W{1} = self.W{1} + 2*mu(1)*delta*X.';
            self.b{1} = self.b{1} + 2*mu(1)*delta;    
            
            % Remaining hidden layers
            for layer = 2:self.numHiddenLayers
                delta = -(self.Y{layer} - self.gamma*self.S{layer}).*(self.dfHL(self.S{layer})  - self.gamma);
                self.W{layer} = self.W{layer} + 2*mu(layer)*delta*self.Y{layer-1}.';
                self.b{layer} = self.b{layer} + 2*mu(layer)*delta;                 
            end
            
            % Output layer
            delta = self.errorO(self.S{end}, D); 
            self.W{end} = self.W{end} + 2*mu(end)*delta*self.Y{end-1}.';
            self.b{end} = self.b{end} + 2*mu(end)*delta;
        end
        
        function hebbian_lms(self, X, D, mu)
            %% Hebbian-LMS with original gradient estimate
            % Inputs:
            % - X: input vector
            % - D: desired response      
            % - mu: adaptation constant vector. mu(i) is the adaptation
            % constant of the ith layer.
            self.forward(X); % calculate responses for input X
            
            % Updates first layer
            X = [X; zeros(self.numNeuronsHL-size(X, 1), 1)];
            delta = (self.Y{1} - self.gamma*self.S{1});
            self.W{1} = self.W{1} + 2*mu(1)*delta*X.';
            self.b{1} = self.b{1} + 2*mu(1)*delta;    
            
            % Remaining hidden layers
            for layer = 2:self.numHiddenLayers
                delta = (self.Y{layer} - self.gamma*self.S{layer});
                self.W{layer} = self.W{layer} + 2*mu(layer)*delta*self.Y{layer-1}.';
                self.b{layer} = self.b{layer} + 2*mu(layer)*delta;                 
            end
            
            % Output layer
            delta = self.errorO(self.S{end}, D); 
            self.W{end} = self.W{end} + 2*mu(end)*delta*self.Y{end-1}.';
            self.b{end} = self.b{end} + 2*mu(end)*delta;
        end
        
        function backpropagation(self, X, D, mu)
            %% Backpropagation algorithm
            [~, Sout] = self.forward(X); 
                        
            % Update output layer
            delta = cell(self.numHiddenLayers+1, 1);
            delta{end} = self.errorO(Sout, D); 
            self.W{end} = self.W{end} + 2*mu(end)*delta{end}*self.Y{end-1}.';
            self.b{end} = self.b{end} + 2*mu(end)*delta{end};
            
            % Update hidden layers with the exception of the first one
            for layer = self.numHiddenLayers:-1:2
                delta{layer} = (self.W{layer+1}.'*delta{layer+1}).*self.dfHL(self.S{layer});
                self.W{layer} = self.W{layer} + 2*mu(layer)*delta{layer}*self.Y{layer-1}.';
                self.b{layer} = self.b{layer} + 2*mu(layer)*delta{layer};
            end
            
            % Update first layer
            delta{1} = (self.W{2}.'*delta{2}).*self.dfHL(self.S{1});
            X = [X; zeros(self.numNeuronsHL-size(X, 1), 1)];
            self.W{1} = self.W{1} + 2*mu(1)*delta{1}*X.';
            self.b{1} = self.b{1} + 2*mu(1)*delta{1};            
        end
        
        function [error_rate, outputs] = test(self, X, D)
            %% Test neural network with the target response D
            outputs = zeros(size(D));
            for k = 1:size(X, 2)
                outputs(:, k) = self.forward(X(:, k));
            end        

            decisions = zeros(size(outputs));
            for k = 1:size(outputs, 2)
                [~, idx] = max(outputs(:, k));
                decisions(idx, k) = 1;
            end

            error_rate = sum(any(decisions ~= D, 1))/size(D, 2);
        end
        
        function [y, Sout] = forward(self, X, verbose)
            %% Calculate outputs of the neural network (forward propagation)
            X = [X; zeros(self.numNeuronsHL-size(X, 1), 1)];
            
            % First layer
            self.S{1} = self.W{1}*X + self.b{1};
            self.Y{1} = self.fHL(self.S{1});
            
            % Remaining hidden layers
            for layer = 2:self.numHiddenLayers
                self.S{layer} = self.W{layer}*self.Y{layer-1} + self.b{layer};
                self.Y{layer} = self.fHL(self.S{layer});
            end
            
            % Output layer
            self.S{end} = self.W{end}*self.Y{end-1} + self.b{end};
            self.Y{end} = self.fO(self.S{end});
            Sout = self.S{end};
            y = self.Y{end};
            
            if exist('verbose', 'var') && verbose
                self.plot_outputs();
            end
        end
        
        function [Consis, Nwords] = consistency(self, X, Cidx, C, verbose)
            %% Calculate the consistency of each layer. Consistency is defined as the number of distinct binary words per cluster
            % Inputs:
            % - X: input patterns
            % - Cidx: cluster that each input pattern belongs to
            % - C: centroids
            % - verbose (optional, default=false): whether to plot results
            % Outputs
            % - Consis: consistency of each cluster i.e., how many distinct
            % binary words represent that cluster
            % - Nwords: number of unique binary words at the output of each
            % layer
            X = [X; zeros(self.numNeuronsHL-size(X, 1), size(X, 2))];
                 
            Nclusters = max(Cidx);
            Consis = zeros(Nclusters, self.numHiddenLayers);
            Nwords = zeros(1, self.numHiddenLayers);
            
            % First layer
            Stemp{1} = bsxfun(@plus, self.W{1}*X, self.b{1});
            Ytemp{1} = self.fHL(Stemp{1});
            Dtemp{1} = sign(Ytemp{1});
            Consis(:, 1) = calc_layer_consistency(Cidx, Dtemp{1}, Nclusters);
            uniqueBwords = unique(Dtemp{1}.', 'rows');
            hammingDistance{1} = hist(self.numNeuronsHL*pdist(uniqueBwords, 'hamming'), 0:self.numNeuronsHL);
            Nwords(1) = size(uniqueBwords, 1);
            
            % Remaining hidden layers
            for layer = 2:self.numHiddenLayers
                Stemp{layer} = bsxfun(@plus, self.W{layer}*Ytemp{layer-1}, self.b{layer});
                Ytemp{layer} = self.fHL(Stemp{layer});
                Dtemp{layer} = sign(Ytemp{layer});
                Consis(:, layer) = calc_layer_consistency(Cidx, Dtemp{layer}, Nclusters);
                uniqueBwords = unique(Dtemp{layer}.', 'rows');
                hammingDistance{layer} = hist(self.numNeuronsHL*pdist(uniqueBwords, 'hamming'), 0:self.numNeuronsHL);
                Nwords(layer) = size(uniqueBwords, 1);
            end
            
            % Overlaps
            for b = 1:size(uniqueBwords, 1)
                idx = all(bsxfun(@eq, Dtemp{end}, uniqueBwords(b, :).')); % indeces of patterns with uniqueBwords(n, :) binary word
                uniqueCidx = unique(Cidx(idx)); % cluster indeces that have binary word uniqueBwords(n, :)
                if numel(uniqueCidx) > 1 % more than one cluster with same binary word
                    table(categorical(uniqueCidx.'), Consis(uniqueCidx, end),...
                        'VariableNames',{'Cluster' 'diff_binary_words'})               
                    fprintf('Relative distances:\n')
                    dists = pdist(C(:, uniqueCidx).', 'euclidean') % euclidean distances between clusters that share the same binary word
                end 
            end
            
            
            if exist('verbose', 'var') && verbose
                figure(112)
                for layer = 1:self.numHiddenLayers
                    subplot(self.numHiddenLayers, 1, layer), hold on, box on
                    stem(1:Nclusters, Consis(:, layer), 'fill')
                    xlabel('Cluster', 'FontSize', 12)
                    ylabel('Distinct binary words', 'FontSize', 12)
                    title(sprintf('Layer: %d', layer))
                end
                
                figure(113), hold on, box on
                for layer = 1:self.numHiddenLayers
                    subplot(self.numHiddenLayers, 1, layer), hold on, box on
                    stem(0:self.numNeuronsHL, hammingDistance{layer}, 'fill')
                    xlabel('Hamming distance', 'FontSize', 12)
                    ylabel('# diff. binary words', 'FontSize', 12)
                    title(sprintf('Layer: %d', layer))
                end
                
                drawnow
            end
                        
            function [Consis, Bwords] = calc_layer_consistency(C, D, Nclusters)
                Consis = zeros(Nclusters, 1);
                Bwords = cell(Nclusters, 1);                
                for n = 1:length(C)
                    if isempty(Bwords{C(n)})
                        Consis(C(n)) = Consis(C(n)) + 1;
                        Bwords{C(n)} = D(:, n);
                    else
                        if any(all(bsxfun(@eq, Bwords{C(n)}, D(:, n))))
                            continue;
                        else
                            Consis(C(n)) = Consis(C(n)) + 1;
                            Bwords{C(n)} = [Bwords{C(n)} D(:, n)]; 
                        end
                    end
                end
            end
        end
           
        function set_functions(self, hlFcn, oFcn)
            %% Get function for the hidden layers (fhl) and its deriviative (dfhl) and the function for the output layer (fo) and its derivatieve (dfo)
            if exist('hlFcn', 'var')
                self.hiddenFcn = hlFcn;
            end
            
            if exist('oFcn', 'var')
                self.outputFcn = oFcn;
            end
            
            switch lower(self.hiddenFcn)
                case 'sigmoid'
                    self.fHL = @(x) self.sigmoid(x);
                    self.dfHL = @(x) self.dsigmoid(x);
                case 'sign' % only for HLMS
                    self.fHL = @(x) sign(x);
                    self.dfHL = @(x) 0;
                case 'rectifier'
                    self.fHL = @(x) max(x, 0);
                    self.dfHL = @(x) (sign(x)+1)/2;
                otherwise
                    error('NeuralNetwork/get_functions: invalid hidden layer function')
            end
            
            switch lower(self.outputFcn)
                case 'sigmoid'
                    % Important: signal for error calculation is measured
                    % prior to sigmoid i.e., it is the output of the sum
                    self.fO = @(x) self.sigmoid(x);
                    self.errorO = @(S, D) (D - S); %(d - self.S{end});
%                     self.errorO = @(S, D) (D - self.sigmoid(S)); % error computed after sigmoid
                case 'softmax' % loss function is cross-entropy
                    self.fO = @(x) exp(x)/sum(exp(x));
                    self.errorO  = @(S, D) (D - exp(S)/sum(exp(S))); % Note: D(i) in this case represents 1{y == i}
                case 'linear'
                    self.fO = @(x) x;
                    self.errorO = @(S, D) (D - S); %(d - self.S{end}); 
                case 'unsupervised-sigmoid'
                    self.fO = @(x) self.sigmoid(x);
                    self.errorO = @(S, ~) (self.unsupervised_sigmoid(S) - S);
                otherwise
                    error('NeuralNetwork/get_functions: invalid output layer function')
            end
        end
        
        function D = unsupervised_sigmoid(~, S)
            %% Unsupervised sigmoid: Return 1 for the maximum output, and zero to all others
            [~, idx] = max(S);
            D = zeros(size(S));
            D(idx) = 1;
        end
        
        function y = sigmoid(~, x)
            %% Sigmoid function from -1 to 1 with slope 1 about x = 0
            y = (1 - exp(-2*x))./(1 + exp(-2*x)); % = tansig(x);
%             y = 1./(1 + exp(-2*x)); 
        end
        
        function y = dsigmoid(~, x)
            %% First derivative of self.sigmoid()
            y = 4*exp(2*x)./(1 + exp(2*x)).^2;
%             y = 2*exp(2*x)./(1 + exp(2*x)).^2;
        end 
        
        function [train, valid, test] = partition_data(self, X, d)
            %% Partition data into trainig, validation, and testing
            % Partition ratios are given by the property dataPartitioning
            % Patitioning is not random. To partition the data randomly, it
            % is necessary to scramble X and d first
            N = size(X, 2);
            
            % Training
            train.idx = 1:self.dataPartitioning(1)*N;
            train.X = X(:, train.idx);
            train.d = d(:, train.idx);
            idxend = train.idx(end);
            
            % Validation
            if self.dataPartitioning(2) == 0
                valid = [];
            else
                valid.idx = idxend + (1:self.dataPartitioning(2)*N);
                valid.X = X(:, valid.idx);
                valid.d = d(:, valid.idx);
                idxend = valid.idx(end);
            end
            
            % Testing
            if self.dataPartitioning(3) == 0
                test = [];
            else
                test.idx = idxend + (1:self.dataPartitioning(3)*N);
                test.X = X(:, test.idx);
                test.d = d(:, test.idx);
            end
        end
        
        function obj = reset(obj, Wrange)
            %% Reset weights of neural network            
            if not(exist('Wrange', 'var'))
                Wrange = 1;
            end
            obj.W = cell(obj.numHiddenLayers+1, 1);
            obj.b = cell(obj.numHiddenLayers+1, 1);
            obj.S = cell(obj.numHiddenLayers+1, 1);
            obj.Y = cell(obj.numHiddenLayers+1, 1);
            for k = 1:obj.numHiddenLayers
                obj.W{k} = Wrange*(rand(obj.numNeuronsHL)-0.5); % uniformly distributed in [-Wrange/2, Wrange/2]
                obj.b{k} = zeros(obj.numNeuronsHL, 1);
                obj.S{k} = zeros(obj.numNeuronsHL, 1);
                obj.Y{k} = zeros(obj.numNeuronsHL, 1);
            end
            obj.W{obj.numHiddenLayers+1} = Wrange*(rand(obj.numNeuronsOut, obj.numNeuronsHL)-0.5); %  uniformly distributed in [-Wrange/2, Wrange/2]
            obj.b{obj.numHiddenLayers+1} = zeros(obj.numNeuronsOut, 1);
            obj.S{obj.numHiddenLayers+1} = zeros(obj.numNeuronsOut, 1);
            obj.Y{obj.numHiddenLayers+1} = zeros(obj.numNeuronsOut, 1);
        end  
        
        function plot_outputs(self)
            %% Plot output of all summers and all ouptut layer functions as an image.
            Smat = [];
            Ymat = [];
            for k = 1:length(self.S)-1
                Smat = [Smat self.S{k}];
                Ymat = [Ymat self.Y{k}];
            end
            
            figure(101), image(Smat, 'CDataMapping','scaled')
            title('Adders output')
            xlabel('Hidden layer')
            figure(102), image(Ymat, 'CDataMapping','scaled')
            title('Sigmoids output')
            xlabel('Hidden layer')
        end
    end
end
    