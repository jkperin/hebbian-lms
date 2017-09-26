classdef Layer < handle
    %% Layer of a neural network
    properties
        numInputs % number of inputs per neuron
        numNeurons % number of neurons in layer
        W % weights
        b % bias weights
        gamma = 0.3 % slope in HLMS neuron
        outputFcn='sigmoid' % function of output layer neurons: {'linear', 'sigmoid', 'softmax'}
        Wstd=0.1; % standard deviation of distribution of weights
        S % output of the adders
        Y % neurons outputs
    end
            
    properties (Hidden)
        fO % handle for output function of output layer neurons
        errorO % handle for error function of output layer neurons
    end
    
    methods
        function obj = Layer(numInputs, numNeurons, Wstd)
            %% Constructor
            obj.numInputs = numInputs;
            obj.numNeurons = numNeurons;
            obj.set_functions;
            
            if exist('Wstd', 'var')
                obj.Wstd = Wstd;
            end
            obj = obj.init;
        end
                        
        function obj = init(obj)
            %% Reset weights of neural network            
            obj.W = obj.Wstd*(randn(obj.numNeurons, obj.numInputs)); % normally distributed with standard deviation Wstd
            obj.b = zeros(obj.numNeurons, 1);
            obj.S = zeros(obj.numNeurons, 1);
            obj.Y = zeros(obj.numNeurons, 1);
        end 
        
        function set_functions(self, Fcn)
            %% Get function for the hidden layers (fhl) and its deriviative (dfhl) and the function for the output layer (fo) and its derivatieve (dfo)           
            if exist('Fcn', 'var')
                self.outputFcn = Fcn;
            end
                        
            switch lower(self.outputFcn)
                case 'sigmoid'
                    % Important: signal for error calculation is measured
                    % prior to sigmoid i.e., it is the output of the sum
                    self.fO = @(x) self.sigmoid(x);
                    self.errorO = @(S, D) (D - S); %(d - self.S{end});
%                     self.errorO = @(S, D) (D - self.sigmoid(S)); % error computed after sigmoid
                case 'norm sigmoid'
                    self.fO = @(x) 1/N*self.sigmoid(x);
                    self.errorO = @(S, D) (D - 1/N*S); %(d - self.S{end});
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
                    error('Layer/set_functions: invalid layer function')
            end
        end
        
        function [train, valid, test] = train_supervised(self, Data, mu, Ncycles)
            %% Train layer neurons with inputs |X|, desired response |D|, and algorithm |algorithm|
            % Inputs:
            % Data: vector containing training, valdiation, and test data
            % structs
            % mu: adaptation constant
            % Ncycles: number of cycles
            
            self.set_functions(); % set hidden and output layers functions
                               
            train = Data{1};
            valid = Data{2};
            test = Data{3};
            
            n = 1;
            while n < Ncycles
                for k = 1:size(train.X, 2) % go through the data
                    self.supervised(train.X(:, k), train.d(:, k), mu) 
                end
  
                train.error(n+1) = self.test(train.X, train.d);
                fprintf('- Training cycle #%d\n', n)
                fprintf('Training error = %G\n', train.error(n+1))
                if not(isempty(valid))
                    valid.error(n+1) = self.test(valid.X, valid.d);
                    fprintf('Validation error = %G\n', valid.error(n+1))  
                end
                if not(isempty(test))
                    test.error(n+1) = self.test(test.X, test.d);
                    fprintf('Testing error = %G\n', test.error(n+1))  
                end   
                n = n + 1;
            end
            
            % Plot learning curve
            figure(1), hold on, box on
            hplot = plot(1:n-1, 100*train.error(2:n), 'LineWidth', 2, 'DisplayName', 'Training error');
            if not(isempty(valid))
                plot(1:n-1, 100*valid.error(2:n), ':', 'Color', get(hplot, 'Color'), 'LineWidth', 2, 'DisplayName', 'Validation error')
            end
            if not(isempty(test))
                plot(1:n-1, 100*test.error(2:n), '--', 'Color', get(hplot, 'Color'), 'LineWidth', 2, 'DisplayName', 'Testing error')
            end
            xlabel('Training cycles')
            ylabel('Error rate %')
            legend('-dynamiclegend')
            set(gca, 'FontSize', 12)
            drawnow
        end
        
        function train_HLMS(self, Xtrain, mu, Ncycles)
            %% Train layer neurons with inputs |X|, desired response |D|, and algorithm |algorithm|
            % Inputs:
            % Xtrain: trainig data
            % mu: adaptation constant
            % Ncycles: number of cycles
            
            self.set_functions(); % set hidden and output layers functions
            
            n = 1;
            while n < Ncycles
                for k = 1:size(Xtrain, 2) % go through the data
                    self.hebbian_lms(Xtrain(:, k), mu)
                end
                n = n + 1;
            end
        end
        
        function hebbian_lms(self, X, mu)
            %% Hebbian-LMS with original gradient estimate
            % Inputs:
            % - X: input vector    
            self.output(X); % calculate responses for input X
            
            delta = (self.Y - self.gamma*self.S);
            self.W = self.W + 2*mu*delta*X.';
            self.b = self.b + 2*mu*delta;    
        end
        
        function supervised(self, X, D, mu)
            %% Supervised learning
            % Inputs:
            % - X: input vector    
            % - D: Desired response
            self.output(X); % calculate responses for input X
            
            delta = self.errorO(self.S, D);
            self.W = self.W + 2*mu*delta*X.';
            self.b = self.b + 2*mu*delta;    
        end
        
        function [error_rate, outputs] = test(self, X, D)
            %% Test neural network with the target response D
            self.output(X);

            outputs = self.Y;
            decisions = zeros(size(outputs));
            for k = 1:size(outputs, 2)
                [~, idx] = max(outputs(:, k));
                decisions(idx, k) = 1;
            end

            error_rate = sum(any(decisions ~= D, 1))/size(D, 2);
        end
        
        function Y = output(self, X)
            %% Calculate layer output
            self.S = bsxfun(@plus, self.W*X, self.b); % bsxfun in case X is a matrix
            Y = self.fO(self.S);
            self.Y = Y;
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
    end
end
        