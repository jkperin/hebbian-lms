%% Compare Hebbian-LMS and Backpropagation
clear

addpath f/                  % auxiliary functions folder

numHiddenLayers = 3;        % number of hidden layers
numNeuronsHL = 150;         % number of neurons in the hidden layers
dimInputVector = 50;       % dimensionality of input vector space
Nclusters = 100;            % number of clusters
Npatterns = 100;            % number of patterns per cluster

% Generate centroids whose coordinates are uniformly distributed in [-D, D] 
% of each dimension of the input vector space
Omega = 1;                % standard deviation of centroids
rho = 0.1;                % ratio of standard deviation of centroids and standard deviation of the cluster points.
sigma = rho*Omega*sqrt(2*dimInputVector);        % standard deviation of the cluster points.

Vhyper_sphere = @(s, N) pi^(N/2)*(5*s)^N/gamma(1 + N/2);
Vcluster = Vhyper_sphere(sigma, dimInputVector);
Vspace = Vhyper_sphere(Omega, dimInputVector);
Vratio = Vcluster/Vspace

% Generate \Nclusters\ clusters with \Npatterns\ patterns per cluster
dataPartitioning = [0.5 0 0.5]; % 20% for training, 0% for validation, and 80% for testing
[Xtrain, Dtrain, C, Cidx] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(1)*Npatterns, Omega, sigma);
[Xval, Dval] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(2)*Npatterns, C, sigma);
[Xtest, Dtest] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(3)*Npatterns, C, sigma);

Dist = pdist(C.', 'euclidean');
meanDist = min(Dist)
newRho = sigma/meanDist

% Concatanate 
X = [Xtrain Xval Xtest];
D = [Dtrain Dval Dtest];

X = X - mean(X, 2);
X = X./std(X, 0, 2);

%
state = rng;
%% HLMS
HLMSoriginal = NeuralNetwork(numHiddenLayers, numNeuronsHL, Nclusters); % 0.2e-3 for 20 patterns
% HLMSoriginal.gamma = 0.3;
HLMSoriginal.dataPartitioning = dataPartitioning;  % all for training
%% Backpropagation
rng(state); % reset seed of RNG so that both networks have same initial conditions
BP = NeuralNetwork(numHiddenLayers, numNeuronsHL, Nclusters); % 0.5e-3 for sigmoid
BP.dataPartitioning = dataPartitioning;  % all for training

% Original Hebbian-LMS (HLMS)
rng('shuffle')
Wrange = 0.2;
for n = 1:length(Wrange)
    state = rng;
    HLMSoriginal.reset(Wrange(n));
    rng(state)
    BP.reset(0.1);

    %% Training
    tic
    disp('Hebbian-LMS-Original')
    HLMSoriginal.set_functions('sigmoid', 'softmax')
    HLMSoriginal.train(X, D, 'Hebbian-LMS', 1e-3, 20); % for sigmoid output layer
    [ConsisHLMS, Nwords] = HLMSoriginal.consistency(Xtrain, Cidx, C, true)

    toc, tic
    disp('Backpropagation')
    BP.set_functions('sigmoid', 'softmax')
    BP.train(X, D, 'Backpropagation', 1e-3, 20); % for sigmoid output layer
    toc
end

% axis([1 50 0 5])

% %% Compare with Matlab's neural network (output layer is softmax)
% net = patternnet(numNeuronsHL*ones(1, 3));
% 
% % Set up Division of Data for Training, Validation, Testing
% net.divideParam.trainRatio = 1;
% net.divideParam.valRatio = 0;
% net.divideParam.testRatio = 0;
%  
% % Train the Network
% [net,tr] = train(net,Xtrain, Dtrain);
% 
% % Test the Network
% outputs = net(Xtest);
% 
% dval = outputs;
% dout = zeros(size(dval));
% for k = 1:size(dval, 2)
%     [~, idx] = max(dval(:, k));
%     dout(idx, k) = 1;
% end
% 
% error_rate = sum(any(dout ~= Dtest, 1))/size(Dtest, 2)
% 
% 
% ValNet = NeuralNetwork(numHiddenLayers, numNeuronsHL, Nclusters, 0.5e-3); % 0.5e-3 for sigmoid
% 
% ValNet.W{1} = net.IW{1};
% ValNet.W{2} = net.LW{2, 1};
% ValNet.W{3} = net.LW{3, 2};
% ValNet.W{4} = net.LW{4, 3};
% % ValNet.W{5} = net.LW{5, 4};
% % ValNet.W{6} = net.LW{6, 5};
% ValNet.b = net.b;
% 
% ValNet.test(Xtest, Dtest)
