%% Compare Hebbian-LMS and Backpropagation
clear, clc

rng(0)

addpath f/                  % auxiliary functions folder

numHiddenLayers = 3;        % number of hidden layers
numNeuronsHL = 50;         % number of neurons in the hidden layers
dimInputVector = 50;       % dimensionality of input vector space
Nclusters = 100;            % number of clusters
Npatterns = 100;            % number of patterns per cluster
NdisturbClusters = 0;      % number of clusters not not included in training
Nrealizations = 1;

% Generate centroids whose coordinates are uniformly distributed in [-D, D] 
% of each dimension of the input vector space
D = 2;                      
rho = 0.5;                % ratio of standard deviation of centroids and standard deviation of the cluster points.
Omega = sqrt((2*D)^2/12); % standard deviation of centroids
sigma = rho*Omega;        % standard deviation of the cluster points.

Vcluster = pi^(dimInputVector/2)*(5*sigma)^dimInputVector/gamma(1 + dimInputVector/2);
Vspace = (2*D)^dimInputVector;

dimInputVector*Vcluster/Vspace

% Generate \Nclusters\ clusters with \Npatterns\ patterns per cluster
dataPartitioning = [0.01 0 0.99]; % 50% for training, 0% for validation, and 50% for testing
[Xtrain, Dtrain, C] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(1)*Npatterns, Omega, sigma, NdisturbClusters);
[Xval, Dval] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(2)*Npatterns, C, sigma);
[Xtest, Dtest] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(3)*Npatterns, C, sigma);

% Concatanate 
X = [Xtrain Xval Xtest];
D = [Dtrain Dval Dtest];

%
state = rng;
HLMSoriginal = NeuralNetwork(numHiddenLayers, numNeuronsHL, Nclusters, 1e-3); % 0.2e-3 for 20 patterns
HLMSoriginal.dataPartitioning = dataPartitioning;  % all for training
% Backpropagation
rng(state); % reset seed of RNG so that both networks have same initial conditions
BP = NeuralNetwork(numHiddenLayers, numNeuronsHL, Nclusters, 1e-3); % 0.5e-3 for sigmoid
BP.dataPartitioning = dataPartitioning;  % all for training

% Original Hebbian-LMS (HLMS)
rng('shuffle')
Wrange = [0.5 1 2];
for n = 1:Nrealizations
    state = rng;
    HLMSoriginal.reset(2*Wrange(n));
    rng(state)
    BP.reset(Wrange(n));

    %% Training
    tic
    disp('Hebbian-LMS-Original')
    HLMSoriginal.set_functions('sigmoid', 'sigmoid')
    HLMSoriginal.train(X, D, 'Hebbian-LMS', 50, true); % for sigmoid output layer
    % HLMSoriginal.train(X, D, 'Modified-Hebbian-LMS', 30, true); % for sigmoid output layer
    toc, tic
    disp('Backpropagation')
    BP.set_functions('sigmoid', 'softmax')
    BP.train(X, D, 'Backpropagation', 50, true); % for sigmoid output layer
    toc
end

% axis([1 50 0 5])

%% Compare with Matlab's neural network (output layer is softmax)
net = patternnet(numNeuronsHL*ones(1, 3));

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 1;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;
 
% Train the Network
[net,tr] = train(net,Xtrain, Dtrain);

% Test the Network
outputs = net(Xtest);

dval = outputs;
dout = zeros(size(dval));
for k = 1:size(dval, 2)
    [~, idx] = max(dval(:, k));
    dout(idx, k) = 1;
end

error_rate = sum(any(dout ~= Dtest, 1))/size(Dtest, 2)


ValNet = NeuralNetwork(numHiddenLayers, numNeuronsHL, Nclusters, 0.5e-3); % 0.5e-3 for sigmoid

ValNet.W{1} = net.IW{1};
ValNet.W{2} = net.LW{2, 1};
ValNet.W{3} = net.LW{3, 2};
ValNet.W{4} = net.LW{4, 3};
% ValNet.W{5} = net.LW{5, 4};
% ValNet.W{6} = net.LW{6, 5};
ValNet.b = net.b;

ValNet.test(Xtest, Dtest)
