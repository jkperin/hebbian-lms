%% Compare Hebbian-LMS and Backpropagation
clear, clc, close all

addpath f/                  % auxiliary functions folder

numHiddenLayers = 3;        % number of hidden layers
numNeuronsHL = 50;         % number of neurons in the hidden layers
dimInputVector = 50;       % dimensionality of input vector space
Nclusters = 50;            % number of clusters
Npatterns = 20;            % number of patterns per cluster
NdisturbClusters = 3;      % number of clusters not not included in training

% Generate centroids whose coordinates are uniformly distributed in [-D, D] 
% of each dimension of the input vector space
D = 2;                      
rho = 0.2;                % ratio of standard deviation of centroids and standard deviation of the cluster points.
Omega = sqrt((2*D)^2/12); % standard deviation of centroids
sigma = rho*Omega;        % standard deviation of the cluster points.

% Generate \Nclusters\ clusters with \Npatterns\ patterns per cluster
dataPartitioning = [0.5 0 0.5]; % 50% for training, 0% for validation, and 50% for testing
[Xtrain, Dtrain, C] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(1)*Npatterns, Omega, sigma, NdisturbClusters);
[Xval, Dval] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(2)*Npatterns, C, sigma);
[Xtest, Dtest] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(3)*Npatterns, C, sigma);

% Concatanate 
X = [Xtrain Xval Xtest];
D = [Dtrain Dval Dtest];

% Original Hebbian-LMS (HLMS)
seed = rng;
HLMSoriginal = NeuralNetwork(numHiddenLayers, numNeuronsHL, Nclusters, 10e-3);
HLMSoriginal.dataPartitioning = dataPartitioning;  % all for training
% Backpropagation
rng(seed); % reset seed of RNG so that both networks have same initial conditions
BP = NeuralNetwork(numHiddenLayers, numNeuronsHL, Nclusters, 1e-3); % 0.5e-3 for sigmoid
BP.dataPartitioning = dataPartitioning;  % all for training

% Choose output layer
output_layer_fun = 'softmax';
Dtrain = D;
if strcmpi(output_layer_fun, 'sigmoid')
    disp('Using sigmoid function in output layer (on-out-of-many code)')
    Dtrain = 2*D - 1;  % make outputs {-1, 1}
else
    disp('Using softmax function at output layer')
end

%% Training
tic
disp('Hebbian-LMS-Original')
HLMSoriginal.set_functions('sigmoid', output_layer_fun)
HLMSoriginal.train(X, Dtrain, 'Hebbian-LMS', 0.01, true); % for sigmoid output layer
toc, tic
disp('Backpropagation')
BP.set_functions('sigmoid', output_layer_fun)
BP.train(X, Dtrain, 'Backpropagation', 0.01, true); % for sigmoid output layer
toc

%% Compare with Matlab's neural network (output layer is softmax)
% net = patternnet(hiddenLayerSize*ones(1, 3));
% 
% % Set up Division of Data for Training, Validation, Testing
% net.divideParam.trainRatio = 100/100;
% net.divideParam.valRatio = 0;
% net.divideParam.testRatio = 0;
%  
% % Train the Network
% [net,tr] = train(net,X, D);
% 
% % Test the Network
% outputs = net(X);
% 
% dout = double(outputs > 0.5);
% error_rate = sum(any(dout ~= D, 1))/size(D, 2)

