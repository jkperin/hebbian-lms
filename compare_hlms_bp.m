%% Compare Hebbian-LMS and Backpropagation
clear, clc, close all

numHiddenLayers = 3;        % number of hidden layers
numNeuronsHL = 100;         % number of neurons in the hidden layers
dimInputVector = 100;       % dimensionality of input vector space
Nclusters = 105;            % number of clusters
Npatterns = 100;            % number of patterns per cluster

% Generate centroids whose coordinates are uniformly distributed in [-D, D] 
% of each dimension of the input vector space
D = 2;                      
C = 2*D*rand([dimInputVector, Nclusters]) - D;

Omega = sqrt((2*D)^2/12); % standard deviation of centroids
rho = 0.3;                % ratio of standard deviation of centroids and standard deviation of the cluster points.
sigma = rho*Omega;        % standard deviation of the cluster points.

% Generate \Nclusters\ clusters with \Npatterns\ patterns per cluster
X = sigma*randn(dimInputVector, Nclusters*Npatterns);
D = zeros(Nclusters, Nclusters*Npatterns);
for cluster = 1:Nclusters
    X(:, Npatterns*(cluster-1) + (1:Npatterns)) = bsxfun(@plus, X(:, Npatterns*(cluster-1) + (1:Npatterns)), C(:, cluster));
    D(cluster, Npatterns*(cluster-1) + (1:Npatterns)) = 1;
end

% Randomly permute clusters
idx = randperm(Nclusters*Npatterns);
X = X(:, idx);
D = D(:, idx);

% Original Hebbian-LMS (HLMS)
seed = rng;
HLMSoriginal = NeuralNetwork(numHiddenLayers, numNeuronsHL, Nclusters, 7e-3);
HLMSoriginal.dataPartitioning = [1 0 0];  % all for training
% Backpropagation
rng(seed); % reset seed of RNG so that both networks have same initial conditions
BP = NeuralNetwork(numHiddenLayers, numNeuronsHL, Nclusters, 0.1e-3);
BP.dataPartitioning = [1 0 0];  % all for training

% Choose output layer
output_layer_fun = 'sigmoid';
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
HLMSoriginal.train(X, Dtrain, 'hebbian-lms', 0.01, true); % for sigmoid output layer
toc, tic
disp('Backpropagation')
BP.set_functions('sigmoid', output_layer_fun)
BP.train(X, Dtrain, 'backpropagation', 0.01, true); % for sigmoid output layer
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

