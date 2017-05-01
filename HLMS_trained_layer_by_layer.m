%% Train HLMS sequentially 
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

[X, D, C, idx] = generate_clusters(dimInputVector, Nclusters, Npatterns, Omega, sigma, NdisturbClusters);

dataPartitioning = [0.2 0 0.8]; % 50% for training, 0% for validation, and 50% for testing
[train, valid, test] = partition_data(X, D, dataPartitioning);

%
state = rng;
L(1) = Layer(dimInputVector, numNeuronsHL, 0.1); % 0.2e-3 for 20 patterns
L(2) = Layer(numNeuronsHL, numNeuronsHL, 0.1); % 0.2e-3 for 20 patterns
L(3) = Layer(numNeuronsHL, numNeuronsHL, 0.1); % 0.2e-3 for 20 patterns
L(4) = Layer(numNeuronsHL, Nclusters, 0.1); % 0.2e-3 for 20 patterns

% Original Hebbian-LMS (HLMS)
L(1).train_HLMS(train.X, 0.5e-3, 50);
L(1).output(train.X);

L(2).train_HLMS(L(1).Y, 0.5e-3, 50);
L(2).output(L(1).Y);

L(3).train_HLMS(L(2).Y, 0.5e-3, 50);
L(3).output(L(2).Y);

Y = cascade_layers(L(1:3), X);
[train, valid, test] = partition_data(Y, D, dataPartitioning);
Data = {train, valid, test};
L(4).train_supervised(Data, 0.5e-3, 50);




