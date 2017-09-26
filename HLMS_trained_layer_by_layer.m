%% Train HLMS sequentially 
clear, clc

rng(0)

addpath f/                  % auxiliary functions folder

numHiddenLayers = 3;        % number of hidden layers
numNeuronsHL = 150;         % number of neurons in the hidden layers
dimInputVector = 50;       % dimensionality of input vector space
Nclusters = 100;            % number of clusters
Npatterns = 100;            % number of patterns per cluster

% Generate centroids whose coordinates are uniformly distributed in [-D, D] 
% of each dimension of the input vector space
Omega = 1;                % standard deviation of centroids
rho = 0.5;                % ratio of standard deviation of centroids and standard deviation of the cluster points.
sigma = rho*Omega;        % standard deviation of the cluster points.

Vhyper_sphere = @(s, N) pi^(N/2)*(5*s)^N/gamma(1 + N/2);
Vcluster = Vhyper_sphere(sigma, dimInputVector);
Vspace = Vhyper_sphere(Omega, dimInputVector);
Vratio = Vcluster/Vspace

% Generate \Nclusters\ clusters with \Npatterns\ patterns per cluster

[X, D, C, idx] = generate_clusters(dimInputVector, Nclusters, Npatterns, Omega, sigma);

dataPartitioning = [0.2 0 0.8]; % 50% for training, 0% for validation, and 50% for testing
[train, valid, test] = partition_data(X, D, dataPartitioning);

%
state = rng;
L(1) = Layer(dimInputVector, numNeuronsHL, 0.1); % 0.2e-3 for 20 patterns
L(2) = Layer(numNeuronsHL, numNeuronsHL, 0.1); % 0.2e-3 for 20 patterns
L(3) = Layer(numNeuronsHL, numNeuronsHL, 0.1); % 0.2e-3 for 20 patterns
L(4) = Layer(numNeuronsHL, Nclusters, 0.1); % 0.2e-3 for 20 patterns

% Original Hebbian-LMS (HLMS)
L(1).train_HLMS(train.X, 1e-3, 50);
L(1).output(train.X);

L(2).train_HLMS(L(1).Y, 1e-3, 50);
L(2).output(L(1).Y);

L(3).train_HLMS(L(2).Y, 1e-3, 50);
L(3).output(L(2).Y);

Y = cascade_layers(L(1:3), X);
[train, valid, test] = partition_data(Y, D, dataPartitioning);
Data = {train, valid, test};
L(4).train_supervised(Data, 0.5e-3, 50);




