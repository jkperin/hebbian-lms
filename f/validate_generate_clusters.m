%% Validate function generate_clusters
clear, clc, close all

dimInputVector = 3;        % dimensionality of input vector space
Nclusters = 6;            % number of clusters
Npatterns = 50;           % number of patterns per cluster
NdisturbClusters = 0;      % number of clusters not not included in training

% Generate centroids whose coordinates are uniformly distributed in [-D, D] 
% of each dimension of the input vector space
D = 2;                      
rho = 0.1;                % ratio of standard deviation of centroids and standard deviation of the cluster points.
Omega = sqrt((2*D)^2/12); % standard deviation of centroids
sigma = rho*Omega;        % standard deviation of the cluster points.

% Generate \Nclusters\ clusters with \Npatterns\ patterns per cluster
dataPartitioning = [0.5 0 0.5]; % 50% for training, 0% for validation, and 50% for testing
[Xtrain, Dtrain, C] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(1)*Npatterns, Omega, sigma, NdisturbClusters);
[Xval, Dval] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(2)*Npatterns, C, sigma);
[Xtest, Dtest] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(3)*Npatterns, C, sigma);

figure, hold on, box on
plot3(C(1, :), C(2, :), C(3, :))
plot3(Xtrain(1, :), Xtrain(2, :), Xtrain(3, :), 'o')
plot3(Xtest(1, :), Xtest(2, :), Xtest(3, :), 'o')
view(3)