%% Validate function generate_clusters
clear, clc, close all

dimInputVector = 2;        % dimensionality of input vector space
Nclusters = 8;            % number of clusters
Npatterns = 100;           % number of patterns per cluster
NdisturbClusters = 0;      % number of clusters not not included in training

% Generate centroids whose coordinates are uniformly distributed in [-D, D] 
% of each dimension of the input vector space
rho = 0.05;                % ratio of standard deviation of centroids and standard deviation of the cluster points.
Omega = 1; % standard deviation of centroids
sigma = rho*Omega*sqrt(2*dimInputVector);        % standard deviation of the cluster points.

% Generate \Nclusters\ clusters with \Npatterns\ patterns per cluster
dataPartitioning = [0.5 0 0.5]; % 50% for training, 0% for validation, and 50% for testing
[Xtrain, Dtrain, C] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(1)*Npatterns, Omega, sigma);
[Xval, Dval] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(2)*Npatterns, C, sigma);
[Xtest, Dtest] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(3)*Npatterns, C, sigma);

figure, hold on, box on
plot(C(1, :), C(2, :))
plot(Xtrain(1, :), Xtrain(2, :), 'o')
axis equal
% plot3(C(1, :), C(2, :), C(3, :))
% % plot3(Xtrain(1, :), Xtrain(2, :), Xtrain(3, :), 'o')
% plot3(Xtest(1, :), Xtest(2, :), Xtest(3, :), 'o')
% view(3)