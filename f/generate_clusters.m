function [X, D, C, idx] = generate_clusters(Ndim, Nclusters, Npatterns, centroidParam, sigma, NdisturbClusters)
%% Generate \Nclusters\ clusters in \Ndim\-dimensional space.
% Inputs:
% - Ndim: dimensionality of input vector space
% - Nclusters: number of clusters
% - Npatters: number of patterns per cluster
% - Omega: standard deviation of centrois of each cluster. Centroids are
% distributed uniformly in the \Ndim\-dimensional space
% - sigma: standard deviation of patterns (vectors) in each cluster. Vectors 
% are normally distributed about the centroid
% - NdisturbClusters (optional, default = 0): number of centroids added as
% disturbance. They not appear in X.
% Outputs:
% X: vectors
% D: desired response 
% C: centroids
% idx: indices to recover random permutation

if length(centroidParam) == 1 % if clusterParam is scalar, generate centroids
    Omega = centroidParam;
    C = sqrt(12)*Omega*(rand([Ndim, Nclusters]) - 0.5);
else % otherwise, use centroids provided in C
    C = centroidParam;
end

if Npatterns == 0 % if no pattern was assigned
    X = [];
    D = [];
    idx = 1:Nclusters;
end

if not(exist('NdisturbClusters', 'var'))
    NdisturbClusters = 0;
end

X = sigma*randn(Ndim, (Nclusters-NdisturbClusters)*Npatterns);
D = zeros(Nclusters, (Nclusters-NdisturbClusters)*Npatterns);
n = 1;
for cluster = 1:Nclusters-NdisturbClusters
    for pattern = 1:Npatterns
        X(:, n) = C(:, cluster) + X(:, n); % X(:, Npatterns*(cluster-1) + (1:Npatterns)) = bsxfun(@plus, X(:, Npatterns*(cluster-1) + (1:Npatterns)), C(:, cluster));
        D(cluster, n) = 1;
        n = n + 1;
    end
end

% Randomly permute clusters
idx = randperm((Nclusters-NdisturbClusters)*Npatterns);
X = X(:, idx);
D = D(:, idx);