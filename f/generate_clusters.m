function [X, D, C, Cidx] = generate_clusters(Ndim, Nclusters, Npatterns, centroidParam, sigma)
%% Generate \Nclusters\ clusters in \Ndim\-dimensional space.
% Inputs:
% - Ndim: dimensionality of input vector space
% - Nclusters: number of clusters
% - Npatters: number of patterns per cluster
% - Omega: standard deviation of centrois of each cluster. Centroids are
% distributed uniformly in the \Ndim\-dimensional space
% - sigma: standard deviation of patterns (vectors) in each cluster. Vectors 
% are normally distributed about the centroid
% Outputs:
% X: vectors
% D: desired response 
% C: centroids
% Cidx: indices of input vectors i.e., X(:, 1) belongs to centroid Cidx(1)

if length(centroidParam) == 1 % if clusterParam is scalar, generate centroids
    Omega = centroidParam;
    C = Omega*randn([Ndim, Nclusters]);
else % otherwise, use centroids provided in C
    C = centroidParam;
end

if Npatterns == 0 % if no pattern was assigned
    X = [];
    D = [];
    idx = 1:Nclusters;
end

X = sigma*randn(Ndim, Nclusters*Npatterns);
D = zeros(Nclusters, Nclusters*Npatterns);
n = 1;
for cluster = 1:Nclusters
    for pattern = 1:Npatterns
        X(:, n) = C(:, cluster) + X(:, n); % X(:, Npatterns*(cluster-1) + (1:Npatterns)) = bsxfun(@plus, X(:, Npatterns*(cluster-1) + (1:Npatterns)), C(:, cluster));
        D(cluster, n) = 1;
        n = n + 1;
    end
end

% Randomly permute clusters
idx = randperm(Nclusters*Npatterns);
Cidx = reshape(repmat((1:Nclusters), Npatterns, 1), 1, Nclusters*Npatterns);
X = X(:, idx);
D = D(:, idx);
Cidx = Cidx(idx);