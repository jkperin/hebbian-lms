%% Distribution of minimum distance between centroids
clear, clc

Nrealizations = 10000;
Nclusters = 50:50:500;
Ndim = 50;
D = 2; % centroids are uniformly distributed in [-D, D]

avgminDist = zeros(size(Nclusters));
for n = 1:length(Nclusters)
    minDist = zeros(1, Nrealizations);
    for k = 1:Nrealizations
        C = 2*D*(rand([Ndim, Nclusters(n)]) - 0.5);

        Dist = pdist(C.', 'euclidean');

        minDist(k) = min(Dist);
    end

    figure(n), clf
    hist(minDist)
    title(sprintf('%d clusters, Average min distance = %.2f', Nclusters(n), mean(minDist)))
    
    avgminDist(n) = mean(minDist);
end

figure, hold on, box on
plot(Nclusters, avgminDist, '-ok', 'LineWidth', 2)
xlabel('Number of centroids', 'FontSize', 12)
ylabel('Average min. distance between centroids', 'FontSize', 12)
title(sprintf('Average min. distance in %d-dimensional space', Ndim))

rho = linspace(0, 1);                % ratio of standard deviation of centroids and standard deviation of the cluster points.
Omega = sqrt((2*D)^2/12); % standard deviation of centroids
sigma = 	;        % standard deviation of the cluster points.

figure, hold on, box on
plot(rho*100, sigma, 'LineWidth', 2)
xlabel('\rho (%)', 'FontSize', 12)
ylabel('Standard deviation of clusters', 'FontSize', 12)
