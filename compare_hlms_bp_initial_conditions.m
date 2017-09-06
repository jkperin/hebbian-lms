%% Compare Hebbian-LMS and Backpropagation
clear, clc, close all

addpath f/                  % auxiliary functions folder

numHiddenLayers = 3;        % number of hidden layers
numNeuronsHL = 150;         % number of neurons in the hidden layers
dimInputVector = 50;       % dimensionality of input vector space
Nclusters = 100;            % number of clusters
Npatterns = 40;            % number of patterns per cluster

% Generate centroids whose coordinates are uniformly distributed in [-D, D] 
% of each dimension of the input vector space
Omega = 1;                % standard deviation of centroids
rho = 0.75;                % ratio of standard deviation of centroids and standard deviation of the cluster points.
sigma = rho*Omega;        % standard deviation of the cluster points.

Vhyper_sphere = @(s, N) pi^(N/2)*(5*s)^N/gamma(1 + N/2);
Vcluster = Vhyper_sphere(sigma, dimInputVector);
Vspace = Vhyper_sphere(Omega, dimInputVector);
Vratio = Vcluster/Vspace

% Generate \Nclusters\ clusters with \Npatterns\ patterns per cluster
dataPartitioning = [0.3 0 0.7]; % 20% for training, 0% for validation, and 80% for testing
[Xtrain, Dtrain, C, Cidx] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(1)*Npatterns, Omega, sigma);
[Xval, Dval] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(2)*Npatterns, C, sigma);
[Xtest, Dtest] = generate_clusters(dimInputVector, Nclusters, dataPartitioning(3)*Npatterns, C, sigma);

Dist = pdist(C.', 'euclidean');
meanDist = min(Dist)
newRho = sigma/meanDist

% Concatanate 
X = [Xtrain Xval Xtest];
D = [Dtrain Dval Dtest];

%
state = rng;
%% HLMS
HLMSoriginal = NeuralNetwork(numHiddenLayers, numNeuronsHL, Nclusters); % 0.2e-3 for 20 patterns
% HLMSoriginal.gamma = 0.3;
HLMSoriginal.dataPartitioning = dataPartitioning;  % all for training
%% Backpropagation
rng(state); % reset seed of RNG so that both networks have same initial conditions
BP = NeuralNetwork(numHiddenLayers, numNeuronsHL, Nclusters); % 0.5e-3 for sigmoid
BP.dataPartitioning = dataPartitioning;  % all for training

%% Range of initial weights
% Original Hebbian-LMS (HLMS)
rng('shuffle')
Wrange = 0.05:0.05:0.3;
for n = 1:length(Wrange)
    state = rng;
    HLMSoriginal.reset(Wrange(n));
    rng(state)
    BP.reset(Wrange(n));

    %% Training
    tic
    disp('Hebbian-LMS-Original')
    HLMSoriginal.set_functions('sigmoid', 'softmax')
    [HLMStrain, HLMSvalid, HLMStest] = HLMSoriginal.train(X, D, 'Hebbian-LMS', 1e-3, 21, false); % for sigmoid output layer
%     [ConsisHLMS, Nwords] = HLMSoriginal.consistency(Xtrain, Cidx, C, true)

    figure(1), hold on, box on
    hplot = plot(100*HLMStest.error(2:end), 'LineWidth', 2, 'DisplayName', ['\sigma_w = ', sprintf('%.2f', Wrange(n))]);
%     plot(100*HLMStest.error(2:end), '--', 'Color', get(hplot, 'Color'), 'LineWidth', 2, 'HandleVisibility','off')
    xlabel('Training cycles')
    ylabel('Error rate (%)')
    legend('-dynamiclegend')
    set(gca, 'FontSize', 12)
%     axis([1 20 0 30])
    drawnow

    toc, tic
    disp('Backpropagation')
    BP.set_functions('sigmoid', 'softmax')
    [BPtrain, BPvalid, BPtest] = BP.train(X, D, 'Backpropagation', 1e-3, 21, false); % for sigmoid output layer
%     ConsisBP = BP.consistency(Xtrain, Cidx, true)
    toc
   
    figure(2), hold on, box on
    hplot = plot(100*BPtest.error(2:end), 'LineWidth', 2, 'DisplayName', ['\sigma_w = ', sprintf('%.2f', Wrange(n))]);
%     plot(100*BPtest.error(2:end), '--', 'Color', get(hplot, 'Color'), 'LineWidth', 2, 'HandleVisibility', 'off')
    xlabel('Training cycles')
    ylabel('Error rate (%)')
    legend('-dynamiclegend')
    set(gca, 'FontSize', 12)
%     axis([1 20 0 30])
    drawnow
end
% 
figure(1), saveas(gca, 'doc/figs/w0_hlms_300', 'epsc')
figure(2), saveas(gca, 'doc/figs/w0_bp_300', 'epsc')

%% Sensibility to initial conditions
% Nrealizations = 3;
% for n = 1:Nrealizations
%     rng('shuffle')
%     state = rng;
%     state.Seed
%     state.State
%     HLMSoriginal.reset(0.2);
%     rng(state)
%     BP.reset(0.1);
% 
%     %% Training
%     tic
%     disp('Hebbian-LMS-Original')
%     HLMSoriginal.set_functions('sigmoid', 'linear')
%     [HLMStrain, HLMSvalid, HLMStest] = HLMSoriginal.train(X, D, 'Hebbian-LMS', 1e-3, 21, false); % for sigmoid output layer
% 
%     figure(1), hold on, box on
%     hplot = plot(100*HLMStrain.error(2:end), 'LineWidth', 2, 'DisplayName', sprintf('seed = %d', n));
%     plot(100*HLMStest.error(2:end), '--', 'Color', get(hplot, 'Color'), 'LineWidth', 2, 'HandleVisibility','off')
%     xlabel('Training cycles')
%     ylabel('Error rate (%)')
%     legend('-dynamiclegend')
%     set(gca, 'FontSize', 12)
%     axis([1 20 0 30])
%     drawnow
% 
%     toc, tic
%     disp('Backpropagation')
%     BP.set_functions('sigmoid', 'softmax')
%     [BPtrain, BPvalid, BPtest] = BP.train(X, D, 'Backpropagation', 1e-3, 21, false); % for sigmoid output layer
%     toc
%    
%     figure(2), hold on, box on
%     hplot = plot(100*BPtrain.error(2:end), 'LineWidth', 2, 'DisplayName', sprintf('seed = %d', n));
%     plot(100*BPtest.error(2:end), '--', 'Color', get(hplot, 'Color'), 'LineWidth', 2, 'HandleVisibility', 'off')
%     xlabel('Training cycles')
%     ylabel('Error rate (%)')
%     legend('-dynamiclegend')
%     set(gca, 'FontSize', 12)
%     axis([1 20 0 30])
%     drawnow
% end
% 
% figure(1), saveas(gca, 'doc/figs/w0_sens_hlms', 'epsc')
% figure(2), saveas(gca, 'doc/figs/w0_sens_bp2', 'epsc')
% 