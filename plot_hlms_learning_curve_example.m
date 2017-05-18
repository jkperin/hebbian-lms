%% Validate Hebbian-LMS behaviour
clear, clc, close all
addpath ../codes/f/

% rng('default')

sigmoid = @(x) (1 - exp(-2*x))./(1 + exp(-2*x));
dsigmid = @(x) 4*exp(2*x)./(1 + exp(2*x)).^2;

Ndim = 50;
Npatterns = 40;

D = 2;                      
rho = 0.5;                % ratio of standard deviation of centroids and standard deviation of the cluster points.
Omega = sqrt((2*D)^2/12); % standard deviation of centroids
sigma = rho*Omega;        % standard deviation of the cluster points.

[~, ~, C] = generate_clusters(Ndim, Npatterns, 1, Omega, sigma);
X = C;
W = 1e-3*(rand(Ndim, 1)-0.5);
b = 0;
d = sign(X.'*W + b);

gamma = 0.3;
mu = 1e-3;
l = linspace(-6, 6);

% Initial conditions
Y = zeros(size(X, 2), 1);
S = zeros(size(X, 2), 1);
for k = 1:size(X, 2)
    s = X(:, k).'*W + b;
    y = sigmoid(s);
    Y(k) = y;
    S(k) = s; 
end
% figure, clf, hold on, box on
% plot(l, sigmoid(l), 'k', l, gamma*l, 'k')
% plot(S(d == -1), Y(d==-1), 'xr', 'MarkerSize', 10)
% plot(S(d == 1), Y(d==1), 'xb', 'MarkerSize', 10)
% axis([-6 6 -1.2 1.2])
% title('Initial conditions')
% m = matlab2tikz(gca);
% m.write('initial conditions.tex')

for cycles = 1:2000
    for k = 1:size(X, 2)
        s = X(:, k).'*W + b;
        y = sigmoid(s);
%         delta = -(y - gamma*s)*(dsigmid(s) - gamma);
        delta = (y - gamma*s);
        W = W + 2*mu*delta*X(:, k);
        b = b + 2*mu*delta;
        Y(k) = y;
        S(k) = s;
        MSE(cycles, k) = delta^2;
    end
    
%     if ismember(cycles, [10 100 500 1000 2000])
%         figure, clf, hold on, box on
%         plot(l, sigmoid(l), 'k', l, gamma*l, 'k')
%         plot(S(d == -1), Y(d==-1), 'xr', 'MarkerSize', 10)
%         plot(S(d == 1), Y(d==1), 'xb', 'MarkerSize', 10)
%         axis([-6 6 -1.2 1.2])
%         title(sprintf('%d', cycles))
%         drawnow
%         m = matlab2tikz(gca);
% %         m.write(sprintf('cycles%d.tex', cycles))
%     end
end

% figure, box on
% plot(mean(MSE, 2))
% m = matlab2tikz(gca);
% % m.write('learning_curve.tex')
    
Nselect = 10;
Ntest = 1000;
Dref = sign(X(:, Nselect).'*W + b)
Dnew =zeros(Ntest, 1);
for n = 1:Ntest
    Xnew = X(:, Nselect) + 0.4*randn(Ndim, 1);
    Dnew(n) = sign(Xnew.'*W + b);
end

sum(Dnew ~= Dref)

    



