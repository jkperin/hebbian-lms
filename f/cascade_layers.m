function Y = cascade_layers(L, X)
%% Calculate output of layers in L for input X
if isempty(X)
    Y = X;
    return;
end

Y = X;
for k = 1:length(L)
    Y = L(k).output(Y);
end