function [train, valid, test] = partition_data(X, D, dataPartitioning)
    %% Partition data into trainig, validation, and testing
    % Partition ratios are given by the property dataPartitioning
    % Patitioning is not random. To partition the data randomly, it
    % is necessary to scramble X and d first
    N = size(X, 2);

    % Training
    train.idx = 1:dataPartitioning(1)*N;
    train.X = X(:, train.idx);
    train.d = D(:, train.idx);
    idxend = train.idx(end);

    % Validation
    if dataPartitioning(2) == 0
        valid = [];
    else
        valid.idx = idxend + (1:dataPartitioning(2)*N);
        valid.X = X(:, valid.idx);
        valid.d = D(:, valid.idx);
        idxend = valid.idx(end);
    end

    % Testing
    if dataPartitioning(3) == 0
        test = [];
    else
        test.idx = idxend + (1:dataPartitioning(3)*N);
        test.X = X(:, test.idx);
        test.d = D(:, test.idx);
    end