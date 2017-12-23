%% Compare HLMS and backprop in the MNIST data set
clear, clc, close all

addpath mnist/
addpath f/                  % auxiliary functions folder

num_classes = 10;
train.X = loadMNISTImages('train-images-idx3-ubyte');
train.labels = loadMNISTLabels('train-labels-idx1-ubyte');
test.X = loadMNISTImages('t10k-images-idx3-ubyte');
test.labels = loadMNISTLabels('t10k-labels-idx1-ubyte');

train.N = size(train.X, 2);
test.N = size(test.X, 2);

% Convert images
train.images = reshape(train.X, 28, 28, train.N);
test.images = reshape(test.X, 28, 28, test.N);

% Convert to categorical
% train.D = 2*to_categorical(train.labels, num_classes)-1;
% test.D = 2*to_categorical(test.labels, num_classes)-1;

train.D = to_categorical(train.labels, num_classes);
test.D = to_categorical(test.labels, num_classes);

for k = 1:9
    subplot(3,3,k)
    idx = randi(test.N);
    imagesc(squeeze(test.images(:, :, idx)))
    colormap(gray)
    title(num2str(test.labels(idx)))
    set(gca, 'xtick', '')
    set(gca, 'ytick', '')
    axis square
end

numHiddenLayers = 5;        % number of hidden layers
numNeuronsHL = 1500;         % number of neurons in the hidden layers
dimInputVector = 28*28;       % dimensionality of input vector space

% disp('Backpropagation')
% BP = NeuralNetwork(dimInputVector, numHiddenLayers, numNeuronsHL, num_classes); 
% BP.initialize('glorot');
% BP.set_functions('sigmoid', 'softmax')
% BP.train(train.X, train.D, test.X, test.D, 'Backpropagation', 0.001, 20, true); 

disp('HLMS')
mean_train = mean(train.X, 2);
mean_test =  mean(test.X, 2);
HLMS = NeuralNetwork(dimInputVector, numHiddenLayers, numNeuronsHL, num_classes); 
HLMS.initialize('hlms');
HLMS.set_functions('sigmoid', 'softmax')
HLMS.train(train.X-mean_train, train.D, test.X-mean_test, test.D, 'hebbian-lms', 0.1/numNeuronsHL, 5, true); 
