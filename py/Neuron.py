import tensorflow as np
import numpy as np

def sgm(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.neg(x))));


class NeuralNetwork:
    Nhl # number of hidden layers
    Nhlneurons # number of neurons in each hidden layer
    Noutneurons # number of neurons in the output layer
    W # weights for the entire network
    b # bias weights for the entire network
    mu # adaptation constant
    S # output each hidden layer
    Y # output each hidden layer
    seed # random number generator state before intializing the weights
    dataPartitioning = [0.7 0.15 0.15] # how much is allocated for training, validation, and testing
    outputFcn='sigmoid' # output function of the output layers {'linear',  
    hiddenFcn='sigmoid' # output function of the hidden layers {'sigmoid'

	def __init__(Nhl, Nhlneurons, Noutneurons, mu):
		''' Initialize class Neuron
		Arguments:
		- 
		'''
		self.Nhl = Nhl;
		self.Nhlneurons = Nhlneurons;
		self.Noutneurons = Noutneurons;
		self.mu = tf.constant(mu);

	    self.W = [tf.Variable(tf.random_normal([Nhlneurons, Nhlneurons], mean=0, stddev=1), name="weights") for k in range(Nhlneurons)];
	    self.b = [tf.Variable(tf.zeros([Nhlneurons, 1]), name="bias") for k in range(Nhlneurons)]; 
	    self.S = [tf.Variable(tf.zeros([Nhlneurons, 1]), name="adder_output") for k in range(Nhlneurons)]; 
	    self.Y = [tf.Variable(tf.zeros([Nhlneurons, 1]), name="neuron_output") for k in range(Nhlneurons)]; 

    	self.W.append(tf.Variable(tf.random_normal([Noutneurons, Nhlneurons], mean=0, stddev=1), name="weights"));
		self.b.append(tf.Variable(tf.zeros([Noutneurons, 1]), name="bias"));
		self.S.append(tf.Variable(tf.zeros([Noutneurons, 1]), name="adder_output"));
		self.Y.append(tf.Variable(tf.zeros([Noutneurons, 1]), name="neuron_output"));

	def yout = forward(self, X):
		tf.assign(self.S[0], tf.add(tf.matmul(self.W[0], X, transpose_a=false), self.b[0]));
		tf.assign(self.Y[0], sgm(self.S[0]));

		for k in range(1, Nhl):
			tf.assign(self.S[k], tf.add(tf.matmul(self.W[k], self.Y[k-1], transpose_a=false), self.b[k]));
			tf.assign(self.Y[k], sgm(self.S[k]));

		# Output layer
		tf.assign(self.S[-1], tf.add(tf.matmul(self.W[-1], self.Y[-2], transpose_a=false), self.b[-1]));
        tf.assign(self.Y[-1], sgm(self.S[-1]));
        yout = self.Y[-1];


net = NeuralNetwork(2, 10, 2, 1e-3);
net.forward(Xx)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
sess.run(net.forward, feed_dict = {X: np.ones(10, 1)});