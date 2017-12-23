import numpy as np 

def randn_initializer(shape, std=0.2):
    return std*np.random.randn(shape)

class Layer:
	def __init__(self, 
		input_shape, 
		num_neurons, 
		gamma=0.3
		kernel_initializer=lambda s : randn_initializer(s, std=0.2), 
		activation=np.tahn, 
		use_bias=True, 
		bias_initializer=np.zeros):

		self.num_weights = input_shape
		self.num_neurons = num_neurons
		self.gamma = gamma
		self.kernel_initializer = kernel_initializer
		self.activation = activation
		self.use_bias = use_bias
		self.bias_initializer = bias_initializer
		self.kernel = self.kernel_initializer((self.num_weights, num_neurons))
		self.bias = self.bias_initializer(num_neurons)

	def call(self, x, training=False, lr=1e-3):
		'''
		Compute layer output and update weights if 'training' == True
		x : input data. np array of shape (B, num_weights), where B is the batch size
		training: boolean that controls whether weights are updated
		lr : learning rate. Only used when 'training' == True
		'''
		S = np.dot(x, self.kernel) # output of linear combiner
		if self.use_bias:
			S += self.bias

		Y = self.activation(S) # neuron output

		if training: # update weights and kernel
			delta = (Y - self.gamma*S)
			self.kernel += 2*lr*np.dot(np.transpose(x), delta)
			if self.use_bias:
				self.bias += 2*lr*np.mean(x, axis=0)

		return Y


class Network:
	def __init__(self, input_shape, num_hidden_layers, num_neurons):
		self.Layers = []
		self.Layers = [Layer(input_shape, 
							 num_neurons, 
								gamma=0.3
								kernel_initializer=lambda s : randn_initializer(s, std=0.2), 
								activation=np.tahn, 
								use_bias=True, 
								bias_initializer=np.zeros))]
		for _ in range(1, num_hidden_layers):
			self.Layers.append(Layer(num_neurons, 
									 num_neurons, 
										gamma=0.3
										kernel_initializer=lambda s : randn_initializer(s, std=0.2), 
										activation=np.tahn, 
										use_bias=True, 
										bias_initializer=np.zeros))

	def train(self, train_set, lr=1e-3):
		b, w = train_set.shape
		output = []
		for i in range(b):
			y = train_set[i,:]
			for l in self.Layers:
				y = l.call(y, training=True, lr=lr)
			output.append(y)

		return np.array(output)

	def forward(self, x):
		b, w = x.shape
		output = []
		for i in range(b):
			y = x[i,:]
			for l in self.Layers:
				y = l.call(y, training=False)
			output.append(y)

		return np.array(output)

