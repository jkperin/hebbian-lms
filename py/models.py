from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import initializers, activations
from keras import backend as K
from keras.engine.topology import Layer

def fully_connected(input_shape, hidden_layers_shape, output_layer_shape):
    ''' Create keras model of fully-connected network
    hidden_layers_shape = (number of hidden layers, number of neurons per hidden layer)
    output_layer_shape = number of neurons in the output layer
    '''
    
    model = Sequential()
    model.add(Dense(hidden_layers_shape[1], input_shape=input_shape, activation='tanh'))
    
    for _ in range(1, hidden_layers_shape[0]):
        model.add(Dense(hidden_layers_shape[1], activation='tanh'))
    
    model.add(Dense(output_layer_shape, activation='softmax'))

    return model

def hlms_network(input_shape, hidden_layers_shape, output_layer_shape):
    ''' Create keras model of fully-connected network trained with HLMS
    hidden_layers_shape = (number of hidden layers, number of neurons per hidden layer)
    output_layer_shape = number of neurons in the output layer
    '''
    
    model = Sequential()
    model.add(HLMS(hidden_layers_shape[1], input_shape=input_shape, activation='tanh', gamma=0.3, lr=1e-3, use_bias=True,
                 kernel_initializer=hlms_initializer, 
                 bias_initializer='zeros'))
    
    for _ in range(1, hidden_layers_shape[0]):
        model.add(HLMS(hidden_layers_shape[1], input_shape=input_shape, activation='tanh', gamma=0.3, lr=1e-3, use_bias=True,
                 kernel_initializer=hlms_initializer, 
                 bias_initializer='zeros'))
    
    model.add(Dense(output_layer_shape, activation='softmax'))

    return model

def hlms_initializer(shape, dtype=None, std=0.2):
    return K.random_normal(shape, dtype=dtype)
    
class HLMS(Layer):
    """ HLMS densely-connected layer.
    `HLMS` implements the operation:
    `output = activation(dot(input, kernel) + bias)` as a regular `Dense` layer. 
    However, learing is unsupervised.
    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    # Arguments
        output_dim: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        gamma: 0 < gamma < 1 is the parameter of HLMS neuron
    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
        nD tensor with shape: `(batch_size, ..., output_dim)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, output_dim)`.
    """

    def __init__(self, 
                 output_dim, 
                 activation='tanh', 
                 gamma=0.3,
                 lr=1e-3,
                 use_bias=True,
                 kernel_initializer=hlms_initializer, 
                 bias_initializer='zeros',                 
                 **kwargs):               
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        
        super(HLMS, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.gamma = gamma
        self.lr = lr
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        #self.kernel = self.add_weight(name='kernel', 
        #                              shape=(input_shape[1], self.output_dim),
        #                              initializer='uniform',
        #                              trainable=True)
        #super(HLMS, self).build(input_shape)  # Be sure to call this somewhere!
        
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      trainable=False)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        trainable=False)
        else:
            self.bias = None
        super(HLMS, self).build(input_shape)  # Be sure to call this somewhere!
        

    def call(self, inputs):
        
        S = K.dot(inputs, self.kernel)
        if self.use_bias:
            S = K.bias_add(S, self.bias)
        if self.activation is not None:
            output = self.activation(S)
               
        delta = (output - self.gamma*S);
        K.update_add(self.kernel, 2*self.lr*K.dot(K.transpose(inputs), delta))
        if self.use_bias:
            K.update_add(self.bias, 2*self.lr*K.mean(delta, axis=0));    

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)    
    
    
    class HLMS_Network(num_hidden_layers, layers_shape, gamma, initializer=hlms_initializer):
        def __init__():
            self.num_hidden_layers = num_hidden_layers
        def build():
            # Hidden 1
            with tf.name_scope('hidden1'):
                weights = tf.Variable(
                    tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                        stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
                    name='weights')
                biases = tf.Variable(tf.zeros([hidden1_units]),
                                     name='biases')
                hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
                # Hidden 2
                with tf.name_scope('hidden2'):
                    weights = tf.Variable(
                        tf.truncated_normal([hidden1_units, hidden2_units],
                                            stddev=1.0 / math.sqrt(float(hidden1_units))),
                        name='weights')
                    biases = tf.Variable(tf.zeros([hidden2_units]),
                                         name='biases')
                    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
                    # Linear
                    with tf.name_scope('softmax_linear'):
                        weights = tf.Variable(
                            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                                stddev=1.0 / math.sqrt(float(hidden2_units))),
                            name='weights')
                        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                                             name='biases')
                        logits = tf.matmul(hidden2, weights) + biases
                        return logits

