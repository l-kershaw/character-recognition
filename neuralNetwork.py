import numpy

# neural network class definition
class neuralNetwork:
	
	# initialise the neural network
	def __init__(self, layer_shape, learning_rate):
		# set number of nodes in each layer (hidden may be a list)
		self.shape = layer_shape
		# set the learning rate
		self.lr = learning_rate
		
		# initialise weights between each layer
		self.weights = []
		for layer in range(0,length(self.shape)-1):
			layer_weights = numpy.random.normal(0.0, pow(self.shape[layer+1],-0.5),(self.shape[layer+1],self.shape[layer]))
			self.weights.append(layer_weights)
		# initialise biases at each layer
		self.biases = []
		for layer in range(1,length(self.shape)):
			layer_biases = numpy.random.normal(0.0, pow(self.shape[layer],-0.5),(self.shape[layer],1))
		pass
		
	# train the neural network
	def train():
		pass
		
	# query the neural network
	def query():
		pass
