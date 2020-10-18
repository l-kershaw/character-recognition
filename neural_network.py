import numpy
from scipy.special import expit

# neural network class definition
class NeuralNetwork:
	
	# initialise the neural network
	def __init__(self, layer_shape, learning_rate):
		# set number of nodes in each layer (hidden may be a list)
		self.shape = layer_shape
		# set the learning rate
		self.lr = learning_rate
		# set activation function
		self.activation_function = lambda x: expit(x)
		
		# initialise weights between each layer
		self.weights = []
		for layer in range(0,len(self.shape)-1):
			layer_weights = numpy.random.normal(0.0, pow(self.shape[layer+1],-0.5),(self.shape[layer+1],self.shape[layer]))
			self.weights.append(layer_weights)
		# initialise biases at each layer
		self.biases = []
		for layer in range(1,len(self.shape)):
			layer_biases = numpy.random.normal(0.0, pow(self.shape[layer],-0.5),(self.shape[layer],1))
			self.biases.append(layer_biases)
		pass
		
	# train the neural network
	def train():
		pass
		
	# query the neural network
	def query(self,inputs_list):
		# convert list to numpy array
		inputs = numpy.array(inputs_list,ndmin=2).T
		activations = inputs
		# for each layer in the network
		for i in range(len(self.shape)-1):
			# print(i, activations)
			outputs = numpy.dot(self.weights[i],activations) + self.biases[i]
			activations = self.activation_function(outputs)
		# return final activations
		return activations
	
	
	
	
	

if __name__ == "__main__":
	pass