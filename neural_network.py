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
	def train(self,inputs_list,targets_list):
		# calculate outputs of each layer
		# convert list to numpy array
		inputs = numpy.array(inputs_list,ndmin=2).T
		activations = [inputs]
		# for each layer in the network
		for i in range(len(self.shape)-1):
			# print(i, activations)
			outputs = numpy.dot(self.weights[i],activations[i]) + numpy.tile(self.biases[i],(1,len(inputs_list)))
			activations.append(self.activation_function(outputs))
		errors = [False]*(len(self.shape))
		errors[-1] = numpy.array(targets_list,ndmin=2).T-activations[-1]
		# adjust the weights and biases of each layer
		new_weights = []
		new_biases = []
		print(self.shape)
		for i in reversed(range(1,len(self.shape))):
			# adjust weights of connections between layer i-1 and layer i
			#print("weights")
			#print(errors[i])
			#print("e:",errors[i].shape,"a:",activations[i].shape,activations[i-1].shape)
			#print(activations[i][0])
			# 
			adjust_weights = self.lr * numpy.dot((errors[i] * activations[i] * (1.0 - activations[i])),numpy.transpose(activations[i-1]))
			#print(self.weights[i-1],adjust_weights)
			#print(self.weights[i-1],"\n",adjust_weights)
			new_layer_weights = self.weights[i-1] + adjust_weights
			new_weights.append(new_layer_weights)
			# adjust biases of nodes in layer i
			#print("biases")
			adjust_biases = self.lr * numpy.dot(errors[i],numpy.ones((errors[i].shape[1],1)))/errors[i].shape[1]
			#print(self.biases[i-1],"\n",adjust_biases)
			#print(self.biases[i-1],adjust_biases)
			new_layer_biases = self.biases[i-1] + adjust_biases
			new_biases.append(new_layer_biases)
			# calculate errors for layer i-1 
			errors[i-1] = numpy.dot(self.weights[i-1].T,errors[i])
		new_weights = new_weights[::-1]
		new_biases = new_biases[::-1]
		self.weights = new_weights
		self.biases = new_biases
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
	
	
def main():
	numpy.random.seed(1)
	net = NeuralNetwork([5,3,2],0.1)
	print("0", net.query([10,-10,99,0,5]))
	for i in range(1,10):
		inputs_list = numpy.random.rand(1000,5)
		inputs_list = inputs_list.tolist()
		targets_list = [[x[0]+x[1],x[3]+x[4]] for x in inputs_list]
		net.train(inputs_list,targets_list)
		print(i,": ",net.query([10,-10,99,0,5]))
		
	return net
	
	
	
if __name__ == "__main__":
	pass