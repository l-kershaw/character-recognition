import numpy
import scipy.special
import random
import pandas as pd

class NeuralNetwork:
		
		
	# initialise the neural network
	def __init__(self, layer_shape, learningrate):
		# set number of nodes in each layer
		self.shape = layer_shape
		self.layers = len(self.shape)
		
		# link weight matrices
		self.weights = []
		for i in range(0,self.layers-1):
			self.weights.append(numpy.random.normal(0.0, pow(self.shape[i+1], -0.5), (self.shape[i+1], self.shape[i])))

		# learning rate
		self.lr = learningrate
		
		# activation function is the sigmoid function
		self.activation_function = lambda x: scipy.special.expit(x)
		
		pass

		
	# train the neural network
	def train(self, inputs_list, targets_list):
		# convert inputs list to 2d array
		inputs = numpy.array(inputs_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T
		
		activations = [inputs]
		for i in range(0,self.layers-1):
			layer_outputs = numpy.dot(self.weights[i],activations[i])
			activations.append(self.activation_function(layer_outputs))
		
		errors = [False]*self.layers
		# output layer error is the (target - actual)
		# print(targets)
		# print(activations[-1])
		errors[-1] = targets.T - activations[-1]
		for i in range(self.layers-2,-1,-1):
			errors[i] = numpy.dot(self.weights[i].T,errors[i+1])
		# hidden layer error is the output_errors, split by weights, recombined at hidden nodes
		# hidden_errors = numpy.dot(self.who.T, output_errors) 
		
		for i in range(0,self.layers-1):
			# print(errors[i+1].shape,activations[i+1].shape,activations[i].shape)
			self.weights[i] += self.lr * numpy.dot((errors[i+1] * activations[i+1] * (1.0 - activations[i+1])), numpy.transpose(activations[i]))
		# # update the weights for the links between the hidden and output layers
		# self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
		
		# # update the weights for the links between the input and hidden layers
		# self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
		
		pass

	
	# query the neural network
	def query(self, inputs_list):
		# convert inputs list to 2d array
		inputs = numpy.array(inputs_list, ndmin=2).T
		activations = [inputs]
		for i in range(0,self.layers-1):
			layer_outputs = numpy.dot(self.weights[i],activations[i])
			activations.append(self.activation_function(layer_outputs))
		
		return activations[-1]

def target(choice,length):
	result = numpy.zeros(length) + 0.001
	result[choice] = 0.999
	return numpy.array(result,ndmin=2).T

def init_trained_network(data_csv,weight_csv):
	init_data = pd.read_csv(data_csv)
	init_data = init_data.values.tolist()
	shape = init_data[0]
	shape = [int(x) for x in shape]
	lr = init_data[1][0]
	# print(shape,lr)
	n = NeuralNetwork(shape,lr)
	df = pd.read_csv(weight_csv)
	read_weights = []
	row = 0
	for i in range(0,n.layers-1):
		layer_weights = df[row:row+n.shape[i+1]].to_numpy()
		layer_weights = layer_weights[:,:n.shape[i]]
		read_weights.append(layer_weights)
		row += n.shape[i+1]
	n.weights = read_weights
	return n
				
def main():
	numpy.random.seed(1)

	layer_shape = [784,100,10]
	learningrate = 0.1
	n = NeuralNetwork(layer_shape,learningrate)
	
	# load the mnist training data CSV file into a list
	training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
	training_data_list = training_data_file.readlines()
	training_data_file.close()
	
	# load the mnist test data CSV file into a list
	test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
	test_data_list = test_data_file.readlines()
	test_data_file.close()
	
	# train the neural network

	# epochs is the number of times the training data set is used for training
	epochs = 200

	for e in range(epochs):
		print("Begin epoch:",e,"lr:",lr)
		random.shuffle(training_data_list)
		# go through all records in the training data set
		for record in training_data_list[0:1000]:
			# split the record by the ',' commas
			all_values = record.split(',')
			# scale and shift the inputs
			inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
			# create the target output values (all 0.01, except the desired label which is 0.99)
			targets = target(int(all_values[0]),layer_shape[-1])
			
			n.train(inputs, targets.tolist())
			pass
		pass
			
	
	
		# test the neural network

		# scorecard for how well the network performs, initially empty
		scorecard = []

		# go through all the records in the test data set
		for record in test_data_list:
				# Split the record by the ',' commas
				all_values = record.split(',')
				# Correct answer is first value
				correct_label = int(all_values[0])
				# Scale the inputs from all but first value
				inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.999) + 0.001
				# Query the network
				outputs = n.query(inputs)
				
				# Calculate the mean square error
				score = (target(correct_label,layer_shape[-1])-outputs)
				score = numpy.dot(score.T,score)
				scorecard.append(score)
				# Print first output and its associated score (for verification)
				# if record == test_data_list[0]:
					# print("outputs:\n",outputs)
					# print("score:\n", score)
	
		scorecard_array = numpy.asarray(scorecard)
		print ("loss = ", scorecard_array.sum() / scorecard_array.size)
		
		# Reduce learning rate by 5%
		n.lr = 0.95*n.lr
	# save weights
	matrices = []
	for i in range(0,n.layers-1):
		matrices.append(pd.DataFrame.from_records(n.weights[i]))
	print(matrices)
	df = pd.concat(matrices)
	print(df)
	df.to_csv('./trained_network/weights.csv',index=False)
	return n
	
if __name__ == "__main__":
	pass