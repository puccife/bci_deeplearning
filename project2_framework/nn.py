import torch
import numpy as np
import math
from sequential import Sequential
from activation_functions import *
from linear import Linear
from loss import MSELoss
from sgd import SGD


def generate_data(row_dimension, col_dimension):
	data_input = np.random.uniform(0,1,(row_dimension,col_dimension))
	indices = np.arange(0,row_dimension)	
	# apply a function on indices if input belongs to circle assign 1 otherwise assign 0
	data_output = np.asarray(list(map(lambda index: inCircle(data_input[index]),indices)))
	input_tensor = torch.from_numpy(data_input)
	target_tensor = torch.from_numpy(data_output)
	return input_tensor.float(),target_tensor.float()


# check if (x,y) point is inside circle or not
def inCircle(values):
	radius_squared = 1 / (2 * math.pi)
	x_squared = math.pow(values[0], 2)
	y_squared = math.pow(values[1], 2)
	if ((x_squared + y_squared) <= radius_squared):
		return 1
	else:
		return 0

# returns mini batch of data from start till the start + batch size
def get_data(train_input, train_target, start, mini_batch_size):
	current_sample = train_input.narrow(0, start, mini_batch_size)
	current_target = train_target.narrow(0, start, mini_batch_size)
	return current_sample, current_target


def run_network(train_input,train_target,test_input,test_target):

	eta = 0.001
	mini_batch_size = 50
	epochs = 10

	criterion,optmizer = MSELoss(),SGD()
	nn = Sequential(	Linear(2,25),
				ReLU(),
				Linear(25,25),
				ReLU(),
				Linear(25,25),
				ReLU(),
				Linear(25,2),
				Tanh()
			)
	
	
	
	for e in range(0,epochs):
	
		print("Epoch = {}".format(e))
		
		loss_sum = 0
		nn.init_Parameters()					

		# at every epoch start from a different point
		shuffle_indexes = torch.randperm(train_input.size(0))
		shuffle_train = train_input[shuffle_indexes]
		shuffle_target = train_target[shuffle_indexes]		
		
		
		# shuffle_train.size(0)
		for b in range(0,shuffle_train.size(0), mini_batch_size):

		
			# data	
			sample_data, sample_target = get_data(shuffle_train, shuffle_target, b, mini_batch_size)  # get mini batch
			column_sample = torch.transpose(sample_data, 0, 1)  # compute column vector of input
		
			# forward train phase
			forward_pass = nn.forward(column_sample)
		
			
			# prediction
			_,predicted_labels = torch.max(forward_pass[len(forward_pass) - 1][1],0) # argmax of last layer gets us node with highest proba
			mse_loss = criterion.loss(predicted_labels.float(),sample_target)
			loss_sum +=mse_loss
			
		
			print("Iteration = {} and MSE Loss = {}".format(b/mini_batch_size,mse_loss))		

			# backward
			backward_pass = criterion.backward_pass(nn.get_activations(),forward_pass,predicted_labels.float(),sample_target,nn.parameters()[0] )

						
			gradient = optmizer.compute_stoch_gradient(backward_pass,forward_pass )		
			nn.update_parameters(eta,gradient)
			
		
	
		# test
		test_transpose = torch.transpose(test_input, 0, 1) 
		test_net = nn.forward(test_transpose)
		_,prediction = torch.max(test_net[len(test_net) - 1][1],0)	
		test_loss = criterion.loss(prediction.float(),test_target)
		train_acc = (train_input.size(0) - loss_sum)/train_input.size(0)
		test_acc = (test_input.size(0) - test_loss)/test_input.size(0)
		print("Train accuracy = {} and Test accuracy = {} ".format(train_acc,test_acc))	
