import torch
import math
from activation_functions import *
from linear import Linear

class Sequential:


	# takes the layers of the network as input and stores them in a list
	def __init__ (self,* argv):

		neural_net = []
		for arg in argv:
			neural_net.append(arg)
		self.neural_net = neural_net
		self.weights,self.bias = self.init_Parameters() 


	def init_Parameters(self):
		
		weights,bias = [],[]
		for i in range(0,len(self.neural_net),2): # iterate over the sturcture of the network
			
			inputL,outputL = self.neural_net[i].get_input_output() # get number of inputs and outputs
			activation = self.neural_net[i+1] # get the activation
			
			# we do xavier initialization , based on the type of activation 
			# we select the constant to multiply the random vector with
			if isinstance(activation,ReLU):
				cnst = math.sqrt(2/inputL)
			elif isinstance(activation,Tanh):
				cnst = math.sqrt(1/inputL)
						

			weight = torch.Tensor(inputL,outputL).normal_() * cnst # initialize weight
			b = torch.Tensor(outputL).uniform_() # initialize bias
			
			weights.append(weight)
			bias.append(b)

		return weights,bias

	# getter method returns activation functions in the network
	def get_activations(self):
		activations = list(filter(lambda layer: isinstance(layer,Activation) ,self.neural_net))
		return activations

	# getter method returns weights and bias in the network
	def parameters(self):
		return (self.weights,self.bias)

	# subtracts the gradient from the weight 
	def update_parameters(self,learning_rate,gradient):

		for i in range(0, len(gradient)):			
			weight_loss = learning_rate * gradient[i][0]  # multiply step size with gradient				
			self.weights[i] = (self.weights[i] - weight_loss) 
		
	
	# forward pass	
	def forward(self,input_data):

		
		j = 0
		# add initial input to list storing tuple of layerwise data before and after applying activation per layer
		layer_data = [(input_data,input_data)]  
		current_data = input_data
		for i in range(0,len(self.neural_net),2): 
	
			# iterate over the structure of the network in pairs , for every data point its respective activation
							
			current_weight_matrix = torch.transpose(self.weights[j], 0, 1)
			current_bias_matrix = (self.bias[j]).view(-1, 1)

			s_l = self.neural_net[i].forward_layer(current_data,current_weight_matrix,current_bias_matrix) # compute before activation
			x = self.neural_net[i+1].forward(s_l) # data after activation			
			
			layer_data.append((s_l,x))
			current_data = x # next layer the input is the data point after performing activation
			j = j + 1
		
		return layer_data
