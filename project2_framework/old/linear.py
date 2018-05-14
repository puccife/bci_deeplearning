import torch

class Linear:

	def __init__(self,inputLayers,outputLayers):

		self.input = inputLayers
		self.bias = outputLayers

	# getter function to return input and output neurons of a layer
	def get_input_output(self):

		return self.input,self.bias


	def forward_layer(self,input_data,weight,bias):
		
		s_l = torch.add(torch.mm(weight, input_data),bias)  # compute layerwise z before apply activation : (w(l).x(l-1) + b(l)
		return s_l

	
