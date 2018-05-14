import torch
from activation_functions import Activation

class MSELoss:


	def loss(self,prediction, target):
		return (target - prediction).pow(2).sum()  # MSE loss
		


	def last_layer_loss(self,train_target, predicted_label, last_layer_vec, last_layer_activation):
		# compute last layer delta
		diff = 	train_target.sub(predicted_label) * (-2)
		# for mse (yn - xn) derivative of data before applying activation
		# eg if last layer activation is tanh then we compute the derivative of tanh with the z of last layer as input
		# elementwise multiplication is done	
		ll_delta =  torch.mul(diff, last_layer_activation.derrivative(last_layer_vec) )	
		return ll_delta


	# compute backward pass in backprop algorithm
	# backward layer = component_wise_product(dot(w(l),delta(l+1)),phi_prime(z before activation))
	def backward_pass(self,activations,forward_data,predicted_label,train_target,weights):
		
			
		init_delta = self.last_layer_loss(train_target,predicted_label,forward_data[len(forward_data)-1][0],activations[len(activations) - 1])		

		deltas_per_layer = []
		
		deltas_per_layer.append(init_delta)  # append last layer delta
		iterator = 0
		
		# iterate over the remaining activations and compute the derivative of the loss wrt data poins
		for i in reversed(range(1,len(weights))):
						
			# compute the derivative of the activation with the data before activation of the current layer as input
			# please note that i is used in all cases since we appended the initial datapoint into the array computed by the
			# forward pass thus deltas and weights are shifted one ahead
			inverse_activation = activations[i-1].derrivative((forward_data[i])[0])	 
			delta = torch.mul(torch.mm(weights[i], deltas_per_layer[iterator]), inverse_activation)
			iterator += 1
			deltas_per_layer.append(delta)

		return list(reversed(deltas_per_layer))  # return a revesed list since we start from end to begining
