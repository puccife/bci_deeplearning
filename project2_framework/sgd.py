import torch

class SGD:

	def compute_stoch_gradient(self,backward_pass,forward_pass):
			
		gradient = []
			
		for i in range(0, len(backward_pass)):
			 # gradient for sgd w.r.t weight = dot(x(l),delta(l+1).T)	
			
			weight_loss = torch.mm(backward_pass[i], torch.transpose(forward_pass[i][1], 0,1)) 
			bias_loss = backward_pass[i]  # bias is just equal to delta
			gradient.append((torch.transpose(weight_loss,0,1), bias_loss )) 
			# we transpose final result since we are using column wise ops
			# where the weights are transposed

		return gradient
		
