import torch
import math

################ DATA ##################################################


def generate_data(row_dimension,col_dimension):	
	data_input = torch.Tensor(row_dimension,col_dimension).uniform_() # generate unifromly distributed points
	data_output = torch.arange(0,row_dimension) # generate a tensor having indices of previous tensor
	 # apply a function on indices if input belongs to circle assign 0 otherwise assign 1
	data_output.apply_(lambda index: 0 if inCircle(data_input[int(index)]) else 1)
	return data_input,data_output

# check if (x,y) point is inside circle or not
def inCircle(values):
	radius_squared = 1/(2*math.pi)
	x_squared = math.pow(values[0],2)
	y_squared = math.pow(values[1],2)
	if( (x_squared + y_squared) <= radius_squared):
		return True
	else:
		return False


# method to initialize random weights
def initialize_weights(layers,nodes_per_hidden_layer):

	weights = []
	for layer in range(0,layers-1):
		neurons_current_layer = nodes_per_hidden_layer[layer] # find dimension of current layer
		neurons_next_layer = nodes_per_hidden_layer[layer+1] # find dimension of next layer
		# generate a normaly distributed tensor having dimensions of (current layer,next layer)
		weight_matrix = torch.Tensor(neurons_current_layer,neurons_next_layer).normal_() 
		weights.append(weight_matrix)
	return weights

# same as before but bias only depends on current layer 1d tensor
def initialize_bias(layers,nodes_per_hidden_layer):
	bias = []
	for layer in range(1,layers):
		current_layer_bias = torch.Tensor(nodes_per_hidden_layer[layer]).normal_()
		bias.append(current_layer_bias)
	return bias

# returns mini batch of data from start till the start + batch size
def get_data(train_input,train_target,start,mini_batch_size):

	current_sample = train_input.narrow(0, start, mini_batch_size)
	current_target = train_target.narrow(0,start, mini_batch_size)
	return current_sample,current_target

########################################################################

############### Activations ###########################################

# calculates reLU activation
def sigma(x):
	scalar = torch.FloatTensor(1,1).fill_(0)
	reLU = torch.max(x, scalar.expand_as(x)) # check max between x and 0
	return reLU

def dsigma(x):
	result = sigma(x) # get reLU
	result[result>0] = 1 # replace all positive values by one thus corresponding to  derivative of reLU
	return result


'''
def tanh(x):
	return torch.tanh(x)

def dtanh(x):
	cnst_row,cnst_col = x.size()[0],x.size()[1]
	constantA = torch.Tensor(cnst_row,cnst_col).fill_(2)
	constantB = torch.Tensor(cnst_row,cnst_col).fill_(1)    	
	expo = torch.exp(torch.mul(x,constantA))
	derivative = torch.div(constantA,constantB.add(expo))
	return derivative
'''	

####################################################################

###################LOSS ###################################################

def MSE(prediction,target):
	return (prediction - target).pow(2).sum() # MSE loss

# an implementation of softmax
def apply_sigmoid(x):
	exponential = torch.exp(x)
	total_sum = torch.sum(exponential)
	softmax = exponential.apply_(lambda expos: expos/total_sum) # softmax is equal to e^xj/(sum over k e^xk)
	return softmax

	
############################################################################


##################Backprop#########################################################

def forward_layer(input_data,weight,bias):
   
	s_l = torch.add(torch.mm(weight,input_data),bias) # compute layerwise z before apply activation : (w(l).x(l-1) + b(l)
	x = sigma(s_l) # apply sigmoid on computed z
	return (s_l,x)

def forward_pass(input_data,layers,weights,bias):

 
	layer_data = [(input_data,input_data)] # add initial input to list storing tuple of layerwise data before and after applying activation per layer
	current_data = input_data
	for layer in range(1,layers):
		
		current_weight_matrix = torch.transpose(weights[layer-1],0,1)	
		current_bias_matrix = (bias[layer-1]).view(-1,1)
		
		node_transformation = forward_layer(current_data,current_weight_matrix ,current_bias_matrix) # apply calculation of forward pass
		layer_data.append(node_transformation)
		current_data = node_transformation[0]
		

	return layer_data


# last layer loss for mse is -2(yn -xn)^2 dot(phi_prime(z))
def last_layer_loss(train_target,predicted_label,last_layer_vec):

	cnst_row = train_target.size()[0]
	constant = torch.Tensor(cnst_row).fill_(-2) # fill a tensor with -2 (only way to multiply by constant)
	test = 	torch.pow(train_target.sub(predicted_label),2)	# compute squared difference between prediction and actual label
	# compute last layer delta
	ll_delta = torch.mul(constant,torch.mul(torch.pow(train_target.sub(predicted_label),2),dsigma(last_layer_vec)))
	return ll_delta


# compute backward pass in backprop algorithm
# backward layer = component_wise_product(dot(w(l),delta(l+1)),phi_prime(z before activation))
def backward_pass(init_delta,ll_forward_data,initial_input,weights,layers):

	deltas_per_layer = []
	deltas_per_layer.append(init_delta) # append last layer delta
	iterator = 0	
	for i in reversed(range(layers - 1)):					
		inverse_activation = dsigma((ll_forward_data[i])[0]) # compute derivative of delta before applying activation
		delta = torch.mul(torch.mm(weights[i],deltas_per_layer[iterator]),inverse_activation) # compute back pass
		iterator+=1
		deltas_per_layer.append(delta)
	return list(reversed(deltas_per_layer)) # return a revesed list since we start from end to begining
	


##################################################################################


########################### SGD ##################################################

def compute_stoch_gradient(sample_target,sample_data,weights,bias,layers):
	
	
	layer_data = forward_pass(sample_data,layers,weights,bias) # compute forward pass
	soft_max_compute = apply_sigmoid(layer_data[len(layer_data) - 1][1]) # apply softmax to compute most probable label
	mlp_prediction = ((torch.max(soft_max_compute,0))[1]).float() # label corresponds to the index of the maximum
	
	mse_loss = MSE(mlp_prediction,sample_target)	# compute mse and log
	print("Loss = {} ".format(mse_loss))
	print("Prediction = {} target = {}".format(mlp_prediction[0],sample_target[0]))

	ll_delta = last_layer_loss(sample_target,mlp_prediction,layer_data[len(layer_data) - 1][0]) # compute last layer delta
	backward_data = backward_pass(ll_delta,layer_data,sample_data,weights,layers) # compute backward pass
	gradient = []
	for i in range(0,len(backward_data) - 1):
		weight_loss = torch.mm(backward_data[i],torch.transpose(layer_data[i+1][0],0,1)) # gradient for sgd w.r.t weight = dot(x(l),delta(l+1).T)
		bias_loss = backward_data[i+1].view(-1) # bias is just equal to delta
		gradient.append((weight_loss,bias_loss))

	return gradient
	


def stochastic_gradient_descent(train_input,train_target,layers,nodes_per_hidden_layer,gamma,mini_batch_size):
    
	# initialize parameters
	weights = initialize_weights(layers,nodes_per_hidden_layer)	
	bias = initialize_bias(layers,nodes_per_hidden_layer)
	train_error = 0	

	for b in range(0,  train_input.size(0) , mini_batch_size):

		sample_data,sample_target = get_data(train_input,train_target,b,mini_batch_size) # get mini batch
		column_sample = torch.transpose(sample_data,0,1) # compute column vector of input
		gradient = compute_stoch_gradient(sample_target,column_sample,weights,bias,layers) # get gradient
	
				
		for i in range(0,len(weights)):
			
			cnst_row,cnst_col = weights[i].size()[0],weights[i].size()[1]
			learning_rate = torch.Tensor(cnst_row,cnst_col).fill_(gamma) # get stepsize
			weight_loss = torch.mul(learning_rate,gradient[i][0]) # multiply step size with gradient
			weights[i].sub_(weight_loss)	# subtract gradient from loss	
			vanishing_gradient = (weights[i] != weights[i])			
			if(vanishing_gradient.any()):
				print("hello nan's every where ")
	
	return weights,bias
	


##################### NN #############################################



def neural_net(layers,nodes_per_hidden_layer):

	
	num_samples = 1000	
	gamma = 0.001
	mini_batch_size = 100
	train_input,train_target = generate_data(num_samples,2)
	test_input,test_target = generate_data(num_samples,2)		
	data_dimension = train_input.size()[1]
	nodes_per_hidden_layer = [data_dimension] + nodes_per_hidden_layer 
	stochastic_gradient_descent(train_input,train_target,layers,nodes_per_hidden_layer,gamma,mini_batch_size)
	
	
		

neural_net(5,[25,25,25,2])














