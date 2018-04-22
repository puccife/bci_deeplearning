import torch
import dlc_practical_prologue as prologue
import random

################ DATA ##################################################

def load_data():
	train_input, train_target, test_input, test_target = prologue.load_data(cifar = False, one_hot_labels = True)
	return train_input, train_target, test_input, test_target


def initialize_weights(layers,nodes_per_hidden_layer):

	weights = []
	for layer in range(0,layers-1):
		neurons_current_layer = nodes_per_hidden_layer[layer]
		neurons_next_layer = nodes_per_hidden_layer[layer+1]
		weight_matrix = torch.Tensor(neurons_current_layer,neurons_next_layer).normal_()
		weights.append(weight_matrix)
	return weights

def initialize_bias(layers,nodes_per_hidden_layer):
	bias = []
	for layer in range(1,layers):
		current_layer_bias = torch.Tensor(nodes_per_hidden_layer[layer]).normal_()
		bias.append(current_layer_bias)
	return bias

def get_sample(train_input,train_target):

	row_dimension = train_input.size()[0]
	random_index = random.randint(0,row_dimension-1)
	current_sample = train_input.narrow(0,random_index,1)
	current_target = train_target.narrow(0,random_index,1)
	return current_sample,current_target

########################################################################

############### Activations ###########################################

def sigma(x):
	return x.apply_(lambda values:max(values,0))

def dsigma(x):
	return x.apply_(lambda x: 0 if x <= 0 else 1)

def tanh(x):
	return torch.tanh(x)

def dtanh(x):
	cnst_row,cnst_col = x.size()[0],x.size()[1]
	constantA = torch.Tensor(cnst_row,cnst_col).fill_(2)
	constantB = torch.Tensor(cnst_row,cnst_col).fill_(1)    	
	expo = torch.exp(torch.mul(x,constantA))
	derivative = torch.div(constantA,constantB.add(expo))
	return derivative
	

####################################################################

###################LOSS ###################################################

def MSE(prediction,target):
	
	return (prediction - target).pow(2).sum()

############################################################################


##################Backprop#########################################################

def forward_layer(input_data,weight,bias):
   
	s_l = torch.add(torch.mm(input_data,weight),bias)
	x = sigma(s_l)
	return (s_l,x)

def forward_pass(input_data,layers,weights,bias):

 
	layer_data = [(input_data,sigma(input_data))]
	current_data = input_data
	for layer in range(1,layers):
		
		current_weight_matrix = weights[layer-1]	
		current_bias_matrix = (bias[layer-1]).view(1,-1)
			
		node_transformation = forward_layer(current_data,current_weight_matrix ,current_bias_matrix)
		layer_data.append(node_transformation)
		current_data = node_transformation[0]
		

	return layer_data


def last_layer_loss(train_target,ll_forward_data):

	cnst_row,cnst_col = train_target.size()[0],train_target.size()[1]
	constant = torch.Tensor(cnst_row,cnst_col).fill_(-2)
	ll_delta = torch.mul(constant,torch.mul(torch.pow(train_target.sub(ll_forward_data[1]),2),dsigma(ll_forward_data[0])))
	return ll_delta


def backward_pass(init_delta,ll_forward_data,initial_input,weights,layers):

	deltas_per_layer = []
	deltas_per_layer.append(init_delta)
	iterator = 0	
	for i in reversed(range(layers - 1)):					
		inverse_activation = dsigma((ll_forward_data[i])[0])
		delta = torch.mul(torch.mm(deltas_per_layer[iterator],torch.transpose(weights[i],0,1)),inverse_activation)
		iterator+=1
		deltas_per_layer.append(delta)
	return list(reversed(deltas_per_layer))
	


##################################################################################


########################### SGD ##################################################

def compute_stoch_gradient(sample_target,sample_data,weights,bias,layers):
	
	
	layer_data = forward_pass(sample_data,layers,weights,bias)
	ll_delta = last_layer_loss(sample_target,layer_data[len(layer_data)-1])
	backward_data = backward_pass(ll_delta,layer_data,sample_data,weights,layers)
	
	weight_gradients = []
	bias_gradients = []

	for i in range(0,len(weights)):
		weight_loss = torch.mm(torch.transpose(layer_data[i][1],0,1),backward_data[i+1])
		bias_loss = backward_data[i+1][0,:]
		weight_gradients.append(weight_loss)
		bias_gradients.append(bias_loss)

	return weight_gradients,bias_gradients,layer_data


def zero_gradient(row_dimension,col_dimension):
	return torch.Tensor(row_dimension,col_dimension).fill_(0)

def stochastic_gradient_descent(train_input,train_target,layers,nodes_per_hidden_layer,max_iters,gamma):
    

	weights = initialize_weights(layers,nodes_per_hidden_layer)	
	bias = initialize_bias(layers,nodes_per_hidden_layer)
	train_error = 0	

	for n_iter in range(max_iters):

		sample_data,sample_target = get_sample(train_input,train_target)

		gradient,bias_gradient,layer_data = compute_stoch_gradient(sample_target,sample_data,weights,bias,layers)
		
		train_error = MSE(layer_data[len(layer_data) - 1][1],sample_target)
		print("Step = {} and loss = {} ".format(n_iter,train_error))
		
		
		for i in range(0,len(weights)):
			
			cnst_row,cnst_col = gradient[i].size()[0],gradient[i].size()[1]
			learning_rate = torch.Tensor(cnst_row,cnst_col).fill_(gamma)
			#cnst_rows = bias_gradient[i].size()[0]
			#bias_rate = torch.Tensor(cnst_rows).fill_(gamma)
			weight_loss = torch.mul(learning_rate,gradient[i])
			#bias_loss = torch.mul(bias_rate,bias_gradient[i])
			weights[i] = weights[i].sub(weight_loss)
			#bias[i] = bias[i].sub(bias_loss)
			vanishing_gradient = (weights[i] != weights[i])			
			if(vanishing_gradient.any()):				
				dimensions = weights[i].size()
				weights[i] = zero_gradient(dimensions[0],dimensions[1])
			
			
	
	return weights,bias
	

#########################################################################

#####################TEST##############################################

'''
def test_network(test_input,test_target,weights,bias,layers,max_iters):

	test_error = 0
	index = 0
	for n_iter in range(max_iters):

		sample_data,sample_target = get_sample(test_input,test_target,index)
		layer_data = forward_pass(sample_data,layers,weights,bias)
		
		train_error = MSE(layer_data[len(layer_data) - 1][1],sample_target)
		print("Step = {} and loss = {} ".format(n_iter,train_error)
		index = index + 1
'''	

##############################################################

##################### NN #############################################


def neural_net(layers,nodes_per_hidden_layer):

	train_input, train_target, test_input, test_target = load_data()
	data_dimension = train_input.size()[1]
	nodes_per_hidden_layer = [data_dimension] + nodes_per_hidden_layer 
	max_iters = 100
	gamma = 0.2
	weights,bias = stochastic_gradient_descent(train_input,train_target,layers,nodes_per_hidden_layer,max_iters,gamma)
	
neural_net(4,[3,2,10])












