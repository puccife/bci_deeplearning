import torch
import dlc_practical_prologue as prologue



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


def sigma(x):
	return torch.tanh(x)

def dsigma(x):
	cnst_row,cnst_col = x.size()[0],x.size()[1]
	constantA = torch.Tensor(cnst_row,cnst_col).fill_(2)
	constantB = torch.Tensor(cnst_row,cnst_col).fill_(4)
	constantC = torch.Tensor(cnst_row,cnst_col).fill_(1)    	
	expo = torch.exp(torch.mul(x,constantA))
	derivative = torch.div(torch.mul(constantB,x),torch.pow(torch.add(expo,constantC),2))
	return derivative

def forward_layer(input_data,weight,bias):
   
	s_l = torch.add(torch.mm(input_data,weight),bias)
	x = sigma(s_l)
	return (s_l,x)

def forward_pass(input_data,layers,weights,bias):

 
	layer_data = []
	current_data = input_data
	for layer in range(1,layers):
		
		current_weight_matrix = weights[layer-1]	
		current_bias_matrix = (bias[layer-1]).view(1,-1)
			
		node_transformation = forward_layer(current_data,current_weight_matrix ,current_bias_matrix)
		layer_data.append(node_transformation)
		current_data = node_transformation[0]
		

	return layer_data


def last_layer_loss(train_target,ll_forward_data):

	
	ll_delta = torch.mul(torch.pow(train_target.sub(ll_forward_data[1]),2),dsigma(ll_forward_data[0]))
	return ll_delta
	
def backward_pass(init_delta,ll_forward_data,initial_input,weights,layers):

	deltas_per_layer = []
	deltas_per_layer.append(init_delta)
	iterator = 0	
	for i in reversed(range(layers - 1)):					
		
		inverse_activation = []
		if (i-1 < 0):
			inverse_activation = dsigma(initial_input)
		else:
			inverse_activation = dsigma((ll_forward_data[i-1])[0])	
		delta = torch.mul(torch.mm(deltas_per_layer[iterator],torch.transpose(weights[i],0,1)),inverse_activation)
		iterator+=1
		deltas_per_layer.append(delta)
	return deltas_per_layer


def get_sample(train_input,train_target,index):

	current_sample = torch.transpose(train_input.narrow(0,index,1),0,1)
	current_target = train_target.narrow(0,index,1)
	return current_sample,current_target
	

def compute_stoch_gradient(sample_target,sample_data,weights,bias,layers):
	
	
	layer_data = forward_pass(sample_data,layers,weights,bias)
	ll_delta = last_layer_loss(sample_target,layer_data[len(layer_data)-1])
	backward_data = backward_pass(ll_delta,layer_data,sample_data,weights,layers)
	gradient = []
	for layer_gradient in range(len(layer_data)-1,-1,-1):	
		print(backward_data[layer_gradient].size())		
		print(layer_data[len(layer_data) - layer_gradient][0].size())
		
	# return delta * xl


def stochastic_gradient_descent(train_input,train_target,layers,nodes_per_hidden_layer,initial_w,max_iters,gamma):
    

	w = initial_w
	weights = initialize_weights(layers,nodes_per_hidden_layer)	
	bias = initialize_bias(layers,nodes_per_hidden_layer)
	index = 0
	    
	
	for n_iter in range(max_iters):

		sample_data,sample_target = get_sample(train_input,train_target,index)
		grad = compute_stoch_gradient(sample_target,sample_data,weights,bias,layers)
		# , loss 		
		#w= np.subtract(w,np.dot(gamma0,grad))
		#print("Gradient Descent({bi}/{ti}): objective={l}".format(bi=n_iter, ti=max_iters - 1, l=obj))
		index+=1
	



def neural_net(layers,nodes_per_hidden_layer):

	train_input, train_target, test_input, test_target = load_data()
	data_dimension = train_input.size()[1]
	max_iters = 1
	gamma = 0.2
	initial_w = []
	for layer in range(0,layers - 1):
		initial_w.append(torch.Tensor(1,data_dimension).fill_(0))
	stochastic_gradient_descent(train_input,train_target,layers,nodes_per_hidden_layer,initial_w,max_iters,gamma)

		
	
neural_net(4,[1,3,2,10])













