from nn import *


def augment_data(inputTensor):
	
	sqrt_sum = torch.sum(torch.pow(inputTensor,2), dim=1).view(-1,1)
	augmented_data = torch.cat((inputTensor,sqrt_sum), 1)
	return augmented_data

def normalize(data,mean,std):
	return (data - mean)/std

def run():
	train_input,train_target = generate_data(1000,2)
	test_input,test_target = generate_data(1000,2)	
	run_network(train_input,train_target,test_input,test_target)

run()


def check_distribution(train_input,train_target):
	import numpy
	import pylab

	np_in = train_input.numpy()
	np_tr = (train_target.numpy()).reshape((-1,1))

	z = np.hstack((np_in,np_tr))
	colors = [ int(i[2]) for i in z]
	pylab.scatter(np_in[:,0], np_in[:,1],c=colors)
	pylab.show()
