import torch

class Activation:

	def forward(self,inputTensor):
		pass

	def derrivative(self,inputTensor):
		pass


class ReLU(Activation):

	# compute relu activation by 
	def forward(self,inputTensor):
		scalar = torch.FloatTensor(1, 1).fill_(0.0)
		reLU = torch.max(inputTensor, scalar.expand_as(inputTensor))  # check max between x and 0
		return reLU

	# check that relu computation is correct
	def test(self,x,y):
		m = torch.nn.ReLU()
		z = m(x)
		print(z.data - y)

	# derivative of relu is equal to 
	# derivative = 0 if x <= 0
	# derivative = 1 if x>0
	# we compute relu and replace every nonzero value with a 1
	def derrivative(self,inputTensor):
		result = self.forward(inputTensor)  # get reLU
		result[result > 0.0] = 1.0  # replace all positive values by one thus corresponding to  derivative of reLU
		return result

class Tanh(Activation):

	# hyperbolic tangent activation
	# e^(2x) - 1/ e^(2x) + 1
	def forward(self,inputTensor):
		exponential = torch.exp(inputTensor * 2)
		res = (exponential - 1)/(exponential + 1)
		#print(res)
		return res

	# check that the activation is correctly computed
	def test(self,x,y):
		q = torch.tanh(x)
		print(q - y)

	# derivative of tanh
	# 4 * e^(2x)/(e^(2x) + 1)^2
	def derrivative(self,inputTensor):
		exponential = torch.exp(inputTensor * 2)
		return (4 * exponential)/ (torch.pow(exponential + 1,2))



