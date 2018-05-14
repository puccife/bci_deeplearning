import torch

class DataLoader():
    
    def __init__(self, input_tensor, target_tensor, batch_size):
        self.input_tensor = input_tensor
        self.target_tensor = target_tensor
        self.batch_size = batch_size
    
    def get_loader(self):
        
        input_ = self.input_tensor.clone()
        target_ = self.target_tensor.clone()
        
        size = len(input_)
        perm = torch.randperm(size)
        
        input_ = input_[perm]
        target_ = target_[perm]
        
        for index in range(0, size, self.batch_size):
            if len(input_) - index < self.batch_size:
                yield input_.narrow(0, index, len(self.input_tensor) - index), target_.narrow(0, index, len(self.input_tensor) - index)
            else:
                yield input_.narrow(0, index, self.batch_size), target_.narrow(0, index, self.batch_size)
