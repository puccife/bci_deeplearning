import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import tensorly as tl
from tensorly.decomposition import parafac, tucker

class EegDataset(Dataset):
    """
    dataset having images as inputs and targets
    """

    def __init__(self, data_path, model='SVM', training=True):
        """
        :param data_path: path for the directory containing all input images
        :param target_img_path: path for the directory containing all target images
        :param input_transform: combination of torchVision transforms applied on the input images
        """

        self.training = training
        self.model = model
        self.train_inputs, self.train_targets = bci.load(root=data_path, one_khz=True)
        self.test_inputs, self.test_targets = bci.load(root=data_path, train=False, one_khz=True)

        means = self.train_inputs.mean(dim=1).mean(dim=0)
        stds = self.train_inputs.std(dim=1).mean(dim=0)
        self.train_inputs = (self.train_inputs - means) / stds
        self.test_inputs = (self.test_inputs - means) / stds
            
        inputs = torch.cat((self.train_inputs, self.test_inputs), 0)
        self.train_inputs, self.test_input = self.decompose_tucker(inputs)
        self.prepare_data()

    def __getitem__(self, index):

        inputs = self.train_inputs if self.training else self.test_inputs
        targets = self.train_targets if self.training else self.test_targets

        return inputs[index], targets[index]

    def __len__(self):
        return len(self.train_inputs) if self.training else len(self.test_inputs)

    def prepare_data(self):
        if self.model in ('SVM', 'log'):
            self.train_inputs = self.train_inputs.view((self.train_inputs.size()[0], -1))

            if not self.training:
                self.test_inputs = self.test_inputs.view((self.test_inputs.size()[0], -1))
        
        if self.model in ('EEGnet'):
                self.train_inputs = self.train_inputs.contiguous().view(self.train_inputs.shape[0], 1, self.train_inputs.shape[1], self.train_inputs.shape[2])
                self.test_inputs = self.test_inputs.contiguous().view(self.test_inputs.shape[0], 1, self.test_inputs.shape[1], self.test_inputs.shape[2])

    def decompose_tucker(self, inputs, decomposed_channels=3):
        tucker_rank = [self.train_inputs.shape[0]+self.test_inputs.shape[0], self.train_inputs.shape[1], decomposed_channels]
        inputs = torch.cat((self.train_inputs, self.test_inputs), 0)
        inputs = inputs.numpy()
        core, tucker_factors = tucker(inputs, ranks=tucker_rank, n_iter_max=100, init="svd", tol=1e-06, random_state=None, verbose=True)
        core = torch.Tensor(core)
        return core[:self.train_inputs.shape[0]], core[-self.test_inputs.shape[0]:]