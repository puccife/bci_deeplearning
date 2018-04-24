import dlc_bci as bci
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


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

        self.input_img_path = data_path
        self.training = training
        self.model = model
        self.train_inputs, self.train_targets = bci.load(root=data_path, one_khz=True)
        means = self.train_inputs.mean(dim=1).mean(dim=0)
        stds = self.train_inputs.std(dim=1).mean(dim=0)
        self.train_inputs = (self.train_inputs - means) / stds

        if not training:
            self.test_inputs, self.test_targets = bci.load(root=data_path, train=False, one_khz=True)
            self.test_inputs = (self.test_inputs - means) / stds

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
