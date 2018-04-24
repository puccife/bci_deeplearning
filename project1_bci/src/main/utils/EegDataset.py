import dlc_bci as bci
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class EegDataset(Dataset):
    """
    dataset having images as inputs and targets
    """

    def __init__(self, data_path, model, training=True):
        """
        :param data_path: path for the directory containing all input images
        :param target_img_path: path for the directory containing all target images
        :param input_transform: combination of torchVision transforms applied on the input images
        """

        self.input_img_path = data_path
        self.training = training
        self.train_inputs, self.train_targets = bci.load(root=data_path, one_khz=True)
        self.model = model

        if not training:
            self.test_inputs, self.test_targets = bci.load(root=data_path, train=False, one_khz=True)

        means, stds = self.get_dataset_stats()

        self.input_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(means, stds)])

    def __getitem__(self, index):

        inputs = self.train_inputs if self.training else self.test_inputs
        targets = self.train_targets if self.training else self.test_targets

        # applying the transforms to the sample
        if self.input_transform is not None:
            inputs = self.input_transform(inputs[index])

        return self.prepare_data(self.model, inputs) , targets[index]

    def __len__(self):
        return len(self.train_inputs)

    def get_dataset_stats(self):
        samples_means = []
        samples_stds = []
        for inputs in self.train_inputs:
            samples_means.append(inputs.squeeze().mean(dim=0))
            samples_stds.append(inputs.squeeze().std(dim=0))

        # Stacking means/stds of all examples
        stacked_stds = torch.stack(samples_stds, 0)
        stacked_means = torch.stack(samples_means, 0)

        # combined means/stds
        means = stacked_means.mean(dim=0)
        stds = stacked_stds.mean(dim=0)

        return means, stds

    def prepare_data(self, model, input):
        if model in ('SVM', 'log'):
            return input.view((input.size()[0], -1))

        return input

