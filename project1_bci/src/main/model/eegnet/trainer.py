import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import os

from .net import EEGNet
from visualization.graphviz import GraphViz

class EEGNetTrainer:

    def __init__(self, num_epochs, batch_size):
        self.num_epochs = num_epochs
        self.batch_size = batch_size


    def train(self, train_loader, test_loader):
        net = EEGNet()
        criterion = nn.BCELoss()
        # Setting optimizer
        optimizer = optim.Adam(net.parameters())
        best_accuracy = 0
        # Training
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            print("\n ------ Epoch n°", epoch + 1, "/", self.num_epochs, "------")
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = Variable(inputs)
                labels = Variable(labels.float().view(self.batch_size, 1))
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward(retain_graph=True)
                optimizer.step()
                running_loss += loss.data[0]
            print("\t• Training Loss ", running_loss)
            # Evaluating on training dataset
            self.__evaluate("\t\t- Train", net, train_loader)
            # Returning best test accuracy
            best_accuracy = self.__evaluate("\t\t- Test", net, test_loader, testing=True, best_accuracy=best_accuracy)
        print("Finished.")
        self.graph_output = outputs
        self.params = dict(net.named_parameters())

    # Evaluation
    def __evaluate(self, label, net, loader, testing=False, best_accuracy=0):
        # Test the Model
        net.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        correct = 0
        total = 0
        for inputs, labels in loader:
            inputs = Variable(inputs)
            labels = labels.float().view(self.batch_size, 1)
            outputs = net(inputs)
            predicted = torch.round(outputs.data)
            total += labels.size(0)
            correct += (predicted == labels.float()).sum()
        accuracy = (100 * correct / total)
        print(label, ' accuracy of the model : %d %%' % accuracy)
        # Save the Trained Model
        if testing:
            if best_accuracy <= accuracy:
                best_accuracy = accuracy
                torch.save(net.state_dict(), '../../model/eegnet.pkl')
            return best_accuracy

    def create_graph(self):
        gv = GraphViz()
        gv.create_graph('EEGnet', self.graph_output, params=self.params)
        print("Structure saved successfully.")