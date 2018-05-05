import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from .nets.cnn import CNN
from .nets.lstm import LSTM
from .baselines.logistic import LogisticRegression

from visualization.graphviz import GraphViz

from sklearn.metrics import accuracy_score

from tensorboardX import SummaryWriter

class NetTrainer:

    def __init__(self, num_epochs, batch_size, model='CNN'):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = model
        if model == 'CNN':
            self.net = CNN()
        elif model == 'LSTM':
            self.net = LSTM()
        elif model == 'LOG':
            self.net = LogisticRegression()
        else:
            raise RuntimeError('No model found')

    def train(self, train_loader, test_loader):
        self.writer=SummaryWriter()
        args = {
            'model':self.model,
            'batchsize':self.batch_size,
            'epochs':self.num_epochs,
        }
        self.writer.add_text('model', str(args))
        self.criterion = nn.CrossEntropyLoss()
        # Setting optimizer
        optimizer = optim.Adamax(self.net.parameters())
        best_accuracy = 0
        # Training
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            print("\n ------ Epoch n°", epoch + 1, "/", self.num_epochs, "------")
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = Variable(inputs.float())
                labels = Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward(retain_graph=True)
                optimizer.step()
                running_loss += loss.data[0]
            print("\t• Training Loss (avg)", torch.mean(running_loss))
            # Evaluating on training dataset
            self.__evaluate("\t\t- Train", self.net, train_loader, epoch)
            # Returning best test accuracy
            best_accuracy = self.__evaluate("\t\t- Test", self.net, test_loader, epoch, testing=True, best_accuracy=best_accuracy)
        print("Best registered accuracy on test set = ", best_accuracy)
        self.graph_output = outputs
        self.params = dict(self.net.named_parameters())

    # Evaluation
    def __evaluate(self, label, net, loader, epoch, testing=False, best_accuracy=0):
        # Test the Model
        net.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        size = 100 if testing else 316
        running_loss = 0.0
        predictions = []
        correct_targets = []

        for i, (inputs, labels) in enumerate(loader):
            inputs = Variable(inputs.float())
            labels = Variable(labels)

            correct_targets.extend(labels)
            outputs = net(inputs)
            loss = self.criterion(outputs, labels)
            predicted = outputs.max(1)[1]
            predictions.extend(predicted)
            self.writer.add_scalar(label+'/loss', loss, epoch * size + i)
            running_loss += loss.data[0]
        if testing:
            print("\t• Testing Loss (avg)", torch.mean(running_loss))

        accuracy = accuracy_score(correct_targets, predictions)
        self.writer.add_scalar(label + '/accuracy', (accuracy * 100), epoch)
        print(label, ' accuracy of the model : ', accuracy)
        # Save the Trained Model

        if testing:
            if best_accuracy <= accuracy:
                best_accuracy = accuracy
                torch.save(net.state_dict(), '../../model/'+self.model+'.pkl')
            return best_accuracy

    def create_graph(self):
        gv = GraphViz()
        gv.create_graph(self.model, self.graph_output, params=self.params)
        print("Structure saved successfully.")