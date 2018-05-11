import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.autograd import Variable

from sklearn.metrics import accuracy_score

from .nets.cnn import CNN
from .nets.tsnet import TSnet
from .nets.lstm import LSTM
from .baselines.logistic import LogisticRegression

# from torchsummary import summary
# from visualization.graphviz import GraphViz
# from tensorboardX import SummaryWriter

"""
Trainer for the models
"""
class NetTrainer:

    def __init__(self, num_epochs, batch_size, weight_decay, model='TS', pretrained=False):

        """
        Initialization for the net trainer
        :param num_epochs: how many epochs of training
        :param batch_size: batch size dimension
        :param weight_decay: weight decay value
        :param model: model to train between [LSTM, LOG, CNN, TS]
        :param pretrained: parameter to load weights of pretrained network
        """

        # Loss of the models
        self.criterion = nn.CrossEntropyLoss()

        # Storing hyperparams
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.pretrained = pretrained
        self.model = model

        # Setting optimizers for each model
        if model == 'CNN':
            self.net = CNN()
            self.optimizer = optim.Adamax(self.net.parameters(), weight_decay=0)
        elif model == 'LSTM':
            self.net = LSTM()
            self.optimizer = optim.Adamax(self.net.parameters(), weight_decay=self.weight_decay)
        elif model == 'LOG':
            self.net = LogisticRegression()
            self.optimizer = optim.Adamax(self.net.parameters(), weight_decay=self.weight_decay)
        elif model == 'TS':
            self.net = TSnet()
            self.optimizer = optim.Adam(self.net.parameters(), weight_decay=self.weight_decay, lr=0.001)
            self.scheduler = MultiStepLR(self.optimizer, milestones=[135, 145], gamma=0.1)
            if pretrained:
                self.net.load_state_dict(torch.load('../../model/tsnet_best.pkl'))
        else:
            raise RuntimeError('No model found')

    def train(self, train_loader, test_loader):
        """
        Methods that train the model given two loaders
        :param train_loader: training set loader
        :param test_loader: testing set loader
        :return: best accuracy and loss of the model
        """

        # Used to write on tensorboard
        # self.writer=SummaryWriter()
        # args = {
        #     'model':self.model,
        #     'batchsize':self.batch_size,
        #     'epochs':self.num_epochs,
        # }
        # self.writer.add_text('model', str(args))
        # summary(self.net, (28, 50))

        # variables used to store the performances of the models
        best_accuracy = 0
        best_loss = 333

        # Training process
        for epoch in range(self.num_epochs):

            # Setting scheduler if model = TSNET
            if self.model == 'TS':
                self.scheduler.step()

            # Variable to store the running loss
            running_loss = 0.0
            if not self.pretrained:
                for i, (inputs, labels) in enumerate(train_loader):

                    # Loading inputs and labels
                    inputs = Variable(inputs.float())
                    labels = Variable(labels)
                    # zeroing gradients parameter
                    self.optimizer.zero_grad()

                    # forward
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)

                    # backward
                    loss.backward()

                    # optimize
                    self.optimizer.step()
                    running_loss += loss

            # Eval training
            self.__evaluate("\t\t- Train", self.net, train_loader, epoch)
            # Eval testing
            best_accuracy, best_loss = self.__evaluate("\t\t- Test", self.net, test_loader, epoch, testing=True, best_accuracy=best_accuracy, best_loss=best_loss)
            if self.pretrained:
                break
        # Used to build the graph of the network
        #self.graph_output = outputs
        #self.params = dict(self.net.named_parameters())
        return best_accuracy, best_loss

    def __evaluate(self, label, net, loader, epoch, testing=False, best_accuracy=0, best_loss=333):
        """
        Method used to evaluate the network at every epoch
        :param label: testing or training -- for debug purpose
        :param net: nn 
        :param loader: dataset loader (either test or train)
        :param epoch: epoch number for debug purpose
        :param testing: if it is testing
        :param best_accuracy: best accuracy registered so far
        :param best_loss: best loss registered so far
        :return: besst accuracy and loss from evaluation
        """
        # Changing model to evaluate
        net.eval()

        # Loss of the network
        running_loss = 0.0

        # Vectors used to calculate accuracy
        predictions = []
        correct_targets = []
        for i, (inputs, labels) in enumerate(loader):

            # inputs and labels
            inputs = Variable(inputs.float())
            labels = Variable(labels)

            # forward
            outputs = net(inputs)
            loss = self.criterion(outputs, labels)
            predicted = outputs.max(1)[1]
            predictions.extend(predicted.data.numpy())
            correct_targets.extend(labels.data.numpy())
            #self.writer.add_scalar(label+'/loss', loss, epoch * (100 if testing else 316) + i)
            running_loss += loss#.data[0]

        # calculating avg loss
        avgloss = running_loss / len(loader)
        #print("\t• "+label+" Loss (avg)", avgloss)

        # calculating accuracy score
        accuracy = accuracy_score(correct_targets, predictions)
        #self.writer.add_scalar(label + '/accuracy', (accuracy * 100), epoch)
        #print(label, ' accuracy of the model : ', accuracy)
        # Returning best accuracy and loss
        # Save the Trained Model if better
        if testing:
            if avgloss <= best_loss:
                best_loss = avgloss
            if best_accuracy <= accuracy:
                best_accuracy = accuracy
                torch.save(net.state_dict(), '../../model/'+self.model+'.pkl')
            return best_accuracy, best_loss