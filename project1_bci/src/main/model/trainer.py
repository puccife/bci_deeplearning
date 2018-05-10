import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.autograd import Variable

from .nets.cnn import CNN
from .nets.res import Res
from .nets.deepcnn import TheNet
from .nets.lstm import LSTM
from .baselines.logistic import LogisticRegression

# from visualization.graphviz import GraphViz

from torchvision.models.resnet import BasicBlock, model_urls, model_zoo
from torchvision import models

from sklearn.metrics import accuracy_score

from tensorboardX import SummaryWriter

class NetTrainer:

    def __init__(self, num_epochs, batch_size, weight_decay, model='CNN', pretrained=False):
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.model = model
        if model == 'CNN':
            self.net = CNN()
            self.optimizer = optim.Adamax(self.net.parameters(), weight_decay=0)
        elif model == 'LSTM':
            self.net = LSTM()
            self.optimizer = optim.Adamax(self.net.parameters(), weight_decay=self.weight_decay)
        elif model == 'LOG':
            self.net = LogisticRegression()
            self.optimizer = optim.Adamax(self.net.parameters(), weight_decay=self.weight_decay)
        elif model == 'CONV2D':
            # self.net = Res(BasicBlock, [2, 2])
            # self.optimizer = optim.Adamax(self.net.parameters(), lr=0.00001)
            self.net = TheNet()
            self.optimizer = optim.Adam(self.net.parameters(), weight_decay=self.weight_decay, lr=0.005)
            self.scheduler = StepLR(self.optimizer, step_size=15, gamma=0.5)
            if pretrained:
                self.net.load_state_dict(torch.load('../../model/CONV2D_best.pkl'))
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

        # Setting optimizer
        best_accuracy = 0
        # Training
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            print("\n ------ Epoch n°", epoch + 1, "/", self.num_epochs, "------")
            self.scheduler.step()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = Variable(inputs.float())
                labels = Variable(labels)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                running_loss += loss.data[0]
            # Evaluating on training dataset
            self.__evaluate("\t\t- Train", self.net, train_loader, epoch)
            # Returning best test accuracy
            best_accuracy, val_loss = self.__evaluate("\t\t- Test", self.net, test_loader, epoch, testing=True, best_accuracy=best_accuracy)

        print("Best registered accuracy on test set = ", best_accuracy)
        self.graph_output = outputs
        self.params = dict(self.net.named_parameters())

    # Evaluation
    def __evaluate(self, label, net, loader, epoch, testing=False, best_accuracy=0):
        # Test the Model
        net.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        running_loss = 0.0
        predictions = []
        correct_targets = []
        for i, (inputs, labels) in enumerate(loader):
            inputs = Variable(inputs.float())
            labels = Variable(labels)
            outputs = net(inputs)
            loss = self.criterion(outputs, labels)
            predicted = outputs.max(1)[1]
            predictions.extend(predicted.data.numpy())
            correct_targets.extend(labels.data.numpy())
            self.writer.add_scalar(label+'/loss', loss, epoch * (100 if testing else 316) + i)
            running_loss += loss.data[0]
        print("\t• "+label+" Loss (avg)", running_loss / len(loader))
        accuracy = accuracy_score(correct_targets, predictions)
        self.writer.add_scalar(label + '/accuracy', (accuracy * 100), epoch)
        print(label, ' accuracy of the model : ', accuracy)
        # Save the Trained Model

        if testing:
            if best_accuracy <= accuracy:
                best_accuracy = accuracy
                torch.save(net.state_dict(), '../../model/'+self.model+'.pkl')
            return best_accuracy, running_loss / len(loader)

    def create_graph(self):
        gv = GraphViz()
        gv.create_graph(self.model, self.graph_output, params=self.params)
        print("Structure saved successfully.")