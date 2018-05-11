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

class NetTrainer:

    def __init__(self, num_epochs, batch_size, weight_decay, model='TS', pretrained=False):
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.pretrained = pretrained
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
        elif model == 'TS':
            self.net = TSnet()
            self.optimizer = optim.Adam(self.net.parameters(), weight_decay=self.weight_decay, lr=0.001)
            self.scheduler = MultiStepLR(self.optimizer, milestones=[135, 145], gamma=0.1)
            if pretrained:
                self.net.load_state_dict(torch.load('../../model/tsnet_best.pkl'))
        else:
            raise RuntimeError('No model found')

    def train(self, train_loader, test_loader):
        #self.writer=SummaryWriter()
        args = {
            'model':self.model,
            'batchsize':self.batch_size,
            'epochs':self.num_epochs,
        }
        #self.writer.add_text('model', str(args))
        #summary(self.net, (28, 50))
        best_accuracy = 0
        best_loss = 333
        # Training
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            #print("\n ------ Epoch n°", epoch + 1, "/", self.num_epochs, "------")
            if self.model == 'TS':
                self.scheduler.step()
            running_loss = 0.0
            if not self.pretrained:
                for i, (inputs, labels) in enumerate(train_loader):
                    inputs = Variable(inputs.float())
                    labels = Variable(labels)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss
            # Evaluating on training dataset
            self.__evaluate("\t\t- Train", self.net, train_loader, epoch)
            # Returning best test accuracy
            best_accuracy, best_loss = self.__evaluate("\t\t- Test", self.net, test_loader, epoch, testing=True, best_accuracy=best_accuracy, best_loss=best_loss)
            if self.pretrained:
                break
        #self.graph_output = outputs
        #self.params = dict(self.net.named_parameters())
        return best_accuracy, best_loss

    # Evaluation
    def __evaluate(self, label, net, loader, epoch, testing=False, best_accuracy=0, best_loss=333):
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
            #self.writer.add_scalar(label+'/loss', loss, epoch * (100 if testing else 316) + i)
            running_loss += loss#.data[0]
        avgloss = running_loss / len(loader)
        #print("\t• "+label+" Loss (avg)", avgloss)
        accuracy = accuracy_score(correct_targets, predictions)
        #self.writer.add_scalar(label + '/accuracy', (accuracy * 100), epoch)
        #print(label, ' accuracy of the model : ', accuracy)
        # Save the Trained Model
        if testing:
            if avgloss <= best_loss:
                best_loss = avgloss
            if best_accuracy <= accuracy:
                best_accuracy = accuracy
                torch.save(net.state_dict(), '../../model/'+self.model+'.pkl')
            return best_accuracy, best_loss