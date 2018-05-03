import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .nets.cnn import CNN
from .nets.lstm import LSTM

from visualization.graphviz import GraphViz

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
        else:
            raise RuntimeError('No model found')

    def train(self, train_loader, test_loader):
        self.writer=SummaryWriter()
        self.writer.add_text('model', self.model)
        self.criterion = nn.CrossEntropyLoss()
        # Setting optimizer
        optimizer = optim.Adamax(self.net.parameters())
        best_accuracy = 0
        # Training
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            print("\n ------ Epoch n°", epoch + 1, "/", self.num_epochs, "------")
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = Variable(inputs)
                labels = Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward(retain_graph=True)
                optimizer.step()
                running_loss += loss.data[0]
            print("\t• Training Loss ", running_loss)
            # Evaluating on training dataset
            self.__evaluate("\t\t- Train", self.net, train_loader, epoch)
            # Returning best test accuracy
            best_accuracy = self.__evaluate("\t\t- Test", self.net, test_loader, epoch, testing=True, best_accuracy=best_accuracy)
        self.graph_output = outputs
        self.params = dict(self.net.named_parameters())

    # Evaluation
    def __evaluate(self, label, net, loader, epoch, testing=False, best_accuracy=0):
        # Test the Model
        net.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        size = 100 if testing else 316
        correct = 0
        total = 0
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(loader):
            inputs = Variable(inputs)
            labels = Variable(labels)
            outputs = net(inputs)
            loss = self.criterion(outputs, labels)
            predicted = outputs.max(1)[1]
            self.writer.add_scalar(label+'/loss', loss, epoch * size + i)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            running_loss += loss.data[0]
        if testing:
            print("\t• Testing Loss ", running_loss)
        accuracy = (100 * correct / total)
        self.writer.add_scalar(label + '/accuracy', accuracy, epoch)
        print(label, ' accuracy of the model : %d %%' % accuracy)
        # Save the Trained Model
        if testing:
            if best_accuracy <= accuracy:
                best_accuracy = accuracy
                torch.save(net.state_dict(), '../../model/eegnet.pkl')
            return best_accuracy

    def create_graph(self):
        gv = GraphViz()
        gv.create_graph(self.model, self.graph_output, params=self.params)
        print("Structure saved successfully.")