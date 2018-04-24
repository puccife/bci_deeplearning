import torch
import torch.nn as nn

import torch.utils.data as dt
from torch.autograd import Variable

import utils.dlc_bci as bci

learning_rate = 0.001
batch_size = 1
num_epochs = 15

class CNNTrainer:

    def __init__(self, cnn):
        self.cnn = cnn
        train_input, train_target = bci.load(root='../../data_bci', one_khz = True)
        test_input, test_target = bci.load(root='../../data_bci', train = False, one_khz = True)
        print(train_input)
        #train_input = Preprocessing().PCA(train_input, k=5)
        #test_input = Preprocessing().PCA(test_input, k=5)

        self.train_dataset = dt.TensorDataset(train_input, train_target)
        self.test_dataset = dt.TensorDataset(test_input, test_target)

        self.train_loader = dt.DataLoader(dataset=self.train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

        self.test_loader = dt.DataLoader(dataset=self.test_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)

    def train_cnn(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.cnn.parameters(), lr=learning_rate)
        # Train the Model
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                images = Variable(images)
                labels = Variable(labels)
                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = self.cnn(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if (i + 1) % 100 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                          % (epoch + 1, num_epochs, i + 1, len(self.train_dataset) // batch_size, loss.data[0]))

        # Test the Model
        self.cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        correct = 0
        total = 0
        for images, labels in self.test_loader:
            images = Variable(images)
            outputs = self.cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

        # Save the Trained Model
        torch.save(self.cnn.state_dict(), '../../model/cnn.pkl')