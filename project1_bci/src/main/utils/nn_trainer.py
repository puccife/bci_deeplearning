from model.eegnet.trainer import EEGNetTrainer


class Trainer:

    def __init__(self, model, batch_size, epochs):
        if model == 'EEG':
            self.trainer = EEGNetTrainer(num_epochs=epochs, batch_size=batch_size)
        else:
            raise RuntimeError('No model found')

    def train(self, train_loader, test_loader):
        self.trainer.train(train_loader=train_loader, test_loader=test_loader)

    def create_graph(self):
        self.trainer.create_graph()
