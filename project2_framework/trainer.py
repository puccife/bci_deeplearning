
class Trainer:

    def __init__(self, model, optimizer, loss, epochs, train_loader, test_loader):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self):

        # writer = SummaryWriter()
        for epoch in range(self.epochs):

            # computing training loss, accuracy and the backward pass
            t_loss, t_accuracy = self.compute_loss_and_accuracy()

            # computing validation loss, accuracy and skipping backward pass
            v_loss, v_accuracy = self.compute_loss_and_accuracy(is_training=False)

            print('Epoch [ %d / %d ], train loss: %.6f, val loss: %.6f, train acc: %.6f, val acc: %.6f'% (epoch + 1,
                                                                                                          self.epochs,
                                                                                                          t_loss,
                                                                                                          v_loss,
                                                                                                          t_accuracy,
                                                                                                          v_accuracy))

    def compute_loss_and_accuracy(self, is_training=True):

        epoch_val_losses = []
        total = 0
        correct = 0

        loader = self.train_loader if is_training else self.test_loader

        for j, (x_batch, y_batch) in enumerate(loader.get_loader()):
            batch_loss = 0
            for x, y in zip(x_batch, y_batch):
                predicted = self.model.forward(x)
                total += 1
                if predicted.max(0)[1][0] == y.max(0)[1][0]:
                    correct += 1
                loss_value = self.loss.compute(predicted, y)
                batch_loss += loss_value
                epoch_val_losses.append(loss_value)

                # computing the backward pass for the training part
                if is_training:
                    derivative_loss = self.loss.derivative(predicted, y)
                    self.model.backward(derivative_loss)

            # updating the model weights during training
            if is_training:
                self.optimizer.step()

        epoch_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        return epoch_val_loss, correct/total
