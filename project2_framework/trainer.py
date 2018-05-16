from nn.optimizers.SGD import sgd

def train(model, epochs, train_loader, test_loader, loss, lr=0.01):

    for epoch in range(epochs):
        epoch_train_losses = []
        for x_batch, y_batch in train_loader.get_loader():

            model, batch_loss = sgd(model=model, x_batch=x_batch, y_batch=y_batch, loss=loss, lr=lr)
            epoch_train_losses.append(batch_loss)
            # print('Epoch [%d/%d], batch Loss: %.9f' % (epoch + 1, epochs, batch_loss))

        epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)

        # validation
        epoch_val_losses = []
        total = 0
        correct = 0
        for x_batch, y_batch in test_loader.get_loader():
            for x, y in zip(x_batch, y_batch):
                predicted = model.forward(x)
                total+=1
                if predicted.max(0)[1][0] == y.max(0)[1][0]:
                    correct+=1
                loss_value = loss.compute(predicted, y)
                epoch_val_losses.append(loss_value)

        epoch_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)

        print("Accuracy: ", correct/total)
        print('Epoch [%d/%d], train Loss: %.6f, val loss: %.6f'% (epoch + 1, epochs,
                 epoch_train_loss, epoch_val_loss))







