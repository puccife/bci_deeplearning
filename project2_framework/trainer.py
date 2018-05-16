from nn.optimizers.SGD import SGD


def batch_step(model, x_batch, y_batch, loss):
    batch_losses = []
    for x, y in zip(x_batch, y_batch):
        predicted = model.forward(x)
        # print(predicted)
        loss_value = loss.compute(predicted, y)
        batch_losses.append(loss_value)

        derivative_loss = loss.derivative(predicted, y)
        model.backward(derivative_loss)

    # return the updated model and the average loss of the batch
    return sum(batch_losses) / len(batch_losses)


def train(model, epochs, train_loader, test_loader, loss, optimizer):

    for epoch in range(epochs):
        epoch_train_losses = []
        for x_batch, y_batch in train_loader.get_loader():
            batch_loss = batch_step(model=model, x_batch=x_batch, y_batch=y_batch, loss=loss)
            epoch_train_losses.append(batch_loss)

            # updating the model parameters
            optimizer.step()

        # compute the average epoch train loss
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









