from nn.optimizers.SGD import sgd

# from tensorboardX import SummaryWriter

def train(model, epochs, train_loader, test_loader, loss, lr=0.01):

    # writer = SummaryWriter()
    for epoch in range(epochs):
        for k, (x_batch, y_batch) in enumerate(train_loader.get_loader()):
            model, batch_loss = sgd(model=model, x_batch=x_batch, y_batch=y_batch, loss=loss, lr=lr)

        t_loss, t_accuracy = validate('train', train_loader, model, epoch, loss)#, writer)
        v_loss, v_accuracy = validate('validation', test_loader, model, epoch, loss)#, writer)

        print("Epoch [",epoch,"/",epochs,"], train acc: ", t_accuracy, ", val acc: ", v_accuracy)
        print('Epoch [ %d / %d ], train loss: %.6f, val loss: %.6f'% (epoch + 1, epochs,
                 t_loss, v_loss))


def validate(label, loader, model, epoch, loss):#, writer):
    epoch_val_losses = []
    total = 0
    correct = 0
    for j, (x_batch, y_batch) in enumerate(loader.get_loader()):
        batch_loss = 0
        for x, y in zip(x_batch, y_batch):
            predicted = model.forward(x)
            total += 1
            if predicted.max(0)[1][0] == y.max(0)[1][0]:
                correct += 1
            loss_value = loss.compute(predicted, y)
            batch_loss += loss_value
            epoch_val_losses.append(loss_value)
        # writer.add_scalars('loss', {label: batch_loss}, epoch * 1000 + 10 * j)
        # writer.add_scalars('accuracy', {label: correct/total}, epoch * 1000 + 10 * j)
    epoch_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
    return epoch_val_loss, correct/total
