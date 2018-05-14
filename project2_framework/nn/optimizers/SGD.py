def sgd(model, x_batch, y_batch, loss, lr):
    batch_losses = []
    for x, y in zip(x_batch, y_batch):
        predicted = model.forward(x)
        # print(predicted)
        loss_value = loss.compute(predicted, y)
        batch_losses.append(loss_value)

        derivative_loss = loss.derivative(predicted, y)
        model.backward(derivative_loss)

    # TODO: temporaneo sgd update fatto cosi
    model.update_weights(lr)

    # return the updated model and the average loss of the batch
    return model, sum(batch_losses) / len(batch_losses)