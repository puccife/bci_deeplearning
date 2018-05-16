import argparse
import torch
import torch.utils.data as dt
import utils.dlc_bci as bci
import utils.hyperparams as hyperparams
from preprocessing.preprocessing import Preprocess
from model.trainer import NetTrainer


# asserting pytorch version == 0.4.0
try:
    assert torch.__version__ == '0.4.0'
except AssertionError as e:
    e.args += ('Incompatible version of torch package','please use VERSION 0.4.0')
    raise

# data path of the dataset
DATA_PATH = '../../data_bci'

def main(epochs, batch_size, weight_decay):
    """
    Main method to call
    :param epochs: epochs of training
    :param batch_size: batch size dimension
    :param weight_decay: weight decay
    :return: 0
    """

    # prettify output
    print("=================================================================")
    print("MODEL\ttrain_accuracy\tval_accuracy\ttrain_loss\tval_loss")
    print("-----------------------------------------------------------------")

    # models to be trained
    models = ['TS','LOG', 'CNN', 'LSTM']
    # loop over all the models
    for model in models:

        # loading inputs - targets and creating dataset loaders
        train_inputs, train_targets = bci.load(root=DATA_PATH, one_khz=False)
        test_inputs, test_targets = bci.load(root=DATA_PATH, train=False, one_khz=False)
        train_inputs, test_inputs = Preprocess().transform(train_inputs, test_inputs, train_targets, test_targets, to_filter=model!='TS')
        train_dataset = dt.TensorDataset(train_inputs, train_targets)
        test_dataset = dt.TensorDataset(test_inputs, test_targets)
        train_loader = dt.DataLoader(dataset=train_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)
        test_loader = dt.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=True)

        # creating the net trainer given the model
        t = NetTrainer(epochs, batch_size, weight_decay, model)#, pretrained=model=='TS')

        # training model
        best_accuracy, v_loss, best_t_accuracy, t_loss = t.train(train_loader, test_loader)

        # printing output
        print(model + '\t\t' + str(round(best_t_accuracy, 3)) + '\t\t' + str(best_accuracy) + '\t     ' + str(round(t_loss.item(), 3)) + '\t   ' + str(round(v_loss.item(), 3)))
    print("=================================================================")
    return 0

if __name__ == '__main__':
    epochs, \
    batch_size, \
    weight_decay = hyperparams.load_config()

    parser = argparse.ArgumentParser(description='Run deep learning model for BCI data classification.')
    parser.add_argument("-e", "--epochs", help="Number of training epochs", type=int, default=epochs)
    parser.add_argument("-b", "--batch_size", help="Batch size for training", type=int, default=batch_size)
    args = parser.parse_args()
    main(args.epochs, args.batch_size, weight_decay)