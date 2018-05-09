import argparse

import torch.utils.data as dt
import utils.dlc_bci as bci
import utils.hyperparams as hyperparams
from preprocessing.preprocessing import Preprocess
from model.trainer import NetTrainer

DATA_PATH = '../../data_bci'

def main(model, epochs, batch_size, weight_decay):


    train_inputs, train_targets = bci.load(root=DATA_PATH, one_khz=False)
    test_inputs, test_targets = bci.load(root=DATA_PATH, train=False, one_khz=False)

    train_inputs, test_inputs = Preprocess().transform(train_inputs, test_inputs, train_targets, test_targets)

    # Datasets
    train_dataset = dt.TensorDataset(train_inputs, train_targets)
    test_dataset = dt.TensorDataset(test_inputs, test_targets)
    train_loader = dt.DataLoader(dataset=train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
    test_loader = dt.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=True)

    t = NetTrainer(epochs, batch_size, weight_decay, model)
    t.train(train_loader, test_loader)
    t.create_graph()
    return 0

if __name__ == '__main__':
    model, \
    epochs, \
    batch_size, \
    weight_decay = hyperparams.load_config()

    parser = argparse.ArgumentParser(description='Run deep learning model for BCI data classification.')
    parser.add_argument("-m", "--model", help="Model for training", type=str, default=model)
    parser.add_argument("-e", "--epochs", help="Number of training epochs", type=int, default=epochs)
    parser.add_argument("-b", "--batch_size", help="Batch size for training", type=int, default=batch_size)
    args = parser.parse_args()
    main(args.model, args.epochs, args.batch_size, weight_decay)