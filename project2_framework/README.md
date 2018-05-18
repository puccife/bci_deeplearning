# EPFL Deep Learning Course CS-433, Spring 2018, second project

## Description

Implementation of deep learning framework using only the torch Tensors and pytorch tensor operations. Implementation of Linear layer, Tanh, Relu and Softmax activation functions, MSE and Cross Entropy losses

## Getting Started

First of all, make sure you have installed:

1. python 3.5 or higher

2. pytorch 0.4.0

### Description of directories
1. In the "nn" directory you can find subdirectories containing the activations funtions implementations, the layers implementations, the Sequential class, the losses and the optimizer.

2. In the "utils" directory there are the data generator, the Loader class and the trainer class
 

## description of the script
The "test.py" script contains a training session on the model described on the assignment. The training is done on 500 epochs with a learning rate of 0.01 and batch size equal to 10. In order to run the script just use "python test.py" on command line.


This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


