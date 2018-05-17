### README ###

The purpose of this readme is to better understand the structure of the project.
We briefly describe the content of each folder to navigate it easily.

## CONFIG:
	JSON file containing the running parameters of the network.

## MODEL:
	Folder containing the models of the network.
	*baselines* folder contain the logistic regression model
	*nets* folder contains cnn, lstm and cnn(no preproc. called tsnet to distinguish)
	*trainer* class to train every model

## PREPROCESSING:
	Folder containing the methods used to preprocess the data

## UTILS:
	Contains methods to load data and hyperparameters from the jsonfile

## VISUALIZATION:
	Folder used mainly to debug and visualise the structure of the network

## test.py:
	Main script to run without parameters