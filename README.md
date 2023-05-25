# MultiVariable-Logistic-Regression-Pytorch
A simple PyTorch implementation of multivariable logistic regression on the health index dataset from kaggle

There are 6 input variables and 1 output variable, the model is trained for 20 epochs with a learning rate of 0.01.
The training is performed on the 2018 dataset while the 2019 dataset is used for testing. The max MSE I got was between 2.0 and 3.0.
The code for train-test split is commented out for the 2018 dataset and can be used to make a validation dataset during training for tuning hyper-parameters
