# DNN Project with OneDNN using the Boston Housing Dataset

## Introduction
This project is an example of a deep neural network (DNN) using OneDNN with the Boston Housing dataset. The goal of this project is to predict house prices based on 13 input features such as crime rate, number of rooms, and proximity to highways. OneDNN (formerly known as Intel MKL-DNN) is a library for deep learning that provides optimized performance on Intel CPUs and GPUs.

## Dataset
The Boston Housing dataset consists of 506 samples of housing prices in the Boston area. There are 13 input features such as crime rate, number of rooms, and proximity to highways, and the target variable is the median value of owner-occupied homes in thousands of dollars.

## OneAPI and OneDNN
OneAPI is a programming model for heterogeneous computing that allows developers to write code that runs on different types of hardware such as CPUs, GPUs, and FPGAs. OneDNN is a library for deep learning that provides optimized performance on Intel CPUs and GPUs using the OneAPI programming model.

## Methodology
The project consists of the following steps:

1) Load the Boston Housing dataset using scikit-learn.
2) Preprocess the data by standardizing the input features and splitting the data into training and test sets.
3) Build a DNN model using OneDNN with two hidden layers with 64 units each, and an output layer with a single unit. The relu activation function is used for the hidden  layers, and no activation function is used for the output layer.
4) Compile the model using the Adam optimizer and the mean squared error (MSE) loss function. The metric used to evaluate the model during training is the mean absolute error (MAE).
5) Train the model on the training data for 100 epochs with a batch size of 32, and use 20% of the training data for validation during training.
6) Evaluate the model on the test data and print the test loss (MSE) and MAE.
7) Visualize the training and validation loss and MAE over epochs using matplotlib.

## Model
The DNN model used in this project has two hidden layers with 64 units each, and an output layer with a single unit. The input layer has 13 units, corresponding to the 13 input features of the Boston Housing dataset. The relu activation function is used for the hidden layers, which has been shown to work well for many DNN applications. No activation function is used for the output layer since we want to predict a continuous value (the house price) rather than a binary classification label. The model is compiled using the Adam optimizer and the MSE loss function, which is a common choice for regression problems. The metric used to evaluate the model during training is the MAE, which is the average absolute difference between the predicted house prices and the true house prices in the validation set.

## Results
The model achieves a test MAE of approximately 2.5. This means that the model's predictions are, on average, off by $2,500 in terms of house prices. However, this value can be improved with further tuning of the model's hyperparameters or by using a more complex architecture.Since the Boston Housing dataset is a regression problem (predicting house prices), we cannot use accuracy as a metric to evaluate the model. Instead, we can use the mean absolute error (MAE)

![download (5)](https://user-images.githubusercontent.com/111365771/224477784-58873edc-6f70-40f8-9350-fe36ee113fe7.png)

![download (6)](https://user-images.githubusercontent.com/111365771/224477785-6b40cf03-559d-4762-ab04-0421f3c61355.png)

## Acknowledgements
This project was inspired by the Intel OneDNN samples repository and the scikit-learn documentation. The Boston Housing dataset is available from the UCI Machine Learning Repository.
