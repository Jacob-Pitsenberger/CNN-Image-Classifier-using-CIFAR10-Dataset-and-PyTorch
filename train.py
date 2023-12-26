"""
Author: Jacob Pitsenberger
Date: 12-26-23
Module: train.py

This module implements the training process for a Convolutional Neural Network (CNN) using PyTorch.
It covers GPU availability check, data loading, model creation, loss function specification, optimization, training, and model saving.

Key functionalities:
- CNN training process with specified loss function and optimizer.
- Model evaluation and saving the best-performing model.

Functions:
- train(net_info, train_loader, valid_loader, n_epochs): Function for training the CNN model.
- main(): Main function to execute the training process.

Run this script to train the CNN model on the CIFAR-10 dataset.
"""

from net import Net
from load_and_visualize_data import load_data, test_for_CUDA
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch

# Check for CUDA
train_on_gpu = test_for_CUDA()


def train(net_info: list[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer],
          train_loader: torch.utils.data.DataLoader,
          valid_loader: torch.utils.data.DataLoader,
          n_epochs: int) -> None:
    """
    Train the CNN model.

    Args:
    - net_info (list): A list containing the CNN model, loss function, and optimizer.
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - valid_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    - n_epochs (int): Number of training epochs.

    Returns:
    - None
    """
    # Unpack the net_info to get the model, criterion, and optimizer
    model, criterion, optimizer = net_info

    valid_loss_min = np.Inf  # track change in validation loss

    # Loop through each training epoch
    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # Train the model #
        ###################

        # Set the model to training mode, enabling gradients and dropout
        model.train()

        # Iterate over each batch in the training DataLoader
        for data, target in train_loader:
            # Move input data and target labels to GPU if available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Zero the gradients to clear any accumulated values from previous iterations
            optimizer.zero_grad()

            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)

            # Calculate the batch loss using the specified criterion (loss function)
            loss = criterion(output, target)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Perform a single optimization step (parameter update)
            optimizer.step()

            # Update the training loss by adding the product of batch loss and batch size
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################

        # Set the model to evaluation mode, disabling gradients and dropout
        model.eval()

        # Iterate over each batch in the validation DataLoader
        for data, target in valid_loader:
            # Move input data and target labels to GPU if available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)

            # Calculate the batch loss using the specified criterion (loss function)
            loss = criterion(output, target)

            # Update the validation loss by adding the product of batch loss and batch size
            valid_loss += loss.item() * data.size(0)

        # Calculate average losses by dividing the accumulated losses by the total number of samples
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)

        # Print training/validation statistics for the current epoch
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # Save the model if the validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model_cifar.pt')
            valid_loss_min = valid_loss


def main() -> None:
    """
    Main function to execute the training process.

    Returns:
    - None
    """

    # Load the training, validation, and test data using DataLoader
    train_loader, valid_loader, test_loader = load_data()

    # Create a complete Convolutional Neural Network (CNN) model
    model = Net()

    # Move the model to the GPU if CUDA is available
    if train_on_gpu:
        model.cuda()

    # Specify the loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # Specify the optimizer (Stochastic Gradient Descent)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Bundle the model, loss function, and optimizer into a list
    net_info = [model, criterion, optimizer]

    # Number of training epochs
    n_epochs = 30

    # Call the train function to train the model
    train(net_info, train_loader, valid_loader, n_epochs)


if __name__ == "__main__":
    main()
