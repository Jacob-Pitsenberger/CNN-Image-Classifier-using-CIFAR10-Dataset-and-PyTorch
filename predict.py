"""
Author: Jacob Pitsenberger
Date: 12-26-23
Module: predict.py

This module implements the prediction and visualization process for a Convolutional Neural Network (CNN) using PyTorch.
It checks for GPU availability, loads a pre-trained model, specifies the loss function, and evaluates and visualizes predictions.

Key functionalities:
- Prediction and evaluation of a pre-trained CNN on the CIFAR-10 dataset.
- Visualization of predicted and true labels for a sample batch.
"""

from net import Net
from load_and_visualize_data import load_data, get_batch, test_for_CUDA, imshow
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Check for CUDA
train_on_gpu = test_for_CUDA()


def print_predictions(model: torch.nn.Module,
                      criterion: torch.nn.Module,
                      test_loader: torch.utils.data.DataLoader,
                      classes: list[str]) -> None:
    """
    Print test loss and accuracy.

    Args:
        - model (torch.nn.Module): The CNN model.
        - criterion (torch.nn.Module): The loss function.
        - test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        - classes (list): List of image classes.

    Returns:
        - None
    """
    # how many samples per batch to load
    batch_size = 20

    # track test loss
    test_loss = 0.0
    # List to track correct predictions for each class
    class_correct = list(0. for _ in range(10))
    # List to track the total number of samples for each class
    class_total = list(0. for _ in range(10))

    # Set the model to evaluation mode
    model.eval()
    # iterate over test data
    for data, target in test_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to the true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        # convert the correct_tensor to a numpy array and squeeze it
        correct = np.squeeze(correct_tensor.cpu().numpy()) if train_on_gpu else np.squeeze(correct_tensor.numpy())
        # calculate test accuracy for each object class
        for i in range(batch_size):
            # get the true label for the i-th sample in the batch
            label = target.data[i]
            # update class_correct and class_total based on the correctness of the prediction
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    # Calculate the average test loss by dividing the total test loss by the number of test samples
    test_loss = test_loss / len(test_loader.dataset)
    # Print the average test loss
    print('Test Loss: {:.6f}\n'.format(test_loss))

    # Loop through each class
    for i in range(10):
        # Check if there are any samples for the current class
        if class_total[i] > 0:
            # Print the test accuracy for the current class
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                int(np.sum(class_correct[i])), int(np.sum(class_total[i]))))
        else:
            # Print a message indicating that there are no training examples for the current class
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    # Print the overall test accuracy
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        int(np.sum(class_correct)), int(np.sum(class_total))))


def visualize_predictions(model: torch.nn.Module,
                          test_loader: torch.utils.data.DataLoader,
                          classes: list[str]) -> None:
    """
    Visualize predictions for a sample batch.

    Args:
        - model (torch.nn.Module): The CNN model.
        - test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        - classes (list): List of image classes.

    Returns:
        - None
    """
    # obtain one batch of test images
    images, labels = get_batch(test_loader)
    images.numpy()

    # move model inputs to cuda, if GPU available
    if train_on_gpu:
        images = images.cuda()

    # get sample outputs
    output = model(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy()) if train_on_gpu else np.squeeze(preds_tensor.numpy())

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, int(20 / 2), idx + 1, xticks=[], yticks=[])
        imshow(images[idx].cpu().numpy() if train_on_gpu else images[idx].numpy())
        ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                     color=("green" if preds[idx] == labels[idx].item() else "red"), fontsize=8)
    # Adjust width between subplots
    fig.subplots_adjust(wspace=0.8)

    plt.show()


def main() -> None:
    """
    Main function to execute the prediction and visualization process.

    Returns:
        - None
    """
    # Load the data
    train_loader, valid_loader, test_loader = load_data()

    # create a complete CNN
    model = Net()

    model.load_state_dict(torch.load('model_cifar.pt'))

    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # specify the image classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print_predictions(model, criterion, test_loader, classes)

    visualize_predictions(model, test_loader, classes)


if __name__ == "__main__":
    main()
