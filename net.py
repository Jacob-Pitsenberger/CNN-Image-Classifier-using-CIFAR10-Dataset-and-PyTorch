"""
Author: Jacob Pitsenberger
Date: 12-26-23
Module: net.py

This module defines the architecture of a Convolutional Neural Network (CNN) for image classification using PyTorch.
The CNN includes convolutional layers, max pooling, fully connected layers, and dropout to enhance learning and prevent overfitting.

Key functionalities:
- CNN architecture definition for image classification.
- Utilizes convolutional and fully connected layers with ReLU activation and dropout.

Classes:
- Net: CNN model class.

Attributes:
- Conv1 (nn.Conv2d): First convolutional layer.
- Conv2 (nn.Conv2d): Second convolutional layer.
- Conv3 (nn.Conv2d): Third convolutional layer.
- Pool (nn.MaxPool2d): Max pooling layer.
- fc1 (nn.Linear): First fully connected layer.
- fc2 (nn.Linear): Second fully connected layer.
- fc3 (nn.Linear): Third fully connected layer.
- dropout (nn.Dropout): Dropout layer to prevent overfitting.

Methods:
- __init__(): Initializes the CNN architecture.
- forward(x: Tensor) -> Tensor: Defines the forward pass of the CNN.
"""

# Import necessary libraries
import torch.nn as nn
import torch.nn.functional as F


# Define the CNN architecture in a class called Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolutional layers with specified input channels, output channels, kernel size, and padding
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # convolutional layer 1 (input channels: 3, output channels: 16, kernel size: 3x3, padding: 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # convolutional layer 2 (input channels: 16, output channels: 32, kernel size: 3x3, padding: 1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # convolutional layer 3 (input channels: 32, output channels: 64, kernel size: 3x3, padding: 1)

        # Max pooling layer with kernel size 2x2 and stride 2
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers with specified input and output dimensions
        self.fc1 = nn.Linear(64 * 4 * 4, 500)  # Fully connected layer 1 (input features: 64 * 4 * 4, output features: 500)
        self.fc2 = nn.Linear(500, 10)  # Fully connected layer 2 (input features: 500, output features: 10)

        # Dropout layer with a specified dropout rate of 0.25
        self.dropout = nn.Dropout(0.25)

    # Define the forward pass of the network
    def forward(self, x):
        # Apply convolutional and max pooling layers with ReLU activation
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the image input by reshaping it to be compatible with the fully connected layers
        x = x.view(-1, 64 * 4 * 4)

        # Apply dropout to the tensor to prevent overfitting
        x = self.dropout(x)

        # Apply the first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Apply dropout to the tensor to prevent overfitting
        x = self.dropout(x)

        # Apply the second fully connected layer
        x = self.fc2(x)

        # Return the final output tensor
        return x


# Run the model creation if this script is executed
if __name__ == "__main__":
    # Create a complete CNN and print its architecture
    model = Net()
    print(model)
