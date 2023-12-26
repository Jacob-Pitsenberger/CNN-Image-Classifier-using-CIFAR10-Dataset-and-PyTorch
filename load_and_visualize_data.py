"""
Author: Jacob Pitsenberger
Date: 12-26-23
Module: load_and_visualize_data.py

This module handles the loading and preprocessing of data for training a Convolutional Neural Network (CNN)
for image classification based on the CIFAR10 dataset. It includes functions for downloading, augmenting,
and visualizing the dataset.

Key functionalities:
- Data loading and processing, including image augmentation techniques.
- Visualization functions for batches and detailed images.

Functions:
- test_for_CUDA(): Test for the availability of CUDA for GPU acceleration.
- load_data(): Load and prepare training, validation, and test datasets.
- imshow(img): Helper function to unnormalize and display an image.
- get_batch(loader): Obtain one batch of training images from a data loader.
- visualize_batch(images, labels): Visualize a batch of images with corresponding labels.
- visualize_detailed_image(images): View an image in more detail by examining RGB color channels.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def test_for_CUDA() -> bool:
    """
    Test for the availability of CUDA for GPU acceleration.

    Returns:
    - train_on_gpu: Boolean whether CUDA is available
    """
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    return train_on_gpu

def load_data() -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load and prepare training, validation, and test datasets.

    Returns:
    - DataLoader: Training data loader.
    - DataLoader: Validation data loader.
    - DataLoader: Test data loader.
    """
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20
    # percentage of training set to use as validation
    valid_size = 0.2
    # Data transform to convert data to a tensor and apply normalization

    # Here we augment the train and validation dataset with RandomHorizontalFlip and RandomRotation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # randomly flip and rotate
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Data transform to convert test data to a tensor and apply normalization
    test_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image data to a PyTorch tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the tensor values to have a mean and standard deviation of 0.5
    ])

    # Choose the training and test datasets
    train_data = datasets.CIFAR10('data', train=True,  # Load the CIFAR-10 training dataset
                                  download=True,  # Download the dataset if not already present
                                  transform=train_transform)  # Apply the specified transformation to the training data

    test_data = datasets.CIFAR10('data', train=False,  # Load the CIFAR-10 test dataset
                                 download=True,  # Download the dataset if not already present
                                 transform=test_transform)  # Apply the specified transformation to the test data

    # Obtain training indices that will be used for validation
    num_train = len(train_data)  # Get the total number of samples in the training dataset
    indices = list(range(num_train))  # Create a list of indices corresponding to the training samples
    np.random.shuffle(indices)  # Shuffle the indices randomly
    split = int(np.floor(valid_size * num_train))  # Calculate the split index for validation set
    train_idx, valid_idx = indices[split:], indices[:split]  # Split the indices into training and validation sets

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              num_workers=num_workers)

    # Return the data loaders
    return train_loader, valid_loader, test_loader


def imshow(img: torch.Tensor) -> None:
    """
    Helper function to unnormalize and display an image.

    Args:
    - img: Image to be displayed.

    Returns:
    - None
    """
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


def get_batch(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Obtain one batch of training images from a data loader.

    Args:
    - loader: DataLoader from which to obtain a batch.

    Returns:
    - images: Batch of images.
    - labels: Labels corresponding to the images.
    """
    # Retrieve the next batch of images and labels and return them.
    images, labels = next(iter(loader))
    return images, labels


def visualize_batch(images: torch.Tensor, labels: torch.Tensor) -> None:
    """
    Visualize a subset of images from a batch with corresponding labels.

    Args:
    - images: Batch of images.
    - labels: Labels corresponding to the images.

    Returns:
    - None
    """
    images = images.numpy()  # convert images to numpy for display
    print(images.shape)  # (number of examples: 20, number of channels: 3, pixel sizes: 32x32)

    # Plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))  # Create a figure with a specified size for displaying images
    # Display 20 images
    for idx in np.arange(20):
        ax = fig.add_subplot(2, int(20 / 2), idx + 1, xticks=[], yticks=[])  # Add subplots for each image
        imshow(images[idx])  # Display the image using the imshow function
        ax.set_title(classes[labels[idx]])  # Set the title of the subplot to the corresponding class label
    plt.show()  # Show the entire figure with the batch of images


def visualize_detailed_image(images: torch.Tensor) -> None:
    """
    View an image in more detail by examining RGB color channels.

    Args:
    - images: Torch Tensor representing an image.

    Returns:
    - None
    """

    # Extract the normalized RGB image from the batch
    rgb_img = np.squeeze(images[3])

    # Define color channels for visualization
    channels = ['red channel', 'green channel', 'blue channel']

    # Create a figure for displaying detailed RGB channels
    fig = plt.figure(figsize=(36, 36))
    for idx in np.arange(rgb_img.shape[0]):
        ax = fig.add_subplot(1, 3, idx + 1)  # Add subplots for each color channel
        img = rgb_img[idx]
        ax.imshow(img, cmap='gray')  # Display the grayscale representation of the color channel
        ax.set_title(channels[idx])  # Set the title to the corresponding color channel
        width, height = img.shape
        thresh = img.max() / 2.5
        # Annotate pixel values on the image for visual inspection
        for x in range(width):
            for y in range(height):
                val = np.round(img[x][y], 2) if img[x][y] != 0 else 0
                ax.annotate(str(val), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center', size=8,
                            color='white' if img[x][y] < thresh else 'black')
    plt.show()  # Show the detailed RGB channel visualization

def main() -> None:
    """
    Main function to demonstrate data loading and visualization.
    """
    # Load the data
    train_loader, valid_loader, test_loader = load_data()

    # obtain one batch of training images
    images, labels = get_batch(train_loader)

    # visualize a batch of images
    visualize_batch(images, labels)

    # visualize an image in detail
    visualize_detailed_image(images)


if __name__ == "__main__":
    main()
