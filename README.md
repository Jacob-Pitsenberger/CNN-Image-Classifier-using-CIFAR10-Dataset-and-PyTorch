# Project Overview

This project encompasses a series of modules designed to facilitate the creation, training, and prediction using a PyTorch CNN Neural Network for Image classification based on the CIFAR10 dataset. The key modules include:

- **load_and_visualize_data.py**: Downloads the CIFAR10 dataset, applies data augmentation techniques, loads data, and provides visualization functions.
- **net.py**: Defines the architecture for a PyTorch CNN Neural Network.
- **train.py**: Offers functions for training the neural network.
- **predict.py**: Provides functions for making predictions using trained models and visualizing the results.

Under the root directory "CNN Image Classifier using CIFAR10 Dataset and Pytorch," you'll find these modules, the 'data' directory (storing the CIFAR10 dataset), and a .pt file titled 'model_cifar.pt' which contains the trained model.

# Purpose of the Project

This project serves as a practical application and demonstration of the knowledge gained during the Udacity course "Intro to Deep Learning with PyTorch," specifically focusing on Convolutional Neural Networks (CNNs). The primary objectives were:

## 1. Data Loading and Preprocessing (`load_and_visualize_data.py`)

Understanding and handling real-world datasets is a crucial skill in machine learning. In this module, I applied the concepts of data loading and preprocessing learned in the course. Techniques such as data augmentation were employed to enhance the diversity of the CIFAR10 dataset, preparing it for training.

## 2. CNN Architecture Definition (`net.py`)

The `net.py` module reflects the key topics related to Convolutional Neural Networks (CNNs) covered in the course:
- Convolutional Layers: The architecture includes nn.Conv2d layers to capture hierarchical features in input images.
- Fully Connected Layers: nn.Linear layers follow convolutional layers to learn global patterns in feature maps.
- Activation Functions and Dropout: ReLU activation (F.relu) is applied, and Dropout is used to prevent overfitting.
- Forward Pass: The forward method defines the sequence of operations during the forward pass.
- Model Initialization: The model is initialized by creating an instance of the Net class.

## 3. Model Training (`train.py`)

The `train.py` module incorporates several key topics learned during the course section:
- GPU Acceleration: The module checks for GPU availability to leverage GPU acceleration.
- Data Loading and Preprocessing: It covers loading and preprocessing data using PyTorch data loaders.
- Loss Functions and Optimization: The module specifies the loss function (CrossEntropyLoss) and optimizer (Adam).
- Model Training: The training process for a CNN is demonstrated, including forward and backward passes.
- Model Evaluation: The best-performing model is saved for later use.

## 4. Model Prediction and Visualization (`predict.py`)

The `predict.py` module focuses on testing and visualizing predictions made by a trained CNN on the CIFAR-10 dataset:
- Model Evaluation: It checks for GPU availability, loads the trained model, and specifies the loss function for model evaluation.
- Visualization of Predictions: The module visualizes predictions, showing the model's performance on a sample batch.

Feel free to explore and adapt the project to your needs, and I hope you find it informative and enjoyable!

# Usage

To utilize this project, you can either train your own models from scratch or use the pre-trained model ('model_cifar.pt').

## Starting From Scratch

### Visualize CIFAR10 Data

Run the module as the main loop to execute the main function, creating data loaders, fetching a batch, and visualizing the data using `visualize_batch(images, labels)` and `view_detailed_image(images)`.

### Train Model(s)

With downloaded data and proper directories, run the `train(net_info, train_loader, valid_loader, n_epochs)` functions in `train.py` to train the model (defined in `net.py`). The trained model's weights and information will be saved under the root directory.

### Make Predictions

After training, run `predict.py`, ensuring the model dict specified in the main function is loaded from your trained model (.pt) file path.

## Using Pre-trained Models

If using the provided pre-trained models in the .pt file titled 'model_cifar.pt', run `predict.py` as the main loop to make predictions. 

You can also visualize CIFAR10 data using `load_and_visualize_data.py` as described in the 'Starting From Scratch' section.

Feel free to explore and adapt the project to your needs, and I hope you find it informative and enjoyable!

# Author
Jacob Pitsenberger - 2023

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.
