# Neural-Net-Project

## Convolutional Neural Network (CNN) for CIFAR-10 Classification

This repository contains code for training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using PyTorch. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Prerequisites

Make sure you have the following dependencies installed:

- Python 3.x
- PyTorch
- torchvision
- numpy

## Dataset

The CIFAR-10 dataset will be automatically downloaded and preprocessed when running the code. It will be split into a training set (80% of the data) and a validation set (20% of the data).

## Model Architecture

The CNN model used for this classification task consists of several convolutional layers followed by fully connected layers. The architecture is as follows:

1. Convolutional layer (input: 3 channels, output: 64 channels, kernel size: 3x3, stride: 1, padding: 1)
2. ReLU activation
3. Convolutional layer (input: 64 channels, output: 128 channels, kernel size: 3x3, stride: 1, padding: 1)
4. ReLU activation
5. Max pooling layer (kernel size: 2x2, stride: 2)
6. Convolutional layer (input: 128 channels, output: 256 channels, kernel size: 3x3, stride: 1, padding: 1)
7. Batch normalization
8. ReLU activation
9. Convolutional layer (input: 256 channels, output: 256 channels, kernel size: 3x3, stride: 1, padding: 1)
10. ReLU activation
11. Max pooling layer (kernel size: 2x2, stride: 2)
12. Convolutional layer (input: 256 channels, output: 512 channels, kernel size: 3x3, stride: 1, padding: 1)
13. ReLU activation
14. Convolutional layer (input: 512 channels, output: 512 channels, kernel size: 3x3, stride: 1, padding: 1)
15. Batch normalization
16. ReLU activation
17. Max pooling layer (kernel size: 2x2, stride: 2)
18. Fully connected layer (input: 512*4*4, output: 1024)
19. ReLU activation
20. Dropout (p=0.5)
21. Fully connected layer (input: 1024, output: 512)
22. ReLU activation
23. Dropout (p=0.5)
24. Fully connected layer (input: 512, output: 10)

## Training

The model is trained using stochastic gradient descent (SGD) optimizer with a learning rate of 0.5. The learning rate is adjusted using an exponential decay scheduler with a decay factor of 0.9.

The training is performed for a total of 15 epochs. During each epoch, the model is trained on the training set in batches of size 128. The loss function used is cross-entropy loss.

After each epoch, the accuracy and loss on the training set are recorded. Additionally, the accuracy on the validation set is computed.

## Results

The final validation accuracy achieved by the model is printed after training. The model is evaluated on the validation set to measure its performance.

## Usage

To run the code, execute the following jupyter notebook: `CIFAR_10.ipynb`

You can modify the hyperparameters such as batch size, learning rate, and number of epochs in the code. Additionally, you can experiment with different model architectures by uncommenting the alternative CNN architecture provided in the code.
Remember to have the CIFAR-10 dataset downloaded or set the `download` parameter to `True` in the `torchvision.datasets.CIFAR10` function to download the dataset automatically.

## Conclusion

This code provides a basic implementation of a CNN for CIFAR-10 classification using PyTorch. You can use it as a starting point to explore and experiment with different CNN architectures, hyperparameters, and training techniques to improve the model's performance.
