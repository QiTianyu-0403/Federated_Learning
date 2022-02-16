# Federated_Learning 💻
This projest include common test models for federated learning, like CNN, Resnet18 and lstm, controlled by different parser. It can also handle common noniid data. We can change the parameters to change the model and dataset. Here is the related introduction.

---


## Network Models

This project contains network models commonly used in FL：Resnet18, CNN and LSTM.
- **Resnet18**

A ***Residual Neural Network*** (ResNet) is an artificial neural network (ANN). Typical ResNet models are implemented with double- or triple- layer skips that contain nonlinearities (ReLU) and batch normalization in between. An additional weight matrix may be used to learn the skip weights. These models are known as HighwayNets.

> Working with datasets: Cifar10, MNIST, FMNIST.

- **CNN**

In deep learning, the **Convolutional Neural Network** (CNN) is a class of artificial neural network, most commonly applied to analyze visual imagery. CNNs are a specialized type of neural networks that use convolution in place of general matrix multiplication in at least one of their layers.

> Working with datasets: Cifar10, MNIST, FMNIST.

- **LSTM**

**Long short-term memory** (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. LSTM networks are well-suited to classifying, processing and making predictions based on time series data.

> Working with datasets: Shakespeare.

## Environment

- Python >= 3.6.0
- Pytorch >= 1.7.0
- Torchvision >= 0.8.0

## Datasets

- Cifar10: Consist of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
- MNIST: Consist of 70000 28x28 gray images in 10 classes. There are 60000 training images and 10000 test images.
- FasionMNIST: Consist of 70000 28x28 gray images in 10 classes, with 7000 images per class. There are 60000 training images and 10000 test images.
- Shakespeare: Consist of 1146 local devices, a txt file.
