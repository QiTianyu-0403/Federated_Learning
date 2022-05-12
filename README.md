# Federated_Learning 💻
This projest include common test models for federated learning, like CNN, Resnet18 and lstm, controlled by different parser. It can also handle common noniid data. We can change the parameters to change the model and dataset. Here is the related introduction.

---


## Network Models 🔑

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

## Datasets 📝

- Cifar10: Consist of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
- MNIST: Consist of 70000 28x28 gray images in 10 classes. There are 60000 training images and 10000 test images.
- FasionMNIST: Consist of 70000 28x28 gray images in 10 classes, with 7000 images per class. There are 60000 training images and 10000 test images.
- Shakespeare: Consist of 1146 local devices, a txt file.

All datasets will be saved at ``/data/``.

***The models and the datasets must match!!! Otherwise an error will occur.*** ❗️

## Environment 🐍

- Python >= 3.6.0
- Pytorch >= 1.7.0
- Torchvision >= 0.8.0

## Train ⏳

### Parameter Introduction

#### 1. setting.py ⚙️

This file is used to assign the datasets. The address is ``/noniid/setting.py``. You should ``python setting.py`` before formal training. And the assignment result will be saved at ``/noniid/temp/``. Here is the parsers.

- ``--data``: Select the datasets which will be assigned, including ``Cifar``, ``MNIST``, ``FMNIST``.
- ``--noniid_model``: There are 3 models of noniid setting, including ``iid``, ``label_noniid`` and ``quantity_noniid``. ``iid`` means each clients has the same distribution of data. `label_noniid` means each client has data with different labels. ``quantity_noniid`` means each client has a different amount of data.
- ``--num_users``: Select the number of clients.
- ``--total_samples``: In ``iid`` and ``label_noniid`` models, this means the number of data of each client. In ``quantity_noniid`` model, this means the mean number of data of all clients.
- ``--rate_unbalance``: In ``label_noniid``, this means the offset of the data label. It is a number less than 1, which represents the proportion of noniid.
- ``--num_class``: In ``label_noniid``, this means the number of labels of each client. 

If you don't have dataset, you should change the ``download=True`` in ``/noniid/data_noniid/Cifar_noniid.py/get_dataset_cifar10_noniid``, ``/noniid/data_noniid/MNIST_noniid.py/get_dataset_mnist_noniid`` and ``/noniid/data_noniid/FMNIST_noniid.py/get_dataset_fmnist_noniid``.

#### 2. main.py

This file is used to train the datasets. The information from training will be saved at ``/log/`` and ``/acc/``. Here is the parsers.

- ``--model``: Select the network models, including ``resnet18``, ``cnn`` and ``lstm``.
- ``--data``: Select the datasets which will be trained, including ``Cifar``, ``MNIST``, ``FMNIST``.
- ``--batchsize``: The batch size of each epoch, better multiples of 128.
- ``--EPOCH``: The number of epochs.
- ``--noniid_model``: There are 3 models of noniid setting, including ``iid``, ``label_noniid`` and ``quantity_noniid``.
- ``--num_users``: Select the number of clients.
- ``--total_samples``: In ``iid`` and ``label_noniid`` models, this means the number of data of each client. In ``quantity_noniid`` model, this means the mean number of data of all clients.
- ``--rate_unbalance``: In ``label_noniid``, this means the offset of the data label. It is a number less than 1, which represents the proportion of noniid.
- ``--num_class``: In ``label_noniid``, this means the number of labels of each client. 
- ``--idx_user``: The serial number of client we want to train.

Some parameters here are the same as in ``setting.py``, so that it can match the data just processed.

---

### Exemples 🙋

🔸 **(a)** Slice dataset ``Cifar`` to make the ``iid`` data with ``3`` clients, ``1000`` data per client.

```python /noniid/setting.py -d Cifar -nm iid -nu 3 -ts 1000```

🔸 **(b)** Slice dataset ``MNIST`` to make the ``label_iid`` data with ``4`` clients, ``1000`` data per client, ``0.6`` related unbalanced and ``2`` related labels.

```python /noniid/setting.py -d MNIST -nm label_noniid -nu 4 -ts 1000 -ru 0.6 -nc 2```

In this way we can get the number of data from 10 classes like ``[50 50 300 50 300 50 50 50 50 50]``.

🔸 **(c)** Slice dataset ``FMNIST`` to make the ``quantity_noniid`` data with ``3`` clients, average ``500`` data per client.

```python /noniid/setting.py -d FMNIST -nm quantity_noniid -nu 3 -ts 500```

In this way we can get the number of data from 10 classes as ``0:[33 33 33 33 33 33 33 33 33 33]`` ``1:[50 50 50 50 50 50 50 50 50 50]`` ``2:[67 67 67 67 67 67 67 67 67 67]``.

🔸 **(d)** Train the ``1th`` client in (c) with ``cnn`` model, ``128`` batchsize, ``1000`` epochs.

```python main.py -m cnn -d FMNIST -bs 128 -e 1000 -nm quantity_noniid -nu 3 -ts 500 -iu 1```

---

## Federated Learning

### FL

If you want to run FL, you should set the file in ``main.py``. You can run this code in different equipments.
