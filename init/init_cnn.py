import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from model.resnet import *
from model.cnn import *
from torchsummary import summary
from noniid.file_flow import select_trainset


def tmp_func(x):
    return x.repeat(3, 1, 1)


def normalize_data_cifar():
    """
    Get the normalize picture (cifar)
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        # The mean and variance of R,G, and B for each level of normalization
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_train, transform_test


def normalize_data_mnist():
    """
    Get the normalize picture (MNIST and FMNIST)
    make the gray_picture * 3 layers
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(tmp_func),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    return transform


def load_data(args):
    """
    Get the train and test dataloader
    """
    transform_train_cifar, transform_test_cifar = normalize_data_cifar()
    transform_mnist = normalize_data_mnist()

    if args.data == 'Cifar':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train_cifar)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test_cifar)
        trainset_select = select_trainset(trainset, args)
    if args.data == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_mnist)
        trainset_select = select_trainset(trainset, args)
    if args.data == 'FMNIST':
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False,transform=transform_mnist)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_mnist)
        trainset_select = select_trainset(trainset, args)

    trainloader = torch.utils.data.DataLoader(trainset_select, batch_size=args.batchsize, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader, len(trainset_select), len(testset)


def init(args):
    """
    Make the net/device/data/criterion/optimizer
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    trainloader, testloader, train_data_num, test_data_num = load_data(args)

    if args.data != 'Cifar':
        net = CNN4lite().to(device)
    if args.data == 'Cifar':
        net = CNN4Cifar().to(device)

    # Define loss functions and optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    return device, trainloader, testloader, net, criterion, optimizer, train_data_num, test_data_num
