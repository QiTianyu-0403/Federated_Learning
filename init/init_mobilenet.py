import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from model.resnet import *
from model.mobilenet import *
from torchsummary import summary
from noniid.file_flow import select_trainset


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
    transform_mnist = normalize_data_mnist()

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader, train_data_num, test_data_num = load_data(args)

    net = mobilenetv2().to(device)

    # Define loss functions and optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    return device, trainloader, testloader, net, criterion, optimizer, train_data_num, test_data_num
