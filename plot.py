import pandas as pd
import argparse
import matplotlib.pyplot as plt

def per2float(per):
    """
    Delete the '%' in acc
    """
    return float(per[:-1])

def GetList(train_consult,test_consult):
    """
    Get the prasers of 'x'&'y'
    """
    x = []
    # x_test = []
    train_loss = []
    train_acc = []
    test_acc = []
    for i in range(len(train_consult)-1):
        if train_consult.iloc[i, 0] + 1 == train_consult.iloc[i+1,0] or i == len(train_consult)-2:
            if train_consult.iloc[i, 0] > test_consult.iloc[len(test_consult)-1, 1]:
                break
            x.append(train_consult.iloc[i, 0])
            train_loss.append(train_consult.iloc[i, 4])
            train_acc.append(per2float(train_consult.iloc[i, 7]))
            test_acc.append(per2float(test_consult.iloc[train_consult.iloc[i, 0]-1, 4]))
    return x, train_loss, train_acc, test_acc

def draw(x, train_loss, train_acc, test_acc, args):
    title = args.model + '---' + args.data

    plt.figure()
    plt.suptitle(title)

    plt.subplot(2, 1, 1)
    plt.grid()
    plt.plot(x, train_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.grid()
    plt.plot(x, train_acc, label='train')
    plt.plot(x, test_acc, label='test')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Acc')

    plt.tight_layout()

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot')
    parser.add_argument("-m", "--model", help="resnet18 or cnn or lstm", type=str, default='cnn')
    parser.add_argument("-d", "--data", help="Cifar or  MINIST or FMNIST or Shakespeare", type=str, default='MNIST')
    args = parser.parse_args()

    train_path = "./log/log_" + args.model + "_" + args.data + ".txt"
    test_path = "./acc/acc_" + args.model + "_" + args.data + ".txt"

    train_consult = pd.read_csv(train_path, sep="\s", header=None, engine='python',
                                names=['0', '1', '2', '3', '4', '5', '6', '7'])
    test_consult = pd.read_csv(test_path, sep="\s|,|=", header=None, engine='python',
                                names=['0', '1','2','3','4'])


    x, train_loss, train_acc, test_acc = GetList(train_consult, test_consult)
    draw(x, train_loss, train_acc, test_acc, args)

    print(train_consult)


if __name__ == "__main__":
    main()
