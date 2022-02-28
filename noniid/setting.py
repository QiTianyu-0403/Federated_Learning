import argparse
from noniid.data_noniid.Cifar_noniid import get_dataset_cifar10_noniid
from noniid.data_noniid.MNIST_noniid import get_dataset_mnist_noniid
from noniid.data_noniid.FMNIST_noniid import get_dataset_fmnist_noniid
from noniid.data_noniid.Shakespeare_noniid import divide_in_txt
from file_flow import user_noniid_in_file

def main():
    parser = argparse.ArgumentParser(description='NonIID')
    parser.add_argument("-d", "--data", help="Cifar or  MINIST or FMNIST", type=str, default='Shakespeare')
    parser.add_argument("-nm", "--noniid_model", help="quantity_noniid or label_noniid or iid/Shakespeare:iid or noniid",
                        type=str, default='iid')
    parser.add_argument("-nu", "--num_users", help="The number of clients", type=int, default=3)
    parser.add_argument("-ts", "--total_samples", help="The total samples of each clients", type=int, default=500)
    parser.add_argument("-ru", "--rate_unbalance", help="The proportion of noniid (<=1.0) ", type=float, default=0.6)
    parser.add_argument("-nc", "--num_class", help="The classes number of noniid (<=10) ", type=int, default=2)
    args = parser.parse_args()

    if args.data == 'Cifar':
        dict_users_train = get_dataset_cifar10_noniid(args)
        user_noniid_in_file(dict_users_train, args)
    if args.data == 'MNIST':
        dict_users_train = get_dataset_mnist_noniid(args)
        user_noniid_in_file(dict_users_train, args)
    if args.data == 'FMNIST':
        dict_users_train = get_dataset_fmnist_noniid(args)
        user_noniid_in_file(dict_users_train, args)
    if args.data == 'Shakespeare':
        divide_in_txt(args)


if __name__ == "__main__":
    main()