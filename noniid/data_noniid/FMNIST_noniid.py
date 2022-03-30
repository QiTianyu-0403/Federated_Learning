import torchvision
import numpy as np
from init.init_resnet18 import normalize_data_mnist
from noniid.file_flow import split_integer, test_label


def get_dataset_fmnist_noniid(args):
    transform_train_mnist = normalize_data_mnist()

    train_dataset = torchvision.datasets.FashionMNIST('../data', train=True, download=False,
                                   transform=transform_train_mnist)

    if args.noniid_model == 'label_noniid':
        data_users_train = fmnist_extr_label_noniid(train_dataset, args)
    if args.noniid_model == 'quantity_noniid':
        data_users_train = fmnist_extr_quantity_noniid(train_dataset, args)
    if args.noniid_model == 'iid':
        data_users_train = fmnist_extr_iid(train_dataset, args)
    return data_users_train


def fmnist_extr_label_noniid(train_dataset, args):
    dict_users_train = {i: np.array([]) for i in range(args.num_users)}
    idxs = np.arange(60000)
    labels = np.array(train_dataset.targets)

    # probability
    prop_equal = 10 * (1-args.rate_unbalance)/(10-args.num_class)
    num_per_equal = int(args.total_samples * prop_equal / 10)
    assert ((1-args.rate_unbalance)/(10-args.num_class) <= args.rate_unbalance/args.num_class)

    # verb and sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

    # shuffle
    for i in range(10):
        np.random.shuffle(idxs_labels[0][6000 * i:6000 * (i+1)])
    idxs = idxs_labels[0, :]

    # equal part
    for i in range(args.num_users):
        for j in range(10):
            dict_users_train[i] = np.concatenate((dict_users_train[i], idxs[j * 6000 + i * num_per_equal: j * 6000 + (1 + i) * num_per_equal]), axis=0)

    # noniid init part
    residue_noniid = 60000 - num_per_equal * 10 * args.num_users
    num_samples = int((args.total_samples - num_per_equal*10) / args.num_class)
    num_shards_train, num_imgs_train = int(residue_noniid / num_samples), num_samples
    assert (args.num_class * args.num_users <= num_shards_train)
    assert (args.num_class <= 10)
    idx_shard = [i for i in range(num_shards_train)]

    # delete rank
    delete_rank = np.array([])
    delete_res = residue_noniid - num_shards_train * num_samples
    surplus_list = split_integer(delete_res,10)
    for i in range(10):
        delete_rank = np.append(delete_rank, np.arange(i * 6000,  args.num_users * num_per_equal + i * 6000, 1))
    for i in range(len(surplus_list)):
        delete_rank = np.append(delete_rank, np.arange((i + 1) * 6000 - surplus_list[i], (i + 1) * 6000, 1))
    delete_rank = delete_rank.astype(int)
    idxs_labels = np.delete(idxs_labels, delete_rank, axis=1)
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(args.num_users):
        rand_set = set(np.random.choice(idx_shard, args.num_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_train[i] = np.concatenate((dict_users_train[i], idxs[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)

    # change type of dict_user_train
    for i in range(len(dict_users_train)):
        dict_users_train[i] = dict_users_train[i].astype(int)

    test_label(dict_users_train, train_dataset)
    return dict_users_train


def fmnist_extr_quantity_noniid(train_dataset, args):
    dict_users_train = {i: np.array([]) for i in range(args.num_users)}
    idxs = np.arange(60000)
    labels = np.array(train_dataset.targets)

    # verb and sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

    # shuffle
    for i in range(10):
        np.random.shuffle(idxs_labels[0][6000 * i:6000 * (i+1)])
    idxs = idxs_labels[0, :]

    # equal difference compute
    equal_diff = 2 * args.total_samples * args.num_users / (args.num_users * (args.num_users + 3))
    num_list, num_per_user_list = [], []
    assert (args.num_users > 1)
    for i in range(args.num_users - 1):
        num_list.append(round(equal_diff*(i+2),-1))
    num_list.append(args.num_users * args.total_samples - sum(num_list))
    for i in range(len(num_list)):
        num_per_user_list.append(int(num_list[i]/10))

    # equal part
    for i in range(args.num_users):
        for j in range(10):
            dict_users_train[i] = np.concatenate((dict_users_train[i], idxs[j * 6000 + i * num_per_user_list[i]: j * 6000 + (1 + i) * num_per_user_list[i]]), axis=0)

    # change type of dict_user_train
    for i in range(len(dict_users_train)):
        dict_users_train[i] = dict_users_train[i].astype(int)

    test_label(dict_users_train, train_dataset)
    return dict_users_train


def fmnist_extr_iid(train_dataset, args):
    dict_users_train = {i: np.array([]) for i in range(args.num_users)}
    idxs = np.arange(60000)
    labels = np.array(train_dataset.targets)

    # verb and sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

    # shuffle
    for i in range(10):
        np.random.shuffle(idxs_labels[0][6000 * i:6000 * (i+1)])
    idxs = idxs_labels[0, :]

    num_per_user = int(args.total_samples/10)

    for i in range(args.num_users):
        for j in range(10):
            dict_users_train[i] = np.concatenate((dict_users_train[i], idxs[j * 6000 + i * num_per_user: j * 6000 + (1 + i) * num_per_user]), axis=0)

    # change type of dict_user_train
    for i in range(len(dict_users_train)):
        dict_users_train[i] = dict_users_train[i].astype(int)

    test_label(dict_users_train, train_dataset)
    return dict_users_train
