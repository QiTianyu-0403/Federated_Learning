import pandas as pd
import torch
import numpy as np
import math
from torch.utils.data import TensorDataset

def split_integer(m, n):
    assert n > 0
    quotient = int(m / n)
    remainder = m % n
    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [quotient] * n

def test_label(dict_user_train, train_dataset):
    for i in range(len(dict_user_train)):
        test_num = np.zeros((10,), dtype=np.int)
        for j in range(len(dict_user_train[i])):
            idxs = dict_user_train[i][j]
            label = train_dataset[idxs][1]
            test_num[label] += 1
        print('----------', end=' ')
        print('Client : ',i, end=' ')
        print('----------')
        print(test_num)

def user_noniid_in_file(dict_users_train, args):
    if args.noniid_model == 'label_noniid':
        file_name = './temp/' + args.data + '/' + args.data + '_' + args.noniid_model + '_users' + str(args.num_users) +\
                    '_data' + str(args.total_samples) + '_unbalance' + str(args.rate_unbalance) + '.csv'
    if args.noniid_model == 'quantity_noniid':
        file_name = './temp/' + args.data + '/' + args.data + '_' + args.noniid_model + '_users' + str(args.num_users) +\
                    '_data' + str(args.total_samples) + '.csv'
    if args.noniid_model == 'iid':
        file_name = './temp/' + args.data + '/' + args.data + '_' + args.noniid_model + '_users' + str(args.num_users) +\
                    '_data' + str(args.total_samples) + '.csv'
    frame = pd.DataFrame.from_dict(dict_users_train, orient='index')

    frame.to_csv(file_name)

def user_out_file(args):
    if args.noniid_model == 'label_noniid':
        file_name = './noniid/temp/' + args.data + '/' + args.data + '_' + args.noniid_model + '_users' + str(args.num_users) +\
                    '_data' + str(args.total_samples) + '_unbalance' + str(args.rate_unbalance) + '.csv'
    if args.noniid_model == 'quantity_noniid':
        file_name = './noniid/temp/' + args.data + '/' + args.data + '_' + args.noniid_model + '_users' + str(args.num_users) +\
                    '_data' + str(args.total_samples) + '.csv'
    if args.noniid_model == 'iid':
        file_name = './noniid/temp/' + args.data + '/' + args.data + '_' + args.noniid_model + '_users' + str(args.num_users) +\
                    '_data' + str(args.total_samples) + '.csv'
    frame = pd.read_csv(file_name)
    train_idx = []
    for i in range(frame.shape[1]-1):
        if math.isnan(frame.iloc[args.rank-1, i+1]):
            break
        train_idx.append(frame.iloc[args.rank-1, i+1])

    return train_idx

def select_trainset(trainset, args):
    train_idx = user_out_file(args)
    train_select_list = []
    train_label_list = []
    for i in range(len(train_idx)):
        train_select_list.append(trainset[i][0])
        train_label_list.append(trainset[i][1])
    train_select_tens = torch.stack(train_select_list)
    train_label_tens = torch.tensor(train_label_list)
    trainset_select = TensorDataset(train_select_tens,train_label_tens)

    return trainset_select
