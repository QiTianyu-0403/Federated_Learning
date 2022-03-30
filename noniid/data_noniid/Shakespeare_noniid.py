import pandas as pd
from noniid.file_flow import split_integer
import shutil
import os
pd.set_option('display.max_columns', None)


def sum_idx_list(list):
    sum_list = [0]
    for i in range(len(list)):
        sum = 0
        for j in range(i+1):
            sum += list[j]
        sum_list.append(sum)
    return sum_list


def assign_list(lenth, n):
    assignment = []
    cout_list = split_integer(lenth,n)
    for i in range(len(cout_list)):
        for j in range(cout_list[i]):
            assignment.append(i)
    return assignment


def get_sort_data(args):
    # data_path = '../shakespeare.txt'
    data_path = '../data/Shakespeare/Shakespeare.txt'
    data = open(data_path, 'r').read()
    data = list(data)

    delete_list = []
    for i in range(1, len(data)-1):
        if data[i] == '\n' and data[i-1] == '\n' and data[i+1] == '\n':
            delete_list.append(i)
    x = 0
    for y in delete_list:
        data.pop(y-x)
        x+=1

    name_list = []
    idx_list = [0]
    part_list = []
    name = ""
    part = ""
    name_flag = 1

    for i in range(len(data)):
        if data[i - 1] == ':' and data[i] == '\n':
            if name_flag == 1:
                name_flag = 0
                name_list.append(name)
                name = ""
        if name_flag == 1:
            name += data[i]
        if data[i] == '\n' and data[i - 1] == '\n':
            name_flag = 1
            idx_list.append(i)
    idx_list.append(len(data))

    for i in range(len(idx_list) - 1):
        temp = data[idx_list[i]:idx_list[i + 1]]
        for j in range(len(temp)):
            part += temp[j]
        part_list.append(part)
        part = ""

    df = pd.DataFrame({'Name': name_list, 'Part': part_list, 'idx': idx_list[0:len(idx_list) - 1]})

    df_sort = df.sort_values(by=['Name'])
    user_list = assign_list(len(name_list), args.num_users)
    df_sort['User'] = user_list
    return df_sort


def get_iid_data(args):
    data_path = '../data/Shakespeare/Shakespeare.txt'
    data = open(data_path, 'r').read()
    data = list(data)
    lenth_list = split_integer(len(data), args.num_users)
    return data, lenth_list


def divide_in_txt(args):
    shutil.rmtree('./temp/Shakespeare/')
    os.mkdir('./temp/Shakespeare/')

    if args.noniid_model == 'noniid':
        dataframe_sort = get_sort_data(args)

        for i in range(args.num_users):
            df = dataframe_sort[dataframe_sort['User']==i]
            df = df.sort_values(by=['idx'])

            data_path = './temp/Shakespeare/' + str(i) + '.txt'

            with open(data_path, "w") as f:
                for j in range(df.shape[0]):
                    f.write(df.iloc[j][1])
                f.close()

    if args.noniid_model == 'iid':
        data, lenth_list = get_iid_data(args)
        sum_list = sum_idx_list(lenth_list)

        for i in range(args.num_users):
            data_path = './temp/Shakespeare/' + str(i) + '.txt'
            with open(data_path, "w") as f:
                for item in data[sum_list[i]:sum_list[i+1]-1]:
                    f.write(item)
                f.close()
