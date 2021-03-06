from model.lstm import *
import torch
import torch.optim as optim


def load_data(args):
    """
    load data from txt (or make Non-IID data)
    """
    data_path = './data/' + args.data + '/' + args.data + '.txt'
    data_idx_path = './noniid/temp/Shakespeare/' + str(args.idx_user) + '.txt'
    data = open(data_path, 'r').read()
    data_idx = open(data_idx_path, 'r').read()
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    chars_idx = sorted(list(set(data_idx)))
    data_idx_size, vocab_idx_size = len(data_idx), len(chars_idx)

    print("-----------load data...-----------------")
    print("Data has {} characters, {} unique".format(data_size, vocab_size))
    print("Client {} has {} characters, {} unique".format(args.idx_user-1, data_idx_size, vocab_idx_size))
    print("----------------------------------------")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # char to index and index to char maps
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    data_train_select = list(data_idx)
    data_test_select = list(data)

    data_train = data_train_select
    data_test = data_test_select[int(data_size * 0.9): data_size-1]
    
    train_num = len(data_train)
    test_num = len(data_test)

    for i, ch in enumerate(data_train):
        data_train[i] = char_to_ix[ch]
    for i, ch in enumerate(data_test):
        data_test[i] = char_to_ix[ch]

    data_train = torch.tensor(data_train).to(device)
    data_test = torch.tensor(data_test).to(device)
    data_train = torch.unsqueeze(data_train, dim=1)
    data_test = torch.unsqueeze(data_test, dim=1)

    return data_train, data_test, data_size, vocab_size, char_to_ix, ix_to_char, device, train_num, test_num


def init(args):
    """
    Make the net/device/data/criterion/optimizer/train_num/test_num
    """
    hidden_size = 512  # size of hidden state
    num_layers = 3  # num of layers in LSTM layer stack

    data_train, data_test, data_size, vocab_size, char_to_ix, ix_to_char, device, train_num, test_num = load_data(args)

    rnn = RNN(vocab_size, vocab_size, hidden_size, num_layers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=args.learning_rate)

    return device, rnn, data_train, data_test, criterion, optimizer, train_num, test_num
