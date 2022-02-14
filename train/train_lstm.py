from init.init_lstm import init
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

def train_lstm(args):
    device, rnn, data_train, data_test, criterion, optimizer = init(args)
    pre_epoch = 0
    iter = 0

    print("Start Training: " + args.model + "--" + args.data)
    with open("./acc/"+"acc_"+args.model+"_"+args.data+".txt", "w") as f:
        with open("./log/"+"log_"+args.model+"_"+args.data+".txt", "w")as f2:

            # avg epoch : train
            for epoch in range(pre_epoch, args.EPOCH):
                print('\nEpoch: %d' % (epoch + 1))

                data_ptr = np.random.randint(100)
                n = 0
                sum_loss = 0
                correct = 0.0
                total = 0.0
                hidden_state = None

                while True:
                    input_seq = data_train[data_ptr: data_ptr + args.batchsize]
                    target_seq = data_train[data_ptr + 1: data_ptr + args.batchsize + 1]
                    input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    output, hidden_state = rnn(input_seq, hidden_state)
                    loss = criterion(torch.squeeze(output), torch.squeeze(target_seq))
                    loss.backward()
                    optimizer.step()

                    # loss + acc
                    sum_loss += loss.item()
                    _, predicted = torch.max(torch.squeeze(output).data, 1)
                    total += torch.squeeze(target_seq).size(0)
                    correct += predicted.eq(torch.squeeze(target_seq).data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, iter + 1, sum_loss / (n + 1), 100. * correct / total))
                    f2.write('%03d  %07d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, iter + 1, sum_loss / (n + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                    data_ptr += args.batchsize
                    n += 1
                    iter += 1

                    if data_ptr + args.batchsize + 1 > data_train.size(0):
                        break

                #  after epoch : test
                print("Waiting Test!")
                with torch.no_grad():
                    data_ptr = 0
                    hidden_state = None
                    sum_correct = 0
                    sum_test = 0

                    # random character from data to begin
                    rand_index = np.random.randint(100)

                    while True:
                        input_seq = data_test[rand_index + data_ptr: rand_index + data_ptr + 1]
                        target_seq = data_test[rand_index + data_ptr + 1: rand_index + data_ptr + 2]
                        output, hidden_state = rnn(input_seq, hidden_state)

                        output = F.softmax(torch.squeeze(output), dim=0)
                        dist = Categorical(output)
                        index = dist.sample()

                        if index.item() == target_seq[0][0]:
                            sum_correct += 1
                        sum_test += 1
                        data_ptr += 1

                        if data_ptr > data_test.size(0) - rand_index - 2:
                            break
                    print('测试分类准确率为：%.3f%%' % (100. * sum_correct / sum_test))
                    acc = 100. * sum_correct / sum_test
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()

            print("Training Finished, TotalEPOCH=%d" % args.EPOCH)