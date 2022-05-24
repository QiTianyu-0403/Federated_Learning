import os
# import torch.distributed.rpc as rpc
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import init.init_cnn as cnn_module
import init.init_mobilenet as mobilenet_module
import init.init_resnet18 as resnet18_module
import init.init_lstm as lstm_module
from collections import OrderedDict
import ray


@ray.remote
class Server(object):
    def __init__(self, args):
        self.device, _, self.test_loader, self.model, _, self.optimizer, _, self.test_num = init_fuc(args)
        # self.server_rref = rpc.RRef(self)
        self.worker_rrefs = []
        self.world_size = args.world_size
        # print(f"{rpc.get_worker_info().name} has received the {self.test_num} data successfully!")

    def run_episode(self, epoch_s, args, workers):
        print(f'Round: {epoch_s + 1}')
        update_paras, data_num = [], []
        para = self.model.state_dict()
        
        for i in range(len(workers)):
            data_num.append(ray.get(workers[i].get_data_num.remote()))
            update_paras.append(ray.get(workers[i].run_episode.remote(para, args)))
        # # futs: run_episode ; weight_futs: get_data_num 
        # futs, update_paras = [], []
        # weight_futs, data_num = [], []

        # para = self.model.state_dict()
        # for worker_rref in self.worker_rrefs:
        #     futs.append(rpc.rpc_async(worker_rref.owner(), _call_method, args=(Worker.run_episode, \
        #     worker_rref, para, args), timeout=0))
        #     weight_futs.append(rpc.rpc_async(worker_rref.owner(), _call_method, args=(Worker.get_data_num, \
        #     worker_rref), timeout=0))
        # update_paras.extend(fut.wait() for fut in futs)
        # data_num.extend(weight_fut.wait() for weight_fut in weight_futs)
        print(data_num)
        self.model_average(*update_paras, data_num = data_num)
        self.evaluate(args, epoch_s)

    def model_average(self, *local_weights, data_num):
        global_weight = OrderedDict()
        server_data_sum = sum(data_num)
        for index, local_update in enumerate([*local_weights]):
            weight = data_num[index]/server_data_sum
            for key in self.model.state_dict().keys():
                if index == 0:
                    global_weight[key] = weight*local_update[key]    
                else:
                    global_weight[key] += weight*local_update[key]
        self.model.set_weights(global_weight)

    def evaluate(self, args, epoch_s):
        with open("./acc/" + "acc_" + args.model + "_" + args.data + ".txt", "w") as f:
            print("Waiting Test!")
            self.model.eval()
            with torch.no_grad():
                # for model: CNN / MobileNet / ResNet18
                if args.model != 'lstm': 
                    correct = 0
                    total = 0
                    for data in self.test_loader:
                        self.model.eval()
                        images, labels = data
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = self.model(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('The Global Test Accuracy is: %.3f%%' % (100. * correct / total))
                    acc = 100. * correct / total
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch_s + 1, acc))
                    f.write('\n')
                    f.flush()
                
                # for model: LSTM
                if args.model == 'lstm':
                    data_ptr = 0
                    hidden_state = None
                    sum_correct = 0
                    sum_test = 0

                    # random character from data to begin
                    rand_index = np.random.randint(100)

                    while True:
                        input_seq = self.test_loader[rand_index + data_ptr: rand_index + data_ptr + 1]
                        target_seq = self.test_loader[rand_index + data_ptr + 1: rand_index + data_ptr + 2]
                        output, hidden_state = self.model(input_seq, hidden_state)

                        output = F.softmax(torch.squeeze(output), dim=0)
                        dist = Categorical(output)
                        index = dist.sample()

                        if index.item() == target_seq[0][0]:
                            sum_correct += 1
                        sum_test += 1
                        data_ptr += 1

                        if data_ptr > self.test_loader.size(0) - rand_index - 2:
                            break
                    print('The Global Test Accuracy is: %.3f%%' % (100. * sum_correct / sum_test))
                    acc = 100. * sum_correct / sum_test
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch_s + 1, acc))
                    f.write('\n')
                    f.flush()


@ray.remote
class Worker(object):
    def __init__(self, args):
        self.device, self.train_loader, _, self.model, self.criterion, self.optimizer, self.train_num, _ = init_fuc(args)
        self.idx = args.idx_user + 1
        # print(f"{rpc.get_worker_info().name} has received the {self.train_num} data successfully!")

    def run_episode(self, para, args):
        # for model: CNN / MobileNet / ResNet-18 / LSTM
        print(f'-----------------Worker {self.idx} is running!-----------------')
        if args.model != "lstm":
            self.model.load_state_dict(para)
            self.model.zero_grad()
            pre_epoch = 0
            for epoch in range(pre_epoch, args.epoch_worker):
                print('\nEpoch: %d' % (epoch + 1))
                self.model.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(self.train_loader, 0):
                    length = len(self.train_loader)
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()

                    # forward + backward
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)    # predicted返回的是tensor每行最大的索引值
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ----Rank%d'
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total, self.idx))
            local_para = self.model.state_dict()
            return local_para

        if args.model == "lstm":
            self.model.load_state_dict(para)
            self.model.zero_grad()
            pre_epoch = 0
            iter = 0
            for epoch in range(pre_epoch, args.epoch_worker):
                print('\nEpoch: %d' % (epoch + 1))

                data_ptr = np.random.randint(100)
                n = 0
                sum_loss = 0
                correct = 0.0
                total = 0.0
                hidden_state = None

                while True:
                    input_seq = self.train_loader[data_ptr: data_ptr + args.batchsize]
                    target_seq = self.train_loader[data_ptr + 1: data_ptr + args.batchsize + 1]
                    input_seq, target_seq = input_seq.to(self.device), target_seq.to(self.device)
                    self.optimizer.zero_grad()

                    # forward + backward
                    output, hidden_state = self.model(input_seq, hidden_state)
                    loss = self.criterion(torch.squeeze(output), torch.squeeze(target_seq))
                    loss.backward()
                    self.optimizer.step()

                    # loss + acc
                    sum_loss += loss.item()
                    _, predicted = torch.max(torch.squeeze(output).data, 1)
                    total += torch.squeeze(target_seq).size(0)
                    correct += predicted.eq(torch.squeeze(target_seq).data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, iter + 1, sum_loss / (n + 1), 100. * correct / total))

                    data_ptr += args.batchsize
                    n += 1
                    iter += 1

                    if data_ptr + args.batchsize + 1 > self.train_loader.size(0):
                        break
            local_para = self.model.state_dict()
            return local_para
            
    def get_data_num(self):
        return self.train_num


# get init informations according to args
def init_fuc(args):
    if args.model == "cnn":
        device, trainloader, testloader, net, criterion, optimizer, train_num, test_num = cnn_module.init(args)
    if args.model == "mobilenet":
        device, trainloader, testloader, net, criterion, optimizer, train_num, test_num = mobilenet_module.init(args)
    if args.model == "resnet18":
        device, trainloader, testloader, net, criterion, optimizer, train_num, test_num = resnet18_module.init(args)
    if args.model == "lstm":
        device, net, trainloader, testloader, criterion, optimizer, train_num, test_num = lstm_module.init(args)
    return device, trainloader, testloader, net, criterion, optimizer, train_num, test_num


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def run_worker(args):
    ray.init()
    server = Server.remote(args)
    workers = []
    for work_rank in range(1, args.world_size):
        args.idx_user = work_rank - 1
        workers.append(Worker.remote(args))
    print('hello')
    for i in range(args.EPOCH):
        ray.get(server.run_episode.remote(i, args, workers))
    ray.shutdown()