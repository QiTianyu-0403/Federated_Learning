import os
import torch.distributed.rpc as rpc
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import init.init_cnn as cnn_module
import init.init_mobilenet as mobilenet_module
import init.init_resnet18 as resnet18_module
import init.init_lstm as lstm_module
from collections import OrderedDict


class Server(object):
    def __init__(self, args):
        self.device, _, self.test_loader, self.model, _, self.optimizer, _, self.test_num = init_fuc(args)
        self.server_rref = rpc.RRef(self)
        self.edge_rrefs = []
        print("{} has received the {} data successfully!".format(rpc.get_worker_info().name, self.test_num))

    # Get the worker's RRef
    def make_subnet(self, topo, args):
        futs = []
        for edge_rref in self.edge_rrefs:
            futs.append(rpc.rpc_async(edge_rref.owner(), _call_method, args=(Edge.get_worker_rref, \
                edge_rref, edge_rref.owner().id, topo, args), timeout=0))
        for fut in futs:
            fut.wait()
        print('The subnet has been finished!')
        
    def run_edge_episode(self, epoch_s, args):
        print('Server Round: {}'.format(epoch_s + 1))
        para = self.model.state_dict()
        # The 2st epoch Edge_Round
        for i in range(args.epoch_edge):
            # The Edge data information (Sum the workers data they link) is in data_num
            futs, update_paras, data_num, data_sum_num = [], [], [], []
            for edge_rref in self.edge_rrefs:
                futs.append(rpc.rpc_async(edge_rref.owner(), _call_method, args=(Edge.run_worker_episode, \
                    edge_rref, i, para, args), timeout=0))
            for fut in futs:
                update_model, edge_data = fut.wait()
                update_paras.append(update_model)
                data_sum_num.append(edge_data)
            # Get the each Edge sum data
            for j in range(len(data_sum_num)):
                data_num.append(sum(data_sum_num[j]))
            print('server', data_num)
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
        with open("./acc/" + "acc_" + args.model + "_" + args.data + ".txt", "a") as f:
            print("Waiting Test!")
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


class Edge(object):
    def __init__(self, args):
        self.device, _, self.test_loader, self.model, _, self.optimizer, _, _ = init_fuc(args)
        self.edge_rref = rpc.RRef(self)
        self.worker_rrefs = []
        
    def get_worker_rref(self, id, topo, args):
        for worker_rank in topo[id]:
            args.idx_user = worker_rank - len(topo[0])
            worker_info = rpc.get_worker_info('worker{}'.format(worker_rank))
            self.worker_rrefs.append(rpc.remote(worker_info, Worker, args=(args,)))
        print("The Edge {} RRef map has been created successfully!".format(len(topo[id])))
        print("The length of RRef is {}".format(len(self.worker_rrefs)))
        
    def run_worker_episode(self, epoch_e, para, args):
        print('Edge Round: {}'.format(epoch_e + 1))
        self.model.load_state_dict(para)
        futs, update_paras = [], []
        weight_futs, data_num = [], []
        for worker_rref in self.worker_rrefs:
            futs.append(rpc.rpc_async(worker_rref.owner(), _call_method, args=(Worker.run_episode, \
                worker_rref, para, args), timeout=0))
            weight_futs.append(rpc.rpc_async(worker_rref.owner(), _call_method, args=(Worker.get_data_num, \
                worker_rref), timeout=0))
        for fut in futs:
            update_paras.append(fut.wait())
        for weight_fut in weight_futs:
            data_num.append(weight_fut.wait())
        self.model_average(*update_paras, data_num = data_num)
        print('edge', data_num)
        return self.model.state_dict(), data_num
    
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


class Worker(object):
    def __init__(self, args):
        self.device, self.train_loader, _, self.model, self.criterion, self.optimizer, self.train_num, _ = init_fuc(args)
        self.idx = args.idx_user + 1
        print("{} has received the {} data successfully!".format(rpc.get_worker_info().name, self.train_num))

    def run_episode(self, para, args):
        # for model: CNN / MobileNet / ResNet-18 / LSTM
        # The 3st epoch Worker_Round
        print('-----------------Worker {} is running!-----------------'.format(self.idx))
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


def get_rank_list(args):
    topo = []
    count = 0
    for i in range(args.topo_num[0]):
        topo.append([])
    for i in range(len(args.topo_num)):
        if i == 0:
            for j in range(args.topo_num[0]):
                topo[0].append(j)
        else:
            count = topo[i - 1][-1] + 1
            for j in range(args.topo_num[i] - 1):
                topo[i].append(count)
                count += 1
    return topo


def run_worker(args):
    print("waiting for connecting......")
    topo = get_rank_list(args)
    # The server
    if args.rank == 0:
        os.environ['MASTER_ADDR'] = args.addr
        os.environ['MASTER_PORT'] = args.port
        os.environ["GLOO_SOCKET_IFNAME"] = "wlan0"
        rpc.init_rpc(name='server', rank=args.rank, world_size=args.world_size)
        print("{} has been initialized successfully".format(rpc.get_worker_info().name))
        server = Server(args)
        for edge_rank in range(1, args.topo_num[0]):
            edge_info = rpc.get_worker_info('edge{}'.format(edge_rank))
            server.edge_rrefs.append(rpc.remote(edge_info, Edge, args=(args,)))
        print("The Server {} RRef map has been created successfully!".format(args.topo_num[0]))
        print("The length of RRef is {}".format(len(server.edge_rrefs)))
        server.make_subnet(topo, args)
        if (os.path.exists("./acc/" + "acc_" + args.model + "_" + args.data + ".txt")) :
            os.remove("./acc/" + "acc_" + args.model + "_" + args.data + ".txt")
        # The 1st epoch Server_Round
        for i in range(args.EPOCH):
            server.run_edge_episode(i, args)

    if 0 < args.rank < args.topo_num[0]:
        os.environ['MASTER_ADDR'] = args.addr
        os.environ['MASTER_PORT'] = args.port
        os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
        rpc.init_rpc(name='edge{}'.format(args.rank), rank=args.rank, world_size=args.world_size)
        print("{} has been initialized successfully".format(rpc.get_worker_info().name))
    
    else:
        os.environ['MASTER_ADDR'] = args.addr
        os.environ['MASTER_PORT'] = args.port
        os.environ["GLOO_SOCKET_IFNAME"] = "wlan0"
        rpc.init_rpc(name='worker{}'.format(args.rank), rank=args.rank, world_size=args.world_size)
        print("{} has been initialized successfully".format(rpc.get_worker_info().name))

    rpc.shutdown()