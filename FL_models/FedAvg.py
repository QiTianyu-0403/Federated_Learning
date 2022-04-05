import os
import torch.distributed.rpc as rpc
import torch
import init.init_cnn as cnn_module
from collections import OrderedDict


class Server(object):
    def __init__(self, args):
        self.device, _, self.test_loader, self.model, _, self.optimizer = init_fuc(args)
        self.server_rref = rpc.RRef(self)
        self.worker_rrefs = []
        self.world_size = args.world_size
        print("{} has received the {} data successfully!".format(rpc.get_worker_info().name, len(self.test_loader)))

    def run_episode(self, epoch_s, args):
        futs, update_paras = [], []
        para = self.model.state_dict()
        for worker_rref in self.worker_rrefs:
            futs.append(rpc.rpc_async(worker_rref.owner(), _call_method, args=(Worker.run_episode, \
            worker_rref, para, args), timeout=0))
        for fut in futs:
            update_paras.append(fut.wait())
        self.model_average(*update_paras)
        self.evaluate(args, epoch_s)

    def model_average(self, *local_weights):
        global_weight = OrderedDict()
        weight = 1/len(local_weights)
        for index, local_update in enumerate([*local_weights]):
            for key in self.model.state_dict().keys():
                if index == 0:
                    global_weight[key] = weight*local_update[key]    # 修改系数为0.1，否则就不是平均了
                else:
                    global_weight[key] += weight*local_update[key]
        self.model.set_weights(global_weight)

    def evaluate(self, args, epoch_s):
        with open("./acc/" + "acc_" + args.model + "_" + args.data + ".txt", "w") as f:
            print("Waiting Test!")
            self.model.eval()
            with torch.no_grad():
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


class Worker(object):
    def __init__(self, args):
        self.device, self.train_loader, _, self.model, self.criterion, self.optimizer = init_fuc(args)
        self.idx = args.idx_user + 1

    def run_episode(self, para, args):
        # for model: CNN / MobileNet / ResNet
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


# get init informations according to args
def init_fuc(args):
    if args.model == "cnn":
        device, trainloader, testloader, net, criterion, optimizer = cnn_module.init(args)
    return device, trainloader, testloader, net, criterion, optimizer


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def run_worker(args):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port
    print("waiting for connecting......")

    if args.rank == 0:
        os.environ["GLOO_SOCKET_IFNAME"] = "wlp4s0"
        rpc.init_rpc(name='server', rank=args.rank, world_size=args.world_size)
        print("{} has been initialized successfully".format(rpc.get_worker_info().name))
        server = Server(args)
        for work_rank in range(1, args.world_size):
            args.idx_user = work_rank - 1
            work_info = rpc.get_worker_info('worker{}'.format(work_rank))
            server.worker_rrefs.append(rpc.remote(work_info, Worker, args=(args,)))
        print("RRef map has been created successfully!")
        print("The length of RRef is {}".format(len(server.worker_rrefs)))
        for i in range(args.EPOCH):
            server.run_episode(i, args)

    else:
        os.environ["GLOO_SOCKET_IFNAME"] = "wlan0"
        rpc.init_rpc(name='worker{}'.format(args.rank), rank=args.rank, world_size=args.world_size)
        print("{} has been initialized successfully".format(rpc.get_worker_info().name))

    rpc.shutdown()
