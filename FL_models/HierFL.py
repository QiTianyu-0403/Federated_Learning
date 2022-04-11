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
        self.world_size = args.world_size
        print("{} has received the {} data successfully!".format(rpc.get_worker_info().name, self.test_num))


class Edge(object):
    def __init__(self, args):
        self.device, _, self.test_loader, self.model, _, self.optimizer, _, _ = init_fuc(args)
        self.edge_rref = rpc.RRef(self)
        self.worker_rrefs = []
        self.world_size = args.world_size
        # print("{} has received the {} data successfully!".format(rpc.get_worker_info().name, self.test_num))


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
    # The 0 subnet
    if args.subnet == 0:
        os.environ['MASTER_ADDR'] = '192.168.1.101'
        os.environ['MASTER_PORT'] = args.port
        print("waiting for connecting......")
        # The server
        if args.rank == 0:
            os.environ["GLOO_SOCKET_IFNAME"] = "wlan0"
            rpc.init_rpc(name='server', rank=args.rank, world_size=args.world_size)
            print("{} has been initialized successfully".format(rpc.get_worker_info().name))
            server = Server(args)
            for edge_rank in range(1, args.world_size):
                edge_info = rpc.get_worker_info('edge{}'.format(edge_rank))
                server.edge_rrefs.append(rpc.remote(edge_info, Edge, args=(args,)))
            print("The subnet {} RRef map has been created successfully!".format(args.subnet))
            print("The length of RRef is {}".format(len(server.edge_rrefs)))
        
        else:
            os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
            rpc.init_rpc(name='edge{}'.format(args.rank), rank=args.rank, world_size=args.world_size)
            print("{} has been initialized successfully".format(rpc.get_worker_info().name))
    print('fuck')