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
        
    def run_episode(self, epoch_s, args):
        futs = []
        for edge_rref in self.edge_rrefs:
            futs.append(rpc.rpc_async(edge_rref.owner(), _call_method, args=(Edge.run_episode, \
            edge_rref, args), timeout=0))
        for fut in futs:
            fut.wait()
            


class Edge(object):
    def __init__(self, args, edge_rank, topo):
        self.device, _, self.test_loader, self.model, _, self.optimizer, _, _ = init_fuc(args)
        self.edge_rref = rpc.RRef(self)
        self.worker_rrefs = []
        self.world_size = args.world_size
        os.environ['MASTER_ADDR'] = topo[edge_rank - 1]
        os.environ['MASTER_PORT'] = '29501'
        os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
        rpc.init_rpc(name='edge{}'.format(edge_rank), rank=0, world_size=2)
        print("{} has been initialized successfully".format(rpc.get_worker_info().name))
        for worker_rank in range(1, 2):
            worker_info = rpc.get_worker_info('worker{}'.format(worker_rank))
            self.worker_rrefs.append(rpc.remote(worker_info, Worker, args=(args,edge_rank)))
        print("The subnet {} RRef map has been created successfully!".format(args.subnet))
        print("The length of RRef is {}".format(len(self.edge_rrefs)))

    def run_episode(self, args):
        futs = []
        for worker_rref in self.worker_rrefs:
            futs.append(rpc.rpc_async(worker_rref.owner(), _call_method, args=(Worker.run_episode, \
            worker_rref, args), timeout=0))
        for fut in futs:
            fut.wait()


class Worker(object):
    def __init__(self, args):
        self.device, self.train_loader, _, self.model, self.criterion, self.optimizer, self.train_num, _ = init_fuc(args)
        self.idx = args.idx_user + 1
        print("{} has received the {} data successfully!".format(rpc.get_worker_info().name, self.train_num))

    def run_episode(self, args):
        import time
        from itertools import cycle

        cycle_iter = cycle([100, 200, 300])
        for item in cycle_iter:
            print(item)
        time.sleep(1)


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
    topo = ['192.168.1.112', '192.168.1.113']
    if args.subnet == 0:
        print("waiting for connecting......")
        # The server
        if args.rank == 0:
            os.environ['MASTER_ADDR'] = args.addr
            os.environ['MASTER_PORT'] = args.port
            os.environ["GLOO_SOCKET_IFNAME"] = "wlan0"
            rpc.init_rpc(name='server', rank=args.rank, world_size=args.world_size)
            print("{} has been initialized successfully".format(rpc.get_worker_info().name))
            server = Server(args)
            for edge_rank in range(1, args.world_size):
                edge_info = rpc.get_worker_info('edge{}'.format(edge_rank))
                server.edge_rrefs.append(rpc.remote(edge_info, Edge, args=(args, edge_rank, topo)))
            print("The subnet {} RRef map has been created successfully!".format(args.subnet))
            print("The length of RRef is {}".format(len(server.edge_rrefs)))
            for i in range(args.EPOCH):
                server.run_episode(i, args)
        
        else:
            os.environ['MASTER_ADDR'] = args.addr
            os.environ['MASTER_PORT'] = args.port
            os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
            rpc.init_rpc(name='edge{}'.format(args.rank), rank=args.rank, world_size=args.world_size)
            print("{} has been initialized successfully".format(rpc.get_worker_info().name))
    
    else:
        print("waiting for connecting......")
        os.environ['MASTER_ADDR'] = topo[args.subnet - 1]
        os.environ['MASTER_PORT'] = '29501'
        os.environ["GLOO_SOCKET_IFNAME"] = "wlan0"
        rpc.init_rpc(name='worker{}'.format(args.rank), rank=args.rank, world_size=2)
        print("{} has been initialized successfully".format(rpc.get_worker_info().name))
        
    rpc.shutdown()
    print('fuck')