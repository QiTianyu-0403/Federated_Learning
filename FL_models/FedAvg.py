import os
import torch.distributed.rpc as rpc
import torch
import init.init_cnn as cnn_module

def train_image(worker, para, args):
    worker.model.load_state_dict(para)
    worker.model.zero_grad()
    print("Start Training: " + args.model + "--" + args.data)

class Server(object):
    def __init__(self,args):
        _, _, self.test_loader, self.model, _, self.optimizer = init_fuc(args)
        self.server_rref=rpc.RRef(self)
        self.worker_rrefs=[]
        self.world_size=args.world_size
        print(self.model)
        print("{} has received the {} data successfully!".format(rpc.get_worker_info().name,len(self.test_loader)))

class Worker(object):
    def __init__(self, args):
        _, self.train_loader, _, self.model, _, self.optimizer = init_fuc(args)

def init_fuc(args):
    if args.model == "cnn":
        device, trainloader, testloader, net, criterion, optimizer = cnn_module.init(args)
    return device, trainloader, testloader, net, criterion, optimizer

def run_worker(args):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port
    os.environ["GLOO_SOCKET_IFNAME"] = 'wlan0'
    print("waiting for connecting......")

    if args.rank == 0:
        rpc.init_rpc(name='server', rank=args.rank, world_size=args.world_size)
        print("{} has been initialized successfully".format(rpc.get_worker_info().name))
        server = Server(args)

    else:
        rpc.init_rpc(name='worker{}'.format(args.rank), rank=args.rank, world_size=args.world_size)
        print("{} has been initialized successfully".format(rpc.get_worker_info().name))

    rpc.shutdown()