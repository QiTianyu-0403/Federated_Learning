import os
import torch.distributed.rpc as rpc

def run_worker(rank,world_size,args):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port
    os.environ["GLOO_SOCKET_IFNAME"] = 'wlan0'

    if rank == 0:
        rpc.init_rpc(name='server', rank=rank, world_size=world_size)
        print("{} has been initialized successfully".format(rpc.get_worker_info().name))

    else:
        rpc.init_rpc(name='worker{}'.format(rank), rank=rank, world_size=world_size)
        print("{} has been initialized successfully".format(rpc.get_worker_info().name))

    rpc.shutdown()