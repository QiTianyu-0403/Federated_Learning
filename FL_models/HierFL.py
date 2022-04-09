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

def run_worker(args):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port
    print("waiting for connecting......")