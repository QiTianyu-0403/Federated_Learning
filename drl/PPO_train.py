from drl.agent import Agent
from drl.environment import Env
from drl.memory import Memory
from drl.trajectory import Sec, Tem


class PPO_server(object):
    def __init__(self, args):
        self.sec = Sec() # Trajectory
        self.tem = Tem()
        self.env = Env(args.topo_num[0] - 1) # enviroment
        self.agent = Agent(args.topo_num[0] - 1, args.greedy) # agent
        self.memory = Memory() # memory
        
    def train(self, epoch_n, args):
        print('hello')