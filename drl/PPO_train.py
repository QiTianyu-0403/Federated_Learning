from drl.agent import Agent
from drl.environment import Env
from drl.memory import Memory
from drl.trajectory import Sec, Tem
from prettytable import PrettyTable


class PPO_server(object):
    def __init__(self, args):
        self.sec = Sec() # Trajectory
        self.tem = Tem()
        self.env = Env(args.topo_num[0] - 1) # enviroment
        self.agent = Agent(args.topo_num[0] - 1, args.greedy) # agent
        self.memory = Memory() # memory
        
    def train(self, epoch_n, args):
        print('*****************************')
        print(f'Start DRL training: {epoch_n + 1}/{args.EPOCH}')
        training_count = 0
        last_accuracy, accuracy = 0, 0
        state, observer = self.env.reset()
        print('hello')
        # Round learning
        while True:
            self.env.done, don=False, False
            self.sec.clear_sec()
            table = PrettyTable(['Current state', 'Action', 'Next state'])
            while not don:
                action, probs, value = self.agent.choose_action(observer)
                next_state, next_observer, don = self.env.step(action)
                self.sec.append_sec(observer, action, probs, value, don)
                if not don:
                    table.add_row([state.numpy().astype(int), action, next_state.numpy().astype(int)])
                state, observer = next_state,next_observer
            print(table)
            break
        print(action, probs, value)