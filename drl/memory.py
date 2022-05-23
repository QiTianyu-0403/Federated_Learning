import numpy as np

class Memory():                 
    def __init__(self):
        self.observers = []
        self.actions = []
        self.rewards = []
        self.done = []
        self.probs = []
        self.values = []

    def clear_memory(self):
        del self.observers[:]
        del self.actions[:]
        del self.rewards[:]
        del self.done[:]
        del self.probs[:]
        del self.values[:]
    
    def push(self, tem):
        self.rewards.append(tem.rewards.copy())
        self.observers.append(tem.obs.copy())
        self.actions.append(tem.actions.copy())
        self.values.append(tem.values.copy())
        self.probs.append(tem.probs.copy())
        self.done.append(tem.done.copy())
        
    def sample(self, i):
        # print('++++++++++++++++++++')
        # print(self.rewards)
        # print(self.values)
        # print(self.done)
        # print('++++++++++++++++++++')
        indi = np.arange(len(self.observers), dtype=np.int64)
        np.random.shuffle(indi)
        return np.array(self.rewards[i]), np.array(self.observers[i]), np.array(self.actions[i]), \
            np.array(self.values[i]), np.array(self.probs[i]), np.array(self.done[i]), indi