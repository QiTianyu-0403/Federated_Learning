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
        return self.rewards[i], self.observers[i], self.actions[i], self.values[i], \
            self.probs[i], self.done[i]