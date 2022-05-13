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
        self.rewards.append(tem.rewards)
        self.observers.append(tem.obs)
        self.actions.append(tem.actions)
        self.values.append(tem.values)
        self.probs.append(tem.probs)
        self.done.append(tem.done)