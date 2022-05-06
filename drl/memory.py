class Memory():                 
    def __init__(self):
        self.observers=[]
        self.next_observers=[]
        self.actions=[]
        self.rewards=[]
        self.adv=[]
        self.done=[]
        self.discount_reward=[]

    def clear_memory(self):
        del self.observers[:]
        del self.next_observers[:]
        del self.actions[:]
        del self.rewards[:]
        del self.done[:]
        del self.discount_reward[:]
        del self.adv[:]