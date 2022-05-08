class Sec():
    def __init__(self):
        self.obs = []
        self.action = []
        self.don = []
        self.probs = []
        self.value = []
    
    def clear_sec(self):
        del self.obs[:]
        del self.action[:]
        del self.don[:]
        del self.probs[:]
        del self.value[:]
        
    def append_sec(self, observer, action, probs, value, don):
        self.obs.append(observer.numpy())
        self.action.append(action)
        self.probs.append(probs)
        self.value.append(value)
        self.don.append(don)


class Tem():
    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.don = []
        self.v = []
    
    def clear_tem(self):
        del self.obs[:]
        del self.action[:]
        del self.reward[:]
        del self.don[:]
        del self.v[:]
