class Sec():
    def __init__(self):
        self.obs = []
        self.action = []
        self.don = []
    
    def clear_sec(self):
        del self.obs[:]
        del self.action[:]
        del self.don[:]


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
