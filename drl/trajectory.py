class Sec():
    def __init__(self):
        self.obs = []
        self.action = []
        self.don = []
        self.prob = []
        self.value = []
    
    def clear_sec(self):
        del self.obs[:]
        del self.action[:]
        del self.don[:]
        del self.prob[:]
        del self.value[:]
        
    def append_sec(self, observer, action, probs, value, don):
        self.obs.append(observer.numpy())
        self.action.append(action)
        self.prob.append(probs)
        self.value.append(value)
        self.don.append(don)


class Tem():
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.values = []
        self.done = []
    
    def clear_tem(self):
        del self.obs[:]
        del self.actions[:]
        del self.rewards[:]
        del self.probs[:]
        del self.values[:]
        del self.done[:]
    
    def append_tem(self, reward, sec_obs, sec_action, sec_probs, sec_values, sec_don):
        self.rewards.append(reward)
        self.obs.extend(sec_obs.copy())
        self.actions.extend(sec_action.copy())
        self.probs.extend(sec_probs.copy())
        self.values.extend(sec_values.copy())
        self.done.extend(sec_don.copy())
