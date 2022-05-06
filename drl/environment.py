import torch

class Env(object):
    def __init__(self,number_worker):
        self.num_worker=number_worker
        self.horizon=number_worker
        self.cur_state=torch.ones((1,self.num_worker),dtype=torch.float)
        self.observer=torch.cat((torch.zeros((self.num_worker,self.horizon),dtype=torch.float),self.cur_state.T),dim=1)[:,1:]
        self.border=10
        self.done=False

    def reset(self):
        self.cur_state=torch.ones(self.num_worker,dtype=torch.float)
        self.observer = torch.cat((torch.zeros((self.num_worker, self.horizon), dtype=torch.float), self.cur_state.unsqueeze(0).T),dim=1)[:, 1:]
        self.done=False
        return self.cur_state.clone(),self.observer.clone()

    def step(self,action):
        index=action % self.num_worker
        if (0<=action<self.num_worker) and (self.cur_state[index].item()>0):
            self.cur_state[index]-=1
            self.observer = torch.cat((self.observer, self.cur_state.unsqueeze(0).T),  dim=1)[:,1:]
        elif (self.num_worker<=action<2*self.num_worker) and (self.cur_state[index].item()<self.border):
            self.cur_state[index]+=1
            self.observer = torch.cat((self.observer, self.cur_state.unsqueeze(0).T), dim=1)[:, 1:]
        else:
            self.done=True
        return self.cur_state.clone(),self.observer.clone(),self.done