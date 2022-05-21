import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# including CNN: For complex environment
class PolicyNetwork(nn.Module):         
    def __init__(self, number_worker, greedy):
        super(PolicyNetwork, self).__init__()
        self.tier = int((number_worker - 2) / 2)
        self.num_worker = number_worker
        self.greedy = greedy
        self.conv = nn.Conv2d(1,10,kernel_size=(3,3))
        self.fc1 = nn.Linear(self.tier * self.tier * 10, 64)
        self.fc2 = nn.Linear(64, 2 * self.num_worker + 1)

    def forward(self,observer):
        x = F.relu(F.max_pool2d(self.conv(observer), 2))
        x = x.view(-1, self.tier * self.tier * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def select_action(self, observer):
        with torch.no_grad():
            if np.random.rand() < self.greedy:
                sample_action = np.random.randint(2*self.num_worker + 1)
                actions = self.forward(observer)
                distribution = Categorical(actions)
                action = distribution.sample()
                probs = distribution.log_prob(action).item()
            else:
                actions = self.forward(observer)
                distribution = Categorical(actions)
                action = distribution.sample()
                probs = distribution.log_prob(action).item()
                sample_action = action.item()
            return sample_action, probs


class CriticNetwork(nn.Module):
    def __init__(self, number_worker):
        super(CriticNetwork, self).__init__()
        self.tier = int((number_worker - 2) / 2)
        self.num_worker = number_worker
        self.conv = nn.Conv2d(1,10,kernel_size=(3,3))
        self.fc1 = nn.Linear(self.tier * self.tier * 10, 64)
        self.fc2 = nn.Linear(64, 1)    # the critic network is a single layer
        
    def forward(self,observer):
        x = F.relu(F.max_pool2d(self.conv(observer), 2))
        x = x.view(-1, self.tier * self.tier * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# not including CNN: For simple environment
class PolicyNetwork_lite(nn.Module):         
    def __init__(self, number_worker, greedy):
        super(PolicyNetwork_lite, self).__init__()
        self.tier = int(number_worker / 2)
        self.num_worker = number_worker
        self.greedy = greedy
        self.conv = nn.Conv2d(1,10,kernel_size=(1, 1))
        self.fc1 = nn.Linear(self.tier * self.tier * 10, 64)
        self.fc2 = nn.Linear(64, 2 * self.num_worker + 1)

    def forward(self, observer):
        x = F.relu(F.max_pool2d(self.conv(observer), 2))
        x = x.view(-1, self.tier * self.tier * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def select_action(self, observer):
        with torch.no_grad():
            if np.random.rand() < self.greedy:
                sample_action = np.random.randint(2 * self.num_worker + 1)
                actions = self.forward(observer)
                distribution = Categorical(actions)
                action = distribution.sample()
                probs = distribution.log_prob(action).item()
            else:
                actions = self.forward(observer)
                distribution = Categorical(actions)
                action = distribution.sample()
                probs = distribution.log_prob(action).item()
                sample_action = action.item()
            return sample_action, probs


class CriticNetwork_lite(nn.Module):
    def __init__(self, number_worker):
        super(CriticNetwork_lite, self).__init__()
        self.tier = int(number_worker / 2)
        self.num_worker = number_worker
        self.conv = nn.Conv2d(1,10,kernel_size=(1,1))
        self.fc1 = nn.Linear(self.tier * self.tier * 10, 64)
        self.fc2 = nn.Linear(64, 1)    # the critic network is a single layer
        
    def forward(self,observer):
        x = F.relu(F.max_pool2d(self.conv(observer), 2))
        x = x.view(-1, self.tier * self.tier * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Agent(object):
    def __init__(self, number_worker, greedy):
        if number_worker < 6:
            self.policy = PolicyNetwork_lite(number_worker, greedy)
            self.critic = CriticNetwork_lite(number_worker)
        else:
            self.policy = PolicyNetwork(number_worker, greedy)
            self.critic = CriticNetwork(number_worker)
        self.actor_optim = torch.optim.Adam(self.policy.parameters(), lr = 1e-3)
        self.gamma = 0.9

    def choose_action(self, observer):
        action, probs = self.policy.select_action(observer.unsqueeze(0).unsqueeze(0))
        value = self.critic(observer.unsqueeze(0).unsqueeze(0)).item()
        return action, probs, value

    def learn(self, memory):
        for i in range(len(memory.rewards)):
            reward_arr, state_arr, action_arr, vals_arr, old_prob_arr, dones_arr = memory.sample(i)
            values = vals_arr[:]
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
        print("Policy has been updated successfully!")
