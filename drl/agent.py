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
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr= 1e-3)
        self.gamma = 0.9
        self.gae_lambda = 0.9
        self.policy_clip = 0.2
        self.loss = 0

    def choose_action(self, observer):
        action, probs = self.policy.select_action(observer.unsqueeze(0).unsqueeze(0))
        value = self.critic(observer.unsqueeze(0).unsqueeze(0)).item()
        return action, probs, value

    def learn(self, memory):
        ### compute one trajectory ###
        for i in range(len(memory.rewards)):
            reward_arr, state_arr, action_arr, vals_arr, old_prob_arr, dones_arr, indi = memory.sample(i)
            reward_G_arr = torch.zeros(len(dones_arr))
            num_T = 0
            
            # make reward[a, b, c] to reward_G[0, ..., a, 0, ..., b, 0, ..., c]
            for index, flag in enumerate(dones_arr):
                reward_G_arr[index] = 0
                if flag == True: 
                    reward_G_arr[index] = float(reward_arr[num_T])
                    num_T += 1
            
            ### compute advantage ###
            values = vals_arr[:]
            advantage = np.zeros(len(reward_G_arr), dtype=np.float32)
            for t in range(len(reward_G_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_G_arr)-1):
                    a_t += discount*(reward_G_arr[k] + self.gamma * values[k+1] - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage[len(reward_G_arr)-1] = reward_G_arr[len(reward_G_arr)-1] - values[len(reward_G_arr)-1]
            advantage = torch.tensor(advantage)
            
            ### SGD update ###
            values = torch.tensor(values)
            states = torch.tensor(state_arr[indi], dtype=torch.float).unsqueeze(1)
            old_probs = torch.tensor(old_prob_arr[indi])
            actions = torch.tensor(action_arr[indi])
            dist = Categorical(self.policy(states))
            critic_value = self.critic(states)
            critic_value = torch.squeeze(critic_value)
            new_probs = dist.log_prob(actions)
            prob_ratio = new_probs.exp() / old_probs.exp()
            weighted_probs = advantage[indi] * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, \
                        1+self.policy_clip) * advantage[indi]
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
            returns = advantage[indi] + values[indi]
            critic_loss = (returns-critic_value)**2
            critic_loss = critic_loss.mean()
            total_loss = actor_loss + 0.5 * critic_loss
            self.loss  = total_loss
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            total_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()
            
        print("Policy has been updated successfully!")
