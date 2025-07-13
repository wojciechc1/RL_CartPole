import torch
import torch.nn.functional as F
import numpy as np
from model.model import DQN, Q_Net
import random

class Agent:
    def __init__(self):
        #self.model = DQN(4, 2)  # CartPole: 4 obserwacje, 2 akcje
        self.model = Q_Net(4, 256, 2)  # CartPole: 4 obserwacje, 2 akcje
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0


    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(obs_tensor)
            return torch.argmax(q_values).item()

    def store_transition(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def learn(self):
        if len(self.memory) < 32:
            return

        batch = random.sample(self.memory, 32)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(obs).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.model(next_obs).max(1)[0]
        target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def decent_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.991)
        #print('[AGENT] decent epsilon: ', self.epsilon)