import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
    
    import torch
import random
import numpy as np

def train_step(self):
    if len(self.memory) < self.batch_size:
        return

    batch = random.sample(self.memory, self.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to tensors
    states = torch.FloatTensor(states).to(self.device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
    rewards = torch.FloatTensor(rewards).to(self.device)
    next_states = torch.FloatTensor(next_states).to(self.device)
    dones = torch.FloatTensor(dones).to(self.device)

    # Q(s, a)
    q_values = self.q_network(states).gather(1, actions).squeeze()

    # max_a' Q_target(s', a')
    with torch.no_grad():
        max_next_q = self.target_network(next_states).max(1)[0]

    # Target: r + γ * max_a' Q(s', a')
    target_q = rewards + (1 - dones) * self.gamma * max_next_q

    loss = self.criteria(q_values, target_q)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()