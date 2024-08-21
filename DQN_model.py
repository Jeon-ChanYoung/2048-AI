import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

            

class Agent:
    def __init__(self, env):
        self.env = env
        self.model = DQNetwork(env.observation_space, env.action_space)
        self.target_model = DQNetwork(env.observation_space, env.action_space)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = deque(maxlen=100000)
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.min_epsilon = 0.01

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        state = torch.tensor([state], dtype=torch.float)
        q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        q_values = self.model(states)
        next_q_values = self.target_model(next_states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)

        loss = self.loss_fn(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def save_model(self, filename: str):
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename: str):
        self.model.load_state_dict(torch.load(filename))
        print(f"Model loaded from {filename}")