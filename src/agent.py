import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from durak_env import DurakEnv

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).squeeze(0)
                
                filtered_q_values = torch.full((self.action_dim,), float('-inf'))
                filtered_q_values[available_actions] = q_values[available_actions]
                
                return torch.argmax(filtered_q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        q_values = self.policy_net(state).gather(1, action)
        next_q_values = self.target_net(next_state).max(1)[0].detach()
        target = reward + (1 - done) * self.gamma * next_q_values

        loss = self.loss_fn(q_values.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = DurakEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent_0 = DQNAgent(state_dim, action_dim)
agent_1 = DQNAgent(state_dim, action_dim)

episodes = 700
win_rate = 0

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = [0, 0]

    while not done:
        available_actions = env.get_available_actions()
        turn_to_action = env.get_turn_to_action(state)

        if turn_to_action == 0:
            action = agent_0.select_action(state, available_actions)
        else:
            action = agent_1.select_action(state, available_actions)

        next_state, reward, done, info = env.step(action)
        
        if turn_to_action == 0:
            agent_0.store_transition(state, action, reward, next_state, done)
            total_reward[0] += reward
            agent_0.train()
        else:
            agent_1.store_transition(state, action, reward, next_state, done)
            total_reward[1] += reward
            # agent_1.train()


        state = next_state

    agent_0.update_target_network()
    # agent_1.update_target_network()

    # if (episode + 1) % 15 == 0:
    print(f"Episode {episode + 1}: Total Reward Agent 1: {total_reward[0]}, Total Reward Agent 2: {total_reward[1]}")
    if total_reward[1] > 100:
        win_rate += 1

print(f"Win rate: {win_rate / episodes}")
