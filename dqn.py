import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import gym
import rl_utils
import matplotlib.pyplot as plt
from environment import *


# 定义神经网络模型
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# 初始化参数和超参数
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.001
gamma = 0.99
batch_size = 64
replay_buffer = ReplayBuffer(capacity=100000)
max_episode = 1500
learning_rate = 1e-2


# 初始化环境、模型和优化器
env = RIS_SISO()
input_size = env.state_dim
output_size = env.action_dim
q_network = QNetwork(input_size, output_size)
target_q_network = QNetwork(input_size, output_size)
target_q_network.load_state_dict(q_network.state_dict())
target_q_network.eval()
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
_LOSS = []
_rewards = []

# 训练循环
for episode in range(max_episode):
    print(episode)
    state = env.reset()
    total_reward = 0
    _loss = 0
    while True:
        # # epsilon-greedy策略选择动作
        if random.random() < epsilon:
            action = random.randint(0, 7)
        else:
            with torch.no_grad():
                q_values = q_network(torch.tensor(state, dtype=torch.float32))
                action = torch.argmax(q_values).item()



        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        # 将经验存入回放缓冲区
        replay_buffer.add((state, action, reward, next_state, done))

        state = next_state

        # 经验回放，更新Q网络
        if len(replay_buffer.buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            q_values = q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = target_q_network(next_states).max(1)[0].detach()
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values.squeeze(), target_q_values)
            _loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            _rewards.append(total_reward/20)
            _LOSS.append(_loss)
            break

    # 更新目标Q网络
    if episode % 4 == 0:
        target_q_network.load_state_dict(q_network.state_dict())

    # 衰减epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # print(f"Episode: {episode+1}, Reward: {total_reward}")

env.close()

# model_path = "trained_model.pth"
# torch.save(q_network.state_dict(), model_path)
# print("Model saved:", model_path)

episodes_list = list(range(len(_rewards)))
# _return = rl_utils.average(_rewards)
# filename = "partdqn.npy"
# np.save(filename, _return)
_loss = rl_utils.Exponential_Moving_Average(_LOSS,5)



window_size = 15  # The size of the smoothing window
smoothed_values = np.convolve(_rewards, np.ones(window_size) / window_size, mode='valid')
# np.save("pppDqn.npy", smoothed_values)

plt.plot(smoothed_values, color="Slateblue", alpha=0.6, linewidth=2)
    # plt.plot(episodes_list, _return1, label='d')
    # plt.plot(episodes_list, _return2, label='k')
    # # plt.plot(episodes_list, _return_1, label='w_d = 1,w_k = 0')
    # # plt.plot(episodes_list, _return_2, label='w_d = 0,w_k = 1')
    # # plt.fill_between(episodes_list, np.array(_return) - np.array(_rewards_std), np.array(_return) + np.array(_rewards_std), alpha=0.3, label='Reward Std')
    #
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Training Curve')
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()
# print(_LOSS)

# plt.plot(episodes_list, _loss, label='Average LOSS')
# plt.xlabel('Episode')
# plt.ylabel('loss')
# plt.title('loss Curve')
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.legend()
# plt.show()