import gym
import math
import random
from itertools import count
import torch
from eval_policy import eval_policy, device
from model import MyModel
from replay_buffer import ReplayBuffer
import numpy as np
import torch.nn as nn


BATCH_SIZE = 256
GAMMA = 0.99
EPS_EXPLORATION = 0.2
TARGET_UPDATE = 10
NUM_EPISODES = 4000
TEST_INTERVAL = 25
LEARNING_RATE = 10e-4
RENDER_INTERVAL = 20
ENV_NAME = 'CartPole-v1'
PRINT_INTERVAL = 1

env = gym.make(ENV_NAME)
state_shape = len(env.reset())
n_actions = env.action_space.n
model = MyModel(state_shape, n_actions).to(device)
target = MyModel(state_shape, n_actions).to(device)
target.load_state_dict(model.state_dict())
target.eval()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer()


def choose_action(state, test_mode=False):
    if not torch.is_tensor(state):
        state = torch.tensor([state], device=device, dtype=torch.float32)
    greedy_action = model.select_action(state)
    if np.random.rand() > EPS_EXPLORATION:
        action = greedy_action.to(device)
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    return action


def optimize_model(state, action, next_state, reward, done):
    y = torch.zeros([1, len(done)], dtype=torch.double).to(device)
    for idx in range(len(done)):
        if done[idx]:
            y[0][idx] = reward[idx]
        else:
            max_action = torch.argmax(target(next_state)[idx])
            y[0][idx] = reward[idx]+GAMMA*target(next_state)[idx][max_action]
    q_values_training = torch.zeros([1, len(done)], dtype=torch.double).to(device)
    for i in range(len(done)):
        action_i = action[i].item()
        q_all_values_i = model(state)[i]
        q_value_i = q_all_values_i[action_i]
        q_values_training[0][i] = q_value_i
    loss = nn.MSELoss()(q_values_training, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_reinforcement_learning(render=False):
    steps_done = 0
    best_score = -float("inf")

    for i_episode in range(1, NUM_EPISODES+1):
        episode_total_reward = 0
        state = env.reset()
        for t in count():
            action = choose_action(state)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0][0])
            steps_done += 1
            episode_total_reward += reward
            memory.push(state, action, next_state, reward, done)
            if len(memory) > BATCH_SIZE:
                state_1, action_1, next_state_1, reward_1, done_1 = memory.sample(batch_size=BATCH_SIZE)
                action_1 = torch.tensor(action_1, dtype=torch.long)
                optimize_model(state_1, action_1, next_state_1, reward_1, done_1)
            else:
                pass
            state = next_state
            if render:
                env.render(mode='human')
            if done:
                if i_episode % PRINT_INTERVAL == 0:
                    print('[Episode {:4d}/{}] [Steps {:4d}] [reward {:.1f}]'
                        .format(i_episode, NUM_EPISODES, t, episode_total_reward))
                break
        if i_episode % TARGET_UPDATE == 0:
            target.load_state_dict(model.state_dict())

        if i_episode % TEST_INTERVAL == 0:
            print('-'*10)
            score = eval_policy(policy=model, env=ENV_NAME, render=render)
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), "best_model_{}_2.pt".format(ENV_NAME))
                print('saving model.')
            print("[TEST Episode {}] [Average Reward {}]".format(i_episode, score))
            print('-'*10)


if __name__ == "__main__":
    train_reinforcement_learning()
