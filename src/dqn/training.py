import sys
sys.path.append('../')
#import dqn.myenv
from simulator.simulator import Price
from utils.types import PairType, GranularityType, ActionType, PositionType  # type: ignore
from myenv.fxenv import FxEnv, EnvParameter
from typing import NamedTuple, List
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from datetime import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        y = self.fc3(h)
        return y


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mock_forex_data: List[List[Price]] = []
date = datetime(2019, 2, 1, 12, 15, 30, 2000)
for i in range(100):
    price1D = Price(date=date, open=100.0+5 * random.random(), close=100.0+5 * random.random(), low=100.0+5 * random.random(), high=100.0+5 * random.random(), type=GranularityType.D)
    price4H = Price(date=date, open=100.0+5 * random.random(), close=100.0+5 * random.random(), low=100.0+5 * random.random(), high=100.0+5 * random.random(), type=GranularityType.H4)
    price1H = Price(date=date, open=100.0+5 * random.random(), close=100.0+5 * random.random(), low=100.0+5 * random.random(), high=100.0+5 * random.random(), type=GranularityType.H1)
    price15M = Price(date=date, open=100.0+5 * random.random(), close=100.0+5 * random.random(), low=100.0+5 * random.random(), high=100.0+5 * random.random(), type=GranularityType.M15)
    #print(price15M)
    day_prices: List[Price] = [price1D, price4H, price1H, price15M]
    mock_forex_data.append(day_prices)
    date = date + timedelta(days=1)

sim_param = EnvParameter(spread=0.005, window_size=30, max_episode_steps=30)
env = FxEnv(mock_forex_data, sim_param)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
WINDOW_SIZE = 30
HIDDEN_SIZE = 10

# Get number of actions from gym action space
n_actions = env.get_action_space()

policy_net = DQN(WINDOW_SIZE, HIDDEN_SIZE, n_actions).to(device)
target_net = DQN(WINDOW_SIZE, HIDDEN_SIZE, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def show_episode_result():
    history = env.get_history()
    print("------------------")
    print("StartDate", history[-1].start_date)
    print("Position", history[-1].position)
    print("Profit", history[-1].profit)
    print("------------------")


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


class Model():
    def __init__(self):
        pass 

    # データの前処理
    def preprocessing(self):
        return 0

    def get_action(self):
        lot = 0.01
        rand = random.random()
        if rand < 0.3:
            return ActionType.BUY, lot
        elif rand < 0.6:
            return ActionType.SELL, lot
        else:
            return ActionType.NONE, lot


num_episodes = 0
state = env.reset()
#for i_episode in range(num_episodes):
while env.is_finish() == False:
    print("episode: ", num_episodes)
    # Initialize the environment and state
    #state = env.observe()
    for t in count():
        print("step: ", t)
        # Select and perform an action
        action = select_action(state)
        model = Model()
        action, lot = model.get_action()
        next_state, reward, done, _ = env.step(action, 0.1)
        reward = torch.tensor([reward], device=device)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            show_episode_result()
            env.reset()
            break

    # Update the target network, copying all weights and biases in DQN
    if num_episodes % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print("isFinish: ", env.is_finish())
    num_episodes += 1

print('Complete')

env.render()
env.close()
plt.ioff()
plt.show()

