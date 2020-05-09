import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from gym import wrappers
from datetime import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from enum import Enum
from typing import NamedTuple, List
import pandas as pd
import os
import uuid
from sklearn.model_selection import train_test_split
from gym import spaces
from gym.spaces.box import Box


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#################################
#####     Replay Memory    ######
#################################
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward):
        """Saves a transition."""
        #print("args", state, action, next_state, reward)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


#################################
#####        Agent         ######
#################################
class Agent:
    def __init__(self, brain):
        '''エージェントが行動を決定するための頭脳を生成'''
        self.brain = brain
        
    def learn(self):
        '''Q関数を更新する'''
        loss = self.brain.optimize()
        return loss
        
    def modify_goal(self):
        '''Target Networkを更新する'''
        self.brain.update_target_model()
        
    def select_action(self, state):
        '''行動を決定する'''
        action = self.brain.decide_action(state)
        return action
    
    def memorize(self, state, action, next_state, reward):
        '''memoryオブジェクトに、state, action, state_next, rewardの内容を保存する'''
        self.brain.memory.push(state, action, next_state, reward)
    
    def predict_action(self, state):
        '''行動を予測する'''
        action = self.brain.predict(state)
        return action
    
    def record(self, name):
        '''モデルを保存する'''
        self.brain.save_model(name)
        
    def remember(self, name):
        '''モデルを読み込む'''
        self.brain.read_model(name)

#################################
#####      Environment     ######
#################################
import cv2
cv2.ocl.setUseOpenCL(False)
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        '''工夫1のNo-Operationです。リセット後適当なステップの間何もしないようにし、
        ゲーム開始の初期状態を様々にすることｆで、特定の開始状態のみで学習するのを防ぐ'''
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        '''工夫2のEpisodic Lifeです。1機失敗したときにリセットし、失敗時の状態から次を始める'''
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        '''5機とも失敗したら、本当にリセット'''
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        '''工夫3のMax and Skipです。4フレーム連続で同じ行動を実施し、最後の3、4フレームの最大値をとった画像をobsにする'''
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        '''工夫4のWarp frameです。画像サイズをNatureのDQN論文と同じ84x84の白黒にします'''
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        '''PyTorchのミニバッチのインデックス順に変更するラッパー'''
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

class EnvParameter(NamedTuple):
    max_lot: int    # 最大数量
    spread: int     # スプレッド
    window_size: int
        
class Environment():
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        #self.env = NoopResetEnv(self.env, noop_max=30)
        #self.env = MaxAndSkipEnv(self.env, skip=4)
        #self.env = EpisodicLifeEnv(self.env)
        #self.env = WarpFrame(self.env)
        #self.env = WrapPyTorch(self.env)
        
    def reset(self):
        return self.env.reset()
        
    def step(self, action): # action: 0 is None, 1 is Buy, 2 is Sell
        return self.env.step(action)

    def get_action_num(self):
        return self.env.action_space.n
        
    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]

        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0).to(device)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
    

#################################
#####        Trainer       ######
#################################

class Trainer():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.loss_durations = []
        self.episode_durations = []
        self.TARGET_UPDATE = 10
        self.episode = 0
        
    def train(self, episode_num, save_name):
        for episode_i in range(episode_num):
            print("episode: ", episode_i)
            env.reset()
            last_screen = self.env.get_screen()
            current_screen = self.env.get_screen()
            state = current_screen - last_screen
            for t in count():
                env.render()
                ''' 行動を決定する '''
                action = self.agent.select_action(state) # input ex: <list> [0, 0, 0, 0], output ex: <int> 0 or 1

                ''' 行動に対する環境や報酬を取得する '''
                _, reward, done, _ = self.env.step(action)  # state [0,0,0,0...window_size], reward 1.0, done False, input: action 0 or 1 or 2

                ''' 終了時はnext_state_valueをNoneとする '''
                # Observe new state
                last_screen = current_screen
                current_screen = self.env.get_screen()
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                ''' エージェントに記憶させる '''
                # Store the transition in memory
                #print("state: ", state)
                #print("action: ", action)
                #print("next_state: ", next_state)
                #print("reward: ", reward)
                # 全てtesor型にして保存
                self.agent.memorize(
                    state, 
                    torch.tensor([[action]], device=device), 
                    next_state, 
                    torch.tensor([reward], device=device)
                )

                # Move to the next state
                state = next_state

                ''' エージェントに学習させる '''
                # Perform one step of the optimization (on the target network)
                # update q network
                loss = self.agent.learn()
                if loss != None:
                    self.loss_durations.append(loss)

                if done:
                    ''' 終了時に結果をプロット '''
                    print("finish episode")
                    print("step: ", t)
                    self.episode_durations.append(t + 1)
                    #self.plot_durations()
                    #self.episode += 1
                    break
            # Update the target network, copying all weights and biases in DQN
            if self.episode % self.TARGET_UPDATE == 0:
                ''' 目標を修正する '''
                self.agent.modify_goal()

        ''' モデルを保存する '''
        # モデルの保存
        self.agent.record(save_name)
        print('Complete')
        self.plot_durations()
        
        
    def plot_durations(self):
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        x = [s for s in range(len(self.loss_durations))]
        y = self.loss_durations
        x2 = [s for s in range(len(self.episode_durations))]
        y2 = self.episode_durations
        ax.plot(x, y, color="red", label="loss")
        ax2.plot(x2, y2, color="blue", label="episode")
        ax.legend(loc = 'upper right') #凡例
        ax2.legend(loc = 'upper right') #凡例
        fig.tight_layout()              #レイアウトの設定
        plt.show()

#################################
#####        Examiner      ######
#################################

class Examiner():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.episode = 0
        
    def evaluate(self, file_name):
        print('Complete')

#################################
#####         Net          ######
#################################
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

#################################
#####         Brain        ######
#################################

class BrainParameter(NamedTuple):
    batch_size: int
    gamma : float
    eps_start : float
    eps_end: float
    eps_decay: int
    capacity: int
    hidden_size: int

class Brain:
    def __init__(self, param, screen_height, screen_width, num_actions):
        self.steps_done = 0
        
        # Brain Parameter
        self.BATCH_SIZE = param.batch_size
        self.GAMMA = param.gamma
        self.EPS_START = param.eps_start
        self.EPS_END = param.eps_end
        self.EPS_DECAY = param.eps_decay
        self.CAPACITY = param.capacity
        self.HIDDEN_SIZE = param.hidden_size
        
        # 経験を保存するメモリオブジェクトを生成
        self.memory = ReplayMemory(self.CAPACITY)
        
        #print(self.model) # ネットワークの形を出力
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.num_actions = num_actions
        #print(self.num_observ)
        self.num_actions = num_actions # 行動の数を取得
        self.policy_net = DQN(self.screen_height, self.screen_width, self.num_actions).to(device)
        self.target_net = DQN(self.screen_height, self.screen_width, self.num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # 最適化手法の設定
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        
    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        # 訓練モード
        self.policy_net.train()
        
        ''' batch化する '''
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        #print("reward_action: ", batch.action)
        #print("reward_batch: ", batch.reward)
        state_batch = torch.cat(batch.state) # state: tensor([[0.5, 0.4, 0.5, 0], ...]) size(32, 4)
        action_batch = torch.cat(batch.action) # action: tensor([[1],[0],[0]...]) size(32, 1) 
        reward_batch = torch.cat(batch.reward) # reward: tensor([1, 1, 1, 0, ...]) size(32)
        #print("state_batch: ", state_batch, state_batch.size())
        #print("action_batch: ", action_batch, action_batch.size())
        #print("reward_batch: ", reward_batch, reward_batch.size())


        ''' 出力データ：行動価値を作成 '''
        # 出力actionの値のうちaction_batchが選んだ方を抽出（.gather()）
        # action_batch = [[0], [1], [1]...] action_value = [[0.01, 0.03], [0.03, 0], [0, 0.02]...]
        # state_action_values = [[0.01], [0], [0.02]]
        #print("screen: ", self.screen_width, self.screen_height)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch) # size(32, 1)
        #print("state_action_values2", self.policy_net(state_batch), self.policy_net(state_batch).size())
        #print("state_action_values", state_action_values, state_action_values.size())

        ''' 教師データを作成する '''
        ''' target = 次のステップでの行動価値の最大値 * 時間割引率 + 即時報酬 '''
         # doneされたかどうか doneであればfalse
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        #print("non_final_mask: ", non_final_mask, non_final_mask.size())
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        #print("non_final_next_state: ", non_final_next_states, non_final_next_states.size())
        
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        
        # 大きい方を選択して一次元にする
        # done時は0
        # target_net: [[0, 0.1], [2, 0.2]...], size(32, 2)      next_state_values: [0.1, 2...], size(32)
        # 次の環境での行動価値
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach() # size(32)

        # target = 次のステップでの行動価値の最大値 * 時間割引率 + 即時報酬
        #print("next_state_values: ", next_state_values, next_state_values.size())
        #print("reward_batch: ", reward_batch, reward_batch.size())
        #print("expected: ", (next_state_values * self.GAMMA) + reward_batch, ((next_state_values * self.GAMMA) + reward_batch).size())
        expected_state_action_values = ((next_state_values * self.GAMMA) + reward_batch).unsqueeze(1) # size(32, 1)
        #print("expected_state_value: ", expected_state_action_values, expected_state_action_values.size())

        ''' Loss を計算'''
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        ''' 勾配計算、更新 '''
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss
    
    def update_target_model(self):
        # モデルの重みをtarget_networkにコピー
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decide_action(self, state):
        state = torch.tensor(state, device=device).float()
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                action = np.argmax(self.policy_net(state).tolist())
                return action
        else:
            return random.randrange(self.num_actions)
    
    
    def save_model(self, name):
        torch.save(self.policy_net.state_dict(), name)
        
    def read_model(self, name):
        param = torch.load(name)
        self.policy_net.load_state_dict(param)
    
    def predict(self, state):
        state = torch.tensor(state, device=device).float()
        self.policy_net.eval() # ネットワークを推論モードに切り替える
        with torch.no_grad():
            action = np.argmax(self.policy_net(state).tolist())
        return action

#################################
#####         Main         ######
#################################

if __name__ == "__main__":
    
    ''' 環境生成 '''
    env = Environment('Breakout-v0')
    
    ''' エージェント生成 '''
    init_screen = env.get_screen()
    _, _, screen_height, screen_width = init_screen.shape
    num_actions = env.get_action_num()
    brain_param = BrainParameter(batch_size=32, gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=200, capacity=10000, hidden_size=100)
    brain = Brain(brain_param, screen_height, screen_width, num_actions)
    agent = Agent(brain)
    
    ''' Trainer '''
    breakout_trainer = Trainer(env, agent)
    breakout_trainer.train(600, 'dqn_breakout_600.pth')