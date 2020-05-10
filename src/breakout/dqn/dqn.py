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
import os
from sklearn.model_selection import train_test_split
from gym import spaces
from gym.spaces.box import Box
from tqdm import tqdm
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import time
import yaml
from common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame, ScaledFloatFrame, ClipRewardEnv, FrameStack, WrapPyTorch
from common.util import get_optimizer

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)

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
    
    def record(self, filename, dir_path):
        '''モデルを保存する'''
        self.brain.save_model(filename, dir_path)
        
    def remember(self, name):
        '''モデルを読み込む'''
        self.brain.read_model(name)





#################################
#####        Examiner      ######
#################################

class EvalParameter(NamedTuple):
    save_path: str
    weights_name: str
    episode_num: int

class Examiner():
    def __init__(self, env, agent, param):
        self.env = env
        self.agent = agent
        self.param = param
        self.writer = SummaryWriter(log_dir="{}/logs".format(param.save_path))
        
    def evaluate(self):
        for episode_i in tqdm(range(self.param.episode_num)):
            state = self.env.reset()
            for t in count():
                time.sleep(0.05)
                self.env.render()
                ''' 行動を決定する '''
                action = self.agent.predict_action(state)

                ''' 行動に対する環境や報酬を取得する '''
                next_state, _, done, _ = self.env.step(action)  # state [0,0,0,0...window_size], reward 1.0, done False, input: action 0 or 1 or 2

                state = next_state
                if done:
                    ''' 終了時に結果をプロット '''
                    self.writer.add_scalar("step-eval", t+1, episode_i)
                    break


        ''' 終了 '''
        print('Complete')
        self.writer.close()


#################################
#####        Trainer       ######
#################################

class TrainParameter(NamedTuple):
    save_path: str
    save_name: str
    target_update_iter: int
    episode_num: int
    record_iter: int

class Trainer():
    def __init__(self, env, agent, param):
        self.env = env
        self.agent = agent
        self.param = param
        self.writer = SummaryWriter(log_dir="{}/logs".format(param.save_path))
        
    def train(self):
        for episode_i in tqdm(range(self.param.episode_num)):
            state = self.env.reset()
            score = 0

            for t in count():
                self.env.render()

                ''' 行動を決定する '''
                action = self.agent.select_action(state)

                ''' 行動に対する環境や報酬を取得する '''
                next_state, reward, done, info = self.env.step(action)  # state [0,0,0,0...window_size], reward 1.0, done False, input: action 0 or 1 or 2

                if reward > 0:
                    score += reward
                ''' 終了時はnext_state_valueをNoneとする '''

                if done:
                    next_state = None
                    self.agent.memorize(
                        torch.tensor(state, device=device), 
                        torch.tensor([[action]], device=device), 
                        next_state, 
                        torch.tensor([reward], device=device)
                    )

                else:
                    self.agent.memorize(
                        torch.tensor(state, device=device), 
                        torch.tensor([[action]], device=device), 
                        torch.tensor(next_state, device=device), 
                        torch.tensor([reward], device=device)
                    )

                # Move to the next state
                state = next_state

                ''' エージェントに学習させる '''
                loss = self.agent.learn()
                if loss != None:
                    self.writer.add_scalar("loss", loss, episode_i)

                if done:
                    ''' 終了時に結果をプロット '''
                    self.writer.add_scalar("step", t+1, episode_i)
                    self.writer.add_scalar("score", score, episode_i)
                    print("step: {}, score: {}, loss: {}".format(t+1, score, loss))
                    break

            # Update the target network, copying all weights and biases in DQN
            if episode_i % self.param.target_update_iter == 0:
                ''' 目標を修正する '''
                self.agent.modify_goal()

            if episode_i % self.param.record_iter == 0 and episode_i != 0:
                ''' 途中経過を保存 '''
                self.agent.record('{}_{}'.format(self.param.save_name, episode_i), '{}/weights'.format(self.param.save_path))

        ''' モデルを保存する '''
        # モデルの保存
        self.agent.record('{}_{}.pth'.format(self.param.save_name, self.param.episode_num), '{}/weights'.format(self.param.save_path))

        print('Complete')
        self.writer.close()
        

#################################
#####         Net          ######
#################################
class DQN(nn.Module):
    def __init__(self, output_size: int):
        super(DQN, self).__init__()
        self.feature = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(inplace=True))
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.output = nn.Linear(512, output_size)

    def forward(self, x):
        #print(x.shape)
        # torch.Size(B,C,H,W)
        x = self.feature(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc(x))
        x = self.output(x)
        return x

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
    optimizer_name: str
    learning_rate: float
    eps: float

class Brain:
    def __init__(self, param, num_actions):
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
        self.num_actions = num_actions # 行動の数を取得
        self.policy_net = DQN(self.num_actions).to(device)
        self.target_net = DQN(self.num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # 最適化手法の設定

        self.optimizer = get_optimizer(param.optimizer_name, self.policy_net.parameters(), lr=param.learning_rate, eps=param.eps)
        # self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=param., eps=1e-5)
        
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
        #print("statebatch", batch.state)
        state_batch = torch.cat(batch.state).reshape(32,4,84,84).float() # size(32, 1, 84, 84) バッチ数に次元を変更
        action_batch = torch.cat(batch.action) # action: tensor([[1],[0],[0]...]) size(32, 1) 
        reward_batch = torch.cat(batch.reward) # reward: tensor([1, 1, 1, 0, ...]) size(32)
        #print("state_batch: ", state_batch, state_batch.size())
        #print("action_batch: ", action_batch, action_batch.size())
        #print("reward_batch: ", reward_batch, reward_batch.size())
        #for i in range(len(state_batch[0])):
        #    print(state_batch[0][i].shape)
        #    var = input()
        #    pilOUT = Image.fromarray(np.uint8(state_batch[0][i]))
        #    pilOUT.show()


        ''' 出力データ：行動価値を作成 '''
        # 出力actionの値のうちaction_batchが選んだ方を抽出（.gather()）
        # action_batch = [[0], [1], [1]...] action_value = [[0.01, 0.03], [0.03, 0], [0, 0.02]...]
        # state_action_values = [[0.01], [0], [0.02]]
        #print("statebatch", state_batch.size())
        #print("actionbatch", action_batch)
        #print("stateactionbatch", self.policy_net(state_batch))
        ''' 出力action: [[0.54, 0.21, 0.11, 0.13], [...]], action_batchはindexを表す。gatherによってindexの場所を取得'''
        ''' state_action_values: [[0.54], []] '''
        state_action_values = self.policy_net(state_batch).gather(1, action_batch) # size(32, 1)

        #print("state_action_values2", self.policy_net(state_batch), self.policy_net(state_batch).size())
        #print("state_action_values", state_action_values, state_action_values.size())

        ''' 教師データを作成する '''
        ''' target = 次のステップでの行動価値の最大値 * 時間割引率 + 即時報酬 '''
         # doneされたかどうか doneであればfalse
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool) # torch.Size([32]) [True, True, False, ...]
        
        #print("non_final_mask: ", non_final_mask, non_final_mask.size())
        # 終了時じゃない場合のnextStatesを作成
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None]) # torch.Size(31, 84, 84)
        non_final_next_states = non_final_next_states.reshape( int(len(non_final_next_states)/4), 4, 84, 84).float() # torch.Size(31, 1, 84, 84)
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

        #print("state_action_values", state_action_values, state_action_values.size())
        ''' Loss を計算'''
        # Compute Huber loss
        # FIX: lossが以上に高い state_action_batchが高すぎるため,1回目の更新で大きく変えすぎ？
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        #print("loss", loss)
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
        state = torch.tensor(state, device=device).unsqueeze(0).float()
        #print("state", state.shape)
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
    
    
    def save_model(self, name, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), "{}/{}".format(dir_path, name))
        
    def read_model(self, name):
        param = torch.load(name, map_location='cpu')
        self.policy_net.load_state_dict(param)
    
    def predict(self, state):
        state = torch.tensor(state, device=device).unsqueeze(0).float()
        self.policy_net.eval() # ネットワークを推論モードに切り替える
        with torch.no_grad():
            action = np.argmax(self.policy_net(state).tolist())
        return action


#################################
#####      Environment     ######
#################################

def wrap_breakout_env(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

#################################
#####         Main         ######
#################################

def get_config():
    path = "./result"
    files = os.listdir(path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))] 
    for i, dir_name in enumerate(files_dir):
        print('{}: {}'.format(i, dir_name))
    print("Please select dirctory")
    select_dir = files_dir[int(input())]
    print('select {}'.format(select_dir))
    with open('./result/{}/config.yaml'.format(select_dir), 'r') as yml:
        config = yaml.load(yml)
    #print(config) 
    return config

if __name__ == "__main__":
    
    ''' config取得 '''
    config = get_config()
    print(config) 
    
    ''' 環境生成 '''
    env = gym.make(config["env_param"]["env_name"])
    print(env.spec.id)
    # env = make_atari('Breakout-v0')
    env = wrap_breakout_env(env, frame_stack=config["env_param"]["frame_stack"], scale=config["env_param"]["scale"])  
    env = WrapPyTorch(env) # output (1, 84, 84)
    
    ''' エージェント生成 '''
    init_screen = env.reset() #(1, 84, 84)
    _, screen_height, screen_width = init_screen.shape
    num_actions = env.action_space.n
    brain_param = BrainParameter(
        batch_size=config["brain_param"]["batch_size"], 
        gamma=config["brain_param"]["gamma"],
        eps_start=config["brain_param"]["eps_start"],
        eps_end=config["brain_param"]["eps_end"],
        eps_decay=config["brain_param"]["eps_decay"],
        capacity=config["brain_param"]["capacity"],
        hidden_size=config["brain_param"]["hidden_size"],
        optimizer_name=config["brain_param"]["optimizer_name"],
        learning_rate=config["brain_param"]["learning_rate"],
        eps=config["brain_param"]["eps"]
    )
    brain = Brain(brain_param, num_actions)
    agent = Agent(brain)


    action = config["type"]
    if action == "train": # Train
        print("start training...")
        ''' Trainer '''
        train_param = TrainParameter(
            save_path=config["train_param"]["save_path"], 
            episode_num=config["train_param"]["episode_num"],
            record_iter=config["train_param"]["record_iter"],
            target_update_iter=config["train_param"]["target_update_iter"],
            save_name=config["train_param"]["save_name"]
        )
        breakout_trainer = Trainer(env, agent, train_param)
        breakout_trainer.train()

    elif action == "eval": # Evaluete
        print("start evaluating...")
        ''' Examiner '''
        eval_param = EvalParameter(
            save_path=config["eval_param"]["save_path"], 
            weights_name=config["eval_param"]["weights_name"],
            episode_num=config["eval_param"]["episode_num"]
        )
        agent.remember("{}/weights/{}".format(eval_param.save_path, eval_param.weights_name))
        breakout_examiner = Examiner(env, agent, eval_param)
        breakout_examiner.evaluate()

    else:
        print("unknown number")