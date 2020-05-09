import sys
sys.path.append('../../')
from utils.types import PairType, GranularityType, ActionType, PositionType  # type: ignore
from simulator.simulator import Price, Transaction, SimParameter  # type: ignore
import gym
import numpy as np
import gym.spaces
from typing import NamedTuple, List


##################################################
# FXの強化学習用環境
# Action: Sell Buy None
# 一度に一つの取引のみを行う（同時に複数の取引はしない）
# 一回の取引につき>0であれば報酬+1, <0であれば報酬-1
##################################################

class EnvParameter(NamedTuple):
    spread: int     # スプレッド
    window_size: int    # 使用する期間
    max_episode_steps: int # 1取引における最大期間

class FxEnv():

    def __init__(self, forex_data: List[List[Price]], sim_param: SimParameter):
        # action_space, observation_space, reward_range を設定する
        self.action_space = 3  # Buy, Sell, None
        self.reward_range = [-1., 1.]
        self.forex_data = forex_data
        self.param = sim_param
        self.history: List[Transaction] = []      # トレード履歴
        self.steps = self.param.window_size
        self.reset()

    def reset(self):
        self.done = False
        self.transaction = None
        return self.get_observe()

    def step(self, action: ActionType, lot: float):
        step_data = self.forex_data[self.steps]
        date = step_data[-1].date
        # 新規取引
        if self.transaction == None and (action == ActionType.SELL or action == ActionType.BUY):
            position = self.get_build_position(action)
            self.transaction = Transaction(start_date=date, pair=PairType.USD_JPY, lot=lot, position=position, entry_rate=step_data[-1].close)

        # 決済した場合、statsにデータを送る
        elif self.transaction != None and self.is_release(self.transaction.position, action):
            # 取引内容を追記
            self.transaction.settle(end_date=date, settle_rate=step_data[-1].close)
            self.history.append(self.transaction)
            # エピソード終了
            self.done = True
        
        reward = self._get_reward()

        self.steps += 1
        next_observation = self.get_observe()

        return next_observation, reward, self.done, {}


        # 利益確定したかどうか
    def is_release(self, position: PositionType, action: ActionType) -> bool:
        if position == PositionType.LONG and action == ActionType.SELL or position == PositionType.SHORT and action == ActionType.BUY:
            return True

        return False

    # ポジションを建てたかどうか
    def get_build_position(self, action: ActionType) -> PositionType:
        if action == ActionType.SELL:
            return PositionType.SHORT
        elif action == ActionType.BUY:
            return PositionType.LONG
        else:
            return PositionType.NONE
    
    def get_history(self) -> List[Transaction]:
        return self.history

    def render(self):
        print("---------------------------------------------")
        print("Transaction: ", self.transaction)
        print("---------------------------------------------")

    def close(self):
        pass

    def _seed(self, seed=None):
        pass

    def _get_reward(self) -> int:
        if self.done:
            if self.transaction.profit > 0:
                return 1
            elif self.transaction.profit < 0:
                return -1
        else:
            return 0


    def get_action_space(self):
        return self.action_space

    def get_observe(self):
        return self.forex_data[self.steps - self.param.window_size: self.steps]

    def is_finish(self) -> bool:
        if self.steps + self.param.max_episode_steps > len(self.forex_data):
            return True
        else:
            return False

