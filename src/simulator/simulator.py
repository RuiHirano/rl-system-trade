import sys
sys.path.append('../')
from utils.types import PairType, GranularityType, ActionType, PositionType  # type: ignore
from utils.util import import_csv_data  # type: ignore
from typing import NamedTuple, List
from datetime import *
import uuid
import pandas as pd  # type: ignore
import numpy as np  # type: ignore


class Parameter(NamedTuple):
    initial_assets: int     # 初期投資額
    max_lot: int    # 最大数量
    spread: int     # スプレッド

class Price(NamedTuple):
    type: GranularityType
    date: datetime
    open: float
    close: float
    high: float
    low: float


class Transaction():
    def __init__(self, start_date: datetime, pair: PairType, lot: float, position: PositionType, entry_rate: float):
        self.id = str(uuid.uuid4())
        self.start_date = start_date
        self.end_date = start_date
        self.pair = pair
        self.lot = lot
        self.position = position
        self.entry_rate = entry_rate
        self.losscut_rate = self.calc_losscut_line()
        self.settle_rate = entry_rate
        self.profit = 0

    def settle(self, end_date, settle_rate):
        self.end_date = end_date
        self.settle_rate = settle_rate
        self.profit = self.calc_profit()
    
    # 利益計算
    def calc_profit(self) -> int:
        return 0

    # losscutlineを計算
    def calc_losscut_line(self) -> float:
        return 0

    # losscutになるかどうか
    def is_losscut(self, rate: float) -> bool:
        if self.position == PositionType.LONG and rate <= self.losscut_rate or self.position == PositionType.SHORT and rate >= self.losscut_rate:
            return True
        return False

class Model():
    def __init__(self, window_size: int, allowable_loss: int):
        self.window_size = window_size
        self.allowable_loss = allowable_loss

    # データの前処理
    def preprocessing(self):
        return 0

    def get_action(self):
        lot = 0.01
        return ActionType.Buy, lot

class StatsCalculator():
    def __init__(self, initial_assets):
        self.initial_assets = initial_assets    # 初期投資額
        self.transaction_list: List[Transaction] = []    # 取引内容
        self.total_trade_num = 0    # 総取引数
        self.win_trade_num = 0  # 勝ち回数
        self.lose_trade_num = 0 # 負け回数
        self.buy_num = 0    # 買いの回数
        self.sell_num = 0 # 売りの回数
        self.win_rate = 0 # 勝率
        self.total_profit = 0   # 総利益
        self.total_loss = 0 # 総損失
        self.profit_average = 0 # 平均利益
        self.loss_average = 0   # 平均損失
        self.profit_loss_rate = 0 # 損益率
        self.earn_power = 0 # 稼力：1より大きければ必ず資産は増えていく

    # 取引内容を追加
    def add_transaction(self, transaction: Transaction):
        self.transaction_list.append(transaction)

    def calc_stats(self):
        return 0


class Simulator():
    def __init__(self, forex_data: List[Price], model: Model, param: Parameter):
        self.forex_data = forex_data
        self.model = model
        self.param = param
        self.stats = StatsCalculator(param.initial_assets)
        self.transactions = []      # 取引中のトレード

    # シミュレーション開始
    def run(self):
        # 1日毎にループ
        for i, day_data in enumerate(self.forex_data):
            date = day_data.date
            # window_size分待機
            if i <= self.model.window_size:
                continue

            # window_size分のデータを取得
            window_data = self.forex_data[i-self.model.window_size: i]

            new_transactions = []
            # 保持中の取引に関して
            for transaction in self.transactions:

                action, _ = self.model.get_action(window_data, transaction.position)

                # 決済した場合、statsにデータを送る
                if self.is_release(transaction.position, action):
                    # 取引内容を追記
                    transaction.settle(end_date=date, settle_rate=day_data.close)
                    # statsに追加
                    self.stats.add_transaction(transaction)

                else:
                    # 決済してない場合、追加
                    new_transactions.append(transaction)


            # 新規で行う取引に関して
            action, lot = self.model.get_action(window_data, PositionType.NONE)

            # ポジションを建てた場合
            position = self.is_build(action)
            if position != PositionType.NONE:
                # 取引内容を作成し、transactionsに追加
                transaction = Transaction(start_date=date, pair=day_data.pair, lot=lot, position=position, entry_rate=day_data.close)
                new_transactions.append(transaction)
            
            # 保持リストを更新
            self.transactions = new_transactions



    # 利益確定したかどうか
    def is_release(self, position: PositionType, action: ActionType) -> bool:
        if position == PositionType.LONG and action == ActionType.SELL or position == PositionType.SHORT and action == ActionType.BUY:
            return True
        
        return False
    
    # ポジションを建てたかどうか
    def is_build(self, action: ActionType) -> PositionType:
        if action == ActionType.SELL:
            return PositionType.SHORT

        elif action == ActionType.BUY:
            return PositionType.LONG

        return PositionType.NONE

    # 結果取得
    def get_stats(self):
        return self.stats.calc_stats()


if __name__ == "__main__":
    # 為替データの取得
    d_forex_df = import_csv_data(PairType.USD_JPY, GranularityType.D)
    d_forex_df["granularity"] = "D"
    h4_forex_df = import_csv_data(PairType.USD_JPY, GranularityType.H4)
    h4_forex_df["granularity"] = "H4"
    h1_forex_df = import_csv_data(PairType.USD_JPY, GranularityType.H1)
    h1_forex_df["granularity"] = "H1"
    #d_forex_data = import_csv_data(PairType.USD_JPY, GranularityType.M15)

    # 一番多いやつにindexを合わせる(少ないやつにはNanが作成される)
    #date_df = h1_forex_df["time"]
    #print(date_df)
    h1_forex_df = h1_forex_df.set_index(["time"])   # indexをtimeに変更
    h4_forex_df = h4_forex_df.set_index(["time"])   # indexをtimeに変更
    h4_forex_df = h1_forex_df.merge(h4_forex_df, how="outer",right_on="time")
    print(h4_forex_df)
    print("finish")

    # asfreqを用いて前方補完する

    # データフレームを縦に結合
    df = pd.concat([d_forex_df, h4_forex_df, h1_forex_df], ignore_index=True)
    df = df.set_index(["time", "granularity"])   # indexをtimeに変更
    #df.index = pd.to_datetime(df.index) # datetime化
    df = df.sort_index(ascending=True)  # dateでソート
    
    #df.set_index(["time", "granularity"])
    #for date, new_df in df.head(30).groupby(level=0):
    #    print(date)
    #    #print(new_df[new_df.index.get_level_values("time")=="H4"])
    #    print(new_df.loc[("high")])
    
    #print(df.head(30))
    #print(datetime(2005, 7, 1, 5, 00, 00, 0000))
    #print(df.loc[(datetime(2005, 7, 1, 5, 00, 00, 0000), "H1"), :])


'''
    # パラメータ指定
    param = Parameter(initial_assets=500000, max_lot=10, spread=0.005)

    # モデル選択
    model = Model(window_size=30, allowable_loss=1)

    # Simulator
    simulator = Simulator(forex_data, model, param)
    simulator.run()

    simulator.get_stats()
'''

