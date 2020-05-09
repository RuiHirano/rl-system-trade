import sys
sys.path.append('../')
from utils.types import PairType, GranularityType, ActionType, PositionType  # type: ignore
from utils.util import import_csv_data  # type: ignore
from typing import NamedTuple, List
from datetime import *
import uuid
import pandas as pd  # type: ignore
import numpy as np  # type: ignore




class Price(NamedTuple):
    type: GranularityType
    date: datetime
    open: float
    close: float
    high: float
    low: float

class Transaction2(NamedTuple):
    id: str
    date: datetime
    pair: PairType
    lot: float
    position: PositionType
    rate: float


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
    def __init__(self):
        pass 

    # データの前処理
    def preprocessing(self):
        return 0

    def get_action(self, data):
        lot = 0.01
        return ActionType.BUY, lot


class StatsParameter(NamedTuple):
    allowable_loss: float
    initial_assets: int     # 初期投資額

class StatsData(NamedTuple):
    total_trade_num: int    # 総取引数
    win_trade_num: int    # 勝ち回数
    lose_trade_num: int    # 負け回数
    buy_num: int    # 買いの回数
    sell_num: int    # 売りの回数
    win_rate: float    # 勝率
    total_profit: int    # 総利益
    total_loss: int    # 総損失
    profit_average: float    # 平均利益
    loss_average: float    # 平均損失
    profit_loss_rate: float    # 損益率
    earn_power: float    # 稼力：1より大きければ必ず資産は増えていく


class StatsCalculator():
    def __init__(self, history: List[Transaction2], params: StatsParameter):
        self.params = params
        self.history = history
        self.stats = StatsData(
            total_trade_num=0, 
            win_trade_num=0, 
            lose_trade_num=0, 
            buy_num=0, 
            sell_num=0,
            win_rate=0,
            total_profit=0,
            total_loss=0,
            profit_average=0,
            loss_average=0,
            profit_loss_rate=0,
            earn_power=0
        )
        

    # 取引内容を追加
    #def add_transaction(self, transaction: Transaction):
    #    self.transaction_list.append(transaction)

    def calc_stats(self):
        for transaction in history:
            if transaction.
        return 0

    def show(self):
        print("--------------------------------------")
        print("TotalTradeNum: ", self.stats.total_trade_num)
        print("WinTradeNum: ", self.stats.win_trade_num)
        print("LoseTradeNum: ", self.stats.lose_trade_num)
        print("BuyNum: ", self.stats.buy_num)
        print("SellNum: ", self.stats.sell_num)
        print("WinRate: ", self.stats.win_rate)
        print("TotalProfit: ", self.stats.total_profit)
        print("TotalLoss: ", self.stats.total_loss)
        print("ProfitAverage: ", self.stats.profit_average)
        print("LossAverage: ", self.stats.loss_average)
        print("ProfitLossRate: ", self.stats.profit_loss_rate)
        print("EarnPower: ", self.stats.earn_power)
        print("--------------------------------------")




class SimParameter(NamedTuple):
    max_lot: int    # 最大数量
    spread: int     # スプレッド
    window_size: int

class Simulator():
    def __init__(self, forex_data: List[List[Price]], param: SimParameter):
        self.forex_data = forex_data
        self.model = model
        self.param = param
        self.history: List[Transaction2] = []      # トレード履歴
        self.steps = 0

    def get_window_data(self):
        return self.forex_data[self.steps: self.steps+self.param.window_size]

    def done(self) -> bool:
        if len(self.forex_data) == self.steps+self.param.window_size:
            return True
        else:
            return False

    def step(self, action: ActionType, lot: float):
        step_data = self.get_window_data()[-1]
        if action == ActionType.BUY:
            position_type = PositionType.LONG
            transaction = Transaction2(id=str(uuid.uuid4()),date=date, pair=PairType.USD_JPY, lot=lot, position=position_type, rate=step_data[-1].close)
            self.history.append(transaction)
        elif action == ActionType.SELL:
            position_type = PositionType.SHORT
            transaction = Transaction2(id=str(uuid.uuid4()), date=date, pair=PairType.USD_JPY, lot=lot, position=position_type, rate=step_data[-1].close)
            self.history.append(transaction)
        
        self.steps += 1

    def get_history(self) -> List[Transaction2]:
        return self.history



if __name__ == "__main__":
    '''# 為替データの取得
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
    #print(df.loc[(datetime(2005, 7, 1, 5, 00, 00, 0000), "H1"), :])'''


    mock_forex_data: List[List[Price]] = []
    date = datetime(2019, 2, 1, 12, 15, 30, 2000)
    for i in range(100):
        price1D = Price(date=date, open=120.0, close=120.0, low=100.0, high=100.0, type=GranularityType.D)
        price4H = Price(date=date, open=120.0, close=120.0, low=100.0, high=100.0, type=GranularityType.H4)
        price1H = Price(date=date, open=120.0, close=120.0, low=100.0, high=100.0, type=GranularityType.H1)
        price15M = Price(date=date, open=120.0, close=120.0, low=100.0, high=100.0, type=GranularityType.M15)
        day_prices: List[Price] = [price1D, price4H, price1H, price15M]
        mock_forex_data.append(day_prices)
        date = date + timedelta(days=1)

    #print(mock_forex_data)


    # モデル選択
    model = Model()

    # Simulator
    sim_param = SimParameter(max_lot=10, spread=0.005, window_size=30)
    simulator = Simulator(mock_forex_data, sim_param)

    while simulator.done() == False: 
        window_data = simulator.get_window_data()
        action, lot = model.get_action(window_data)
        simulator.step(action, lot)
        #simulator.show()

    history = simulator.get_history()

    # StatsCalculator
    stats_param = StatsParameter(initial_assets=500000, allowable_loss=1)
    stats_calculator = StatsCalculator(history, stats_param)
    stats_calculator.calc_stats()
    stats_calculator.show()


