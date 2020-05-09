import sys
sys.path.append('../')
from utils.types import PairType, GranularityType, ActionType, PositionType  # type: ignore
from utils.util import import_csv_data  # type: ignore
from typing import NamedTuple, List
from datetime import *
import uuid
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import random



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
        if self.position == PositionType.LONG:
            return self.lot * (self.settle_rate - self.entry_rate) * 10000
        else:
            return self.lot * (self.entry_rate - self.settle_rate) * 10000

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
        rand = random.random()
        if rand < 0.5:
            return ActionType.BUY, lot
        else:
            return ActionType.SELL, lot


class StatsParameter(NamedTuple):
    allowable_loss: float
    initial_assets: int     # 初期投資額


class StatsCalculator():
    def __init__(self, history: List[Transaction], params: StatsParameter):
        self.params = params
        self.history = history

        self.total_trade_num=0
        self.win_trade_num=0
        self.lose_trade_num=0
        self.buy_num=0
        self.sell_num=0
        self.win_rate=0
        self.total_profit=0
        self.total_loss=0
        self.profit_average=0
        self.loss_average=0
        self.profit_loss_rate=0
        self.earn_power=0
    

    def calc_stats(self):
        print(len(self.history))
        self.total_trade_num = len(self.history)
        for transaction in self.history:
            print(transaction.profit)
            if transaction.profit >= 0:
                # Win
                self.win_trade_num += 1
                self.total_profit += transaction.profit
            else:
                # Lose
                self.lose_trade_num += 1
                self.total_loss += transaction.profit

            if transaction.position == PositionType.LONG:
                self.buy_num += 1
            elif transaction.position == PositionType.SHORT:
                self.sell_num += 1

        if self.total_trade_num == 0:
            self.win_rate = 0
        else:  
            self.win_rate = self.win_trade_num / self.total_trade_num

        if self.win_trade_num == 0:
            self.profit_average = 0
        else:  
            self.profit_average = self.total_profit / self.win_trade_num

        if self.lose_trade_num == 0:
            self.loss_average = 0
        else:  
            self.loss_average = self.total_loss / self.lose_trade_num
        
        if self.loss_average == 0:
            self.profit_loss_rate = 10000
        else:  
            self.profit_loss_rate = abs(self.profit_average / self.loss_average)

        self.earn_power = self.win_rate * self.profit_loss_rate

    def show(self):
        print("--------------------------------------")
        print("TotalTradeNum: ", self.total_trade_num)
        print("WinTradeNum: ", self.win_trade_num)
        print("LoseTradeNum: ", self.lose_trade_num)
        print("BuyNum: ", self.buy_num)
        print("SellNum: ", self.sell_num)
        print("WinRate: ", self.win_rate)
        print("TotalProfit: ", self.total_profit)
        print("TotalLoss: ", self.total_loss)
        print("ProfitAverage: ", self.profit_average)
        print("LossAverage: ", self.loss_average)
        print("ProfitLossRate: ", self.profit_loss_rate)
        print("EarnPower: ", self.earn_power)
        print("--------------------------------------")




class SimParameter(NamedTuple):
    max_lot: int    # 最大数量
    spread: int     # スプレッド
    window_size: int

class Simulator():
    def __init__(self, forex_data: List[List[Price]], param: SimParameter):
        self.forex_data = forex_data
        self.param = param
        self.history: List[Transaction] = []      # トレード履歴
        self.transaction = None
        self.steps = 0
        self._test()

    def get_window_data(self):
        return self.forex_data[self.steps: self.steps+self.param.window_size]

    def done(self) -> bool:
        if len(self.forex_data) == self.steps+self.param.window_size:
            return True
        else:
            return False


    def get_history(self) -> List[Transaction]:
        return self.history

    # シミュレーション開始
    # FIX
    def step(self, action: ActionType, lot: float):
        step_data = self.forex_data[self.steps+self.param.window_size]
        date = step_data[-1].date

        if self.transaction == None and (action == ActionType.SELL or action == ActionType.BUY):
            position = self.get_build_position(action)
            print(position)
            self.transaction = Transaction(start_date=date, pair=PairType.USD_JPY, lot=lot, position=position, entry_rate=step_data[-1].close)

        # 決済した場合、statsにデータを送る
        elif self.is_release(self.transaction.position, action):
            # 取引内容を追記
            self.transaction.settle(end_date=date, settle_rate=step_data[-1].close)
            # historyに追加
            self.history.append(self.transaction)
            # 初期化
            self.transaction = None

        self.steps += 1



    # 利益確定したかどうか
    def is_release(self, position: PositionType, action: ActionType) -> bool:
        if position == PositionType.LONG and action == ActionType.SELL or position == PositionType.SHORT and action == ActionType.BUY:
            return True
        else:
            return False
    

    # ポジションを建てたかどうか
    def get_build_position(self, action: ActionType) -> PositionType:
        if action == ActionType.SELL:
            return PositionType.SHORT
        elif action == ActionType.BUY:
            return PositionType.LONG
        else:
            return PositionType.NONE

    def _test(self):
        # is_release
        assert self.is_release(PositionType.LONG, ActionType.SELL) == True, '[LongでSellをしたら売却] 期待する値[{0}], 出力値[{1}]'.format(True, False)
        assert self.is_release(PositionType.LONG, ActionType.BUY) == False, '[LongでBuyなら売却しない] 期待する値[{0}], 出力値[{1}]'.format(False, True)
    

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
        price1D = Price(date=date, open=100.0+5 * random.random(), close=100.0+5 * random.random(), low=100.0+5 * random.random(), high=100.0+5 * random.random(), type=GranularityType.D)
        price4H = Price(date=date, open=100.0+5 * random.random(), close=100.0+5 * random.random(), low=100.0+5 * random.random(), high=100.0+5 * random.random(), type=GranularityType.H4)
        price1H = Price(date=date, open=100.0+5 * random.random(), close=100.0+5 * random.random(), low=100.0+5 * random.random(), high=100.0+5 * random.random(), type=GranularityType.H1)
        price15M = Price(date=date, open=100.0+5 * random.random(), close=100.0+5 * random.random(), low=100.0+5 * random.random(), high=100.0+5 * random.random(), type=GranularityType.M15)
        #print(price15M)
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


