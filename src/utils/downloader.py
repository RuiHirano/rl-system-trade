# datasetをダウンロードする

# coding: utf-8
from .util import PairType, GranularityType  # type: ignore
import csv
import os
import os.path
import json
from oandapyV20 import API  # type: ignore
from oandapyV20.exceptions import V20Error  # type: ignore
from oandapyV20.endpoints.pricing import PricingStream  # type: ignore
from datetime import *
import oandapyV20.endpoints.instruments as instruments  # type: ignore
from typing import List, Dict, Tuple, NamedTuple, Optional
import pandas as pd  # type: ignore
from enum import Enum
from tqdm import tqdm  # type: ignore





class Pair():
    def __init__(self, pairType: PairType):
        self.type = pairType

    def get_name(self) -> str:
        return self.type.name


class Period(NamedTuple):
    start_date: datetime
    end_date: datetime


class Account(NamedTuple):
    account_id: str
    access_token: str


class Granularity():
    def __init__(self, granularityType: GranularityType):
        self.type = granularityType

    def get_name(self) -> str:
        return self.type.name


class DateManager():
    def __init__(self, granularity: Granularity, period: Period):
        self.granularity = granularity
        self.start_date = period.start_date
        self.end_date = period.end_date
        self.from_date = period.start_date

    def get_next_from_to(self, delta_num: int) -> Tuple[datetime, datetime]:
        # from_date取得
        from_date: datetime = self.from_date

        # to_dateの計算
        to_date: datetime = datetime.now()
        if self.granularity.type == GranularityType.W:
            to_date = self.from_date + timedelta(weeks=delta_num)

        elif self.granularity.type == GranularityType.D:
            to_date = self.from_date + timedelta(days=delta_num)

        elif self.granularity.type == GranularityType.H4:
            to_date = self.from_date + timedelta(hours=4 * delta_num)

        elif self.granularity.type == GranularityType.H1:
            to_date = self.from_date + timedelta(hours=delta_num)

        elif self.granularity.type == GranularityType.M15:
            to_date = self.from_date + timedelta(minutes=15 * delta_num)

        elif self.granularity.type == GranularityType.M5:
            to_date = self.from_date + timedelta(minutes=5 * delta_num)

        # to_dateがend_dateを超えるかどうか
        if to_date > self.end_date:
            to_date = self.end_date

        # from_dateを更新
        self.from_date = to_date
        return from_date, to_date

    # 次のループを回せるかどうか
    def is_next(self) -> bool:
        return self.from_date < self.end_date


class Downloader():
    def __init__(self, save_path: str,  pairs: List[Pair], granularities: List[Granularity], period: Period, account: Account):
        self.save_path = save_path
        self.pairs = pairs
        self.granularities = granularities
        self.period = period
        self.delta_num = 5000   # 一度に取得できる数
        self.api = API(access_token=access_token, environment="practice")

    def download(self):
        for pair in self.pairs:
            print("\n---------------")
            print("Pair: " + pair.get_name())

            # datasetディレクトリがなければ作成
            os.chdir(self.save_path)
            if not os.path.isdir("dataset"):
                os.mkdir("dataset")

            # 通貨ディレクトリがなければ作成
            os.chdir(self.save_path + "/dataset")
            if not os.path.isdir(pair.get_name()):
                os.mkdir(pair.get_name())

            # データをロード
            self.download_rate_from_oanda(pair)

    def download_rate_from_oanda(self, pair: Pair):
        # 粒度毎に計算
        for granularity in self.granularities:
            print("\nPair: " + pair.get_name() +
                  ", Granularity: " + granularity.get_name())

            # すでにファイルが作成されている場合スキップ
            csv_name = pair.get_name() + "_" + granularity.get_name() + ".csv"
            os.chdir(self.save_path + "/dataset/" + pair.get_name())
            if os.path.isfile(csv_name):
                print("Already Created!")
                continue

            # dataframe初期化
            data = pd.DataFrame({
                'time': [],
                'volume': [],
                'high': [],
                'low': [],
                'open': [],
                'close': [],
            },
            )

            # 一度に取得できるのが5000までなのでfrom-toで回す
            dateManager = DateManager(granularity, self.period)
            while dateManager.is_next():
                from_date, to_date = dateManager.get_next_from_to(
                    self.delta_num)
                params = {
                    "from": from_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "to": to_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "granularity": granularity.get_name()
                }
                # データ取得
                r = instruments.InstrumentsCandles(
                    instrument=pair.get_name(), params=params)
                self.api.request(r)
                candles = r.response['candles']
                print("from: " + from_date.strftime("%Y-%m-%dT%H:%M:%SZ") +
                      ", to: " + to_date.strftime("%Y-%m-%dT%H:%M:%SZ"))

                # candle毎でdataに格納する
                for candle in tqdm(candles):
                    data = data.append(
                        pd.Series(
                            {
                                'time': candle["time"],
                                'volume': candle["volume"],
                                'high': candle["mid"]["h"],
                                'low': candle["mid"]["l"],
                                'open': candle["mid"]["o"],
                                'close': candle["mid"]["c"],
                            },
                        ),
                        ignore_index=True
                    )

            # CSVに保存
            # pair_nameのディレクトリに移動
            # csv_name = pair.get_name() + "_" + granularity.get_name() + ".csv"
            os.chdir(self.save_path + "/dataset/" + pair.get_name())
            data = data.set_index("time")
            data.index = pd.to_datetime(data.index)
            data.to_csv(csv_name)
            print("CSV created! name: ", csv_name)


if __name__ == "__main__":

    # 保存するディレクトリパス
    save_path = os.path.abspath("../../")

    # IDとTokenを取得
    config_dir = os.path.abspath("../../config")
    with open(config_dir+"/config.json") as f:
        j = json.load(f)

    # accountID, accessTokenを取得（本来は環境変数にすべき）
    account_id = j["demo"]["account_id"]
    access_token = j["demo"]["api_key"]
    account = Account(account_id, access_token)
    print("Account ID: ", account_id)
    print("Access Token: ", access_token)

    # 取得する通貨を選択
    pairs = [
        Pair(PairType.USD_JPY),
        Pair(PairType.GBP_JPY),
        Pair(PairType.EUR_JPY),
        Pair(PairType.GBP_USD),
        Pair(PairType.EUR_USD),
        Pair(PairType.USD_CHF)
    ]
    # 取得する周期を選択
    granularities = [
        Granularity(GranularityType.W),
        Granularity(GranularityType.D),
        Granularity(GranularityType.H4),
        Granularity(GranularityType.H1),
        Granularity(GranularityType.M15),
        Granularity(GranularityType.M5)
    ]
    # 取得する期間を選択
    start_date: datetime = datetime(2005, 7, 1, 0, 0, 0)
    end_date: datetime = datetime.today() - timedelta(days=1)
    period = Period(start_date, end_date)

    downloader = Downloader(save_path, pairs, granularities, period, account)
    downloader.download()
