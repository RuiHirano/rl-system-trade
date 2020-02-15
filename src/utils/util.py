import os
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Tuple, NamedTuple, Optional
from enum import Enum
from .types import PairType, GranularityType  # type: ignore


def import_csv_data(pair_type: PairType, granularityType:GranularityType):

    dataset_dir = os.path.abspath("../../dataset")
    os.chdir(dataset_dir + "/" + pair_type.name)
    df_data = pd.read_csv(pair_type.name+"_"+granularityType.name+".csv", encoding="utf-8")
    print(df_data)
    return df_data

def log_diff_series(data):
    df['ratio_log']=np.log(df['close'])-np.log(df['close'].shift(1))