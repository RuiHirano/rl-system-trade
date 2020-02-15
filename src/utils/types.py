
from enum import Enum

class PairType(Enum):
    USD_JPY = 1
    GBP_JPY = 2
    EUR_JPY = 3
    GBP_USD = 4
    EUR_USD = 5
    USD_CHF = 6


class GranularityType(Enum):
    W = 1
    D = 2
    H4 = 3
    H1 = 4
    M15 = 5
    M5 = 6

class ActionType(Enum):
    BUY = 1
    SELL = 2
    NONE = 3

class PositionType(Enum):
    LONG = 1
    SHORT = 2
    NONE = 3