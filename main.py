from os import environ
from datetime import datetime, timezone, timedelta
from cybotrade.strategy import Strategy as BaseStrategy
from cybotrade.runtime import Runtime
from cybotrade.models import (
    DatahubConfig,
    Direction,
    Exchange,
    Interval,
    OrderParams,
    OrderSide,
    StopParams,
    OrderSize,
    OrderSizeUnit,
    RuntimeConfig,
    RuntimeMode,
    Symbol,
)
import numpy as np
import asyncio
import logging
import colorlog
import random
import requests
from binance import Client
import json
import collections
import time
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

class TradingStrategy:
    def __init__(self, lower_percentile_threshold, upper_percentile_threshold):
        self.price_df = None
        self.lower_percentile_threshold = lower_percentile_threshold
        self.upper_percentile_threshold = upper_percentile_threshold

    def fetch_kline_price_data(self):
        API_Library='v1/klines'
        ticker = 'BTCUSDT'
        time_interval= '1d'
        start_time = datetime(2023,12,21) #获取数据开始的时间，YYYY/M/D
        end_time = datetime.now() #数据最新更新的时间

        price_data=[]
        
        while start_time < end_time:
            start_time_2 = int(start_time.timestamp() * 1000)
            url = 'https://fapi.binance.com/fapi/'+str(API_Library)+'?symbol='+str(ticker)+'&interval='+str(time_interval)+'&limit=1500&startTime='+str(start_time_2)
            resp = requests.get(url)
            resp = json.loads(resp.content.decode())  

            price_data.extend(resp)
            
            start_time = start_time + timedelta(days=1500)
        
        price_data = pd.DataFrame(price_data)

        price_data[0] = pd.to_datetime(price_data[0], unit="ms")
        price_data[6] = pd.to_datetime(price_data[6], unit="ms")

        price_data.columns = ['Time', 'Open','High', 'Low', 'Close', 'Volume','Close Time','Ignore','Ignore','Ignore','Ignore','Ignore']
        price_data = price_data.set_index('Time')

        extracted_df = price_data[['Close', 'High', 'Low']]
        
        self.price_df = extracted_df

    def max_magnitude(self, x, y):
        return x if abs(x) > abs(y) else y
    
    def signal(self, Max_Significant_Value, lower_percentile_threshold, upper_percentile_threshold):
        return '0' if Max_Significant_Value > lower_percentile_threshold and Max_Significant_Value < upper_percentile_threshold else '1' if Max_Significant_Value < lower_percentile_threshold else '-1' if Max_Significant_Value > upper_percentile_threshold else ''

    def position_trigger(self, threshold_interval, Signal):
        return '1' if Signal > threshold_interval else '-1' if Signal < -threshold_interval else '0'

    def hoot(self, Close, Ma_close_Price):
        try:
            close_float = float(Close)
            ma_close_float = float(Ma_close_Price)
            return '1' if close_float > ma_close_float else '-1' if close_float < ma_close_float else '0'
        except ValueError:
            return 'NaN'
    

    def pnl_calculation(self, previous, today, position):
        if previous is None or today is None:
            return np.nan
        try:
            today = float(today)
            previous = float(previous)
        except ValueError:
            return np.nan
        
        if position == '1':
            return 1 * (today / previous - 1)
        elif position == '-1':
            return -1 * (today / previous - 1)
        else:
            return 0.0

    

    def run_strategy(self):
        self.fetch_kline_price_data()
        lookback = 60
        threshold_interval = 0.847882165 #延遲下單

        self.price_df['MaxSignificantValue'] = self.price_df.apply(lambda row: self.max_magnitude(float(row['High']), float(row['Low'])), axis=1)
        
        self.price_df['Signal'] = self.price_df.apply(lambda row: self.signal(row['MaxSignificantValue'],  self.lower_percentile_threshold, self.upper_percentile_threshold), axis=1)

        self.price_df['Sma_signal'] = self.price_df['Signal'].rolling(window=lookback).mean()
        self.price_df['MA_close_Price'] = self.price_df['Close'].rolling(window=lookback).mean()

        self.price_df['position_Trigger'] = self.price_df.apply(lambda row: self.position_trigger(threshold_interval, row['Sma_signal']), axis=1)

        self.price_df['Signal'] = self.price_df.apply(lambda row: self.hoot(row['Close'], row['MA_close_Price']), axis=1)

        self.price_df['previous_Close'] = self.price_df['Close'].shift(1)

        self.price_df['Daliy_PnL'] = self.price_df.apply(lambda row: self.pnl_calculation(row['previous_Close'], row['Close'], row['Signal']), axis=1)

        self.price_df ['cum_PnL'] = self.price_df['Daliy_PnL'].cumsum()

        print(self.price_df)
        self.price_df = self.price_df.to_csv('testing.csv')
        

if __name__ == "__main__":
    # 定義 lower 和 upper threshold 值
    lower_threshold = 0.5
    upper_threshold = 0.8
    
    # 創建 TradingStrategy 實例並傳遞閾值參數
    trading_strategy = TradingStrategy(lower_threshold, upper_threshold)
    
    # 運行策略
    trading_strategy.run_strategy()

