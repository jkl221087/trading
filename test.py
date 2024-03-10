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

def fetch_kline_price_data():
    API_Library='v1/klines'
    ticker = 'BTCUSDT'
    time_interval= '1d'
    start_time = datetime(2023,12,21) #获取数据开始的时间，YYYY/M/D
    end_time = datetime.now() #数据最新更新的时间

    price_data=[]
    
    
    while start_time < end_time:
        start_time_2 = int(start_time.timestamp() * 1000)
        url = 'https://fapi.binance.com/fapi/'+str(API_Library)+'?symbol='+str(ticker)+'&interval='+str(time_interval)+'&limit=1500&startTime='+str(start_time_2)
        print(start_time)
        resp = requests.get(url)
        resp = json.loads(resp.content.decode())  

        price_data.extend(resp)
        
        start_time = start_time + timedelta(days=1500)
    
    price_data = pd.DataFrame(price_data)

    price_data[0] = pd.to_datetime(price_data[0], unit="ms")
    price_data[6] = pd.to_datetime(price_data[6], unit="ms")

    #项目命名Columns Rename
    price_data.columns = ['Time', 'Open','High', 'Low', 'Close', 'Volume','Close Time','Ignore','Ignore','Ignore','Ignore','Ignore']
    price_data = price_data.set_index('Time')

    #price_data = price_data[price_data.index >= datetime(2020, 12, 1)]

    extracted_df = price_data[['Close', 'High', 'Low']]
    
    return extracted_df


price_request = fetch_kline_price_data() 

price_df = pd.DataFrame(price_request)

# amaxsignificantValue 比較hight low最大值
def max_magnitude(x, y):
    return x if abs(x) > abs(y) else y

price_df['MaxSignificantValue'] = price_df.apply(lambda row: max_magnitude(float(row['High']), float(row['Low'])), axis=1)



# # signal 訊號 
# #如果Max_Significant_Value大于lower_percentile_threshold且小于upper_percentile_threshold，函数返回字符串'0'。这可能表示市场处于一种相对稳定的状态，在两个阈值之间波动，没有显著的买入或卖出信号。
# #如果Max_Significant_Value小于lower_percentile_threshold，函数返回字符串'1'。这可能表示市场出现了一种比较极端的下跌状态，超过了下界阈值，可能是一个买入信号。
# 如果Max_Significant_Value大于upper_percentile_threshold，函数返回字符串'-1'。这可能表示市场出现了一种比较极端的上升状态，超过了上界阈值，可能是一个卖出信号。
# 如果Max_Significant_Value不满足上述任何条件，函数返回空字符串''
def signal(Max_Significant_Value, lower_percentile_threshold, upper_percentile_threshold):
    return '0' if Max_Significant_Value > lower_percentile_threshold and Max_Significant_Value < upper_percentile_threshold else '1' if Max_Significant_Value < lower_percentile_threshold else '-1' if Max_Significant_Value > upper_percentile_threshold else ''


# 上 下 阈值
upper_percentile_threshold = 0.001210026
lower_percentile_threshold = -0.000643092

price_df['Signal'] = price_df.apply(lambda row: signal(row['MaxSignificantValue'], lower_percentile_threshold, upper_percentile_threshold), axis=1)

# position_trigger 位置觸發
def position_trigger(threshold_interval, Signal):
    return '1' if Signal > threshold_interval else '-1' if Signal < -threshold_interval else '0'

lookback = 60
threshold_interval = 0.847882165 #延遲下單

price_df['Sma_signal'] = price_df ['Signal'].rolling(window=lookback).mean()
price_df['MA_close_Price'] = price_df['Close'].rolling(window=lookback).mean()

price_df['position_Trigger'] = price_df.apply(lambda row: position_trigger(threshold_interval, row['Sma_signal']), axis=1)


def hoot(Close, Ma_close_Price):
    try:
        close_float = float(Close)
        ma_close_float = float(Ma_close_Price)
        return '1' if close_float > ma_close_float else '-1' if close_float < ma_close_float else '0'
    except ValueError:
        return 'NaN'


price_df['Signal'] = price_df.apply(lambda row: hoot(row['Close'], row['MA_close_Price']), axis=1)


#計算虧損
def pnl_calculation(previous, today, position):
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
        return -1 * (today / previous -1)
    else:
        return 0.0# 返回浮点数值，表示没有持仓或者其他情况


price_df['previous_Close'] = price_df['Close'].shift(1)

price_df['Daliy_PnL'] = price_df.apply(lambda row: pnl_calculation((row['Close']), row['previous_Close'],row['Signal']), axis=1)

price_df ['cum_PnL'] = price_df['Daliy_PnL'].cumsum()

print(price_df)

price_df = price_df.to_csv('testing.csv')