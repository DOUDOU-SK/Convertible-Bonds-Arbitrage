# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
from datetime import datetime, timedelta, time
import os
from xy.data.query import load_lv2
import GET_TradingDates
from GET_TradingDates import read_trading_days


############PART 1:Get All the trading days from 2020 to 2024
trade_dates_list = pd.read_csv()


##############PART 2

def get_start_data(trade_date_str, stock, ticker):

    ds = load_lv2('Transe', trade_date_str, market = 'stock')
    dt = load_lv2('Transe', trade_date_str, market = 'bond')
   
    ds_stock_raw = pd.DataFrame(ds[stock])
    dt_ticker_raw = pd.DataFrame(dt[ticker])
    
    if ticker not in list(dt.tickers):
        print('No valid data')
        ds_stock = ds_stock_raw
        dt_ticker = dt_ticker_raw
    
    else:
    
        ds_stock = ds_stock_raw[ds_stock_raw['FunctionCode'] != 'C'].copy()
        dt_ticker = dt_ticker_raw[dt_ticker_raw['FunctionCode']!= 'C'].copy()

        ds_stock['date_time'] = pd.to_datetime(ds_stock['date_time'])
        dt_ticker['date_time'] = pd.to_datetime(dt_ticker['date_time'])

        ds_stock.sort_values(by = 'date_time', inplace = True)
        dt_ticker.sort_values(by = 'date_time', inplace = True)
        
        
    return ds_stock, dt_ticker
    

def GET_LIMIT_UP(trade_date_str, ds_stock, stock):

    target_index = trade_dates_list.index(trade_date_str)
    prev_trade_date = trade_dates_list[target_index - 1] if target_index > 0 else None

    if prev_trade_date is None:

        ds_stock.sort_values(by = 'date_time', inplace = True)
        LIMIT_UP_PRICE = ds_stock[ds_stock['TradeVolume'] > 0]['TradePrice'].max()
    
    else:

        temp_df = load_lv2('Transe', prev_trade_date, market = 'stock')
        temp_stock_raw = pd.DataFrame(temp_df[stock])
        
        if temp_stock_raw.empty: 
            ds_stock.sort_values(by = 'date_time', inplace = True)
            LIMIT_UP_PRICE = ds_stock[ds_stock['TradeVolume'] > 0]['TradePrice'].max()
            
        else:     
            temp_stock = temp_stock_raw[temp_stock_raw['FunctionCode'] != 'C'].copy()
            temp_stock['date_time'] = pd.to_datetime(temp_stock['date_time'])
            temp_stock.sort_values(by='date_time', inplace=True)

            Prev_CLOSEPRICE = temp_stock[temp_stock['TradeVolume'] > 0]['TradePrice'].iloc[-1]
    
            #print(f'The previous closeprice of {stock} on {prev_trade_date} is: {Prev_CLOSEPRICE}')
        
            LIMIT_UP_PRICE_cc = round(Prev_CLOSEPRICE * 1.1, 2) if str(stock).startswith(('00', '60')) else round(Prev_CLOSEPRICE * 1.2, 2)
            LIMIT_UP_PRICE_ch = ds_stock[ds_stock['TradeVolume'] > 0]['TradePrice'].max()
    
            LIMIT_UP_PRICE = min(LIMIT_UP_PRICE_cc, LIMIT_UP_PRICE_ch)


    LIMIT_UP_TIME = ds_stock[ds_stock['TradePrice'] == LIMIT_UP_PRICE]['date_time'].min()
    
    # 设定容忍时间（炸板 < 2分钟的不算）
    TOLERANCE_MINUTES = 2
    TOLERANCE_SECONDS = TOLERANCE_MINUTES * 60

    # 添加标志：是否低于涨停价（视为炸板）
    ds_stock_x = ds_stock.copy()
    ds_stock_x['is_break'] = ds_stock_x['TradePrice'] < (LIMIT_UP_PRICE - 1e-6)
    
    # 只分析 entry_cb_time 之后的数据
    after_entry = ds_stock_x[ds_stock_x['date_time'] >= LIMIT_UP_TIME].copy()
    
    # 标记是否炸板段的开始和结束（用分组编号分段）
    after_entry['break_shift'] = after_entry['is_break'].shift(1, fill_value=False)
    after_entry['break_group'] = (after_entry['is_break'] & ~after_entry['break_shift']).cumsum()

    # 提取所有炸板段
    break_groups = after_entry[after_entry['is_break']].groupby('break_group')

    BREAK_LIMIT_UP_TIME = ds_stock['date_time'].max()

    for _, group in break_groups:
        start_time = group['date_time'].iloc[0]
        end_time = group['date_time'].iloc[-1]
        duration = (end_time - start_time).total_seconds()
    
        if duration >= TOLERANCE_SECONDS:
            BREAK_LIMIT_UP_TIME = start_time
            break  # 找到第一个有效炸板段就停止
    
    #print(f'The real time point to leave the market:{BREAK_LIMIT_UP_TIME}')

    return LIMIT_UP_PRICE, LIMIT_UP_TIME, BREAK_LIMIT_UP_TIME  



###################PART 3

def Factors(dt_ticker, LIMIT_UP_TIME, GAP = '2T'):

    dt = dt_ticker.copy()
    dt['TradeAmount'] = dt['TradePrice'] * dt['TradeVolume']
    dt.sort_values(by = 'date_time', inplace = True)

    dt.set_index('date_time', inplace = True)

    resampled = dt.resample(GAP).agg(
        {
            'TradePrice': 'last',
            'TradeAmount': 'sum',
            'AskOrder': 'sum',
            'BidOrder': 'sum',
            'BSFlag': lambda x: x.value_counts().to_dict()
        }
    )

    trade_count_per_sec = dt.resample(GAP).size().rename('TradeCount')
    resampled = resampled.merge(trade_count_per_sec, left_index=True, right_index=True, how='left')

    # log_return_30s：实际上基于1min前的价格（30s × 6 = 3mins）
    resampled['log_return_6mins'] = np.log(resampled['TradePrice']) - np.log(resampled['TradePrice'].shift(3))

    # 2. trade_rate_amount_30s：过去6个10s内的均值成交金额（即60s窗口）
    resampled['trade_rate_amount_2T'] = resampled['TradeAmount'].rolling(window=6, min_periods=1).mean()
    resampled['trade_rate_count_2T'] = resampled['TradeCount'].rolling(window = 6, min_periods=1).mean()
    resampled['trade_price_2T'] = resampled['TradePrice'].rolling(window=6, min_periods=1).mean()

    # 3. order_depth_diff：当前买盘 - 卖盘差
    resampled['order_depth_diff'] = (resampled['BidOrder'] - resampled['AskOrder'])/(resampled['BidOrder'] + resampled['AskOrder']+1e-6)
    resampled['order_diff_trend'] = resampled['order_depth_diff'].rolling(7).mean().diff()
    # 4. order_depth_diff_change：盘口反转迹象（即差值变化）
    resampled['order_depth_change'] = np.sign(resampled['order_depth_diff'] * resampled['order_depth_diff'].shift(1))

    
    resampled['amplitude_6mins'] = resampled['TradePrice'].rolling(3).apply(lambda x: x.max() - x.min())
    resampled['price_rollmax'] = resampled['trade_price_2T'].rolling(3, min_periods=1).max()
    resampled['is_price_rolling_top'] = resampled['trade_price_2T'] >= resampled['price_rollmax'].shift(1)


    # 6. volatility_30s：最近30s价格波动率（标准差）
    resampled['log_volatility'] = resampled['log_return_6mins'].rolling(window=6, min_periods=1).std()
    resampled['log_sharpe'] = np.where(
    resampled['log_volatility'].fillna(0) != 0,
    resampled['log_return_6mins'] / resampled['log_volatility'],
    np.nan  )
    
    # 7. 动量持续性衰减指标（Momentum Decay）
    resampled['momentum'] = resampled['log_return_6mins'].rolling(5).mean()
    resampled['momentum_decay'] = resampled['momentum'].diff()

    # 8. 成交量反转强度
    resampled['trade_amount_growth'] = resampled['trade_rate_amount_2T'].diff()


    return resampled




