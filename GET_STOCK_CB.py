# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
from datetime import datetime, timedelta, time
import os
import pickle
import matplotlib.pyplot as plt 



##############PART 1:Get All the trading days from 2020 to 2024

trade_dates = pd.read_csv('TRADINGDAYS.csv', index_col = 0)
trade_dates.reset_index(drop = True)
trade_dates['Date'] = trade_dates['Date'].astype(str)

##############PART 2: GET All the Ever LIMIT_UP Tickers and their information

LIMIT_UPs = pd.read_csv('Tickers_LIMITUP_TIME.csv')
LIMIT_UPs['LIMIT_UP_TIME'] = pd.to_datetime(LIMIT_UPs['LIMIT_UP_TIME'])
LIMIT_UPs['BREAK_LIMIT_UP_TIME'] = pd.to_datetime(LIMIT_UPs['BREAK_LIMIT_UP_TIME'])
LIMIT_UPs['Date'] = LIMIT_UPs['Date'].astype(str) 

##############PART 3: GET All the Transe Data of convertible bonds saved in pickles splitted by trade_date

def Split_tickers(dict):
    keys = list(dict.keys())
    xlist = []
    for t in keys:
        temp_df = pd.DataFrame(dict[t])
        temp_df['Ticker'] = t
        
        xlist.append(temp_df) 

    df = pd.concat(xlist, axis = 0)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df.sort_values(by = ['Ticker','date_time'], inplace = True)
    df.reset_index(drop = True, inplace = True)
    
    cols = ['Ticker'] + [col for col in df.columns if col != 'Ticker']
    return df[cols]

def GET_PICKLE(trade_date_str, base_path,  tickertype):
    ticker_path = os.path.join(base_path, f'Transe_{tickertype}_{trade_date_str}.pkl')
    temp_dict = pd.read_pickle(ticker_path)
    temp_df = Split_tickers(temp_dict)
    return temp_df     

##############PART 4: GET the LIMIT_UP Information of Convertible Bonds

def GET_LIMIT_UP(trade_date_str, df, tickertype):
    """
    trade_date_str: %Y%m%d, string of trading days
    df: dataframe of Transe level-2 data of convertible bonds
    tickertype: the type of df, 'Stock' or 'Ticker'
    """

    ### Step1: get the tickers traded at trade_date_str
    tickers = df['Ticker'].drop_duplicates().tolist()
    price_dict = {}
    entrytime_dict = {}
    exittime_dict = {}    
    
    ### Step2: obtain the index of target trading date and the date before it 
    target_index = trade_dates_list.index(trade_date_str) 
    prev_trade_date = trade_dates_list[target_index - 1] if target_index > 0 else None

    if prev_trade_date is not None:
        prev_dff = load_lv2('Transe', prev_trade_date, market = tickertype)

    ### Step3: Get LIMIT_UP_PRICE, LIMIT_UP_TIME, BREAK_LIMIT_UP_TIME     
    
    for t in tickers:
        
        if prev_trade_date is not None:
            prev_dff_t = pd.DataFrame(prev_dff[t])
            
        else:
            prev_dff_t = pd.DataFrame()    

        ######## Empty dataset may have two reasons: 1.it's the first date on trading_dates_list 2.this convertible bond is not traded on the previous one date
        if prev_dff_t.empty: 
            
            LIMIT_UP_PRICE = df[(df['Ticker'] == t) & (df['FunctionCode'] != 'C') & (df['TradeVolume']>0)]['TradePrice'].max()
            LIMIT_UP_TIME = df[ (df['Ticker'] == t) & (df['TradePrice'] == LIMIT_UP_PRICE)]['date_time'].min()
            
        else:   
        
            dfs = prev_dff_t[prev_dff_t['FunctionCode'] != 'C'].copy()
            dfs['date_time'] = pd.to_datetime(dfs['date_time'])          
            dfs.sort_values(by='date_time', inplace=True)
            Prev_CLOSEPRICE = dfs[dfs['TradeVolume'] > 0]['TradePrice'].iloc[-1]
    
            print(f'The closeprice of Ticker {t} on {prev_trade_date} is: {Prev_CLOSEPRICE}')
        
            LIMIT_UP_PRICE_cc = round(Prev_CLOSEPRICE * 1.1, 2) if str(t).startswith(('00', '60')) else round(Prev_CLOSEPRICE * 1.2, 2)
            LIMIT_UP_PRICE_ch = df[(df['Ticker'] == t) & (df['FunctionCode'] != 'C') & (df['TradeVolume']>0)]['TradePrice'].max()

            LIMIT_UP_TIME_cc = df[ (df['Ticker'] == t) & (df['TradePrice'] == LIMIT_UP_PRICE_cc)]['date_time'].min()
            LIMIT_UP_TIME_ch = df[ (df['Ticker'] == t) & (df['TradePrice'] == LIMIT_UP_PRICE_ch)]['date_time'].min()
    
            if LIMIT_UP_PRICE_ch <= LIMIT_UP_PRICE_cc:
                LIMIT_UP_PRICE = LIMIT_UP_PRICE_ch
                LIMIT_UP_TIME = LIMIT_UP_TIME_ch
            elif LIMIT_UP_PRICE_ch > LIMIT_UP_PRICE_cc and (LIMIT_UP_TIME_cc is None or pd.isna(LIMIT_UP_TIME_cc)) :
                LIMIT_UP_PRICE = LIMIT_UP_PRICE_ch
                LIMIT_UP_TIME = LIMIT_UP_TIME_ch
            elif LIMIT_UP_PRICE_ch > LIMIT_UP_PRICE_cc and LIMIT_UP_TIME_cc is not None and not pd.isna(LIMIT_UP_TIME_cc) :
                LIMIT_UP_PRICE = LIMIT_UP_PRICE_cc
                LIMIT_UP_TIME = LIMIT_UP_TIME_cc
                
                
        ######## Find and save the BREAK_LIMIT_UP_TIME        
        # set the minimum period of time to leave the upper line
        TOLERANCE_MINUTES = 2 
        TOLERANCE_SECONDS = TOLERANCE_MINUTES * 60

        # 添加标志：是否低于涨停价（视为炸板）
        ds_stock_x = df[(df['Ticker'] == t) & (df['FunctionCode'] != 'C') & (df['TradeVolume'] >0)].copy()
        ds_stock_x['is_break'] = ds_stock_x['TradePrice'] < (LIMIT_UP_PRICE - 1e-6)
    
        # 只分析 entry_cb_time 之后的数据
        after_entry = ds_stock_x[ds_stock_x['date_time'] >= LIMIT_UP_TIME].copy()
    
        # 标记是否炸板段的开始和结束（用分组编号分段）
        after_entry['break_shift'] = after_entry['is_break'].shift(1, fill_value=False)
        after_entry['break_group'] = (after_entry['is_break'] & ~after_entry['break_shift']).cumsum()

        # 提取所有炸板段
        break_groups = after_entry[after_entry['is_break']].groupby('break_group')

        BREAK_LIMIT_UP_TIME = ds_stock_x['date_time'].max()
        #print(f'Initial Break_Limitup_Time is {BREAK_LIMIT_UP_TIME}')

        for _, group in break_groups:
            start_time = group['date_time'].iloc[0]
            end_time = group['date_time'].iloc[-1]
            duration = (end_time - start_time).total_seconds()
    
            if duration >= TOLERANCE_SECONDS :
                BREAK_LIMIT_UP_TIME = start_time
                break  # 找到第一个有效炸板段就停止
        
        ### Step4: Save LIMIT_UP_PRICE, LIMIT_UP_TIME, BREAK_LIMIT_UP_TIME            
        price_dict[t] = LIMIT_UP_PRICE        
        entrytime_dict[t] =  LIMIT_UP_TIME 
        exittime_dict[t] = BREAK_LIMIT_UP_TIME
        
    result = pd.DataFrame({
        'Ticker': tickers,
        'LIMIT_UP_PRICE': list(price_dict.values()),
        'LIMIT_UP_TIME': list(entrytime_dict.values()),
        'BREAK_LIMIT_UP_TIME': list(exittime_dict.values())
        
        })

    return result    


##############PART 5: GET the Factors Information of Convertible Bonds

def Factors(group, Win ,GAP = '1min', FixTime = True):
    """
    This function 'Factors' calculates various factors and indicators based on time-series market data.
    Parameters:
    - group: Slice DataFrame about Transe level-2 info for fixed ticker and trade_date
    - Win: Rolling window size for calculating indicators
    - GAP: Time interval for resampling, default '1min'
    - FixTime: Get Factors of all day or Factors at special time(limit_up/break_limit_up time), default = True
    """
    ################################ 初始化数据  ################################
    LIMIT_UP_TIME = pd.to_datetime(group['LIMIT_UP_TIME'].iloc[0])
    BREAK_LIMIT_UP_TIME = pd.to_datetime(group['BREAK_LIMIT_UP_TIME'].iloc[0])
    
    dt_ticker = group.drop(columns = ['Ticker', 'LIMIT_UP_TIME', 'BREAK_LIMIT_UP_TIME']).copy()
    dt = dt_ticker[(dt_ticker['FunctionCode'] != 'C') & (dt_ticker['TradeVolume'] > 0)].copy()    
    dt['TradeAmount'] = dt['TradePrice'] * dt['TradeVolume']
    dt.sort_values(by = 'date_time', inplace = True)    
    
    ################################ 获取涨停时刻的基本特征数据  ################################
    after_limitup_dt = pd.Series(dt['date_time'] - LIMIT_UP_TIME)
    
    if after_limitup_dt.empty:
        entry_price = None
        entry_amount = None
        entry_size = None
        print('No data after limit_up_time,Type 1 Error')
        
    else:   
        try: 
            entry_idx_min = after_limitup_dt.abs().idxmin()
        
            entry_price = dt.loc[entry_idx_min, 'TradePrice']
            entry_amount = dt.loc[entry_idx_min, 'TradeAmount']
            entry_size = dt.loc[entry_idx_min, 'TradeVolume']
        
        except (ValueError, KeyError, IndexError):
            print('Type 2 Error, something wrong in index or key-value')
    
    ################################ 准备 Resample 数据  ################################
    dt.set_index('date_time', inplace = True)
    resampledx = dt.resample(GAP).agg(
        {
            'TradePrice': 'last',
            'TradeVolume': 'sum',
            'TradeAmount': 'sum',
            'AskOrder': 'sum',
            'BidOrder': 'sum',
            'BSFlag': lambda x: x.value_counts().to_dict()
        }
    )

    ######## 1. 收集窗口时间内的成功的交易笔数以及买卖交易笔数 ############
    buy_counts = dt[dt['BSFlag'] == 'B'].resample(GAP).size() #########3 但是这里索引会出现错位，所以后面紧接着要对它进行调整
    sell_counts = dt[dt['BSFlag'] == 'S'].resample(GAP).size() #########3 但是这里索引会出现错位，所以后面紧接着要对它进行调整
    
    ## Adjustment For BUY/SELL
    # 统一索引，填充缺失值为0
    trade_count_per_sec = dt.resample(GAP).size().rename('TradeCount')
    buy_counts_per_sec = buy_counts.reindex(trade_count_per_sec.index, fill_value=0).rename('BuyCount')
    sell_counts_per_sec = sell_counts.reindex(trade_count_per_sec.index, fill_value=0).rename('SellCount')    
    
    ## Concat All the New features: TradeCount, BuyCount, SellCount
    resampledy = resampledx.merge(trade_count_per_sec, left_index=True, right_index=True, how='left')
    resampledz = resampledy.merge(buy_counts_per_sec, left_index=True, right_index=True, how='left')
    resampled = resampledz.merge(sell_counts_per_sec, left_index=True, right_index=True, how='left')
    
    
    ######### 2. rolling平滑获取交易额变化趋势，这些才有预测未来的能力 ########### 
    resampled['trade_amount_rolling_5'] = resampled['TradeAmount'].rolling(window=Win, min_periods=1).mean() #其中函数参数Win代表统一平滑的窗口长度
    resampled['trade_amount_growth'] = (resampled['trade_amount_rolling_5'].pct_change()).replace([np.inf, -np.inf], np.nan)  
     
    # 先计算每个窗口的资金变动斜率
    resampled['trade_amount_slope'] = resampled['TradeAmount'].rolling(Win).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]) 
    # 再计算斜率的差分，近似加速度
    resampled['amount_slope_growth'] = (resampled['trade_amount_slope'].pct_change()).replace([-np.inf, np.inf], np.nan)

    if entry_amount is not None:
        resampled['amount_rel_entry'] = (np.log(resampled['trade_amount_rolling_5']) - np.log(entry_amount)).replace([np.inf, -np.inf], np.nan)  
    else:
        resampled['amount_rel_entry'] = None
        print('Attention Please !!! Trading Amount at limit_up_time is None, there are something wrong')    
    
    ######### 3. rolling平滑获取交易笔数变化趋势，这些才有预测未来的能力 ###########
    resampled['trade_count_rolling_5'] = resampled['TradeCount'].rolling(window = Win, min_periods=1).mean() 
    resampled['trade_count_growth'] = (resampled['trade_count_rolling_5'].pct_change()).replace([np.inf, -np.inf], np.nan)
    
    # 先计算每个窗口的资金变动斜率
    resampled['trade_count_slope'] = resampled['TradeCount'].rolling(Win).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]) 
    # 再计算斜率的差分，近似加速度
    resampled['count_slope_growth'] = (resampled['trade_count_slope'].pct_change()).replace([-np.inf, np.inf], np.nan) 


    ######### 4. 计算窗口时间段内买卖力量的差距，反映市场多空力量的差异 ###########
    resampled['buy_ratio'] = resampled['BuyCount']/(resampled['BuyCount'] + resampled['SellCount']+1e-6) ##比值越接近1说明多方强，越接近0说明空方占优
    resampled['delta_buy_ratio'] = (resampled['buy_ratio'].pct_change()).replace([np.inf, -np.inf], np.nan)


    ######### 5. 动态收益率及其相关特征，比如变化动量之类的 ###########
    resampled['log_return_3GAP'] = (np.log(resampled['TradePrice']) - np.log(resampled['TradePrice'].shift(3))).replace([np.inf, -np.inf], np.nan) # 原来是3
    resampled['momentum'] = resampled['log_return_3GAP'].rolling(window=Win, min_periods=1).mean() ######### momentum是log_return的平滑均值结果                         
    resampled['momentum_decay'] = resampled['momentum'].diff()                          


    ######### 6. 买卖盘特征：当前买盘 - 卖盘差 #########
    resampled['order_depth_diff'] = (resampled['BidOrder'] - resampled['AskOrder'])/(resampled['BidOrder'] + resampled['AskOrder']+1e-6)
    resampled['order_diff_growth'] = (resampled['order_depth_diff'].pct_change()).replace([np.inf, -np.inf], np.nan)
    # order_depth_diff_change：盘口反转迹象（即差值变化）
    resampled['order_depth_change'] = np.sign(resampled['order_depth_diff'] * resampled['order_depth_diff'].shift(1))
    ######### 7. impact_cost #########
    resampled['impact_cost'] = resampled['TradeAmount'] / (resampled['BidOrder'] + resampled['AskOrder'] + 1e-6)

    
#############################
    if FixTime: 
        cols = ['trade_amount_growth','amount_slope_growth', 'buy_ratio', 'delta_buy_ratio', 'log_return_3GAP', 
            'momentum', 'impact_cost', 'order_diff_growth', 'order_depth_change']
    
        dfs = resampled[cols].copy()
        dfs.reset_index(inplace = True)
        
        limit_up_diff = pd.Series(dfs['date_time'] - LIMIT_UP_TIME)
        break_limit_up_diff = pd.Series(dfs['date_time'] - BREAK_LIMIT_UP_TIME)

        if limit_up_diff.empty or break_limit_up_diff.empty:
            ans = pd.DataFrame()

        else:   
            try: 
                if limit_up_diff.iloc[0] >= pd.Timedelta(0):
                    entry_idx_min = limit_up_diff.abs().idxmin()
                else:    
                    entry_idx_min = limit_up_diff[limit_up_diff <pd.Timedelta(0)].abs().idxmin()
                    
                break_idx_min = break_limit_up_diff.abs().idxmin()

                rows = [entry_idx_min, break_idx_min]    
                ans = dfs.loc[rows, ['date_time'] + cols]

                ans['special_time'] = [LIMIT_UP_TIME, BREAK_LIMIT_UP_TIME]
                new_cols = ['special_time', 'date_time'] + cols
        
            except (ValueError, KeyError, IndexError):
                print('Type 2 Error, something wrong in index or key-value')

        return ans[new_cols]

    else:
        cols = ['trade_amount_growth','trade_amount_slope', 'amount_slope_growth', 'amount_rel_entry', 'trade_count_growth', 'trade_count_slope', 'count_slope_growth',
    
    'buy_ratio', 'delta_buy_ratio', 'log_return_3GAP', 'momentum', 'momentum_decay', 'impact_cost', 'order_diff_growth', 'order_depth_change'] 

        ans = resampled[cols].copy()

        return ans
    







