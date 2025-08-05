# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
from joblib import Parallel, delayed
from datetime import datetime, timedelta, time
import os
import pickle
import matplotlib.pyplot as plt 
from xy.data.query import load_lv2



##############PART 1:Get All the trading days from 2020 to 2024

trade_dates = pd.read_csv('TRADINGDAYS.csv', index_col = 0)
trade_dates.reset_index(drop = True)
trade_dates['Date'] = trade_dates['Date'].astype(str)

##############PART 2: GET All the Ever LIMIT_UP Tickers and their information

LIMIT_UPs = pd.read_csv('Tickers_LIMITUP_TIME.csv')
LIMIT_UPs['LIMIT_UP_TIME'] = pd.to_datetime(LIMIT_UPs['LIMIT_UP_TIME'])
LIMIT_UPs['BREAK_LIMIT_UP_TIME'] = pd.to_datetime(LIMIT_UPs['BREAK_LIMIT_UP_TIME'])
LIMIT_UPs['Date'] = LIMIT_UPs['Date'].astype(str) 

##############PART 3: GET All the Transaction Data of convertible bonds and stocks saved in pickles splitted by trade_date

class GetPickle:
    """
    This class read several Transe-level-2 transaction data of convertible bonds saved in pickles
    Parameters:
    - df: dataframe contains all the trading/target dates string like '20240102'
    - base_path: reading dir for pickles
    - tickertype: 'stock' or 'ticker'
    Returns:
    - result_dict: keys are trading date strings, values are corresponding transaction dataframe of stock or ticker within the day
    """
    def __init__(self, df, base_path, tickertype):
        self.base_path = base_path
        self.tickertype = tickertype
        self.df = df

    def split_tickers(self, data_dict):
        keys = list(data_dict.keys())
        xlist = []
        for t in keys:
            temp_df = pd.DataFrame(data_dict[t])
            temp_df['Ticker'] = t
            xlist.append(temp_df) 

        df = pd.concat(xlist, axis = 0)
        df['date_time'] = pd.to_datetime(df['date_time'])
        df.sort_values(by = ['Ticker','date_time'], inplace = True)
        df.reset_index(drop = True, inplace = True)
    
        cols = ['Ticker'] + [col for col in df.columns if col != 'Ticker']
        return df[cols]

    def get_every_date(self, trade_date_str):
        ticker_path = os.path.join(self.base_path, f'Transe_{self.tickertype}_{trade_date_str}.pkl')
        temp_dict = pd.read_pickle(ticker_path)
        temp_df = self.split_tickers(temp_dict)
        return temp_df     

    def get_pickles(self):

        unique_dates = self.df['Date'].drop_duplicates()

        resultx = Parallel(n_jobs=-1)(
            delayed(self.get_every_date)(date_str) for date_str in unique_dates
        )
        resultx_dict = {
            date_str: res for date_str, res in zip(unique_dates, resultx) } 
        
        return resultx_dict



##############PART 4: GET the LIMIT_UP Information of Convertible Bonds

class GET_LIMIT_UP:
    """
    This class can obtain the basic information about limit_ups: LIMIT_UP_TIME, BREAK_LIMIT_UP_TIME, LIMIT_UP_PRICE of stock
    Parameters:
    - data_dict: dict of transaction data of convertible bonds on different trading date, dict.keys are date strings, like '20240102'
    Returns:
    - resultx_dict: dict of Limit_UPs information of convertible bonds on different trading date strings
    """
    def __init__(self, data_dict, trade_dates_list, tickertype):
        self.data_dict = data_dict
        self.trade_dates_list = trade_dates_list
        self.tickertype = tickertype

    def every_date_limitup(self, trade_date_str, df):

        ### Step1: get the tickers traded at trade_date_str
        tickers = df['Ticker'].drop_duplicates().tolist()
        price_dict = {}
        entrytime_dict = {}
        exittime_dict = {}    
    
        ### Step2: obtain the index of target trading date and the date before it 
        target_index = self.trade_dates_list.index(trade_date_str) 
        prev_trade_date = self.trade_dates_list[target_index - 1] if target_index > 0 else None

        if prev_trade_date is not None:
            prev_dff = load_lv2('Transe', prev_trade_date, market = self.tickertype)

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
        result['Date'] = trade_date_str

        return result    

    def get_limit_ups(self):

        unique_dates = list(self.data_dict.keys())
        resultx = Parallel(n_jobs=-1)(
            delayed(self.every_date_limitup)(date_str, self.data_dict[date_str]) for date_str in unique_dates
        )
        resultx_dict = {
            date_str: res for date_str, res in zip(unique_dates, resultx) } 
        
        return resultx_dict
  

##############PART 5: GET the Factors Information of Convertible Bonds

class Factors:
    """
    This class 'Factors' calculates various factors and indicators based on time-series market data.
    Parameters:
    - data_dict: dict of transaction data of convertible bonds on different trading date, dict.keys are date strings, like '20240102'
    - limit_ups: dataframe which saves the information about: Ticker, Stock, limit_up_time, break_limit_up_time, date
    - Win: Rolling window size for calculating indicators
    - GAP: Time interval for resampling, default '1min'
    - FixTime: Get Factors of all day or Factors at special time(limit_up/break_limit_up time), default = True
    """
    def __init__(self, data_dict, limit_ups, Win, GAP, FixTime):
        self.limit_ups = limit_ups
        self.Win = Win
        self.GAP = GAP
        self.FixTime = FixTime

    def get_factors(self, group):
        """
        - group: the slice of transaction data on the target date, split by 'Ticker'
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
            print('No data after limit_up_time,Type 1 Error')
        
        else:   
            try: 
                entry_idx_min = after_limitup_dt.abs().idxmin()
                entry_price = dt.loc[entry_idx_min, 'TradePrice']
                entry_amount = dt.loc[entry_idx_min, 'TradeAmount']
        
            except (ValueError, KeyError, IndexError):
                print('Type 2 Error, something wrong in index or key-value')
    
        ################################ 准备 Resample 数据  ################################
        dt.set_index('date_time', inplace = True)
        resampledx = dt.resample(self.GAP).agg(
            {
                'TradePrice': 'last',
                'TradeVolume': 'sum',
                'TradeAmount': 'sum',
                'AskOrder': 'sum',
                'BidOrder': 'sum',
                'BSFlag': lambda x: x.value_counts().to_dict()
            })

        ######## 1. 收集窗口时间内的成功的交易笔数以及买卖交易笔数 ############
        buy_counts = dt[dt['BSFlag'] == 'B'].resample(self.GAP).size() #########3 但是这里索引会出现错位，所以后面紧接着要对它进行调整
        sell_counts = dt[dt['BSFlag'] == 'S'].resample(self.GAP).size() #########3 但是这里索引会出现错位，所以后面紧接着要对它进行调整
    
        ## Adjustment For BUY/SELL
        # 统一索引，填充缺失值为0
        trade_count_per_sec = dt.resample(self.GAP).size().rename('TradeCount')
        buy_counts_per_sec = buy_counts.reindex(trade_count_per_sec.index, fill_value=0).rename('BuyCount')
        sell_counts_per_sec = sell_counts.reindex(trade_count_per_sec.index, fill_value=0).rename('SellCount')    
    
        ## Concat All the New features: TradeCount, BuyCount, SellCount
        resampledy = resampledx.merge(trade_count_per_sec, left_index=True, right_index=True, how='left')
        resampledz = resampledy.merge(buy_counts_per_sec, left_index=True, right_index=True, how='left')
        resampled = resampledz.merge(sell_counts_per_sec, left_index=True, right_index=True, how='left')
    
        ######### 2. rolling平滑获取交易额变化趋势，这些才有预测未来的能力 ########### 
        resampled['trade_amount_rolling_5'] = resampled['TradeAmount'].rolling(window=self.Win, min_periods=1).mean() #其中函数参数Win代表统一平滑的窗口长度
        resampled['trade_amount_growth'] = (resampled['trade_amount_rolling_5'].pct_change()).replace([np.inf, -np.inf], np.nan)  
     
        # 先计算每个窗口的资金变动斜率
        resampled['trade_amount_slope'] = resampled['TradeAmount'].rolling(self.Win).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]) 
        # 再计算斜率的差分，近似加速度
        resampled['amount_slope_growth'] = (resampled['trade_amount_slope'].pct_change()).replace([-np.inf, np.inf], np.nan)

        if entry_amount is not None:
            resampled['amount_rel_entry'] = (np.log(resampled['trade_amount_rolling_5']) - np.log(entry_amount)).replace([np.inf, -np.inf], np.nan)  
        else:
            resampled['amount_rel_entry'] = None
            print('Attention Please !!! Trading Amount at limit_up_time is None, there are something wrong')    
    
        ######### 3. rolling平滑获取交易笔数变化趋势，这些才有预测未来的能力 ###########
        resampled['trade_count_rolling_5'] = resampled['TradeCount'].rolling(window = self.Win, min_periods=1).mean() 
        resampled['trade_count_growth'] = (resampled['trade_count_rolling_5'].pct_change()).replace([np.inf, -np.inf], np.nan)
    
        # 先计算每个窗口的资金变动斜率
        resampled['trade_count_slope'] = resampled['TradeCount'].rolling(self.Win).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]) 
        # 再计算斜率的差分，近似加速度
        resampled['count_slope_growth'] = (resampled['trade_count_slope'].pct_change()).replace([-np.inf, np.inf], np.nan) 


        ######### 4. 计算窗口时间段内买卖力量的差距，反映市场多空力量的差异 ###########
        resampled['buy_ratio'] = resampled['BuyCount']/(resampled['BuyCount'] + resampled['SellCount']+1e-6) ##比值越接近1说明多方强，越接近0说明空方占优
        resampled['delta_buy_ratio'] = (resampled['buy_ratio'].pct_change()).replace([np.inf, -np.inf], np.nan)


        ######### 5. 动态收益率及其相关特征，比如变化动量之类的 ###########
        resampled['log_return_3GAP'] = (np.log(resampled['TradePrice']) - np.log(resampled['TradePrice'].shift(3))).replace([np.inf, -np.inf], np.nan) # 原来是3
        resampled['momentum'] = resampled['log_return_3GAP'].rolling(window=self.Win, min_periods=1).mean() ######### momentum是log_return的平滑均值结果                         
        resampled['momentum_decay'] = resampled['momentum'].diff()                          


        ######### 6. 买卖盘特征：当前买盘 - 卖盘差 #########
        resampled['order_depth_diff'] = (resampled['BidOrder'] - resampled['AskOrder'])/(resampled['BidOrder'] + resampled['AskOrder']+1e-6)
        resampled['order_diff_growth'] = (resampled['order_depth_diff'].pct_change()).replace([np.inf, -np.inf], np.nan)
        # order_depth_diff_change：盘口反转迹象（即差值变化）
        resampled['order_depth_change'] = np.sign(resampled['order_depth_diff'] * resampled['order_depth_diff'].shift(1))
        ######### 7. impact_cost #########
        resampled['impact_cost'] = resampled['TradeAmount'] / (resampled['BidOrder'] + resampled['AskOrder'] + 1e-6)

    
        #############################
        if self.FixTime: 
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
                    ans1 = dfs.loc[rows, ['date_time'] + cols]

                    ans1['special_time'] = [LIMIT_UP_TIME, BREAK_LIMIT_UP_TIME]
                    new_cols = ['special_time', 'date_time'] + cols
                    ans = ans1[new_cols]
        
                except (ValueError, KeyError, IndexError):
                    print('Type 2 Error, something wrong in index or key-value')

            return ans

        else:
            cols = ['trade_amount_growth','trade_amount_slope', 'amount_slope_growth', 'amount_rel_entry', 'trade_count_growth', 'trade_count_slope', 'count_slope_growth',
    
                        'buy_ratio', 'delta_buy_ratio', 'log_return_3GAP', 'momentum', 'momentum_decay', 'impact_cost', 'order_diff_growth', 'order_depth_change'] 

            ans = resampled[cols].copy()

            return ans


    def every_date_factors(self, trade_date_str, df): 
        """
        - trade_date_str: trading date string
        - df: corresponding transaction dataframe on trade_date_str
        """
        print(f'Process Trading Date is : {trade_date_str}')
    
        ###### Here the df is NOT Stock, df IS convertible bond
        df = df.reset_index(drop=True)
        dff = df[(df['FunctionCode'] != 'C') & (df['TradeVolume'] > 0)].copy()
    
        df_limit_ups = self.limit_ups[self.limit_ups['Date'] == trade_date_str].copy()
    
        dfx = dff.merge(df_limit_ups[['Ticker', 'LIMIT_UP_TIME', 'BREAK_LIMIT_UP_TIME']], on = 'Ticker', how = 'left')
        dfx['LIMIT_UP_TIME'] = pd.to_datetime(dfx['LIMIT_UP_TIME'])
        dfx['BREAK_LIMIT_UP_TIME'] = pd.to_datetime(dfx['BREAK_LIMIT_UP_TIME'])
 
        ### result_df 是一个dataframe, 包含：Ticker, LIMIT_UP_TIME, entry_cb_price三列
        result_df = dfx.groupby('Ticker').apply(lambda group: self.get_factors(group)).reset_index()
    
        if not result_df.empty:
            print(f'GET Valid Result on trading date {trade_date_str}')
        else:
            print(f'Something Wrong happends on trading date {trade_date_str}')
        
        return result_df

    def factor_dicts(self):
        
        unique_dates = list(self.data_dict.keys())  
        resultx = Parallel(n_jobs=-1)(
            delayed(self.every_date_factors)(date_str, self.data_dict[date_str]) for date_str in unique_dates
        )
        resultx_dict = {
            date_str: res for date_str, res in zip(unique_dates, resultx) } 
        return resultx_dict
        


##############PART 7: Adjust the Exit Time with EntryTime and HoldTime

morning_start = pd.to_datetime("09:25:00").time()
morning_end = pd.to_datetime("11:30:00").time()

afternoon_start = pd.to_datetime("13:00:00").time()
afternoon_end = pd.to_datetime("15:00:00").time()

class ExitReturns:
  """  
    Class to compute exit returns, drawdowns, and price trajectories for convertible bonds.  
    
    Parameters:  
    - data_dict: dict, transaction data indexed by date, each value is a DataFrame with trade info.  
    - TimeList: list of int, minute offsets from entry time to adjust exit times.  
    - limit_ups: DataFrame, containing 'LIMIT_UP_TIME' and 'BREAK_LIMIT_UP_TIME' for each bond.  
  """ 
    
    def __init__(self, data_dict, TimeList, limit_ups):
        self.data_dict = data_dict
        self.TimeList = TimeList
        self.limit_ups = limit_ups

    
    def adjust_exit_time(self, entry_cb_time, x):
        ### x可以是负数，相应地往前平移一段时间
        entry_cb_time_x = pd.to_datetime(entry_cb_time)
        exit_time = entry_cb_time_x + timedelta(seconds=x*60)
    
        exit_dt = pd.Timestamp(exit_time)
        morning_end_dt = exit_dt.replace(hour=11, minute=30, second=0, microsecond=0)
    
        ##分区间讨论
        if morning_end < exit_dt.time() < afternoon_start:
            overshoot_seconds = (exit_dt - morning_end_dt).total_seconds()
            exit_time = exit_dt.replace(hour=13, minute=0, second=0, microsecond=0) + timedelta(seconds=overshoot_seconds)
        
        elif exit_dt.time() > afternoon_end:  
            exit_time = exit_dt.replace(hour=15, minute=0, second=0, microsecond=0)
        
        elif  exit_dt.time() < morning_start:
            exit_time = exit_dt.replace(hour=9, minute=25, second=0, microsecond=0)
        
        return exit_time


    def get_price_at_time(self, group, Time): 
        # 要确保Time和'date_time'的数据类型都是datetime
    
        tempt = group[(group['FunctionCode'] != 'C') & (group['TradeVolume']>0)].copy()
        time_dff = pd.Series(tempt['date_time'] - Time)
    
        if time_dff.empty:
            price = None
            print('time_diff is none')
        else:
            try:
                time_idx_min = time_dff.abs().idxmin()
                price = tempt.loc[time_idx_min, 'TradePrice']     
            
            except (ValueError, KeyError, IndexError):
                price = None
                print('second error')
    
        return price             


    def get_drawdown(self, group, entry_cb_time, exit_time):
    
        xy_temp = group.copy()
        xy_slice = xy_temp[(xy_temp['FunctionCode'] != 'C')& (xy_temp['TradeVolume']>0) &
            (xy_temp['date_time'] >= entry_cb_time) & (xy_temp['date_time'] <= exit_time)].copy()
    
        xy_slice['peak'] = xy_slice['TradePrice'].cummax()
        xy_slice['drawdown'] = (xy_slice['peak'] - xy_slice['TradePrice']) / xy_slice['peak']
    
        ### 最大回撤与最大回撤发生的时刻
        max_dd = xy_slice['drawdown'].max()*100 # 把它转变成百分数
        #max_dd_time = xy_slice.loc[xy_slice['drawdown'].idxmax(), 'datetime']
    
        return max_dd 

    def compute_prices(self, group):
    
        ######## 确认转债的入场时间，也即是对应正股的涨停时间，并计算转债入场时的入场价格
        limit_up_time = pd.to_datetime(group['LIMIT_UP_TIME'].iloc[0])
        break_cb_time = pd.to_datetime(group['BREAK_LIMIT_UP_TIME'].iloc[0])
    
        entry_cb_time = limit_up_time
        after_limitup_diff = pd.Series(group['date_time'] - limit_up_time)
    
        if after_limitup_diff.empty:
            entry_cb_price = None
            print('No data after limit_up_time, entry_cb_price is none')
        
        else:
            try:
                #### Calculate The enter Price(买入转债的时间）
                entry_idx_min = after_limitup_diff.abs().idxmin()
                entry_cb_price = group.loc[entry_idx_min, 'TradePrice']  
                
            except (ValueError, KeyError, IndexError):
                # 发生异常时，也将price设为None
                entry_cb_price = None
                print('Type 2 Error, entry_cb_price is none')

        ######## 获取其余离场时间对应的离场价格
        
        ExitTimeList_raw = [ self.adjust_exit_time(entry_cb_time, x) for x in self.TimeList] 
    
        close_time = pd.Timestamp(entry_cb_time).replace(hour=15, minute=0, second=0, microsecond=0)
    
        ExitTimeList = ExitTimeList_raw + [break_cb_time] + [close_time]
        
        ExitPriceList = [ self.get_price_at_time(group, t) for t in ExitTimeList ]
        
        DrawdownList = [self.get_drawdown(group, entry_cb_time, t) for t in ExitTimeList]
        
        #### 对应顺序： TimeList（这个是一部分列名字）, ExitTimeList, ExitPriceList
        
        Prices = [entry_cb_price] + ExitPriceList
        Times = [entry_cb_time] + ExitTimeList
    
        LogReturns = [
            (np.log(p) - np.log(entry_cb_price)) if p is not None and entry_cb_price is not None and entry_cb_price != 0 else None
            for p in Prices]  
    
        Drawdowns = [0] + DrawdownList
        Re_TIMELIST = ['entry'] + [str(item) for item in self.TimeList] + ['break'] + ['close']
        
        ans = pd.DataFrame({
            'HoldTime': Re_TIMELIST,
            'Time': Times,
            'TradePrice': Prices,
            'LogReturn': LogReturns,
            'Drawdown': Drawdowns
            })

    def every_date_returns(self, trade_date_str, df):
        
        print(f'Process Trading Date is : {trade_date_str}')
    
        df = df.reset_index(drop=True)
        ###### Here the df is NOT Stock, df IS convertible bond
        dff = df[(df['FunctionCode'] != 'C') & (df['TradeVolume'] > 0)].copy()
        df_limit_ups = self.limit_ups[self.limit_ups['Date'] == trade_date_str].copy()
    
        dfx = dff.merge(df_limit_ups[['Ticker', 'LIMIT_UP_TIME', 'BREAK_LIMIT_UP_TIME']], on = 'Ticker', how = 'left')
        dfx['LIMIT_UP_TIME'] = pd.to_datetime(dfx['LIMIT_UP_TIME'])
        dfx['BREAK_LIMIT_UP_TIME'] = pd.to_datetime(dfx['BREAK_LIMIT_UP_TIME'])
 
        ### result_df 是一个dataframe, 包含：Ticker, LIMIT_UP_TIME, entry_cb_price三列
        result_df = dfx.groupby('Ticker').apply(lambda group: self.compute_prices(group)).reset_index()
    
        if not result_df.empty:
            print(f'GET Valid Result on trading date {trade_date_str}')
        else:
            print(f'Something Wrong happends on trading date {trade_date_str}')
        
        return result_df

    def get_returns(self):
        
        unique_dates = list(self.data_dict.keys())  
        resultx = Parallel(n_jobs=-1)(
            delayed(self.every_date_returns)(date_str, self.data_dict[date_str]) for date_str in unique_dates
        )          
        resultx_dict = {
            date_str: res for date_str, res in zip(unique_dates, resultx) } 
        
        return resultx_dict
        

##############PART 11: Get LagReturns prepared for forecasting

def LAG_Returns(dt_ticker, Lag, LIMIT_UP_TIME, GAP = '1min'):
    
    ######### Attention ! 目前LIMIT_UP_TIME和Lag在这里被使用
    ###### 得到延迟一定时间后的收益/价格的变动走势
    dt = dt_ticker[(dt_ticker['FunctionCode'] != 'C') & (dt_ticker['TradeVolume'] > 0)].copy()
    dt.sort_values(by = 'date_time', inplace = True)
    
    
    ###### GET T0 every usseful data 
    LIMIT_UP_TIME = pd.to_datetime(LIMIT_UP_TIME)
    after_limitup_diff = pd.Series(dt['date_time'] - LIMIT_UP_TIME)
    
    if after_limitup_diff.empty:
    
        entry_cb_price = None
        print('No data after limit_up_time,Type 1 Error')
        
        resampled = pd.DataFrame()
        
    else:
        try:
            
            #### Calculate The enter Price(买入转债的时间）
            entry_idx_min = after_limitup_diff.abs().idxmin()
            entry_cb_price = dt.loc[entry_idx_min, 'TradePrice']
            
            #### Calculate the log return from limit_up_time
            dt.set_index('date_time', inplace = True)
            resampled = dt.resample(GAP).agg(
                  {
                    'TradePrice': 'mean'
                  })
            
            ######## 其实我是存疑的
            resampled['log_return_min'] = np.log(resampled['TradePrice']) - np.log(entry_cb_price)  #.shift(-Lag)原先再TradePrice先做了一个延迟，做未来的收益
            
            before_entry = resampled.index < LIMIT_UP_TIME
            resampled.loc[before_entry, 'log_return_min'] = -resampled.loc[before_entry, 'log_return_min']
            
            resampled['log_return'] = resampled['log_return_min'].rolling(window = 5, min_periods = 1).mean()
            
            resampled['cumulative_log_return'] = resampled['log_return'].dropna().cumsum()
            
        except (ValueError, KeyError, IndexError):
                
            entry_cb_price = None
            print('Type 2 Error, something wrong in index or key-value')
    
    
    return resampled


##############PART 12: Get correlation coefficients of Factors and Lag_Returns to test the quality of factors

def GET_IC(df_factor, df_return):

      common_valid_idx = (df_factor[df_factor.notna()].index).intersection(df_return[df_return.notna()].index)
      
      valid_factor = df_factor.loc[common_valid_idx]
      valid_return = df_return.loc[common_valid_idx]
      
      IC,pval = spearmanr(valid_factor, valid_return)

      return pd.Series({'IC': IC, 'P-Value': pval})


##############PART 13: Calculate the ICs of factors and order by the coefficients

def ParaSelection(dt_ticker,LIMIT_UP_TIME, GAP, Lag, Win):

    df_factors = Factors(dt_ticker, LIMIT_UP_TIME, Win ,GAP )
    df_returns = LAG_Returns(dt_ticker, Lag , LIMIT_UP_TIME , GAP)
    
    res = df_factors.apply(lambda col: GET_IC(col, df_returns['log_return']), axis=0)
    res = res.T
    
    res.index = ['trade_amount_growth', 'trade_amount_slope', 'amount_slope_growth', 'amount_rel_entry', 'trade_count_growth', 
    'trade_count_slope', 'count_slope_growth', 'buy_ratio', 'delta_buy_ratio', 'log_return_3GAP', 'momentum',
    'momentum_decay', 'impact_cost', 'order_depth_diff', 'order_diff_growth']

    res.sort_values(by = 'IC', inplace = True)
    
    return res

##############PART 14: visualize the changing 'log-return' of one event(fixed ticker and fixed trade_date)  

def ReturnDist(df_return, LIMIT_UP_TIME, BREAK_LIMIT_UP_TIME):

    resampled = df_return.copy()
    resampled.set_index('date_time', inplace = True)
    resampled.index = pd.to_datetime(resampled.index)
    
    # 设置绘图为上下堆叠
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 上面绘制对数收益率的变化
    ax1.plot(resampled.index, resampled['log_return'], label='Log Return')
    ax1.axvline(LIMIT_UP_TIME, color='red', linestyle='--', label='LIMIT UP Time')
    ax1.axvline(BREAK_LIMIT_UP_TIME, color='green', linestyle='--', label='BREAK_LIMIT UP Time')
    ax1.set_ylabel('Log Return')
    ax1.set_title('Log Return of Convertible Bonds over Time')
    ax1.grid(True)
    ax1.legend()

    # 下面绘制累计收益CAR的变化
    ax2.plot(resampled.index, resampled['cumulative_log_return'], label='Cumulative Log Return', color='orange')
    ax2.axvline(LIMIT_UP_TIME, color='red', linestyle='--', label='LIMIT UP Time')
    ax2.axvline(BREAK_LIMIT_UP_TIME, color='green', linestyle='--', label='BREAK_LIMIT UP Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cumulative Log Return')
    ax2.set_title('Cumulative Log Return over Time')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


##############PART 15: visualize the changing 'log-return' of fixed trade_date

def GET_Dist_EveryDate(trade_date_str, df, limit_ups):

    df = df.reset_index(drop=True)
    dff = df[(df['FunctionCode'] != 'C') & (df['TradeVolume'] > 0)].copy()
    df_limit_ups = limit_ups[limit_ups['Date'] == trade_date_str].copy()
    
    dfx = dff.merge(df_limit_ups[['Ticker', 'LIMIT_UP_TIME', 'BREAK_LIMIT_UP_TIME']], on = 'Ticker', how = 'left')
    dfx['LIMIT_UP_TIME'] = pd.to_datetime(dfx['LIMIT_UP_TIME']) 
    dfx['BREAK_LIMIT_UP_TIME'] =  pd.to_datetime(dfx['BREAK_LIMIT_UP_TIME']) 
    
    
    df_returns = dfx.groupby('Ticker').apply(lambda group: LAG_Returns(dt_ticker = group, Lag = 0, LIMIT_UP_TIME = group['LIMIT_UP_TIME'].iloc[0],
                         GAP = '1min')).reset_index()
    print(df_returns.columns)                   
    Tickers = dfx['Ticker'].drop_duplicates().tolist()
    print(Tickers)
    
    for t in Tickers:
    
        df_return = df_returns[df_returns['Ticker'] == t].copy()
        LIMIT_UP_TIME = df_limit_ups[df_limit_ups['Ticker'] == t]['LIMIT_UP_TIME'].iloc[0]
        BREAK_LIMIT_UP_TIME = df_limit_ups[df_limit_ups['Ticker'] == t]['BREAK_LIMIT_UP_TIME'].iloc[0]
    
        print(df_return.head(5))
        print(LIMIT_UP_TIME)
        print(BREAK_LIMIT_UP_TIME)
    
        ReturnDist(df_return, LIMIT_UP_TIME, BREAK_LIMIT_UP_TIME)
                         

##############PART 16: visualize the hist distribution of LogReturns of different HoldTime

def Dist(data_list, title_list):
    """
    Parameter:
    - data_list: LogReturns dataframe of several HoldTime
    - title_list: HoldTime Labels

    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # 存储最高频率点的收益值和频率，用于连线
    peaks = []

    for i, data in enumerate(data_list):
        ax = axes[i]
        # 绘制直方图
        counts, bins, patches = ax.hist(data, bins=100, alpha=0.7, color='skyblue')
    
        # 找到最高频率点的索引
        max_idx = np.argmax(counts)
        peak_freq = counts[max_idx]
        peak_bin_center = (bins[max_idx] + bins[max_idx + 1]) / 2    
        
        # 标注最高频率点，使用绿色

        ax.axvline(peak_bin_center, color='green',alpha=1.0, linestyle='--', lw=1.0, label='Mode Log Return')            

        # 画平均值直线，颜色改为蓝色
        mean_value = data.mean()
        ax.axvline(mean_value, color='red', alpha=0.6, linestyle='--', lw=1.0 ,label='Mean Log Return')

        # 在右上角添加文本显示平均值
        ax.text(0.95, 0.95, f'Mean: {mean_value:.4f}',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                
        ax.text(0.95, 0.9, f'Mode: {peak_bin_center:.4f}',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))        

    
        # 记录最高点，用于连线
        peaks.append((peak_bin_center, peak_freq))
    
        ax.set_title(title_list[i])
        ax.set_xlabel(f'{title_list[2][6:]}')
        ax.set_ylabel('Frequency')
        ax.grid(True)

    plt.tight_layout()
    plt.show()


##############PART 17: T test of LogReturns of one HoldTime strategy
def T_test(df_return):

    t_stat, p_value = stats.ttest_1samp(df_return, 0)
    
    df = len(df_return) - 1  
    
    if t_stat > 0:
        p_one_sided = 1 - stats.t.cdf(t_stat, df)
    else:
        p_one_sided = stats.t.cdf(t_stat, df)
        
    return t_stat, p_one_sided    



