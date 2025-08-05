# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np 
from datetime import datetime, time, timedelta 
import os 
import pickle 
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import GET_STOCK_CB
from GET_STOCK_CB import 


##########################################################PART 1: 
m_start = datetime.time(9, 25)
m_end = datetime.time(11, 30)
a_start = datetime.time(13, 0)
a_end = datetime.time(15, 0)

##########基础函数1

def adjust_break_time(break_time):
    
    # 先获取其时间部分
    t = break_time.time()
    # 判断是否在上午时间段
    if m_start <= t <= m_end:
        return break_time
    # 判断是否在下午时间段
    elif a_start <= t <= a_end:
        return break_time
    else:
        # 不在允许时间段，调整到下一个时间段
        if t > m_end and t < a_start:
        # 如果在上午时间之后，但未到下午范围，可能在11:30-13:00之间，调到下午开始
            return break_time.replace(hour=13, minute=0, second=0)
            
        else:
            return break_time.replace(hour=15, minute=0, second=0)
            
##########基础函数2
def ExitSingal(after_limitup_df, threshold, weights, smooth_window=4):
    # 所需的字段
    required_cols = [
        'trade_rate_amount_2T', 'trade_price_2T', 'trade_rate_count_2T',
        'log_return_6mins', 'log_volatility', 'log_sharpe', 'momentum', 'momentum_decay',
        'order_depth_diff', 'order_diff_trend', 'order_depth_change'
    ]
    
    
    # 检查缺失列
    missing_cols = [col for col in required_cols if col not in after_limitup_df.columns]
    if missing_cols:
        raise ValueError(f"Missing Columns: {missing_cols}")
    
    # 筛出非空行
    valid_idx = after_limitup_df[required_cols].dropna().index

    if len(valid_idx) == 0:
        # 所有信号都为NaN，无法处理，直接返回原始数据并提示
        print("Warning: All signal rows are NaN-skip exit scoring.")
        after_limitup_df['exit_score'] = np.nan
        after_limitup_df['exit_score_smooth'] = 0
        return after_limitup_df

    # 提取信号
    signal1 = after_limitup_df.loc[valid_idx, ['trade_rate_amount_2T', 'trade_price_2T', 'trade_rate_count_2T']]
    signal2 = after_limitup_df.loc[valid_idx, ['log_return_6mins', 'log_sharpe', 'momentum', 'momentum_decay']]
    signal3 = after_limitup_df.loc[valid_idx, ['order_depth_diff', 'order_diff_trend']]
    signal4 = after_limitup_df.loc[valid_idx, 'log_volatility'].values.reshape(-1, 1)
    signal5 = after_limitup_df.loc[valid_idx, 'order_depth_change'].values.reshape(-1, 1)
    
    
    

    # 标准化
    try:
        scaler1 = StandardScaler().fit(signal1)
        scaler2 = StandardScaler().fit(signal2)
        scaler3 = StandardScaler().fit(signal3)
        scaler4 = StandardScaler()

        scaled1 = scaler1.transform(signal1)
        scaled2 = scaler2.transform(signal2)
        scaled3 = scaler3.transform(signal3)
        scaled4 = scaler4.fit_transform(signal4)
        
    except Exception as e:
        print(f"Error in StandardScaler: {e}")
        after_limitup_df['exit_score'] = np.nan
        after_limitup_df['exit_score_smooth'] = 0
        return after_limitup_df

    # 打分逻辑
    exit_score = (
        -weights[0] * scaled1[:, 0] +
        -weights[1] * scaled1[:, 1] +
        -weights[2] * scaled1[:, 2] +
        -weights[3] * scaled2[:, 0] +
        -weights[4] * scaled2[:, 1] +
        -weights[5] * scaled2[:, 2] +
        -weights[6] * scaled2[:, 3] +
        -weights[7] * scaled3[:, 0] +
        -weights[8] * scaled3[:, 1] +
         weights[9] * scaled4[:, 0] +
        -signal5[:,0]
    )

    after_limitup_df.loc[valid_idx, 'exit_score'] = exit_score

    # 生成平滑信号列（> threshold 为触发信号）
    signal_flag = (after_limitup_df['exit_score'] >= threshold).astype(float)
    after_limitup_df['exit_score_smooth'] = (
        signal_flag.rolling(window=smooth_window, min_periods=2).sum().fillna(0)
    )

    return after_limitup_df
  

################################

def SIMULATE_ONE_TRADE_SIGNAL(limitup_time, stock_df, cb_df, cb_resampled, thresholdx, weight_lists, smooth_window, trigger_times, MIN_HOLD):

    morning_start = datetime.time(9, 25)
    morning_end = datetime.time(11, 30)
    afternoon_start = datetime.time(13, 0)
    afternoon_end = datetime.time(15, 0)
    
    
    if cb_df is None or cb_resampled is None or cb_df.empty or cb_resampled.empty :
    
        print('No valid Data')
        ans = pd.DataFrame()
    
    else:    
    
        ########### 确定入场时间 Entry Time
        entry_cb_time = pd.to_datetime(limitup_time)
        
        ######### Find the entry_cb_price 入手转债时的价格
        cb_time_diff = pd.Series(cb_df['date_time'] - entry_cb_time)
        if cb_time_diff.empty:
            entry_cb_price = None

        else:
            try:
                idx_min = cb_time_diff.abs().idxmin()
                entry_cb_price = cb_df.loc[idx_min, 'TradePrice']
                
            except ValueError:
                # 发生异常时，也将 price 设为 None
                entry_cb_price = None

        
        ######### 找出正股炸板离开涨停板的时间 Break the Upper Limit Exit Time
        
        # 设定容忍时间（炸板 < 2分钟的不算）
        TOLERANCE_MINUTES = 2
        TOLERANCE_SECONDS = TOLERANCE_MINUTES * 60

        
        # 找出涨停价（可加精度控制）
        limit_up_price = stock_df[(stock_df['date_time'] >= entry_cb_time)]['TradePrice'].max()
        
        # 添加标志：是否低于涨停价（视为炸板）
        stock_df_x = stock_df.copy()
        stock_df_x['is_break'] = stock_df_x['TradePrice'] < (limit_up_price - 1e-6)
        
        # 只分析 entry_cb_time 之后的数据
        after_entry = stock_df_x[stock_df_x['date_time'] >= entry_cb_time].copy()
        
        # 标记是否炸板段的开始和结束（用分组编号分段）
        after_entry['break_shift'] = after_entry['is_break'].shift(1, fill_value=False)
        after_entry['break_group'] = (after_entry['is_break'] & ~after_entry['break_shift']).cumsum()

        # 提取所有炸板段
        break_groups = after_entry[after_entry['is_break']].groupby('break_group')

        break_time = stock_df['date_time'].max()

        for _, group in break_groups:
            start_time = group['date_time'].iloc[0]
            end_time = group['date_time'].iloc[-1]
            duration = (end_time - start_time).total_seconds()
        
            if duration >= TOLERANCE_SECONDS:
                break_time = start_time
                break  
        
        print(f'The real time point to leave the upper limit is:{break_time}')
        
        ################################################################## 按照当天收盘时刻退场的收益率        
        exit_price_close = cb_df['TradePrice'].iloc[-1] if not cb_df.empty else np.nan
            
        if entry_cb_price is None:
            pnl_close = np.nan
        else:
                
            pnl_close = (exit_price_close - entry_cb_price) / entry_cb_price
        
        ################################################################## 以正股炸盘为退场信号的收益率 
        
        # 设定最短持仓时间
        MIN_HOLD_SECONDS = MIN_HOLD * 60
        
        # 在开始之前，先定义为收盘退场时间，避免未被赋值
        exit_price_break = exit_price_close
        pnl_break = pnl_close

        if pd.notnull(break_time):
            # 计算持股时间
            hold_duration = (break_time - entry_cb_time).total_seconds()
            
            if hold_duration < MIN_HOLD_SECONDS:
            
                break_time_raw = entry_cb_time + timedelta(seconds=MIN_HOLD_SECONDS)
                break_time = adjust_break_time(break_time_raw)
                print(f"Time holding the Ticker is {MIN_HOLD_SECONDS} seconds, then change the Breakleave time to {break_time}")

            try:
                # 试图获取符合条件的交易价格
                exit_price_series = cb_df[cb_df['date_time'] >= break_time]['TradePrice']
                
                if not exit_price_series.empty:
                    exit_price_break = exit_price_series.iloc[0]
                else:
                # 没有满足条件的数据
                    exit_price_break = exit_price_close
                # 计算盈亏
                if entry_cb_price is None:
                    pnl_break = np.nan
                else:
                    pnl_break = (exit_price_break - entry_cb_price) / entry_cb_price
            except:
            # 捕获指数错误或其他异常，确保变量已定义
                exit_price_break = exit_price_close
                pnl_break = pnl_close
        else:
        # 没有break_time或为null的情况
            exit_price_break = exit_price_close
            pnl_break = pnl_close
                
        
        ################################################################## 以多因子信号为触发退场条件的收益率        
        cb_resampled.reset_index(inplace = True)
        cb_resampled['date_time'] = pd.to_datetime(cb_resampled['date_time'], errors='coerce')
        
        ####### 时间限制及其筛选
        time_part = cb_resampled['date_time'].dt.time
        
        time_condition = (
          ((time_part >= morning_start) & (time_part <= morning_end)) |
          ((time_part >= afternoon_start) & (time_part <= afternoon_end)))

        entry_cb_time = pd.to_datetime(limitup_time)
        after_limitup_df_raw = cb_resampled[ (cb_resampled['date_time'] >= entry_cb_time) &time_condition].copy()

        # 入场前的成交量交易量基准 #
        # 检查时间差序列是否为空
        resampled_time_diff = pd.Series(cb_resampled['date_time'] - entry_cb_time)
        
        if resampled_time_diff.empty:
            
            entry_cb_amount = cb_resampled['TradeAmount'].iloc[0]
            entry_cb_count =  cb_resampled['TradeCount'].iloc[0]


        else:
            try:
                min_idx = resampled_time_diff.abs().idxmin()
                entry_cb_amount = cb_resampled.loc[min_idx, 'TradeAmount']
                entry_cb_count = cb_resampled.loc[min_idx, 'TradeCount']
                
            except ValueError:
                # 发生异常时，也将 price 设为 None
                entry_cb_amount = cb_resampled['TradeAmount'].iloc[0]
                entry_cb_count =  cb_resampled['TradeCount'].iloc[0]
        
        # 添加确保最小持股时间MIN_HOLD_SECONDS
        after_limitup_df_raw['hold_seconds'] = (after_limitup_df_raw['date_time'] - entry_cb_time).dt.total_seconds()
        
        
        ########################## Step1: 计算基础信号的动量变化指标（已有）
        #after_limitup_df['order_depth_change'] = np.sign(after_limitup_df['order_depth_diff'] * after_limitup_df['order_depth_diff'].shift(1)) ### 买卖订单差是否出现反转
        #after_limitup_df['is_price_rolling_top'] = after_limitup_df['trade_price_30s'] >= after_limitup_df['price_rollmax'].shift(1)
        #after_limitup_df['trade_rate_amount_growth'] = after_limitup_df['trade_rate_amount_30s'].diff()

        
        after_limitup_df = ExitSingal(after_limitup_df_raw, threshold = thresholdx, weights = weight_lists , smooth_window= smooth_window)
        
        ############################ Step 4: 动量持仓保护机制
        after_limitup_df['keep_holding'] = (
          (after_limitup_df['log_return_6mins'] > 0.03) #& 
          #(after_limitup_df['trade_price_2T'] >= after_limitup_df['price_rollmax'].shift(1)) &
          #(after_limitup_df['is_price_rolling_top'] >= 0)
)

        
        final_exit_mask = ((
          (after_limitup_df['hold_seconds'].fillna(0) >= MIN_HOLD_SECONDS) &
          (after_limitup_df['exit_score_smooth'].fillna(0) >= trigger_times) & #########连续触发n次才算真正触发退出强烈信号
          (~after_limitup_df['keep_holding'].fillna(False))) ) #  | (after_limitup_df['order_depth_change'] < 0 )
                 
          #& 
          #(after_limitup_df['TradeAmount'].fillna(np.inf) <= entry_cb_amount) &
          #(after_limitup_df['TradeCount'].fillna(np.inf) <= entry_cb_count)
        #)

        
        temp_df = after_limitup_df[final_exit_mask].copy()        
        

        # 3. 找出满足退出信号的时间点和其对应收益率
        exit_time_signal = break_time
        
        if not temp_df.empty and temp_df['date_time'].notnull().all():
        
            exit_time_signal_raw = temp_df['date_time'].min()
            
            #if exit_time_signal_raw <= break_time:
            exit_time_signal = exit_time_signal_raw
            exit_price_signal = cb_df[cb_df['date_time'] >= exit_time_signal]['TradePrice'].iloc[0]
                
            #else:    
            #    exit_time_signal = break_time
            #    exit_price_signal = exit_price_break

            
            if entry_cb_price is None:
                pnl_signal = np.nan
            else:
                pnl_signal = (exit_price_signal - entry_cb_price) / entry_cb_price
            
        else:
            exit_price_signal = exit_price_break #np.nan
            pnl_signal = pnl_break
            
        print(f'The real time point to hit the factors threshold is:{exit_time_signal}')    
        ##########################


        ans = pd.DataFrame({
                'EntryTime': [entry_cb_time],
                'EntryPrice': [entry_cb_price],
                'BreakTime': [break_time],
                'ExitBreakPrice': [exit_price_break],
                'PnlBreakExit': [pnl_break],
                'SignalTime': [exit_time_signal],
                'ExitSignalPrice':[exit_price_signal],
                'PnlSignalExit':[pnl_signal],
                'ExitClosePrice': [exit_price_close],
                'PnlCloseExit': [pnl_close]
            })
        
    return ans

        


        ########################## Step1: 构建单因子退出信号（趋势/反转）
#        after_limitup_df['exit_signal_1'] =(
#            (after_limitup_df['log_return_3mins'] < -0.03) &     ## 负收益
#            (after_limitup_df['log_return_diff'] < -0) 
#        )
        
#        after_limitup_df['exit_signal_2'] = (
#            (after_limitup_df['order_depth_change'] <= 0)  # 主买主卖反转    
#        )
        
#        after_limitup_df['exit_signal_3'] = (
#            (after_limitup_df['TradeCount'] < after_limitup_df['trade_rate_count_30s'] * 0.7)&  ## 收益率加速下跌
#            (after_limitup_df['trade_rate_amount_growth'] < 0) &  ## 量能减少
#            (after_limitup_df['trade_rate_amount_30s'] < 0.7*after_limitup_df['trade_rate_amount_30s'].rolling(4).mean() )
#            )
            
            
         
#        after_limitup_df['exit_score'] = (
#          1.0 * after_limitup_df['exit_signal_1'].astype(int) +
#          1.0 * after_limitup_df['exit_signal_2'].astype(int) +
#          1.0 * after_limitup_df['exit_signal_3'].astype(int))
        
        
#        after_limitup_df['exit_score_smooth'] = after_limitup_df['exit_score'].rolling(window=4, min_periods=3).sum()


        
#        after_limitup_df['keep_holding'] = (
#          (after_limitup_df['log_return_3mins'] > 0.01) & 
#          (after_limitup_df['trade_price_30s'] >= after_limitup_df['price_rollmax'].shift(1)) &
#          (after_limitup_df['trade_rate_amount_growth'] > 0) &
#          (after_limitup_df['is_price_rolling_top'] >= 0)
#)


        
        
#        final_exit_mask = (
#          (after_limitup_df['hold_seconds'] >= MIN_HOLD_SECONDS) &
#          (after_limitup_df['exit_score_smooth'] >= 2) &
#          (~after_limitup_df['keep_holding']) &
#          (after_limitup_df['TradeAmount'] <= entry_cb_amount) &
#          (after_limitup_df['TradeCount'] <= entry_cb_count)
#)
        
#        temp_df = after_limitup_df[final_exit_mask]




        ########################## Step2: 构建多因子退出信号（趋势/反转）
        #scaler = StandardScaler()
        #exit_signal1 = after_limitup_df[['trade_rate_amount_30s', 'trade_price_30s', 'trade_rate_count_30s']].copy().dropna()
        #exit_signal2 = after_limitup_df[['log_volatility_3mins', 'momentum', 'momentum_decay']].copy().dropna()
        #exit_signal3 = after_limitup_df[['order_depth_diff', 'order_diff_trend']].copy().dropna()
        
        #exit_signal_scaled1 = scaler.fit_transform(exit_signal1.values)
        #exit_signal_scaled2 = scaler.fit_transform(exit_signal2.values)
        #exit_signal_scaled3 = scaler.fit_transform(exit_signal3.values)


        # 你可以人为设置加权组合：
        #exit_score = (
        #        -0.3*exit_signal1[:, 0] +  # 成交额减少
        #        -0.5*exit_signal1[:, 1] +  # 价格走势变差
        #        -0.2*exit_signal1[:, 2] +  # 买单减少
        #        -0.3*exit_signal2[:, 0] +
        #        -0.4*exit_signal2[:, 1] +
        #        -0.3*exit_signal2[:, 2] +
        #        -0.6*exit_signal3[:, 0] +
        #        -0.4*exit_signal3[:, 1] +                
        #  )
           

        # 得分高于阈值即退出（注意这里越大越坏）
        #after_limitup_df.loc[after_limitup_df.index, 'exit_score'] = exit_score
        
        
        ############################ Step3: 平滑合成总信号
        # 平滑信号，避免误触发
        #after_limitup_df.loc[after_limitup_df.index, 'exit_score_smooth'] = (after_limitup_df['exit_score'] >= 2.5).rolling(window=4, min_periods=3).sum()








        #after_limitup_df['exit_signal'] = (
        #                ((after_limitup_df['log_return_3mins'] < -0.03) &
        #                 (after_limitup_df['TradeCount'] < after_limitup_df['trade_rate_count_30s'] * 0.3)) |
        #                (after_limitup_df['order_depth_diff'] * after_limitup_df['order_depth_diff'].shift(1) <= 0))

        #after_limitup_df['exit_signal_smooth'] = after_limitup_df['exit_signal'].rolling(window=4, min_periods=4).sum()
        #after_limitup_df['trade_rate_amount_growth'] = after_limitup_df['trade_rate_amount_30s'].diff()

        # 最终综合退出判断
        #final_exit_mask = (
        #        (after_limitup_df['hold_seconds'] >= MIN_HOLD_SECONDS) &  # 至少持仓360秒
         #         (after_limitup_df['exit_signal_smooth'] >= 2) & 
        #          (after_limitup_df['TradeAmount'] <= entry_cb_amount) & # 至少说明后续退出时的交易量应该比刚开始要少了，有大部分人已经退市
        #          (after_limitup_df['TradeCount'] <= entry_cb_count)
        #)       
               
    

        ############################# Step5: 最终退出信号判断，最终退出标准：至少持仓时间 + 信号强度足够 + 动量不足 + 成交量下降
        #final_exit_mask = (
        #  (after_limitup_df_with_exitsignals['hold_seconds'] >= MIN_HOLD_SECONDS) &
        #  (after_limitup_df_with_exitsignals['exit_score_smooth'] >= 2) &
        #  (~after_limitup_df_with_exitsignals['keep_holding']) &
        #  (after_limitup_df_with_exitsignals['TradeAmount'] <= entry_cb_amount) &
        #  (after_limitup_df_with_exitsignals['TradeCount'] <= entry_cb_count)) 
