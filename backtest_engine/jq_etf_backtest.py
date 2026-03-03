from jqdata import *
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

plt.style.use('ggplot') # 用来设置作图风格
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

STRATEGY_FILES = {
    'wy03': './ETFs/ETF_wy03_opt.py',
    'long': './ETFs/ETF_long_opt.py',
    'yj15': './ETFs/ETF_yj15_modular.py'
}
ID_Save_Path = './ETFs/saved_name_id_mapper.json'
Base_Back_Id = '6e21bbaee8cc3f8423def84436a2bf49'

def get_name_id_mapper(update = {}):
    if not os.path.exists(ID_Save_Path):
        with open(ID_Save_Path, 'w') as f:
            json.dump(update, f)
            print('Saved Name-ID json file.')
    with open(default_save_path, 'r') as f:
        res = json.load(f)
    res.update(update)
    with open(default_save_path, 'w') as f:
        json.dump(update, f)
        print('Saved Name-ID json file.')
    return res

def generate_strategy_code(strategy_file_path, params=dict()):
    """
    读取策略文件并替换占位符参数
    params: dict, 例如 {'EXECUTION_TIME_PLACEHOLDER': "'14:30'"}
    """
    new_lines = []
    try:
        with open(strategy_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                # 检查当前行是否是我们要替换的参数定义行
                replaced = False
                if stripped.startswith('EXECUTION_') and '=' in stripped:
                    key = stripped.split('=', 1)[0].strip()
                    if key in params:
                        # 替换为新值
                        new_lines.append(f"{key} = {params[key]}\n")
                        replaced = True
                
                if not replaced:
                    new_lines.append(line)
        return ''.join(new_lines)
    except Exception as e:
        print(f"Error processing {strategy_file_path}: {e}")
        return ""
    
def wrapped_create_backtest(name, code,
                            initial_cash=100000,
                            start_day='2018-01-01',
                            end_day='2026-01-10',
                            frequency='day'):
    if name in S_Name_Id:
        pass
    else:
        backtest_id = create_backtest(Base_Back_Id, start_day, end_day, frequency=frequency,
                                      initial_cash=initial_cash, initial_positions=None, extras=None, name=name,
                                      code=code, benchmark=None, python_version=3, use_credit=False)
        S_Name_Id[name] = backtest_id
        print(f"Create {name} backtest.")
        while True:
            gt = get_backtest(backtest_id)
            status = gt.get_status()
            if status == 'done':
                print(f"Backtest {name} Done.")
                return {'status': 'done'}
            elif status in ['failed', 'canceled', 'deleted']:
                print(f"\n[CRITICAL] Backtest {name} ({backtest_id}) FAILED with status: {status}")
                return {'status': 'failed'}
            time.sleep(5)

class PoolEvaluator:

    def __init__(self, backtest_id, etf_list, price_field='open', percent = (0.25, 0.75)):
        """
        初始化评估器
        :param backtest_id: str, 回测ID (from create_backtest)
        :param etf_list: list, 标的池代码列表
        :param price_field: str, 使用的价格字段
        :param percent: tuple, 排名分档阈值 (high, low), 默认(0.25, 0.75)即前25%和后25%
        """

        self.percent = percent
        print(f"正在加载回测数据 ID: {backtest_id}...")
        self.gt = get_backtest(backtest_id)

        results_list = self.gt.get_results()
        df_results = pd.DataFrame(results_list)
        df_results['time'] = pd.to_datetime(df_results['time'])
        df_results.set_index('time', inplace=True)

        self.strategy_total_ret = df_results['returns']
        self.strategy_ret = (1 + self.strategy_total_ret).pct_change().fillna(0)

        if len(self.strategy_ret) > 0:
            self.strategy_ret.iloc[0] = self.strategy_total_ret.iloc[0]

        self.strategy_ret.index = self.strategy_ret.index.normalize()

        pos_list = self.gt.get_positions()
        if pos_list:
            df_pos = pd.DataFrame(pos_list)
            df_pos['time'] = pd.to_datetime(df_pos['time']).dt.normalize()

            df_pos = df_pos[df_pos['amount'] > 0]

            daily_holdings = df_pos.sort_values('amount', ascending=False).groupby('time')['security'].first()
            self.strategy_positions = daily_holdings.reindex(self.strategy_ret.index).fillna("CASH")
        else:
            self.strategy_positions = pd.Series("CASH", index=self.strategy_ret.index)

        start_date = self.strategy_ret.index[0]
        end_date = self.strategy_ret.index[-1]
        
        print(f"正在获取 {len(etf_list)} 个标的池资产行情...")
        raw_data = get_price(list(etf_list), start_date=start_date, end_date=end_date, frequency='daily', fields=[price_field],panel=False)

        if hasattr(raw_data, 'to_frame'): 
            self.pool_prices = raw_data[price_field]
        elif isinstance(raw_data, pd.DataFrame):
            if 'code' in raw_data.columns:
                self.pool_prices = raw_data.pivot(index='time', columns='code', values=price_field)
            else:
                self.pool_prices = raw_data

        self.pool_prices.index = pd.to_datetime(self.pool_prices.index).normalize()

        market_index = self.pool_prices.index

        self.strategy_ret = self.strategy_ret.reindex(market_index).fillna(0.0)

        self.strategy_positions = self.strategy_positions.reindex(market_index).fillna("CASH")

        common_index = self.strategy_ret.index.intersection(self.pool_prices.index)
        
        self.strategy_ret = self.strategy_ret.loc[common_index]
        self.strategy_positions = self.strategy_positions.loc[common_index]
        self.pool_prices = self.pool_prices.loc[common_index]

        self.pool_rets = self.pool_prices.pct_change().fillna(0)
        self.pool_index_ret = self.pool_rets.mean(axis=1)

        self.strategy_nav = (1 + self.strategy_ret).cumprod()
        self.pool_index_nav = (1 + self.pool_index_ret).cumprod()

    def evaluate_rolling_returns(self, window=20, fig=False, percent=None):
        """
        方法1: 滚动收益对比 (Rolling Return Comparison)
        计算策略与池内所有资产的N日滚动收益，并分析策略处于排名的百分位。
        """
        if percent is None:
            percent = self.percent
        high, low = percent

        strategy_roll = self.strategy_ret.rolling(window).apply(lambda x: (1+x).prod() - 1, raw=True)
        pool_roll = self.pool_rets.rolling(window).apply(lambda x: (1+x).prod() - 1, raw=True)

        combined = pool_roll.copy()
        combined['Strategy'] = strategy_roll

        ranks = combined.rank(axis=1, pct=True, ascending=True)
        strategy_rank = ranks['Strategy']
        
        print(f"\n[滚动收益分析 (N={window}日)]")
        print(f"平均排名百分位: {1-strategy_rank.mean():.2%}")
        print(f"位列前{high:.0%}的时间占比: {(strategy_rank >= (1-high)).mean():.2%}")
        print(f"位列后{(1-low):.0%}的时间占比: {(strategy_rank <= (1-low)).mean():.2%}")
        if fig:
            self.plot_rolling_returns(window, percent=percent)
        
        return strategy_rank

    def plot_rolling_returns(self, window=20, percent=None):
        """
        方法1绘图: 滚动收益可视化
        绘制两个子图:
        1. 滚动收益对比: 策略 vs 标的池(最大/最小/平均)
        2. 排名分位数: 策略在池子中的排名 (1.0=最优)
        """
        if percent is None:
            percent = self.percent
        high, low = percent

        strategy_roll = self.strategy_ret.rolling(window).apply(lambda x: (1+x).prod() - 1, raw=True)
        pool_roll = self.pool_rets.rolling(window).apply(lambda x: (1+x).prod() - 1, raw=True)

        pool_mean = pool_roll.mean(axis=1)
        pool_max = pool_roll.max(axis=1)
        pool_min = pool_roll.min(axis=1)

        combined = pool_roll.copy()
        combined['Strategy'] = strategy_roll
        ranks = combined.rank(axis=1, pct=True, ascending=True)['Strategy']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        ax1.plot(strategy_roll, label='Strategy', color='red', linewidth=2)
        ax1.plot(pool_mean, label='Pool Average', color='gray', linestyle='--')

        ax1.fill_between(pool_roll.index, pool_min, pool_max, color='gray', alpha=0.2, label='Pool Range (Min-Max)')
        
        ax1.set_title(f'{window}-Day Rolling Returns Comparison')
        ax1.set_ylabel('Rolling Return')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax2.plot(ranks, label='Strategy Rank Percentile', color='blue')
        ax2.axhline(high, color='green', linestyle=':', label=f'Top {high:.0%}')
        ax2.axhline(0.50, color='orange', linestyle=':', label='Median')
        ax2.axhline(low, color='red', linestyle=':', label=f'Bottom {(1-low):.0%}')
        
        ax2.fill_between(ranks.index, 1-high, 1.0, color='green', alpha=0.1) # 优秀区
        ax2.fill_between(ranks.index, 0, 1-low, color='red', alpha=0.1)     # 差区
        
        ax2.set_title(f'Strategy Rank Percentile (1.0=Best)')
        ax2.set_ylabel('Percentile')
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='lower right')
        ax2.grid(True)
        
        plt.tight_layout()
        print(f"\n[滚动绘图] 已生成 {window}日 滚动收益分析图。")

    def evaluate_holding_attribution(self, simple_df=False):
        """
        方法2: 持仓周期归因 (Holding Period Attribution)
        自动使用 self.strategy_positions 进行分析
        """
        strategy_positions = self.strategy_positions
        groups = (strategy_positions != strategy_positions.shift()).cumsum()
        results = []
        
        for g_id, group_data in strategy_positions.groupby(groups):
            start_date = group_data.index[0]
            end_date = group_data.index[-1]
            asset_held = group_data.iloc[0]
            period_pool_rets = self.pool_rets.loc[start_date:end_date]
            if len(period_pool_rets) < 1:
                continue
            period_ret = (1 + period_pool_rets).prod() - 1
            strat_period_trace = (1 + self.strategy_ret.loc[start_date:end_date]).cumprod()
            if len(strat_period_trace) > 0:
                 held_ret = strat_period_trace.iloc[-1] - 1
            else:
                 held_ret = 0.0
            if asset_held != "CASH" and asset_held not in period_ret:
                 pass
            best_asset = period_ret.idxmax()
            best_ret = period_ret.max()
            worst_asset = period_ret.idxmin()
            worst_ret = period_ret.min()
            pool_index_period = self.pool_index_ret.loc[start_date:end_date]
            if len(pool_index_period) > 0:
                avg_ret = (1 + pool_index_period).prod() - 1
            else:
                avg_ret = 0.0
            
            # 计算排名 (Rank)
            compare_set = period_ret.copy()
            if asset_held in compare_set:
                compare_set[asset_held] = held_ret
            better_count = (compare_set > held_ret).sum()
            rank = better_count + 1
            pool_size = len(period_ret)
            
            # 判断是否"赢了"
            is_best = (asset_held == best_asset)
            if asset_held == "CASH" and held_ret > best_ret:
                is_best = True
                
            results.append({
                'Start': start_date,
                'End': end_date,
                'Days': len(group_data),
                'Held_Asset': asset_held,
                'Held_Return': held_ret,
                'Pool_Best': best_asset,
                'Best_Return': best_ret,
                'Pool_Avg': avg_ret,
                'Rank': f"{rank}/{pool_size}",
                'Is_Best': is_best,
                'Beats_Avg': (held_ret > avg_ret)
            })
            
        df_res = pd.DataFrame(results, columns=['Start', 'End', 'Days', 'Held_Asset', 'Held_Return', 'Pool_Best', 'Best_Return', 'Pool_Avg', 'Rank', 'Is_Best', 'Beats_Avg'])
        
        if df_res.empty:
            print("\n[持仓归因] 未找到有效的持仓周期。")
            return df_res

        print(f"\n[持仓归因分析]")
        print(f"总持仓段数: {len(df_res)}")
        
        # 按段数统计
        print(f"命中率 (按段数): {df_res['Is_Best'].mean():.2%}")
        print(f"胜率 (按段数): {df_res['Beats_Avg'].mean():.2%}")
        
        # 按天数统计
        total_days = df_res['Days'].sum()
        best_days = df_res.loc[df_res['Is_Best'], 'Days'].sum()
        beat_days = df_res.loc[df_res['Beats_Avg'], 'Days'].sum()
        
        print(f"命中率 (按天数): {best_days/total_days:.2%} ({best_days}/{total_days} days)")
        print(f"胜率 (按天数): {beat_days/total_days:.2%} ({beat_days}/{total_days} days)")
        
        print(f"平均超额收益 (vs 池均值): {(df_res['Held_Return'] - df_res['Pool_Avg']).mean():.2%} (算术平均)")

        total_strat_geo = (1 + df_res['Held_Return']).prod() - 1
        total_pool_geo = (1 + df_res['Pool_Avg']).prod() - 1
        
        print(f"\n[几何累积收益对比]")
        print(f"分段累积策略收益: {total_strat_geo:.2%}")
        print(f"分段累积基准收益: {total_pool_geo:.2%}")
        print(f"实际总超额收益: {total_strat_geo - total_pool_geo:.2%}")
        
        if simple_df:
            # 简单模式: 聚合 Rank 和 Held_Asset 并排显示
            total_segs = len(df_res)
            total_days = df_res['Days'].sum()
            df_res['Rank_Num'] = df_res['Rank'].apply(lambda x: int(x.split('/')[0]))
            rank_seg_counts = df_res['Rank_Num'].value_counts().sort_index()
            rank_day_sums = df_res.groupby('Rank_Num')['Days'].sum()
            
            df_rank = pd.DataFrame({
                'Rank': rank_seg_counts.index,
                'Rank_Seg_Count': rank_seg_counts.values,
                'Rank_Day_Sum': rank_day_sums.reindex(rank_seg_counts.index).values
            })
            df_rank['Rank_Seg_Pct'] = (df_rank['Rank_Seg_Count'] / total_segs).apply(lambda x: f"{x:.2%}")
            df_rank['Rank_Day_Pct'] = (df_rank['Rank_Day_Sum'] / total_days).apply(lambda x: f"{x:.2%}")
            df_rank_final = df_rank[['Rank', 'Rank_Seg_Pct', 'Rank_Day_Pct']]
            asset_seg_counts = df_res['Held_Asset'].value_counts()
            asset_day_sums = df_res.groupby('Held_Asset')['Days'].sum()
            
            df_asset = pd.DataFrame({
                'Asset': asset_seg_counts.index,
                'Asset_Name': [get_security_info(etf).display_name if hasattr(get_security_info(etf),'display_name')\
                               else '' for etf in asset_seg_counts.index],
                'Asset_Seg_Count': asset_seg_counts.values,
                'Asset_Day_Sum': asset_day_sums.reindex(asset_seg_counts.index).values
            })
            df_asset['Sort_Key'] = df_asset['Asset_Day_Sum']
            df_asset = df_asset.sort_values('Sort_Key', ascending=False)
            df_asset['Asset_Seg_Pct'] = (df_asset['Asset_Seg_Count'] / total_segs).apply(lambda x: f"{x:.2%}")
            df_asset['Asset_Day_Pct'] = (df_asset['Asset_Day_Sum'] / total_days).apply(lambda x: f"{x:.2%}")
            df_asset_final = df_asset[['Asset', 'Asset_Name', 'Asset_Seg_Pct', 'Asset_Day_Pct']]
            df_rank_final = df_rank_final.reset_index(drop=True)
            df_asset_final = df_asset_final.reset_index(drop=True)
            df_combined = pd.concat([df_rank_final, df_asset_final], axis=1)
            df_combined.columns = ['Rank', 'Seg_Pct', 'Days_Pct', 'Asset', 'Asset_Name', 'Seg_Pct', 'Days_Pct']
            df_combined = df_combined.fillna('')
            return df_combined

        return df_res

    def evaluate_switching_effect(self):
        """
        方法4: 换仓效果分析 (Switching Effectiveness)
        分析每一次换仓 (Asset A -> Asset B) 后，新持仓是否跑赢了旧持仓？
        """
        pos = self.strategy_positions
        trades = pos[pos != pos.shift(1)]
        if len(trades) > 0 and trades.index[0] == self.strategy_ret.index[0]:
            trades = trades.iloc[1:]
        switching_results = []
        for date, new_asset in trades.items():
            old_asset = pos.shift(1).loc[date]
            next_dates = trades.index[trades.index > date]
            if len(next_dates) > 0:
                end_date = next_dates[0] 
            else:
                end_date = self.strategy_ret.index[-1]
            period_pool_rets = self.pool_rets.loc[date:end_date]
            period_pool_rets_lagged = self.pool_rets.loc[date:end_date].iloc[1:]

            if len(period_pool_rets) < 1:
                continue
                
            def get_asset_ret(asset, rets_df):
                if len(rets_df) == 0: return 0.0
                comp_ret = (1 + rets_df).prod() - 1
                if asset == "CASH": return 0.0
                if asset in comp_ret: return comp_ret[asset]
                return 0.0

            ret_new = get_asset_ret(new_asset, period_pool_rets)
            ret_old = get_asset_ret(old_asset, period_pool_rets)
            switch_alpha = ret_new - ret_old
            ret_new_lag = get_asset_ret(new_asset, period_pool_rets_lagged)
            ret_old_lag = get_asset_ret(old_asset, period_pool_rets_lagged)
            switch_alpha_lag = ret_new_lag - ret_old_lag
            
            switching_results.append({
                'Date': date,
                'Old_Asset': old_asset,
                'New_Asset': new_asset,
                'New_Return': ret_new,
                'Old_Return': ret_old,
                'Switch_Alpha': switch_alpha,
                'Is_Correct': (switch_alpha > 0),
                'Switch_Alpha_Lag': switch_alpha_lag,
                'Is_Correct_Lag': (switch_alpha_lag > 0)
            })
            
        df_switch = pd.DataFrame(switching_results, columns=['Date', 'Old_Asset', 'New_Asset', 'New_Return', 'Old_Return', 'Switch_Alpha', 'Is_Correct', 'Switch_Alpha_Lag', 'Is_Correct_Lag'])
        
        if df_switch.empty:
            print("\n[换仓分析] 未发现换仓操作。")
            return df_switch
            
        print(f"\n[换仓效果分析]")
        print(f"总换仓次数: {len(df_switch)}")
        print(f"换仓成功率 (新>旧) [含T日]: {df_switch['Is_Correct'].mean():.2%}")
        print(f"平均换仓Alpha [含T日]: {df_switch['Switch_Alpha'].mean():.2%}")
        print(f"累计换仓Alpha [含T日]: {df_switch['Switch_Alpha'].sum():.2%}")
        print(f"----------")
        print(f"换仓成功率 (新>旧) [T+1日]: {df_switch['Is_Correct_Lag'].mean():.2%} (排除换仓当日涨跌幅)")
        print(f"平均换仓Alpha [T+1日]: {df_switch['Switch_Alpha_Lag'].mean():.2%}")
        print(f"累计换仓Alpha [T+1日]: {df_switch['Switch_Alpha_Lag'].sum():.2%}")
        
        return df_switch

    def plot_relative_strength(self, start_date=None, end_date=None):
        """
        方法3: 相对强弱曲线 (Relative Strength Curve)
        绘制 策略净值 / 标的池等权净值 (上图)
        绘制 策略与标的池等权净值曲线 (下图)
        """
        # 1. 准备数据
        rs = self.strategy_nav / self.pool_index_nav
        strat_nav = self.strategy_nav.copy()
        pool_nav = self.pool_index_nav.copy()

        # 日期过滤与重归一化
        if start_date is not None:
            rs = rs.loc[start_date:]
            strat_nav = strat_nav.loc[start_date:]
            pool_nav = pool_nav.loc[start_date:]
            
            # 如果截取后非空，重新归一化到1.0，便于区间对比
            if not rs.empty:
                rs = rs / rs.iloc[0]
                strat_nav = strat_nav / strat_nav.iloc[0]
                pool_nav = pool_nav / pool_nav.iloc[0]

        if end_date is not None:
            rs = rs.loc[:end_date]
            strat_nav = strat_nav.loc[:end_date]
            pool_nav = pool_nav.loc[:end_date]
        
        # 2. 绘图 (2个子图)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # 子图1: 相对强弱 (比值)
        ax1.plot(rs, label='Relative Strength (Strategy / Pool Index)', color='#800080') # Purple
        ax1.axhline(1.0, color='gray', linestyle='--')
        ax1.set_title('Relative Strength Analysis (Strategy / Pool)')
        ax1.set_ylabel('Ratio')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # 子图2: 净值曲线
        ax2.plot(strat_nav, label='Strategy NAV', color='red', linewidth=1.5)
        ax2.plot(pool_nav, label='Pool Equal Weighted NAV', color='blue', linestyle='--', linewidth=1.5)
        ax2.set_title('Net Value Comparison')
        ax2.set_ylabel('Net Value')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        plt.tight_layout()
        # plt.show() # Uncomment to show locally
        print("\n[相对强弱] 已生成RS曲线 (上) 与净值对比曲线 (下)。")
        # return rs

class BacktestAnalyzer:
    def __init__(self, registry=None):
        """
        registry: dict or str (path to registry file)
        """
        self.registry = registry

    def compare_results(self, strategy_names=None,print_df=False):
        """
        对比回测结果 (实时获取指标)
        strategy_names: list, 要对比的策略名称列表 (e.g. ['wy03']). None=全部
        """
        data = []
        print("\nFetching metrics for comparison...")
        
        for task_name, record in self.registry.items():
            # 1. 基本过滤
            bt = get_backtest(record)
            if not bt or bt.get_status() != 'done':
                continue
            
            # 2. 策略名称过滤
            if strategy_names and task_name not in strategy_names:
                continue
            
            # 3. 动态获取指标
            try:
                metrics = bt.get_risk()
                if not metrics:
                    continue
                    
                row = {
                    'Test Name': task_name,
                    'Return': f"{metrics.get('algorithm_return', 0):.2%}",
                    'Volatility': f"{metrics.get('algorithm_volatility', 0):.2%}",
                    'Ann. Return': f"{metrics.get('annual_algo_return', 0):.2%}",
                    'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                    'Max Drawdown Period': f"{metrics.get('max_drawdown_period', 'N/A')}",
                    'Alpha': f"{metrics.get('alpha', 0):.2f}",
                    'Beta': f"{metrics.get('beta', 0):.2f}",
                    'Sharpe': f"{metrics.get('sharpe', 0):.2f}",
                    'Sortino': f"{metrics.get('sortino', 0):.2f}",
                    'IF.': f"{metrics.get('information', 0):.2f}",
                    'Lose Count': f"{metrics.get('lose_count', 0)}",
                    'Win Count': f"{metrics.get('win_count', 0)}",
                    'Win Rate': f"{metrics.get('win_ratio', 0):.2%}",
                    'Turnover Rate': f"{metrics.get('turnover_rate', 0):.2f}",
                    'Avg. Trade Return': f"{metrics.get('avg_trade_return', 0):.2%}",
                    'Avg. Position Days': f"{metrics.get('avg_position_days', 0):.2f}",
                }
                data.append(row)
            except Exception as e:
                print(f"Error fetching metrics for {e}")

        if not data:
            print("No completed backtests found for comparison.")
            return None

        # 4. 生成 DataFrame 并转置
        df_raw = pd.DataFrame(data)
        
        # 定义期望的行顺序 (Metrics)
        row_order = [
            'Return', 'Ann. Return', 'Volatility', 'Max Drawdown', 'Max Drawdown Period', 
            'Alpha', 'Beta', 'Sharpe', 'Sortino', 'IF.', 
            'Lose Count', 'Win Count', 'Win Rate', 'Turnover Rate', 
            'Avg. Trade Return', 'Avg. Position Days'
        ]
        df_raw['Label'] = df_raw.apply(lambda x: f"{x['Test Name']}", axis=1)
        
        # 设置 Label 为索引
        df_raw.set_index('Label', inplace=True)
        if 'Strategy' in df_raw.columns:
            del df_raw['Strategy']
        if 'Test Name' in df_raw.columns:
            del df_raw['Test Name']
            
        # 转置
        df_t = df_raw.T
        
        # 尝试按指定顺序排序行 (如果存在)
        existing_rows = [r for r in row_order if r in df_t.index]
        other_rows = [r for r in df_t.index if r not in row_order]
        df_t = df_t.reindex(existing_rows + other_rows)
        
        if print_df:
            print("\n" + "="*100)
            print(f"BACKTEST COMPARISON ({len(data)} tests)")
            print("="*100)
            print(df_t.to_string())
            print("="*100 + "\n")
        
        return df_t

    def plot_curves(self, strategy_names=None, log_scale=False, start_date=None, end_date=None):
        """
        画出收益曲线
        strategy_names: list, 策略名过滤
        log_scale: bool, 是否使用对数坐标
        """

        plt.figure(figsize=(20, 11))
        has_data = False

        for task_name, record in self.registry.items():
            # 1. 基本过滤
            bt = get_backtest(record)
            if not bt or bt.get_status() != 'done':
                continue
            
            # 2. 策略名称过滤
            if strategy_names and task_name not in strategy_names:
                continue

            try:
                results = bt.get_results()
                if not results:
                    continue

                # 转换为 DataFrame
                df_res = pd.DataFrame(results)
                # time 字段转为 datetime
                df_res['time'] = pd.to_datetime(df_res['time'])
                df_res.set_index('time', inplace=True)
                
                # 计算净值 (Net Value)
                df_res['net_value'] = df_res['returns'] + 1

                # 过滤日期
                if start_date:
                    original_len = len(df_res)
                    df_res = df_res[df_res.index >= pd.to_datetime(start_date)]
                    # 如果指定了开始日期，且数据不为空，进行归一化处理（以此日为单位1）
                    if not df_res.empty and len(df_res) < original_len:
                         df_res['net_value'] = df_res['net_value'] / df_res['net_value'].iloc[0]

                if end_date:
                    df_res = df_res[df_res.index <= pd.to_datetime(end_date)]

                if df_res.empty:
                    print(f"No data for {s_name} in specified range.")
                    continue
                
                # 绘制收益曲线
                plt.plot(df_res.index, df_res['net_value'], label=task_name)
                has_data = True
                
                
            except Exception as e:
                print(f"Error processing plot for {bts_id}: {e}")

        if has_data:
            plt.title('Backtest Equity Curves')
            plt.xlabel('Date')
            
            if log_scale:
                plt.yscale('log')
                plt.ylabel('Net Value (Log Scale)')
            else:
                plt.ylabel('Net Value')
                
            plt.legend(loc='upper left')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.show()
        else:
            print("No data available for plotting.")

    def show_monthly_returns(self, strategy_names=None, mode='individual', plot_type='both'):
        """
        展示月度/年度收益表及热力图
        mode: 'individual' (单独展示) | 'compare' (对比展示)
        plot_type: 'month' (仅月度) | 'year' (仅年度) | 'both' (两者都画) - 仅在 compare 模式有效
        """
        target_records = list()
        for task_name, record in self.registry.items():
            # 1. 基本过滤
            bt = get_backtest(record)
            if not bt or bt.get_status() != 'done':
                continue
            
            # 2. 策略名称过滤
            if strategy_names and task_name not in strategy_names:
                continue
            target_records.append((task_name, record))
        
        if not target_records:
            print("No valid backtests found.")
            return

        if mode == 'compare':
            self._show_compare_mode(target_records, plot_type)
        else:
            self._show_individual_mode(target_records)

    def _fetch_monthly_data(self, bts_id, full_name):
        try:
            bt = get_backtest(bts_id)
            risks = bt.get_period_risks()
            if not risks or 'algorithm_return' not in risks:
                return None
                
            df_src = risks['algorithm_return']
            if 'one_month' not in df_src.columns:
                return None
            
            return df_src['one_month']
        except Exception as e:
            print(f"Error fetching data for {full_name}: {e}")
            return None

    def _calculate_yearly_returns(self, monthly_series):
        """
        根据月度收益计算年度收益
        Input: pd.Series (index=Date, values=MonthlyReturn)
        Output: pd.Series (index=Year, values=YearlyReturn)
        """
        if monthly_series is None or monthly_series.empty:
            return pd.Series(dtype=float)
            
        # 转换为 DataFrame 以便处理
        df = monthly_series.to_frame(name='ret')
        df['Year'] = df.index.map(lambda x: int(x.split('-')[0]))
        
        # Group by Year and compound
        yearly = df.groupby('Year')['ret'].apply(lambda x: np.prod(1 + x) - 1)
        return yearly

    def _show_compare_mode(self, target_records, plot_type='both'):
        monthly_data = {}
        yearly_data = {}
        
        for task_name, record in target_records:          
            series = self._fetch_monthly_data(record, task_name)
            if series is not None:
                monthly_data[task_name] = series
                yearly_data[task_name] = self._calculate_yearly_returns(series)
        
        if not monthly_data:
            print("No return data available for comparison.")
            return

        # --- Monthly Comparison ---
        if plot_type in ['month', 'both']:
            df_compare = pd.DataFrame(monthly_data)
            df_compare.sort_index(inplace=True,ascending=True)
            
            print("\n" + "="*80)
            print(f"MONTHLY RETURN COMPARISON")
            print("="*80)
            print(df_compare.applymap(lambda x: f"{x:.2%}" if pd.notnull(x) else "").to_string())
            print("="*80 + "\n")
            
            try:
                # Time on Y-axis -> barh
                # Adjust height based on number of months to avoid overcrowding
                height = len(df_compare) * 0.8
                ax = df_compare.plot(kind='barh', figsize=(12, min(16,height)), width=0.8)
                plt.title('Monthly Return Comparison')
                plt.xlabel('Return')
                plt.ylabel('Month')
                plt.grid(True, axis='x', alpha=0.5) # Grid on X for horizontal
                plt.axvline(0, color='black', linewidth=0.8) # Add zero line
                plt.gca().invert_yaxis()
                #plt.legend(loc='best')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error plotting monthly comparison: {e}")

        # --- Yearly Comparison ---
        if plot_type in ['year', 'both']:
            df_yearly = pd.DataFrame(yearly_data)
            df_yearly.sort_index(inplace=True,ascending=True)
            
            if plot_type == 'year': 
                print("\n" + "="*80)
                print(f"YEARLY RETURN COMPARISON")
                print("="*80)
                print(df_yearly.applymap(lambda x: f"{x:.2%}" if pd.notnull(x) else "").to_string())
                print("="*80 + "\n")
            
            try:
                height = len(df_yearly) * 0.8
                ax = df_yearly.plot(kind='barh', figsize=(12, height), width=0.8)
                plt.title('Yearly Return Comparison')
                plt.xlabel('Return')
                plt.ylabel('Year')
                plt.grid(True, axis='x', alpha=0.5)
                plt.axvline(0, color='black', linewidth=0.8)
                plt.gca().invert_yaxis()
                #plt.legend(loc='best')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                # plt.xticks(rotation=0) # Not needed for numeric/barh
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error plotting yearly comparison: {e}")

    def _show_individual_mode(self, target_records):
        for task_name, record in target_records:
            series = self._fetch_monthly_data(record, task_name)
            if series is None:
                continue
            
            try:
                # Pivot and Calculate
                df_calc = series.to_frame(name='one_month')
                df_calc['Year'] = df_calc.index.map(lambda x: int(x.split('-')[0]))
                df_calc['Month'] = df_calc.index.map(lambda x: int(x.split('-')[1]))
                
                pivot = df_calc.pivot(index='Year', columns='Month', values='one_month')
                
                # Yearly Calc
                yearly_ret = []
                years = pivot.index.tolist()
                for y in years:
                    row_rets = pivot.loc[y].dropna()
                    if len(row_rets) == 0:
                        yearly_ret.append(0.0)
                        continue
                    comp = np.prod(1 + row_rets) - 1
                    yearly_ret.append(comp)
                
                pivot['Yearly'] = yearly_ret
                
                # Print Table
                print(f"\n--- Monthly Returns: {task_name} ---")
                
                print_df = pivot.copy()
                for c in print_df.columns:
                    print_df[c] = print_df[c].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
                
                for m in range(1, 13):
                    if m not in print_df.columns:
                            print_df[m] = ""
                
                cols = list(range(1, 13)) + ['Yearly']
                print_df = print_df[cols]
                print(print_df.to_string())
                
                # Plot Heatmap
                self._plot_monthly_heatmap(pivot, task_name)
                    
            except Exception as e:
                print(f"Error processing individual view for {bts_id}: {e}")

    def _plot_monthly_heatmap(self, pivot_df, title):
        """
        绘制月度收益热力图
        """
        # 准备数据 (Drop Yearly column for heatmap)
        data = pivot_df.drop(columns=['Yearly'], errors='ignore')
        
        # 补全 1-12 月以便对齐
        for m in range(1, 13):
            if m not in data.columns:
                data[m] = np.nan
        data = data[sorted(data.columns)] # Sort 1..12
        
        years = data.index.tolist()
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        values = data.values # shape (n_years, 12)
        
        fig, ax = plt.subplots(figsize=(10, len(years) * 0.8 + 2))
        
        # Plot Heatmap
        # 使用 RdYlGn (红绿) colormap, center=0
        # 为了让 0 显示为白色/中性，通常需要自定义 norm 或 cmap range
        # 这里简单使用 RdYlGn
        im = ax.imshow(values, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
        
        # Label Ticks
        ax.set_xticks(np.arange(len(months)))
        ax.set_xticklabels(months)
        ax.set_yticks(np.arange(len(years)))
        ax.set_yticklabels(years)
        
        # Rotated x labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(years)):
            for j in range(len(months)):
                val = values[i, j]
                if pd.notnull(val):
                    text = ax.text(j, i, f"{val:.1%}",
                                   ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(f"Monthly Returns - {title}")
        fig.tight_layout()
        plt.colorbar(im, ax=ax, label='Return')
        plt.show()