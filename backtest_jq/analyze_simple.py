"""轻量级参数对比分析模块。

根据 compare_params_plan.md 设计，输出纯表格，不做绘图。
在 JQ 研究环境中运行。
"""
from jqdata import *
import numpy as np
import pandas as pd
from generate_params import Rules


def _parse_param_columns(config, Rules=Rules):
    """将参数字典解析为表格用的参数列值。
    
    开关类参数: 开启显示 ✓, 关闭显示 -
    范围类参数 (S/v/r): 显示核心数值
    """
    toggleable = {r[0] for r in Rules if len(r[2]) == 2}
    row = {}
    
    for rule in Rules:
        short_name = rule[0]
        if short_name not in config:
            row[short_name] = '-'
            continue
            
        val = config[short_name]
        
        if short_name == 'S':
            # 评分上限，始终显示数值
            row['S'] = f"{val[1]:.1f}"
        elif short_name == 'v':
            # 成交量阈值
            row['v'] = f"{val[2]:.1f}" if val[0] else '-'
        elif short_name == 'r':
            # R² 阈值
            row['r'] = f"{val[1]:.1f}" if val[0] else '-'
        elif short_name in toggleable:
            # 普通开关
            row[short_name] = '✓' if val[0] else '-'
        else:
            row[short_name] = str(val)
    
    return row


def _fetch_risk_metrics(backtest_id):
    """从 JQ 平台获取回测风险指标。"""
    try:
        bt = get_backtest(backtest_id)
        if not bt or bt.get_status() != 'done':
            return None
        metrics = bt.get_risk()
        if not metrics:
            return None
        
        ann_ret = metrics.get('annual_algo_return', 0)
        algo_return = metrics.get('algorithm_return', 0)
        max_dd = metrics.get('max_drawdown', 0)
        
        # 计算 Calmar (避免除零)
        calmar = ann_ret / max_dd if max_dd > 0.001 else 0.0
        
        return {
            'Return': algo_return,
            'Ann.Ret': ann_ret,
            'MaxDD': -max_dd,       # 显示为负数更直观
            'Calmar': calmar,
            'Sharpe': metrics.get('sharpe', 0),
            'Volatility': metrics.get('algorithm_volatility', 0),
            'WinRate': metrics.get('win_ratio', 0),
            'Trades': metrics.get('win_count', 0) + metrics.get('lose_count', 0),
            'AvgHoldDays': metrics.get('avg_position_days', 0),
            'Turnover': metrics.get('turnover_rate', 0),
        }
    except Exception as e:
        print(f"  获取指标失败 ({backtest_id}): {e}")
        return None


def _fetch_yearly_returns(backtest_id):
    """从 JQ 平台获取每年收益率，返回 {2018: 0.12, 2019: -0.05, ...} 或 None。"""
    try:
        bt = get_backtest(backtest_id)
        if not bt or bt.get_status() != 'done':
            return None
        risks = bt.get_period_risks()
        if not risks or 'algorithm_return' not in risks:
            return None
        df_src = risks['algorithm_return']
        if 'one_month' not in df_src.columns:
            return None
        monthly = df_src['one_month']
        # 按年复合
        df_m = monthly.to_frame(name='ret')
        df_m['Year'] = df_m.index.map(lambda x: int(x.split('-')[0]))
        yearly = df_m.groupby('Year')['ret'].apply(lambda x: np.prod(1 + x) - 1)
        return {f'Y{y}': r for y, r in yearly.items()}
    except Exception as e:
        return None


def compare_params(tasks, sort_by='Calmar', ascending=False, yearly=False):
    """多参数回测结果对比表。
    
    tasks: list of (name, param_dict, backtest_id) — batch_run 的输出格式
    sort_by: str, 排序列名 (默认 'Calmar')
    ascending: bool, 升序/降序
    yearly: bool, 是否在表格中显示每年收益列
    
    返回: pd.DataFrame, 每行一个参数组合, 列 = 参数列 + 指标列 [+ 年收益列]
    """
    rows = []
    param_keys = [r[0] for r in Rules]
    
    print(f"正在获取 {len(tasks)} 组回测指标...")
    
    for name, config, bt_id in tasks:
        risk = _fetch_risk_metrics(bt_id)
        if risk is None:
            print(f"  跳过 {name} (无有效指标)")
            continue
        
        # 参数列
        param_cols = _parse_param_columns(config)
        
        # 年收益列
        year_cols = {}
        if yearly:
            yr = _fetch_yearly_returns(bt_id)
            if yr:
                year_cols = yr
        
        # 合并
        row = {'ID': name}
        row.update(param_cols)
        row.update(risk)
        row.update(year_cols)
        rows.append(row)
    
    if not rows:
        print("没有有效的回测结果。")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # 设置 ID 为索引
    df.set_index('ID', inplace=True)
    
    # --- 显式排列列顺序 ---
    # 1. 参数列: Rules 中的短名
    param_keys = [r[0] for r in Rules if r[0] in df.columns]
    
    # 2. 指标列: _fetch_risk_metrics 返回的顺序
    indicator_order = ['Return', 'Ann.Ret', 'MaxDD', 'Calmar', 'Sharpe', 'Volatility', 'WinRate', 'Trades', 'Turnover']
    indicator_keys = [k for k in indicator_order if k in df.columns]
    
    # 3. 年收益列: Y20XX 格式，按年份排序
    year_keys = sorted([c for c in df.columns if c.startswith('Y') and c[1:].isdigit()])
    
    # 合并所有排好序的列
    new_columns = param_keys + indicator_keys + year_keys
    # 确保没有遗漏其他可能的列
    remaining_cols = [c for c in df.columns if c not in new_columns]
    df = df[new_columns + remaining_cols]
    
    # 排序
    if sort_by in df.columns:
        df.sort_values(sort_by, ascending=ascending, inplace=True)
    
    return df


def format_table(df):
    """格式化 DataFrame 用于打印，百分比和数值对齐。"""
    fmt = df.copy()
    pct_cols = ['Return','Ann.Ret', 'MaxDD', 'Volatility', 'WinRate']
    float_cols = ['Calmar', 'Sharpe', 'Turnover']
    
    # 年收益列也用百分比格式
    year_cols = [c for c in fmt.columns if c.startswith('Y') and c[1:].isdigit()]
    pct_cols = pct_cols + year_cols
    
    for col in pct_cols:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else x)
    
    for col in float_cols:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    
    # 其他列处理
    if 'Trades' in fmt.columns:
        fmt['Trades'] = fmt['Trades'].apply(lambda x: f"{int(x)}" if isinstance(x, (int, float)) else x)
    
    if 'AvgHoldDays' in fmt.columns:
        fmt['AvgHoldDays'] = fmt['AvgHoldDays'].apply(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x)
    
    return fmt


def print_compare(tasks, sort_by='Calmar', ascending=False, yearly=False):
    """一键对比并打印格式化表格。"""
    df = compare_params(tasks, sort_by=sort_by, ascending=ascending, yearly=yearly)
    if df.empty:
        return df
    
    fmt = format_table(df)
    
    # 使用 option_context 确保打印时不换行，且显示所有列
    with pd.option_context('display.max_columns', None, 
                           'display.width', 1000, 
                           'display.expand_frame_repr', False):
        output = fmt.to_string(line_width=1000)
        line_len = max(len(line) for line in output.split('\n'))
        sep = "=" * max(120, line_len)
        
        print("\n" + sep)
        print(f"参数优化对比表 (按 {sort_by} {'升序' if ascending else '降序'} 排列)")
        print(sep)
        print(output)
        print(sep)
        print(f"共 {len(df)} 组参数")
    
    return df


def get_best_config(tasks, sort_by='Calmar'):
    """从 tasks 中找到指标最优的配置并返回其 config 字典。
    
    tasks: list of (name, config_dict, backtest_id)
    返回: (best_name, best_config_dict) 或 (None, None)
    """
    df = compare_params(tasks, sort_by=sort_by, ascending=False)
    if df.empty:
        return None, None
    
    best_name = df.index[0]
    best_config = next(cfg for name, cfg, _ in tasks if name == best_name)
    return best_name, best_config
