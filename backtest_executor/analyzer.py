"""参数优化结果分析模块。

从 backtest_jq/analyze_simple.py 复制并解耦。
从 YAML 配置读取参数定义，而非 generate_params。

在 JQ 研究环境中运行。
"""
import argparse
import numpy as np
import pandas as pd
import yaml
import os
import json
import logging
from .logger import logger

try:
    from IPython.display import display, HTML
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

try:
    from jqdata import get_backtest
    JQ_AVAILABLE = True
except ImportError:
    JQ_AVAILABLE = False


def _detect_param_types(params_def):
    """根据参数定义自动判断类型。
    
    返回: (toggleable, rangeable) 两个集合
    - toggleable: 第一个值为 bool 的参数（开关类型）
    - rangeable: 值为 list 且长度 > 1 的参数（范围类型）
    """
    toggleable = set()
    rangeable = set()
    
    for key, cfg in params_def.items():
        values = cfg.get('values', [])
        default = cfg.get('default')
        
        if isinstance(default, list) and len(default) > 0:
            if isinstance(default[0], bool):
                toggleable.add(key)
            elif len(default) > 2:
                rangeable.add(key)
        elif values and isinstance(values[0], list):
            if isinstance(values[0][0], bool):
                toggleable.add(key)
            if len(values[0]) > 2:
                rangeable.add(key)
    
    return toggleable, rangeable


def _parse_param_columns(config, params_def, toggleable, rangeable):
    """将参数字典解析为表格用的参数列值。
    
    参数显示逻辑：
    - 开关类型(bool): 显示 ✓ / -
    - 范围类型(list): 显示最后一个数值
    - 其他: 显示原值
    """
    row = {}
    
    for key, cfg in params_def.items():
        var_name = cfg.get('var')
        if key in config:
            val = config[key]
        elif var_name and var_name in config:
            val = config[var_name]
        else:
            row[key] = '-'
            continue

        if isinstance(val, tuple):
            val = list(val)
        
        if key in toggleable and isinstance(val, list) and val:
            row[key] = '✓' if val[0] else '-'
        elif key in rangeable:
            if isinstance(val, list) and val and isinstance(val[0], bool) and val[0]:
                row[key] = f"{val[-1]}" if isinstance(val[-1], (int, float)) else str(val[-1])
            else:
                row[key] = '-'
        else:
            row[key] = str(val)
    
    return row


def _fetch_risk_metrics(backtest_id):
    """从 JQ 平台获取回测风险指标。"""
    if not JQ_AVAILABLE:
        return _mock_risk_metrics()
    
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
        
        calmar = ann_ret / max_dd if max_dd > 0.001 else 0.0
        
        return {
            'Return': algo_return,
            'Ann.Ret': ann_ret,
            'MaxDD': -max_dd,
            'Calmar': calmar,
            'Sharpe': metrics.get('sharpe', 0),
            'Volatility': metrics.get('algorithm_volatility', 0),
            'WinRate': metrics.get('win_ratio', 0),
            'Trades': metrics.get('win_count', 0) + metrics.get('lose_count', 0),
            'AvgHoldDays': metrics.get('avg_position_days', 0),
            'Turnover': metrics.get('turnover_rate', 0),
        }
    except Exception as e:
        logger.warning("获取指标失败 (%s): %s", backtest_id, e)
        return None


def _mock_risk_metrics():
    """模拟指标，用于测试。"""
    import random
    return {
        'Return': random.uniform(0.1, 0.5),
        'Ann.Ret': random.uniform(0.1, 0.4),
        'MaxDD': random.uniform(0.05, 0.2),
        'Calmar': random.uniform(0.5, 3.0),
        'Sharpe': random.uniform(0.5, 2.5),
        'Volatility': random.uniform(0.1, 0.3),
        'WinRate': random.uniform(0.4, 0.7),
        'Trades': random.randint(50, 200),
        'AvgHoldDays': random.uniform(5, 30),
        'Turnover': random.uniform(0.5, 2.0),
    }


def _fetch_yearly_returns(backtest_id):
    """从 JQ 平台获取每年收益率。"""
    if not JQ_AVAILABLE:
        return None
    
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
        df_m = monthly.to_frame(name='ret')
        df_m['Year'] = df_m.index.map(lambda x: int(x.split('-')[0]))
        yearly = df_m.groupby('Year')['ret'].apply(lambda x: np.prod(1 + x) - 1)
        return {f'Y{y}': r for y, r in yearly.items()}
    except Exception as e:
        return None


def load_results(mapper_path):
    """从 mapper.json 加载回测结果。
    
    返回: list of (name, params_dict, metrics_dict, bt_id)
    """
    if not os.path.exists(mapper_path):
        logger.error("mapper.json not found: %s", mapper_path)
        return []
    
    with open(mapper_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    runs = data.get('runs', {})
    results = []
    
    for key, run_info in runs.items():
        if run_info.get('status') != 'done':
            continue
        # 优先使用 name 字段，如果没有则使用 Key（兼容旧数据）
        display_name = run_info.get('name', key)
        results.append((
            display_name, 
            run_info.get('params', {}), 
            run_info.get('metrics', {}),
            run_info.get('bt_id')
        ))
    
    return results


def compare_params(results, params_def, sort_by='Calmar', ascending=False, yearly=False):
    """多参数回测结果对比表。
    
    results: list of (name, params_dict, metrics_dict, bt_id) — load_results 的输出
    params_def: dict, YAML 中的 params 定义
    sort_by: str, 排序列名 (默认 'Calmar')
    ascending: bool, 升序/降序
    
    返回: pd.DataFrame
    """
    toggleable, rangeable = _detect_param_types(params_def)
    
    rows = []
    
    for name, params, metrics, bt_id in results:
        if not metrics:
            # 如果 metrics 不存在，尝试使用 bt_id 获取（如果 bt_id 为空则尝试用 name）
            risk = _fetch_risk_metrics(bt_id or name)
            if risk is None:
                continue
        else:
            risk = {
                'Return': metrics.get('return', 0),
                'Ann.Ret': metrics.get('annual_return', 0),
                'MaxDD': abs(metrics.get('max_drawdown', 0)),
                'Calmar': metrics.get('calmar', 0),
                'Sharpe': metrics.get('sharpe', 0),
                'Volatility': metrics.get('volatility', metrics.get('algorithm_volatility', 0)),
                'WinRate': metrics.get('win_rate', metrics.get('win_ratio', 0)),
                'Trades': metrics.get('trades', 0),
                'AvgHoldDays': metrics.get('avg_hold_days', metrics.get('avg_position_days', 0)),
                'Turnover': metrics.get('turnover', metrics.get('turnover_rate', 0)),
            }
        
        param_cols = _parse_param_columns(params, params_def, toggleable, rangeable)
        
        year_cols = {}
        year_cols = {}
        if yearly and bt_id:
            yr = _fetch_yearly_returns(bt_id)
            if yr:
                year_cols = yr
        
        row = {'ID': name}
        row.update(param_cols)
        row.update(risk)
        row.update(year_cols)
        rows.append(row)
    
    if not rows:
        logger.info("没有有效的回测结果。")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df.set_index('ID', inplace=True)
    
    param_keys = list(params_def.keys())
    
    indicator_order = ['Return', 'Ann.Ret', 'MaxDD', 'Calmar', 'Sharpe', 'Volatility', 'WinRate', 'Trades', 'AvgHoldDays', 'Turnover']
    indicator_keys = [k for k in indicator_order if k in df.columns]
    
    year_keys = sorted([c for c in df.columns if c.startswith('Y') and c[1:].isdigit()])
    
    new_columns = param_keys + indicator_keys + year_keys
    remaining_cols = [c for c in df.columns if c not in new_columns]
    df = df[new_columns + remaining_cols]
    
    if sort_by in df.columns:
        df.sort_values(sort_by, ascending=ascending, inplace=True)
    
    return df


def format_table(df):
    """格式化 DataFrame 用于打印。"""
    fmt = df.copy()
    pct_cols = ['Return', 'Ann.Ret', 'MaxDD', 'Volatility', 'WinRate']
    float_cols = ['Calmar', 'Sharpe', 'Turnover']
    
    year_cols = [c for c in fmt.columns if c.startswith('Y') and c[1:].isdigit()]
    pct_cols = pct_cols + year_cols
    
    for col in pct_cols:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) and not isinstance(x, bool) else x)
    
    for col in float_cols:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else x)
    
    if 'Trades' in fmt.columns:
        fmt['Trades'] = fmt['Trades'].apply(lambda x: f"{int(x)}" if isinstance(x, (int, float)) else x)

    if 'AvgHoldDays' in fmt.columns:
        fmt['AvgHoldDays'] = fmt['AvgHoldDays'].apply(
            lambda x: f"{x:.1f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else x
        )
    
    return fmt


def print_compare(df, sort_by='Calmar', ascending=False):
    """打印格式化表格。"""
    if df.empty:
        return
    
    fmt = format_table(df)
    
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


def jupyter_display_compare(df, sort_by='Calmar', ascending=False):
    """在 Jupyter Notebook 中展示格式化对比表。"""
    if df.empty:
        return df

    if not IPYTHON_AVAILABLE:
        print_compare(df, sort_by=sort_by, ascending=ascending)
        return df

    fmt = format_table(df)
    order_str = '升序' if ascending else '降序'
    display(HTML(f"<b>参数优化对比表 (按 {sort_by} {order_str} 排列)</b>"))
    with pd.option_context('display.max_columns', None):
        display(fmt)
    display(HTML(f"<i>共 {len(df)} 组参数</i>"))
    return df


def markdown_table_print(df, sort_by='Calmar', ascending=False):
    """输出可复制的 Markdown 表格。"""
    if df.empty:
        return df

    if not TABULATE_AVAILABLE:
        print_compare(df, sort_by=sort_by, ascending=ascending)
        return df

    fmt = format_table(df)
    order_str = '升序' if ascending else '降序'
    md_title = f"### 参数优化对比表 (按 {sort_by} {order_str} 排列)\n"
    md_table = tabulate(fmt, headers='keys', tablefmt='pipe', showindex=True)
    md_footer = f"\n\n*共 {len(df)} 组参数*"
    print(md_title)
    print(md_table)
    print(md_footer)
    return df


def analyze_results(mapper_path, config_path, sort_by='Calmar', ascending=False, yearly=False, output='print'):
    """一站式分析函数。
    
    读取 mapper.json 和 YAML 配置，生成对比表格。
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    results = load_results(mapper_path)
    if not results:
        logger.info("No completed results found.")
        return pd.DataFrame()
    
    params_def = cfg.get('params', {})
    df = compare_params(results, params_def, sort_by=sort_by, ascending=ascending, yearly=yearly)

    if output == 'jupyter':
        jupyter_display_compare(df, sort_by=sort_by, ascending=ascending)
    elif output == 'markdown':
        markdown_table_print(df, sort_by=sort_by, ascending=ascending)
    else:
        print_compare(df, sort_by=sort_by, ascending=ascending)
    
    return df


def get_best_config(results, params_def, sort_by='Calmar'):
    """从结果中找到指标最优的配置。
    
    返回: (best_name, best_params) 或 (None, None)
    """
    df = compare_params(results, params_def, sort_by=sort_by, ascending=False)
    if df.empty:
        return None, None
    
    best_name = df.index[0]
    best_params = next((params for name, params, _, _ in results if name == best_name), None)
    return best_name, best_params



def nb_analyze(mapper_path, config_path, sort_by='Calmar', ascending=False, yearly=False):
    """
    Jupyter Notebook 友好的分析入口函数。

    在 JQ 研究环境的 Notebook Cell 中直接调用即可:

        from backtest_executor import nb_analyze
        df = nb_analyze(
            'backtest_executor/results/ETF_gao_opt/mapper.json',
            'backtest_executor/config/etf_gao.yaml'
        )

    Args:
        mapper_path (str): mapper.json 文件路径。
        config_path (str): YAML 配置文件路径。
        sort_by (str): 排序字段，默认 'Calmar'。
        ascending (bool): 是否升序，默认 False（降序）。
        yearly (bool): 是否增加年度收益列。

    Returns:
        pd.DataFrame: 包含参数和指标的对比表格。
    """
    return analyze_results(
        mapper_path,
        config_path,
        sort_by=sort_by,
        ascending=ascending,
        yearly=yearly,
        output='jupyter'
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='参数优化结果分析工具')
    parser.add_argument('mapper', help='mapper.json 文件路径')
    parser.add_argument('config', nargs='?', help='YAML 配置文件路径 (可选)')
    parser.add_argument('--sort', '-s', default='Calmar', help='排序字段 (默认: Calmar)')
    parser.add_argument('--ascending', '-a', action='store_true', help='升序排列')
    parser.add_argument('--yearly', action='store_true', help='显示年度收益列')
    parser.add_argument('--output', choices=['print', 'jupyter', 'markdown'], default='print', help='输出格式')

    args = parser.parse_args()

    if args.config:
        df = analyze_results(
            args.mapper,
            args.config,
            sort_by=args.sort,
            ascending=args.ascending,
            yearly=args.yearly,
            output=args.output
        )
    else:
        results = load_results(args.mapper)
        logger.info("Loaded %s results", len(results))
        for name, params, metrics in results[:5]:
            logger.info("  %s: %s", name, params)
