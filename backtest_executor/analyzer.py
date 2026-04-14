"""参数优化结果分析模块。

从 backtest_jq/analyze_simple.py 复制并解耦。
从 YAML 配置读取参数定义，而非 generate_params。

在 JQ 研究环境中运行。
"""
import argparse
import builtins
from collections import Counter
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
except ImportError:
    get_backtest = None


def _resolve_get_backtest():
    """在运行时解析 get_backtest，兼容 JQ Notebook 的延迟注入。"""
    if callable(get_backtest):
        return get_backtest

    builtins_get_backtest = getattr(builtins, 'get_backtest', None)
    if callable(builtins_get_backtest):
        return builtins_get_backtest

    try:
        from jqdata import get_backtest as jq_get_backtest
        return jq_get_backtest
    except ImportError:
        return None


def _compress_display_ids(index_values, max_len=40):
    """压缩展示用 ID，优先去掉共有前缀，再对过长部分做中间省略。"""
    labels = [str(v) for v in index_values]
    if not labels:
        return labels

    prefix = os.path.commonprefix(labels)
    shared_prefix = ''
    if len(prefix) >= 8 and '_' in prefix:
        shared_prefix = prefix[:prefix.rfind('_') + 1]

    if shared_prefix:
        trimmed = [label[len(shared_prefix):] or label for label in labels]
        if len(set(trimmed)) == len(trimmed):
            labels = trimmed

    compressed = []
    for label in labels:
        if len(label) <= max_len:
            compressed.append(label)
            continue
        head_len = max(10, max_len // 3)
        tail_len = max(12, max_len - head_len - 1)
        compressed.append(label[:head_len] + '…' + label[-tail_len:])

    if len(set(compressed)) != len(compressed):
        return labels

    return compressed


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
    get_bt = _resolve_get_backtest()
    if not callable(get_bt):
        return _mock_risk_metrics()
    
    try:
        bt = get_bt(backtest_id)
        status = bt.get_status() if hasattr(bt, 'get_status') else bt.get('status') if isinstance(bt, dict) else None
        if not bt or status != 'done':
            return None
        metrics = bt.get_risk() if hasattr(bt, 'get_risk') else bt.get('risk', {}) if isinstance(bt, dict) else None
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


def _fetch_yearly_returns_with_reason(backtest_id):
    """从 JQ 平台获取每年收益率，并返回失败原因。"""
    get_bt = _resolve_get_backtest()
    if not callable(get_bt):
        return None, 'jq_unavailable'

    def _yearly_from_monthly(monthly, source_name):
        if monthly is None:
            return None, 'monthly_missing'

        df_m = monthly.to_frame(name='ret').copy()
        df_m = df_m[pd.notnull(df_m['ret'])]
        if df_m.empty:
            return None, 'monthly_empty'

        years = pd.to_datetime(df_m.index, errors='coerce').year
        if pd.isnull(years).all():
            years = pd.Series(df_m.index.astype(str), index=df_m.index).str.extract(r'(\d{4})', expand=False)
            years = pd.to_numeric(years, errors='coerce')

        df_m['Year'] = years
        df_m = df_m[pd.notnull(df_m['Year'])]
        if df_m.empty:
            logger.warning("年度收益提取失败 (%s): %s 的索引中无法解析年份。", backtest_id, source_name)
            return None, 'year_parse_failed'

        df_m['Year'] = df_m['Year'].astype(int)
        yearly = df_m.groupby('Year')['ret'].apply(lambda x: np.prod(1 + x) - 1)
        return ({f'Y{y}': r for y, r in yearly.items()}, None) if not yearly.empty else (None, 'yearly_empty')

    try:
        bt = get_bt(backtest_id)
        if not bt:
            return None, 'backtest_missing'
    except Exception as e:
        logger.warning("获取年度收益失败 (%s): get_backtest() 异常: %s", backtest_id, e)
        return None, 'get_backtest_error'

    try:
        if hasattr(bt, 'get_period_risks'):
            risks = bt.get_period_risks()
            if risks:
                monthly = None
                candidate_keys = ['algorithm_return', 'algo_return', 'returns', 'return']

                for key in candidate_keys:
                    if key not in risks:
                        continue
                    src = risks[key]

                    if isinstance(src, pd.Series):
                        monthly = src
                        break

                    if isinstance(src, pd.DataFrame):
                        preferred_cols = ['one_month', 'monthly', 'month', 'return']
                        for col in preferred_cols:
                            if col in src.columns:
                                monthly = src[col]
                                break
                        if monthly is None and len(src.columns) == 1:
                            monthly = src.iloc[:, 0]
                        if monthly is not None:
                            break

                yearly, reason = _yearly_from_monthly(monthly, 'get_period_risks()')
                if yearly:
                    return yearly, None

                logger.warning(
                    "年度收益提取失败 (%s): get_period_risks() 中未找到可用月收益列，可用键: %s；将尝试回退到 get_results()。",
                    backtest_id,
                    list(risks.keys()) if hasattr(risks, 'keys') else type(risks).__name__
                )
    except Exception as e:
        logger.warning("年度收益提取失败 (%s): get_period_risks() 异常: %s；将尝试回退到 get_results()。", backtest_id, e)

    try:
        if not hasattr(bt, 'get_results'):
            logger.warning("年度收益提取失败 (%s): backtest 对象不支持 get_results()。", backtest_id)
            return None, 'get_results_unsupported'

        results = bt.get_results()
        if not results:
            return None, 'results_empty'

        df_res = pd.DataFrame(results)
        if 'time' not in df_res.columns or 'returns' not in df_res.columns:
            logger.warning(
                "年度收益提取失败 (%s): get_results() 缺少 time/returns 列，实际列: %s",
                backtest_id,
                list(df_res.columns)
            )
            return None, 'results_missing_columns'

        df_res['time'] = pd.to_datetime(df_res['time'], errors='coerce')
        df_res = df_res[pd.notnull(df_res['time'])].copy()
        if df_res.empty:
            return None, 'results_time_empty'

        df_res.sort_values('time', inplace=True)
        total_ret = pd.to_numeric(df_res['returns'], errors='coerce').fillna(0.0)
        daily_ret = (1 + total_ret).pct_change().fillna(0.0)
        if len(daily_ret) > 0:
            daily_ret.iloc[0] = total_ret.iloc[0]

        monthly = daily_ret.groupby(df_res['time'].dt.to_period('M')).apply(lambda x: np.prod(1 + x) - 1)
        monthly.index = monthly.index.astype(str)
        yearly, reason = _yearly_from_monthly(monthly, 'get_results()')
        if yearly:
            return yearly, None
        return None, reason or 'yearly_build_failed'
    except Exception as e:
        logger.warning("年度收益提取失败 (%s): get_results() 回退方案异常: %s", backtest_id, e)
        return None, 'get_results_error'


def _fetch_yearly_returns(backtest_id):
    yearly, _ = _fetch_yearly_returns_with_reason(backtest_id)
    return yearly


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
    yearly_cols_found = False
    missing_bt_id_count = 0
    yearly_failure_reasons = Counter()
    
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
            yr, reason = _fetch_yearly_returns_with_reason(bt_id)
            if yr:
                year_cols = yr
                yearly_cols_found = True
            elif reason:
                yearly_failure_reasons[reason] += 1
        elif yearly:
            missing_bt_id_count += 1
        
        row = {'ID': name}
        row.update(param_cols)
        row.update(risk)
        row.update(year_cols)
        rows.append(row)
    
    if not rows:
        logger.info("没有有效的回测结果。")
        return pd.DataFrame()

    if yearly and not yearly_cols_found:
        reason_summary = ', '.join(f'{k}:{v}' for k, v in yearly_failure_reasons.most_common()) or 'unknown'
        logger.warning(
            "yearly=True，但没有获取到任何年度收益列。缺少 bt_id 的结果数: %s。"
            "年度收益提取失败统计: %s。"
            "当前代码已兼容多种 get_period_risks() 结构，也兼容 get_backtest() 返回 dict/object 两种形态。",
            missing_bt_id_count,
            reason_summary
        )
    
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

    fmt.index = pd.Index(_compress_display_ids(fmt.index), name=fmt.index.name)
    
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
    display(HTML(fmt.to_html()))
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



def nb_analyze(mapper_path, config_path, sort_by='Calmar', ascending=False, yearly=False, output='jupyter'):
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
        output (str): 输出格式，可选 jupyter|markdown|print

    Returns:
        pd.DataFrame: 包含参数和指标的对比表格。
    """
    return analyze_results(
        mapper_path,
        config_path,
        sort_by=sort_by,
        ascending=ascending,
        yearly=yearly,
        output=output
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
