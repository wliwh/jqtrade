from jqdata import *
import time
import json
import os
import itertools
import pandas as pd
import numpy as np

ID_Save_Path = './ETFs/saved_name_id_mapper.json'
Base_Back_Id = '6e21bbaee8cc3f8423def84436a2bf49'

def get_name(short_name, value, Rules):
    rule = next(r for r in Rules if r[0] == short_name)
    return rule[3](value)

def make_id_name(config, Rules, Baseline):
    """根据配置生成 ID 名称，按 Baseline 中的参数顺序排列"""
    toggleable = {r[0] for r in Rules if isinstance(r[2][0][0], bool)}
    tokens = []
    for key in Baseline.keys():
        if key in config:
            val = config[key]
            if key in toggleable and val[0] == False:
                continue
            n = get_name(key, val, Rules)
            if n: tokens.append(n)
    return "_".join(tokens)

def generate_stage_1(Rules, Baseline, mode='cartesian', exclude=None):
    """第一阶段：确定开关项的最优组合。
    """
    if exclude is None:
        exclude = []
    
    params_list = []
    all_switches = [r[0] for r in Rules if len(r[2])==2]
    
    if mode == 'ablation':
        # --- 逐一反转模式 ---
        params_list.append((make_id_name(Baseline, Rules, Baseline), Baseline.copy()))
        for s in all_switches:
            config = Baseline.copy()
            rule_values = next(r[2] for r in Rules if r[0] == s)
            other_value = next(v for v in rule_values if v[0] != Baseline[s][0])
            config[s] = other_value
            params_list.append((make_id_name(config, Rules, Baseline), config))
    
    elif mode == 'cartesian':
        # --- 笛卡尔积模式 ---
        cart_switches = [s for s in all_switches if s not in exclude]
        switch_values = []
        for s in cart_switches:
            rule_values = next(r[2] for r in Rules if r[0] == s)
            switch_values.append(rule_values)
        
        for combo in itertools.product(*switch_values):
            config = Baseline.copy()
            for i, s in enumerate(cart_switches):
                config[s] = combo[i]
            params_list.append((make_id_name(config, Rules, Baseline), config))
    
    return params_list

def generate_stage_2(Rules, Baseline, enabled_switches, fixed_params):
    """第二阶段：核心寻优
    """
    toggleable = {r[0] for r in Rules if len(r[2])==2}
    range_items = [r[0] for r in Rules if len(r[2])>2]
    
    base_switches = {}
    for key in toggleable:
        rule_values = next(r[2] for r in Rules if r[0] == key)
        if key in enabled_switches:
            on_value = next(v for v in rule_values if v[0])
            base_switches[key] = on_value
        else:
            on_value = next(v for v in rule_values if not v[0])
            base_switches[key] = on_value # Baseline[key]
    
    fixed_dict = {}
    if fixed_params:
        for item in fixed_params:
            if isinstance(item, str):
                fixed_dict[item] = Baseline[item]
            elif isinstance(item, dict):
                fixed_dict.update(item)
    
    search_items = [item for item in range_items if item not in fixed_dict]
    
    combinations = []
    item_values = []
    for item in search_items:
        rule_values = next(r[2] for r in Rules if r[0] == item)
        item_values.append(rule_values)
    for combo in itertools.product(*item_values):
        config = base_switches.copy()
        config.update(fixed_dict)
        
        names = []
        for i, item in enumerate(search_items):
            config[item] = combo[i]
            
        full_name = make_id_name(config, Rules, Baseline)
        combinations.append((f"{full_name}", config))
        
    return combinations

def config_to_execution_params(config, Rules):
    """将短名参数字典转换为 EXECUTION_ 格式字典。
    """
    mapped = {}
    for short_name, value in config.items():
        full_name = next(r[1] for r in Rules if r[0] == short_name)
        mapped[full_name] = value
    return mapped

def get_name_id_mapper(update = {}, save_path = ID_Save_Path):
    if not os.path.exists(save_path):
        print('No Saved json file.')
        with open(save_path, 'w') as f:
            json.dump(update, f)
            print('Saved Name-ID json file.')
    with open(save_path, 'r') as f:
        res = json.load(f)
    res.update(update)
    with open(save_path, 'w') as f:
        json.dump(res, f)
        print('Saved Name-ID json file.')
    return res

def generate_strategy_code(strategy_file_path, params=dict(), print_para_line = False):
    """读取策略文件并替换占位符参数"""
    new_lines = []
    try:
        with open(strategy_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                # 检查当前行是否是我们要替换的参数定义行
                replaced = False
                if stripped.startswith('EXECUTION_') and '=' in stripped:
                    if print_para_line: print(stripped)
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
                            frequency='day',
                            saved_names = dict(),
                            use_credit = False,
                            verbose = True):
    if name in saved_names:
        if verbose: print(f"Backtest {name} already exists.")
        return {'status': 'saved'}, saved_names[name]
    else:
        backtest_id = create_backtest(Base_Back_Id, start_day, end_day, frequency=frequency,
                                      initial_cash=initial_cash, initial_positions=None, extras=None, name=name,
                                      code=code, benchmark=None, python_version=3, use_credit=use_credit)
        saved_names[name] = backtest_id
        if verbose: print(f"Create {name}-{backtest_id} backtest.")
        while True:
            gt = get_backtest(backtest_id)
            status = gt.get_status()
            if status == 'done':
                if verbose: print(f"Backtest {name} Done.")
                return {'status': 'done'}, backtest_id
            elif status in ['failed', 'canceled', 'deleted']:
                print(f"\n[CRITICAL] Backtest {name} ({backtest_id}) FAILED with status: {status}")
                return {'status': 'failed'}, None
            time.sleep(5)

def batch_run(strategy_file, params_list, Rules,
             prefix_name = None, save_path=ID_Save_Path,
             start_day='2018-01-01', end_day='2026-01-10',
             initial_cash=100000, use_credit=False, sleep_between=2):
    """批量提交回测并收集结果"""
    saved_names = get_name_id_mapper(save_path=save_path)
    results = []
    total = len(params_list)
    
    for idx, (name, config) in enumerate(params_list, 1):
        if prefix_name:
            name = f"{prefix_name}_{name}"
        print(f"\n[{idx}/{total}] {name}")
        
        exec_params = config_to_execution_params(config, Rules)
        code = generate_strategy_code(strategy_file, exec_params)
        if not code:
            print(f"  ✗ 策略代码生成失败，跳过")
            continue
        
        status_info, bt_id = wrapped_create_backtest(
            name, code,
            initial_cash=initial_cash,
            start_day=start_day,
            end_day=end_day,
            saved_names=saved_names,
            use_credit=use_credit
        )
        
        if status_info['status'] in ('done', 'saved'):
            results.append((name, config, bt_id))
            get_name_id_mapper(update={name: bt_id}, save_path=save_path)
        else:
            print(f"  ✗ 回测失败: {name}")
        
        if idx < total:
            time.sleep(sleep_between)
    
    print(f"\n=== 批量回测完成: {len(results)}/{total} 成功 ===")
    return results

def _parse_param_columns(config, Rules):
    """将参数字典解析为表格用的参数列值"""
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
        max_dd = metrics.get('max_drawdown', 0)
        
        # 计算 Calmar (避免除零)
        calmar = ann_ret / max_dd if max_dd > 0.001 else 0.0
        
        return {
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
    """从 JQ 平台获取每年收益率"""
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


def compare_params(tasks, Rules, sort_by='Calmar', ascending=False, yearly=False):
    """多参数回测结果对比表"""
    rows = []
    param_keys = [r[0] for r in Rules]
    
    print(f"正在获取 {len(tasks)} 组回测指标...")
    
    for name, config, bt_id in tasks:
        risk = _fetch_risk_metrics(bt_id)
        if risk is None:
            print(f"  跳过 {name} (无有效指标)")
            continue
        
        param_cols = _parse_param_columns(config, Rules)
        year_cols = {}
        if yearly:
            yr = _fetch_yearly_returns(bt_id)
            if yr:
                year_cols = yr
        row = {'ID': name}
        row.update(param_cols)
        row.update(risk)
        row.update(year_cols)
        rows.append(row)
    
    if not rows:
        print("没有有效的回测结果。")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df.set_index('ID', inplace=True)
    # 排序
    if sort_by in df.columns:
        df.sort_values(sort_by, ascending=ascending, inplace=True)
    
    return df


def format_table(df):
    """格式化 DataFrame 用于打印，百分比和数值对齐。"""
    fmt = df.copy()
    pct_cols = ['Ann.Ret', 'MaxDD', 'Volatility', 'WinRate']
    float_cols = ['Calmar', 'Sharpe', 'AvgHoldDays', 'Turnover']
    
    # 年收益列也用百分比格式
    year_cols = [c for c in fmt.columns if c.startswith('Y') and c[1:].isdigit()]
    pct_cols = pct_cols + year_cols
    
    for col in pct_cols:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)
    
    for col in float_cols:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    
    if 'Trades' in fmt.columns:
        fmt['Trades'] = fmt['Trades'].apply(lambda x: f"{int(x)}" if isinstance(x, (int, float)) else x)
    
    return fmt


def print_compare(tasks, Rules, sort_by='Calmar', ascending=False, yearly=False):
    """一键对比并打印格式化表格。"""
    df = compare_params(tasks, Rules, sort_by=sort_by, ascending=ascending, yearly=yearly)
    if df.empty:
        return df
    
    fmt = format_table(df)
    print("\n" + "=" * 120)
    print(f"参数优化对比表 (按 {sort_by} {'升序' if ascending else '降序'} 排列)")
    print("=" * 120)
    print(fmt.to_string())
    print("=" * 120)
    print(f"共 {len(df)} 组参数")
    
    return df


def get_best_config(tasks, Rules, sort_by='Calmar'):
    """从 tasks 中找到指标最优的配置并返回其 config 字典。"""
    df = compare_params(tasks, Rules, sort_by=sort_by, ascending=False)
    if df.empty:
        return None, None
    
    best_name = df.index[0]
    best_config = next(cfg for name, cfg, _ in tasks if name == best_name)
    return best_name, best_config