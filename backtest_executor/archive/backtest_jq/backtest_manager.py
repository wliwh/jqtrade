from jqdata import *
import time
import json
import os
from generate_params import config_to_execution_params

ID_Save_Path = './ETFs/saved_name_id_mapper.json'
Base_Back_Id = '6e21bbaee8cc3f8423def84436a2bf49'

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

def batch_run(strategy_file, params_list, save_path=ID_Save_Path,
             start_day='2018-01-01', end_day='2026-01-10',
             initial_cash=100000, use_credit=False, sleep_between=2):
    """批量提交回测并收集结果。
    
    strategy_file: str, 策略文件路径 (如 'ETFs/ETF_gao_opt.py')
    params_list:   list of (name, config_dict), generate_stage_1/2 的输出
    save_path:     str, name→id 映射 JSON 路径 (断点续跑)
    sleep_between: int, 两次提交间休眠秒数
    
    返回: list of (name, config_dict, backtest_id)，供 analyze_simple.compare_params 使用
    """
    saved_names = get_name_id_mapper(save_path=save_path)
    results = []
    total = len(params_list)
    
    for idx, (name, config) in enumerate(params_list, 1):
        print(f"\n[{idx}/{total}] {name}")
        
        # 转换参数并生成策略代码
        exec_params = config_to_execution_params(config)
        code = generate_strategy_code(strategy_file, exec_params)
        if not code:
            print(f"  ✗ 策略代码生成失败，跳过")
            continue
        
        # 提交回测
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
            # 持久化映射
            get_name_id_mapper(update={name: bt_id}, save_path=save_path)
        else:
            print(f"  ✗ 回测失败: {name}")
        
        # 限流休眠 (最后一个不需要)
        if idx < total:
            time.sleep(sleep_between)
    
    print(f"\n=== 批量回测完成: {len(results)}/{total} 成功 ===")
    return results