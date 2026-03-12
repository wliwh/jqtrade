import os
import yaml
import json
import itertools
import random
import time
import hashlib
from .executor import BacktestExecutorV3

class ParameterGenerator:
    def __init__(self, params_def):
        self.params_def = params_def

    def generate(self, round_cfg):
        method = round_cfg.get('method', 'grid')
        if method == 'grid':
            return self._generate_grid(round_cfg)
        elif method == 'random':
            return self._generate_random(round_cfg)
        elif method == 'list':
            return self._generate_list(round_cfg)
        elif method == 'sensitivity':
            return self._generate_sensitivity(round_cfg)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _generate_grid(self, round_cfg):
        keys = round_cfg['search']
        values_lists = [self.params_def[k]['values'] for k in keys]
        for combo in itertools.product(*values_lists):
            yield dict(zip(keys, combo))

    def _generate_random(self, round_cfg):
        count = round_cfg.get('count', 10)
        keys = round_cfg['search']
        # 简单随机采样（可能有重复，依靠 mapper 去重）
        for _ in range(count):
            yield {k: random.choice(self.params_def[k]['values']) for k in keys}

    def _generate_list(self, round_cfg):
        for combo in round_cfg.get('combinations', []):
            yield combo

    def _generate_sensitivity(self, round_cfg):
        base = round_cfg.get('base', {})
        keys = round_cfg['search']
        yield base
        for k in keys:
            for val in self.params_def[k]['values']:
                if val != base.get(k):
                    new_combo = base.copy()
                    new_combo[k] = val
                    yield new_combo

def get_param_id(short_params):
    """生成一个简短且唯一的标识符用于回测名称"""
    parts = []
    for k, v in sorted(short_params.items()):
        if isinstance(v, (list, tuple)):
            # 处理像 [True, 0.95] 这样的参数 -> T95
            val_str = "".join([str(x)[0] if isinstance(x, bool) else str(x).replace("0.", ".") for x in v])
        elif isinstance(v, bool):
            val_str = "T" if v else "F"
        else:
            val_str = str(v)
        parts.append(f"{k}{val_str}")
    return "_".join(parts)

def run_optimization(config_path, round_name, create_bt_func, get_bt_func):
    """
    主入口函数。
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 1. 查找指定的 Round
    round_cfg = next((r for r in cfg['rounds'] if r['name'] == round_name), None)
    if not round_cfg:
        print(f"[ERROR] Round '{round_name}' not found in {config_path}")
        return

    # 2. 初始化执行器
    strategy_file = cfg['strategy']['file']
    # 结果存放在 results/{strategy_name}/mapper.json
    strategy_name = os.path.splitext(os.path.basename(strategy_file))[0]
    result_dir = os.path.join("backtest_executor", "results", strategy_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    mapper_path = os.path.join(result_dir, "mapper.json")
    executor = BacktestExecutorV3(mapper_path, strategy_file)

    # 3. 生成参数组合
    gen = ParameterGenerator(cfg['params'])
    combos = list(gen.generate(round_cfg))
    print(f"[INFO] Round '{round_name}' generated {len(combos)} combinations.")

    # 4. 准备基础参数 (Defaults + Round Fixed)
    base_params = {k: v.get('default') for k, v in cfg['params'].items()}
    if 'fixed' in round_cfg:
        base_params.update(round_cfg['fixed'])

    # 5. 执行循环
    for i, short_combo in enumerate(combos):
        # 合并当前组合
        current_short_params = base_params.copy()
        current_short_params.update(short_combo)
        
        # 转换为 EXECUTION_ 全名
        full_params = {}
        for k, v in current_short_params.items():
            var_name = cfg['params'][k]['var']
            full_params[var_name] = v

        # 生成任务名称
        param_id = get_param_id(short_combo) if short_combo else "BASE"
        task_name = f"{round_name}_{param_id}"

        print(f"\n[PROGRESS] {i+1}/{len(combos)}: {task_name}")
        
        # 执行（executor 自带去重和状态检查）
        executor.run_single_task(
            name=task_name,
            params=full_params,
            create_bt_func=create_bt_func,
            get_bt_func=get_bt_func,
            start_day=cfg['backtest'].get('start_day', '2018-01-01'),
            end_day=cfg['backtest'].get('end_day', '2026-01-10'),
            initial_cash=cfg['backtest'].get('initial_cash', 100000)
        )

    print(f"\n[FINISH] Round '{round_name}' completed. Results in {mapper_path}")
