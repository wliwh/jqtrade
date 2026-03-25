import os
import sys
import argparse
import yaml
import json
import itertools
import random
import time
import hashlib
import logging
from .logger import logger

try:
    from jqdata import create_backtest, get_backtest
    JQ_AVAILABLE = True
except ImportError:
    JQ_AVAILABLE = False

# 兼容包导入 (from backtest_executor import ...) 和直接运行 (python optimize.py) 两种方式
try:
    from .executor import BacktestExecutorV3
except ImportError:
    from executor import BacktestExecutorV3

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

def get_param_id(short_params, max_total_len=64):
    """
    生成一个简短且具有人类可读性的标识符作为回测名称。
    支持多种类型（None, bool, float, list, dict 等）并进行递归处理。
    同时控制总长度不超限。
    """
    def format_val(v):
        if v is None:
            return "N"
        if isinstance(v, bool):
            return "T" if v else "F"
        if isinstance(v, (int, float)):
            s = str(v)
            if "." in s:
                s = s.rstrip("0").rstrip(".")  # 移除末尾无意义的 0
            if s.startswith("0."):
                s = s[1:]  # 0.95 -> .95
            return s
        if isinstance(v, (list, tuple, set)):
            return "-".join(format_val(x) for x in v)
        if isinstance(v, dict):
            # 字典递归：k1v1-k2v2，键截断取前 3 位
            return "-".join(f"{str(k)[:3]}{format_val(v[k])}" for k in sorted(v.keys()))
        # 字符串或其他类型：只保留字母数字，且截断取前 6 位
        s_v = "".join(c for c in str(v) if c.isalnum())
        return s_v[:6]

    parts = []
    for k, v in sorted(short_params.items()):
        # 键名截断（通常 YAML 里的参数 ID 较短，为保险取前 6 位）
        k_short = str(k)[:6]
        parts.append(f"{k_short}{format_val(v)}")
    
    res = "_".join(parts)
    
    # 长度控制：如果超过阈值，截断并追加哈希后缀以保证唯一性
    if len(res) > max_total_len:
        h_suffix = hashlib.md5(res.encode('utf-8')).hexdigest()[:6]
        res = res[:max_total_len - 7] + "_" + h_suffix
        
    return res

def run_optimization(config_path, round_name, create_bt_func, get_bt_func):
    """
    主入口函数。
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 1. 查找指定的 Round
    round_cfg = next((r for r in cfg['rounds'] if r['name'] == round_name), None)
    if not round_cfg:
        logger.error("Round '%s' not found in %s", round_name, config_path)
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
    logger.info("Round '%s' generated %s combinations.", round_name, len(combos))

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

        logger.info("[PROGRESS] %s/%s: %s", i+1, len(combos), task_name)
        
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

    logger.info("Round '%s' completed. Results in %s", round_name, mapper_path)


def main():
    parser = argparse.ArgumentParser(description='策略参数优化工具')
    parser.add_argument('--config', '-c', required=True, help='YAML 配置文件路径')
    parser.add_argument('--round', '-r', required=True, help='要执行的轮次名称')
    parser.add_argument('--base-id', '-b', help='JoinQuant base_id (可选)')
    
    args = parser.parse_args()
    
    if not JQ_AVAILABLE:
        logger.error("jqdata not available. Please run in JQ environment.")
        sys.exit(1)
    
    run_optimization(args.config, args.round, create_backtest, get_backtest)


def nb_run(config_path, round_name):
    """
    Jupyter Notebook 友好的入口函数。

    自动从 JQ 全局命名空间获取 create_backtest / get_backtest，
    无需手动传入。在 JQ 研究环境的 Notebook Cell 中直接调用即可:

        from backtest_executor import nb_run
        nb_run('backtest_executor/config/etf_gao.yaml', 'round1_grid')

    Args:
        config_path (str): YAML 配置文件路径。
        round_name (str): 要执行的轮次名称（对应 YAML 中 rounds[].name）。
    """
    # 在 JQ Notebook 中，create_backtest / get_backtest 由平台注入全局命名空间
    # 通过 __builtins__ 或直接 import 均可获取
    import builtins
    _create_bt = getattr(builtins, 'create_backtest', None)
    _get_bt = getattr(builtins, 'get_backtest', None)

    # 如果全局没有，尝试从 jqdata 导入（兼容本地测试）
    if _create_bt is None or _get_bt is None:
        try:
            from jqdata import create_backtest as _create_bt, get_backtest as _get_bt
        except ImportError:
            raise RuntimeError(
                "[ERROR] 无法获取 create_backtest / get_backtest。"
                "请确认在 JQ 研究环境中运行，或已安装 jqdata。"
            )

    run_optimization(config_path, round_name, _create_bt, _get_bt)


if __name__ == '__main__':
    main()
