import os
import json
import time
import re

import ast

# [DESIGN] Robust Header Injection System
def inject_params_to_code(strategy_code: str, params: dict) -> str:
    """
    通过头部注入方式将参数字典注入策略源码。
    1. 使用 AST 解析源码，找出所有以 EXECUTION_ 开头的全局变量赋值。
    2. 记录这些赋值所在的行号范围并将其剔除。
    3. 在源码最前端添加带有标记的参数定义块。
    """
    try:
        tree = ast.parse(strategy_code)
    except SyntaxError:
        # 如果代码本身有错，回退到普通处理（或抛出错误）
        print("  [WARNING] Syntax error in strategy code, skipping AST-based purging.")
        return strategy_code

    # 找出所有要删除的行号范围
    ranges_to_remove = []
    for node in tree.body:
        # 处理基本赋值 EXECUTION_X = ...
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith('EXECUTION_'):
                    ranges_to_remove.append((node.lineno, node.end_lineno))
                    break
        # 处理带类型注解的赋值 EXECUTION_X: int = ...
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id.startswith('EXECUTION_'):
                ranges_to_remove.append((node.lineno, node.end_lineno))

    # 按行分解代码
    lines = strategy_code.splitlines()
    
    # 将要删除的行标记为 None
    for start, end in ranges_to_remove:
        # ast 的行号是从 1 开始的
        for i in range(start - 1, end):
            if i < len(lines):
                lines[i] = None
    
    # 过滤掉被标记删除的行
    filtered_lines = [line for line in lines if line is not None]
    
    # 2. 生成新的参数定义块
    header_block = [
        "# " + "="*40,
        "# [Optimizer Generated Parameters]",
    ]
    for key, value in params.items():
        full_key = key if key.startswith('EXECUTION_') else f"EXECUTION_{key}"
        formatted_val = repr(value)
        header_block.append(f"{full_key} = {formatted_val}")
    
    header_block.append("# " + "="*40 + "\n")
    
    # 3. 合并
    new_code = "\n".join(header_block) + "\n".join(filtered_lines)
    return new_code

class BacktestExecutorV3:
    def __init__(self, record_path: str):
        self.record_path = record_path
        self.records = self._load_records()

    def _load_records(self):
        if os.path.exists(self.record_path):
            with open(self.record_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"metadata": {}, "runs": {}}

    def _save_records(self):
        with open(self.record_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)

    def run_single_task(self, strategy_file: str, name: str, params: dict, 
                        create_bt_func, get_bt_func, 
                        start_day='2018-01-01', end_day='2026-01-10', 
                        initial_cash=100000, frequency='day', verbose=True):
        """
        执行单个回测任务。
        create_bt_func: 对应 JQ 的 create_backtest
        get_bt_func: 对应 JQ 的 get_backtest
        """
        # [Instant Flush Support]
        if name in self.records["runs"]:
            run_info = self.records["runs"][name]
            if run_info.get("status") == "done":
                if verbose: print(f"  [SKIPPED] Task '{name}' already completed.")
                return run_info["bt_id"]

        # 读取并注入代码
        with open(strategy_file, 'r', encoding='utf-8') as f:
            original_code = f.read()
        
        injected_code = inject_params_to_code(original_code, params)
        
        # 提交回测
        if verbose: print(f"  [SUBMITTING] Task '{name}'...")
        bt_id = create_bt_func(
            algorithm_id=None, # 此处根据 JQ 实际 API 填写，通常需要一个 base_backtest_id
            start_date=start_day,
            end_date=end_day,
            frequency=frequency,
            initial_cash=initial_cash,
            code=injected_code,
            name=name
        )
        
        if not bt_id:
            if verbose: print(f"  [FAILED] Failed to submit task '{name}'.")
            return None

        # 记录初始状态
        self.records["runs"][name] = {
            "bt_id": bt_id,
            "status": "running",
            "params": params,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_records()

        # [Serial Polling]
        while True:
            bt = get_bt_func(bt_id)
            status = bt.get_status()
            if verbose: print(f"  [POLLING] Task '{name}' ({bt_id}): {status}")
            
            if status == 'done':
                self.records["runs"][name]["status"] = "done"
                # 此处可以进一步通过 bt.get_risk() 获取 metrics 并存储 (对应规划4)
                self._save_records()
                if verbose: print(f"  [SUCCESS] Task '{name}' Done.")
                return bt_id
            elif status in ['failed', 'canceled', 'deleted']:
                self.records["runs"][name]["status"] = status
                self._save_records()
                if verbose: print(f"  [ERROR] Task '{name}' FAILED with status: {status}")
                return None
            
            time.sleep(10) # 建议频率不要太高

def mock_batch_runner():
    """本地测试逻辑，不依赖 JQ 平台。"""
    print("Testing Header Injection Logic with Multi-line Parameters...")
    sample_code = """
import pandas as pd
# Multi-line param that should be removed
EXECUTION_SCORE_RANGE = {
    'min': 0.0,
    'max': 10.0,
    'weight': 1.0
} 
EXECUTION_MA_PARAM = True

def initialize(context):
    pass
    """
    params = {"EXECUTION_SCORE_RANGE": (0.0, 6.0), "EXECUTION_V": (True, 5, 0.6)}
    new_code = inject_params_to_code(sample_code, params)
    print("\n--- Injected Code Preview ---")
    print(new_code)
    print("--- End of Preview ---\n")

if __name__ == "__main__":
    mock_batch_runner()
