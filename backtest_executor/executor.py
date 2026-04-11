import os
import json
import time
import hashlib
import ast
import logging
from .logger import logger

Default_Back_Id = '6e21bbaee8cc3f8423def84436a2bf49'

def get_logical_hash(strategy_code: str) -> str:
    """
    计算策略逻辑的哈希值 (兼容 Python 3.6)。
    1. 使用 AST 解析代码。
    2. 移除所有 EXECUTION_ 开头的全局变量赋值。
    3. 使用 ast.dump 生成结构化字符串（忽略行号和注释）。
    4. 计算 MD5。
    """
    try:
        tree = ast.parse(strategy_code)
        
        # 过滤掉以 EXECUTION_ 开头的赋值
        new_body = []
        for node in tree.body:
            should_remove = False
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith('EXECUTION_'):
                        should_remove = True
                        break
            elif isinstance(node, (getattr(ast, 'AnnAssign', type(None)))): # 兼容 3.6 的 AnnAssign
                if isinstance(node.target, ast.Name) and node.target.id.startswith('EXECUTION_'):
                    should_remove = True
            
            if not should_remove:
                new_body.append(node)
        
        tree.body = new_body
        
        # ast.dump 在 3.6 中可用，include_attributes=False 确保不包含行号
        structural_repr = ast.dump(tree, include_attributes=False)
        return hashlib.md5(structural_repr.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.warning("AST Hash failed: %s. Falling back to raw MD5.", e)
        return hashlib.md5(strategy_code.strip().encode('utf-8')).hexdigest()

def extract_execution_params(strategy_code: str) -> dict:
    """
    从策略源码中提取所有 EXECUTION_ 开头的全局变量及其默认值 (兼容 Python 3.6)。
    """
    params = {}
    try:
        tree = ast.parse(strategy_code)
        for node in tree.body:
            target_id = None
            value_node = None
            
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith('EXECUTION_'):
                        target_id = target.id
                        value_node = node.value
                        break
            elif isinstance(node, (getattr(ast, 'AnnAssign', type(None)))):
                if isinstance(node.target, ast.Name) and node.target.id.startswith('EXECUTION_'):
                    target_id = node.target.id
                    value_node = node.value
            
            if target_id and value_node:
                try:
                    # 使用 literal_eval 安全提取字面量值
                    params[target_id] = ast.literal_eval(value_node)
                except ValueError:
                    # 如果不是字面量（例如函数调用），保留 None 或跳过，
                    # 但在回测配置中，EXECUTION_ 变量通常是简单值或配置字典
                    params[target_id] = None
    except Exception as e:
        logger.warning("Extracting execution params failed: %s", e)
    return params

def inject_params_to_code(strategy_code: str, params: dict) -> str:
    """
    通过头部注入方式将参数字典注入策略源码 (兼容 Python 3.6)。
    """
    lines = strategy_code.splitlines()
    
    # 剥离已存在的 [Optimizer Generated Parameters] 注释块及其分隔符
    for i in range(len(lines)):
        if lines[i] is None:
            continue
        if "[Optimizer Generated Parameters]" in lines[i]:
            lines[i] = None
            # 向上找起始分隔符
            if i > 0 and lines[i-1].strip().startswith("#"):
                lines[i-1] = None
            # 向下找结束分隔符
            curr = i + 1
            while curr < len(lines):
                l_strip = lines[curr].strip()
                if l_strip.startswith("#"):
                    lines[curr] = None
                    if "===" in l_strip:
                        break
                elif l_strip.startswith("EXECUTION_"):
                    lines[curr] = None
                else:
                    break
                curr += 1

    try:
        # 重新解析（此时已部分标记为 None 的 lines 仅用于最后过滤）
        # AST 解析仍然使用原始代码获取正确的行号
        tree = ast.parse(strategy_code)
        # Python 3.6 没有 end_lineno，我们通过获取下一个节点的 lineno 来推算范围
        nodes = sorted(tree.body, key=lambda x: x.lineno)
        ranges_to_remove = []
        
        for i, node in enumerate(nodes):
            is_exec_param = False
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith('EXECUTION_'):
                        is_exec_param = True
                        break
            elif isinstance(node, (getattr(ast, 'AnnAssign', type(None)))):
                if isinstance(node.target, ast.Name) and node.target.id.startswith('EXECUTION_'):
                    is_exec_param = True
            
            if is_exec_param:
                start_line = node.lineno
                # 如果有下一个节点，结束行是下一个节点的起始行减1；否则直到末尾
                end_line = nodes[i+1].lineno - 1 if i+1 < len(nodes) else len(lines)
                ranges_to_remove.append((start_line, end_line))
        
        # 标记 AST 识别出的参数定义行
        for start, end in ranges_to_remove:
            for idx in range(start - 1, end):
                if idx < len(lines):
                    lines[idx] = None
        
        filtered_lines = [line for line in lines if line is not None]
    except Exception as e:
        logger.warning("AST Injection failed: %s. Falling back to simple filtering.", e)
        # 兜底方案：过滤掉所有 EXECUTION 开头的行，以及我们已经标为 None 的行
        filtered_lines = [l for l in lines if l is not None and not l.strip().startswith('EXECUTION_')]

    header_block = [
        "# " + "="*50,
        f"# [Optimizer Generated Parameters] Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    for key, value in params.items():
        full_key = key if key.startswith('EXECUTION_') else f"EXECUTION_{key}"
        header_block.append(f"{full_key} = {repr(value)}")
    header_block.append("# " + "="*50 + "\n")
    
    return "\n".join(header_block) + "\n".join(filtered_lines)

class BacktestExecutorV3:
    def __init__(self, record_path: str, strategy_file: str):
        self.record_path = record_path
        self.strategy_file = os.path.abspath(strategy_file)
        self.records = self._load_records()
        self._verify_integrity()

    def _load_records(self):
        if os.path.exists(self.record_path):
            with open(self.record_path, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except:
                    return {"metadata": {}, "runs": {}}
        return {"metadata": {}, "runs": {}}

    def _save_records(self):
        with open(self.record_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)

    def _verify_integrity(self):
        if not os.path.exists(self.strategy_file):
            raise FileNotFoundError(f"Strategy file not found: {self.strategy_file}")

        with open(self.strategy_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        current_hash = get_logical_hash(code)
        rel_path = os.path.relpath(self.strategy_file)
        
        meta = self.records.get("metadata", {})
        if not meta:
            logger.info("Initializing metadata for strategy: %s", rel_path)
            self.records["metadata"] = {
                "strategy_path": rel_path,
                "strategy_logic_hash": current_hash,
                "python_version": "3.6_compat",
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            self._save_records()
        else:
            old_hash = meta.get("strategy_logic_hash")
            if old_hash and old_hash != current_hash:
                logger.critical("Strategy logic has been modified!")
                logger.critical("Expected Hash: %s", old_hash)
                logger.critical("Current Hash:  %s", current_hash)
                logger.critical("If you intentionally changed the strategy, please delete 'mapper.json'.")
                raise RuntimeError("Strategy integrity check failed.")

    def run_single_task(self, name: str, params: dict, 
                        create_bt_func, get_bt_func, 
                        start_day='2018-01-01', end_day='2026-01-10', 
                        initial_cash=100000, frequency='day', use_credit=True):
        # 读取源码
        with open(self.strategy_file, 'r', encoding='utf-8') as f:
            original_code = f.read()

        # 1. 参数归一化 (Normalization)
        # 获取源码中硬编码的默认参数，并用传入参数覆盖
        # 这样可以保证：即使 params 缺省了某些参数，哈希值和注入后的代码依然是完整的
        default_params = extract_execution_params(original_code)
        
        # 将传入参数转为 EXECUTION_ 前缀全名以便合并
        normalized_input = {}
        for k, v in params.items():
            full_key = k if k.startswith('EXECUTION_') else f"EXECUTION_{k}"
            normalized_input[full_key] = v
            
        full_params = default_params.copy()
        full_params.update(normalized_input)

        # 2. 生成基于全量执行参数的唯一 ID (task_id)
        execution_ctx = {
            "params": full_params,
            "start_day": start_day,
            "end_day": end_day,
            "initial_cash": initial_cash,
            "frequency": frequency
        }
        ctx_json = json.dumps(execution_ctx, sort_keys=True)
        task_id = hashlib.md5(ctx_json.encode('utf-8')).hexdigest()

        if task_id in self.records["runs"]:
            run_info = self.records["runs"][task_id]
            if run_info.get("status") == "done":
                logger.info("Task '%s' (ID: %s) already completed. (Skipped)", name, task_id)
                return run_info["bt_id"]

        injected_code = inject_params_to_code(original_code, full_params)
        
        logger.info("Submitting task '%s'...", name)
        bt_id = create_bt_func(
            algorithm_id=Default_Back_Id, 
            start_date=start_day,
            end_date=end_day,
            frequency=frequency,
            initial_cash=initial_cash,
            code=injected_code,
            name=name,
            benchmark=None,
            python_version=3,
            use_credit=use_credit
        )
        
        if not bt_id:
            logger.error("Failed to submit task '%s' (ID: %s).", name, task_id)
            return None

        self.records["runs"][task_id] = {
            "name": name,
            "bt_id": bt_id,
            "status": "running",
            "params": full_params,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_records()

        while True:
            bt = get_bt_func(bt_id)
            # 兼容不同版本的 JQ API，有些返回对象，有些返回字典
            status = bt.get_status() if hasattr(bt, 'get_status') else bt['status']
            logger.info("Polling task '%s' (BTID: %s, TaskID: %s): %s", name, bt_id, task_id, status)
            
            if status == 'done':
                self.records["runs"][task_id]["status"] = "done"
                risk = bt.get_risk() if hasattr(bt, 'get_risk') else {}
                annual_return = risk.get("annual_algo_return", 0)
                max_drawdown = risk.get("max_drawdown", 0)
                win_count = risk.get("win_count", 0) or 0
                lose_count = risk.get("lose_count", 0) or 0
                self.records["runs"][task_id]["metrics"] = {
                    "return": risk.get("algorithm_return", 0),
                    "annual_return": annual_return,
                    "max_drawdown": max_drawdown,
                    "sharpe": risk.get("sharpe"),
                    "calmar": annual_return / abs(max_drawdown) if max_drawdown and max_drawdown != 0 else 0,
                    "volatility": risk.get("algorithm_volatility", 0),
                    "win_rate": risk.get("win_ratio", 0),
                    "trades": win_count + lose_count,
                    "avg_hold_days": risk.get("avg_position_days", 0),
                    "turnover": risk.get("turnover_rate", 0),
                }
                self._save_records()
                logger.info("Task '%s' Done. (Success)", name)
                return bt_id
            elif status in ['failed', 'canceled', 'deleted']:
                self.records["runs"][task_id]["status"] = status
                self._save_records()
                return None
            
            time.sleep(10)

def mock_test():
    code_v1 = "import os\n# Comment\nEXECUTION_S = 10\ndef run():\n    print('hello')\n"
    code_v2 = "import os\nEXECUTION_S = 20\n\ndef run():\n    print('hello')\n"
    
    h1 = get_logical_hash(code_v1)
    h2 = get_logical_hash(code_v2)
    print(f"Hash V1: {h1}")
    print(f"Hash V2: {h2} (Match: {h1==h2})")
    
    # 测试多行参数注入
    code_multi = "import os\nEXECUTION_CONFIG = {\n    'a': 1,\n    'b': 2\n}\ndef start(): pass"
    injected = inject_params_to_code(code_multi, {"CONFIG": "new"})
    print("\n--- Injected Multi-line Test ---")
    print(injected)

if __name__ == "__main__":
    mock_test()
