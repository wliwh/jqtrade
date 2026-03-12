import os
import json
import time
import hashlib
import ast

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
        print(f"  [WARNING] AST Hash failed: {e}. Falling back to raw MD5.")
        return hashlib.md5(strategy_code.strip().encode('utf-8')).hexdigest()

def inject_params_to_code(strategy_code: str, params: dict) -> str:
    """
    通过头部注入方式将参数字典注入策略源码 (兼容 Python 3.6)。
    """
    lines = strategy_code.splitlines()
    try:
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
        
        # 从后往前标记，避免行号偏移（虽然这里用 None 标记不用担心偏移）
        for start, end in ranges_to_remove:
            for i in range(start - 1, end):
                if i < len(lines):
                    lines[i] = None
        
        filtered_lines = [line for line in lines if line is not None]
    except Exception as e:
        print(f"  [WARNING] AST Injection failed: {e}. Falling back to simple filtering.")
        filtered_lines = [l for l in lines if not l.strip().startswith('EXECUTION_')]

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
            print(f"  [INIT] Initializing metadata for strategy: {rel_path}")
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
                print("\n" + "!"*60)
                print("[CRITICAL] Strategy logic has been modified!")
                print(f"Expected Hash: {old_hash}")
                print(f"Current Hash:  {current_hash}")
                print("If you intentionally changed the strategy, please delete 'mapper.json'.")
                print("!"*60 + "\n")
                raise RuntimeError("Strategy integrity check failed.")

    def run_single_task(self, name: str, params: dict, 
                        create_bt_func, get_bt_func, 
                        start_day='2018-01-01', end_day='2026-01-10', 
                        initial_cash=100000, frequency='day', verbose=True):
        if name in self.records["runs"]:
            run_info = self.records["runs"][name]
            if run_info.get("status") == "done":
                if verbose: print(f"  [SKIPPED] Task '{name}' already completed.")
                return run_info["bt_id"]

        with open(self.strategy_file, 'r', encoding='utf-8') as f:
            original_code = f.read()
        
        injected_code = inject_params_to_code(original_code, params)
        
        if verbose: print(f"  [SUBMITTING] Task '{name}'...")
        bt_id = create_bt_func(
            algorithm_id=None, 
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

        self.records["runs"][name] = {
            "bt_id": bt_id,
            "status": "running",
            "params": params,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_records()

        while True:
            bt = get_bt_func(bt_id)
            # 兼容不同版本的 JQ API，有些返回对象，有些返回字典
            status = bt.get_status() if hasattr(bt, 'get_status') else bt['status']
            if verbose: print(f"  [POLLING] Task '{name}' ({bt_id}): {status}")
            
            if status == 'done':
                self.records["runs"][name]["status"] = "done"
                risk = bt.get_risk() if hasattr(bt, 'get_risk') else {}
                self.records["runs"][name]["metrics"] = {
                    "annual_return": risk.get("annual_return"),
                    "max_drawdown": risk.get("max_drawdown"),
                    "sharpe": risk.get("sharpe"),
                    "calmar": risk.get("annual_return") / abs(risk.get("max_drawdown")) if risk.get("max_drawdown") and risk.get("max_drawdown") != 0 else 0
                }
                self._save_records()
                if verbose: print(f"  [SUCCESS] Task '{name}' Done.")
                return bt_id
            elif status in ['failed', 'canceled', 'deleted']:
                self.records["runs"][name]["status"] = status
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
