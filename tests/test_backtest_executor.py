import os
import json
import shutil
import pytest
from unittest.mock import MagicMock

from backtest_executor.executor import (
    get_logical_hash, 
    extract_execution_params, 
    inject_params_to_code, 
    BacktestExecutorV3
)
from backtest_executor.optimize import ParameterGenerator, get_param_id

# --- AST & Tools Tests ---

def test_get_logical_hash():
    code_v1 = "import os\n# Comment\nEXECUTION_SCORE = 10\ndef run():\n    pass\n"
    code_v2 = "import os\nEXECUTION_SCORE = 99\n\ndef run():\n    pass\n"
    
    # 逻辑哈希应忽略注释、空行和 EXECUTION_ 参数变动
    assert get_logical_hash(code_v1) == get_logical_hash(code_v2)
    
    code_diff_logic = "import os\ndef run():\n    print('diff')\n"
    assert get_logical_hash(code_v1) != get_logical_hash(code_diff_logic)

def test_extract_execution_params():
    code = """
import os
EXECUTION_STR = "hello"
EXECUTION_INT = 123
EXECUTION_LIST = [1, 2, 3]
EXECUTION_DICT = {"a": 1}
def start(): pass
    """
    params = extract_execution_params(code)
    assert params["EXECUTION_STR"] == "hello"
    assert params["EXECUTION_INT"] == 123
    assert params["EXECUTION_LIST"] == [1, 2, 3]
    assert params["EXECUTION_DICT"] == {"a": 1}

def test_inject_params_to_code():
    code = "EXECUTION_P = 1\ndef start(): pass"
    params = {"EXECUTION_P": 2, "EXECUTION_NEW": "val"}
    injected = inject_params_to_code(code, params)
    
    assert "EXECUTION_P = 2" in injected
    assert "EXECUTION_NEW = 'val'" in injected
    # 旧的定义应该被由于注入头部的逻辑而“屏蔽”或过滤掉（取决于实现，当前实现是头部注入并过滤掉旧的）
    assert "EXECUTION_P = 1" not in injected

# --- Parameter Generation Tests ---

def test_parameter_generator_grid():
    params_def = {
        "A": {"values": [1, 2]},
        "B": {"values": ["x", "y"]}
    }
    gen = ParameterGenerator(params_def)
    combos = list(gen.generate({"method": "grid", "search": ["A", "B"]}))
    assert len(combos) == 4
    assert {"A": 1, "B": "x"} in combos

def test_parameter_generator_sensitivity():
    params_def = {
        "A": {"values": [1, 2, 3]},
        "B": {"values": ["x", "y"]}
    }
    gen = ParameterGenerator(params_def)
    # base 是 A=1, B=x。变动 A 会产生 A=2, A=3。
    combos = list(gen.generate({
        "method": "sensitivity", 
        "search": ["A"], 
        "base": {"A": 1, "B": "x"}
    }))
    assert len(combos) == 3 # base + (2, 3)
    assert {"A": 1, "B": "x"} in combos
    assert {"A": 2, "B": "x"} in combos

def test_get_param_id():
    assert get_param_id({"S": 0.5}) == "S.5"
    assert get_param_id({"B": True}) == "BT"
    assert get_param_id({"L": [True, 0.95]}) == "LT-.95"

# --- Executor Core Tests ---

@pytest.fixture
def temp_env(tmp_path):
    strat_file = tmp_path / "strategy.py"
    strat_file.write_text("EXECUTION_P1 = 'v1'\nEXECUTION_P2 = 100\ndef start(): pass")
    mapper_file = tmp_path / "mapper.json"
    return str(mapper_file), str(strat_file)

def test_executor_normalization_and_hashing(temp_env):
    mapper_path, strat_path = temp_env
    executor = BacktestExecutorV3(mapper_path, strat_path)
    
    mock_create = MagicMock(return_value="bt_auto_1")
    
    # 模拟 BT 对象
    mock_bt = MagicMock()
    mock_bt.get_status.return_value = "done"
    mock_bt.get_risk.return_value = {"annual_return": 0.1, "max_drawdown": 0.05, "sharpe": 2.0}
    
    mock_get = MagicMock(return_value=mock_bt)
    
    # 1. 隐式传参（使用源码默认值 v1）
    executor.run_single_task("Task1", {"P2": 200}, mock_create, mock_get)
    
    # 2. 显式传参（传默认值 v1）
    # 应该命中 Hash，不触发第二次创建
    executor.run_single_task("Task2", {"P1": "v1", "P2": 200}, mock_create, mock_get)
    
    assert mock_create.call_count == 1
    
    with open(mapper_path, "r") as f:
        data = json.load(f)
    assert len(data["runs"]) == 1

def test_executor_id_persistence(temp_env):
    mapper_path, strat_path = temp_env
    executor = BacktestExecutorV3(mapper_path, strat_path)
    
    mock_create = MagicMock(return_value="bt_id_999")
    
    mock_bt = MagicMock()
    mock_bt.get_status.return_value = "done"
    mock_bt.get_risk.return_value = {"annual_return": 0.1, "max_drawdown": 0.05, "sharpe": 2.0}
    
    mock_get = MagicMock(return_value=mock_bt)
    
    # 运行任务
    executor.run_single_task("MyTask", {"P1": "new"}, mock_create, mock_get)
    
    # 重新加载 Executor
    executor2 = BacktestExecutorV3(mapper_path, strat_path)
    # 再次运行相同参数
    executor2.run_single_task("AnotherNameSameParams", {"P1": "new"}, mock_create, mock_get)
    
    # 应该只调用了一次 create
    assert mock_create.call_count == 1
