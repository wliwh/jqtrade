
import pytest
import hashlib
from backtest_executor.optimize import get_param_id

def test_get_param_id_basic():
    # Int and Float
    assert get_param_id({"A": 10, "B": 20}) == "A10_B20"
    assert get_param_id({"S": 0.5}) == "S.5"
    assert get_param_id({"P": 1.50}) == "P1.5"
    
    # Boolean
    assert get_param_id({"B1": True, "B2": False}) == "B1T_B2F"
    
    # None
    assert get_param_id({"N": None}) == "NN"

def test_get_param_id_collections():
    # List / Tuple
    assert get_param_id({"L": [1, 2, 3]}) == "L1-2-3"
    assert get_param_id({"T": (True, 0.95)}) == "TT-.95"
    
    # Mixed
    assert get_param_id({"M": [1, "hello", False]}) == "M1-hello-F"

def test_get_param_id_dict():
    # Nested Dict
    params = {"D": {"a": 1, "b": 2}}
    # dict keys are truncated to 3 chars in format_val
    assert get_param_id(params) == "Da1-b2"
    
    # More complex dict
    params2 = {"CFG": {"threshold": 0.05, "enable": True}}
    # threshold -> thr, enable -> ena
    assert get_param_id(params2) == "CFGenaT-thr.05"

def test_get_param_id_string_truncation():
    # Long strings and keys
    params = {"EXTREME_LONG_KEY": "VERY_LONG_VALUE"}
    res = get_param_id(params)
    assert res == "EXTREMVERYLO"

def test_get_param_id_total_length():
    # Total length constraint (64 chars)
    # create enough params to exceed 64 chars
    params = {f"P{i:02d}": "V" * 10 for i in range(10)}
    # Each part: P00VVVVVV (3+6=9)
    # Total approx 10 * 9 + 9 = 99
    res = get_param_id(params)
    assert len(res) <= 64
    assert "_" in res
    # Checked that it contains a hash at the end
    last_part = res.split("_")[-1]
    assert len(last_part) == 6

def test_get_param_id_readability():
    # Nested list/dict mix
    params = {
        "POOL": ["000300.XSHG", "399006.XSHE"],
        "OPTS": {"ma": 20, "std": 2}
    }
    res = get_param_id(params)
    print(res)
    assert "000300" in res
    assert "399006" in res
    assert "ma20" in res


if __name__ == "__main__":
    test_get_param_id_readability()