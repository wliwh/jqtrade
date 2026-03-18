
import sys
import os

# 导入 inject_params_to_code
# 因为是在 jqtrade 目录下，我们需要确保路径正确
sys.path.append(os.getcwd())
from backtest_executor.executor import inject_params_to_code

def test_duplicate_injection():
    original_code = "import os\nEXECUTION_PARAM = 1\nprint('hello')\n"
    
    # 第一次注入
    params1 = {"PARAM": 10}
    injected1 = inject_params_to_code(original_code, params1)
    print("--- First Injection ---")
    print(injected1)
    
    # 第二次在已注入的代码基础上再次注入
    params2 = {"PARAM": 20}
    injected2 = inject_params_to_code(injected1, params2)
    print("\n--- Second Injection (on top of first) ---")
    print(injected2)

    # 检查是否有重复的 Header
    header_marker = "[Optimizer Generated Parameters]"
    if injected2.count(header_marker) > 1:
        print("\n[DETECTED] Duplicate header detected!")
    else:
        print("\n[OK] No duplicate header.")

if __name__ == "__main__":
    test_duplicate_injection()
