"""
backtest_executor 包初始化文件。

在 JQ Jupyter Notebook 中使用示例:

    import sys
    sys.path.insert(0, '/path/to/jqtrade')  # 根据实际路径修改

    # 运行参数优化（自动使用 JQ 全局的 create_backtest/get_backtest）
    from backtest_executor import nb_run
    nb_run('backtest_executor/config/etf_7star_opt_dynamic.yaml',
           'round1a_switches')

    # 分析结果
    from backtest_executor import nb_analyze
    df = nb_analyze(
        'backtest_executor/results/ETF_7star_opt_dynamic/mapper.json',
        'backtest_executor/config/etf_7star_opt_dynamic.yaml'
    )
"""

from .optimize import nb_run, run_optimization
from .analyzer import nb_analyze, analyze_results, load_results

__all__ = [
    'nb_run',
    'run_optimization',
    'nb_analyze',
    'analyze_results',
    'load_results',
]
