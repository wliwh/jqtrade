"""策略参数优化主控程序。

在 JQ 研究环境中运行。将参数生成、批量回测、结果分析串联为一键流程。

使用方式:
  1. 修改下方 === 配置区 === 中的参数
  2. 在 JQ 研究环境中运行本脚本
"""
from generate_params import generate_stage_1, generate_stage_2, print_params
from backtest_manager import batch_run
from analyze_simple import print_compare, compare_params, get_best_config

# ==============================================================================
# === 配置区 ===
# ==============================================================================

# 策略文件路径 (相对于 JQ 研究环境的工作目录)
STRATEGY_FILE = 'ETFs/ETF_gao_opt.py'

# 回测时间范围
START_DAY = '2018-01-01'
END_DAY   = '2026-01-10'

# 初始资金
INITIAL_CASH = 100000

# name→id 映射保存路径
ID_SAVE_PATH = './ETFs/saved_name_id_mapper.json'

# 排序指标
SORT_BY = 'Calmar'

# 执行阶段: 'stage1' / 'stage2' / 'stage2_multi' / 'analyze_only'
RUN_MODE = 'stage1'

# --- Stage 1 配置 ---
STAGE1_MODE    = 'cartesian'   # 'cartesian' 或 'ablation'
STAGE1_EXCLUDE = ['dl']        # 从笛卡尔积中排除的开关

# --- Stage 2 配置 (单轮模式) ---
# 根据 Stage 1 的结果填入最优开关组合
STAGE2_ENABLED_SWITCHES = ['ls', 'ma']    # 需要开启的开关
STAGE2_FIXED_PARAMS     = ['v']      # 固定的参数 (使用 Baseline 值)

# --- Stage 2 多轮配置 (坐标下降模式) ---
# 每轮指定 搜索项(search) 和 固定项(fix)
# fix 中的参数：第一轮使用 Baseline 值；后续轮使用上一轮最优值
STAGE2_ROUNDS = [
    {'search': ['S', 'r'], 'fix': ['v']},      # Round 1: 固定v, 搜索 S×r
    {'search': ['v'],      'fix': ['S', 'r']},  # Round 2: 固定S*,r*, 搜索 v
]

# ==============================================================================
# === 执行区 ===
# ==============================================================================

if __name__ == '__main__':
    
    if RUN_MODE == 'stage1':
        # ── Step 1: 生成参数组合 ──
        print("=" * 60)
        print("Stage 1: 开关项搜索")
        print("=" * 60)
        params = generate_stage_1(mode=STAGE1_MODE, exclude=STAGE1_EXCLUDE)
        print(f"生成 {len(params)} 组参数：")
        print_params(params)
        
        # ── Step 2: 批量回测 ──
        print("\n" + "=" * 60)
        print("开始批量回测...")
        print("=" * 60)
        results = batch_run(
            STRATEGY_FILE, params,
            save_path=ID_SAVE_PATH,
            start_day=START_DAY, end_day=END_DAY,
            initial_cash=INITIAL_CASH
        )
        
        # ── Step 3: 分析排序 ──
        df = print_compare(results, sort_by=SORT_BY)
    
    elif RUN_MODE == 'stage2':
        # ── Step 1: 生成参数组合 ──
        print("=" * 60)
        print("Stage 2: 范围项搜索")
        print("=" * 60)
        params = generate_stage_2(STAGE2_ENABLED_SWITCHES, STAGE2_FIXED_PARAMS)
        print(f"生成 {len(params)} 组参数：")
        print_params(params)
        
        # ── Step 2: 批量回测 ──
        print("\n" + "=" * 60)
        print("开始批量回测...")
        print("=" * 60)
        results = batch_run(
            STRATEGY_FILE, params,
            save_path=ID_SAVE_PATH,
            start_day=START_DAY, end_day=END_DAY,
            initial_cash=INITIAL_CASH
        )
        
        # ── Step 3: 分析排序 ──
        df = print_compare(results, sort_by=SORT_BY)
    
    elif RUN_MODE == 'stage2_multi':
        # ── 多轮坐标下降优化 ──
        best_values = {}  # 累积各轮最优参数值
        all_results = []  # 所有轮次的回测结果
        
        for round_idx, round_cfg in enumerate(STAGE2_ROUNDS, 1):
            search_items = round_cfg['search']
            fix_items = round_cfg['fix']
            
            print("\n" + "#" * 60)
            print(f"Round {round_idx}: 搜索 {search_items}, 固定 {fix_items}")
            print("#" * 60)
            
            # 构建 fixed_params: 已有最优值用字典，没有的用 Baseline
            fixed_params = []
            for item in fix_items:
                if item in best_values:
                    fixed_params.append({item: best_values[item]})
                    print(f"  {item} = {best_values[item]}  (来自上轮最优)")
                else:
                    fixed_params.append(item)  # 字符串 → Baseline 值
                    print(f"  {item} = Baseline")
            
            # 生成参数并回测
            params = generate_stage_2(STAGE2_ENABLED_SWITCHES, fixed_params)
            print(f"生成 {len(params)} 组参数")
            print_params(params)
            
            results = batch_run(
                STRATEGY_FILE, params,
                save_path=ID_SAVE_PATH,
                start_day=START_DAY, end_day=END_DAY,
                initial_cash=INITIAL_CASH
            )
            
            df = print_compare(results, sort_by=SORT_BY)
            all_results.extend(results)
            
            # 提取本轮最优配置，传递给下一轮
            best_name, best_config = get_best_config(results, sort_by=SORT_BY)
            if best_config:
                for item in search_items:
                    if item in best_config:
                        best_values[item] = best_config[item]
                print(f"\n  ★ Round {round_idx} 最优: {best_name}")
                for item in search_items:
                    print(f"    {item} = {best_values.get(item)}")
            else:
                print(f"\n  ✗ Round {round_idx} 无有效结果，终止")
                break
        
        # 全轮次汇总
        if all_results:
            print("\n" + "#" * 60)
            print(f"=== 全轮次汇总 ({len(all_results)} 组) ===")
            print("#" * 60)
            print_compare(all_results, sort_by=SORT_BY)
    
    elif RUN_MODE == 'analyze_only':
        # ── 仅分析已有回测结果 ──
        import json
        with open(ID_SAVE_PATH, 'r') as f:
            saved = json.load(f)
        
        # 注：analyze_only 模式下参数列将为空，仅显示指标
        tasks = [(name, {}, bt_id) for name, bt_id in saved.items()]
        print(f"从 {ID_SAVE_PATH} 加载 {len(tasks)} 组回测")
        
        df = print_compare(tasks, sort_by=SORT_BY)
