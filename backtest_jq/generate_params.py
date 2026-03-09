import itertools

Rules = [
    ['S', 'EXECUTION_SCORE_RANGE', [(0.0, float(x)) for x in (4, 5, 6, 7, 8, 10)],
     lambda x: 'S' + ('0' if x[1] < 10 else '') + '{:.1f}'.format(x[1]).replace('.','')],
    ['ls', 'EXECUTION_LOSE_PARAM', [(False, 0.95), (True, 0.95)],
     lambda x: 'ls' + '{:.2f}'.format(x[1])[2:] if x[0] else ''],
    ['ma', 'EXECUTION_MA_PARAM', [(False, 20), (True, 20)], lambda x: 'ma' if x[0] else ''],
    ['v', 'EXECUTION_VOLUME_PARAM', [(False, 5, 0.6)] + [(True, 5, x) for x in (0.4, 0.6, 0.8, 1.0)],
     lambda x: 'v{:.1f}'.format(x[2]).replace('.', '') if x[0] else ''],
    ['r', 'EXECUTION_R2_PARAM', [(False, 0.4)] + [(True, t) for t in (0.4, 0.5, 0.6, 0.7)],
     lambda x: 'r{:.1f}'.format(x[1]).replace('.', '') if x[0] else ''],
    ['st', 'EXECUTION_SHORT_MOMENTUM_PARAM', [(False, 10, 0.0), (True, 10, 0.0)], lambda x: 'st' if x[0] else ''],
    ['ar', 'EXECUTION_ANNUAL_RETURN_PARAM', [(False, 1.0), (True, 1.0)], lambda x: 'ar' if x[0] else ''],
    ['dl', 'EXECUTION_DAY_LIMIT_PARAM', [(True, 0.95), (False, 0.95)], lambda x: 'dl' if x[0] else '']
]

# 默认 Baseline 参数 (基于策略默认值)
Baseline = {
    'S': (0.0, 6.0),
    'ls': (False, 0.95),
    'ma': (False, 20),
    'v': (False, 5, 0.6),
    'r': (False, 0.4),
    'st': (False, 10, 0.0),
    'ar': (False, 1.0),
    'dl': (True, 0.95)
}

def get_name(short_name, value):
    rule = next(r for r in Rules if r[0] == short_name)
    return rule[3](value)

def make_id_name(config):
    """根据配置生成 ID 名称，按 Baseline 中的参数顺序排列"""
    # 自动识别带开关的参数（候选值的第一个元素是 bool 类型的参数）
    toggleable = {r[0] for r in Rules if isinstance(r[2][0][0], bool)}
    tokens = []
    for key in Baseline.keys():
        if key in config:
            val = config[key]
            # 带开关的参数，如果关闭则跳过显示
            if key in toggleable and val[0] == False:
                continue
            n = get_name(key, val)
            if n: tokens.append(n)
    return "_".join(tokens)

def generate_stage_1(mode='cartesian', exclude=None):
    """第一阶段：确定开关项 (ls, ma, st, ar, dl) 的最优组合。
    
    mode: 
        'ablation'  - 逐一反转（6组）
        'cartesian' - 笛卡尔积（2^N 组）
    S: 固定使用的评分范围
    exclude: 从笛卡尔积中排除的开关列表，单独测试（如 ['dl']）
    """
    if exclude is None:
        exclude = []
    
    params_list = []
    # all_switches = ['ls', 'ma', 'st', 'ar', 'dl']
    all_switches = [r[0] for r in Rules if len(r[2])==2]
    
    if mode == 'ablation':
        # --- 逐一反转模式 (6组) ---
        params_list.append((make_id_name(Baseline), Baseline.copy()))
        for s in all_switches:
            config = Baseline.copy()
            rule_values = next(r[2] for r in Rules if r[0] == s)
            other_value = next(v for v in rule_values if v[0] != Baseline[s][0])
            config[s] = other_value
            params_list.append((make_id_name(config), config))
    
    elif mode == 'cartesian':
        # --- 笛卡尔积模式 ---
        # 将开关分为"参与笛卡尔积"和"排除（固定）"两组
        cart_switches = [s for s in all_switches if s not in exclude]
        
        # 获取每个参与笛卡尔积的开关的所有候选值
        switch_values = []
        for s in cart_switches:
            rule_values = next(r[2] for r in Rules if r[0] == s)
            switch_values.append(rule_values)
        
        # 笛卡尔积
        for combo in itertools.product(*switch_values):
            config = Baseline.copy()
            for i, s in enumerate(cart_switches):
                config[s] = combo[i]
            params_list.append((make_id_name(config), config))
    
    return params_list

def generate_stage_2(enabled_switches=None, fixed_params=None):
    """第二阶段：核心寻优 (Core Optimization)
    针对范围项 S, v, r 进行组合搜索。
    enabled_switches: list, 需要开启的开关项短名列表，例如 ['ma', 'ls']
    fixed_params: list, 需要固定的参数，例如 ['v', {'S': (0.0, 5.0)}]
                  - 字符串: 固定为 Baseline 默认值
                  - 字典:   固定为给定的值
    """
    toggleable = {r[0] for r in Rules if len(r[2])==2}
    range_items = [r[0] for r in Rules if len(r[2])>2]
    
    # 从 Baseline 构建完整配置，然后开启指定的开关
    base_switches = {}
    for key in toggleable:
        rule_values = next(r[2] for r in Rules if r[0] == key)
        if key in enabled_switches:
            on_value = next(v for v in rule_values if v[0])
            base_switches[key] = on_value
        else:
            on_value = next(v for v in rule_values if not v[0])
            base_switches[key] = on_value # Baseline[key]
    
    # 解析 fixed_params 列表为字典
    fixed_dict = {}
    if fixed_params:
        for item in fixed_params:
            if isinstance(item, str):
                fixed_dict[item] = Baseline[item]
            elif isinstance(item, dict):
                fixed_dict.update(item)
    
    search_items = [item for item in range_items if item not in fixed_dict]
    
    combinations = []
    
    # 获取需要搜索的项的所有候选值
    item_values = []
    for item in search_items:
        rule_values = next(r[2] for r in Rules if r[0] == item)
        item_values.append(rule_values)
    
    # 笛卡尔积组合
    for combo in itertools.product(*item_values):
        config = base_switches.copy()
        config.update(fixed_dict) # 加入固定的参数
        
        names = []
        for i, item in enumerate(search_items):
            config[item] = combo[i]
            
        # 生成名称 (按照 Baseline 顺序排列)
        full_name = make_id_name(config)
        combinations.append((f"{full_name}", config))
        
    return combinations

def config_to_execution_params(config):
    """将短名参数字典转换为 EXECUTION_ 格式字典。
    
    输入: {'S': (0.0, 6.0), 'ls': (True, 0.95), ...}
    输出: {'EXECUTION_SCORE_RANGE': (0.0, 6.0), 'EXECUTION_LOSE_PARAM': (True, 0.95), ...}
    """
    mapped = {}
    for short_name, value in config.items():
        full_name = next(r[1] for r in Rules if r[0] == short_name)
        mapped[full_name] = value
    return mapped

def print_params(params):
    print(f"{'ID':<40} | {'Parameters'}")
    print("-" * 100)
    for name, p in params:
        mapped = config_to_execution_params(p)
        print(f"{name:<40} | {mapped}")

if __name__ == "__main__":
    print("=== 自动生成测试参数程序 ===")

    # --- 第一步：开关项搜索 ---
    
    # 模式 1: 笛卡尔积 (排除 dl，2^4 = 16组)
    # print("\n[Stage 1: 开关项笛卡尔积 (排除 dl)]")
    # stage1 = generate_stage_1(mode='cartesian', exclude=['dl'])
    # print(f"生成了 {len(stage1)} 组：")
    # print_params(stage1)
    
    # # 模式 2: 笛卡尔积 (全部5个开关，2^5 = 32组)
    # stage1_full = generate_stage_1(mode='cartesian')
    # print(f"生成了 {len(stage1_full)} 组。")
    
    # # 模式 3: 逐一反转 (6组)
    # stage1_abl = generate_stage_1(mode='ablation')
    # print(f"生成了 {len(stage1_abl)} 组。")

    # --- 第二步：范围项搜索 ---
    stage2 = generate_stage_2(['ls','ma'], ['S','r'])
    print_params(stage2)

