import pickle

file_path = "/home/hh01/Documents/jqtrade/Pools/cluster_results/cluster_data_cache2.pkl"
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Find ETF codes by name keyword from any period's label_map
name_mapping = {}
for period in data:
    if 'label_map' in period:
        for code, name in period['label_map'].items():
            name_mapping[name] = code

def find_code(keyword):
    for name, code in name_mapping.items():
        if keyword in name:
            return code, name
    return None, None

# Categories to track based on ap_pools_2.md
categories = {
    "1. 核心资产 (Core Assets)": ["300", "A500"],
    "2. 中小盘 (Small-Mid Cap)": ["中证500", "1000", "2000"],
    "3. 科技风格 (Tech Style)": ["创业板", "光伏", "人工智能", "科创板", "半导体", "科创100"],
    "4. 红利策略 (Dividend Strategy)": ["红利", "红利低波", "香港红利", "银行"],
    "5. 跨境与商品 (Cross-border & Commodities)": ["纳指", "日经", "生物", "德国", "沙特", "黄金", "油气", "豆粕"]
}

# Resolve actual codes
target_etfs = {}
for category, keywords in categories.items():
    target_etfs[category] = {}
    for kw in keywords:
        code, name = find_code(kw)
        if code:
            target_etfs[category][name] = code
        else:
            print(f"Warning: Could not find ETF matching '{kw}'")

print("\n================= ETF CLUSTER TRACKING REPORT =================")

for category, etfs in target_etfs.items():
    print(f"\n>> {category}")
    for name, code in etfs.items():
        print(f"\nTracking {name} ({code}):")
        
        # Track its cluster membership over the 12 periods
        cluster_history = []
        for i, period in enumerate(data):
            date_str = period['date'].strftime("%Y-%m-%d")
            details = period['details']
            
            if code in details:
                label = details[code]
                cluster_size = len(period['clusters'][label])
                rep_code = period['representatives'].get(label)
                rep_name = period.get('label_map', {}).get(rep_code, rep_code)
                cluster_history.append(f"  Period {i} ({date_str}): Cluster {label} -> Rep: {rep_name} (Size: {cluster_size})")
            else:
                cluster_history.append(f"  Period {i} ({date_str}): Not selected in this period")
                
        # condense history for readability
        condensed = []
        prev_rep = None
        start_period = 0
        for idx, hist in enumerate(cluster_history):
            rep = hist.split("Rep: ")[1].split(" (")[0] if "Rep: " in hist else "None"
            if rep != prev_rep:
                if prev_rep is not None:
                    print(f"  Periods {start_period}-{idx-1}: -> Rep: {prev_rep}")
                prev_rep = rep
                start_period = idx
        if prev_rep is not None:
            print(f"  Periods {start_period}-{len(cluster_history)-1}: -> Rep: {prev_rep}")
