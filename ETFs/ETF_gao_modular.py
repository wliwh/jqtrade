# 策略名称：高收益ETF轮动策略（模块化重构版）
# 原始逻辑参考：ETF_gao.py
# 重构作者：Antigravity
# 说明：
# 1. 严格保留原策略逻辑：R2动量评分、均线过滤、成交量异常过滤、3日跌幅限制。
# 2. 严格保留原交易时点：10:29卖出，10:30买入。
# 3. 采用模块化架构：Config -> Data -> Logic -> Execution

import numpy as np
import math
import pandas as pd
from datetime import datetime, timedelta
from jqdata import *

class Config:
    # ==================== 交易环境设置 ====================
    AVOID_FUTURE_DATA = True
    USE_REAL_PRICE = True
    BENCHMARK = "513100.XSHG"
    
    # 滑点与费率
    SLIPPAGE_FUND = 0.001
    SLIPPAGE_STOCK = 0.003
    
    COMMISSION_STOCK_OPEN = 0.0002
    COMMISSION_STOCK_CLOSE = 0.0002
    COMMISSION_MIN = 0

    # 策略参数
    HOLD_COUNT = 1          # 持仓数量
    M_DAYS = 25             # 动量参考天数
    MIN_MONEY = 500         # 最小交易额
    
    # 评分筛选参数
    MIN_SCORE = 0.0
    MAX_SCORE = 5.0
    DROP_3DAY_LIMIT = 0.97  # 3日跌幅限制
    
    # 均线过滤
    ENABLE_MA_FILTER = False # 原代码默认False，若需启用改为True
    MA_FILTER_DAYS = 20
    
    # 成交量检测
    ENABLE_VOLUME_CHECK = True
    VOLUME_LOOKBACK = 5
    VOLUME_THRESHOLD = 1.0
    
    # 评分方式: 'wls' (原版), 'er' (效率系数), 'frama' (分形维数), 'quad' (二次拟合), 'trendflex'
    SCORING_METHOD = 'wls' 

    # ==================== 策略核心参数 ====================
    ETF_POOL = [
        # 境外
        "513100.XSHG",  # 纳指ETF
        "159509.XSHE",  # 纳指科技ETF
        "513520.XSHG",  # 日经ETF
        "513030.XSHG",  # 德国ETF
        # 商品
        "518880.XSHG",  # 黄金ETF
        "159980.XSHE",  # 有色ETF
        "159985.XSHE",  # 豆粕ETF
        "159981.XSHE",  # 能源化工ETF
        # "159870.XSHE", # 化工
        
        "501018.XSHG",  # 南方原油
        # 债券
        "511090.XSHG",  # 30年国债ETF
        # 国内
        "513130.XSHG",  # 恒生科技
        "513690.XSHG",  # 港股红利
        
        "510180.XSHG",   #上证180
        "159915.XSHE",   #创业板ETF
        
        "510410.XSHG",   #资源
        "515650.XSHG",   #消费50
        "512290.XSHG",   #生物医药
        "588120.XSHG",   #科创100
        "515070.XSHG",   #人工智能ETF
        
        "159851.XSHE",   #金融科技
        "159637.XSHE",   #新能源车
        "516160.XSHG",   #新能源
        
        "159550.XSHE",   #互联网ETF
        "512710.XSHG",   #军工ETF
        "159692.XSHE",   #证券
        "512480.XSHG",   #半导体
        "515250.XSHG",   #智能汽车
        "159378.XSHE",   #通用航空
        "516510.XSHG",   #云计算
        "515050.XSHG",   #5G通信
        "159995.XSHE",   #芯片 
        "515790.XSHG",   #光伏
        "515000.XSHG"    #科技
    ]


# ==================== 初始化 ====================
def initialize(context):
    set_benchmark(Config.BENCHMARK)
    set_option("avoid_future_data", Config.AVOID_FUTURE_DATA)
    set_option("use_real_price", Config.USE_REAL_PRICE)
    
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    
    set_slippage(FixedSlippage(Config.SLIPPAGE_FUND), type="fund")
    set_slippage(FixedSlippage(Config.SLIPPAGE_STOCK), type="stock")
    
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, # 0.001
        open_commission=Config.COMMISSION_STOCK_OPEN, 
        close_commission=Config.COMMISSION_STOCK_CLOSE, 
        close_today_commission=0, min_commission=Config.COMMISSION_MIN
    ), type="stock")
    
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, 
        open_commission=0, close_commission=0, 
        close_today_commission=0, min_commission=0
    ), type="mmf")
    
    # 全局变量
    g.etf_pool = Config.ETF_POOL
    g.target_list = [] # 存储每日计算出的目标列表
    
    # 定时任务 (与原逻辑一致)
    run_daily(etf_trade_sell, "10:29") # 卖出
    run_daily(etf_trade_buy, "10:30")  # 买入
    run_daily(end_trade, "14:59")      # 盘后处理

# ==================== 数据获取模块 ====================
def get_safe_price(security, context):
    """获取防未来当前价"""
    try:
        current_dt = context.current_dt
        # 取最近1分钟bar的close，如果是回测，context.current_dt是分钟开始时间还是结束时间取决于平台
        # 稳妥起见，取前一分钟
        data = get_price(security, end_date=current_dt, frequency='1m', fields=['close'], count=2, panel=False)
        if data is not None and not data.empty:
            return data['close'].iloc[-1]
        return 0
    except:
        return 0

def get_current_vol_sum(security, context):
    """获取当日截止当前的累计成交量"""
    try:
        today = context.current_dt.date()
        df = get_price(security, start_date=today, end_date=context.current_dt, frequency='1m', fields=['volume'], panel=False)
        if df is not None and not df.empty:
            return df['volume'].sum()
        return 0
    except:
        return 0

# ==================== 逻辑计算模块 ====================
def calculate_ma_filter(stocks, days=20):
    if not stocks: return []
    filtered = []
    
    # 批量获取数据优化
    hists = attribute_history(stocks, days, '1d', ['close'], skip_paused=True)
    
    # 获取当前快照
    current_data = get_current_data()
    
    for stock in stocks:
        if stock not in hists or len(hists[stock]) < days: continue
        ma_price = hists[stock]['close'].mean()
        cur_price = current_data[stock].last_price
        
        if cur_price >= ma_price:
            filtered.append(stock)
            
    return filtered

def check_volume_anomaly(etf, context, lookback=5, threshold=1.0):
    """检测成交量是否异常放量"""
    try:
        # 1. 历史平均成交量
        hist = attribute_history(etf, lookback, '1d', ['volume'], skip_paused=True)
        if len(hist) < lookback: return False, 0
        avg_vol = hist['volume'].mean()
        if avg_vol == 0: return False, 0
        
        # 2. 当日实时成交量
        cur_vol = get_current_vol_sum(etf, context)
        
        ratio = cur_vol / avg_vol
        if ratio > threshold:
            return True, ratio
        return False, ratio
    except:
        return False, 0

class ScoringMethods:
    """评分方法集合"""
    
    @staticmethod
    def wls_score(prices):
        """原版加权线性回归评分"""
        y = np.log(prices)
        x = np.arange(len(y))
        w = np.linspace(1, 2, len(y)) # 线性加权
        
        slope, intercept = np.polyfit(x, y, 1, w=w)
        ann_ret = math.exp(slope * 250) - 1
        
        y_pred = slope * x + intercept
        ss_res = np.sum(w * (y - y_pred) ** 2)
        ss_tot = np.sum(w * (y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        
        score = ann_ret * r2
        return score, {"r2": r2, "ret": ann_ret}

    @staticmethod
    def calc_fractal_dimension(prices, n1=10, n2=20, n3=20):
        """计算分形维数 D"""
        if len(prices) < n3: return 1.5 # 默认中性
        
        # 取最近 n3 天
        data = prices[-n3:]
        
        mid = n3 // 2
        p1 = data[:mid]
        p2 = data[mid:]
        
        # 极差
        r1 = np.max(p1) - np.min(p1)
        r2 = np.max(p2) - np.min(p2)
        r3 = np.max(data) - np.min(data)
        
        if r3 == 0: return 1.0 # 直线
        if r1 + r2 == 0: return 1.0
        
        d = (np.log(r1 + r2) - np.log(r3)) / np.log(2)
        
        # D 理论范围 [1, 2], 但计算值可能溢出，限制一下
        if d < 1.0: d = 1.0
        if d > 2.0: d = 2.0
        
        return d

    @staticmethod
    def frama_modified_score(prices):
        """原版改良: WLS Slope * (2 - D)"""
        # 1. 计算 WLS Slope
        y = np.log(prices)
        x = np.arange(len(y))
        w = np.linspace(1, 2, len(y))
        slope, intercept = np.polyfit(x, y, 1, w=w)
        ann_ret = math.exp(slope * 250) - 1
        
        # 2. 计算分形维数 D
        # 使用全长度作为窗口
        d = ScoringMethods.calc_fractal_dimension(prices, n3=len(prices))
        
        # 3. 计算分数
        # D越大(越乱)，(2-D)越小，分数越低 -> 符合R2逻辑
        score = ann_ret * (2 - d)
        
        return score, {"d": d, "ret": ann_ret}

    @staticmethod
    def calc_super_smoother(prices, length):
        """Ehlers SuperSmoother Filter"""
        # prices: list or np.array
        if len(prices) < 3: return prices
        
        a1 = math.exp(-1.414 * np.pi / length)
        b1 = 2 * a1 * math.cos(1.414 * np.pi / length)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        
        filt = np.zeros_like(prices)
        # copy first 2 values
        filt[0] = prices[0] 
        filt[1] = prices[1]
        
        for i in range(2, len(prices)):
            filt[i] = c1 * (prices[i] + prices[i-1]) / 2 + c2 * filt[i-1] + c3 * filt[i-2]
            
        return filt

    @staticmethod
    def trendflex_score(prices):
        """Trendflex 评分"""
        # 1. 计算 SuperSmoother
        length = 20 # 默认周期
        if len(prices) < length + 5: return 0.0, {}
        
        filt = ScoringMethods.calc_super_smoother(prices, length)
        
        # 2. 计算 Trendflex
        # Trendflex = Sum(Filt[i] - Filt[i-k]) for k in 1..Length
        # 然后进行归一化 (MS)
        
        # 我们只计算最后一个点的 Trendflex 值
        i = len(prices) - 1
        sum_trendflex = 0.0
        
        for count in range(1, length + 1):
            sum_trendflex += filt[i] - filt[i-count]
            
        sum_trendflex /= length
        
        # 虽然 Ehlers 原版会进行 MS (Mean Square) 归一化，
        # 但我们这里直接用 sum_trendflex 作为动量分值即可，
        # 因为我们是在不同 ETF 间横向对比，且 Sum 已经除以 Length 类似于平均差
        
        # 为了与 score 维度 (0~5) 匹配:
        # Trendflex 本质是价格差的平均值。如果价格是 1.0, 涨了 10% -> 1.1
        # filt ~ 1.1, filt_old ~ 1.0
        # sum_diff ~ 0.1
        # 我们通常的 score 是 ann_ret (e.g. 0.5) * r2 (e.g. 0.9) = 0.45
        # 所以 Trendflex 值可能偏小（如果是绝对价格差）。
        # !重要!: Trendflex 原版是应用于 Price，这里 Input Prices 是绝对价格。
        # 不同 ETF 价格不同 (e.g. 1.0 vs 5.0)，直接减会导致高价 ETF 分数高。
        # 必须使用 Log Price 或者 归一化 Price (Change)。
        
        # 修正: 使用 Log Prices 计算 SuperSmoother
        log_prices = np.log(prices)
        filt_log = ScoringMethods.calc_super_smoother(log_prices, length)
        
        sum_tf_log = 0.0
        for count in range(1, length + 1):
            sum_tf_log += filt_log[i] - filt_log[i-count]
            
        sum_tf_log /= length
        
        # sum_tf_log 代表了 "平均对数收益偏离"，类似于平均涨幅
        # 乘以一个系数放大到 0~3 左右，或者直接用
        # 尝试还原为年化收益率量级? 
        # 简单起见，乘以 250 (年化) ? 
        # sum_tf_log 是 20天内的平均由于趋势带来的LogPrice增量
        # 乘 20 还原为总增量? -> 20 * sum_tf_log
        # 再年化? roughly: sum_tf_log * 250 is huge.
        
        # 让我们直接用 sum_tf_log * 100 作为分数，或者 * 50
        # 经验值: 一个强趋势，20天涨10%，log_diff sum avg approx 0.05?
        # 0.05 * 50 = 2.5 (符合 Score 范围)
        
        score = sum_tf_log * 50
        
        return score, {"trendflex": sum_tf_log}

    @staticmethod
    def quad_score(prices):
        """二次拟合评分: 瞬时速度 * R2"""
        # 1. 拟合抛物线 y = ax^2 + bx + c
        y = np.log(prices)
        x = np.arange(len(y))
        w = np.linspace(1, 2, len(y)) # 线性加权
        
        # polyfit 返回系数从高次到低次: a, b, c
        coeffs = np.polyfit(x, y, 2, w=w)
        a, b, c = coeffs
        
        # 2. 计算末端瞬时速度 (Velocity)
        # y' = 2ax + b
        end_x = x[-1]
        velocity = 2 * a * end_x + b
        
        # 转换为年化收益率 (近似)
        # velocity 是 log return 的每周期变化率
        ann_ret = math.exp(velocity * 250) - 1
        
        # 3. 计算稳定性 (R2)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum(w * (y - y_pred) ** 2)
        ss_tot = np.sum(w * (y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        
        # 4. 计算得分
        score = ann_ret * r2
        
        return score, {"velocity": ann_ret, "r2": r2, "accel": a}

    @staticmethod
    def er_score(prices):
        """Efficiency Ratio (ER) 评分"""
        # ER = |Net Change| / Sum(|Change|)
        # 价格通常为序列
        if len(prices) < 2: return 0.0, {}

        y = np.log(prices)
        x = np.arange(len(y))
        w = np.linspace(1, 2, len(y)) # 线性加权
        
        slope, intercept = np.polyfit(x, y, 1, w=w)
        ann_ret = math.exp(slope * 250) - 1
        
        net_change = abs(prices[-1] - prices[0])
        
        abs_changes = np.abs(np.diff(prices))
        sum_abs_change = np.sum(abs_changes)
        
        if sum_abs_change == 0:
            er = 0.0
        else:
            er = net_change / sum_abs_change
        
        score = er * ann_ret
        return score, {"er": er}

def get_etf_score(etf, context):
    try:
        current_data = get_current_data()
        cur_price = current_data[etf].last_price
        if math.isnan(cur_price): return None
        
        # 获取历史数据
        hist = attribute_history(etf, Config.M_DAYS, '1d', ['close'])
        if len(hist) < Config.M_DAYS: return None
        
        prices = np.append(hist['close'].values, cur_price)
        
        # 1. 3日跌幅限制
        if len(prices) >= 4:
            # 只要有一天跌幅超过5% (ratio < 0.95)，就排除
            # 原逻辑：min(p[-1]/p[-2], ...) < 0.95
            drops = [prices[-1]/prices[-2], prices[-2]/prices[-3], prices[-3]/prices[-4]]
            if min(drops) < Config.DROP_3DAY_LIMIT:
                # log.info(f"排除 {etf}: 近3日有大跌")
                return None

        # 2. 计算分数
        score = 0.0
        details = {}
        
        if Config.SCORING_METHOD == 'wls':
            score, details = ScoringMethods.wls_score(prices)
        elif Config.SCORING_METHOD == 'er':
            score, details = ScoringMethods.er_score(prices)
            # ER 模式下，分数通常在 -1.0 到 1.0 之间
            # 原有的 MIN_SCORE=0, MAX_SCORE=5 依然适用 (0-1在范围内)
        elif Config.SCORING_METHOD == 'frama':
            score, details = ScoringMethods.frama_modified_score(prices)
        elif Config.SCORING_METHOD == 'quad':
            score, details = ScoringMethods.quad_score(prices)
        elif Config.SCORING_METHOD == 'trendflex':
            score, details = ScoringMethods.trendflex_score(prices)
        else:
            log.warn(f"Unknown SCORING_METHOD: {Config.SCORING_METHOD}")
            return None
            
        # 3. 分数过滤
        if Config.MIN_SCORE < score < Config.MAX_SCORE:
            result = {'etf': etf, 'score': score}
            result.update(details)
            return result
        
        return None
        
    except Exception as e:
        log.warn(f"Error scoring {etf}: {e}")
        return None

def get_target_list(context):
    """筛选目标ETF列表"""
    pool = g.etf_pool
    
    # 1. 均线过滤
    if Config.ENABLE_MA_FILTER:
        pool = calculate_ma_filter(pool, Config.MA_FILTER_DAYS)
        log.info(f"均线过滤后剩余: {len(pool)}")
        
    # 2. 评分排序
    scored_list = []
    for etf in pool:
        res = get_etf_score(etf, context)
        if res: scored_list.append(res)
        
    # 按分数降序
    scored_list.sort(key=lambda x: x['score'], reverse=True)
    
    # 记录日志
    if scored_list:
        top3_info = [f"{x['etf']}:{x['score']:.3f}" for x in scored_list[:3]]
        log.info(f"Top 3: {top3_info}")
    
    # 返回纯代码列表
    return [x['etf'] for x in scored_list]


# ==================== 交易执行模块 ====================
def smart_order_target(security, value, context):
    """智能下单封装"""
    current_data = get_current_data()
    if current_data[security].paused: return False
    
    # 涨跌停检查
    price = current_data[security].last_price
    if price >= current_data[security].high_limit or price <= current_data[security].low_limit:
        return False
        
    # 计算目标股数 (向下取整到100)
    if price == 0: return False
    target_amount = (int(value / price) // 100) * 100
    
    # 最小交易额检查
    pos = context.portfolio.positions[security]
    current_val = pos.total_amount * price
    diff_val = abs(value - current_val)
    
    if diff_val < Config.MIN_MONEY and value > 0: # 只有开仓/调仓时检查，清仓不受限
        return False
        
    order_target_value(security, value)
    return True

# ---------- 卖出逻辑 (10:29) ----------
def etf_trade_sell(context):
    log.info("=== 开始执行卖出逻辑 (10:29) ===")
    
    # 1. 计算今日目标
    # 这里我们只计算 filter 结果，真正根据成交量剔除在买入时做，或者在这里如果持仓放量也卖
    raw_targets = get_target_list(context)
    g.target_list = raw_targets # 存给买入逻辑用
    
    # 截取前 N 名作为“理想持仓”
    ideal_targets = raw_targets[:Config.HOLD_COUNT]
    
    current_positions = list(context.portfolio.positions.keys())
    
    # 2. 检查持仓是否需要卖出
    for etf in current_positions:
        pos = context.portfolio.positions[etf]
        if pos.total_amount == 0: continue
        
        should_sell = False
        reason = ""
        
        # A. 成交量异常放量 -> 强制卖出
        if Config.ENABLE_VOLUME_CHECK:
            is_anomaly, ratio = check_volume_anomaly(etf, context, Config.VOLUME_LOOKBACK, Config.VOLUME_THRESHOLD)
            if is_anomaly:
                should_sell = True
                reason = f"放量异常(Ratio:{ratio:.2f})"
        
        # B. 不在目标池中
        if etf not in ideal_targets and not should_sell:
            should_sell = True
            reason = "不在目标列表"
            
        # C. 持仓超标且排名靠后 (针对持有多个标的的情况，本策略默认HOLD_COUNT=1)
        if etf in ideal_targets:
            # 如果在目标里，但排名在 HOLD_COUNT 之外 (例如持仓多了)
            if etf not in ideal_targets[:Config.HOLD_COUNT]:
                should_sell = True
                reason = "排名下降"
        
        if should_sell:
            log.info(f"卖出 {etf}: {reason}")
            smart_order_target(etf, 0, context)

# ---------- 买入逻辑 (10:30) ----------
def etf_trade_buy(context):
    log.info("=== 开始执行买入逻辑 (10:30) ===")
    
    # 1. 获取候选名单 (复用卖出时计算的，或者重新计算)
    # 建议重新读取 g.target_list，若为空则重新算
    targets = g.target_list
    if not targets:
        log.info("无目标标的")
        return

    # 2. 筛选最终买入目标 (剔除放量异常的)
    final_buy_targets = []
    
    # 只看前几名候选，甚至可以多看几个替补
    candidates = targets 
    
    for etf in candidates:
        if len(final_buy_targets) >= Config.HOLD_COUNT: break
        
        # 成交量检查
        if Config.ENABLE_VOLUME_CHECK:
            is_anomaly, ratio = check_volume_anomaly(etf, context, Config.VOLUME_LOOKBACK, Config.VOLUME_THRESHOLD)
            if is_anomaly:
                log.info(f"剔除买入目标 {etf}: 放量异常(Ratio:{ratio:.2f})")
                continue
        
        final_buy_targets.append(etf)
        
    if not final_buy_targets:
        log.info("无有效买入目标 (全被过滤)")
        return
        
    log.info(f"最终买入目标: {final_buy_targets}")
    
    # 3. 执行买入
    total_value = context.portfolio.total_value
    per_value = total_value / len(final_buy_targets)
    
    available_cash = context.portfolio.available_cash
    
    for etf in final_buy_targets:
        smart_order_target(etf, per_value, context)

# ---------- 盘后处理 ----------
def end_trade(context):
    pass
