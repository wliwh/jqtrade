# 策略名称：高收益ETF轮动策略（模块化融合 7star 严控版）
# 核心算法：WLS加权线性回归
# 交易机制：7star 多维过滤、极速冷热状态机、O(1)级高效分钟止损

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

    # 交易时间设定
    SELL_TIME = "13:10"
    BUY_TIME = "13:11"

    # 策略基本参数
    HOLD_COUNT = 1          # 持仓数量
    M_DAYS = 25             # 动量参考天数
    MIN_MONEY = 500         # 最小交易额
    
    # === 多维评分筛选参数 ===
    MIN_SCORE = 0.0
    MAX_SCORE = 6.0
    
    ENABLE_R2_FILTER = True # 是否启用R²过滤
    R2_THRESHOLD = 0.4      # R²必须大于此值
    
    ENABLE_ANNUAL_RET_FILTER = False # 是否启用年化收益率过滤
    MIN_ANNUAL_RET = 1.0    # 1.0表示100%
    
    ENABLE_SHORT_MOMENTUM = True # 短期动量过滤
    SHORT_MOMENTUM_DAYS = 10
    SHORT_MOMENTUM_THRESHOLD = 0.0
    
    ENABLE_RSI_FILTER = False # RSI超买过滤
    RSI_PERIOD = 6
    RSI_LOOKBACK = 1
    RSI_THRESHOLD = 98

    # === 短期风控过滤 ===
    DROP_3DAY_LIMIT = 0.97  # 3日单日跌幅限制（1-0.97=3%）
    
    # 均线过滤
    ENABLE_MA_FILTER = False
    MA_FILTER_DAYS = 20
    
    # 成交量异常检测
    ENABLE_VOLUME_CHECK = True
    VOLUME_LOOKBACK = 5
    VOLUME_THRESHOLD = 0.6
    
    # === 分钟级极速风控（止损） ===
    USE_FIXED_STOP_LOSS = True     # 启用固定比例止损
    FIXED_STOP_LOSS_THRESHOLD = 0.95 # 成本价 * 95%
    
    USE_PCT_STOP_LOSS = False      # 启用当日跌幅止损
    PCT_STOP_LOSS_THRESHOLD = 0.95 # 开盘价 * 95%
    
    USE_ATR_STOP_LOSS = False      # 启用ATR跟踪止损
    ATR_PERIOD = 14
    ATR_MULTIPLIER = 2
    ATR_TRAILING_STOP = True       # True为跟踪止损，False为固定成本止损
    ATR_EXCLUDE_DEFENSIVE = True   # 防御标的不参与ATR止损
    
    # === 防御与冷却期机制 ===
    DEFENSIVE_ETF = "511880.XSHG"  # 银华日利（当无符合条件票时买入收息）
    SAFE_HAVEN_ETF = "511660.XSHG" # 建信添益（冷却期间避险）
    
    SELL_COOLDOWN_ENABLED = True   # 是否启用冷却期（触发止损后必须冷静）
    SELL_COOLDOWN_DAYS = 3
    
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
    
    # 新增7star特征状态变量
    g.cooldown_end_date = None      # 止损后的冷却期结束日
    g.position_highs = {}           # ATR跟踪最高价记录
    g.position_stop_prices = {}     # ATR设定的止损价
    
    # 定时任务
    run_daily(check_positions, "09:10") # 盘前检查
    run_daily(etf_trade_sell, Config.SELL_TIME) # 卖出
    run_daily(etf_trade_buy, Config.BUY_TIME)  # 买入
    run_daily(end_trade, "14:59")      # 盘后处理
    
    # 注册分钟级极速风控
    for hour in range(9, 15):
        for minute in range(0, 60):
            current_time = "%02d:%02d" % (hour, minute)
            if ('09:27' < current_time < '11:30') or ('13:00' < current_time < '14:57'):
                run_daily(minute_level_stop_loss, current_time)
                run_daily(minute_level_pct_stop_loss, current_time)
                run_daily(minute_level_atr_stop_loss, current_time)

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

# ==================== 技术指标计算 ====================
def calculate_rsi(prices, period=6):
    """计算RSI序列"""
    if len(prices) < period + 1:
        return np.array([])
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    alpha = 2.0 / (period + 1)
    avg_gains = np.zeros(len(deltas))
    avg_losses = np.zeros(len(deltas))
    avg_gains[period - 1] = np.mean(gains[:period])
    avg_losses[period - 1] = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gains[i] = (gains[i] * alpha) + (avg_gains[i - 1] * (1 - alpha))
        avg_losses[i] = (losses[i] * alpha) + (avg_losses[i - 1] * (1 - alpha))
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    full_rsi = np.full(len(prices), np.nan)
    full_rsi[1:] = rsi
    return full_rsi[period:]

def calculate_atr(security, period=14):
    """计算ATR指标"""
    try:
        needed_days = period + 20
        hist_data = attribute_history(security, needed_days, '1d', ['high', 'low', 'close'])
        if len(hist_data) < period + 1:
            return 0, [], False
        high_prices = hist_data['high'].values
        low_prices = hist_data['low'].values
        close_prices = hist_data['close'].values
        tr_values = np.zeros(len(high_prices))
        for i in range(1, len(high_prices)):
            tr1 = high_prices[i] - low_prices[i]
            tr2 = abs(high_prices[i] - close_prices[i-1])
            tr3 = abs(low_prices[i] - close_prices[i-1])
            tr_values[i] = max(tr1, tr2, tr3)
        atr_values = np.zeros(len(tr_values))
        for i in range(period, len(tr_values)):
            atr_values[i] = np.mean(tr_values[i-period+1:i+1])
        current_atr = atr_values[-1] if len(atr_values) > 0 else 0
        valid_atr = atr_values[period:] if len(atr_values) > period else atr_values
        return current_atr, valid_atr, True
    except:
        return 0, [], False

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

def get_etf_score(etf, context):
    try:
        current_data = get_current_data()
        cur_price = current_data[etf].last_price
        if math.isnan(cur_price) or cur_price == 0: return None
        
        # 停牌跳过
        if current_data[etf].paused: return None
        
        # 取满足各种指标最大所需天数
        max_lookback = max(
            Config.M_DAYS,
            Config.SHORT_MOMENTUM_DAYS if Config.ENABLE_SHORT_MOMENTUM else 0,
            (Config.RSI_PERIOD + Config.RSI_LOOKBACK) if Config.ENABLE_RSI_FILTER else 0
        )
        
        # 获取历史数据
        hist = attribute_history(etf, max_lookback, '1d', ['close'])
        if len(hist) < Config.M_DAYS: return None # 至少满足核心WLS长度
        
        prices = np.append(hist['close'].values, cur_price)
        
        # 1. 3日单日跌幅限制 (短期风控)
        if len(prices) >= 4:
            drops = [prices[-1]/prices[-2], prices[-2]/prices[-3], prices[-3]/prices[-4]]
            if min(drops) < Config.DROP_3DAY_LIMIT:
                return None
                
        # 2. RSI 超买过滤
        if Config.ENABLE_RSI_FILTER and len(prices) >= Config.RSI_PERIOD + Config.RSI_LOOKBACK:
            rsi_values = calculate_rsi(prices, Config.RSI_PERIOD)
            if len(rsi_values) >= Config.RSI_LOOKBACK:
                recent_rsi = rsi_values[-Config.RSI_LOOKBACK:]
                if np.any(recent_rsi > Config.RSI_THRESHOLD):
                    ma5 = np.mean(prices[-5:]) if len(prices) >= 5 else cur_price
                    if cur_price < ma5:
                        # RSI超买并且破5日线直接过滤
                        return None
                        
        # 3. 短期动量过滤
        if Config.ENABLE_SHORT_MOMENTUM and len(prices) >= Config.SHORT_MOMENTUM_DAYS + 1:
            short_return = prices[-1] / prices[-(Config.SHORT_MOMENTUM_DAYS + 1)] - 1
            short_annualized = (1 + short_return) ** (250 / Config.SHORT_MOMENTUM_DAYS) - 1
            if short_annualized < Config.SHORT_MOMENTUM_THRESHOLD:
                return None

        # 4. 计算分数
        # 截取WLS所需长度
        wls_prices = prices[-(Config.M_DAYS + 1):]
        score, details = wls_score(wls_prices)
        
        # 5. R²和年化独立过滤
        if Config.ENABLE_R2_FILTER and details['r2'] < Config.R2_THRESHOLD:
            return None
            
        if Config.ENABLE_ANNUAL_RET_FILTER and details['ret'] < Config.MIN_ANNUAL_RET:
            return None
            
        # 6. 分数过滤
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

def check_positions(context):
    """盘前获取当前持仓与状态"""
    for security in context.portfolio.positions:
        pos = context.portfolio.positions[security]
        if pos.total_amount > 0:
            name = get_current_data()[security].name if security in get_current_data() else security
            log.info(f"持仓检查: {security} {name}, 数量: {pos.total_amount}, 成本: {pos.avg_cost:.3f}")

# ==================== 分钟级风控（止损） ====================
def is_in_cooldown(context):
    """判断是否在冷却期内"""
    if not Config.SELL_COOLDOWN_ENABLED or g.cooldown_end_date is None:
        return False
    return context.current_dt.date() <= g.cooldown_end_date

def set_cooldown(context):
    """设置冷却期结束日期"""
    if Config.SELL_COOLDOWN_ENABLED:
        g.cooldown_end_date = context.current_dt.date() + timedelta(days=Config.SELL_COOLDOWN_DAYS)
        log.info(f"🔒 触发冷却期，结束日期: {g.cooldown_end_date.strftime('%Y-%m-%d')}")

def enter_safe_haven_and_set_cooldown(context, trigger_reason=""):
    """进入冷却期并买入避险ETF"""
    if not Config.SELL_COOLDOWN_ENABLED: return
    
    # 卖出非避险品种
    for security in list(context.portfolio.positions.keys()):
        if security != Config.SAFE_HAVEN_ETF:
            pos = context.portfolio.positions[security]
            if pos.total_amount > 0:
                smart_order_target(security, 0, context)
                g.position_highs.pop(security, None)
                g.position_stop_prices.pop(security, None)
                log.info(f"✅ [冷却期] 卖出全部风险敞口: {security}")
                
    total_value = context.portfolio.total_value
    if total_value > Config.MIN_MONEY:
        if smart_order_target(Config.SAFE_HAVEN_ETF, total_value * 0.99, context):
            log.info(f"🛡️ [冷却期] 买入避险ETF {Config.SAFE_HAVEN_ETF}")
            
    set_cooldown(context)
    log.info(f"🔒 [冷却期] 已进入冷却期，由 [{trigger_reason}] 触发。")

def exit_safe_haven_if_cooldown_ends(context):
    """冷却期结束卖出避险ETF"""
    if not Config.SELL_COOLDOWN_ENABLED or g.cooldown_end_date is None:
        return
    current_date = context.current_dt.date()
    if current_date > g.cooldown_end_date:
        log.info(f"🔓 冷却期结束 ({current_date.strftime('%Y-%m-%d')})")
        if Config.SAFE_HAVEN_ETF in context.portfolio.positions:
            pos = context.portfolio.positions[Config.SAFE_HAVEN_ETF]
            if pos.total_amount > 0:
                smart_order_target(Config.SAFE_HAVEN_ETF, 0, context)
                log.info(f"✅ [冷却期结束] 卖出避险ETF: {Config.SAFE_HAVEN_ETF}")
        g.cooldown_end_date = None

def minute_level_stop_loss(context):
    """分钟级固定比例止损"""
    if not Config.USE_FIXED_STOP_LOSS: return
    if is_in_cooldown(context): return
    
    current_data = get_current_data()
    for security in list(context.portfolio.positions.keys()):
        if security == Config.DEFENSIVE_ETF or security == Config.SAFE_HAVEN_ETF: continue
        position = context.portfolio.positions[security]
        if position.total_amount <= 0: continue
        
        current_price = current_data[security].last_price
        cost_price = position.avg_cost
        if current_price <= 0 or cost_price <= 0: continue
        
        if current_price <= cost_price * Config.FIXED_STOP_LOSS_THRESHOLD:
            loss_percent = (current_price / cost_price - 1) * 100
            log.info(f"🚨 [极速风控] 固定跌幅止损触发: {security}, 现价{current_price}, 成本{cost_price}, 亏损{loss_percent:.2f}%")
            if smart_order_target(security, 0, context):
                enter_safe_haven_and_set_cooldown(context, trigger_reason="分钟级固定止损")

def minute_level_pct_stop_loss(context):
    """分钟级当日跌幅止损"""
    if not Config.USE_PCT_STOP_LOSS: return
    if is_in_cooldown(context): return
    
    current_data = get_current_data()
    for security in list(context.portfolio.positions.keys()):
        if security == Config.DEFENSIVE_ETF or security == Config.SAFE_HAVEN_ETF: continue
        position = context.portfolio.positions[security]
        if position.total_amount <= 0: continue
        
        today_open = current_data[security].day_open
        if not today_open or today_open <= 0: continue
        
        current_price = current_data[security].last_price
        stop_price = today_open * Config.PCT_STOP_LOSS_THRESHOLD
        
        if current_price <= stop_price:
            daily_loss = (current_price / today_open - 1) * 100
            log.info(f"🚨 [极速风控] 日内暴跌止损触发: {security}, 现价{current_price}, 开盘{today_open}, 跌幅{daily_loss:.2f}%")
            if smart_order_target(security, 0, context):
                enter_safe_haven_and_set_cooldown(context, trigger_reason="分钟级当日暴跌止损")

def minute_level_atr_stop_loss(context):
    """分钟级ATR跟踪止损"""
    if not Config.USE_ATR_STOP_LOSS: return
    if is_in_cooldown(context): return
    
    current_data = get_current_data()
    for security in list(context.portfolio.positions.keys()):
        if Config.ATR_EXCLUDE_DEFENSIVE and (security == Config.DEFENSIVE_ETF or security == Config.SAFE_HAVEN_ETF): continue
        
        position = context.portfolio.positions[security]
        if position.total_amount <= 0: continue
        
        current_price = current_data[security].last_price
        cost_price = position.avg_cost
        if current_price <= 0 or cost_price <= 0: continue
        
        current_atr, _, success = calculate_atr(security, Config.ATR_PERIOD)
        if not success or current_atr <= 0: continue
        
        # 记录最高价
        if security not in g.position_highs:
            g.position_highs[security] = current_price
        else:
            g.position_highs[security] = max(g.position_highs[security], current_price)
            
        # 设置止损价
        if Config.ATR_TRAILING_STOP:
            atr_stop_price = g.position_highs[security] - Config.ATR_MULTIPLIER * current_atr
        else:
            atr_stop_price = cost_price - Config.ATR_MULTIPLIER * current_atr
            
        g.position_stop_prices[security] = atr_stop_price
        
        if current_price <= atr_stop_price:
            loss_percent = (current_price / cost_price - 1) * 100
            t_type = "跟踪" if Config.ATR_TRAILING_STOP else "固定"
            log.info(f"🚨 [极速风控] ATR{t_type}止损触发: {security}, 现价{current_price}, 止损价{atr_stop_price:.3f}, 亏损{loss_percent:.2f}%")
            if smart_order_target(security, 0, context):
                g.position_highs.pop(security, None)
                g.position_stop_prices.pop(security, None)
                enter_safe_haven_and_set_cooldown(context, trigger_reason="分钟级ATR动态止损")

# ---------- 卖出逻辑 ----------
def etf_trade_sell(context):
    log.info(f"=== 开始执行卖出逻辑 ({Config.SELL_TIME}) ===")
    
    if is_in_cooldown(context):
        log.info("🔒 处于冷却期，跳过执行常规卖出逻辑")
        return
    
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

    # 检查持股数，如果是固定ATR止损，并且当前不在建仓期的话
    # 在这个阶段，可能可以考虑把防守型基金也卖掉腾出资金，如果是为了进攻的话，由下一阶段决定。

# ---------- 买入逻辑 ----------
def etf_trade_buy(context):
    log.info(f"=== 开始执行买入逻辑 ({Config.BUY_TIME}) ===")
    
    # 退出避险机制检查
    exit_safe_haven_if_cooldown_ends(context)
    
    if is_in_cooldown(context):
        log.info("🔒 处于冷却期，跳过执行常规买入逻辑")
        return
    
    # 1. 获取候选名单
    targets = g.target_list
    final_buy_targets = []
    
    if not targets:
        log.info("今日无有效进攻目标")
        # --- 防御降维机制 ---
        current_data = get_current_data()
        if Config.DEFENSIVE_ETF in current_data and not current_data[Config.DEFENSIVE_ETF].paused:
            log.info(f"🛡️ 启动防御机制，买入防御ETF {Config.DEFENSIVE_ETF}")
            final_buy_targets = [Config.DEFENSIVE_ETF]
        else:
            log.info("防御ETF不可用，保持空仓")
            return
    else:
        # 2. 筛选最终买入进攻目标 (剔除放量异常的)
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
            log.info("有效买入目标全被过滤，将执行防御降维")
            current_data = get_current_data()
            if Config.DEFENSIVE_ETF in current_data and not current_data[Config.DEFENSIVE_ETF].paused:
                final_buy_targets = [Config.DEFENSIVE_ETF]
            else:
                return
                
    log.info(f"最终买入目标: {final_buy_targets}")
    
    # 3. 计算分配金额并执行买入
    current_positions = set(context.portfolio.positions.keys())
    etfs_to_buy = [etf for etf in final_buy_targets if etf not in current_positions]
    num_etfs_to_buy = len(etfs_to_buy)
    
    if num_etfs_to_buy == 0:
        log.info("当前持仓已全部为目标持仓，无需买入新ETF。")
        return

    available_cash = context.portfolio.available_cash
    allocated_value_per_etf = available_cash / num_etfs_to_buy
    
    if allocated_value_per_etf < Config.MIN_MONEY:
        log.info(f"分配金额 {allocated_value_per_etf:.2f} 小于最小交易额，跳过买入")
        return

    for i, etf in enumerate(etfs_to_buy):
        target_value = allocated_value_per_etf
        # 针对最后一只全部买入
        if i == len(etfs_to_buy) - 1 and context.portfolio.available_cash >= Config.MIN_MONEY:
            target_value = context.portfolio.available_cash
            
        smart_order_target(etf, target_value, context)
        
        # 记录ATR最高价等初始状态
        if Config.USE_ATR_STOP_LOSS and not (Config.ATR_EXCLUDE_DEFENSIVE and (etf == Config.DEFENSIVE_ETF)):
            current_data = get_current_data()
            if etf in current_data:
                g.position_highs[etf] = current_data[etf].last_price
                current_atr, _, success = calculate_atr(etf, Config.ATR_PERIOD)
                if success:
                    g.position_stop_prices[etf] = g.position_highs[etf] - Config.ATR_MULTIPLIER * current_atr
    


# ---------- 盘后处理 ----------
def end_trade(context):
    pass
