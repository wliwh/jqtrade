# 策略名称：高收益ETF轮动策略
# 核心算法：WLS加权线性回归
# 交易机制：多维过滤、简单日内止损

import numpy as np
import math
import pandas as pd
from datetime import datetime, timedelta
from jqdata import *

# # 境外
# "513100.XSHG",  # 纳指ETF
# "159509.XSHE",  # 纳指科技ETF
# "513520.XSHG",  # 日经ETF
# "513030.XSHG",  # 德国ETF
# # 商品
# "518880.XSHG",  # 黄金ETF
# "159980.XSHE",  # 有色ETF
# "159985.XSHE",  # 豆粕ETF
# "159981.XSHE",  # 能源化工ETF
# # "159870.XSHE", # 化工
# "501018.XSHG",  # 南方原油
# # 债券
# "511090.XSHG",  # 30年国债ETF
# # 国内
# "513130.XSHG",  # 恒生科技
# "513690.XSHG",  # 港股红利
# "510180.XSHG",   #上证180
# "159915.XSHE",   #创业板ETF
# # 赛道
# "510410.XSHG",   #资源
# "515650.XSHG",   #消费50
# "512290.XSHG",   #生物医药
# "588120.XSHG",   #科创100
# "515070.XSHG",   #人工智能ETF

# "159851.XSHE",   #金融科技
# "159637.XSHE",   #新能源车
# "516160.XSHG",   #新能源

# "159550.XSHE",   #互联网ETF
# "512710.XSHG",   #军工ETF
# "159692.XSHE",   #证券
# "512480.XSHG",   #半导体
# "515250.XSHG",   #智能汽车
# "159378.XSHE",   #通用航空
# "516510.XSHG",   #云计算
# "515050.XSHG",   #5G通信
# "159995.XSHE",   #芯片 
# "515790.XSHG",   #光伏
# "515000.XSHG"    #科技

EXECUTION_ETF_POOLS_PLACEHOLDER = ["513100.XSHG","159509.XSHE","513520.XSHG","513030.XSHG","518880.XSHG","159980.XSHE","159985.XSHE","159981.XSHE","501018.XSHG","511090.XSHG","513130.XSHG","513690.XSHG","510180.XSHG","159915.XSHE","510410.XSHG","515650.XSHG","512290.XSHG","588120.XSHG","515070.XSHG","159851.XSHE","159637.XSHE","516160.XSHG","159550.XSHE","512710.XSHG","159692.XSHE","512480.XSHG","515250.XSHG","159378.XSHE","516510.XSHG","515050.XSHG","159995.XSHE","515790.XSHG","515000.XSHG"]
EXECUTION_BEG_TIME_PLACEHOLDER = '10:31'
EXECUTION_END_TIME_PLACEHOLDER = '10:32'

EXECUTION_SCORE_RANGE = (0.0, 6.0)
EXECUTION_LOSE_PARAM = (False, 0.95)
EXECUTION_MA_PARAM = (False, 20)
EXECUTION_VOLUME_PARAM = (False, 5, 0.6)
EXECUTION_R2_PARAM = (False, 0.4)
EXECUTION_SHORT_MOMENTUM_PARAM = (False, 10, 0.0)
EXECUTION_ANNUAL_RETURN_PARAM = (False, 1.0)
EXECUTION_RSI_PARAM = (False, 6, 1, 98)
EXECUTION_DAY_LIMIT_PARAM = (True, 0.95)

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
    MIN_SCORE = EXECUTION_SCORE_RANGE[0]
    MAX_SCORE = EXECUTION_SCORE_RANGE[1]

    # 短期风控（3日跌幅限制）
    ENABLE_LOSS_FILTER = EXECUTION_LOSE_PARAM[0]
    DROP_3DAY_LIMIT = EXECUTION_LOSE_PARAM[1]  # 3日跌幅限制
    
    # 均线过滤
    ENABLE_MA_FILTER = EXECUTION_MA_PARAM[0]
    MA_FILTER_DAYS = EXECUTION_MA_PARAM[1]
    
    # 成交量检测
    ENABLE_VOLUME_CHECK = EXECUTION_VOLUME_PARAM[0]
    VOLUME_LOOKBACK = EXECUTION_VOLUME_PARAM[1]
    VOLUME_THRESHOLD = EXECUTION_VOLUME_PARAM[2]
    
    # R² 独立过滤
    ENABLE_R2_FILTER = EXECUTION_R2_PARAM[0]
    R2_THRESHOLD = EXECUTION_R2_PARAM[1]
    
    # 短期动量过滤
    ENABLE_SHORT_MOMENTUM = EXECUTION_SHORT_MOMENTUM_PARAM[0]
    SHORT_LOOKBACK_DAYS = EXECUTION_SHORT_MOMENTUM_PARAM[1]
    SHORT_MOMENTUM_THRESHOLD = EXECUTION_SHORT_MOMENTUM_PARAM[2]
    
    # 年化收益过滤
    ENABLE_ANNUAL_RETURN_FILTER = EXECUTION_ANNUAL_RETURN_PARAM[0]
    MIN_ANNUALIZED_RETURN = EXECUTION_ANNUAL_RETURN_PARAM[1]
    
    # RSI 过滤
    ENABLE_RSI_FILTER = EXECUTION_RSI_PARAM[0]
    RSI_PERIOD = EXECUTION_RSI_PARAM[1]
    RSI_LOOKBACK_DAYS = EXECUTION_RSI_PARAM[2]
    RSI_THRESHOLD = EXECUTION_RSI_PARAM[3]
    
    # 简易日内止损
    ENABLE_INTRADAY_STOP_LOSS = EXECUTION_DAY_LIMIT_PARAM[0]
    STOP_LOSS_PCT = EXECUTION_DAY_LIMIT_PARAM[1]
    
    # 防御ETF
    DEFENSIVE_ETF = "511880.XSHG"   # 银华日利（货币ETF）

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
        open_commission=Config.COMMISSION_STOCK_OPEN, 
        close_commission=Config.COMMISSION_STOCK_CLOSE, 
        close_today_commission=0, min_commission=Config.COMMISSION_MIN
    ), type="fund")
    
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, 
        open_commission=0, close_commission=0, 
        close_today_commission=0, min_commission=0
    ), type="mmf")
    
    # 全局变量
    g.etf_pool = EXECUTION_ETF_POOLS_PLACEHOLDER
    g.target_list = [] # 存储每日计算出的目标列表
    
    # 定时任务 (与原逻辑一致)
    run_daily(etf_trade_sell, EXECUTION_BEG_TIME_PLACEHOLDER) # 卖出
    run_daily(etf_trade_buy, EXECUTION_END_TIME_PLACEHOLDER)  # 买入
    
    # 简单的定时止损监控（避免每分钟高频耗时）
    run_daily(simple_stop_loss, "09:31")
    run_daily(simple_stop_loss, "10:00")
    run_daily(simple_stop_loss, "11:00")
    run_daily(simple_stop_loss, "13:30")
    run_daily(simple_stop_loss, "14:30")
    
    # 状态输出
    run_daily(print_positions, "09:30")
    run_daily(print_positions, "15:00")

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

def check_defensive_etf_available(context):
    """检查防御ETF是否可交易"""
    current_data = get_current_data()
    defensive_etf = Config.DEFENSIVE_ETF
    try:
        info = current_data[defensive_etf]
        if info.paused:
            log.info(f"防御性ETF {defensive_etf} 今日停牌")
            return False
            
        if not math.isnan(info.high_limit) and info.high_limit > 0:
            if info.last_price >= info.high_limit:
                return False
        if not math.isnan(info.low_limit) and info.low_limit > 0:
            if info.last_price <= info.low_limit:
                return False
                
        return True
    except Exception as e:
        return False

def calculate_rsi(prices, period=6):
    """计算 RSI 指标"""
    if len(prices) < period + 1: return np.array([])
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
        
    rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses!=0)
    rsi = 100 - (100 / (1 + rs))
    
    full_rsi = np.full(len(prices), np.nan)
    full_rsi[1:] = rsi
    return full_rsi[period:]

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
        if math.isnan(cur_price) or cur_price <= 0: return None
        
        # 确定需要的最大历史天数
        lookback = max(
            Config.M_DAYS,
            Config.SHORT_LOOKBACK_DAYS if Config.ENABLE_SHORT_MOMENTUM else 0,
            Config.RSI_PERIOD + Config.RSI_LOOKBACK_DAYS if Config.ENABLE_RSI_FILTER else 0,
            Config.MA_FILTER_DAYS if Config.ENABLE_MA_FILTER else 0
        ) + 5
        
        # 获取历史数据 (回退至 ETF_gao_modular 的默认取法，规避额外参数造成的隐式拦截)
        hist = attribute_history(etf, lookback, '1d', ['close'])
        if len(hist) < Config.M_DAYS: return None
        
        prices = np.append(hist['close'].values, cur_price)
        
        # 截取动量计算所需的天数
        recent_prices = prices[-(Config.M_DAYS + 1):]
        
        # 1. 计算核心分数
        score, details = wls_score(recent_prices)
        passed_momentum = Config.MIN_SCORE < score < Config.MAX_SCORE
            
        # 2. 短期动量
        short_annualized = -np.inf
        if len(prices) >= Config.SHORT_LOOKBACK_DAYS + 1:
            short_return = prices[-1] / prices[-(Config.SHORT_LOOKBACK_DAYS + 1)] - 1
            short_annualized = (1 + short_return) ** (250 / Config.SHORT_LOOKBACK_DAYS) - 1
            
        # 3. 均线状态
        ma_price = np.mean(prices[-Config.MA_FILTER_DAYS:]) if len(prices) >= Config.MA_FILTER_DAYS else prices[0]
        current_above_ma = cur_price >= ma_price
        
        # 4. 短期风控（3日跌幅）
        day_ratios = []
        passed_loss = True
        if len(prices) >= 4:
            day_ratios = [prices[-1]/prices[-2], prices[-2]/prices[-3], prices[-3]/prices[-4]]
            if min(day_ratios) < Config.DROP_3DAY_LIMIT:
                passed_loss = False
                
        # 5. RSI指标
        current_rsi = 0
        max_recent_rsi = 0
        passed_rsi = True
        if Config.ENABLE_RSI_FILTER and len(prices) >= Config.RSI_PERIOD + Config.RSI_LOOKBACK_DAYS:
            rsi_values = calculate_rsi(prices, Config.RSI_PERIOD)
            if len(rsi_values) >= Config.RSI_LOOKBACK_DAYS:
                recent_rsi = rsi_values[-Config.RSI_LOOKBACK_DAYS:]
                max_recent_rsi = np.max(recent_rsi)
                current_rsi = recent_rsi[-1]
                if np.any(recent_rsi > Config.RSI_THRESHOLD):
                    # 加入MA5趋势保护，若价格跌破MA5且RSI过高才拦截
                    ma5 = np.mean(prices[-5:]) if len(prices) >= 5 else cur_price
                    if cur_price < ma5:
                        passed_rsi = False

        result = {
            'etf': etf,
            'score': score,
            'r2': details['r2'],
            'annualized_returns': details['ret'],
            'short_annualized': short_annualized,
            'day_ratios': day_ratios,
            'passed_momentum': passed_momentum,
            'passed_short_mom': short_annualized >= Config.SHORT_MOMENTUM_THRESHOLD,
            'passed_r2': details['r2'] > Config.R2_THRESHOLD,
            'passed_annual_ret': details['ret'] >= Config.MIN_ANNUALIZED_RETURN,
            'passed_ma': current_above_ma,
            'passed_loss': passed_loss,
            'passed_rsi': passed_rsi
        }
        return result
        
    except Exception as e:
        log.warn(f"Error scoring {etf}: {e}")
        return None

def apply_filters(metrics_list, context):
    """应用所有配置的风险过滤器"""
    steps = [
        ('动量得分', lambda m: m['passed_momentum'], True),
        ('短期动量', lambda m: m['passed_short_mom'], Config.ENABLE_SHORT_MOMENTUM),
        ('R²', lambda m: m['passed_r2'], Config.ENABLE_R2_FILTER),
        ('年化收益率', lambda m: m['passed_annual_ret'], Config.ENABLE_ANNUAL_RETURN_FILTER), 
        ('均线', lambda m: m['passed_ma'], Config.ENABLE_MA_FILTER),
        ('短期风控', lambda m: m['passed_loss'], Config.ENABLE_LOSS_FILTER),
        ('RSI', lambda m: m['passed_rsi'], Config.ENABLE_RSI_FILTER),
    ]
    
    filtered = metrics_list[:]
    log.info(f"== 过滤前初始标的数量: {len(filtered)} ==")
    
    for name, condition, is_enabled in steps:
        if is_enabled:
            before_len = len(filtered)
            filtered = [m for m in filtered if condition(m)]
            after_len = len(filtered)
            log.info(f"[{name}] 过滤器已开启, 过滤前: {before_len}, 过滤后: {after_len}")
    
    # 成交量过滤需要请求实时分时数据，放在最后单算以减少API调用
    if Config.ENABLE_VOLUME_CHECK:
        volume_passed = []
        for m in filtered:
            is_anomaly, ratio = check_volume_anomaly(m['etf'], context, Config.VOLUME_LOOKBACK, Config.VOLUME_THRESHOLD)
            m['volume_ratio'] = ratio
            if not is_anomaly:
                volume_passed.append(m)
            else:
                pass # log.info(f"过滤 {m['etf']}: 成交量异常放量 (Ratio: {ratio:.2f})")
        log.info(f"[成交量] 过滤器已开启, 过滤前: {len(filtered)}, 过滤后: {len(volume_passed)}")
        filtered = volume_passed
        
    return filtered

def get_target_list(context):
    """筛选目标ETF列表"""
    pool = g.etf_pool
    
    # 1. 计算所有 ETF指标
    all_metrics = []
    for etf in pool:
        res = get_etf_score(etf, context)
        if res: all_metrics.append(res)
        
    # 按分数降序
    all_metrics.sort(key=lambda x: x['score'], reverse=True)
    
    # 2. 管道过滤
    filtered_metrics = apply_filters(all_metrics, context)
    
    # 记录日志
    if filtered_metrics:
        top3_info = [f"{x['etf']}:{x['score']:.3f}" for x in filtered_metrics[:3]]
        log.info(f"排名最前的ETF: {top3_info}")
    else:
        log.info(f"所有目标被风险过滤器拦截，当前无符合条件的 ETF。")
    
    # 返回纯代码列表
    return [x['etf'] for x in filtered_metrics]


# ==================== 交易执行模块 ====================

def smart_order_target(security, value, context):
    """智能下单封装"""
    current_data = get_current_data()
    
    # 拿到仓位
    pos = context.portfolio.positions.get(security)
    current_position = pos.total_amount if pos else 0
    closeable_amount = pos.closeable_amount if pos else 0
    
    # === 如果目标为清仓 (value == 0) ===
    if value == 0:
        if closeable_amount > 0:
            o = order(security, -closeable_amount)
            if o:
                log.info(f"💰 [真实交易] 卖出清仓 {security}, 委托数量: {closeable_amount} 股")
        return True
        
    # === 以下是买入或持仓调整逻辑 ===
    if security not in current_data or current_data[security].paused:
        log.info(f"{security}: 今日停牌或无数据，放弃买入")
        return False
        
    price = current_data[security].last_price
    if math.isnan(price) or price <= 0:
        return False
        
    if price >= current_data[security].high_limit or price <= current_data[security].low_limit:
        # 涨跌停不买
        return False
        
    # 最小交易额检查
    current_val = current_position * price
    diff_val = abs(value - current_val)
    if diff_val < Config.MIN_MONEY: 
        return False

    # 计算目标持仓数量，必须按100的整数倍买
    target_position = (int(value / price) // 100) * 100
    adjustment = target_position - current_position
    
    if adjustment > 0:
        o = order(security, adjustment)
        if o:
            log.info(f"💰 [真实交易] 买入建仓/加仓 {security}, 委托买入: {adjustment} 股, 约合 {adjustment * price:.2f} 元")
            return True
    elif adjustment < 0:
        actual_sell = min(abs(adjustment), closeable_amount)
        if actual_sell > 0:
            o = order(security, -actual_sell)
            if o: 
                log.info(f"💰 [真实交易] 卖出减仓 {security}, 委托卖出: {actual_sell} 股, 约合 {actual_sell * price:.2f} 元")
                return True
            
    return False

# ---------- 日内止损监控 ----------
def simple_stop_loss(context):
    if not Config.ENABLE_INTRADAY_STOP_LOSS:
        return
        
    # log.info(f"=== 日内止损监控检查 ({context.current_dt.strftime('%H:%M')}) ===")
    current_data = get_current_data()
    
    for etf in list(context.portfolio.positions.keys()):
        # 忽略防御和避险ETF的止损（如果有的话）
        if etf == Config.DEFENSIVE_ETF:
            continue
            
        pos = context.portfolio.positions[etf]
        if pos.total_amount == 0: continue
        
        # 检查 T+1 限制，只有今天可卖出的份额大于 0 时才处理
        if pos.closeable_amount == 0:
            continue
            
        try:
            info = current_data[etf]
            current_price = info.last_price
        except Exception:
            continue
            
        cost_price = pos.avg_cost
        
        if current_price <= 0 or cost_price <= 0: continue
        
        if current_price <= cost_price * Config.STOP_LOSS_PCT:
            loss_percent = (current_price / cost_price - 1) * 100
            log.info(f"🚨 [日内监控] {context.current_dt.strftime('%H:%M')} 触发止损: {etf}，当前价: {current_price:.3f}, 成本: {cost_price:.3f}, 亏损: {loss_percent:.2f}%")
            
            # 使用 closeable_amount 下单，确保遵循 T+1 抛售旧仓位
            o = order(etf, -pos.closeable_amount)
            if o:
                log.info(f"✅ [日内监控] 止损卖出委托下达: {etf}, 数量: {pos.closeable_amount} 股")

# ---------- 卖出逻辑 (10:29) ----------
def etf_trade_sell(context):
    log.info("=== 开始执行卖出逻辑 (10:29) ===")
    
    # 1. 计算今日目标
    raw_targets = get_target_list(context)
    
    # 截取前 N 名作为“理想持仓”
    ideal_targets = raw_targets[:Config.HOLD_COUNT]
    
    # 如果无有效目标，切入防御模式
    if not ideal_targets:
        if check_defensive_etf_available(context):
            ideal_targets = [Config.DEFENSIVE_ETF]
            log.info(f"🛡️ 开启防御模式，今日理想目标为避险ETF: {ideal_targets}")
        else:
            log.info("💤 无有效目标且防御ETF不可交易，空仓状态")
            ideal_targets = []
            
    g.target_list = ideal_targets # 存给买入逻辑用
    
    current_positions = list(context.portfolio.positions.keys())
    
    # 2. 检查持仓是否需要卖出
    for etf in current_positions:
        pos = context.portfolio.positions[etf]
        if pos.total_amount == 0: continue
        
        should_sell = False
        reason = ""
        
        # 判断卖出条件并细化原因
        if etf not in ideal_targets:
            should_sell = True
            if etf == Config.DEFENSIVE_ETF:
                reason = "由守转攻，防御阶段结束"
            elif not ideal_targets or (len(ideal_targets) == 1 and ideal_targets[0] == Config.DEFENSIVE_ETF):
                reason = "触发防御避险机制"
            else:
                reason = "动量或排名掉队"
            
        if should_sell:
            smart_order_target(etf, 0, context)
            positions = list(context.portfolio.positions.keys())
            pos_str = ", ".join(positions) if positions else "空仓"
            log.info(f"卖出 {etf}: {reason}; 当前持仓: {pos_str}")


# ---------- 买入逻辑 (10:30) ----------
def etf_trade_buy(context):
    log.info("=== 开始执行买入逻辑 (10:30) ===")
    
    # 1. 获取候选名单 (复用卖出时计算的，或者重新计算)
    # 建议重新读取 g.target_list，若为空则重新算
    targets = g.target_list
    if not targets:
        log.info("无目标标的")
        return

    # 2. 确定买入目标
    final_buy_targets = targets 
        
    if not final_buy_targets:
        log.info("无有效买入目标")
        return
        
    log.info(f"最终买入目标: {final_buy_targets}")
    
    # 3. 执行买入
    total_value = context.portfolio.total_value
    per_value = total_value / len(final_buy_targets)
    available_cash = context.portfolio.available_cash
    
    current_data = get_current_data()
    
    for etf in final_buy_targets:
        if etf not in current_data or current_data[etf].paused:
            log.info(f"⏭️ 目标 {etf} 今日停牌或无数据，跳过买入")
            continue
            
        last_price = current_data[etf].last_price
        if math.isnan(last_price) or last_price <= 0:
            continue
            
        current_position = context.portfolio.positions[etf].total_amount if etf in context.portfolio.positions else 0
        current_value = current_position * last_price
        
        # 资金门槛拦截：如果是建仓或加仓，检查可用资金是否充分
        if current_value < per_value * 0.9:
            need_buy_value = per_value - current_value
            actual_buy_value = min(need_buy_value, available_cash)
            # 严格防范资金连一手都不够的情况
            if actual_buy_value <= max(Config.MIN_MONEY, last_price * 100):
                log.info(f"⚠️ 买入金额不足（需{need_buy_value:.2f}元，可用{available_cash:.2f}元），不买入碎片金额。")
                continue
                
            # 使用可真买的资金额加上当前持有额，确定最后要到达的额度
            safe_target_value = current_value + actual_buy_value
            smart_order_target(etf, safe_target_value, context)
        else:
            smart_order_target(etf, per_value, context)
        
        # 每操作完一只更新剩余资金
        available_cash = context.portfolio.available_cash

# ---------- 盘后处理 & 状态输出 ----------
def print_positions(context):
    positions = list(context.portfolio.positions.keys())
    if positions:
        # 为了更直观，可以连同名字和市值一起打印，这里按您的要求输出清单
        log.info(f"=== 当前持仓: {positions} ===")
    else:
        log.info("=== 当前持仓: 空仓 ===")
        
    if context.current_dt.hour >= 14:
        log.info("")