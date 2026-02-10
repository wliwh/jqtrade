# 策略名称：ETF收益率稳定性轮动策略（模块化重构版）
# 原始逻辑参考：ETF_atr.py
# 重构作者：Antigravity
# 说明：保留所有原有功能及未来函数修复逻辑，采用 Config 类及模块化架构。

import numpy as np
import math
import pandas as pd
from datetime import datetime, timedelta
from jqdata import *

class Config:
    # ==================== 交易环境设置 ====================
    BENCHMARK = "513100.XSHG"
    AVOID_FUTURE_DATA = True
    USE_REAL_PRICE = True
    
    # 滑点
    SLIPPAGE_FUND = 0.0005
    SLIPPAGE_STOCK = 0.003
    
    # 交易成本
    COMMISSION_FUND = 0.0003
    MIN_COMMISSION_FUND = 5
    
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
        # "159870.XSHE",   # 化工

        "501018.XSHG",  # 南方原油
        # 债券
        "511090.XSHG",  # 30年国债ETF
        # 国内
        "513130.XSHG",  # 恒生科技
        "513690.XSHG",  # 港股红利

        "510180.XSHG",  # 上证180
        "159915.XSHE",  # 创业板ETF

        "510410.XSHG",  # 资源
        "515650.XSHG",  # 消费50
        "512290.XSHG",  # 生物医药
        "588120.XSHG",  # 科创100
        "515070.XSHG",  # 人工智能ETF

        "159851.XSHE",  # 金融科技
        "159637.XSHE",  # 新能源车
        "516160.XSHG",  # 新能源

        "159550.XSHE",  # 互联网ETF
        "512710.XSHG",  # 军工ETF
        "159692.XSHE",  # 证券
        "512480.XSHG",  # 半导体
        "515250.XSHG",  # 智能汽车
        "159378.XSHE",  # 通用航空
        "516510.XSHG",  # 云计算
        "515050.XSHG",  # 5G通信
        "159995.XSHE",  # 芯片
        "515790.XSHG",  # 光伏
        "515000.XSHG",  # 科技
    ]
    
    LOOKBACK_DAYS = 25
    HOLDINGS_NUM = 1
    STOP_LOSS_RATIO = 0.95   # 固定比例止损 (7star默认0.95)
    LOSS_3DAY_THRESHOLD = 0.97 # 近3日跌幅限制 (7star默认0.97)
    DEFENSIVE_ETF = "511880.XSHG"
    MIN_SCORE_THRESHOLD = 0.0
    MAX_SCORE_THRESHOLD = 6.0
    MIN_MONEY = 5000
    
    # R2 筛选
    ENABLE_R2_FILTER = True
    R2_MIN_THRESHOLD = 0.3    # R²最低阈值
    
    # 短期动量过滤
    ENABLE_SHORT_MOMENTUM_FILTER = False # 7star默认关闭
    SHORT_LOOKBACK_DAYS = 10
    SHORT_MOMENTUM_THRESHOLD = 0.0
    
    # ATR 动态止损
    ENABLE_ATR_STOP_LOSS = False # 7star默认关闭
    ATR_PERIOD = 14
    ATR_MULTIPLIER = 2
    ATR_TRAILING_STOP = False
    ATR_EXCLUDE_DEFENSIVE = True
    
    # MA 过滤
    ENABLE_MA_FILTER = False # 7star默认关闭
    MA_SHORT_PERIOD = 5
    MA_LONG_PERIOD = 20
    MA_FILTER_CONDITION = "above" # "above" or "below"
    
    # RSI 过滤
    ENABLE_RSI_FILTER = False # 7star默认关闭
    RSI_PERIOD = 6
    RSI_LOOKBACK_DAYS = 1
    RSI_THRESHOLD = 98
    
    # MACD 过滤
    ENABLE_MACD_FILTER = False
    MACD_FAST_PERIOD = 12
    MACD_SLOW_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9
    MACD_FILTER_CONDITION = "bullish" # "bullish" or "bearish"
    
    # 成交量过滤 (7star核心升级)
    ENABLE_VOLUME_FILTER = True
    VOLUME_LOOKBACK_DAYS = 5
    VOLUME_THRESHOLD = 2.5
    VOLUME_RETURN_LIMIT = 1.0 # 年化收益率过滤阈值 (7star特有)
    VOLUME_EXCLUDE_DEFENSIVE = True
    
    # 布林带过滤
    ENABLE_BOLLINGER_FILTER = False
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2.0
    BOLLINGER_LOOKBACK_DAYS = 3

# ==================== 初始化 ====================
def initialize(context):
    set_option("avoid_future_data", Config.AVOID_FUTURE_DATA)
    set_option("use_real_price", Config.USE_REAL_PRICE)
    
    set_slippage(FixedSlippage(Config.SLIPPAGE_FUND), type="fund")
    set_slippage(FixedSlippage(Config.SLIPPAGE_STOCK), type="stock")
    
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=Config.COMMISSION_FUND, close_commission=Config.COMMISSION_FUND, close_today_commission=0, min_commission=Config.MIN_COMMISSION_FUND), type="fund")
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0, close_commission=0, close_today_commission=0, min_commission=0), type="mmf")
    
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    
    set_benchmark(Config.BENCHMARK)
    
    # 状态存储
    g.etf_pool = Config.ETF_POOL
    g.positions = {}
    g.position_highs = {}
    g.position_stop_prices = {}
    
    # 任务调度
    run_daily(etf_trade, time='14:00')
    run_daily(check_positions, time='09:30')
    run_daily(check_atr_stop_loss, time='09:30')

# ==================== 数据获取模块 ====================
def get_ref_price(security, context):
    """获取前一分钟价格，避免未来函数"""
    try:
        end_time = context.current_dt
        start_time = end_time - timedelta(minutes=2)
        minute_data = get_price(security, start_date=start_time, end_date=end_time, frequency='1m', fields=['close'], skip_paused=False, fq='pre', panel=False)
        
        if minute_data is None or len(minute_data) < 2:
            hist_data = attribute_history(security, 1, '1d', ['close'], skip_paused=True)
            return hist_data['close'].iloc[-1] if not hist_data.empty else 0
        
        return minute_data['close'].iloc[-2] if len(minute_data) >= 2 else minute_data['close'].iloc[-1]
    except:
        return 0

def get_ref_volume(security, context):
    """获取前一分钟成交量"""
    try:
        end_time = context.current_dt
        start_time = end_time - timedelta(minutes=2)
        minute_data = get_price(security, start_date=start_time, end_date=end_time, frequency='1m', fields=['volume'], skip_paused=False, fq='pre', panel=False)
        
        if minute_data is None or len(minute_data) < 2:
            hist_data = attribute_history(security, 1, '1d', ['volume'], skip_paused=True)
            return hist_data['volume'].iloc[-1] if not hist_data.empty else 0
        
        return minute_data['volume'].iloc[-2] if len(minute_data) >= 2 else minute_data['volume'].iloc[-1]
    except:
        return 0

# ==================== 指标计算模块 ====================
def calculate_atr(security, period=14):
    try:
        hist_data = attribute_history(security, period + 20, '1d', ['high', 'low', 'close'], skip_paused=True)
        if len(hist_data) < period + 1: return 0, False
        
        h, l, cp = hist_data['high'].values, hist_data['low'].values, hist_data['close'].values
        tr_values = np.zeros(len(h))
        for i in range(1, len(h)):
            tr1 = h[i] - l[i]
            tr2 = abs(h[i] - cp[i-1])
            tr3 = abs(l[i] - cp[i-1])
            tr_values[i] = max(tr1, tr2, tr3)
        atr = np.mean(tr_values[-period:])
        return atr, True
    except:
        return 0, False

def calculate_bollinger(prices, period=20, std_dev=2.0):
    if len(prices) < period: return None, None, None
    mid = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    return mid, mid + std_dev * std, mid - std_dev * std

def calculate_rsi(prices, period=6):
    if len(prices) < period + 1: return []
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = np.zeros(len(prices))
    avg_losses = np.zeros(len(prices))
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])
    
    rsi_values = np.zeros(len(prices))
    for i in range(period + 1, len(prices)):
        avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
        avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        if avg_losses[i] == 0: rsi_values[i] = 100
        else:
            rs = avg_gains[i] / avg_losses[i]
            rsi_values[i] = 100 - (100 / (1 + rs))
    return rsi_values[period:]

def calculate_macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal: return 0, 0, 0
    def ema(data, p):
        res = np.zeros_like(data)
        res[0], alpha = data[0], 2 / (p + 1)
        for i in range(1, len(data)): res[i] = alpha * data[i] + (1 - alpha) * res[i-1]
        return res
    ema_f = ema(prices, fast)
    ema_s = ema(prices, slow)
    dif = ema_f - ema_s
    dea = ema(dif, signal)
    return dif[-1], dea[-1], (dif - dea)[-1]

# ==================== 过滤逻辑模块 ====================
def check_filters(etf, context, price_series, current_price, data_cache):
    # 1. MA 过滤
    ma5 = np.mean(price_series[-5:])
    if Config.ENABLE_MA_FILTER:
        ma_l = np.mean(price_series[-Config.MA_LONG_PERIOD:])
        met = (ma5 >= ma_l) if Config.MA_FILTER_CONDITION == "above" else (ma5 <= ma_l)
        if not met: 
            log.info(f"📊 {etf} MA过滤未通过")
            return False

    # 2. RSI 过滤
    if Config.ENABLE_RSI_FILTER:
        rsi_vals = calculate_rsi(price_series, Config.RSI_PERIOD)
        if len(rsi_vals) >= Config.RSI_LOOKBACK_DAYS:
            recent_rsi = rsi_vals[-Config.RSI_LOOKBACK_DAYS:]
            if np.any(recent_rsi > Config.RSI_THRESHOLD) and current_price < ma5:
                log.info(f"⛔ {etf} RSI过滤未通过")
                return False

    # 3. MACD 过滤
    if Config.ENABLE_MACD_FILTER:
        dif, dea, bar = calculate_macd(price_series, Config.MACD_FAST_PERIOD, Config.MACD_SLOW_PERIOD, Config.MACD_SIGNAL_PERIOD)
        met = (dif > dea) if Config.MACD_FILTER_CONDITION == "bullish" else (dif < dea)
        if not met: 
            log.info(f"📉 {etf} MACD过滤未通过")
            return False

    # 4. 成交量过滤 (7star增强版: 放量且高位才过滤)
    if Config.ENABLE_VOLUME_FILTER and not (Config.VOLUME_EXCLUDE_DEFENSIVE and etf == Config.DEFENSIVE_ETF):
        hist = data_cache.get('daily', {}).get(etf)
        if hist is not None and not hist.empty:
            # 使用缓存中的 1d 数据计算均值
            avg_vol = hist['volume'].iloc[-(Config.VOLUME_LOOKBACK_DAYS+1):-1].mean()
            # 获取当日实时成交量 (从data_cache中获取的是累积成交量)
            cur_vol = data_cache.get('volume', {}).get(etf, 0)
            
            if avg_vol > 0:
                vol_ratio = cur_vol / avg_vol
                if vol_ratio > Config.VOLUME_THRESHOLD:
                    # 只有放量且年化收益率过高时才过滤 (避免底部启动被过滤)
                    # 计算 lookback_days 的年化收益
                    # data_cache['daily'][etf]['close'] 是 Series
                    prices = hist['close'].iloc[-(Config.LOOKBACK_DAYS+1):]
                    # 需要加上当前价格来计算最新的年化收益? 
                    # get_etf_score 里传进来的 price_series 包含了当前价格
                    
                    # 计算加权线性回归年化收益
                    try:
                        p_series = price_series[-(Config.LOOKBACK_DAYS+1):]
                        y = np.log(p_series)
                        x = np.arange(len(y))
                        w = np.linspace(1, 2, len(y))
                        slope, _ = np.polyfit(x, y, 1, w=w)
                        ann_ret = math.exp(slope * 250) - 1
                        
                        if ann_ret > Config.VOLUME_RETURN_LIMIT:
                            log.info(f"📊 {etf} 高位放量被过滤: 量比{vol_ratio:.2f} > {Config.VOLUME_THRESHOLD}, 年化{ann_ret:.2f} > {Config.VOLUME_RETURN_LIMIT}")
                            return False
                        else:
                            log.debug(f"{etf} 放量但涨幅未超标: 量比{vol_ratio:.2f}, 年化{ann_ret:.2f}")
                    except Exception as e:
                        log.warn(f"计算年化收益出错: {e}")
                        pass

    # 5. 布林带过滤
    if Config.ENABLE_BOLLINGER_FILTER:
        # 获取多日数据以计算布林带序列
        lookback = Config.BOLLINGER_LOOKBACK_DAYS
        passed = True
        for i in range(lookback):
            idx = len(price_series) - lookback + i
            sub_series = price_series[:idx]
            if len(sub_series) < Config.BOLLINGER_PERIOD: continue
            mid, up, low = calculate_bollinger(sub_series, Config.BOLLINGER_PERIOD, Config.BOLLINGER_STD)
            if price_series[idx-1] > up:
                if current_price < ma5:
                    passed = False
                    break
        if not passed:
            log.info(f"📈 {etf} 布林带过滤未通过")
            return False
                 
    return True

# ==================== 核心评分模块 ====================
def get_etf_score(etf, context, data_cache):
    try:
        # 从缓存读取数据
        hist = data_cache.get('daily', {}).get(etf)
        if hist is None or len(hist) < Config.LOOKBACK_DAYS: return None
        
        # 显式检查停牌 (7star逻辑: 停牌标的直接跳过排名)
        curr_data = get_current_data()
        if curr_data[etf].paused:
            log.debug(f"{etf} 今日停牌，跳过计算")
            return None

        cur_p = data_cache.get('minute', {}).get(etf, 0)
        if cur_p <= 0: return None
        
        price_series = np.append(hist['close'].values, cur_p)
        
        # 短期动量过滤
        if Config.ENABLE_SHORT_MOMENTUM_FILTER:
            short_ret = price_series[-1] / price_series[-(Config.SHORT_LOOKBACK_DAYS + 1)] - 1
            if short_ret < Config.SHORT_MOMENTUM_THRESHOLD:
                log.info(f"📉 {etf} 短期动量不足: {short_ret:.4f}")
                return None
            
        # 3日跌幅限制
        if len(price_series) >= 4:
            if min(price_series[-1]/price_series[-2], price_series[-2]/price_series[-3], price_series[-3]/price_series[-4]) < Config.LOSS_3DAY_THRESHOLD:
                log.info(f"⚠️ {etf} 近3日跌幅过大")
                return None

        # 核心动量得分 (7star逻辑: WLS R2 * Annualized Return)
        # 用 log 价格做线性回归
        # 7star: weights = np.linspace(1, 2, len(y))
        y = np.log(price_series[-Config.LOOKBACK_DAYS:])
        x = np.arange(len(y))
        w = np.linspace(1, 2, len(y))
        
        slope, intercept = np.polyfit(x, y, 1, w=w)
        
        # 年化收益率 (基于回归斜率)
        ann_ret = math.exp(slope * 250) - 1
        
        # R2 计算
        # predicted = slope * x + intercept
        # ss_res = sum(w * (y - predicted)^2)
        # ss_tot = sum(w * (y - mean(y))^2)
        y_pred = slope * x + intercept
        ss_res = np.sum(w * (y - y_pred) ** 2)
        ss_tot = np.sum(w * (y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        
        # R2 过滤
        if Config.ENABLE_R2_FILTER and r2 < Config.R2_MIN_THRESHOLD:
            log.info(f"📉 {etf} R²过滤未通过: {r2:.4f} < {Config.R2_MIN_THRESHOLD}")
            return None
            
        score = ann_ret * r2
        
        # 检查其它过滤器
        if not check_filters(etf, context, price_series, cur_p, data_cache):
            return None
        
        if Config.MIN_SCORE_THRESHOLD <= score < Config.MAX_SCORE_THRESHOLD:
            return {'etf': etf, 'score': score, 'price': cur_p, 'ann_ret': ann_ret, 'r2': r2}
        
        log.info(f"排除异常值ETF: {etf}，得分: {score:.4f}")
        return None
    except Exception as e:
        log.warn(f"计算{etf}得分出错: {e}")
        return None

# ==================== 交易执行模块 ====================
def smart_order(security, target_value, context, data_cache=None):
    # 如果有缓存优先使用缓存价格
    cur_p = data_cache.get('minute', {}).get(security, 0) if data_cache else get_ref_price(security, context)
    if cur_p <= 0: return False
    
    data = get_current_data()
    if data[security].paused or cur_p >= data[security].high_limit or cur_p <= data[security].low_limit:
        return False
        
    # 计算目标数量 (向下取整到100)
    target_amount = int(target_value / cur_p)
    target_amount = (target_amount // 100) * 100
    if target_amount <= 0 and target_value > 0:
        target_amount = 100 # 如果目标价值大于0但算出来是0股，至少买1手? 或者保持0? 7star里是这样的
    
    pos = context.portfolio.positions.get(security)
    cur_amount = pos.total_amount if pos else 0
    diff = target_amount - cur_amount
    
    # 调仓阈值检查 (7star逻辑: 偏差<5%不调仓)
    current_val = cur_amount * cur_p
    # 如果是空仓买入(current_val=0)或者清仓卖出(target_value=0)，则必须执行
    # 只有在持仓调整时才检查阈值
    if current_val > 0 and target_value > 0:
        if abs(current_val - target_value) < target_value * 0.05:
            log.info(f"跳过调仓 {security}: 市值偏差<5% (现{current_val:.0f}/目{target_value:.0f})")
            return False

    # 最小交易额检查 (仅针对有变动的交易)
    if diff != 0:
        trade_val = abs(diff * cur_p)
        if trade_val < Config.MIN_MONEY:
            log.info(f"跳过交易 {security}: 金额 {trade_val:.2f} < 最小限制 {Config.MIN_MONEY}")
            return False

    # 卖出检查 T+1
    if diff < 0:
        closeable = pos.closeable_amount if pos else 0
        if closeable == 0:
            log.info(f"无法卖出 {security}: T+1限制 (可卖数=0)")
            return False
        # 如果要卖出的量超过可卖量，只卖可卖的
        if abs(diff) > closeable:
            diff = -closeable
            log.info(f"修正卖出量 {security}: 调整为可卖量 {diff}")

    if diff != 0:
        if order(security, diff):
            if diff > 0 and security in g.etf_pool:
                g.position_highs[security] = cur_p
                # 初始化 ATR 止损 (如果启用)
                if Config.ENABLE_ATR_STOP_LOSS:
                    atr, ok = calculate_atr(security, Config.ATR_PERIOD)
                    if ok:
                        if Config.ATR_TRAILING_STOP:
                            g.position_stop_prices[security] = cur_p - Config.ATR_MULTIPLIER * atr
                        else:
                            # 固定 ATR 止损基于成本价? 7star里似乎比较复杂，这里简化为由 check_atr_stop_loss 统一处理
                            pass
            action = "买入" if diff > 0 else "卖出"
            log.info(f"📦 {action} {security}: {diff}股, 价格 {cur_p:.3f}")
            return True
        else:
            log.warn(f"下单失败 {security}: {diff}股")
            return False

def etf_trade(context):
    """ETF轮动交易主函数 - 性能优化版 (数据缓存)"""
    # 0. 数据预分析与批量获取
    fetch_len = max(Config.LOOKBACK_DAYS, Config.MA_LONG_PERIOD, Config.BOLLINGER_PERIOD) + 10
    
    # 批量获取日线数据
    daily_data = {etf: attribute_history(etf, fetch_len, '1d', ['close', 'high', 'low', 'volume'], skip_paused=True) for etf in g.etf_pool}
    
    # 获取前一分钟价格 (批量处理)
    end_time = context.current_dt
    start_time = end_time - timedelta(minutes=2)
    minute_prices = {}
    minute_volumes = {}
    
    for etf in g.etf_pool:
        prices = get_price(etf, start_date=start_time, end_date=end_time, frequency='1m', fields=['close', 'volume'], skip_paused=False, fq='pre', panel=False)
        if prices is not None and len(prices) >= 2:
            minute_prices[etf] = prices['close'].iloc[-2]
            minute_volumes[etf] = prices['volume'].iloc[-2]
        elif prices is not None and len(prices) == 1:
            minute_prices[etf] = prices['close'].iloc[-1]
            minute_volumes[etf] = prices['volume'].iloc[-1]
        else:
            # 降级到昨收
            hist = daily_data.get(etf) if isinstance(daily_data, dict) else (daily_data[etf] if etf in daily_data else None)
            minute_prices[etf] = hist['close'].iloc[-1] if hist is not None else 0
            minute_volumes[etf] = hist['volume'].iloc[-1] if hist is not None else 0

    data_cache = {
        'daily': daily_data if isinstance(daily_data, dict) else {etf: daily_data[etf] for etf in g.etf_pool if etf in daily_data},
        'minute': minute_prices,
        'volume': minute_volumes
    }

    # 1. 计算得分
    scores = []
    for etf in g.etf_pool:
        res = get_etf_score(etf, context, data_cache)
        if res: scores.append(res)
    
    scores.sort(key=lambda x: x['score'], reverse=True)
    
    log.info("=== ETF趋势指标分析 ===")
    for m in scores:
        log.info(f"{m['etf']}: 年化={m['ann_ret']:.4f}, R²={m['r2']:.4f}, 得分={m['score']:.4f}, 当前价={m['price']:.3f}")
    
    target_etfs = []
    if scores and scores[0]['score'] >= Config.MIN_SCORE_THRESHOLD:
        target_etfs = [x['etf'] for x in scores[:Config.HOLDINGS_NUM]]
        log.info(f"🎯 正常模式，选择目标ETF: {target_etfs}")
    else:
        if is_defensive_ready(context):
            target_etfs = [Config.DEFENSIVE_ETF]
            log.info(f"🛡️ 进入防御模式: {Config.DEFENSIVE_ETF}")
        else:
            log.info("💤 进入空仓模式")
    
    target_set = set(target_etfs)
    total_val = context.portfolio.total_value
    target_val_per = total_val / len(target_etfs) if target_etfs else 0
    
    # 2. 卖出不在目标列表中的
    # 7star 逻辑: 先卖出非目标持仓，如果有持仓未卖出 (例如停牌)，则不进行买入
    # 这里我们先执行卖出
    has_sell_failed = False
    
    for sec in list(context.portfolio.positions.keys()):
        if sec in g.etf_pool and sec not in target_set:
            if not smart_order(sec, 0, context, data_cache):
                # 如果卖出失败 (例如停牌 or 跌停)，标记
                # 需再次确认是否真的持仓还在
                pos = context.portfolio.positions[sec]
                if pos.total_amount > 0:
                    log.warn(f"⚠️ {sec} 卖出失败或未完全卖出，可能影响后续买入")
                    has_sell_failed = True
            else:
                log.info(f"📤 卖出: {sec} (不在目标列表中)")
            
    # 3. 买入/调仓
    # 7star 核心风控: 如果有非目标持仓未卖出，则暂停买入，防止资金占用或风险敞口
    if has_sell_failed:
        log.warn("⛔ 存在未卖出的非目标持仓，暂停买入操作")
        return

    for etf in target_etfs:
        smart_order(etf, target_val_per, context, data_cache)

def is_defensive_ready(context):
    d = get_current_data()[Config.DEFENSIVE_ETF]
    return not d.paused and get_ref_price(Config.DEFENSIVE_ETF, context) < d.high_limit

# ==================== 风控模块 ====================
def check_positions(context):
    for sec, pos in context.portfolio.positions.items():
        if pos.total_amount > 0:
            log.info(f"持仓: {sec}, 价格: {pos.price}, 成本: {pos.avg_cost}")

def check_atr_stop_loss(context):
    if not Config.ENABLE_ATR_STOP_LOSS: return
    for sec in list(context.portfolio.positions.keys()):
        if Config.ATR_EXCLUDE_DEFENSIVE and sec == Config.DEFENSIVE_ETF: continue
        pos = context.portfolio.positions[sec]
        if pos.total_amount <= 0: continue
        
        cur_p = get_ref_price(sec, context)
        atr, ok = calculate_atr(sec, Config.ATR_PERIOD)
        if not ok: continue
        
        g.position_highs[sec] = max(g.position_highs.get(sec, cur_p), cur_p)
        ref_p = g.position_highs[sec] if Config.ATR_TRAILING_STOP else pos.avg_cost
        stop_p = ref_p - Config.ATR_MULTIPLIER * atr
        
        if cur_p <= stop_p or cur_p <= pos.avg_cost * Config.STOP_LOSS_RATIO:
            # 兼容 7star 的智能卖出
            # 这里调用 smart_order(sec, 0, ...) 会执行清仓
            if smart_order(sec, 0, context):
                loss_pct = (cur_p / pos.avg_cost - 1) * 100
                log.info(f"🚨 止损卖出: {sec}, 现价: {cur_p}, 成本: {pos.avg_cost}, 亏损: {loss_pct:.2f}%")
                
                # 清理状态
                g.position_highs.pop(sec, None)
                g.position_stop_prices.pop(sec, None)
