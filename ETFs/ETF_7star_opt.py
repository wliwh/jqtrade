# 克隆自聚宽文章：https://www.joinquant.com/post/67438
# 修正优化版：集中参数配置，仅使用固定池

import numpy as np                      # 数值计算
import math                             # 数学函数
import pandas as pd                     # 数据处理
from jqdata import *                    # 聚宽数据接口
from datetime import datetime, date     # 日期时间处理

# ==================== 自动化优化参数占位符与 Config 类 ====================
# 这里按照 ETF_gao_opt.py 的形式集中配置参数

EXECUTION_ETF_POOL = [
    '518880.XSHG', '161226.XSHE', '159980.XSHE', '501018.XSHG', '159985.XSHE',
    '513100.XSHG', '159509.XSHE', '513290.XSHG', '513500.XSHG', '159518.XSHE',
    '159502.XSHE', '159529.XSHE', '513400.XSHG', '520830.XSHG', '513520.XSHG',
    '513030.XSHG', '513090.XSHG', '513180.XSHG', '513120.XSHG', '513330.XSHG',
    '513750.XSHG', '159892.XSHE', '159605.XSHE', '513190.XSHG', '510900.XSHG',
    '513630.XSHG', '513920.XSHG', '159323.XSHE', '513970.XSHG', '510500.XSHG',
    '512100.XSHG', '563300.XSHG', '510300.XSHG', '512050.XSHG', '510760.XSHG',
    '159915.XSHE', '159949.XSHE', '159967.XSHE', '588080.XSHG', '588220.XSHG',
    '511380.XSHG', '513310.XSHG', '588200.XSHG', '159852.XSHE', '512880.XSHG',
    '159206.XSHE', '512400.XSHG', '512980.XSHG', '159516.XSHE', '512480.XSHG',
    '515880.XSHG', '562500.XSHG', '159218.XSHE', '159869.XSHE', '159870.XSHE',
    '159326.XSHE', '159851.XSHE', '560860.XSHG', '159363.XSHE', '588170.XSHG',
    '159755.XSHE', '512170.XSHG', '512800.XSHG', '159819.XSHE', '512710.XSHG',
    '159638.XSHE', '517520.XSHG', '515980.XSHG', '159995.XSHE', '159227.XSHE',
    '512660.XSHG', '512690.XSHG', '516150.XSHG', '512890.XSHG', '588790.XSHG',
    '159992.XSHE', '512070.XSHG', '562800.XSHG', '512010.XSHG', '515790.XSHG',
    '510880.XSHG', '159928.XSHE', '159883.XSHE', '159998.XSHE', '515220.XSHG',
    '561980.XSHG', '515400.XSHG', '515120.XSHG', '159566.XSHE', '515050.XSHG',
    '516510.XSHG', '159256.XSHE', '159766.XSHE', '512200.XSHG', '513350.XSHG',
    '159583.XSHE', '159732.XSHE', '516160.XSHG', '516520.XSHG', '562590.XSHG',
    '515030.XSHG', '512670.XSHG', '561330.XSHG', '516190.XSHG', '159840.XSHE',
    '159611.XSHE', '159981.XSHE', '159865.XSHE', '561360.XSHG', '159667.XSHE',
    '515170.XSHG', '513360.XSHG', '159825.XSHE', '515210.XSHG'
]

EXECUTION_SCORE_THRESHOLD = (0, 5)

EXECUTION_SHORT_MOMENTUM_PARAM = (False, 10, 0.0)      # (启用, 天数, 阈值)
EXECUTION_R2_PARAM = (True, 0.4)                      # (启用, 阈值)
EXECUTION_ANNUAL_RETURN_PARAM = (False, 1.0)           # (启用, 阈值)
EXECUTION_MA_PARAM = (False, 20)                      # (启用, 天数)
EXECUTION_VOLUME_PARAM = (True, 5, 1.0)               # (启用, 回看, 阈值)
EXECUTION_LOSS_PARAM = (True, 0.97)                   # (启用, 阈值)
EXECUTION_RSI_PARAM = (False, 6, 1, 98)                # (启用, 周期, 回看, 阈值)

EXECUTION_FIXED_STOPLOSS = (True, 0.95)               # (启用, 阈值)
EXECUTION_PCT_STOPLOSS = (False, 0.95)                # (启用, 阈值)
EXECUTION_ATR_STOPLOSS = (False, 14, 2, True, True)   # (启用, 周期, 倍数, 跟踪, 排除防御)

EXECUTION_SELL_COOLDOWN = (True, 3)                  # (启用, 天数)

class Config:
    # 策略全局配置类
    FIXED_ETF_POOL = EXECUTION_ETF_POOL
    HOLDINGS_NUM = 1
    DEFENSIVE_ETF = "511880.XSHG"     # 防御型ETF
    SAFE_HAVEN_ETF = '511660.XSHG'    # 冷却期避险ETF
    MIN_MONEY = 5000                  # 最小交易金额
    
    # 动量计算
    LOOKBACK_DAYS = 25
    MIN_SCORE_THRESHOLD, MAX_SCORE_THRESHOLD = EXECUTION_SCORE_THRESHOLD

    # 过滤器配置
    USE_SHORT_MOMENTUM_FILTER, SHORT_LOOKBACK_DAYS, SHORT_MOMENTUM_THRESHOLD = EXECUTION_SHORT_MOMENTUM_PARAM

    ENABLE_R2_FILTER, R2_THRESHOLD = EXECUTION_R2_PARAM

    ENABLE_ANNUAL_RETURN_FILTER, MIN_ANNUAL_RETURN = EXECUTION_ANNUAL_RETURN_PARAM

    ENABLE_MA_FILTER, MA_FILTER_DAYS = EXECUTION_MA_PARAM

    ENABLE_VOLUME_CHECK, VOLUME_LOOKBACK, VOLUME_THRESHOLD = EXECUTION_VOLUME_PARAM

    ENABLE_LOSS_FILTER, LOSS = EXECUTION_LOSS_PARAM

    USE_RSI_FILTER, RSI_PERIOD, RSI_LOOKBACK_DAYS, RSI_THRESHOLD = EXECUTION_RSI_PARAM

    # 止损配置
    USE_FIXED_STOP_LOSS, FIXED_STOP_LOSS_THRESHOLD = EXECUTION_FIXED_STOPLOSS
    USE_PCT_STOP_LOSS, PCT_STOP_LOSS_THRESHOLD = EXECUTION_PCT_STOPLOSS
    USE_ATR_STOP_LOSS, ATR_PERIOD, ATR_MULTIPLIER, ATR_TRAILING_STOP, ATR_EXCLUDE_DEFENSIVE = EXECUTION_ATR_STOPLOSS

    # 冷却期
    SELL_COOLDOWN_ENABLED, SELL_COOLDOWN_DAYS = EXECUTION_SELL_COOLDOWN

# --- 策略主控与初始化 ---
def initialize(context):                # 初始化策略
    set_option("avoid_future_data", True)       # 避免未来函数
    set_option("use_real_price", True)          # 使用真实价格
    
    set_slippage(PriceRelatedSlippage(0.0001), type="fund")  # 设置滑点
    
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0.0001, close_commission=0.0001, close_today_commission=0.0001, min_commission=5,), type="fund")  # 设置交易费用

    log.set_level('order', 'error')     # 降低日志级别
    log.set_level('system', 'error')
    log.set_level('strategy', 'info')
    log.info("【优化版】策略初始化完成！已应用参数集中化配置。")

    set_benchmark("510300.XSHG")        # 设置基准

    # 从 Config 类初始全局状态 g
    # 全局变量初始化
    g.positions = {}                    # 记录目标持仓
    g.position_highs = {}               # 记录持仓最高价（用于ATR跟踪）
    g.position_stop_prices = {}         # 记录ATR止损价
    g.target_etfs_list = []             # 今日目标ETF列表
    g.cooldown_end_date = None          # 冷却期结束日期

    run_daily(check_positions, time='09:10')        # 盘前检查持仓
    run_daily(etf_sell_trade, time='13:10')         # 卖出交易
    run_daily(etf_buy_trade, time='13:11')          # 买入交易
    # 原有的 run_daily(update_sector_pool, time='09:00') 已移除

    for tm in ('09:35','10:00','11:00','13:30','14:30'):
        run_daily(minute_level_stop_loss, time=tm)          # 固定比例止损
        run_daily(minute_level_pct_stop_loss, time=tm)      # 当日跌幅止损
        run_daily(minute_level_atr_stop_loss, time=tm)      # ATR动态止损

    log.info(f"""策略参数初始化完成 (固定池版):
=== 过滤条件 ===
- 动量得分过滤: {'启用' if (Config.MIN_SCORE_THRESHOLD > -1e9 or Config.MAX_SCORE_THRESHOLD < 1e9) else '禁用'} (范围: [{Config.MIN_SCORE_THRESHOLD}, {Config.MAX_SCORE_THRESHOLD}])
- 短期动量过滤: {'启用' if Config.USE_SHORT_MOMENTUM_FILTER else '禁用'} (周期: {Config.SHORT_LOOKBACK_DAYS}天)
- R²过滤: {'启用' if Config.ENABLE_R2_FILTER else '禁用'} (阈值 > {Config.R2_THRESHOLD:.3f})
- 年化收益率过滤: {'启用' if Config.ENABLE_ANNUAL_RETURN_FILTER else '禁用'} (阈值 > {Config.MIN_ANNUAL_RETURN:.1%})
- 均线过滤: {'启用' if Config.ENABLE_MA_FILTER else '禁用'} ({Config.MA_FILTER_DAYS}日均线)
- 成交量过滤: {'启用' if Config.ENABLE_VOLUME_CHECK else '禁用'} (均量比 < {Config.VOLUME_THRESHOLD:.2f})
- 短期风控过滤: {'启用' if Config.ENABLE_LOSS_FILTER else '禁用'} (最大单日跌幅限制 < {1 - Config.LOSS:.1%})
- RSI过滤: {'启用' if Config.USE_RSI_FILTER else '禁用'} (周期: {Config.RSI_PERIOD}, 超买阈值 > {Config.RSI_THRESHOLD})
- 止损机制: 固定比例止损({'启用' if Config.USE_FIXED_STOP_LOSS else '禁用'}), 当日跌幅止损({'启用' if Config.USE_PCT_STOP_LOSS else '禁用'}), ATR动下止损({'启用' if Config.USE_ATR_STOP_LOSS else '禁用'})
- 固定ETF池大小: {len(Config.FIXED_ETF_POOL)} 只ETF
- 动量计算周期: {Config.LOOKBACK_DAYS} 天
- 持仓数量: {Config.HOLDINGS_NUM}
- 防御ETF: {Config.DEFENSIVE_ETF}
""")

# --- 指标计算与筛选 ---
def calculate_all_metrics_for_etf(context, etf):  # 计算单个ETF的所有指标
    try:
        etf_name = get_security_name(etf)         # 获取ETF名称
        
        lookback = max(                           # 确定所需历史数据长度
            Config.LOOKBACK_DAYS,
            Config.SHORT_LOOKBACK_DAYS,
            Config.RSI_PERIOD + Config.RSI_LOOKBACK_DAYS,
            Config.MA_FILTER_DAYS,
            Config.VOLUME_LOOKBACK
        ) + 20
        
        prices = attribute_history(etf, lookback, '1d', ['close', 'high', 'low'])  # 获取历史价格
        current_data = get_current_data()
        
        if len(prices) < max(Config.LOOKBACK_DAYS, Config.MA_FILTER_DAYS):
            return None
            
        current_price = current_data[etf].last_price
        price_series = np.append(prices["close"].values, current_price)  # 拼接当前价格

        recent_price_series = price_series[-(Config.LOOKBACK_DAYS + 1):]      # 动量计算序列
        y = np.log(recent_price_series)
        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))                              # 加权拟合
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        annualized_returns = math.exp(slope * 250) - 1                   # 年化收益率
        ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot else 0                 # R²
        momentum_score = annualized_returns * r_squared                  # 动量得分

        if len(price_series) >= Config.SHORT_LOOKBACK_DAYS + 1:              # 短期动量
            short_return = price_series[-1] / price_series[-(Config.SHORT_LOOKBACK_DAYS + 1)] - 1
            short_annualized = (1 + short_return) ** (250 / Config.SHORT_LOOKBACK_DAYS) - 1
        else:
            short_annualized = -np.inf

        ma_price = np.mean(price_series[-Config.MA_FILTER_DAYS:])            # 均线价格
        current_above_ma = current_price >= ma_price                    # 是否站上均线

        volume_ratio = get_volume_ratio(context, etf, show_detail_log=False)  # 成交量比

        day_ratios = []                                                 # 短期风控（近3日跌幅）
        passed_loss_filter = True
        if len(price_series) >= 4:
            day1 = price_series[-1] / price_series[-2]
            day2 = price_series[-2] / price_series[-3]
            day3 = price_series[-3] / price_series[-4]
            day_ratios = [day1, day2, day3]
            if min(day_ratios) < Config.LOSS:
                passed_loss_filter = False

        current_rsi = 0                                                 # RSI指标
        max_recent_rsi = 0
        passed_rsi_filter = True
        if Config.USE_RSI_FILTER and len(price_series) >= Config.RSI_PERIOD + Config.RSI_LOOKBACK_DAYS:
            rsi_values = calculate_rsi(price_series, Config.RSI_PERIOD)
            if len(rsi_values) >= Config.RSI_LOOKBACK_DAYS:
                recent_rsi = rsi_values[-Config.RSI_LOOKBACK_DAYS:]
                max_recent_rsi = np.max(recent_rsi)
                current_rsi = recent_rsi[-1]
                if np.any(recent_rsi > Config.RSI_THRESHOLD):
                    ma5 = np.mean(price_series[-5:]) if len(price_series) >= 5 else current_price
                    if current_price < ma5:
                        passed_rsi_filter = False

        return {
            'etf': etf,
            'etf_name': etf_name,
            'momentum_score': momentum_score,
            'annualized_returns': annualized_returns,
            'r_squared': r_squared,
            'short_annualized': short_annualized,
            'current_price': current_price,
            'ma_price': ma_price,
            'volume_ratio': volume_ratio,
            'day_ratios': day_ratios,
            'current_rsi': current_rsi,
            'max_recent_rsi': max_recent_rsi,
            'passed_momentum': Config.MIN_SCORE_THRESHOLD <= momentum_score <= Config.MAX_SCORE_THRESHOLD,
            'passed_short_mom': short_annualized >= Config.SHORT_MOMENTUM_THRESHOLD,
            'passed_r2': r_squared > Config.R2_THRESHOLD,
            'passed_annual_ret': annualized_returns >= Config.MIN_ANNUAL_RETURN,
            'passed_ma': current_above_ma,
            'passed_volume': volume_ratio is not None and volume_ratio < Config.VOLUME_THRESHOLD,
            'passed_loss': passed_loss_filter,
            'passed_rsi': passed_rsi_filter,
        }
    except Exception as e:
        log.warning(f"计算 {etf} 指标出错: {e}")
        return None

def apply_filters(metrics_list):        # 应用所有过滤条件
    steps = [
        ('动量得分', lambda m: m['passed_momentum'], True),
        ('短期动量', lambda m: m['passed_short_mom'], Config.USE_SHORT_MOMENTUM_FILTER),
        ('R²', lambda m: m['passed_r2'], Config.ENABLE_R2_FILTER),
        ('年化收益率', lambda m: m['passed_annual_ret'], Config.ENABLE_ANNUAL_RETURN_FILTER), 
        ('均线', lambda m: m['passed_ma'], Config.ENABLE_MA_FILTER),
        ('成交量', lambda m: m['passed_volume'], Config.ENABLE_VOLUME_CHECK),
        ('短期风控', lambda m: m['passed_loss'], Config.ENABLE_LOSS_FILTER),
        ('RSI', lambda m: m['passed_rsi'], Config.USE_RSI_FILTER),
    ]
    
    filtered = metrics_list[:]
    for name, condition, is_enabled in steps:  # 逐个应用启用的过滤器
        if is_enabled:
            filtered = [m for m in filtered if condition(m)]
    return filtered
    
def get_final_ranked_etfs(context):     # 主筛选函数：合并池、分类、计算、排序
    all_metrics = []
    # 优化版：仅使用固定池
    etf_set = sorted(list(set(Config.FIXED_ETF_POOL)))

    end_date = context.previous_date    # 昨日日期

    try:
        h = get_price(list(etf_set), count=1, end_date=end_date, frequency='daily', fields=['money'])  # 获取昨日成交额
        yesterday_money = h['money'].iloc[0]
    except Exception:
        yesterday_money = pd.Series(dtype=float)

    for etf in etf_set:                 # 遍历池子计算指标
        try:
            info = get_security_info(etf)
            start_date_raw = info.start_date if info else None
        except Exception:
            start_date_raw = None

        if start_date_raw is None:
            start_date = None
        elif isinstance(start_date_raw, datetime):
            start_date = start_date_raw.date()
        elif isinstance(start_date_raw, date):
            start_date = start_date_raw
        else:
            start_date = None

        if start_date is None or end_date < start_date:  # 跳过未上市
            continue

        if pd.isna(yesterday_money.get(etf, np.nan)):    # 跳过无成交数据
            continue

        current_data = get_current_data()
        if current_data[etf].paused:                     # 跳过停牌
            continue

        metrics = calculate_all_metrics_for_etf(context, etf)
        if metrics:
            all_metrics.append(metrics)

    for item in all_metrics:            # 处理无效动量得分
        score = item.get('momentum_score')
        if pd.isna(score) or (isinstance(score, float) and np.isnan(score)):
            item['momentum_score'] = float('-inf')

    all_metrics.sort(key=lambda x: x.get('momentum_score', float('-inf')), reverse=True)  # 按动量得分排序

    final_list = apply_filters(all_metrics)  # 应用过滤条件
    final_list.sort(key=lambda x: x.get('momentum_score', float('-inf')), reverse=True)
    
    # 简化的评分日志输出
    if final_list:
        top_names = [f"{m['etf_name']}({m['etf']}):{m['momentum_score']:.4f}" for m in final_list[:5]]
        log.info(f"符合条件的候选ETF(前5): {top_names}")
    else:
        log.info("今日无符合条件的候选ETF")

    return final_list

# --- 止损风控 ---
def minute_level_stop_loss(context):    # 分钟级固定比例止损
    if not Config.USE_FIXED_STOP_LOSS:
        log.info("[DEBUG-固定止损] 未启用，return")
        return
    if is_in_cooldown(context):
        log.info(f"[DEBUG-固定止损] 处于冷却期至{g.cooldown_end_date}，return")
        return

    pos_keys = list(context.portfolio.positions.keys())
    log.info(f"[DEBUG-固定止损] 时刻:{context.current_dt} 持仓:{pos_keys}")
    current_data = get_current_data()
    for security in pos_keys:
        position = context.portfolio.positions[security]
        if position.total_amount <= 0:
            log.info(f"[DEBUG-固定止损] {security} 持仓量=0，skip")
            continue
        if security not in current_data:
            log.info(f"[DEBUG-固定止损] {security} 不在current_data，skip")
            continue
        current_price = current_data[security].last_price
        cost_price = position.avg_cost
        stop_line = cost_price * Config.FIXED_STOP_LOSS_THRESHOLD
        log.info(f"[DEBUG-固定止损] {security} 现价:{current_price:.4f} 成本:{cost_price:.4f} 止损线:{stop_line:.4f} 触发:{current_price <= stop_line}")
        if current_price <= 0 or cost_price <= 0: continue
        if current_price <= stop_line:
            security_name = get_security_name(security)
            log.info(f"🚨 [分钟级] 固定比例止损卖出: {security} {security_name}")
            success = smart_order_target_value(security, 0, context)
            log.info(f"[DEBUG-固定止损] smart_order返回: {success}")
            if success:
                g.position_highs.pop(security, None)
                g.position_stop_prices.pop(security, None)
                enter_safe_haven_and_set_cooldown(context, trigger_reason="分钟级固定止损")

def minute_level_pct_stop_loss(context):  # 分钟级当日跌幅止损
    if not Config.USE_PCT_STOP_LOSS: return
    if is_in_cooldown(context): return

    current_data = get_current_data()
    for security in list(context.portfolio.positions.keys()):
        position = context.portfolio.positions[security]
        if position.total_amount <= 0: continue
        if security not in current_data: continue
        today_open = current_data[security].day_open
        if not today_open or today_open <= 0: continue
        current_price = current_data[security].last_price
        if current_price <= 0: continue
        if current_price <= today_open * Config.PCT_STOP_LOSS_THRESHOLD:
            security_name = get_security_name(security)
            log.info(f"🚨 [分钟级] 当日跌幅止损卖出: {security} {security_name}")
            success = smart_order_target_value(security, 0, context)
            if success:
                g.position_highs.pop(security, None)
                g.position_stop_prices.pop(security, None)
                enter_safe_haven_and_set_cooldown(context, trigger_reason="分钟级当日跌幅止损")

def minute_level_atr_stop_loss(context):  # 分钟级ATR动态止损
    if not Config.USE_ATR_STOP_LOSS: return
    if is_in_cooldown(context): return

    current_data = get_current_data()
    for security in list(context.portfolio.positions.keys()):
        position = context.portfolio.positions[security]
        if position.total_amount <= 0: continue
        if Config.ATR_EXCLUDE_DEFENSIVE and security == Config.DEFENSIVE_ETF: continue
        try:
            if security not in current_data: continue
            current_price = current_data[security].last_price
            if current_price <= 0: continue
            cost_price = position.avg_cost
            if cost_price <= 0: continue
            current_atr, _, success, _ = calculate_atr(security, Config.ATR_PERIOD)
            if not success or current_atr <= 0: continue
            if security not in g.position_highs:
                g.position_highs[security] = current_price
            else:
                g.position_highs[security] = max(g.position_highs[security], current_price)
            if Config.ATR_TRAILING_STOP:
                atr_stop_price = g.position_highs[security] - Config.ATR_MULTIPLIER * current_atr
            else:
                atr_stop_price = cost_price - Config.ATR_MULTIPLIER * current_atr
            g.position_stop_prices[security] = atr_stop_price
            if current_price <= atr_stop_price:
                success = smart_order_target_value(security, 0, context)
                if success:
                    g.position_highs.pop(security, None)
                    g.position_stop_prices.pop(security, None)
                    enter_safe_haven_and_set_cooldown(context, trigger_reason="分钟级ATR动态止损")
        except Exception:
            pass

# --- 交易执行 ---
def smart_order_target_value(security, target_value, context):  # 智能下单函数
    current_data = get_current_data()
    security_name = get_security_name(security)
    if current_data[security].paused: return False
    if current_data[security].last_price >= current_data[security].high_limit: return False
    if current_data[security].last_price <= current_data[security].low_limit: return False
    current_price = current_data[security].last_price
    if current_price <= 0: return False
    
    target_amount = int(target_value / current_price)
    target_amount = (target_amount // 100) * 100
    if target_amount <= 0 and target_value > 0: target_amount = 100
    
    current_position = context.portfolio.positions.get(security, None)
    current_amount = current_position.total_amount if current_position else 0
    amount_diff = target_amount - current_amount
    trade_value = abs(amount_diff) * current_price
    
    if 0 < trade_value < Config.MIN_MONEY: return False
    
    if amount_diff < 0:
        closeable = current_position.closeable_amount if current_position else 0
        if closeable == 0: return False
        amount_diff = -min(abs(amount_diff), closeable)
        
    if amount_diff != 0:
        order_result = order(security, amount_diff)
        if order_result:
            if amount_diff > 0:
                log.info(f"📦 买入 {security_name}({security})，金额: {amount_diff * current_price:.2f}")
                if security in Config.FIXED_ETF_POOL:
                    g.position_highs[security] = current_price
            else:
                log.info(f"📤 卖出 {security_name}({security})，金额: {abs(amount_diff) * current_price:.2f}")
            return True
    return False

def etf_sell_trade(context):            # 卖出交易主逻辑
    if is_in_cooldown(context): return
    ranked_etfs = get_final_ranked_etfs(context)
    target_etfs = [m['etf'] for m in ranked_etfs[:Config.HOLDINGS_NUM]] if ranked_etfs else []
    
    if not target_etfs:
        if check_defensive_etf_available(context):
            target_etfs = [Config.DEFENSIVE_ETF]
            log.info(f"🛡️ 防御模式开启: {Config.DEFENSIVE_ETF}")
    
    g.target_etfs_list = target_etfs
    target_set = set(target_etfs)
    
    for security in list(context.portfolio.positions.keys()):
        if security not in target_set:
            smart_order_target_value(security, 0, context)

def etf_buy_trade(context):             # 买入交易主逻辑
    exit_safe_haven_if_cooldown_ends(context)
    if is_in_cooldown(context): return
    
    target_etfs = g.target_etfs_list
    if not target_etfs: return
    
    etfs_to_buy = [e for e in target_etfs if e not in context.portfolio.positions or context.portfolio.positions[e].total_amount == 0]
    if not etfs_to_buy: return
    
    available_cash = context.portfolio.available_cash
    per_value = available_cash // len(etfs_to_buy)
    
    for etf in etfs_to_buy:
        smart_order_target_value(etf, per_value, context)

# --- 工具函数与底层支撑 ---
def check_positions(context):
    for security in context.portfolio.positions:
        pos = context.portfolio.positions[security]
        if pos.total_amount > 0:
            log.info(f"📊 持仓检查: {get_security_name(security)}({security}), 数量: {pos.total_amount}")

def get_security_name(security):
    try: return get_current_data()[security].name
    except: return security

def check_defensive_etf_available(context):
    d = Config.DEFENSIVE_ETF
    info = get_current_data()[d]
    return not info.paused and info.last_price < info.high_limit and info.last_price > info.low_limit

def get_volume_ratio(context, security, lookback_days=None, threshold=None, show_detail_log=True):
    lookback_days = lookback_days or Config.VOLUME_LOOKBACK
    try:
        hist = attribute_history(security, lookback_days, '1d', ['volume'])
        if len(hist) < lookback_days: return None
        avg_v = hist['volume'].mean()
        if avg_v == 0: return None
        cur_v = get_price(security, start_date=context.current_dt.date(), end_date=context.current_dt, frequency='1m', fields=['volume'])['volume'].sum()
        return cur_v / avg_v
    except: return None

def calculate_rsi(prices, period=6):
    if len(prices) < period + 1: return np.array([])
    deltas = np.diff(prices)
    alpha = 2.0 / (period + 1)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g = np.zeros(len(deltas)); avg_l = np.zeros(len(deltas))
    avg_g[period-1] = np.mean(gains[:period]); avg_l[period-1] = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_g[i] = gains[i] * alpha + avg_g[i-1] * (1-alpha)
        avg_l[i] = losses[i] * alpha + avg_l[i-1] * (1-alpha)
    rsi = 100 - (100 / (1 + avg_g / (avg_l + 1e-9)))
    return np.append(np.full(period, np.nan), rsi)

def calculate_atr(security, period=14):
    try:
        hist = attribute_history(security, period + 5, '1d', ['high', 'low', 'close'])
        if len(hist) < period + 1: return 0, [], False, "数据不足"
        h, l, c = hist['high'].values, hist['low'].values, hist['close'].values
        tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
        atr = np.mean(tr[-period:])
        return atr, tr, True, "成功"
    except: return 0, [], False, "失败"

def is_in_cooldown(context):
    return Config.SELL_COOLDOWN_ENABLED and g.cooldown_end_date and context.current_dt.date() <= g.cooldown_end_date

def set_cooldown(context):
    if Config.SELL_COOLDOWN_ENABLED:
        g.cooldown_end_date = context.current_dt.date() + pd.Timedelta(days=Config.SELL_COOLDOWN_DAYS)
        log.info(f"🔒 冷却期开启至: {g.cooldown_end_date}")

def enter_safe_haven_and_set_cooldown(context, trigger_reason=""):
    if not Config.SELL_COOLDOWN_ENABLED: return
    for s in list(context.portfolio.positions.keys()):
        smart_order_target_value(s, 0, context)
        g.position_highs.pop(s, None)
    v = context.portfolio.total_value
    if v > Config.MIN_MONEY: smart_order_target_value(Config.SAFE_HAVEN_ETF, v * 0.99, context)
    set_cooldown(context)

def exit_safe_haven_if_cooldown_ends(context):
    if Config.SELL_COOLDOWN_ENABLED and g.cooldown_end_date and context.current_dt.date() > g.cooldown_end_date:
        if Config.SAFE_HAVEN_ETF in context.portfolio.positions: smart_order_target_value(Config.SAFE_HAVEN_ETF, 0, context)
        g.cooldown_end_date = None
