# 克隆自聚宽文章：https://www.joinquant.com/post/66924
# 标题：七星高照V2.0六年50倍
# 作者：弈剑

# 克隆自聚宽文章：https://www.joinquant.com/post/66676
# 标题：【策略改造】七星高照ETF轮动策略-V1.1
# 作者：屌丝逆袭量化

# 策略名称：七星高照ETF轮动策略-V1.1
# 策略作者：屌丝逆袭量化
# 优化时间：2026-1-31
# 优化内容：
# 1. 增加成交量检测：高位放量则过滤掉
# 2. 在计算动量时进行停牌检查，直接先排除停牌的标的，避免空仓
# 3. 修复防御资产未生效的问题
# 4. 卖出时检查持仓标的是否停牌，如果停牌，则跳过卖出
# 5. 如果发现还持有其他持仓（比如因为交易时间不对，或停牌等原因未卖出），则跳过买入
# 6. 兼容晚1个小时开盘的情况，将交易时间改到14:00左右，并相应的调整成交量过滤函数


# 克隆自聚宽文章：https://www.joinquant.com/post/65654
# 标题：5年30倍回撤18%：ETF动量+布林风控+短线过滤
# 作者：Jean Valjean

import numpy as np
import math
import pandas as pd
from jqdata import *

# ================== 【全局静态常量】==================

ETF_POOL_DEF = [
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
    # "515880.XSHG"

]

# ============== 策略参数默认值（_DEF后缀） ==============

# 动量计算参数
LOOKBACK_DAYS_DEF = 25        # 长期动量计算周期
HOLDINGS_NUM_DEF = 1          # 持仓ETF数量
DEFENSIVE_ETF_DEF = "511880.XSHG"  # 防御性ETF（货币ETF）
MIN_MONEY_DEF = 5000          # 最小交易金额

# 风险控制参数
STOP_LOSS_DEF = 0.95          # 固定百分比止损线（下跌5%止损）
LOSS_DEF = 0.97               # 近3日跌幅止损线

# ATR动态止损参数
USE_ATR_STOP_LOSS_DEF = False # 是否启用ATR动态止损
ATR_PERIOD_DEF = 14           # ATR计算周期
ATR_MULTIPLIER_DEF = 2        # ATR倍数
ATR_TRAILING_STOP_DEF = False # 是否使用跟踪止损
ATR_EXCLUDE_DEFENSIVE_DEF = True # 防御ETF是否豁免ATR止损

# 成交量过滤参数
ENABLE_VOLUME_CHECK_DEF = True # 是否启用成交量过滤
VOLUME_LOOKBACK_DEF = 5       # 成交量历史参考天数
VOLUME_THRESHOLD_DEF = 2.5   # 放量阈值（大于设定值视为放量）
VOLUME_RETURN_LIMIT_DEF = 1   # 年化收益率过滤阈值

# 均线过滤参数
ENABLE_MA_FILTER_DEF = False  # 是否启用均线过滤
MA_FILTER_DAYS_DEF = 20       # 均线过滤天数

# 短期动量过滤参数
USE_SHORT_MOMENTUM_FILTER_DEF = False # 是否启用短期动量过滤
SHORT_LOOKBACK_DAYS_DEF = 10  # 短期动量计算周期
SHORT_MOMENTUM_THRESHOLD_DEF = 0.0 # 短期动量阈值

# RSI过滤参数
USE_RSI_FILTER_DEF = False    # 是否启用RSI过滤
RSI_PERIOD_DEF = 6            # RSI计算周期
RSI_LOOKBACK_DAYS_DEF = 1     # 检查RSI的历史天数
RSI_THRESHOLD_DEF = 98        # RSI阈值

# R²筛选参数
USE_R2_FILTER_DEF = True     # 是否启用R²筛选
R2_MIN_THRESHOLD_DEF = 0.3    # R²最低阈值（0.3≤R²≤1）

# 得分阈值
MIN_SCORE_THRESHOLD_DEF = 0.0 # 最低得分阈值
MAX_SCORE_THRESHOLD_DEF = 5.0 # 最高得分阈值

# =================== 【初始化函数】 =====================
def initialize(context):
    
    # ============== 赋值全局常量到g变量 ==============
    g.etf_pool = ETF_POOL_DEF  # 引用全局ETF池常量
    
    # 设置日志级别
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    log.set_level('strategy', 'debug')
    
    # ================ 聚宽环境初始化 =================
    # 开启「避免未来数据」功能
    set_option("avoid_future_data", True)
    # 开启「使用真实价格」功能
    set_option("use_real_price", True)
    
    # 设置滑点
    set_slippage(PriceRelatedSlippage(0.0001), type="fund")
    
    # 设置交易成本:ETF交易成本较低
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0.0002,
            close_commission=0.0002,
            close_today_commission=0,
            min_commission=5,
        ),
        type="fund",
    )

    # 设置参考基准
    set_benchmark("000300.XSHG")  
    
    # ===== 赋值策略参数到g变量（引用_DEF后缀的全局常量） =====
    # 动量计算参数
    g.lookback_days = LOOKBACK_DAYS_DEF
    g.holdings_num = HOLDINGS_NUM_DEF
    g.defensive_etf = DEFENSIVE_ETF_DEF
    g.min_money = MIN_MONEY_DEF
    
    # 风险控制参数
    g.stop_loss = STOP_LOSS_DEF
    g.loss = LOSS_DEF
    
    # ATR动态止损参数
    g.use_atr_stop_loss = USE_ATR_STOP_LOSS_DEF
    g.atr_period = ATR_PERIOD_DEF
    g.atr_multiplier = ATR_MULTIPLIER_DEF
    g.atr_trailing_stop = ATR_TRAILING_STOP_DEF
    g.atr_exclude_defensive = ATR_EXCLUDE_DEFENSIVE_DEF
    
    # 成交量过滤参数
    g.enable_volume_check = ENABLE_VOLUME_CHECK_DEF
    g.volume_lookback = VOLUME_LOOKBACK_DEF
    g.volume_threshold = VOLUME_THRESHOLD_DEF
    g.volume_return_limit = VOLUME_RETURN_LIMIT_DEF
    
    # 均线过滤参数
    g.enable_ma_filter = ENABLE_MA_FILTER_DEF
    g.ma_filter_days = MA_FILTER_DAYS_DEF
    
    # 短期动量过滤参数
    g.use_short_momentum_filter = USE_SHORT_MOMENTUM_FILTER_DEF
    g.short_lookback_days = SHORT_LOOKBACK_DAYS_DEF
    g.short_momentum_threshold = SHORT_MOMENTUM_THRESHOLD_DEF
    
    # RSI过滤参数
    g.use_rsi_filter = USE_RSI_FILTER_DEF
    g.rsi_period = RSI_PERIOD_DEF
    g.rsi_lookback_days = RSI_LOOKBACK_DAYS_DEF
    g.rsi_threshold = RSI_THRESHOLD_DEF
    
    # R²筛选参数赋值
    g.use_r2_filter = USE_R2_FILTER_DEF
    g.r2_min_threshold = R2_MIN_THRESHOLD_DEF
    
    # 得分阈值
    g.min_score_threshold = MIN_SCORE_THRESHOLD_DEF
    g.max_score_threshold = MAX_SCORE_THRESHOLD_DEF
    
    # ================ 持仓管理 ================
    g.positions = {}  # 记录持仓
    g.position_highs = {}  # 记录持仓期间的最高价
    g.position_stop_prices = {}  # 记录持仓的ATR止损价
    
    # ================ 交易调度 ================
    # 每天开盘后检查持仓
    run_daily(check_positions, time='09:10')
    # 每天开盘后检查ATR动态止损
    run_daily(check_atr_stop_loss, time='10:31')
    # 执行卖出操作
    run_daily(etf_sell_trade, time='14:00')
    # 执行买入操作
    run_daily(etf_buy_trade, time='14:01')
    
    # ================ 打印初始化信息 ================
    log.info(f"""策略参数初始化完成:
    - ETF池大小: {len(g.etf_pool)} 只ETF | 动量周期: {g.lookback_days} 天 | 持仓数量: {g.holdings_num} 只 | 防御ETF: {g.defensive_etf}
    - 成交量过滤: {'启用' if g.enable_volume_check else '禁用'} | 均线过滤: {'启用' if g.enable_ma_filter else '禁用'} | RSI过滤: {'启用' if g.use_rsi_filter else '禁用'} | ATR止损: {'启用' if g.use_atr_stop_loss else '禁用'}
""")

# ============ 持仓检查 ===============
def check_positions(context):
    """每日开盘后检查持仓状态"""
    current_data = get_current_data()
    for security in context.portfolio.positions:
        position = context.portfolio.positions[security]
        if position.total_amount > 0:
            security_name = get_security_name(security)
            log.info(f"📊 持仓检查: {security} {security_name}, 数量: {position.total_amount}, 成本: {position.avg_cost:.3f}, 当前价: {position.price:.3f}")
            if current_data[security].paused:
                log.info(f"⚠️ {security} {security_name} 今日停牌")

# ==================== 卖出函数 ====================
def etf_sell_trade(context):
    """
    卖出函数
    功能：卖出不符合条件的持仓（优先执行固定止损，再卖出非目标持仓）
    """
    log.info("============== 卖出操作开始 ==============")
    
    # 获取当前持仓
    current_positions = list(context.portfolio.positions.keys())
    
    # 如果没有持仓，直接返回
    if not current_positions:
        log.info("当前无持仓，无需卖出")
        log.info("============== 卖出操作完成 ==============")
        return
    
    # 获取符合条件的ETF排名
    ranked_etfs = get_ranked_etfs(context)
    
    # 确定目标ETF
    target_etf = None
    if ranked_etfs and ranked_etfs[0]['score'] >= g.min_score_threshold:
        target_etf = ranked_etfs[0]['etf']
        log.info(f"📌 选中进攻型目标ETF：{target_etf} {get_security_name(target_etf)}")
    else:
        log.info("⚠️ 无符合条件的进攻型ETF，检查防御ETF是否可用")
    
    # 检查防御ETF是否可用
    defensive_etf_available = False
    if target_etf is None:
        defensive_etf_available = check_defensive_etf_available(context)
        if defensive_etf_available:
            target_etf = g.defensive_etf
            log.info(f"📌 切换到防御ETF：{target_etf} {get_security_name(target_etf)}")
        else:
            log.info("⚠️ 防御ETF不可用，本次无目标ETF")
    
    # 构建目标ETF列表
    target_etfs = [target_etf] if target_etf else []
    target_etfs_set = set(target_etfs)
    
    # ============ 检查并执行固定止损 ============
    for security in list(context.portfolio.positions.keys()):
        if security in g.etf_pool:
            position = context.portfolio.positions[security]
            if position.total_amount > 0:
                # 提前定义标的名称，复用
                security_name = get_security_name(security)
                current_price = position.price
                cost_price = position.avg_cost
                
                # 成本价防护：避免除以0/数据异常
                if cost_price > 0 and current_price <= cost_price * g.stop_loss:
                    success = smart_order_target_value(security, 0, context)
                    loss_percent = (current_price/cost_price - 1) * 100
                    
                    if success:
                        log.info(f"🚨 固定百分比止损卖出: {security} {security_name}，亏损: {loss_percent:.2f}%")
                        # 清除记录
                        g.position_highs.pop(security, None)
                        g.position_stop_prices.pop(security, None)
                    else:
                        log.warning(
                            f"❌ 固定止损失败：{security} {security_name}，"
                            f"当前价{current_price:.3f}≤成本价{cost_price:.3f}×{g.stop_loss}={cost_price * g.stop_loss:.3f}，"
                            f"亏损{loss_percent:.2f}%，但无法卖出！"
                        )    
    
    # ============== 卖出不在目标列表中的持仓 ==============
    # 重新获取持仓（避免止损操作后数据不一致）
    latest_positions = list(context.portfolio.positions.keys())
    for security in latest_positions:
        # 只处理策略关注的标的（ETF池 + 防御ETF）
        if (security in g.etf_pool or security == g.defensive_etf)  and security not in target_etfs_set:
            position = context.portfolio.positions[security]
            if position.total_amount > 0:
                # 提前定义标的名称，复用
                security_name = get_security_name(security)
                success = smart_order_target_value(security, 0, context)
                if success:
                    log.info(f"📤 卖出不在目标列表的持仓: {security} {security_name}")
                    # 清除ATR跟踪记录（仅卖出成功时执行）
                    g.position_highs.pop(security, None)       # 有则删，无则不报错
                    g.position_stop_prices.pop(security, None)
                else:
                    log.warning(f"❌ 卖出失败：{security} {security_name}，非目标持仓未清仓")
                   
    log.info("============== 卖出操作完成 ==============")

# ==================== 获取ETF排名函数 ====================
def get_ranked_etfs(context):
    """
    获取符合条件的ETF排名
    返回结果：应用所有过滤条件，返回满足条件的ETF列表，按得分降序
    """
    etf_metrics = []
    
    # 可选：先进行均线过滤（减少计算量）
    filtered_pool = g.etf_pool
    
    current_data = get_current_data()
    for etf in filtered_pool:
        # ========== 新增：停牌过滤 ==========
        if current_data[etf].paused:
            log.debug(f"{etf}: 今日停牌，跳过计算")
            continue

        metrics = calculate_momentum_metrics(context, etf)
        if metrics is not None:
            # 过滤掉得分异常的ETF
            if 0 < metrics['score'] < g.max_score_threshold:
            #if 0 < metrics['score']:
                etf_metrics.append(metrics)
            else: 
                log.info(f"⚠️ {etf} 得分不满足要求！")
                
    # 按得分降序排序
    etf_metrics.sort(key=lambda x: x['score'], reverse=True)
    return etf_metrics

# ==================== 动量指标计算函数 ====================
def calculate_momentum_metrics(context, etf):
    """
    计算ETF的动量指标，整合所有过滤条件
    返回包含各项指标和过滤结果的字典
    """
    try:
        # 获取历史价格数据加20天缓冲，避免数据切片/缺失导致计算不足
        lookback = max(g.lookback_days, g.short_lookback_days, 
                      g.rsi_period + g.rsi_lookback_days) + 20
        prices = attribute_history(etf, lookback, '1d', ['close', 'high'])
        current_data = get_current_data()
        
        if prices.empty or len(prices) < g.lookback_days:
            log.debug(f"{etf}: 历史数据为空或数据不足（仅{len(prices)}天），跳过计算")
            return None
        
        # 获取当前价格并添加到价格序列中
        current_price = current_data[etf].last_price
        if current_price <= 0:
            log.debug(f"{etf}: 实时价格异常（{current_price}），跳过计算")
            return None
        price_series = np.append(prices["close"].values, current_price)
       
        # ========== 成交量过滤检查 ==========
        if g.enable_volume_check and len(price_series) > g.lookback_days:
            volume_ratio = get_volume_ratio(context, etf)
            volume_annualized = get_annualized_returns(price_series,g.lookback_days)
            if volume_ratio is not None:
                if volume_annualized > g.volume_return_limit:
                    log.debug(f"{etf}: 成交量放大{volume_ratio:.2f}倍且折合年化收益{volume_annualized:.2f}超过设置值{g.volume_return_limit}，属于“高位放量”，过滤掉")
                    return None
        
        # ========== RSI过滤检查 ==========
        rsi_filter_pass = True
        current_rsi = 0
        max_rsi = 0
        
        if g.use_rsi_filter and len(price_series) >= g.rsi_period + g.rsi_lookback_days:
            rsi_values = calculate_rsi(price_series, g.rsi_period)
            
            if len(rsi_values) >= g.rsi_lookback_days:
                recent_rsi = rsi_values[-g.rsi_lookback_days:]
                rsi_ever_above_threshold = np.any(recent_rsi > g.rsi_threshold)
                
                # 检查当前价格是否在MA5之下
                if len(price_series) >= 5:
                    ma5 = np.mean(price_series[-5:])
                    current_below_ma5 = current_price < ma5
                else:
                    current_below_ma5 = True
                
                if rsi_ever_above_threshold and current_below_ma5:
                    rsi_filter_pass = False
                    max_rsi = np.max(recent_rsi)
                    current_rsi = recent_rsi[-1] if len(recent_rsi) > 0 else 0
                    log.info(f"⛔ RSI过滤: {etf} 近{g.rsi_lookback_days}日RSI曾达{max_rsi:.1f}，当前价{current_price:.3f}<MA5，当前RSI={current_rsi:.1f}")
                else:
                    max_rsi = np.max(recent_rsi) if len(recent_rsi) > 0 else 0
                    current_rsi = recent_rsi[-1] if len(recent_rsi) > 0 else 0
        
        if not rsi_filter_pass:
            return None
        
        # ========== 短期动量计算 ==========
        if len(price_series) >= g.short_lookback_days + 1:
            short_return = price_series[-1] / price_series[-(g.short_lookback_days + 1)] - 1
            short_annualized = (1 + short_return) ** (250 / g.short_lookback_days) - 1
            #short_annualized = get_annualized_returns(price_series,g.short_lookback_days)
        else:
            short_return = 0
            short_annualized = 0
        
        # ========== 短期动量过滤 ==========
        if g.use_short_momentum_filter and short_annualized < g.short_momentum_threshold:
            log.debug(f"{etf}: 短期动量{short_annualized:.4f} < 阈值{g.short_momentum_threshold}，过滤掉")
            return None
        
        # ========== 长期动量计算 ==========
        # 使用最后g.lookback_days+1天的数据
        recent_price_series = price_series[-(g.lookback_days + 1):]
        y = np.log(recent_price_series)
        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))  # 加权回归，近期权重更高
        
        # ==========计算年化收益率==========
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        annualized_returns = math.exp(slope * 250) - 1
        #annualized_returns = get_annualized_returns(price_series,g.lookback_days)
        
        # ==========计算R²（拟合优度）==========
        ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot else 0
        
        # ========== R²过滤检查==========
        if g.use_r2_filter:
            if not (g.r2_min_threshold <= r_squared <= 1):
                log.debug(f"{etf}: R²={r_squared:.4f} 不在[{g.r2_min_threshold}, 1]范围内，过滤掉")
                return None
        
        # 综合得分 = 年化收益率 * 趋势稳定性
        score = annualized_returns * r_squared
        #score = annualized_returns * (r_squared ** 2)
        #score = annualized_returns * (r_squared + 0.1)
        
        # ========== 短期风控过滤 ==========
        if len(price_series) >= 4:
            day1_ratio = price_series[-1] / price_series[-2]
            day2_ratio = price_series[-2] / price_series[-3]
            day3_ratio = price_series[-3] / price_series[-4]
            
            if min(day1_ratio, day2_ratio, day3_ratio) < g.loss:
                score = 0
                log.info(f"⚠️ {etf} 近3日有单日跌幅超设定值，已排除")
        
        return {
            'etf': etf,
            'annualized_returns': annualized_returns,
            'r_squared': r_squared,
            'score': score,
            'slope': slope,
            'current_price': current_price,
            'short_return': short_return,
            'short_annualized': short_annualized,
            'short_momentum_pass': short_return >= g.short_momentum_threshold,
            'rsi_filter_pass': rsi_filter_pass,
            'current_rsi': current_rsi,
            'max_recent_rsi': max_rsi,
        }
        
    except Exception as e:
        log.warning(f"计算{etf}动量指标时出错: {e}")
        return None
   

# ==================== 新增：成交量过滤函数（参考策略1） ====================
def get_volume_ratio(context, security, lookback_days=None, threshold=None):
    """
    计算成交量比值（当日成交量/历史平均成交量）
    返回：若放量（>threshold）则返回比值，否则返回None
    """
    if lookback_days is None:
        lookback_days = g.volume_lookback
    if threshold is None:
        threshold = g.volume_threshold
    
    try:
        # 1. 获取历史成交量（N天平均）
        hist_data = attribute_history(security, lookback_days, '1d', ['volume'])
        if hist_data.empty or len(hist_data) < lookback_days:
            log.debug(f"{security}: 历史成交量数据不足")
            return None
        
        avg_volume = hist_data['volume'].mean()
        
        # 2. 获取当日实时成交量（分钟数据累加）
        today = context.current_dt.date()
        df_vol = get_price(
            security,
            start_date=today,
            end_date=context.current_dt,
            frequency='1m',
            fields=['volume'],
            skip_paused=False,
            fq='pre',
            panel=True,
            fill_paused=False
        )
        
        if df_vol is None or df_vol.empty:
            log.debug(f"{security}: 当日成交量数据为空")
            return None
        
        current_volume = df_vol['volume'].sum()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # 3. 超过阈值视为放量
        etf_name = get_security_name(security)
        if volume_ratio > threshold:
            log.debug(f"⚠️ {security}-{etf_name}: 成交量比值 {volume_ratio:.2f} > 阈值 {threshold}")
            return volume_ratio
        else:
            log.debug(f"{security}-{etf_name}: 成交量比值 {volume_ratio:.2f} <= 阈值 {threshold}")
            return None
            
    except Exception as e:
        log.warning(f"成交量检测失败 {security}: {e}")
        return None

# ==================== 新增：均线过滤函数（参考策略1） ====================
def filter_below_ma(stocks, days=None):
    """
    过滤掉当前价格小于N日均价的股票/ETF
    返回过滤后的标的列表（仅保留当前价 >= N日均价的标的）
    """
    if days is None:
        days = g.ma_filter_days
    
    if not stocks:
        return []
    
    current_data = get_current_data()
    filtered = []
    
    for stock in stocks:
        try:
            # 获取N日历史收盘价数据
            hist = attribute_history(stock, days, "1d", ["close"])
            if len(hist) < days:
                log.debug(f"{stock}: 历史数据不足{days}天，跳过过滤")
                continue
                
            # 计算N日均价
            ma_n = hist["close"].mean()
            # 获取当前价格
            current_price = current_data[stock].last_price
            
            # 保留当前价 >= N日均价的标的
            if current_price >= ma_n:
                filtered.append(stock)
                log.debug(f"{stock}: 通过{days}日均线过滤，当前价 {current_price:.2f} >= 均线 {ma_n:.2f}")
            else:
                log.debug(f"{stock}: 未通过{days}日均线过滤，当前价 {current_price:.2f} < 均线 {ma_n:.2f}")
                
        except Exception as e:
            log.warning(f"计算{stock} {days}日均价失败: {e}")
            continue
            
    return filtered

# ==================== 原有：ATR计算函数（保持不变） ====================
def calculate_atr(security, period=14):
    """
    计算ATR（平均真实波幅）指标
    """
    try:
        needed_days = period + 20
        hist_data = attribute_history(security, needed_days, '1d', 
                                     ['high', 'low', 'close'])
        
        if len(hist_data) < period + 1:
            return 0, [], False, f"数据不足{period+1}天"
        
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
        
        return current_atr, valid_atr, True, "计算成功"
    
    except Exception as e:
        log.warning(f"计算{security} ATR时出错: {e}")
        return 0, [], False, f"计算出错:{str(e)}"

# ==================== 原有：RSI计算函数（保持不变） ====================
def calculate_rsi(prices, period=6):
    """
    计算RSI指标
    """
    if len(prices) < period + 1:
        return []
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = np.zeros_like(prices)
    avg_losses = np.zeros_like(prices)
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])
    
    rsi_values = np.zeros(len(prices))
    rsi_values[:period] = 50
    
    for i in range(period + 1, len(prices)):
        avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
        avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        if avg_losses[i] == 0:
            rsi_values[i] = 100
        else:
            rs = avg_gains[i] / avg_losses[i]
            rsi_values[i] = 100 - (100 / (1 + rs))
    
    return rsi_values[period:]

     
# ===================计算年化收益===================
def get_annualized_returns(price_series,lookback_days):
    # 使用最后g.lookback_days+1天的数据
    recent_price_series = price_series[-(lookback_days + 1):]
    y = np.log(recent_price_series)
    x = np.arange(len(y))
    weights = np.linspace(1, 2, len(y))  # 加权回归，近期权重更高
    
    # 计算年化收益率
    slope, intercept = np.polyfit(x, y, 1, w=weights)
    annualized_returns = math.exp(slope * 250) - 1
    return annualized_returns




# ==================== 优化：买入函数（下午14:20执行） ====================
def etf_buy_trade(context):
    """
    买入函数
    功能：买入符合条件的ETF
    """
    log.info("========== 买入操作开始 ==========")
    
    # 获取符合条件的ETF排名
    ranked_etfs = get_ranked_etfs(context)
    
    # 记录所有ETF的指标（用于调试）
    if ranked_etfs:
        log.info("=== 符合条件的ETF指标 ===")
        for metrics in ranked_etfs[:5]:  # 只显示前5名
            etf_name = get_security_name(metrics['etf'])
            log.info(f"{metrics['etf']} {etf_name}: 得分={metrics['score']:.4f}, 年化={metrics['annualized_returns']:.4f}, R²={metrics['r_squared']:.4f}, 短期动量={metrics['short_return']:.4f}, RSI={metrics['current_rsi']:.1f}")
    
    # 确定目标ETF
    target_etf = None
    if ranked_etfs and ranked_etfs[0]['score'] >= g.min_score_threshold:
        target_etf = ranked_etfs[0]['etf']
        top_metrics = ranked_etfs[0]
        etf_name = get_security_name(target_etf)
        log.info(f"🎯 选择得分最高的ETF: {target_etf} {etf_name}，得分: {top_metrics['score']:.4f}")
    else:
        # 防御模式
        if check_defensive_etf_available(context):
            target_etf = g.defensive_etf
            etf_name = get_security_name(target_etf)
            log.info(f"🛡️ 进入防御模式，选择防御ETF: {target_etf} {etf_name}")
        else:
            log.info("💤 进入空仓模式，无符合条件的ETF且防御ETF不可用")
    
    # 如果没有目标ETF，直接返回
    if target_etf is None:
        log.info("无目标ETF，保持空仓")
        return
    
    # 如果已有其他持仓，先检查是否已经卖出
    current_positions = list(context.portfolio.positions.keys())
    current_etf_positions = [pos for pos in current_positions if pos in g.etf_pool or pos == g.defensive_etf]
    other_positions = [pos for pos in current_etf_positions if pos != target_etf]
    if other_positions and target_etf not in current_etf_positions:
        # 检查这些持仓是否正在卖出过程中
        for pos in other_positions:
            position = context.portfolio.positions[pos]
            if position.total_amount > 0:
                log.info(f"⚠️ 尚有其他持仓 {get_security_name(pos)} 未卖出，等待卖出完成后再买入新标的")
                return
    
    # 计算目标市值
    total_value = context.portfolio.total_value
    target_value = total_value
    
    # 调整目标ETF的仓位
    # 获取当前持仓价值
    current_value = 0
    if target_etf in context.portfolio.positions:
        position = context.portfolio.positions[target_etf]
        if position.total_amount > 0:
            current_value = position.total_amount * position.price
    
    # 判断是否需要调仓（5%容差）
    if abs(current_value - target_value) > target_value * 0.05 or current_value == 0:
        success = smart_order_target_value(target_etf, target_value, context)
        if success:
            etf_name = get_security_name(target_etf)
            action = "买入" if current_value < target_value else "调仓"
            log.info(f"📦 {action}: {target_etf} {etf_name}，目标金额: {target_value:.2f}")
    
    log.info("========== 买入操作完成 ==========")

# ==================== 原有辅助函数（保持不变） ====================
def get_security_name(security):
    """获取证券名称"""
    current_data = get_current_data()
    #return current_data[security].name if security in current_data else security
    return current_data[security].name

def check_defensive_etf_available(context):
    """检查防御ETF是否可交易"""
    current_data = get_current_data()
    defensive_etf = g.defensive_etf
    
    #if defensive_etf not in g.etf_pool:
    #    return False
        
    if current_data[defensive_etf].paused:
        log.info(f"防御性ETF {defensive_etf} 今日停牌")
        return False
        
    if current_data[defensive_etf].last_price >= current_data[defensive_etf].high_limit:
        log.info(f"防御性ETF {defensive_etf} 当前涨停")
        return False
        
    if current_data[defensive_etf].last_price <= current_data[defensive_etf].low_limit:
        log.info(f"防御性ETF {defensive_etf} 当前跌停")
        return False
        
    return True

def smart_order_target_value(security, target_value, context):
    """
    智能下单函数
    """
    current_data = get_current_data()
    
    # 检查标的是否停牌
    if current_data[security].paused:
        log.info(f"{security} {get_security_name(security)}: 今日停牌，跳过交易")
        return False

    # 检查涨停
    if current_data[security].last_price >= current_data[security].high_limit:
        log.info(f"{security} {get_security_name(security)}: 当前涨停，跳过买入")
        return False

    # 检查跌停
    if current_data[security].last_price <= current_data[security].low_limit:
        log.info(f"{security} {get_security_name(security)}: 当前跌停，跳过卖出")
        return False

    # 获取当前价格
    current_price = current_data[security].last_price
    if current_price == 0:
        log.info(f"{security} {get_security_name(security)}: 当前价格为0，跳过交易")
        return False

    # 计算目标数量
    target_amount = int(target_value / current_price)
    
    # 对于ETF，按100股整数倍调整
    target_amount = (target_amount // 100) * 100
    if target_amount <= 0 and target_value > 0:
        target_amount = 100
    
    # 获取当前持仓
    current_position = context.portfolio.positions.get(security, None)
    current_amount = current_position.total_amount if current_position else 0
    
    # 计算需要调整的数量
    amount_diff = target_amount - current_amount
    
    # 检查最小交易金额
    trade_value = abs(amount_diff) * current_price
    if 0 < trade_value < g.min_money:
        log.info(f"{security} {get_security_name(security)}: 交易金额{trade_value:.2f}小于最小交易额{g.min_money}，跳过交易")
        return False

    # 检查T+1限制
    if amount_diff < 0:  # 卖出操作
        closeable_amount = current_position.closeable_amount if current_position else 0
        if closeable_amount == 0:
            log.info(f"{security} {get_security_name(security)}: 当天买入不可卖出(T+1)")
            return False
        amount_diff = -min(abs(amount_diff), closeable_amount)

    # 执行下单
    if amount_diff != 0:
        order_result = order(security, amount_diff)
        if order_result:
            # 更新持仓记录
            g.positions[security] = target_amount
            
            # 如果买入操作，初始化最高价记录和ATR止损价
            if amount_diff > 0 and security in g.etf_pool:
                g.position_highs[security] = current_price
                
                # 计算ATR止损价
                if g.use_atr_stop_loss and not (g.atr_exclude_defensive and security == g.defensive_etf):
                    current_atr, _, success, _ = calculate_atr(security, g.atr_period)
                    if success:
                        if g.atr_trailing_stop:
                            g.position_stop_prices[security] = current_price - g.atr_multiplier * current_atr
                        else:
                            g.position_stop_prices[security] = current_price - g.atr_multiplier * current_atr
            
            security_name = get_security_name(security)
            if amount_diff > 0:
                log.info(f"📥 买入 {security} {security_name}，数量: {amount_diff}，价格: {current_price:.3f}")
            else:
                log.info(f"📤 卖出 {security} {security_name}，数量: {abs(amount_diff)}，价格: {current_price:.3f}")
            return True
        else:
            log.warning(f"下单失败: {security} {get_security_name(security)}，数量: {amount_diff}")
            return False
    
    return False


        
def check_atr_stop_loss(context):
    """
    检查并执行ATR动态止损
    """
    if not g.use_atr_stop_loss:
        return
    
    current_data = get_current_data()
    
    for security in list(context.portfolio.positions.keys()):
        if security not in g.etf_pool:
            continue
            
        position = context.portfolio.positions[security]
        if position.total_amount <= 0:
            continue
        
        # 防御ETF豁免检查
        if g.atr_exclude_defensive and security == g.defensive_etf:
            continue
        
        try:
            current_price = current_data[security].last_price
            if current_price == 0:
                continue
            
            cost_price = position.avg_cost
            
            # 计算当前ATR值
            current_atr, atr_values, success, atr_info = calculate_atr(security, g.atr_period)
            
            if not success:
                continue
            
            # 更新持仓期间的最高价
            if security not in g.position_highs:
                g.position_highs[security] = current_price
            else:
                g.position_highs[security] = max(g.position_highs[security], current_price)
            
            position_high = g.position_highs[security]
            
            # 计算ATR止损价
            if g.atr_trailing_stop:
                atr_stop_price = position_high - g.atr_multiplier * current_atr
            else:
                atr_stop_price = cost_price - g.atr_multiplier * current_atr
            
            g.position_stop_prices[security] = atr_stop_price
            
            # 检查是否触发ATR止损
            if current_price <= atr_stop_price:
                success = smart_order_target_value(security, 0, context)
                if success:
                    security_name = get_security_name(security)
                    loss_percent = (current_price/cost_price - 1) * 100
                    atr_stop_type = "跟踪" if g.atr_trailing_stop else "固定"
                    log.info(f"🚨 ATR动态止损({atr_stop_type})卖出: {security} {security_name}，亏损: {loss_percent:.2f}%")
                    
                    # 清除记录
                    if security in g.position_highs:
                        del g.position_highs[security]
                    if security in g.position_stop_prices:
                        del g.position_stop_prices[security]
        
        except Exception as e:
            log.warning(f"检查{security} ATR止损时出错: {e}")

# ==================== 主交易函数（保持兼容性） ====================
def trade(context):
    """主交易函数，为了兼容性保留"""
    # 在原有策略二中，trade函数调用了etf_trade
    # 现在我们已经拆分为两个函数，这里可以保持为空或调用买入函数
    pass