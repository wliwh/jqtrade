import numpy as np
import math
import pandas as pd
from jqdata import *
from datetime import datetime, date

# ==========================================
# 策略配置类
# ==========================================
class Config:
    # --- 基础配置 ---
    HOLDINGS_NUM = 1                # 持仓数量
    DEFENSIVE_ETF = '511880.XSHG'   # 防御型ETF (银华日利)
    SAFE_HAVEN_ETF = '511660.XSHG'  # 冷却期避险ETF (建信添益)
    MIN_MONEY = 5000                # 最小交易金额
    
    # --- 选股池相关 ---
    FIXED_ETF_POOL = [
        '518880.XSHG', '161226.XSHE', '159980.XSHE', '501018.XSHG', '159985.XSHE', # 大宗
        '513100.XSHG', '159509.XSHE', '513290.XSHG', '513500.XSHG', '159518.XSHE', # 海外
        '159502.XSHE', '159529.XSHE', '513400.XSHG', '520830.XSHG', '513520.XSHG', 
        '513030.XSHG', 
        '513090.XSHG', '513180.XSHG', '513120.XSHG', '513330.XSHG', '513750.XSHG', # 港股
        '159892.XSHE', '159605.XSHE', '513190.XSHG', '510900.XSHG', '513630.XSHG',
        '513920.XSHG', '159323.XSHE', '513970.XSHG',
        '510500.XSHG', '512100.XSHG', '563300.XSHG', '510300.XSHG', '512050.XSHG', # 指数
        '510760.XSHG', '159915.XSHE', '159949.XSHE', '159967.XSHE', '588080.XSHG',
        '588220.XSHG', '511380.XSHG',
        '513310.XSHG', '588200.XSHG', '159852.XSHE', '512880.XSHG', '159206.XSHE', # 行业
        '512400.XSHG', '512980.XSHG', '159516.XSHE', '512480.XSHG', '515880.XSHG',
        '562500.XSHG', '159218.XSHE', '159869.XSHE', '159870.XSHE', '159326.XSHE',
        '159851.XSHE', '560860.XSHG', '159363.XSHE', '588170.XSHG', '159755.XSHE',
        '512170.XSHG', '512800.XSHG', '159819.XSHE', '512710.XSHG', '159638.XSHE',
        '517520.XSHG', '515980.XSHG', '159995.XSHE', '159227.XSHE', '512660.XSHG',
        '512690.XSHG', '516150.XSHG', '512890.XSHG', '588790.XSHG', '159992.XSHE',
        '512070.XSHG', '562800.XSHG', '512010.XSHG', '515790.XSHG', '510880.XSHG',
        '159928.XSHE', '159883.XSHE', '159998.XSHE', '515220.XSHG', '561980.XSHG',
        '515400.XSHG', '515120.XSHG', '159566.XSHE', '515050.XSHG', '516510.XSHG',
        '159256.XSHE', '159766.XSHE', '512200.XSHG', '513350.XSHG', '159583.XSHE',
        '159732.XSHE', '516160.XSHG', '516520.XSHG', '562590.XSHG', '515030.XSHG',
        '512670.XSHG', '561330.XSHG', '516190.XSHG', '159840.XSHE', '159611.XSHE',
        '159981.XSHE', '159865.XSHE', '561360.XSHG', '159667.XSHE', '515170.XSHG',
        '513360.XSHG', '159825.XSHE', '515210.XSHG'
    ]
    DYNAMIC_ETF_POOL = []           # 动态更新池
    
    # --- 过滤参数 ---
    LOOKBACK_DAYS = 25              # 动量回看
    MIN_SCORE = 0                   # 动量下限
    MAX_SCORE = 5                   # 动量上限
    
    USE_SHORT_MOM = False           # 短期动量开关
    SHORT_LOOKBACK = 10
    SHORT_MOM_THRESHOLD = 0.0
    
    ENABLE_R2_FILTER = True         # R2 过滤
    R2_THRESHOLD = 0.4
    
    ENABLE_ANN_RET_FILTER = False   # 年化收益过滤
    MIN_ANN_RET = 1.0
    
    ENABLE_MA_FILTER = False        # 均线过滤
    MA_DAYS = 20
    
    ENABLE_VOLUME_CHECK = True      # 成交量过滤
    VOLUME_LOOKBACK = 5
    VOLUME_THRESHOLD = 1.0
    
    ENABLE_LOSS_FILTER = True       # 短期跌幅风控
    LOSS_THRESHOLD = 0.97           # 3%
    
    USE_RSI_FILTER = False          # RSI 过滤
    RSI_PERIOD = 6
    RSI_LOOKBACK = 1
    RSI_THRESHOLD = 98
    
    # --- 止损参数 ---
    USE_FIXED_STOP = True           # 固定止损
    FIXED_STOP_THRESHOLD = 0.95
    
    USE_PCT_STOP = False            # 当日跌幅止损
    PCT_STOP_THRESHOLD = 0.95
    
    USE_ATR_STOP = False            # ATR 止损
    ATR_PERIOD = 14
    ATR_MULTIPLIER = 2
    ATR_TRAILING = True
    ATR_EXCLUDE_DEFENSIVE = True
    
    RISK_CHECK_INTERVAL = 5         # 分钟级风控检查间隔 (默认5分钟一次，可改为1)
    
    # --- 冷却期 ---
    SELL_COOLDOWN_ENABLED = False
    SELL_COOLDOWN_DAYS = 3
    COOLDOWN_END_DATE = None
    
    # --- 运行时状态 (替代原 g 对象) ---
    positions_history = {}          # 目标持仓
    position_highs = {}             # 高点记录 (ATR专用)
    position_stop_prices = {}       # 止损价记录
    target_etfs = []                # 今日计划持仓

# ==========================================
# 初始化与调度
# ==========================================
def initialize(context):
    set_option("avoid_future_data", True)
    set_option("use_real_price", True)
    set_slippage(PriceRelatedSlippage(0.0001), type="fund")
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0.0001, close_commission=0.0001, min_commission=5), type="fund")
    
    log.set_level('order', 'error')
    log.set_level('strategy', 'info')
    set_benchmark("510300.XSHG")
    
    # 定时任务调度
    run_daily(update_sector_pool, time='09:00')      # 盘前：刷选动态池
    run_daily(check_positions_before, time='09:10')  # 盘前：数据同步
    run_daily(etf_sell_trade, time='13:10')          # 盘中：轮动卖出
    run_daily(etf_buy_trade, time='13:11')           # 盘中：轮动买入

def handle_data(context, data):
    """聚宽分钟级核心回调，执行日内风控"""
    # 按设定的间隔执行检查，默认 5 分钟一次节省开销
    if context.current_dt.minute % Config.RISK_CHECK_INTERVAL == 0:
        every_minute_check(context)

# ==========================================
# 核心逻辑：池子管理
# ==========================================
def update_sector_pool(context):
    """
    同步原版 7star 的 update_sector_pool 逻辑：
    1. 扫描全市场 ETF
    2. 过滤关键词
    3. 按昨日成交额排序，并进行行业去重（取前2位字符）
    """
    all_etfs = get_all_securities(['etf']).index.tolist()
    exclude_keywords = ['300', '500', '1000', '50', '上证', '创业板', '科创', '恒生', 'H股', '货币', '纳指', '标普', '债']
    
    sector_etfs = []
    for code in all_etfs:
        name = get_security_info(code).display_name
        if not any(k in name for k in exclude_keywords):
            sector_etfs.append(code)
            
    # 动态池更新（采用循环获取，规避平台大批量 get_price 的内部组合错误）
    try:
        final_pool = []
        seen_industries = set()
        etf_money = []
        
        # 逐个获取昨日数据，规避底层 pandas concat bug
        for code in sector_etfs:
            try:
                # 获取该标的昨日数据
                df = attribute_history(code, 1, '1d', ['money'])
                if not df.empty and not np.isnan(df['money'].iloc[0]):
                    val = df['money'].iloc[0]
                    if val > 50000000:
                        etf_money.append((code, val))
            except:
                continue
                
        # 降序排列
        etf_money.sort(key=lambda x: x[1], reverse=True)
        
        for code, val in etf_money:
            try:
                name = get_security_info(code).display_name
                industry_key = name[:2]
                if industry_key not in seen_industries:
                    final_pool.append(code)
                    seen_industries.add(industry_key)
            except:
                final_pool.append(code)
            if len(final_pool) >= 100: break
            
        Config.DYNAMIC_ETF_POOL = final_pool
        log.info(f"动态池更新完成，入选 {len(final_pool)} 只标的 (种子数:{len(sector_etfs)})。")
    except Exception as e:
        log.error(f"动态池更新出错: {e}")

# ==========================================
# 核心逻辑：指标计算与筛选 (高保原度)
# ==========================================
def get_final_ranked_etfs(context):
    """获取最终排名的 ETF 列表"""
    pool = list(set(Config.FIXED_ETF_POOL + Config.DYNAMIC_ETF_POOL))
    log.info(f"选股池总数: {len(pool)} (固定: {len(Config.FIXED_ETF_POOL)}, 动态: {len(Config.DYNAMIC_ETF_POOL)})")
    if not pool: return []
    
    # 确定最大回看长度
    lookback = max(Config.LOOKBACK_DAYS, Config.SHORT_LOOKBACK, Config.MA_DAYS, Config.VOLUME_LOOKBACK, Config.RSI_PERIOD + Config.RSI_LOOKBACK) + 20
    current_data = get_current_data()
    
    metrics_list = []
    stats = {k: 0 for k in ['total', 'price_valid', 'mom', 'short', 'r2', 'ann', 'ma', 'vol', 'loss', 'rsi']}
    
    for etf in pool:
        try:
            stats['total'] += 1
            
            # 使用 attribute_history 逐个获取，避免批量带来的 MultiIndex/Concat 错误
            df = attribute_history(etf, lookback, '1d', ['close', 'volume'])
            if df.empty or len(df) < Config.LOOKBACK_DAYS:
                continue
                
            closes = df['close'].dropna().values
            vols = df['volume'].dropna().values
            if len(closes) < Config.LOOKBACK_DAYS: continue
            
            # 获取当前实时数据
            data = current_data[etf]
            cur_price = data.last_price
            cur_vol = data.volume
            
            # 手动校权：如果 last_price 是 0（通常是停牌或数据延迟），回退到前一根 K 线
            if np.isnan(cur_price) or cur_price <= 0:
                cur_price = closes[-1]
                cur_vol = vols[-1]
            
            stats['price_valid'] += 1
            price_series = np.append(closes, cur_price)
            
            # 1. 动量得分
            recent_y = np.log(price_series[-(Config.LOOKBACK_DAYS + 1):])
            x = np.arange(len(recent_y))
            weights = np.linspace(1, 2, len(recent_y))
            slope, intercept = np.polyfit(x, recent_y, 1, w=weights)
            ann_ret = math.exp(slope * 250) - 1
            ss_res = np.sum(weights * (recent_y - (slope * x + intercept)) ** 2)
            ss_tot = np.sum(weights * (recent_y - np.mean(recent_y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot else 0
            score = ann_ret * r_squared
            
            # 2. 短期动量
            short_ann = -np.inf
            if len(price_series) > Config.SHORT_LOOKBACK:
                short_ret = price_series[-1] / price_series[-(Config.SHORT_LOOKBACK + 1)] - 1
                short_ann = (1 + short_ret) ** (250 / Config.SHORT_LOOKBACK) - 1
            
            # 3. 均线与指标
            ma_val = np.mean(price_series[-Config.MA_DAYS:])
            
            # 成交量比计算
            vol_ratio = 0
            if len(vols) >= Config.VOLUME_LOOKBACK:
                avg_v = np.mean(vols[-Config.VOLUME_LOOKBACK:])
                vol_ratio = cur_vol / avg_v if avg_v > 0 else 0
            
            # RSI 过滤
            passed_rsi = True
            if Config.USE_RSI_FILTER and len(price_series) >= Config.RSI_PERIOD + Config.RSI_LOOKBACK:
                rsi_vals = calculate_rsi(price_series, Config.RSI_PERIOD)
                recent_rsi = rsi_vals[-Config.RSI_LOOKBACK:]
                if np.any(recent_rsi > Config.RSI_THRESHOLD):
                    ma5 = np.mean(price_series[-5:])
                    if price_series[-1] < ma5: passed_rsi = False
            
            # 4. 短期风控
            passed_loss = True
            if Config.ENABLE_LOSS_FILTER and len(price_series) >= 4:
                if min(price_series[-1]/price_series[-2], price_series[-2]/price_series[-3], price_series[-3]/price_series[-4]) < Config.LOSS_THRESHOLD:
                    passed_loss = False
            
            m = {
                'code': etf, 'score': score, 
                'p_mom': Config.MIN_SCORE <= score <= Config.MAX_SCORE,
                'p_short': not Config.USE_SHORT_MOM or short_ann >= Config.SHORT_MOM_THRESHOLD,
                'p_r2': not Config.ENABLE_R2_FILTER or r_squared > Config.R2_THRESHOLD,
                'p_ann': not Config.ENABLE_ANN_RET_FILTER or ann_ret >= Config.MIN_ANN_RET,
                'p_ma': not Config.ENABLE_MA_FILTER or price_series[-1] >= ma_val,
                'p_vol': not Config.ENABLE_VOLUME_CHECK or (vol_ratio < Config.VOLUME_THRESHOLD),
                'p_loss': passed_loss, 'p_rsi': passed_rsi
            }
            
            # 计数统计
            if m['p_mom']: stats['mom'] += 1
            if m['p_short']: stats['short'] += 1
            if m['p_r2']: stats['r2'] += 1
            if m['p_ann']: stats['ann'] += 1
            if m['p_ma']: stats['ma'] += 1
            if m['p_vol']: stats['vol'] += 1
            if m['p_loss']: stats['loss'] += 1
            if m['p_rsi']: stats['rsi'] += 1
            
            metrics_list.append(m)
        except Exception as e:
            continue
        
    # 应用过滤器
    passed = [m for m in metrics_list if all([m['p_mom'], m['p_short'], m['p_r2'], m['p_ann'], m['p_ma'], m['p_vol'], m['p_loss'], m['p_rsi']])]
    
    log.info(f"过滤统计: 有效价格:{stats['price_valid']}, 动量通过:{stats['mom']}, R2通过:{stats['r2']}, 均量通过:{stats['vol']}, 风控通过:{stats['loss']}, 最终通过:{len(passed)}")
    
    # 排序选出前 N
    ranked = sorted(passed, key=lambda x: x['score'], reverse=True)
    top_etfs = [x['code'] for x in ranked[:Config.HOLDINGS_NUM]]
    return top_etfs
    
    # 排序选出前 N
    ranked = sorted(passed, key=lambda x: x['score'], reverse=True)
    top_etfs = [x['code'] for x in ranked[:Config.HOLDINGS_NUM]]
    return top_etfs

# ==========================================
# 交易执行与风险控制
# ==========================================
def every_minute_check(context):
    """整合后的每分钟风控检查器"""
    cur_t = context.current_dt.strftime('%H:%M')
    if not (('09:31' <= cur_t <= '11:30') or ('13:00' <= cur_t <= '14:57')):
        return
        
    # 遍历当前持仓执行止损检查
    for stock in list(context.portfolio.positions.keys()):
        if stock == Config.DEFENSIVE_ETF or stock == Config.SAFE_HAVEN_ETF: continue
        
        pos = context.portfolio.positions[stock]
        cur_price = get_close_price(stock)
        name = get_security_name(stock)
        
        # 1. 固定比例止损
        if Config.USE_FIXED_STOP and cur_price < pos.avg_cost * Config.FIXED_STOP_THRESHOLD:
            if smart_order_target_value(stock, 0, context):
                log.info(f"⚡ [固定止损] 触发: {stock}({name}), 价格:{cur_price:.3f}, 成本:{pos.avg_cost:.3f}")
            continue

        # 2. 当日跌幅止损
        if Config.USE_PCT_STOP:
            p_open = get_open_price(stock)
            if cur_price < p_open * Config.PCT_STOP_THRESHOLD:
                if smart_order_target_value(stock, 0, context):
                    log.info(f"⚡ [跌幅止损] 触发: {stock}({name}), 价格:{cur_price:.3f}, 开盘:{p_open:.3f}")
                continue

        # 3. ATR 跟踪止损
        if Config.USE_ATR_STOP:
            # 更新最高价
            Config.position_highs[stock] = max(Config.position_highs.get(stock, 0), cur_price)
            atr, _, _, _ = calculate_atr(stock, Config.ATR_PERIOD)
            stop_price = Config.position_highs[stock] - Config.ATR_MULTIPLIER * atr
            if cur_price < stop_price:
                if smart_order_target_value(stock, 0, context):
                    log.info(f"⚡ [ATR止损] 触发: {stock}, 价格:{cur_price}, 止损位:{stop_price}")
                continue

def etf_sell_trade(context):
    """轮动卖出逻辑"""
    log.info("--- 轮动卖出开始 ---")
    # 计算今日目标
    Config.target_etfs = get_final_ranked_etfs(context)
    if not Config.target_etfs:
        # 如果没有目标，且不处于冷却期，则持有避险/防御
        if not is_in_cooldown(context):
            Config.target_etfs = [Config.DEFENSIVE_ETF]
            
    target_set = set(Config.target_etfs)
    for stock in list(context.portfolio.positions.keys()):
        if stock not in target_set:
            name = get_security_name(stock)
            if smart_order_target_value(stock, 0, context):
                log.info(f"✅ 卖出轮动不再持有的标的: {stock}({name})")

def etf_buy_trade(context):
    """轮动买入逻辑"""
    log.info("--- 轮动买入开始 ---")
    if is_in_cooldown(context): return
    
    targets = [t for t in Config.target_etfs if t not in context.portfolio.positions]
    if not targets: return
    
    cash = context.portfolio.available_cash
    if cash < Config.MIN_MONEY: return
    
    val_per = cash / len(targets)
    for stock in targets:
        smart_order_target_value(stock, val_per, context)

# ==========================================
# 工具函数
# ==========================================
def smart_order_target_value(stock, value, context):
    """智能下单：处理 T+1、100股取整、最小金额"""
    cur_pos = context.portfolio.positions[stock].total_amount if stock in context.portfolio.positions else 0
    target_amount = 0
    if value > 0:
        price = get_close_price(stock)
        target_amount = int(value / price / 100) * 100
        
    if target_amount == cur_pos: return False
    
    if target_amount == 0: # 卖出
        if stock in context.portfolio.positions:
            sellable = context.portfolio.positions[stock].closeable_amount
            if sellable > 0:
                order_target(stock, 0)
                return True
        return False
    else: # 买入
        if value < Config.MIN_MONEY: return False
        # 买入前清理多余持仓以确保资金充足
        order_target_value(stock, value)
        return True

def get_close_price(stock):
    return get_current_data()[stock].last_price

def get_open_price(stock):
    return get_current_data()[stock].day_open

def get_volume_ratio(context, stock):
    # 简化后的 batch 友好成交量比计算
    try:
        h = attribute_history(stock, Config.VOLUME_LOOKBACK, '1d', ['volume'])
        avg_v = h['volume'].mean()
        cur_v = get_current_data()[stock].volume
        return cur_v / avg_v if avg_v > 0 else None
    except: return None

def calculate_rsi(prices, period=6):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    alpha = 2.0 / (period + 1)
    avg_gains = np.zeros(len(deltas))
    avg_losses = np.zeros(len(deltas))
    avg_gains[period-1] = np.mean(gains[:period])
    avg_losses[period-1] = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gains[i] = (gains[i] * alpha) + (avg_gains[i-1] * (1 - alpha))
        avg_losses[i] = (losses[i] * alpha) + (avg_losses[i-1] * (1 - alpha))
    rs = avg_gains / avg_losses
    return 100 - (100 / (1 + rs))

def calculate_atr(stock, period=14):
    h = attribute_history(stock, period + 1, '1d', ['high', 'low', 'close'])
    tr = np.maximum(h['high'].values[1:] - h['low'].values[1:], 
                   np.maximum(abs(h['high'].values[1:] - h['close'].values[:-1]), 
                             abs(h['low'].values[1:] - h['close'].values[:-1])))
    return np.mean(tr), None, True, ""

def is_in_cooldown(context):
    if not Config.SELL_COOLDOWN_ENABLED or not Config.COOLDOWN_END_DATE: return False
    return context.current_dt.date() <= Config.COOLDOWN_END_DATE

def set_cooldown(context):
    if Config.SELL_COOLDOWN_ENABLED:
        Config.COOLDOWN_END_DATE = context.current_dt.date() + pd.Timedelta(days=Config.SELL_COOLDOWN_DAYS)
        log.info(f"🔒 触发冷却期，结束日期: {Config.COOLDOWN_END_DATE}")

def enter_safe_haven_and_set_cooldown(context, trigger_reason=""):
    """进入冷却期：卖出非固定标的，买入避险 ETF"""
    if not Config.SELL_COOLDOWN_ENABLED: return
    
    # 卖出持有中的轮动标的
    for stock in list(context.portfolio.positions.keys()):
        if stock not in Config.FIXED_ETF_POOL and stock != Config.DEFENSIVE_ETF:
            smart_order_target_value(stock, 0, context)
            
    # 买入避险标的
    total_val = context.portfolio.total_value
    if total_val > Config.MIN_MONEY:
        smart_order_target_value(Config.SAFE_HAVEN_ETF, total_val * 0.99, context)
        
    set_cooldown(context)
    log.info(f"🔒 进入冷却期模式 (触发原因: {trigger_reason})")

def exit_safe_haven_if_cooldown_ends(context):
    """冷却期结束：清理避险标的"""
    if not Config.SELL_COOLDOWN_ENABLED or not Config.COOLDOWN_END_DATE: return
    if context.current_dt.date() > Config.COOLDOWN_END_DATE:
        log.info(f"🔓 冷却期结束 ({Config.COOLDOWN_END_DATE})，清理避险头寸")
        smart_order_target_value(Config.SAFE_HAVEN_ETF, 0, context)
        Config.COOLDOWN_END_DATE = None

def get_security_name(stock):
    try: return get_current_data()[stock].name
    except: return stock

def check_positions_before(context):
    # 同步持仓状态
    for stock in context.portfolio.positions:
        if Config.USE_ATR_STOP:
            Config.position_highs[stock] = max(Config.position_highs.get(stock, 0), context.portfolio.positions[stock].price)
