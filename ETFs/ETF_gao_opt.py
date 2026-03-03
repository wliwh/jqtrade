# 策略名称：高收益ETF轮动策略（模块化融合 7star 严控版）
# 核心算法：WLS加权线性回归
# 交易机制：7star 多维过滤、极速冷热状态机、O(1)级高效分钟止损
# 重构优化：针对回测性能进行了指标静态化缓存处理

import numpy as np
import math
import pandas as pd
from jqdata import *
import datetime

class Config:
    # ==================== 交易环境与基础参数 ====================
    BENCHMARK = "510300.XSHG"
    HOLD_COUNT = 1                   # 持仓数量
    M_DAYS = 25                      # 动量计算观察期 (默认25日)
    MIN_MONEY = 5000                 # 最小交易金额，防止碎股摩擦
    
    # 避险与防御资产
    DEFENSIVE_ETF = "511880.XSHG"    # 防御ETF (银华日利)
    SAFE_HAVEN_ETF = "511660.XSHG"   # 冷却期避险ETF (建信短融)
    
    # ==================== 7Star 多维过滤体系 ====================
    # 1. 短期动量过滤
    ENABLE_SHORT_MOM = False         # 【开关】短期动量
    SHORT_DAYS = 10                  
    SHORT_THRESHOLD = 0.0            
    
    # 2. R² 与 年化绝对收益过滤
    ENABLE_R2_FILTER = True          # 【开关】R²线性度过滤
    R2_THRESHOLD = 0.4               
    ENABLE_ANNUAL_RET = False        # 【开关】绝对收益要求
    MIN_ANNUAL_RET = 1.0             
    
    # 3. 均线与成交量过滤
    ENABLE_MA_FILTER = False         # 【开关】站上均线过滤
    MA_DAYS = 20
    ENABLE_VOLUME_CHECK = True       # 【开关】异常放量过滤
    VOLUME_LOOKBACK = 5
    VOLUME_THRESHOLD = 1.0           # 当日成交量不能比过去5日均量超过的倍数
    
    # 4. 短期暴跌防守
    ENABLE_LOSS_FILTER = True        # 【开关】3日内单日跌幅防守
    MAX_DAILY_DROP = 0.97            # 任一日跌幅不得超过 3%
    
    # 5. RSI 追高防波段过滤
    ENABLE_RSI_FILTER = False        # 【开关】RSI防追高过滤
    RSI_PERIOD = 6
    RSI_LOOKBACK = 1
    RSI_THRESHOLD = 98               # 若RSI触及极值，禁止由于趋势良好而继续买入
    
    # ==================== 分钟级止损与冷却机制 ====================
    ENABLE_COOLDOWN = True           # 【开关】触发止损后必须强制冷却
    COOLDOWN_DAYS = 3                # 冷却天数
    
    ENABLE_FIXED_STOP = True         # 【开关】固定成本止损
    FIXED_STOP_PCT = 0.95            # 亏损5%走人
    
    ENABLE_DAILY_STOP = False        # 【开关】当日由于开盘触发的跌幅止损
    DAILY_STOP_PCT = 0.95
    
    ENABLE_ATR_STOP = False          # 【开关】ATR动态跟踪止损 (优化版)
    ATR_PERIOD = 14
    ATR_MULTI = 2.0
    ATR_TRAILING = True              # 是否记录历史最高价进行滑动跟踪
    
    # ==================== ETF 预设组合池 ====================
    ETF_POOL = [
        # 美股
        '513100.XSHG', '159509.XSHE', '513500.XSHG', '513400.XSHG',
        # 商品 & 资源
        '518880.XSHG', '159980.XSHE', '501018.XSHG', '512400.XSHG', '515220.XSHG',
        # 宽基与科技
        '510500.XSHG', '512100.XSHG', '510300.XSHG', '159915.XSHE', '588080.XSHG',
        # 代表性行业
        '512880.XSHG', '512480.XSHG', '159852.XSHE', '512690.XSHG'
    ]

# ==================== 系统初始化 ====================
def initialize(context):
    set_benchmark(Config.BENCHMARK)
    set_option("avoid_future_data", True)
    set_option("use_real_price", True)
    
    # 日志接管
    log.set_level('order', 'error')
    
    # 交易标的手续费细分 (模拟基金和短融的区别)
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, 
        open_commission=0.0001, close_commission=0.0001, min_commission=5
    ), type="fund")
    
    # 全局状态管理
    g.etf_pool = Config.ETF_POOL
    g.target_list = []
    
    g.cooldown_end_date = None        # 冷却期解冻日
    g.position_highs = {}             # 记录最高价 (用于ATR跟踪)
    g.daily_cache = {}                # 【性能优化核心】缓存每日静态指标，如今日ATR、开盘价
    
    # 定时主任务注册
    run_daily(prepare_daily_metrics, time='09:30') # 早盘预计算耗时指标，极大降低分钟级计算损耗
    run_daily(etf_trade_sell, time='13:10')        # 遵循7star 13:10 卖
    run_daily(etf_trade_buy, time='13:11')         # 遵循7star 13:11 买
    
    log.info("模块化高能版策略启动：持仓 O(1) 极速止损网与多维风控雷达已就绪")

# ==================== O(1) 极速分钟级止损引擎 ====================
def prepare_daily_metrics(context):
    """
    每天开盘统一计算并缓存止损所需的常态化数据（如ATR、今日开盘价）。
    避免分钟级任务调用 get_price 或 attribute_history 导致卡顿。
    """
    g.daily_cache.clear()
    positions = context.portfolio.positions
    
    if not positions: return
        
    for security in positions:
        if security in [Config.DEFENSIVE_ETF, Config.SAFE_HAVEN_ETF]:
            continue # 防御资产和避险资产不参与止损
            
        cache_data = {}
        
        # 1. 获取当天的初始开盘价 (用于当日跌幅止损)
        if Config.ENABLE_DAILY_STOP:
            current_data = get_current_data()
            cache_data['today_open'] = current_data[security].day_open
            
        # 2. 计算并静态化当前的 ATR
        if Config.ENABLE_ATR_STOP:
            # 仅仅在这里调用过去 15 天的数据计算ATR
            h = attribute_history(security, Config.ATR_PERIOD + 1, '1d', ['high', 'low', 'close'])
            if len(h) >= Config.ATR_PERIOD + 1:
                tr = np.maximum.reduce([
                    h['high'].values[1:] - h['low'].values[1:],
                    np.abs(h['high'].values[1:] - h['close'].values[:-1]),
                    np.abs(h['low'].values[1:] - h['close'].values[:-1])
                ])
                atr = np.mean(tr)
                cache_data['atr'] = atr
                
                # 初始化或同步最高价
                if security not in g.position_highs:
                    g.position_highs[security] = context.portfolio.positions[security].avg_cost
                    
        g.daily_cache[security] = cache_data

def handle_data(context, data):
    """
    代替几百个 run_daily 的极速分钟轮询。只在持仓且非冷却期时做 O(1) 运算。
    """
    # 提前异常或状态拦截，提升引擎挂机效率
    if not context.portfolio.positions: return
    if Config.ENABLE_COOLDOWN and g.cooldown_end_date and context.current_dt.date() < g.cooldown_end_date:
        return
        
    current_time = context.current_dt.strftime("%H:%M")
    # 只在交易时段检查
    if not (('09:30' <= current_time < '11:30') or ('13:00' <= current_time < '14:57')):
        return

    triggered_stop = False
    
    # 单次遍历持仓，综合判断全部止损逻辑
    for security, position in list(context.portfolio.positions.items()):
        if position.total_amount <= 0: continue
        if security in [Config.DEFENSIVE_ETF, Config.SAFE_HAVEN_ETF]: continue
        
        cur_price = data[security].price
        if cur_price <= 0 or math.isnan(cur_price): continue
            
        cost_price = position.avg_cost
        reason = ""
        
        # 维护基于当前最新价的最高价水位
        if Config.ENABLE_ATR_STOP and Config.ATR_TRAILING and security in g.position_highs:
            g.position_highs[security] = max(g.position_highs[security], cur_price)

        # ====== 1. 固定成本止损 ======
        if Config.ENABLE_FIXED_STOP and cur_price <= cost_price * Config.FIXED_STOP_PCT:
            reason = f"固定止损(跌破成本{1 - Config.FIXED_STOP_PCT:.1%})"
            
        # ====== 2. 当日跌幅止损 ======
        elif Config.ENABLE_DAILY_STOP and security in g.daily_cache:
            today_open = g.daily_cache[security].get('today_open', 0)
            if today_open > 0 and cur_price <= today_open * Config.DAILY_STOP_PCT:
                reason = f"当日跳水(较开盘跌破{1 - Config.DAILY_STOP_PCT:.1%})"
                
        # ====== 3. ATR动态跟踪止损 ======
        elif Config.ENABLE_ATR_STOP and security in g.daily_cache:
            atr = g.daily_cache[security].get('atr', 0)
            if atr > 0:
                highest = g.position_highs[security] if Config.ATR_TRAILING else cost_price
                stop_line = highest - (Config.ATR_MULTI * atr)
                if cur_price <= stop_line:
                    mode = "跟踪" if Config.ATR_TRAILING else "固定"
                    reason = f"ATR{mode}止损(目前回落{Config.ATR_MULTI}ATR)"

        # 触发止损执行拔线
        if reason:
            log.warning(f"🚨 [分钟级止损]: {security} 当前价 {cur_price:.3f} 触发 {reason}，强制清仓！")
            order_target(security, 0)
            
            # 清理状态机数据
            g.position_highs.pop(security, None)
            g.daily_cache.pop(security, None)
            triggered_stop = True

    # 若任一股票触发止损，立刻全员进入冷却期并切入避险资金池
    if triggered_stop and Config.ENABLE_COOLDOWN:
        g.cooldown_end_date = context.current_dt.date() + datetime.timedelta(days=Config.COOLDOWN_DAYS)
        log.info(f"❄️ 系统进入异常冷却期直至：{g.cooldown_end_date}")
        
        # 【改进1】先清空所有剩余的主力持仓（止损已砍掉部分，将其他仓也一并平掉）
        for sec in list(context.portfolio.positions.keys()):
            pos = context.portfolio.positions[sec]
            if pos.total_amount > 0 and sec != Config.SAFE_HAVEN_ETF:
                order_target(sec, 0)
                g.position_highs.pop(sec, None)
                g.daily_cache.pop(sec, None)
                log.info(f"❄️ [冷却期] 连带清仓: {sec}")
        
        # 【改进1】用总资产的 99% 全量买入避险工具（而非只用剩余现金）
        if Config.SAFE_HAVEN_ETF:
            total_val = context.portfolio.total_value
            if total_val > Config.MIN_MONEY:
                order_target_value(Config.SAFE_HAVEN_ETF, total_val * 0.99)
                log.info(f"🏦 已将全仓转入避险资产(99%总资产): {Config.SAFE_HAVEN_ETF}")

# ==================== 评级与核心过滤引擎 ====================
def get_recent_volume_ratio(etf, context):
    """提取最新的分钟级累计成交量与过去数日的平均成交量对比"""
    try:
        hist = attribute_history(etf, Config.VOLUME_LOOKBACK, '1d', ['volume'])
        if len(hist) < Config.VOLUME_LOOKBACK: return None
        avg_vol = hist['volume'].mean()
        
        # 当日的当前时点累计量
        today = context.current_dt.date()
        df = get_price(etf, start_date=today, end_date=context.current_dt, frequency='1m', fields=['volume'], panel=False)
        cur_vol = df['volume'].sum() if (df is not None and not df.empty) else 0
        
        return cur_vol / avg_vol if avg_vol > 0 else 0
    except:
        return None

def calc_rsi(prices, period):
    if len(prices) < period + 1: return np.array([])
    diff = np.diff(prices)
    pos = np.where(diff > 0, diff, 0)
    neg = np.where(diff < 0, -diff, 0)
    
    # 采用 Pandas 实现计算，保证性能与可靠性
    avg_gain = pd.Series(pos).rolling(window=period).mean().values
    avg_loss = pd.Series(neg).rolling(window=period).mean().values
    
    rsi = np.zeros_like(avg_gain)
    mask = avg_loss != 0
    rsi[mask] = 100 - (100 / (1 + avg_gain[mask] / avg_loss[mask]))
    rsi[~mask] = 100
    return rsi

def evaluate_etf_fitness(etf, context):
    """综合计算该 ETF 并执行所有的过滤逻辑"""
    try:
        max_lookback = max(Config.M_DAYS, Config.MA_DAYS, Config.RSI_PERIOD * 2) + 5
        hists = attribute_history(etf, max_lookback, '1d', ['close', 'high', 'low'], skip_paused=True)
        if len(hists) < Config.M_DAYS: return None
        
        current_data = get_current_data()
        cur_price = current_data[etf].last_price
        if math.isnan(cur_price) or cur_price == 0: return None
        
        prices_series = np.append(hists['close'].values, cur_price)
        
        # ---------------- 动量核心：WLS算法 ----------------
        calc_prices = prices_series[-(Config.M_DAYS + 1):]
        y = np.log(calc_prices)
        x = np.arange(len(y))
        w = np.linspace(1, 2, len(y))
        slope, intercept = np.polyfit(x, y, 1, w=w)
        
        ann_ret = math.exp(slope * 250) - 1
        y_pred = slope * x + intercept
        ss_res = np.sum(w * (y - y_pred) ** 2)
        ss_tot = np.sum(w * (y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        
        score = ann_ret * r2
        
        # ---------------- 过滤逻辑 ----------------
        # 1. R2 和 绝对收益过滤
        if Config.ENABLE_R2_FILTER and r2 <= Config.R2_THRESHOLD: return None
        if Config.ENABLE_ANNUAL_RET and ann_ret < Config.MIN_ANNUAL_RET: return None
        
        # 2. 短期风控过滤 (任意3日内最大幅度)
        if Config.ENABLE_LOSS_FILTER and len(prices_series) >= 4:
            drops = [prices_series[-1]/prices_series[-2], prices_series[-2]/prices_series[-3], prices_series[-3]/prices_series[-4]]
            if min(drops) < Config.MAX_DAILY_DROP: return None
                
        # 3. 均线跌破过滤
        if Config.ENABLE_MA_FILTER:
            ma_price = np.mean(prices_series[-Config.MA_DAYS:])
            if cur_price < ma_price: return None
                
        # 4. 短期动量过滤
        if Config.ENABLE_SHORT_MOM and len(prices_series) >= Config.SHORT_DAYS + 1:
            short_ret = prices_series[-1] / prices_series[-(Config.SHORT_DAYS + 1)] - 1
            if short_ret < Config.SHORT_THRESHOLD: return None
                
        # 5. RSI 反追高过滤
        if Config.ENABLE_RSI_FILTER:
            rsi_vals = calc_rsi(prices_series, Config.RSI_PERIOD)
            if len(rsi_vals) > Config.RSI_LOOKBACK:
                recent_rsi = rsi_vals[-Config.RSI_LOOKBACK:]
                if np.max(recent_rsi) > Config.RSI_THRESHOLD:
                    ma5 = np.mean(prices_series[-5:]) if len(prices_series)>=5 else cur_price
                    if cur_price < ma5: return None

        return {
            'etf': etf,
            'score': score,
            'r2': r2,
            'ann_ret': ann_ret
        }
    except Exception as e:
        log.warn(f"计算分数异常 {etf}: {e}")
        return None

def get_target_list(context):
    """扫描 ETF 池，生成符合风控与动量门槛的多头名单"""
    scored_list = []
    
    for etf in g.etf_pool:
        # A. 成交量实时过滤
        if Config.ENABLE_VOLUME_CHECK:
            vol_ratio = get_recent_volume_ratio(etf, context)
            if vol_ratio is not None and vol_ratio > Config.VOLUME_THRESHOLD:
                continue
                
        # B. 动量计算与技术指标筛查
        res = evaluate_etf_fitness(etf, context)
        if res:
            scored_list.append(res)
            
    scored_list.sort(key=lambda x: x['score'], reverse=True)
    return [x['etf'] for x in scored_list]

# ==================== 交易模块 ====================
def smart_order(security, value, context):
    current_data = get_current_data()
    if current_data[security].paused: return
    price = current_data[security].last_price
    if price >= current_data[security].high_limit or price <= current_data[security].low_limit: return
    if price == 0: return
    
    target_amount = (int(value / price) // 100) * 100
    pos = context.portfolio.positions[security]
    diff_val = abs(value - pos.total_amount * price)
    
    if diff_val >= Config.MIN_MONEY or value == 0:
        order_target_value(security, value)
        
        # 买入同步更新：由于止损引擎依赖 prepare_daily_metrics 的缓存，
        # 日内新买入的品种需要即时补齐 ATR 缓存，否则当天将处于“无止损护甲”状态。
        amount_after = context.portfolio.positions[security].total_amount
        if amount_after > 0 and value > 0 and security not in [Config.DEFENSIVE_ETF, Config.SAFE_HAVEN_ETF]:
            g.position_highs[security] = price
            if Config.ENABLE_ATR_STOP:
                h = attribute_history(security, Config.ATR_PERIOD + 1, '1d', ['high', 'low', 'close'])
                if len(h) >= Config.ATR_PERIOD + 1:
                    tr = np.maximum.reduce([
                        h['high'].values[1:] - h['low'].values[1:],
                        np.abs(h['high'].values[1:] - h['close'].values[:-1]),
                        np.abs(h['low'].values[1:] - h['close'].values[:-1])
                    ])
                    atr = np.mean(tr)
                    if security not in g.daily_cache:
                        g.daily_cache[security] = {}
                    g.daily_cache[security]['atr'] = atr

def etf_trade_sell(context):
    log.info("========== 开始轮动卖出 (13:10) ==========")
    # 冷却期拦截
    if Config.ENABLE_COOLDOWN and g.cooldown_end_date and context.current_dt.date() < g.cooldown_end_date:
        log.info(f"❄️ 状态锁定：当前处于冷却期内，跳过轮动交易")
        return
        
    g.target_list = get_target_list(context)
    ideal_targets = g.target_list[:Config.HOLD_COUNT]

    for etf in list(context.portfolio.positions.keys()):
        if context.portfolio.positions[etf].total_amount == 0: continue
        
        if etf not in ideal_targets:
            log.info(f"退出持仓: {etf}")
            smart_order(etf, 0, context)
            g.daily_cache.pop(etf, None)
            g.position_highs.pop(etf, None)

def exit_safe_haven_if_cooldown_ends(context):
    """【改进2】冷却期结束时主动清仓避险 ETF，为正常轮动腾出资金"""
    if not Config.ENABLE_COOLDOWN or g.cooldown_end_date is None:
        return
    if context.current_dt.date() > g.cooldown_end_date:
        log.info(f"🔓 冷却期已结束 ({g.cooldown_end_date})，开始退出避险仓位")
        if Config.SAFE_HAVEN_ETF in context.portfolio.positions:
            pos = context.portfolio.positions[Config.SAFE_HAVEN_ETF]
            if pos.total_amount > 0:
                order_target(Config.SAFE_HAVEN_ETF, 0)
                log.info(f"✅ 已清仓避险资产: {Config.SAFE_HAVEN_ETF}")
        g.cooldown_end_date = None
        log.info("🔄 策略恢复正常做多模式")

def etf_trade_buy(context):
    log.info("========== 开始轮动买入 (13:11) ==========")
    
    # 【改进2】在买入前，先尝试解除冷却并卖出避险仓
    exit_safe_haven_if_cooldown_ends(context)
    
    if Config.ENABLE_COOLDOWN and g.cooldown_end_date and context.current_dt.date() <= g.cooldown_end_date:
        log.info(f"❄️ 冷却期内 (截止 {g.cooldown_end_date})，跳过买入")
        return
        
    final_buy_targets = g.target_list[:Config.HOLD_COUNT]
    
    # 无强势标的时切入防御性 ETF (如货币基金)
    if not final_buy_targets:
        log.info("市场选不出强势 ETF，切入防御姿态")
        if Config.DEFENSIVE_ETF:
            final_buy_targets = [Config.DEFENSIVE_ETF]
        else:
            return

    # 【改进3】只对「还未持有」的目标下单，用 available_cash 分配，末尾完全吃光碎钱
    current_positions = set(context.portfolio.positions.keys())
    to_buy = [etf for etf in final_buy_targets if etf not in current_positions]
    
    if not to_buy:
        log.info("目标持仓无变化，无需买入")
        return
    
    available_cash = context.portfolio.available_cash
    if available_cash < Config.MIN_MONEY:
        log.info(f"可用资金 {available_cash:.2f} 不足，跳过买入")
        return
    
    per_value = available_cash // len(to_buy)
    log.info(f"可用资金: {available_cash:.2f}，待买 {len(to_buy)} 只，每只分配: {per_value:.2f}")
    
    for i, etf in enumerate(to_buy):
        # 最后一只 ETF 吸收全部剩余可用现金，避免碎钱浪费
        if i == len(to_buy) - 1:
            value = context.portfolio.available_cash
            log.info(f"  最后一只 {etf}：吃光剩余现金 {value:.2f}")
        else:
            value = per_value
        smart_order(etf, value, context)
