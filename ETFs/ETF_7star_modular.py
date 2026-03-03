# -*- coding: utf-8 -*-
# ETF_7star_modular.py
# 基于聚宽平台重构的模块化ETF 7-star策略

import numpy as np
import math
import pandas as pd
from jqdata import *
from datetime import datetime, date

# ==============================================================================
# 1. Config (参数配置模块)
# ==============================================================================
class Config:
    def __init__(self):
        # 基础参数
        self.holdings_num = 1                  # 持仓数量
        self.defensive_etf = "511880.XSHG"     # 防御型ETF (银华日利)
        self.safe_haven_etf = '511660.XSHG'    # 冷却期避险ETF (建信添益)
        self.min_money = 5000                  # 最小交易金额
        
        # 动量因子参数
        self.lookback_days = 25                # 核心动量回看天数
        self.min_score_threshold = 0           # 动量得分下限
        self.max_score_threshold = 5           # 动量得分上限
        
        # 过滤器开关与阈值
        self.use_short_momentum_filter = False # 短期动量过滤
        self.short_lookback_days = 10
        self.short_momentum_threshold = 0.0
        
        self.enable_r2_filter = True           # R²过滤
        self.r2_threshold = 0.4
        
        self.enable_annualized_return_filter = False # 年化收益过滤
        self.min_annualized_return = 1.0
        
        self.enable_ma_filter = False          # 均线过滤
        self.ma_filter_days = 20
        
        self.enable_volume_check = True        # 成交量比过滤
        self.volume_lookback = 5
        self.volume_threshold = 1.0
        
        self.enable_loss_filter = True         # 短期风控跌幅过滤
        self.loss_limit = 0.97                 # 3日最大允许跌幅
        
        self.use_rsi_filter = False            # RSI过滤
        self.rsi_period = 6
        self.rsi_lookback_days = 1
        self.rsi_threshold = 98
        
        # 止损逻辑
        self.use_fixed_stop_loss = True        # 固定比例止损
        self.fixed_stop_loss_threshold = 0.95
        
        self.use_pct_stop_loss = False         # 当日跌幅止损
        self.pct_stop_loss_threshold = 0.95
        
        self.use_atr_stop_loss = False         # ATR动态止损
        self.atr_period = 14
        self.atr_multiplier = 2
        self.atr_trailing_stop = True
        self.atr_exclude_defensive = True
        
        # 冷却期机制
        self.sell_cooldown_enabled = False     # 卖出冷却期机制
        self.sell_cooldown_days = 3
        
        # 预设ETF池
        self.fixed_etf_pool = [
#大宗商品ETF：
        '518880.XSHG',  # (黄金ETF)
        '161226.XSHE',  # (国投白银LOF)
        '159980.XSHE',  # (有色ETF大成)
        '501018.XSHG',  # (南方原油ETF)
        '159985.XSHE',  # (豆粕ETF)

#海外ETF：       
        '513100.XSHG',  # (纳指ETF)
        '159509.XSHE',  # (纳指科技ETF景顺)
        '513290.XSHG',  # (纳指生物)        
        '513500.XSHG',  # (标普500)
        '159518.XSHE',  # (标普油气ETF嘉实)
        '159502.XSHE',  # (标普生物科技ETF嘉实)        
        '159529.XSHE',  # (标普消费ETF)
        '513400.XSHG',  # (道琼斯)
        '520830.XSHG',  # (沙特ETF)
        '513520.XSHG',  # (日经ETF)
        '513030.XSHG',  # (德国ETF)

#港股ETF：
        '513090.XSHG',  # (香港证券)
        '513180.XSHG',  # (恒指科技)
        '513120.XSHG',  # (HK创新药)
        '513330.XSHG',  # (恒生互联)
        '513750.XSHG',  # (港股非银)
        '159892.XSHE',  # (恒生医药ETF)
        '159605.XSHE',  # (中概互联ETF)
        '513190.XSHG',  # (H股金融)
        '510900.XSHG',  # (恒生中国)
        '513630.XSHG',  # (香港红利)
        '513920.XSHG',  # (港股通央企红利)
        '159323.XSHE',  # (港股通汽车ETF)
        '513970.XSHG',  # (恒生消费)
        
#指数ETF：        
        '510500.XSHG',  # (中证500ETF)
        '512100.XSHG',  # (中证1000ETF)
        '563300.XSHG',  # (中证2000)        
        '510300.XSHG',  # (沪深300ETF)
        '512050.XSHG',  # (A500E)        
        '510760.XSHG',  # (上证ETF)        
        '159915.XSHE',  # (创业板ETF易方达)
        '159949.XSHE',  # (创业板50ETF)
        '159967.XSHE',  # (创业板成长ETF)        
        '588080.XSHG',  # (科创板50)
        '588220.XSHG',  # (科创100)
        '511380.XSHG',  # (可转债ETF)
        
#行业ETF：
        '513310.XSHG',  # (中韩芯片)
        '588200.XSHG',  # (科创芯片)
        '159852.XSHE',  # (软件ETF)
        '512880.XSHG',  # (证券ETF)
        '159206.XSHE',  # (卫星ETF)
        '512400.XSHG',  # (有色金属ETF)
        '512980.XSHG',  # (传媒ETF)
        '159516.XSHE',  # (半导体设备ETF)
        '512480.XSHG',  # (半导体)
        '515880.XSHG',  # (通信ETF)
        '562500.XSHG',  # (机器人)
        '159218.XSHE',  # (卫星产业ETF)
        '159869.XSHE',  # (游戏ETF)
        '159870.XSHE',  # (化工ETF)
        '159326.XSHE',  # (电网设备ETF)
        '159851.XSHE',  # (金融科技ETF)
        '560860.XSHG',  # (工业有色)
        '159363.XSHE',  # (创业板人工智能ETF华宝)
        '588170.XSHG',  # (科创半导)
        '159755.XSHE',  # (电池ETF)
        '512170.XSHG',  # (医疗ETF)
        '512800.XSHG',  # (银行ETF)
        '159819.XSHE',  # (人工智能ETF易方达)
        '512710.XSHG',  # (军工龙头)
        '159638.XSHE',  # (高端装备ETF嘉实)
        '517520.XSHG',  # (黄金股)
        '515980.XSHG',  # (人工智能)
        '159995.XSHE',  # (芯片ETF)
        '159227.XSHE',  # (航空航天ETF)
        '512660.XSHG',  # (军工ETF)
        '512690.XSHG',  # (酒ETF)
        '516150.XSHG',  # (稀土基金)
        '512890.XSHG',  # (红利低波)
        '588790.XSHG',  # (科创智能)
        '159992.XSHE',  # (创新药ETF)
        '512070.XSHG',  # (证券保险)
        '562800.XSHG',  # (稀有金属)
        '512010.XSHG',  # (医药ETF)
        '515790.XSHG',  # (光伏ETF)
        '510880.XSHG',  # (红利ETF)
        '159928.XSHE',  # (消费ETF)
        '159883.XSHE',  # (医疗器械ETF)
        '159998.XSHE',  # (计算机ETF)
        '515220.XSHG',  # (煤炭ETF)
        '561980.XSHG',  # (芯片设备)
        '515400.XSHG',  # (大数据)
        '515120.XSHG',  # (创新药)
        '159967.XSHE',  # (创业板成长ETF)
        '159566.XSHE',  # (储能电池ETF易方达)
        '515050.XSHG',  # (5GETF)
        '516510.XSHG',  # (云计算ETF)
        '159256.XSHE',  # (创业板软件ETF华夏)
        '159766.XSHE',  # (旅游ETF)
        '512200.XSHG',  # (地产ETF)
        '513350.XSHG',  # (油气ETF)
        '159583.XSHE',  # (通信设备ETF)
        '159732.XSHE',  # (消费电子ETF)
        '516160.XSHG',  # (新能源)
        '516520.XSHG',  # (智能驾驶)
        '562590.XSHG',  # (半导材料)
        '515030.XSHG',  # (新汽车)
        '512670.XSHG',  # (国防ETF)
        '561330.XSHG',  # (矿业ETF)
        '516190.XSHG',  # (文娱ETF)
        '159840.XSHE',  # (锂电池ETF工银)
        '159611.XSHE',  # (电力ETF)
        '159981.XSHE',  # (能源化工ETF)
        '159865.XSHE',  # (养殖ETF)
        '561360.XSHG',  # (石油ETF)
        '159667.XSHE',  # (工业母机ETF)
        '515170.XSHG',  # (食品饮料ETF)
        '513360.XSHG',  # (教育ETF)
        '159825.XSHE',  # (农业ETF)
        '515210.XSHG',  # (钢铁ETF)
        ]

# ==============================================================================
# 2. DataManager (数据管理与缓存模块)
# ==============================================================================
class DataManager:
    def __init__(self, config):
        self.config = config
        self.security_info_cache = {}    # 标的信息缓存
        self.batch_history = None        # 批量历史数据缓存 (Panel/MultiIndex DF)
        self.yesterday_money = None      # 昨日成交额缓存
        self.dynamic_pool = []           # 动态筛选出来的行业池
        
    def get_info(self, security):
        """获取并缓存证券基本信息"""
        if security not in self.security_info_cache:
            try:
                info = get_security_info(security)
                self.security_info_cache[security] = {
                    'name': info.display_name,
                    'start_date': info.start_date.date() if isinstance(info.start_date, (datetime, date)) else None
                }
            except:
                self.security_info_cache[security] = {'name': security, 'start_date': None}
        return self.security_info_cache[security]

    def update_dynamic_pool(self, context):
        """每日更新热点行业池 (高成交额 + 行业去重)"""
        all_etfs = get_all_securities(['etf']).index.tolist()
        exclude = ['300', '500', '1000', '50', '上证', '创业板', '科创', '恒生', 'H股', '货币', '纳指', '标普', '债']
        
        # 1. 基础关键字过滤
        candidates = []
        for code in all_etfs:
            name = self.get_info(code)['name']
            if not any(k in name for k in exclude):
                candidates.append(code)
                
        # 2. 批量成交额过滤 (避免单只获取)
        end_dt = context.previous_date
        h = get_price(candidates, count=1, end_date=end_dt, frequency='daily', fields=['money'])
        money_series = h['money'].iloc[0]
        
        # 3. 成交额 > 5000万，且行业去重 (取前2个字符匹配)
        qualified = money_series[money_series > 5e7].sort_values(ascending=False)
        self.yesterday_money = money_series
        
        final_pool = []
        seen = set()
        for code in qualified.index:
            name = self.get_info(code)['name']
            industry = name[:2]
            if industry not in seen:
                final_pool.append(code)
                seen.add(industry)
            if len(final_pool) >= 100: break
            
        self.dynamic_pool = final_pool
        return final_pool

    def prepare_batch_data(self, context, security_list):
        """一次性拉取所有标的历史行情"""
        lookback = max(self.config.lookback_days, self.config.ma_filter_days, 60) + 5
        # 聚宽 get_price 支持多标的返回 DataFrame/Panel
        self.batch_history = get_price(
            security_list, 
            count=lookback, 
            end_date=context.previous_date, 
            frequency='daily', 
            fields=['close', 'high', 'low', 'volume', 'money']
        )
        return self.batch_history

# ==============================================================================
# 3. FilterEngine (多因子过滤与计算模块)
# ==============================================================================
class FilterEngine:
    def __init__(self, config, data_manager):
        self.config = config
        self.dm = data_manager

    def calculate_metrics_batch(self, context, security_list):
        """核心：向量化/批量计算所有标的的因子"""
        # 获取基础数据
        hist = self.dm.batch_history # MultiIndex DataFrame: (Time, Security) -> Field
        current_data = get_current_data()
        
        results = []
        for code in security_list:
            try:
                # 获取该标的序列
                df = hist.xs(code, axis=0, level=1) if isinstance(hist.index, pd.MultiIndex) else hist[:, :, code]
                if len(df) < self.config.lookback_days: continue
                
                # 拼接今日最新价 (分钟级数据)
                curr_price = current_data[code].last_price
                closes = np.append(df['close'].values, curr_price)
                
                # --- 1. 动量得分计算 (线性拟合) ---
                y = np.log(closes[- (self.config.lookback_days + 1):])
                x = np.arange(len(y))
                w = np.linspace(1, 2, len(y))
                slope, intercept = np.polyfit(x, y, 1, w=w)
                
                ann_ret = math.exp(slope * 250) - 1
                res = np.sum(w * (y - (slope * x + intercept))**2)
                tot = np.sum(w * (y - np.mean(y))**2)
                r2 = 1 - res/tot if tot else 0
                score = ann_ret * r2

                # 短期动量
                if len(closes) >= self.config.short_lookback_days + 1:
                    short_ret = closes[-1] / closes[-(self.config.short_lookback_days + 1)] - 1
                    short_ann = (1 + short_ret) ** (250 / self.config.short_lookback_days) - 1
                else:
                    short_ann = -np.inf
                
                # --- 2. 其他指标 ---
                ma = np.mean(closes[-self.config.ma_filter_days:])
                
                # 成交量比 (当日累计成交额 / 近5日均值)
                avg_vol = df['volume'][-self.config.volume_lookback:].mean()
                curr_vol = current_data[code].volume # 当日到目前的累计成交量
                vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 0
                
                # RSI计算与过滤逻辑
                rsi_values = self._calc_rsi_series(closes, self.config.rsi_period)
                passed_rsi = True
                curr_rsi = 0
                if self.config.use_rsi_filter and len(rsi_values) >= self.config.rsi_lookback_days:
                    recent_rsi = rsi_values[-self.config.rsi_lookback_days:]
                    curr_rsi = recent_rsi[-1]
                    if np.any(recent_rsi > self.config.rsi_threshold):
                        ma5 = np.mean(closes[-5:]) if len(closes) >= 5 else curr_price
                        if curr_price < ma5:
                            passed_rsi = False
                
                # 跌幅过滤
                passed_loss = True
                if len(closes) >= 4:
                    rets = closes[-3:] / closes[-4:-1]
                    if np.min(rets) < self.config.loss_limit: passed_loss = False
                
                # 打包结果
                results.append({
                    'code': code,
                    'name': self.dm.get_info(code)['name'],
                    'score': score,
                    'ann_ret': ann_ret,
                    'r2': r2,
                    'short_ann': short_ann,
                    'is_ma_ok': curr_price >= ma,
                    'vol_ratio': vol_ratio,
                    'passed_loss': passed_loss,
                    'passed_rsi': passed_rsi,
                    'rsi': curr_rsi
                })
            except Exception as e:
                continue
                
        return results

    def _calc_rsi_series(self, prices, period=6):
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

    def apply_filters(self, metrics_list):
        """执行级联过滤"""
        res = []
        for m in metrics_list:
            # 动量范围
            if not (self.config.min_score_threshold <= m['score'] <= self.config.max_score_threshold): continue
            # R2
            if self.config.enable_r2_filter and m['r2'] <= self.config.r2_threshold: continue
            # MA
            if self.config.enable_ma_filter and not m['is_ma_ok']: continue
            # Volume
            if self.config.enable_volume_check and m['vol_ratio'] >= self.config.volume_threshold: continue
            # Loss
            if self.config.enable_loss_filter and not m['passed_loss']: continue
            
            res.append(m)
        
        # 按得分排序
        res.sort(key=lambda x: x['score'], reverse=True)
        return res

# ==============================================================================
# 4. RiskManager (风控止损模块)
# ==============================================================================
class RiskManager:
    def __init__(self, config, data_manager):
        self.config = config
        self.dm = data_manager
        self.cooldown_end_date = None
        self.position_highs = {} # ATR跟踪最高价

    def _calculate_atr(self, security, period):
        hist_data = attribute_history(security, period + 20, '1d', ['high', 'low', 'close'])
        if len(hist_data) < period + 1: return 0, False
        high = hist_data['high'].values
        low = hist_data['low'].values
        close = hist_data['close'].values
        tr_values = np.zeros(len(high))
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr_values[i] = max(tr1, tr2, tr3)
        atr_values = np.zeros(len(tr_values))
        for i in range(period, len(tr_values)):
            atr_values[i] = np.mean(tr_values[i-period+1:i+1])
        return atr_values[-1] if len(atr_values) > 0 else 0, True

    def check_stop_loss(self, context, trader):
        """分钟级止损轮询"""
        if self.is_in_cooldown(context): return
        
        current_data = get_current_data()
        for security in list(context.portfolio.positions.keys()):
            pos = context.portfolio.positions[security]
            if pos.total_amount <= 0: continue
            
            price = current_data[security].last_price
            reason = None
            cost = pos.avg_cost
            
            # 1. 固定比例止损
            if self.config.use_fixed_stop_loss and not reason:
                if price <= cost * self.config.fixed_stop_loss_threshold:
                    reason = "固定比例止损"
                    
            # 2. 当日跌幅止损
            if self.config.use_pct_stop_loss and not reason:
                today_open = current_data[security].day_open
                if today_open and price <= today_open * self.config.pct_stop_loss_threshold:
                    reason = "当日跌幅止损"
            
            # 3. ATR动态止损
            if self.config.use_atr_stop_loss and not reason:
                if not (self.config.atr_exclude_defensive and security == self.config.defensive_etf):
                    current_atr, success = self._calculate_atr(security, self.config.atr_period)
                    if success and current_atr > 0:
                        if security not in self.position_highs:
                            self.position_highs[security] = price
                        else:
                            self.position_highs[security] = max(self.position_highs[security], price)
                        
                        if self.config.atr_trailing_stop:
                            stop_price = self.position_highs[security] - self.config.atr_multiplier * current_atr
                        else:
                            stop_price = cost - self.config.atr_multiplier * current_atr
                            
                        if price <= stop_price:
                            reason = f"ATR动态止损({'跟踪' if self.config.atr_trailing_stop else '固定'})"

            if reason:
                log.info(f"🚨 [{reason}] 触发卖出: {security}, 当前价: {price:.3f}, 成本价: {cost:.3f}")
                if trader.execute_trade(security, 0, context):
                    self.position_highs.pop(security, None)
                    self.enter_cooldown(context, trader, reason)

    def is_in_cooldown(self, context):
        if not self.config.sell_cooldown_enabled or not self.cooldown_end_date:
            return False
        return context.current_dt.date() <= self.cooldown_end_date

    def enter_cooldown(self, context, trader, reason):
        if not self.config.sell_cooldown_enabled: return
        self.cooldown_end_date = context.current_dt.date() + pd.Timedelta(days=self.config.sell_cooldown_days)
        # 冷却期切换到避险标的
        trader.clear_all(context, exclude=[self.config.safe_haven_etf])
        trader.execute_trade(self.config.safe_haven_etf, context.portfolio.total_value * 0.99, context)
        log.info(f"🔒 进入冷却期，原因: {reason}，结束日期: {self.cooldown_end_date}")

    def check_exit_cooldown(self, context, trader):
        if self.cooldown_end_date and context.current_dt.date() > self.cooldown_end_date:
            log.info("🔓 冷却期结束，清理避险头寸")
            trader.execute_trade(self.config.safe_haven_etf, 0, context)
            self.cooldown_end_date = None

# ==============================================================================
# 5. StrategyTrader (交易执行模块)
# ==============================================================================
class StrategyTrader:
    def __init__(self, config, data_manager):
        self.config = config
        self.dm = data_manager

    def execute_trade(self, security, target_value, context):
        """智能下单抽象层"""
        curr = get_current_data()[security]
        if curr.paused: return False
        
        # 价格与限价检查
        price = curr.last_price
        if price <= 0: return False
        if price >= curr.high_limit or price <= curr.low_limit: return False
        
        # 计算目标股数 (取整100)
        target_amount = (int(target_value / price) // 100) * 100
        current_amount = context.portfolio.positions[security].total_amount
        diff = target_amount - current_amount
        
        # 最小交易额校验
        if 0 < abs(diff) * price < self.config.min_money: return False
        
        # T+1 可卖股数校验
        if diff < 0:
            diff = -min(abs(diff), context.portfolio.positions[security].closeable_amount)
            
        if diff != 0:
            order(security, diff)
            log.info(f"{'买入' if diff > 0 else '卖出'} {security}，数量 {abs(diff)}")
            return True
        return False

    def clear_all(self, context, exclude=[]):
        for s in list(context.portfolio.positions.keys()):
            if s not in exclude:
                self.execute_trade(s, 0, context)

    def rebalance(self, context, target_list):
        """每日轮动核心逻辑"""
        log.info(f"== 调仓开始，目标列表: {target_list} ==")
        
        # 1. 卖出非目标
        target_set = set(target_list)
        for s in list(context.portfolio.positions.keys()):
            if s not in target_set:
                self.execute_trade(s, 0, context)
                
        # 2. 买入目标
        curr_holds = set(context.portfolio.positions.keys())
        to_buy = [s for s in target_list if s not in curr_holds]
        if not to_buy: return
        
        cash_per = context.portfolio.available_cash / len(to_buy)
        for s in to_buy:
            self.execute_trade(s, cash_per, context)

# ==============================================================================
# 6. Main Orchestration (主控流程)
# ==============================================================================

def initialize(context):
    set_option("avoid_future_data", True)
    set_option("use_real_price", True)
    set_benchmark("510300.XSHG")
    
    # 初始化模块
    g.cfg = Config()
    g.dm = DataManager(g.cfg)
    g.fe = FilterEngine(g.cfg, g.dm)
    g.trader = StrategyTrader(g.cfg, g.dm)
    g.rm = RiskManager(g.cfg, g.dm)
    
    # 定时任务
    run_daily(prepare_day, time='09:05')
    run_daily(trade_rotation, time='13:10')
    
    # 分钟级风控
    for h in range(9, 15):
        for m in range(0, 60):
            t = "%02d:%02d" % (h, m)
            if ('09:30' < t < '11:30') or ('13:00' < t < '14:55'):
                run_daily(minute_check, time=t)

def prepare_day(context):
    """盘前数据准备"""
    # 1. 更新动态池
    dynamic = g.dm.update_dynamic_pool(context)
    # 2. 批量拉取行情
    full_list = list(set(g.cfg.fixed_etf_pool + dynamic))
    g.dm.prepare_batch_data(context, full_list)
    # 3. 检查冷却期退出
    g.rm.check_exit_cooldown(context, g.trader)
    log.info(f"盘前准备完成，当前池子大小: {len(full_list)}")

def trade_rotation(context):
    """每日调仓轮动"""
    if g.rm.is_in_cooldown(context): return
    
    # 1. 计算因子
    pool = list(set(g.cfg.fixed_etf_pool + g.dm.dynamic_pool))
    metrics = g.fe.calculate_metrics_batch(context, pool)
    
    # 2. 执行过滤
    ranked = g.fe.apply_filters(metrics)
    
    # 3. 确定最终目标
    targets = [m['code'] for m in ranked[:g.cfg.holdings_num]]
    if not targets:
        # 触发防御模式
        targets = [g.cfg.defensive_etf]
        
    g.trader.rebalance(context, targets)

def minute_check(context):
    """分钟级风控"""
    g.rm.check_stop_loss(context, g.trader)
