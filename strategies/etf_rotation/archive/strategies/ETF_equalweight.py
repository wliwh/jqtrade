# 等权买入持有策略 (Equal-Weight Buy & Hold)
# 买入给定 ETF 池中的所有资产，按等权分配资金，并按指定频率再平衡。
# 不考虑交易成本和滑点。

# ==================== 可配置参数 ====================
EXECUTION_ETF_POOLS_PLACEHOLDER = ['518880.XSHG','513100.XSHG']
EXECUTION_FREQ_PLACEHOLDER = 'yearly'
EXECUTION_TIME_PLACEHOLDER  = '9:30'
# ====================================================

def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option("avoid_future_data", True)
    set_slippage(FixedSlippage(0.000))
    set_order_cost(OrderCost(open_tax=0, close_tax=0,
                             open_commission=0, close_commission=0,
                             close_today_commission=0, min_commission=0), type='fund')
    log.set_level('system', 'error')

    # 参数
    g.etf_pool = EXECUTION_ETF_POOLS_PLACEHOLDER
    g.etf_names = {k:get_security_info(k).display_name for k in EXECUTION_ETF_POOLS_PLACEHOLDER}
    g.rebalance_freq = EXECUTION_FREQ_PLACEHOLDER   # 'daily' / 'monthly' / 'yearly'
    g.execution_time = EXECUTION_TIME_PLACEHOLDER 

    # 状态跟踪
    g.last_rebalance_month = None
    g.last_rebalance_year = None
    g.initialized = False  # 是否已完成首次建仓

    run_daily(rebalance, g.execution_time)


def should_rebalance(context):
    """判断今天是否需要再平衡"""
    freq = g.rebalance_freq
    today = context.current_dt

    # 首次建仓，无论频率都必须执行
    if not g.initialized:
        return True
    if len(g.etf_pool) <= 1:
        return False

    if freq == 'daily':
        return True

    elif freq == 'monthly':
        current_month = (today.year, today.month)
        if g.last_rebalance_month != current_month:
            return True
        return False

    elif freq == 'yearly':
        current_year = today.year
        if g.last_rebalance_year != current_year:
            return True
        return False

    else:
        log.error('未知的再平衡频率: %s' % freq)
        return False


def rebalance(context):
    """执行等权再平衡"""
    if not should_rebalance(context):
        return

    today = context.current_dt
    pool = g.etf_pool
    n = len(pool)

    if n == 0:
        log.warn('ETF 池为空，跳过再平衡')
        return

    total_value = context.portfolio.total_value
    target_value = total_value / n

    # 卖出不在池中的持仓（防御性，正常不应发生）
    for etf in list(context.portfolio.positions):
        if etf not in pool:
            order_target_value(etf, 0)
            print('[卖出] %s (不在池中)' % etf)

    # 先卖后买：先处理需要减仓的，再处理需要加仓的，以释放资金
    # 仅持有一只时直接设置目标值即可
    for etf in pool:
        current_value = context.portfolio.positions[etf].value if etf in context.portfolio.positions else 0
        if current_value > target_value:
            order_target_value(etf, target_value)
            print('[减仓] %s -> %.0f' % (etf, target_value))

    for etf in pool:
        current_value = context.portfolio.positions[etf].value if etf in context.portfolio.positions else 0
        if current_value < target_value:
            order_target_value(etf, target_value)
            dname = g.etf_names[etf]
            dname = f'({dname})' if dname else ''
            print('[加仓/建仓] %s%s -> %.0f' % (etf, dname, target_value))

    # 更新状态
    g.initialized = True
    g.last_rebalance_month = (today.year, today.month)
    g.last_rebalance_year = today.year
    print('[再平衡完成] %s | 频率=%s | 池=%d只 | 每只目标=%.0f\n' % (
        today.strftime('%Y-%m-%d'), g.rebalance_freq, n, target_value))
