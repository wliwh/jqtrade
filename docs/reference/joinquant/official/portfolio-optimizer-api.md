## 投资组合优化

### 概述

投资组合优化器旨在构建最优投资组合，平衡各种竞争目标（例如最大化收益，最小化风险，风险平价等），同时考虑指定的约束，通过数学优化计算为用户提供最优的投资组合建议。投资组合管理者在设定了投资收益预期、风险预算、相关约束和风险模型之后， 依托优化器的快速计算优势，得到资产配置最优化结果。

由于不同的约束条件、目标函数，会形成不同的优化器，优化器的处理结果依赖用户输入的相关信息，因此投资者对收益率的预期和风险模型本身估计的准确性，都会影响最终的分析结果，再考虑到交易成本等各类因素的影响，所以从用户使用上而言， 没有绝对意义上最好的优化器。对于资产组合优化问题， 我们可以通过使用优化器，进行一个较长时间的回测，测试整个投资过程，在所有组合输入一致的情况下通过策略的绩效对比来看哪一个优化器有更好的表现， 或者更符合自己的需求。

组合优化器支持对股票、基金进行投资优化，支持如下优化模型：

- MinVariance - 组合风险最小化（均值-方差优化）
- MaxProfit - 组合收益最大化
- MaxSharpeRatio - 组合夏普比率最大化
- MinTrackingError - 追踪误差最小化
- RiskParity - 风险平价
- MaxScore - 组合标的打分最大化
- MinScore - 组合标的打分最小化
- MaxFactorValue - 因子值最大化
- MinFactorValue - 因子值最小化
- 自定义约束条件的优化模型

对使用优化器的投资组合管理者来说，只需根据收益预期、风险预算，选择恰当的优化模型，并设定相关的约束限制条件。优化器程序可以基于选定的优化模型，输出优化后的投资权重调整建议。我们会对投资组合优化器的进行持续创新与改进。

#### 示例

下面选出上证50成分股的一部分与选定的ETF基金进行组合构成股票池，设定不同的投资组合优化约束条件，并进行回测，测试投资组合优化器对整个投资的影响。

- **模型1：等权重配置**

![enter image description here](https://image.joinquant.com/8f1b312b9885ef2ca4c9b75900a94937)

- **模型2：组合风险平价；股票的总权重限制为0到90%，ETF的总权重限制为0到10%；每只标的权重不超过10%**

![enter image description here](https://image.joinquant.com/424e55790bf46dda649187015e55042e)

- **模型3：组合风险最小化（最小化组合方差）；组合总权重限制为90%到100%；组合年化收益率目标下限为10%**

![enter image description here](https://image.joinquant.com/6060efcc73e32b9d9a19a616e4451c5c)

- **模型4：'人气指标5日均值'最大化；组合年化收益率目标下限为10%；每只标的权重不超过20%**

![enter image description here](https://image.joinquant.com/43b6125b01af7a7377c2f889dceb9786)

- **模型5：组合夏普比率最大化；每只标的权重不超过10%**

![enter image description here](https://image.joinquant.com/4131d055cea287d383ece0f3c0364218)

回测代码如下, 优化行数API详情见 [portfolio_optimizer - 投资组合优化](#portfolio_optimizer)：

``` python
# 导入函数库
import pandas as pd
from jqdata import *
from jqfactor import Factor
from jqlib.optimizer import *

# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)

    # 过滤掉order系列API产生的比error级别低的log
    log.set_level('order', 'error')

    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003,
                            min_commission=5), type='stock')

    # 优化器设置
    g.optimizer = 2 #设定使用的优化模型
    optimize_model = {
                        1:"模型1：等权重配置",
                        2:"模型2：组合风险平价；股票的总权重限制为0到90%，ETF的总权重限制为0到10%；每只标的权重不超过10%",
                        3:"模型3：组合风险最小化（最小化组合方差）；组合总权重限制为90%到100%；组合年化收益率目标下限为10%",
                        4:"模型4：'人气指标5日均值'最大化；组合年化收益率目标下限为10%；每只标的权重不超过20%",
                        5:"模型5：组合夏普比率最大化；每只标的权重不超过10%"
                      }
    print("优化%s"%(optimize_model[g.optimizer]))

    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
      # 开盘前运行
    run_monthly(before_market_open, monthday=1, time='before_open', reference_security='000300.XSHG')
      # 开盘运行
    run_monthly(market_open, monthday=1, time='open', reference_security='000300.XSHG')

## 开盘前运行函数
def before_market_open(context):
    print('调仓日期：%s'%context.current_dt.date())

    # 选出上证50成分股的一部分与选定的ETF基金进行组合,构成股票池。
    etf = [
        '159902.XSHE',
        '159903.XSHE',
        '510050.XSHG',
        '510880.XSHG',
        '510440.XSHG',
        ]
    g.buy_list = list(get_index_stocks('000016.XSHG')[-15:]) + etf

## 开盘时运行函数
def market_open(context):
    # 讲不在股票池中的股票卖出
    sell_list = set(context.portfolio.positions.keys()) - set(g.buy_list)
    for stock in sell_list:
        order_target_value(stock, 0)

    # 组合优化模型
    if g.optimizer == 1:
        # 模型1：等权重配置
        optimized_weight = pd.Series(data=[1.0/len(g.buy_list)]*len(g.buy_list),
                                    index=g.buy_list)
    elif g.optimizer == 2:
        # 模型2：组合风险平价；股票的总权重限制为0到90%，ETF的总权重限制为0到10%；每只标的权重不超过10%
        optimized_weight = portfolio_optimizer(date=context.previous_date,
                                    securities = g.buy_list,
                                    target = RiskParity(count=250, risk_budget=None),# risk_budget 为 None默认为每只股票贡献相等
                                    constraints = [MarketConstraint('stock', low=0.0, high=0.9),
                                                  MarketConstraint('etf', low=0.0, high=0.1)],
                                    bounds=[Bound(0, 0.1)],
                                    default_port_weight_range=[0., 1.0],
                                    ftol=1e-09,
                                    return_none_if_fail=True)
    elif g.optimizer == 3:
        # 模型3：组合风险最小化（最小化组合方差）；组合总权重限制为90%到100%；组合年化收益率目标下限为10%
        optimized_weight = portfolio_optimizer(date=context.previous_date,
                                    securities = g.buy_list,
                                    target = MinVariance(count=250),
                                    constraints = [WeightConstraint(low=0.9, high=1.0),
                                                   AnnualProfitConstraint(limit=0.1, count=250)],
                                    bounds=[],
                                    default_port_weight_range=[0., 1.0],
                                    ftol=1e-09,
                                    return_none_if_fail=True)
    elif g.optimizer == 4:
        # 模型4：组合标的因子值最大化

        # 定义因子：人气指标5日均值
        class AR(Factor):
            name = 'ar'
            # 每天获取过去五日的数据
            max_window = 5
            # 获取的数据是人气指标
            dependencies = ['AR']
            def calc(self, data):
                return data['AR'].mean()
        # 模型4：'人气指标5日均值'最大化；组合年化收益率目标下限为10%；每只标的权重不超过20%
        optimized_weight = portfolio_optimizer(date=context.previous_date,
                                    securities = g.buy_list,
                                    target = MaxFactorValue(factor=AR, count=1),
                                    constraints = [AnnualProfitConstraint(limit=0.2, count=250)],
                                    bounds=[Bound(0, 0.2)],
                                    default_port_weight_range=[0., 1.0],
                                    ftol=1e-09,
                                    return_none_if_fail=True)
    elif g.optimizer == 5:
        # 模型5：组合夏普比率最大化；每只标的权重不超过10%
        optimized_weight = portfolio_optimizer(date=context.previous_date,
                                    securities = g.buy_list,
                                    target = MaxSharpeRatio(rf=0.0,weight_sum_equal=0.5, count=250),#无风险利率为0，最大化夏普比率需要约束组合权重的和为0.5
                                    constraints = [],
                                    bounds=[Bound(0, 0.1)],
                                    default_port_weight_range=[0., 1.0],
                                    ftol=1e-09,
                                    return_none_if_fail=True)

    # 查看优化结果
    print(optimized_weight)

    # 优化失败，给予警告
    if type(optimized_weight) == type(None):
        print('警告：组合优化失败')
    # 按优化结果，执行调仓操作
    else:
        total_value = context.portfolio.total_value # 获取总资产
        for stock in optimized_weight.keys():
            value = total_value * optimized_weight[stock] # 确定每个标的的权重
            order_target_value(stock, value) # 调整标的至目标权重
```

### portfolio_optimizer - 投资组合优化 <span id="portfolio_optimizer"></span>

``` python
portfolio_optimizer(date, securities, target, constraints, bounds=[Bound(0.0, 1.0)], default_port_weight_range=[0.0, 1.0], ftol=1e-9, return_none_if_fail=True)
```

优化函数, 用于计算在某些约束条件下的最优组合权重

- 参数
  - date: 优化发生的日期，请注意未来函数
  - date: 优化发生的日期，请注意未来函数
  - securities: 股票代码列表
  - target: 优化目标函数，只能选择一个，目标函数详见下方
  - constraints: 限制函数，用以对组合总权重进行限制，可设置一个或多个相同/不同类别的函数，限制函数详见下方
  - bounds: 边界函数，用以对组合中单标的权重进行限制，可设置一个或多个相同/不同类别的函数，边界函数详见下方。如果不填，默认为 Bound(0., 1.)；如果有多个 bound，则一只股票的权重下限取所有 Bound 的最大值，上限取所有 Bound 的最小值
  - default_port_weight_range: 长度为2的列表，默认的组合权重之和的范围，默认值为 \[0.0, 1.0\]。如果限制函数(constraints) 中没有 WeightConstraint 或 WeightEqualConstraint 限制，则会添加 WeightConstraint(low=default_port_weight_range\[0\], high=default_port_weight_range\[1\]) 到 constraints列表中。
  - ftol: 默认为 1e-9，优化函数触发结束的函数值。当求解结果精度不够时可以适当降低 ftol 的值，当求解时间过长时可以适当提高 ftol 值
  - return_none_if_fail: 默认为 True，如果优化失败，当 return_none_if_fail 为 True 时返回 None，为 False 时返回全为 0 的组合权重

> 目标函数(target)

- MiniVariance(count=250) - 组合风险最小化（最小化组合方差）

  ​ 最小化组合方差

  - 参数：

    `count`: 默认为 250，向前取 returns 的天数

  - 示例：

    `target = MiniVariance(count=250)`

- MaxProfit(count=250) - 组合收益最大化

  - 参数：

    `count`: 默认为 250，向前取 returns 的天数

  - 示例：

    `target = MaxProfit(count=250)`

- MaxSharpeRatio(rf=0.0, weight_sum_equal=1.0, count=250) - 组合夏普比率最大化

  - 参数：

    `rf`: 年化无风险利率，默认为 0

    `weight_sum_equal`：组合总权重的值（默认值为1.0），在该权重下进行优化，使得组合的夏普比率最大化

    `count`: 默认为 250，向前取 returns 的天数

  - 示例：

    `target = MaxSharpeRatio(count=250)`

- MinTrackingError(benchmark, count=250) - 追踪误差最小化

  - 参数：

    `benchmark`: 基准的 ticker，例如 `'000300.XSHG'`

    `count`: 默认为 250，向前取 returns 的天数

  - 示例：

    `target = MinTrackingError(benchmark='000300.XSHG', count=250)`

- RiskParity(count=250, risk_budget=None) - 风险平价

  风险平价（Risk Parity）是对投资组合中不同资产分配相同的风险权重的一种资产配置理念，资产配置的风险平价方法允许投资者针对具体的风险水平，并在整个投资组合中平均分配风险，以实现每个投资者的最佳投资组合多元化。

  - 参数：

    `count`: 默认为 250，向前取 returns 的天数

    `risk_budget`: pandas.Series，风险预算，股票的每只对组合风险的贡献，risk_budget 为 None默认为每只股票贡献相等

  - 示例：

    `target = RiskParity(count=250, risk_budget=pd.Series([0.3, 0.3, 0.4], index=['000001.XSHE', '000002.XSHE', '000005.XSHE']))`

- MaxScore(scores) - 打分最大化

  在满足约束条件的情况下，给予打分高的标的更高权重（前提假设：用户已知晓打分大的标的表现更好）。

  如有经过因子分析检验，打分越高越有正向效果的\[A,B,C\] 三只标的，打分分别为 \[3,2,1\] , 约束条件为年化波动率小于15%。 如果组合全部配置A可获得最高的收益，但波动率大于15%，不满足约束条件；通过优化器优化，则会配置一定比例的B与C，在满足波动率小于15%的条件下，获得最高收益。

  - 参数：

    `scores`: pandas.Series，每只股票的打分

  - 示例：

    `target = MaxScore(scores=pd.Series([0.1, 0.2, 0.3], index=['000001.XSHE', '000002.XSHE', '000005.XSHE']))`

- MinScore(scores) - 打分最小化

  在满足约束条件的情况下，给予打分低的标的更高权重（前提假设：用户已知晓打分小的标的表现更好）。可参考 \[打分最大化\] 的示例说明。

  - 参数：

    `scores`: pandas.Series，每只股票的打分

  - 示例：

    `target = MinScore(scores=pd.Series([0.1, 0.2, 0.3], index=['000001.XSHE', '000002.XSHE', '000005.XSHE']))`

- MaxFactorValue(factor, count=1) - 因子值最大化

  在满足约束条件的情况下，给予因子值大的标的更高权重（前提假设：用户已知晓因子值大的标的表现更好）。可参考 \[打分最大化\] 的示例说明。

  - 参数：

    `factor`: Factor 的子类

    `count`: 默认为 1，用过去几天的因子取平均

  - 示例：

    ``` python
    from jqfactor import Factor
    # 定义因子：人气指标5日均值
    class AR(Factor):
      name = 'AR_M5'
      # 每天获取过去五日的数据
      max_window = 5
      # 获取的数据是人气指标
      dependencies = ['AR']
      def calc(self, data):
          return data['AR'].mean()

    target = MaxFactorValue(factor=AR, count=1)
    ```

- MinFactorValue(factor, count=1) - 因子值最小化

  在满足约束条件的情况下，给予因子值小的标的更高权重（前提假设：用户已知晓因子值小的标的表现更好）。可参考 \[打分最大化\] 的示例说明。

  - 参数：

    `factor`: Factor 的子类

    `count`: 默认为 1，用过去几天的因子取平均

  - 示例：

    参考 \[因子值最大化\] 的示例

> 限制函数(constraints)

- WeightConstraint(low=0.0, high=1.0) - 组合总权重限制

  设定组合优化结果总权重的上下限，即优化结果的总权重在此范围间。

  - 参数：

    `low`: 默认为 0.0，权重下限

    `high`: 默认为 1.0，权重上限

  - 示例：

    `constraint = WeightConstraint(low=0.5, high=0.9)`

- WeightEqualConstraint(limit=1.0) - 组合总权重和限制

  设定组合优化结果总权重的和，即优化结果的总权重等于该值。

  - 参数：

    `limit`: 默认为 1.0，组合权重等式约束

  - 示例：

    `constraint = WeightEqualConstraint(limit=0.5)`

- AnnualStdConstraint(limit, count=250) - 组合年化收益率标准差限制

  - 参数：

    `limit`: 标准差上限

    `count`: 默认为 250，向前取 returns 的天数

  - 示例：

    `constraint = AnnualStdConstraint(limit=0.15, count=250)`

- AnnualProfitConstraint(limit, count=250) - 组合年化收益率预期限制

  - 参数：

    `limit`: 收益率预期下限

    `count`: 默认为 250，向前取 returns 的天数

  - 示例：

    `constraint = AnnualProfitConstraint(limit=0.1, count=250)`

- IndustryConstraint(industry_code, low=0.0, high=1.0) - 组合行业权重限制

  - 参数：

    `industry_code`: 单一或多个[行业代码](/data/dict/plateData)，如 `'HY001'`。如果为多个行业代码的列表，则表示所有属于列表中行业的股票的权重之和满足限制条件

    `low`: 默认为 0.0，行业权重下限

    `high`: 默认为 1.0，行业权重上限

  - 示例：

    `constraint = IndustryConstraint(['HY007'], low=0.0, high=0.2)`

- IndustriesConstraint(industry_code, low=0.0, high=1.0)- 组合行业分类权重限制

  - 参数：

    `industry_code`: [行业分类代码](/help/api/help?name=api#get_industries)，如 `'jq_l1'`。表示这个行业分类下的所有行业都需满足权重限制

    `low`: 默认为 0.0，行业权重下限

    `high`: 默认为 1.0，行业权重上限

  - 示例：

    `constraint = IndustriesConstraint('jq_l1', low=0.0, high=0.2)`

- MarketConstraint(market_type, low=0.0, high=1.0) - 组合市场权重限制

  - 参数：

    `market_type`: ('stock', 'index', 'fund', 'futures', 'etf', 'lof', 'fja', 'fjb', 'open_fund', 'bond_fund', 'stock_fund', 'QDII_fund', 'money_market_fund', 'mixture_fund') 中的一种

    `low`: 默认为 0.0，市场权重下限

    `high`: 默认为 1.0，市场权重上限

  - 示例：

    `constraint = MarketConstraint('stock', low=0.0, high=0.2)`

- ExposureConstraint(factor, low=0.0, high=1.0, count=1) - 因子暴露限制

  - 参数：

    `factor`: Factor 的子类

    `low`: 默认为 0.0，因子暴露度下限

    `high`: 默认为 1.0，因子暴露度上限

    `count`: 默认为 1，用过去几天的因子取平均

  - 示例：

    ``` python
    from jqfactor import Factor
    # 定义因子：人气指标5日均值
    class AR(Factor):
      name = 'AR_M5'
      # 每天获取过去五日的数据
      max_window = 5
      # 获取的数据是人气指标
      dependencies = ['AR']
      def calc(self, data):
          return data['AR'].mean()

    constraint = ExposureConstraint(AR, low=0.0, high=10.0, count=1)
    ```

> 边界函数(bounds)

- Bound(low=0.0, high=1.0) - 每只标的的权重限制

  - 参数：

    `low`: 默认为 0.0，每只标的的权重下限

    `high`: 默认为 1.0，每只标的的权重上限

  - 示例：

    `bound = Bound(low=0.0, high=0.1)`

- IndustryBound(industry_code, low=0.0, high=1.0) - 属于某一行业的每只股票的权重限制

  - 参数：

    `industry_code`: 单一[行业代码](https://www.joinquant.com/data/dict/plateData)或者行业代码的列表。

    `low`: 默认为 0.0，如果一只股票属于所选行业，则股票权重的下限为 low，否则下限为 0

    `high`: 默认为 1.0，如果一只股票属于所选行业，则股票权重的上限为 high，否则上限为 1

  - 示例：

    `bound = IndustryBound(['HY001', 'HY007'], low=0.0, high=0.05)`

## 概述

投资组合优化器旨在构建最优投资组合，平衡各种竞争目标（例如最大化收益，最小化风险，风险平价等），同时考虑指定的约束，通过数学优化计算为用户提供最优的投资组合建议。投资组合管理者在设定了投资收益预期、风险预算、相关约束和风险模型之后， 依托优化器的快速计算优势，得到资产配置最优化结果。

由于不同的约束条件、目标函数，会形成不同的优化器，优化器的处理结果依赖用户输入的相关信息，因此投资者对收益率的预期和风险模型本身估计的准确性，都会影响最终的分析结果，再考虑到交易成本等各类因素的影响，所以从用户使用上而言， 没有绝对意义上最好的优化器。对于资产组合优化问题， 我们可以通过使用优化器，进行一个较长时间的回测，测试整个投资过程，在所有组合输入一致的情况下通过策略的绩效对比来看哪一个优化器有更好的表现， 或者更符合自己的需求。

组合优化器支持对股票、基金进行投资优化，支持如下优化模型：

- MinVariance - 组合风险最小化（均值-方差优化）
- MaxProfit - 组合收益最大化
- MaxSharpeRatio - 组合夏普比率最大化
- MinTrackingError - 追踪误差最小化
- RiskParity - 风险平价
- MaxScore - 组合标的打分最大化
- MinScore - 组合标的打分最小化
- MaxFactorValue - 因子值最大化
- MinFactorValue - 因子值最小化
- 自定义约束条件的优化模型

对使用优化器的投资组合管理者来说，只需根据收益预期、风险预算，选择恰当的优化模型，并设定相关的约束限制条件。优化器程序可以基于选定的优化模型，输出优化后的投资权重调整建议。我们会对投资组合优化器的进行持续创新与改进。

#### 示例

下面选出上证50成分股的一部分与选定的ETF基金进行组合构成股票池，设定不同的投资组合优化约束条件，并进行回测，测试投资组合优化器对整个投资的影响。

- **模型1：等权重配置**

![enter image description here](http://img2.ph.126.net/aHl652neKVoidzDJ_8OXXA==/6597397421124292478.png)

- **模型2：组合风险平价；股票的总权重限制为0到90%，ETF的总权重限制为0到10%；每只标的权重不超过10%**

![enter image description here](http://img0.ph.126.net/ZjIA6o8tMusvQ-bmLTkbcQ==/6597885604286753274.png)

- **模型3：组合风险最小化（最小化组合方差）；组合总权重限制为90%到100%；组合年化收益率目标下限为10%**

![enter image description here](http://img1.ph.126.net/CARmKcPUlb0QRXDJ97g-Kg==/6631669198563844627.png)

- **模型4：'人气指标5日均值'最大化；组合年化收益率目标下限为10%；每只标的权重不超过20%**

![enter image description here](http://img2.ph.126.net/JYPUqEhuE58m3fAt1-nr5Q==/3263983830037255451.png)

- **模型5：组合夏普比率最大化；每只标的权重不超过10%**

![enter image description here](http://img0.ph.126.net/94ybHamzshdjDaDXUKH0vQ==/6599295178193111856.png)

回测代码如下, 优化行数API详情见 [portfolio_optimizer - 投资组合优化](#portfolio_optimizer)：

``` python
# 导入函数库
import pandas as pd
from jqdata import *
from jqfactor import Factor
from jqlib.optimizer import *

# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)

    # 过滤掉order系列API产生的比error级别低的log
    log.set_level('order', 'error')

    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003,
                            min_commission=5), type='stock')

    # 优化器设置
    g.optimizer = 2 #设定使用的优化模型
    optimize_model = {
                        1:"模型1：等权重配置",
                        2:"模型2：组合风险平价；股票的总权重限制为0到90%，ETF的总权重限制为0到10%；每只标的权重不超过10%",
                        3:"模型3：组合风险最小化（最小化组合方差）；组合总权重限制为90%到100%；组合年化收益率目标下限为10%",
                        4:"模型4：'人气指标5日均值'最大化；组合年化收益率目标下限为10%；每只标的权重不超过20%",
                        5:"模型5：组合夏普比率最大化；每只标的权重不超过10%"
                      }
    print("优化%s"%(optimize_model[g.optimizer]))

    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
      # 开盘前运行
    run_monthly(before_market_open, monthday=1, time='before_open', reference_security='000300.XSHG')
      # 开盘运行
    run_monthly(market_open, monthday=1, time='open', reference_security='000300.XSHG')

## 开盘前运行函数
def before_market_open(context):
    print('调仓日期：%s'%context.current_dt.date())

    # 选出上证50成分股的一部分与选定的ETF基金进行组合,构成股票池。
    etf = [
        '159902.XSHE',
        '159903.XSHE',
        '510050.XSHG',
        '510880.XSHG',
        '510440.XSHG',
        ]
    g.buy_list = list(get_index_stocks('000016.XSHG')[-15:]) + etf

## 开盘时运行函数
def market_open(context):
    # 讲不在股票池中的股票卖出
    sell_list = set(context.portfolio.positions.keys()) - set(g.buy_list)
    for stock in sell_list:
        order_target_value(stock, 0)

    # 组合优化模型
    if g.optimizer == 1:
        # 模型1：等权重配置
        optimized_weight = pd.Series(data=[1.0/len(g.buy_list)]*len(g.buy_list),
                                    index=g.buy_list)
    elif g.optimizer == 2:
        # 模型2：组合风险平价；股票的总权重限制为0到90%，ETF的总权重限制为0到10%；每只标的权重不超过10%
        optimized_weight = portfolio_optimizer(date=context.previous_date,
                                    securities = g.buy_list,
                                    target = RiskParity(count=250, risk_budget=None),# risk_budget 为 None默认为每只股票贡献相等
                                    constraints = [MarketConstraint('stock', low=0.0, high=0.9),
                                                  MarketConstraint('etf', low=0.0, high=0.1)],
                                    bounds=[Bound(0, 0.1)],
                                    default_port_weight_range=[0., 1.0],
                                    ftol=1e-09,
                                    return_none_if_fail=True)
    elif g.optimizer == 3:
        # 模型3：组合风险最小化（最小化组合方差）；组合总权重限制为90%到100%；组合年化收益率目标下限为10%
        optimized_weight = portfolio_optimizer(date=context.previous_date,
                                    securities = g.buy_list,
                                    target = MinVariance(count=250),
                                    constraints = [WeightConstraint(low=0.9, high=1.0),
                                                   AnnualProfitConstraint(limit=0.1, count=250)],
                                    bounds=[],
                                    default_port_weight_range=[0., 1.0],
                                    ftol=1e-09,
                                    return_none_if_fail=True)
    elif g.optimizer == 4:
        # 模型4：组合标的因子值最大化

        # 定义因子：人气指标5日均值
        class AR(Factor):
            name = 'ar'
            # 每天获取过去五日的数据
            max_window = 5
            # 获取的数据是人气指标
            dependencies = ['AR']
            def calc(self, data):
                return data['AR'].mean()
        # 模型4：'人气指标5日均值'最大化；组合年化收益率目标下限为10%；每只标的权重不超过20%
        optimized_weight = portfolio_optimizer(date=context.previous_date,
                                    securities = g.buy_list,
                                    target = MaxFactorValue(factor=AR, count=1),
                                    constraints = [AnnualProfitConstraint(limit=0.2, count=250)],
                                    bounds=[Bound(0, 0.2)],
                                    default_port_weight_range=[0., 1.0],
                                    ftol=1e-09,
                                    return_none_if_fail=True)
    elif g.optimizer == 5:
        # 模型5：组合夏普比率最大化；每只标的权重不超过10%
        optimized_weight = portfolio_optimizer(date=context.previous_date,
                                    securities = g.buy_list,
                                    target = MaxSharpeRatio(rf=0.0,weight_sum_equal=0.5, count=250),#无风险利率为0，最大化夏普比率需要约束组合权重的和为0.5
                                    constraints = [],
                                    bounds=[Bound(0, 0.1)],
                                    default_port_weight_range=[0., 1.0],
                                    ftol=1e-09,
                                    return_none_if_fail=True)

    # 查看优化结果
    print(optimized_weight)

    # 优化失败，给予警告
    if type(optimized_weight) == type(None):
        print('警告：组合优化失败')
    # 按优化结果，执行调仓操作
    else:
        total_value = context.portfolio.total_value # 获取总资产
        for stock in optimized_weight.keys():
            value = total_value * optimized_weight[stock] # 确定每个标的的权重
            order_target_value(stock, value) # 调整标的至目标权重
```

## 优化

##### portfolio_optimizer - 投资组合优化 <span id="portfolio_optimizer"></span>

``` python
portfolio_optimizer(date, securities, target, constraints, bounds=[Bound(0.0, 1.0)], default_port_weight_range=[0.0, 1.0], ftol=1e-9, return_none_if_fail=True)
```

优化函数, 用于计算在某些约束条件下的最优组合权重

- 参数
  - date: 优化发生的日期，请注意未来函数
  - date: 优化发生的日期，请注意未来函数
  - securities: 股票代码列表
  - target: 优化目标函数，只能选择一个，目标函数详见下方
  - constraints: 限制函数，用以对组合总权重进行限制，可设置一个或多个相同/不同类别的函数，限制函数详见下方
  - bounds: 边界函数，用以对组合中单标的权重进行限制，可设置一个或多个相同/不同类别的函数，边界函数详见下方。如果不填，默认为 Bound(0., 1.)；如果有多个 bound，则一只股票的权重下限取所有 Bound 的最大值，上限取所有 Bound 的最小值
  - default_port_weight_range: 长度为2的列表，默认的组合权重之和的范围，默认值为 \[0.0, 1.0\]。如果限制函数(constraints) 中没有 WeightConstraint 或 WeightEqualConstraint 限制，则会添加 WeightConstraint(low=default_port_weight_range\[0\], high=default_port_weight_range\[1\]) 到 constraints列表中。
  - ftol: 默认为 1e-9，优化函数触发结束的函数值。当求解结果精度不够时可以适当降低 ftol 的值，当求解时间过长时可以适当提高 ftol 值
  - return_none_if_fail: 默认为 True，如果优化失败，当 return_none_if_fail 为 True 时返回 None，为 False 时返回全为 0 的组合权重

#### 相关参数

<div class="section">

<div class="header">

参数名称 描述

</div>

<div class="body">

<div class="group">

目标函数(target) 优化目标函数，只能选择一个

\- MiniVariance(count=250) - 组合风险最小化（最小化组合方差） ​ 最小化组合方差 - 参数： count: 默认为 250，向前取 returns 的天数 - 示例： target = MiniVariance(count=250) - MaxProfit(count=250) - 组合收益最大化 - 参数： count: 默认为 250，向前取 returns 的天数 - 示例： target = MaxProfit(count=250) - MaxSharpeRatio(rf=0.0, weight_sum_equal=1.0, count=250) - 组合夏普比率最大化 - 参数： rf: 年化无风险利率，默认为 0 weight_sum_equal：组合总权重的值（默认值为1.0），在该权重下进行优化，使得组合的夏普比率最大化 count: 默认为 250，向前取 returns 的天数 - 示例： target = MaxSharpeRatio(count=250) - MinTrackingError(benchmark, count=250) - 追踪误差最小化 - 参数： benchmark: 基准的 ticker，例如 '000300.XSHG' count: 默认为 250，向前取 returns 的天数 - 示例： target = MinTrackingError(benchmark='000300.XSHG', count=250) - RiskParity(count=250, risk_budget=None) - 风险平价 风险平价（Risk Parity）是对投资组合中不同资产分配相同的风险权重的一种资产配置理念，资产配置的风险平价方法允许投资者针对具体的风险水平，并在整个投资组合中平均分配风险，以实现每个投资者的最佳投资组合多元化。 - 参数： count: 默认为 250，向前取 returns 的天数 risk_budget: pandas.Series，风险预算，股票的每只对组合风险的贡献，risk_budget 为 None默认为每只股票贡献相等 - 示例： target = RiskParity(count=250, risk_budget=pd.Series(\[0.3, 0.3, 0.4\], index=\['000001.XSHE', '000002.XSHE', '000005.XSHE'\])) - MaxScore(scores) - 打分最大化 在满足约束条件的情况下，给予打分高的标的更高权重（前提假设：用户已知晓打分大的标的表现更好）。 如有经过因子分析检验，打分越高越有正向效果的\[A,B,C\] 三只标的，打分分别为 \[3,2,1\] , 约束条件为年化波动率小于15%。 如果组合全部配置A可获得最高的收益，但波动率大于15%，不满足约束条件；通过优化器优化，则会配置一定比例的B与C，在满足波动率小于15%的条件下，获得最高收益。 - 参数： scores: pandas.Series，每只股票的打分 - 示例： target = MaxScore(scores=pd.Series(\[0.1, 0.2, 0.3\], index=\['000001.XSHE', '000002.XSHE', '000005.XSHE'\])) - MinScore(scores) - 打分最小化 在满足约束条件的情况下，给予打分低的标的更高权重（前提假设：用户已知晓打分小的标的表现更好）。可参考 \[打分最大化\] 的示例说明。 - 参数： scores: pandas.Series，每只股票的打分 - 示例： target = MinScore(scores=pd.Series(\[0.1, 0.2, 0.3\], index=\['000001.XSHE', '000002.XSHE', '000005.XSHE'\])) - MaxFactorValue(factor, count=1) - 因子值最大化 在满足约束条件的情况下，给予因子值大的标的更高权重（前提假设：用户已知晓因子值大的标的表现更好）。可参考 \[打分最大化\] 的示例说明。 - 参数： factor: Factor 的子类 count: 默认为 1，用过去几天的因子取平均 - 示例： \`\`\`python from jqfactor import Factor \# 定义因子：人气指标5日均值 class AR(Factor): name = 'AR_M5' \# 每天获取过去五日的数据 max_window = 5 \# 获取的数据是人气指标 dependencies = \['AR'\] def calc(self, data): return data\['AR'\].mean() target = MaxFactorValue(factor=AR, count=1) \`\`\` - MinFactorValue(factor, count=1) - 因子值最小化 在满足约束条件的情况下，给予因子值小的标的更高权重（前提假设：用户已知晓因子值小的标的表现更好）。可参考 \[打分最大化\] 的示例说明。 - 参数： factor: Factor 的子类 count: 默认为 1，用过去几天的因子取平均 - 示例： 参考 \[因子值最大化\] 的示例

</div>

<div class="group">

限制函数(constraints) 用以对组合总权重进行限制，可设置一个或多个相同/不同类别的函数

\- WeightConstraint(low=0.0, high=1.0) - 组合总权重限制 设定组合优化结果总权重的上下限，即优化结果的总权重在此范围间。 - 参数： low: 默认为 0.0，权重下限 high: 默认为 1.0，权重上限 - 示例： constraint = WeightConstraint(low=0.5, high=0.9) - WeightEqualConstraint(limit=1.0) - 组合总权重和限制 设定组合优化结果总权重的和，即优化结果的总权重等于该值。 - 参数： limit: 默认为 1.0，组合权重等式约束 - 示例： constraint = WeightEqualConstraint(limit=0.5) - AnnualStdConstraint(limit, count=250) - 组合年化收益率标准差限制 - 参数： limit: 标准差上限 count: 默认为 250，向前取 returns 的天数 - 示例： constraint = AnnualStdConstraint(limit=0.15, count=250) - AnnualProfitConstraint(limit, count=250) - 组合年化收益率预期限制 - 参数： limit: 收益率预期下限 count: 默认为 250，向前取 returns 的天数 - 示例： constraint = AnnualProfitConstraint(limit=0.1, count=250) - IndustryConstraint(industry_code, low=0.0, high=1.0) - 组合行业权重限制 - 参数： industry_code: 单一或多个\[行业代码\](/data/dict/plateData)，如 'HY001'。如果为多个行业代码的列表，则表示所有属于列表中行业的股票的权重之和满足限制条件 low: 默认为 0.0，行业权重下限 high: 默认为 1.0，行业权重上限 - 示例： constraint = IndustryConstraint(\['HY007'\], low=0.0, high=0.2) - IndustriesConstraint(industry_code, low=0.0, high=1.0)- 组合行业分类权重限制 - 参数： industry_code: \[行业分类代码\](/help/api/help?name=api#get_industries)，如 'jq_l1'。表示这个行业分类下的所有行业都需满足权重限制 low: 默认为 0.0，行业权重下限 high: 默认为 1.0，行业权重上限 - 示例： constraint = IndustriesConstraint('jq_l1', low=0.0, high=0.2) - MarketConstraint(market_type, low=0.0, high=1.0) - 组合市场权重限制 - 参数： market_type: ('stock', 'index', 'fund', 'futures', 'etf', 'lof', 'fja', 'fjb', 'open_fund', 'bond_fund', 'stock_fund', 'QDII_fund', 'money_market_fund', 'mixture_fund') 中的一种 low: 默认为 0.0，市场权重下限 high: 默认为 1.0，市场权重上限 - 示例： constraint = MarketConstraint('stock', low=0.0, high=0.2) - ExposureConstraint(factor, low=0.0, high=1.0, count=1) - 因子暴露限制 - 参数： factor: Factor 的子类 low: 默认为 0.0，因子暴露度下限 high: 默认为 1.0，因子暴露度上限 count: 默认为 1，用过去几天的因子取平均 - 示例： \`\`\`python from jqfactor import Factor \# 定义因子：人气指标5日均值 class AR(Factor): name = 'AR_M5' \# 每天获取过去五日的数据 max_window = 5 \# 获取的数据是人气指标 dependencies = \['AR'\] def calc(self, data): return data\['AR'\].mean() constraint = ExposureConstraint(AR, low=0.0, high=10.0, count=1) \`\`\`

</div>

<div class="group">

边界函数(bounds) 用以对组合中单标的权重进行限制，可设置一个或多个相同/不同类别的函数

\- Bound(low=0.0, high=1.0) - 每只标的的权重限制 - 参数： low: 默认为 0.0，每只标的的权重下限 high: 默认为 1.0，每只标的的权重上限 - 示例： bound = Bound(low=0.0, high=0.1) - IndustryBound(industry_code, low=0.0, high=1.0) - 属于某一行业的每只股票的权重限制 - 参数： industry_code: 单一\[行业代码\](https://www.joinquant.com/data/dict/plateData)或者行业代码的列表。 low: 默认为 0.0，如果一只股票属于所选行业，则股票权重的下限为 low，否则下限为 0 high: 默认为 1.0，如果一只股票属于所选行业，则股票权重的上限为 high，否则上限为 1 - 示例： bound = IndustryBound(\['HY001', 'HY007'\], low=0.0, high=0.05)

</div>

<div class="group">

示例代码 给了五个应用示例，修改参数即可生效

``` python
# 导入函数库
import pandas as pd
from jqdata import *
from jqfactor import Factor
from jqlib.optimizer import *

# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)

    # 过滤掉order系列API产生的比error级别低的log
    log.set_level('order', 'error')

    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003,
                            min_commission=5), type='stock')

    # 优化器设置
    g.optimizer = 2 #设定使用的优化模型
    optimize_model = {
                        1:"模型1：等权重配置",
                        2:"模型2：组合风险平价；股票的总权重限制为0到90%，ETF的总权重限制为0到10%；每只标的权重不超过10%",
                        3:"模型3：组合风险最小化（最小化组合方差）；组合总权重限制为90%到100%；组合年化收益率目标下限为10%",
                        4:"模型4：'人气指标5日均值'最大化；组合年化收益率目标下限为10%；每只标的权重不超过20%",
                        5:"模型5：组合夏普比率最大化；每只标的权重不超过10%"
                      }
    print("优化%s"%(optimize_model[g.optimizer]))

    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
      # 开盘前运行
    run_monthly(before_market_open, monthday=1, time='before_open', reference_security='000300.XSHG')
      # 开盘运行
    run_monthly(market_open, monthday=1, time='open', reference_security='000300.XSHG')

## 开盘前运行函数
def before_market_open(context):
    print('调仓日期：%s'%context.current_dt.date())

    # 选出上证50成分股的一部分与选定的ETF基金进行组合,构成股票池。
    etf = [
        '159902.XSHE',
        '159903.XSHE',
        '510050.XSHG',
        '510880.XSHG',
        '510440.XSHG',
        ]
    g.buy_list = list(get_index_stocks('000016.XSHG')[-15:]) + etf

## 开盘时运行函数
def market_open(context):
    # 讲不在股票池中的股票卖出
    sell_list = set(context.portfolio.positions.keys()) - set(g.buy_list)
    for stock in sell_list:
        order_target_value(stock, 0)

    # 组合优化模型
    if g.optimizer == 1:
        # 模型1：等权重配置
        optimized_weight = pd.Series(data=[1.0/len(g.buy_list)]*len(g.buy_list),
                                    index=g.buy_list)
    elif g.optimizer == 2:
        # 模型2：组合风险平价；股票的总权重限制为0到90%，ETF的总权重限制为0到10%；每只标的权重不超过10%
        optimized_weight = portfolio_optimizer(date=context.previous_date,
                                    securities = g.buy_list,
                                    target = RiskParity(count=250, risk_budget=None),# risk_budget 为 None默认为每只股票贡献相等
                                    constraints = [MarketConstraint('stock', low=0.0, high=0.9),
                                                  MarketConstraint('etf', low=0.0, high=0.1)],
                                    bounds=[Bound(0, 0.1)],
                                    default_port_weight_range=[0., 1.0],
                                    ftol=1e-09,
                                    return_none_if_fail=True)
    elif g.optimizer == 3:
        # 模型3：组合风险最小化（最小化组合方差）；组合总权重限制为90%到100%；组合年化收益率目标下限为10%
        optimized_weight = portfolio_optimizer(date=context.previous_date,
                                    securities = g.buy_list,
                                    target = MinVariance(count=250),
                                    constraints = [WeightConstraint(low=0.9, high=1.0),
                                                   AnnualProfitConstraint(limit=0.1, count=250)],
                                    bounds=[],
                                    default_port_weight_range=[0., 1.0],
                                    ftol=1e-09,
                                    return_none_if_fail=True)
    elif g.optimizer == 4:
        # 模型4：组合标的因子值最大化

        # 定义因子：人气指标5日均值
        class AR(Factor):
            name = 'ar'
            # 每天获取过去五日的数据
            max_window = 5
            # 获取的数据是人气指标
            dependencies = ['AR']
            def calc(self, data):
                return data['AR'].mean()
        # 模型4：'人气指标5日均值'最大化；组合年化收益率目标下限为10%；每只标的权重不超过20%
        optimized_weight = portfolio_optimizer(date=context.previous_date,
                                    securities = g.buy_list,
                                    target = MaxFactorValue(factor=AR, count=1),
                                    constraints = [AnnualProfitConstraint(limit=0.2, count=250)],
                                    bounds=[Bound(0, 0.2)],
                                    default_port_weight_range=[0., 1.0],
                                    ftol=1e-09,
                                    return_none_if_fail=True)
    elif g.optimizer == 5:
        # 模型5：组合夏普比率最大化；每只标的权重不超过10%
        optimized_weight = portfolio_optimizer(date=context.previous_date,
                                    securities = g.buy_list,
                                    target = MaxSharpeRatio(rf=0.0,weight_sum_equal=0.5, count=250),#无风险利率为0，最大化夏普比率需要约束组合权重的和为0.5
                                    constraints = [],
                                    bounds=[Bound(0, 0.1)],
                                    default_port_weight_range=[0., 1.0],
                                    ftol=1e-09,
                                    return_none_if_fail=True)

    # 查看优化结果
    print(optimized_weight)

    # 优化失败，给予警告
    if type(optimized_weight) == type(None):
        print('警告：组合优化失败')
    # 按优化结果，执行调仓操作
    else:
        total_value = context.portfolio.total_value # 获取总资产
        for stock in optimized_weight.keys():
            value = total_value * optimized_weight[stock] # 确定每个标的的权重
            order_target_value(stock, value) # 调整标的至目标权重
```

</div>

</div>

</div>
