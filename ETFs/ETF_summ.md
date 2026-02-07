## 一、策略逻辑概述

1. 原版 (wy03)
    *   定位：基准策略，纯粹的满仓轮动，同时也作为其他策略的修改基准。
    *   核心逻辑：基于收盘价对数的线性回归斜率打分，选取得分最高的一只 ETF。
    *   交易与风控：每日 09:30 调仓，无择时与风控。
2. 长短期动量结合版 (long)
    *   定位：改进版，引入反转因子与差异化风控。
    *   核心逻辑：“短期动量 - 长期反转”。结合 25 日动量（追涨）与 200 日反转（防高位接盘）。
    *   交易与风控：
        *   极值过滤：市场极度分化（过热，分差>15）或无主线（过冷，分差<0.1）时空仓。
        *   ETF RSRS 过滤：即便排名第一，若处于顶部风险区则剔除不买。
3. 综合择时版 (yj15)
    *   定位：完整交易系统，引入大盘择时总开关与盘中风控。
    *   核心逻辑：乖离率动量。基于90日乖离率的斜率打分，并引入持仓加分（降低换手）与涨跌幅修正。
    *   交易与风控：
        *   大盘择时 (总开关)：基于沪深300 RSRS 决定整体仓位（满仓/空仓）。
        *   盘中风控：11:25/11:27 两次检查，若跌破阈值（60分钟线跌破MA20）提前止损。
    *   交易执行：09:30 卖出，09:35 买入（分离买卖）。
4. 高收益防御版 (gao)
    *   定位：稳健改良版，引入多重过滤器。
    *   核心逻辑：基础评分逻辑与wy03相似，但使用了加权线性回归 (WLS)，赋予近期数据更高权重。
    *   交易与风控：
        *   交易时间：10:30 (避开早盘波动)。
        *   数据利用：深度利用当日10:30前的实时价格与成交量。
    *   核心特征：
        1.  3日大跌保护：若近3日（含今日实时）任意一日跌幅剧烈，直接剔除。
        2.  成交量异常：若今日开盘一小时成交量相比历史均值异常放量，视为出货信号，直接剔除。
        3.  分数上限：剔除得分过高(>5)的ETF，认为其动量已透支。
5. 动量加速版 (acc)
    *   定位：激进进攻版，追逐“疯牛”。
    *   核心逻辑：在动量评分的基础上，引入“加速度”考核。
    *   交易与风控：
        *   交易时间：09:35 (早盘确认)。
        *   数据利用：基本依赖昨日收盘数据，9:35主要为了确认开盘不跌停/能成交。
    *   核心特征：
        *   动量加速机制：通常策略会回避得分过高的标的（恐高），但本策略反其道而行。若得分极高(>6)，只要满足今日得分 $\ge$ 昨日得分 $\times$ 1.2 (即动量还在加速)，则果断买入。这是一种针对主升浪末端的搏杀逻辑。

## 二、改进方案

**讨论**

1. 是否需要引入长期动量？
2. 上述5个策略的核心是线性回归收益、再结合R^2来判断动量质量，滞后情况如何？能否使用零（低）滞后指标？
3. 如何平衡低延迟和假信号？

## 三、问题分析与建议 (Analysis & Recommendations)

### 1. 是否需要引入长期动量？
*   **分析**：目前的策略主要集中在短期动量（20-25日）。
    *   `wy03`, `gao`, `acc`, `yj15` 均为短期动量主导。
    *   `long` 策略虽然有名为“long”，但使用的是“短期动量 - 长期反转”（200日反转），即长期看跌。
*   **建议**：
    *   **趋势共振**：可以引入长期动量（如 120日 或 200日 均线/动量）作为**过滤条件**（Filter），而非打分因子。例如：仅在 价格 > 200日均线 时才进行短期动量轮动，有助于规避长期熊市。
    *   **双频轮动**：可以将资金分配为“短期激进”和“长期稳健”两个池子，长期池使用长期动量因子，降低整体换手率和波动。

### 2. 关于线性回归 + R^2 的滞后性与低延迟指标
*   **现状分析**：
    *   `wy03` 使用普通最小二乘法 (OLS)，窗口25天。其“重心”在窗口中间 (t-12.5)，滞后较明显。
    *   `gao` 和 `acc` 使用加权最小二乘法 (WLS)，权重从1线性增加到2。重心向当前时间 t 偏移，滞后减少（约为 t-8 左右），但仍存在。
*   **低/零滞后指标可行性**：
    *   **可用方案**：
        1.  **指数加权移动平均 (EMA) / 分形自适应均线 (FRAMA)**：对近期价格赋予指数级权重，反应更快。
        2.  **Jurik Moving Average (JMA) / Hull Moving Average (HMA)**：这些是专门设计的低滞后均线。
        3.  **Kalman Filter (卡尔曼滤波)**：可以用于估计当前的“真实”价格斜率，对噪声有分离能力，且响应极快。
    *   **替换建议**：可以用 `Jurik Slope` 或 `Kalman Filter Slope` 代替现有的 `Polyfit Slope`。

### 3. 如何平衡低延迟和假信号？
*   **核心矛盾**：低延迟 = 高灵敏度 = 更多噪音（假信号）。高延迟 = 高平滑度 = 错过转折。
*   **解决方案**：
    1.  **多周期共振**：不要单看一个时间窗口。例如，要求 10日动量 和 20日动量 同时由负转正，或者 短周期斜率 > 0 且 长周期斜率 > 0.
    2.  **RCRS (Rank Correlation of Ranks)**：不是看单纯的收益率，而是看排名的稳定性。如果一个ETF连续多日排名上升，信号更可靠。
    3.  **波动率校正**：在计算动量时除以波动率（夏普比率思路）。当市场剧烈波动时，单纯的高收益可能只是噪音；低波动的上涨更可信。目前的 `R^2` 乘子其实已经起到了类似作用（拟合越好，R^2越高，相当于惩罚了高波动）。

## 四、前沿指标引入：Trendflex 与 Reflex

### 1. 指标背景
由 John Ehlers 在 2020 年提出，旨在解决传统动量指标（如均线、RSI）的滞后性问题。
*   **Reflex**: 零滞后循环指标，用于捕捉短期反转。
*   **Trendflex**: 在 Reflex 基础上改进，保留了趋势分量，适合捕捉趋势。

### 2. 计算逻辑 (Python 实现)
以下代码基于 Ehlers 的原始论文公式实现，适用于 `jqdata` 环境：

```python
import numpy as np
import pandas as pd

def calc_ehlers_indicators(close_series, length=20):
    """
    计算 Ehlers 的 Reflex 和 Trendflex 指标
    :param close_series: 收盘价序列 (pandas Series)
    :param length: 周期长度 (默认20)
    :return: (reflex_series, trendflex_series)
    """
    # 1. SuperSmoother Filter
    # 滤波参数
    a1 = np.exp(-1.414 * np.pi / (length / 2))
    b1 = 2 * a1 * np.cos(1.414 * 180 / (length / 2) * np.pi / 180) # 注意角度转弧度
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    
    filt = np.zeros_like(close_series)
    close = close_series.values
    
    # 递归计算 Filt
    for i in range(2, len(close)):
        filt[i] = c1 * (close[i] + close[i-1]) / 2 + c2 * filt[i-1] + c3 * filt[i-2]
        
    # 2. Reflex Calculation
    reflex = np.zeros_like(close)
    ms_reflex = np.zeros_like(close)
    
    # 3. Trendflex Calculation
    trendflex = np.zeros_like(close)
    ms_trendflex = np.zeros_like(close)
    
    for i in range(length, len(close)):
        # Reflex Sum
        sum_reflex = 0.0
        # Slope for Reflex (calculated over 'length' period)
        slope = (filt[i] - filt[i-length]) / length
        
        for count in range(1, length + 1):
            # Reflex: (Filt + count*slope) - Filt[i-count] -> 衡量偏离“当前趋势线”的程度
            sum_reflex += (filt[i] + (i - (i-count)) * slope)  - filt[i-count] # Wait, standard formula check needed
            # Ehlers Reflex Formula:
            # Sum = Sum + (Filt + (i - count) * slope) - Filt[i - count] (Adjust based on relative index)
            # Correct simplified logic: Sum += (Filt[current] - (predicted by slope)) ... actually let's use the straightforward definition:
            # The difference between the data and the line tangent to the data
            current_tangent = filt[i] - slope * count 
            # sum_reflex += (filt[i-count] - current_tangent) # No, let's stick to the official one below
            pass

        # Re-implementing with exact loop for clarity
        sum_reflex = 0
        sum_trendflex = 0
        
        for count in range(1, length + 1):
            # Reflex: Sum of [ (Filt[t] + slope * count) - Filt[t-count] ] ? 
            # Ehlers code: Sum = Sum + Filt + (Length - count) * Slope - Filt[len - count]
            # Actually easier: Trendflex is simpler.
            sum_trendflex += filt[i] - filt[i-count]
        
        sum_trendflex /= length
        
        # Reflex (Corrected):
        # 2-point slope? No, usually slope is (Filt[i] - Filt[i-Length]) / Length
        slope = (filt[i] - filt[i-length]) / length
        sum_reflex = 0
        for count in range(1, length + 1):
             # Forecast backward? 
             # Reflex is the sum of differences between the filter and the slope line extending back
             temp = filt[i] - slope * count 
             sum_reflex += (filt[i-count] - temp) # Difference from the line
             
        sum_reflex /= length

        # MS (Mean Square) Smoothing
        ms_reflex[i] = 0.04 * sum_reflex**2 + 0.96 * ms_reflex[i-1]
        ms_trendflex[i] = 0.04 * sum_trendflex**2 + 0.96 * ms_trendflex[i-1]
        
        if ms_reflex[i] != 0:
            reflex[i] = sum_reflex / np.sqrt(ms_reflex[i])
            
        if ms_trendflex[i] != 0:
            trendflex[i] = sum_trendflex / np.sqrt(ms_trendflex[i])
            
    return pd.Series(reflex, index=close_series.index), pd.Series(trendflex, index=close_series.index)
```

### 3. 在策略中的应用建议
*   **替换动量因子**：
    *   **Trendflex vs Slope**：`Trendflex` 本质上也是在衡量“当前价格相对于过去价格的偏离程度”，但经过了 SuperSmoother 降噪和 RMS 标准化。
    *   **用法**：可以用 `Trendflex` 的值直接作为评分因子。`Trendflex > 0` 表示上升趋势。由于它归一化了波动率，不同 ETF 之间更具可比性。
*   **辅助过滤 (Reflex)**：
    *   `Reflex` 是一个零滞后的震荡指标，在捕捉**为了反转**（Reversal）或**周期顶点**时非常有效。
    *   当 `Trendflex` 高位（趋势强）但 `Reflex` 此刻开始向下穿越零轴时，可能预示着短期顶部的到来，可以用作**止盈信号**。

## 五、其他建议引入的指标 (Alternatives)

除了 Trendflex 和 Reflex，以下三个指标也非常适合 ETF 轮动策略，特别是在解决滞后性和捕捉趋势方面：

### 1. Ehlers Fisher Transform (费舍尔变换)
*   **逻辑**：将价格数据（通常是非正态分布的）转化为高斯正态分布。这使得极值（超买/超卖）非常清晰，且转折点（Turning Points）极其敏锐。
*   **应用**：
    *   **趋势确认**：Fisher 指标 > 0 且向上发散。
    *   **精准抄底/逃顶**：Fisher 指标到达 +/- 2.0 极端值后的回归。
*   **代码实现**：
    ```python
    def calc_fisher(high, low, length=10):
        # 简化版实现
        # 1. 计算中点价
        hl2 = (high + low) / 2
        # 2. 归一化到 -1 ~ 1
        x = np.zeros_like(hl2)
        fish = np.zeros_like(hl2)
        
        for i in range(length, len(hl2)):
            # 滚动窗口内的最大最小值
            mx = np.max(hl2[i-length+1:i+1])
            mn = np.min(hl2[i-length+1:i+1])
            
            if mx - mn == 0:
                x[i] = 0
            else:
                x[i] = 2 * ((hl2[i] - mn) / (mx - mn) - 0.5)
            
            # 平滑
            x[i] = 0.33 * 2 * ((hl2[i] - mn) / (mx - mn) - 0.5) + 0.67 * x[i-1]
            if x[i] > 0.99: x[i] = 0.999
            if x[i] < -0.99: x[i] = -0.999
            
            # Fisher 变换
            fish[i] = 0.5 * np.log((1 + x[i]) / (1 - x[i])) + 0.5 * fish[i-1]
            
        return pd.Series(fish, index=high.index)
    ```

### 2. MESA Adaptive Moving Average (MAMA & FAMA)
*   **逻辑**：自适应均线。在市场横盘时，它走得很慢（避免假突破）；在趋势爆发时，它紧跟价格（低滞后）。通过 Hilbert Transform 测量相位变化率来调整 Alpha 值。
*   **应用**：非常适合做**止损线**或**趋势过滤器**。当 Price 下穿 MAMA 时，往往是趋势结束的确切信号，比固定周期均线（如 MA20）反应更快且不易被震荡洗出。

### 3. Fractal Adaptive Moving Average (FRAMA)
*   **逻辑**：利用分形维数（Fractal Dimension）来调整均线速度。市场越混沌（震荡），分形维数越高，均线变慢；市场越有趋势，分形维数越低，均线变快。
*   **应用**：替代传统的 MA25 或 MA200。作为长期趋势的判断标准，能有效区分震荡市和趋势市，避免在震荡市中频繁止损。

## 六、指标与排序逻辑的适配性讨论 (Discussion on Suitability for Ranking)

针对现有策略“**计算得分 -> 排序 -> 买入TopN**”的核心逻辑，以下是关于新引入指标的适配性分析：

### 1. Trendflex (强烈推荐用于排序)
*   **适配性**：⭐⭐⭐⭐⭐
*   **理由**：Trendflex 的设计初衷就是衡量“趋势成分相对于噪声的比率”。它的值直接反映了趋势的强度和质量。
*   **如何应用**：
    *   **直接替换**：用 `Trendflex` 值替代 `Slope * R^2`。
    *   **逻辑**：
        1.  计算 ETF 池中每个标的的 Trendflex 值。
        2.  **排序**：按 Trendflex 从大到小排序。
        3.  **买入**：选择 Trendflex 最大的 Top N。
    *   **优势**：由于 Trendflex 内部已经包含了降噪（SuperSmoother）和标准化（RMS），它天然具备了 `R^2` 的功能（抗噪），且比简单的线性回归响应更快。

### 2. Reflex (适合作为过滤/择时，不适合直接排序)
*   **适配性**：⭐⭐
*   **理由**：Reflex 是一个震荡指标（Oscillator），主要用于捕捉**周期性反转**。
    *   **风险**：Reflex 值极高通常意味着周期见顶（Overbought），即将回调；Reflex 值极低意味着周期见底（Oversold）。
    *   如果按 Reflex 从大到小排序买入，很可能会买在短期高点（“接盘”）。
*   **如何应用**：
    *   **作为过滤器**：仅当 `Reflex < 顶部阈值` 时才允许买入。
    *   **作为触发器**：当 Trendflex 为正（有趋势）且 Reflex 从下向上穿越零轴（确认反转向上）时，作为买入信号。

### 3. Fisher Transform (适合短期爆发力排序，需防极值)
*   **适配性**：⭐⭐⭐
*   **理由**：Fisher 变换将价格变成正态分布。High Fisher = 价格处于近期极高位。
    *   **优势**：对短期爆发力反应极快。
    *   **风险**：类似于 Reflex，极高的 Fisher 值 (>2.0) 往往对应价格不可持续的极端状态，随时可能均值回归。
*   **如何应用**：
    *   **排序修正**：可以按 `Fisher` 排序，但**剔除**掉 `Fisher > 2.5` 的标的（视为过热）。
    *   **结合 Delta**：按 `Fisher[t] - Fisher[t-1]` 排序，寻找动量**加速**最快的品种，而非绝对位置最高的品种。

### 4. MAMA / FRAMA (适合作为基准线，转化为偏离度后可排序)
*   **适配性**：⭐⭐⭐
*   **理由**：这两个指标输出的是**价格水平**（Price Level），而不是类似 Slope 的动量分数。因此不能直接用于排序。
*   **如何应用**：
    *   **转化**：需要计算“价格相对于均线的偏离度”来生成分数。
        *   `Score = (Close - MAMA) / MAMA`
        *   `Score = (Close - FRAMA) / FRAMA`
    *   **排序**：按上述 Score 排序，逻辑等同于“乖离率”策略。
    *   **特有优势 (FRAMA)**：FRAMA 内部有一个副产品——**分形维数 (Fractal Dimension, D)**。
        *   $D$ 用于衡量序列的混乱程度（1.0=直线趋势，2.0=完全随机噪声）。
        *   **创新用法**：可以按 $D$ 值**从小到大**排序。优先买入 $D$ 值最低的 ETF，意味着买入“走势最平滑、最接近直线趋势”的品种，这与 `R^2` 的追求如出一辙。