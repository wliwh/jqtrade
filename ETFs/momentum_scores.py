
import pandas as pd
import numpy as np
import akshare as ak
import efinance as ef
from scipy.stats import linregress
import math

class AkshareDataLoader:
    """
    负责从akshare获取数据，并进行初步清洗。
    """
    @staticmethod
    def fetch_etf_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取ETF日线数据。
        :param symbol: ETF代码，如 '510300'
        :param start_date: 开始日期 'YYYYMMDD'
        :param end_date: 结束日期 'YYYYMMDD'
        :return: DataFrame ['date', 'open', 'high', 'low', 'close', 'volume']
        """
        try:
            df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            # 重命名列以符合通用习惯
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume'
            })
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            # 确保数值类型
            cols = ['open', 'high', 'low', 'close', 'volume']
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

class MomentumScorer:
    """
    动量与趋势质量计算器。
    设计原则：
    1. 输入均为 pandas Series (通常是 log close)。
    2. 输出均为 pandas Series (与输入索引对齐)。
    3. 方便后期扩展新的指标。
    """
    
    @staticmethod
    def get_log_price(series: pd.Series) -> pd.Series:
        """
        计算对数价格。
        """
        return np.log(series)

    # --- Trend Magnitude (Slope / Momentum) ---

    @staticmethod
    def slope_ols(series: pd.Series, period: int = 25) -> pd.Series:
        """
        普通最小二乘法 (OLS) 线性回归斜率，年化。
        """
        def _calc_slope(y):
            if len(y) < period: return np.nan
            x = np.arange(len(y))
            slope, _, _, _, _ = linregress(x, y)
            return math.exp(slope * 250) - 1 # 年化收益率

        return series.rolling(window=period).apply(_calc_slope, raw=True)

    @staticmethod
    def slope_wls(series: pd.Series, period: int = 25) -> pd.Series:
        """
        加权最小二乘法 (WLS) 线性回归斜率，权重线性递增。
        此方法更重视近期数据。
        """
        weights = np.linspace(1, 2, period) # 线性权重 1 -> 2
        sum_weights = np.sum(weights)
        
        # 预计算 x 的加权相关项，加速运算
        x = np.arange(period)
        w_mean_x = np.average(x, weights=weights)
        
        # 预计算分母: sum(w * (x - mean_x)^2)
        denom = np.sum(weights * (x - w_mean_x)**2)

        def _calc_slope_wls(y):
            if len(y) < period: return np.nan
            w_mean_y = np.average(y, weights=weights)
            # 分子: sum(w * (x - mean_x) * (y - mean_y))
            numer = np.sum(weights * (x - w_mean_x) * (y - w_mean_y))
            slope = numer / denom if denom != 0 else 0
            return math.exp(slope * 250) - 1

        return series.rolling(window=period).apply(_calc_slope_wls, raw=True)

    @staticmethod
    def trendflex(series: pd.Series, period: int = 20) -> pd.Series:
        """
        Ehlers Trendflex 指标。
        利用 SuperSmoother 滤波器剔除噪声，计算趋势分量。
        """
        # 1. SuperSmoother Filter
        # a1 = exp(-1.414 * 3.14159 / period)
        # b1 = 2 * a1 * cos(1.414 * 180 / period)
        # c2 = b1
        # c3 = -a1 * a1
        # c1 = 1 - c2 - c3
        # filt = c1 * (close + close[1]) / 2 + c2 * filt[1] + c3 * filt[2]
        
        # 由于这是IIR滤波器，rolling apply比较困难，这里用向量化或循环实现
        # 为简单起见，这里使用pandas循环或numba(如果可用)。考虑到通用性，用纯Python循环优化。
        
        values = series.values
        n = len(values)
        filt = np.zeros(n)
        ms = np.zeros(n)
        trendflex = np.zeros(n)
        
        sqrt2 = 1.414
        pi = 3.1415926
        a1 = math.exp(-sqrt2 * pi / period)
        b1 = 2 * a1 * math.cos(sqrt2 * pi / period) # 注意: math.cos 输入是弧度，原公式180/period可能是度数，需确认。
        # Ehlers通常用度数: 1.414*180/period 是度数。转弧度 -> 1.414 * PI / period
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        
        for i in range(2, n):
            filt[i] = c1 * (values[i] + values[i-1]) / 2 + c2 * filt[i-1] + c3 * filt[i-2]
            
            # Trendflex 计算: MS = rolling sum of (filt - filt[i-lag]) ... simplified
            # Trendflex = (Filt - Filt[Lag]) / Quantized RMS
            # 这里简化实现 Trendflex 核心思想：在此处直接计算相对于自身的动量
            # 官方 Trendflex 涉及更复杂的 RMS 标准化，这里实现核心趋势强度。
            # 简化版：Trendflex ≈ Filt[i] - Filt[i-period/2] ? 
            # 让我们遵循文档思路：Trendflex 值本身用于排序，衡量趋势强度。
            
        # 由于完整 Trendflex 较复杂，咱们先用一种近似：SuperSmoother 的斜率。
        # 或者直接用 Difference of SuperSmoother.
        # 此处暂且用 Smoothed Momentum: Filt[i] - Filt[i-delta]
        
        s_series = pd.Series(filt, index=series.index)
        return s_series.diff(3) # 简单差分代表趋势方向

    @staticmethod
    def fisher_transform(series: pd.Series, period: int = 10) -> pd.Series:
        """
        Ehlers Fisher Transform.
        此实现需要先将价格归一化到 [-1, 1] 之间（通过RSI或Stochastic），然后应用Fisher变换。
        这里使用类似 Stochastic 的归一化方法。
        """
        # 1. Normalize price to range [-1, 1] based on period window
        roll_min = series.rolling(window=period).min()
        roll_max = series.rolling(window=period).max()
        
        # Value1 = 0.33*2*((Price - Min)/(Max - Min) - 0.5) + 0.67*Value1[1]
        # range_val 映射到 -0.5 ~ 0.5 -> *2 -> -1 ~ 1
        
        def _calc_fisher_steps(arr):
             # 向量化处理稍显复杂，因为有递归依赖
             pass
             
        # 循环实现
        values = series.values
        n = len(values)
        fish = np.zeros(n)
        val1 = np.zeros(n)
        
        # 预计算 rolling min/max
        min_vals = roll_min.fillna(method='bfill').values
        max_vals = roll_max.fillna(method='bfill').values
        
        for i in range(1, n):
            div = (max_vals[i] - min_vals[i])
            if div == 0:
                x = 0
            else:
                x = 2 * ((values[i] - min_vals[i]) / div - 0.5)
            
            # 平滑 Value1
            val1[i] = 0.33 * x + 0.67 * val1[i-1]
            
            # 截断以防 log 负数/溢出
            val1[i] = max(min(val1[i], 0.999), -0.999)
            
            # Fisher = 0.5 * log((1+Value1)/(1-Value1)) + 0.5 * Fisher[1]
            fish[i] = 0.5 * math.log((1 + val1[i]) / (1 - val1[i])) + 0.5 * fish[i-1]
            
        return pd.Series(fish, index=series.index)

    # --- Trend Quality (Stability) ---

    @staticmethod
    def r_squared(series: pd.Series, period: int = 25) -> pd.Series:
        """
        计算 OLS 回归的 R^2。
        """
        def _calc_r2(y):
            if len(y) < period: return np.nan
            x = np.arange(len(y))
            slope, intercept, r_value, _, _ = linregress(x, y)
            return r_value ** 2

        return series.rolling(window=period).apply(_calc_r2, raw=True)

    @staticmethod
    def efficiency_ratio(series: pd.Series, period: int = 10) -> pd.Series:
        """
        Kaufman Efficiency Ratio (ER).
        ER = |Net Change| / Sum of Absolute Changes
        """
        change = series.diff().abs()
        net_change = series.diff(period).abs()
        sum_change = change.rolling(window=period).sum()
        
        return net_change / sum_change

    @staticmethod
    def fractal_dimension(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        基于 FRAMA 原理计算分形维数 (Fractal Dimension).
        D = (log(N1 + N2) - log(N3)) / log(2)
        N1 = (Highest - Lowest) of first half
        N2 = (Highest - Lowest) of second half
        N3 = (Highest - Lowest) of full period
        
        :param data: DataFrame containing 'high' and 'low' columns, or a Series (treated as close price).
        :param period: Calculation window (must be even).
        :return: Fractal Dimension Series.
        """
        if isinstance(data, pd.Series):
            # If Series provided, assume it's close price and estimate range
            high = data
            low = data
        elif isinstance(data, pd.DataFrame):
            if 'high' in data.columns and 'low' in data.columns:
                high = data['high']
                low = data['low']
            else:
                # Fallback to close if specific columns not found
                col = data.columns[0]
                high = data[col]
                low = data[col]
        else:
            raise ValueError("Input must be a pandas Series or DataFrame")

        # Ensure period is even for equal halving
        if period % 2 != 0:
            period += 1
            
        w_half = period // 2
        
        # N3: Amplitude of the full period
        # Note: FRAMA uses (Max(High) - Min(Low)) / period
        # But here we stick to Ehlers' 2020 definition or similar implementation
        
        r_max = high.rolling(window=period).max()
        r_min = low.rolling(window=period).min()
        n3 = (r_max - r_min) / period
        
        # Split interval into two halves
        # Half 2 (recent half): t to t-w_half
        rmax_2 = high.rolling(window=w_half).max()
        rmin_2 = low.rolling(window=w_half).min()
        n2 = (rmax_2 - rmin_2) / w_half
        
        # Half 1 (older half): t-w_half to t-period
        # We can get this by shifting N2 by w_half
        n1 = n2.shift(w_half)
        
        epsilon = 1e-9
        # D = (log(N1 + N2) - log(N3)) / log(2)
        # 验证逻辑：如果直线，N1=k, N2=k, N3=k
        d = (np.log(n1 + n2 + epsilon) - np.log(n3 + epsilon)) / np.log(2)
        
        return d

    # --- Distribution / Statistical Thresholds ---

    @staticmethod
    def quantile_threshold(series: pd.Series, quantile: float = 0.9) -> float:
        """
        计算给定分位数的临界值 (全局)。
        :param series: 输入数据序列
        :param quantile: 分位数 (0.0 ~ 1.0)。例如 0.9 代表前 10% 的门槛 (90th percentile)。
        :return: 临界值 float
        """
        return series.quantile(quantile)

    @staticmethod
    def rolling_quantile_threshold(series: pd.Series, window: int, quantile: float = 0.9) -> pd.Series:
        """
        计算滚动窗口内的分位数临界值 (History Relative)。
        :param series: 输入数据序列
        :param window: 滚动窗口大小 (例如 250天)
        :param quantile: 分位数 (0.0 ~ 1.0)
        :return: 临界值 Series
        """
        return series.rolling(window=window).quantile(quantile)

def signal_score(symbol):
    # symbol = "518880" # 沪深300 ETF
    print(f"Fetching data for {symbol}...")
    df = AkshareDataLoader.fetch_etf_daily(symbol, "20220101", "20260208")
    
    if not df.empty:
        print(f"Data fetched: {len(df)} rows.")
        
        # 预处理：取对数
        log_close = MomentumScorer.get_log_price(df['open'])
        
        # 计算各种指标
        print("Calculating indicators...")
        df['slope_ols'] = MomentumScorer.slope_ols(log_close, period=25)
        df['slope_wls'] = MomentumScorer.slope_wls(log_close, period=25)
        df['trendflex'] = MomentumScorer.trendflex(log_close, period=20)
        
        # Fisher 需要原始价格或归一化价格，这里传入 log_close 也可以，或者 raw close
        df['fisher'] = MomentumScorer.fisher_transform(log_close, period=10)
        
        df['r2'] = MomentumScorer.r_squared(log_close, period=25)
        df['er'] = MomentumScorer.efficiency_ratio(log_close, period=20)
        df['fractal_dim'] = MomentumScorer.fractal_dimension(df, period=20)
        
        # 构造混合得分示例
        # Score = Trendflex * (2 - D)
        df['score_trend_dim'] = df['trendflex'] * (2 - df['fractal_dim'])
        df['score_ols_r2'] = df['slope_ols'] * df['r2']
        df['score_wls_r2'] = df['slope_wls'] * df['r2']
        df['score_trend_er'] = df['trendflex'] * df['er']
        df['score_ols_er'] = df['slope_ols'] * df['er']
        df['score_wls_er'] = df['slope_wls'] * df['er']
        
        # Calculate Top 10% Threshold (90th percentile)
        top_10_threshold = MomentumScorer.quantile_threshold(df['score_wls_r2'], quantile=0.96)
        print(f"\nGlobal Top 10% Threshold for Score (WLS * R2): {top_10_threshold:.4f}")
        
        # Calculate Rolling Top 10% Threshold (e.g., 60-day window)
        df['score_wls_r2_top10'] = MomentumScorer.rolling_quantile_threshold(df['score_wls_r2'], window=200, quantile=0.95)
        
        print("\nTail of the calculated data (with rolling threshold):")
        ndf = df[['open', 'slope_wls', 'r2', 'score_wls_r2', 'score_wls_r2_top10']]
        ndf = ndf[ (ndf['score_wls_r2'] > 2)]

        print(ndf)


if __name__ == "__main__":
    # Test execution

    else:
        print("Failed to fetch data.")
