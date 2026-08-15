This Asset Drift Model is a statistical tool designed to detect whether an asset exhibits a systematic directional tendency in its historical returns. Unlike traditional momentum indicators that react to price movements, this indicator performs a formal hypothesis test to determine if the observed drift is statistically significant, economically meaningful, and structurally stable across time. The result is a classification that helps traders understand whether historical evidence supports a directional bias in the asset.

The core question the indicator answers is simple: Has this asset shown a reliable tendency to move in one direction over the past three years, and is that tendency strong enough to matter?

## What is drift and why does it matter

In financial economics, drift refers to the expected rate of return of an asset over time. The concept originates from the geometric Brownian motion model, which describes asset prices as following a random walk with an added drift component (Black and Scholes, 1973). If drift is zero, price movements are purely random. If drift is positive, the asset tends to appreciate over time. If negative, it tends to depreciate.

The existence of drift has profound implications for trading strategy. Eugene Fama's Efficient Market Hypothesis (Fama, 1970) suggests that in efficient markets, risk-adjusted drift should be minimal because prices already reflect all available information. However, decades of empirical research have documented persistent anomalies. Jegadeesh and Titman (1993) demonstrated that stocks with positive past returns continue to outperform, a phenomenon known as momentum. DeBondt and Thaler (1985) found evidence of long-term mean reversion. These findings suggest that drift is not constant and can vary across assets and time periods.

For practitioners, understanding drift is fundamental. A positive drift implies that long positions have a statistical edge over time. A negative drift suggests short positions may be advantageous. No detectable drift means the asset behaves more like a random walk, where directional strategies have no inherent advantage.


## How professionals use drift analysis

Institutional investors and hedge funds have long incorporated drift analysis into their systematic strategies. Quantitative funds typically estimate drift as part of their alpha generation process, using it to tilt portfolios toward assets with favorable expected returns (Grinold and Kahn, 2000).

The challenge lies not in calculating drift but in determining whether observed drift is genuine or merely statistical noise. A naive approach might conclude that any positive average return indicates positive drift. However, financial returns are noisy, and short samples can produce misleading estimates. This is why professional quants rely on formal statistical inference.

The standard approach involves testing the null hypothesis that expected returns equal zero against the alternative that they differ from zero. The test statistic is typically a t-ratio: the sample mean divided by its standard error. However, financial returns often exhibit serial correlation and heteroskedasticity, which invalidate simple standard errors. To address this, practitioners use heteroskedasticity and autocorrelation consistent standard errors, commonly known as HAC or Newey-West standard errors (Newey and West, 1987).

Beyond statistical significance, professional investors also consider economic significance. A statistically significant drift of 0.5 percent annually may not justify trading costs. Conversely, a large drift that fails to reach statistical significance due to high volatility may still inform portfolio construction. The most robust conclusions require both statistical and economic thresholds to be met.


## Methodology

The Asset Drift Model implements a rigorous inference framework designed to minimize false positives while detecting genuine drift.

Return calculation

The indicator uses logarithmic returns over non-overlapping 60-day periods. Non-overlapping returns are essential because overlapping returns introduce artificial autocorrelation that biases variance estimates (Richardson and Stock, 1989). Using 60-day horizons rather than daily returns reduces noise and captures medium-term drift relevant for position traders.

The sample window spans 756 trading days, approximately three years of data. This provides 12 independent observations for the full sample and 6 observations per half-sample for structural stability testing.

Statistical inference

The indicator calculates the t-statistic for the null hypothesis that mean returns equal zero. To account for potential residual autocorrelation, it applies a simplified HAC correction with one lag, appropriate for non-overlapping returns where autocorrelation is minimal by construction.

Statistical significance requires the absolute t-statistic to exceed 2.0, corresponding to approximately 95 percent confidence. This threshold follows conventional practice in financial econometrics (Campbell, Lo, and MacKinlay, 1997).

## Power analysis

A critical but often overlooked aspect of hypothesis testing is statistical power: the probability of detecting drift when it exists. With small samples, even substantial drift may fail to reach significance due to high standard errors. The indicator calculates the minimum detectable effect at 95 percent confidence and requires observed drift to exceed this threshold. This prevents classifying assets as having no drift when the test simply lacks power to detect it.

## Robustness checks

The indicator applies multiple robustness checks before classifying drift as genuine.

First, the sign test examines whether the proportion of positive returns differs significantly from 50 percent. This non-parametric test is robust to distributional assumptions and verifies that the mean is not driven by outliers.

Second, mean-median agreement ensures that the mean and median returns share the same sign. Divergence indicates skewness that could distort inference.

Third, structural stability splits the sample into two halves and requires consistent signs of both means and t-statistics across sub-periods. This addresses the concern that drift may be an artifact of a specific regime rather than a persistent characteristic (Andrews, 1993).

Fourth, the variance ratio test detects mean-reverting behavior. Lo and MacKinlay (1988) showed that if returns follow a random walk, the variance of multi-period returns should scale linearly with the horizon. A variance ratio significantly below one indicates mean reversion, which contradicts persistent drift. The indicator blocks drift classification when significant mean reversion is detected.

## Classification system

Based on these tests, the indicator classifies assets into three categories.

Strong evidence indicates that all criteria are met: statistical significance, economic significance (at least 3 percent annualized drift), adequate power, and all robustness checks pass. This classification suggests the asset has exhibited reliable directional tendency that is both statistically robust and economically meaningful.

Weak evidence indicates statistical significance without economic significance. The drift is detectable but small, typically below 3 percent annually. Such assets may still have directional tendency but the magnitude may not justify concentrated positioning.

No evidence indicates insufficient statistical support for drift. This does not prove the asset is driftless; it means the available data cannot distinguish drift from random variation. The indicator provides the specific reason for rejection, such as failed power analysis, inconsistent sub-samples, or detected mean reversion.

## Dashboard explanation

The dashboard displays all relevant statistics for transparency.

Classification shows the current drift assessment: Positive Drift, Negative Drift, Positive (weak), Negative (weak), or No Drift.

Evidence indicates the strength of evidence: Strong, Weak, or None, with the specific reason for rejection if applicable.

Inference shows whether the sample is sufficient for analysis. Blocked indicates fewer than 10 observations. Heuristic indicates 10 to 19 observations, where asymptotic approximations are less reliable. Allowed indicates 20 or more observations with reliable inference.

The t-statistics for full sample and both half-samples show the test statistics and sample sizes. Double asterisks denote significance at the 5 percent level.

Power displays OK if observed drift exceeds the minimum detectable effect, or shows the MDE threshold if power is insufficient.

Sign Test shows the z-statistic for the proportion test. An asterisk indicates significance at 10 percent.

Mean equals Median indicates agreement between central tendency measures.

Struct(m) shows structural stability of means across half-samples, including the standardized level deviation.

Struct(t) shows whether t-statistics have consistent signs across half-samples.

VR Test shows the variance ratio and its z-statistic. An asterisk indicates the ratio differs significantly from one.

Econ. Sig. indicates whether drift exceeds the 3 percent annual threshold.

Drift (ann.) shows the annualized drift estimate.

Regime indicates whether the asset exhibits mean-reverting behavior based on the variance ratio test.


## Practical applications for traders

For discretionary traders, the indicator provides a quantitative foundation for directional bias decisions. Rather than relying on intuition or simple price trends, traders can assess whether historical evidence supports their directional thesis.

For systematic traders, the indicator can serve as a regime filter. Trend-following strategies may perform better on assets with detectable positive drift, while mean-reversion strategies may suit assets where drift is absent or the variance ratio indicates mean reversion.

For portfolio construction, drift analysis helps identify assets where long-only exposure has historical justification versus assets requiring more balanced or tactical positioning.


## Limitations

This indicator performs retrospective analysis and does not predict future returns. Past drift does not guarantee future drift. Markets evolve, regimes change, and historical patterns may not persist.

The three-year sample window captures medium-term tendencies but may miss shorter regime changes or longer structural shifts. The 60-day return horizon suits position traders but may not reflect intraday or weekly dynamics.

Small samples yield heuristic rather than statistically robust results. The indicator flags such cases but users should interpret them with appropriate caution.


## References

Andrews, D.W.K. (1993) Tests for parameter instability and structural change with unknown change point. Econometrica, 61(4).

Black, F. and Scholes, M. (1973) The pricing of options and corporate liabilities. Journal of Political Economy, 81(3).

Campbell, J.Y., Lo, A.W. and MacKinlay, A.C. (1997) The econometrics of financial markets. Princeton: Princeton University Press.

DeBondt, W.F.M. and Thaler, R. (1985) Does the stock market overreact? Journal of Finance, 40(3).

Fama, E.F. (1970) Efficient capital markets: a review of theory and empirical work. Journal of Finance, 25(2).

Grinold, R.C. and Kahn, R.N. (2000) Active portfolio management. 2nd ed. New York: McGraw-Hill.

Jegadeesh, N. and Titman, S. (1993) Returns to buying winners and selling losers. Journal of Finance, 48(1).

Lo, A.W. and MacKinlay, A.C. (1988) Stock market prices do not follow random walks. Review of Financial Studies, 1(1).

Newey, W.K. and West, K.D. (1987) A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica, 55(3).

Richardson, M. and Stock, J.H. (1989) Drawing inferences from statistics based on multiyear asset returns. Journal of Financial Economics, 25(2).

```js
//@version=6
// (c) EdgeTools | Asset Drift Model

indicator("Asset Drift Model", shorttitle="ADM", overlay=false, max_bars_back=800)

// Parameters
HORIZON = 60
SAMPLE_TOTAL = 756
MIN_OBS_STRICT = 10
MIN_OBS_ROBUST = 20
T_CRIT_95 = 2.00
T_CRIT_90 = 1.65
MIN_DRIFT_ECON = 0.03
MAX_DEV_SD = 1.5
SIGN_Z_CRIT = 1.65
ANN_FACTOR = 252.0 / HORIZON

// Inputs
i_bgMode    = input.string("On", "Background", options=["Off", "On"], group="Display", tooltip="Background color based on drift classification")
i_bgInt     = input.int(75, "BG Intensity", minval=70, maxval=95, group="Display", tooltip="Transparency of background color")
i_barColor  = input.bool(true, "Bar Coloring", group="Display", tooltip="Color bars based on drift direction")
i_showDash  = input.bool(true, "Dashboard", group="Dashboard", tooltip="Show statistical details panel")
i_dashPos   = input.string("Top Right", "Position", options=["Top Right", "Top Left", "Bottom Right", "Bottom Left"], group="Dashboard")
i_dashSize  = input.string("Small", "Size", options=["Tiny", "Small", "Normal"], group="Dashboard", tooltip="Text size of dashboard")
i_showWM    = input.bool(true, "Watermark", group="Watermark", tooltip="Show ticker and classification summary")
i_wmPos     = input.string("Bottom Right", "Position", options=["Top Right", "Top Left", "Bottom Right", "Bottom Left"], group="Watermark")
i_wmSize    = input.string("Normal", "Size", options=["Small", "Normal", "Large"], group="Watermark", tooltip="Text size of watermark")
i_dark      = input.bool(true, "Dark Mode", group="Style", tooltip="Optimize colors for dark charts")
i_alerts    = input.bool(false, "Alerts", group="Alerts", tooltip="Trigger alerts on classification changes")

// Colors
cBull = i_dark ? #22c55e : #16a34a
cBear = i_dark ? #ef4444 : #dc2626
cWeak = i_dark ? #f59e0b : #d97706
cNeut = i_dark ? #737373 : #525252
cPrim = i_dark ? #3b82f6 : #2563eb
cText = i_dark ? #fafafa : #0a0a0a
cTblBg = i_dark ? #171717 : #f5f5f5
cHdrBg = i_dark ? #262626 : #e5e5e5

// Helpers
f_pos(p) => p == "Top Right" ? position.top_right : p == "Top Left" ? position.top_left : p == "Bottom Right" ? position.bottom_right : position.bottom_left

f_size(s) => s == "Tiny" ? size.tiny : s == "Small" ? size.small : s == "Normal" ? size.normal : s == "Large" ? size.large : size.small

dashSizeH = f_size(i_dashSize == "Tiny" ? "Small" : i_dashSize)
dashSizeD = f_size(i_dashSize)
dashSizeF = f_size(i_dashSize == "Normal" ? "Small" : "Tiny")

wmSizeT = f_size(i_wmSize == "Small" ? "Normal" : i_wmSize == "Normal" ? "Large" : "Large")
wmSizeM = f_size(i_wmSize)
wmSizeS = f_size(i_wmSize == "Large" ? "Small" : "Tiny")

f_median(arr) =>
    n = array.size(arr)
    if n == 0
        0.0
    else
        s = array.copy(arr)
        array.sort(s)
        m = n / 2
        n % 2 == 0 ? (array.get(s, m-1) + array.get(s, m)) / 2 : array.get(s, m)

// Non-overlapping returns
f_collectAllReturns(src, horizon, maxBars) =>
    var float[] arr = array.new_float(0)
    array.clear(arr)
    nPeriods = math.floor(maxBars / horizon)
    for i = 0 to nPeriods - 1
        recentBar = i * horizon
        pastBar = (i + 1) * horizon
        if pastBar <= maxBars
            pRecent = nz(src[recentBar], src)
            pPast = nz(src[pastBar], src)
            if pPast > 0 and pRecent > 0
                array.push(arr, math.log(pRecent / pPast))
    arr

f_arrayStats(arr) =>
    n = array.size(arr)
    if n < 3
        [0.0, 0.0, 0.0, 0, 0]
    else
        mean = array.avg(arr)
        med = f_median(arr)
        sumSq = 0.0
        posCount = 0
        for i = 0 to n - 1
            v = array.get(arr, i)
            sumSq += (v - mean) * (v - mean)
            if v > 0
                posCount += 1
        sd = math.sqrt(sumSq / (n - 1))
        [mean, med, sd, n, posCount]

f_hacSE(arr, mean) =>
    n = array.size(arr)
    if n < 3
        0.0
    else
        g0 = 0.0
        for i = 0 to n - 1
            d = array.get(arr, i) - mean
            g0 += d * d
        g0 := g0 / n
        g1 = 0.0
        for i = 1 to n - 1
            di = array.get(arr, i) - mean
            dj = array.get(arr, i - 1) - mean
            g1 += di * dj
        g1 := g1 / n
        hV = g0 + g1
        math.sqrt(math.max(hV / n, 1e-14))

f_splitArray(arr, isFirstHalf) =>
    n = array.size(arr)
    var float[] result = array.new_float(0)
    array.clear(result)
    if n < 2
        result
    else
        halfN = math.floor(n / 2)
        if isFirstHalf
            for i = 0 to halfN - 1
                array.push(result, array.get(arr, i))
        else
            for i = halfN to n - 1
                array.push(result, array.get(arr, i))
        result

f_vrTest(src, horizon, k, sampleBars) =>
    arr1 = f_collectAllReturns(src, horizon, sampleBars)
    arrK = f_collectAllReturns(src, horizon * k, sampleBars)
    [m1, md1, sd1, n1, pc1] = f_arrayStats(arr1)
    [mK, mdK, sdK, nK, pcK] = f_arrayStats(arrK)
    if sd1 < 1e-10 or n1 < MIN_OBS_STRICT or nK < 3
        [1.0, 0.0, false]
    else
        var1 = sd1 * sd1
        varK = sdK * sdK
        vr = varK / (k * var1)
        vrSE = math.sqrt(2.0 * (2*k - 1) * (k - 1) / (3.0 * k * nK))
        zVR = vrSE > 1e-10 ? math.abs(vr - 1.0) / vrSE : 0.0
        vrSig = zVR > T_CRIT_90
        [vr, zVR, vrSig]

f_mde(sd, n, tCrit) =>
    n < MIN_OBS_STRICT or sd <= 0 ? na : (sd / math.sqrt(n)) * tCrit * ANN_FACTOR

f_signTestZ(posCount, n) =>
    n < MIN_OBS_STRICT ? 0.0 : (float(posCount) - float(n)/2) / math.sqrt(float(n)/4)

// Calculations
allReturns = f_collectAllReturns(close, HORIZON, SAMPLE_TOTAL)
[mF, mdF, sdF, nF, pcF] = f_arrayStats(allReturns)
seF = f_hacSE(allReturns, mF)
tF = seF > 1e-10 ? mF / seF : 0.0

arr1 = f_splitArray(allReturns, true)
arr2 = f_splitArray(allReturns, false)

[m1, md1, sd1, n1, pc1] = f_arrayStats(arr1)
se1 = f_hacSE(arr1, m1)
t1 = se1 > 1e-10 ? m1 / se1 : 0.0

[m2, md2, sd2, n2, pc2] = f_arrayStats(arr2)
se2 = f_hacSE(arr2, m2)
t2 = se2 > 1e-10 ? m2 / se2 : 0.0

[vr, zVR, vrSig] = f_vrTest(close, HORIZON, 5, SAMPLE_TOTAL)
mde95 = f_mde(sdF, nF, T_CRIT_95)
annDrift = mF * ANN_FACTOR
zSign = f_signTestZ(pcF, nF)
hitRate = nF > 0 ? float(pcF) / float(nF) : 0.5

// Inference
inferenceAllowed = nF >= MIN_OBS_STRICT and n1 >= math.floor(MIN_OBS_STRICT/2) and n2 >= math.floor(MIN_OBS_STRICT/2)
isRobust = nF >= MIN_OBS_ROBUST

// Checks
statSig = inferenceAllowed and math.abs(tF) > T_CRIT_95
econSig = inferenceAllowed and math.abs(annDrift) >= MIN_DRIFT_ECON
powerOK = inferenceAllowed and (na(mde95) ? false : math.abs(annDrift) >= mde95)
medianSig = inferenceAllowed and math.abs(zSign) > SIGN_Z_CRIT
medianDir = zSign > 0 ? 1 : zSign < 0 ? -1 : 0
meanDir = mF > 0 ? 1 : mF < 0 ? -1 : 0
signTestOK = inferenceAllowed and (not medianSig or medianDir == meanDir)
meanMedOK = inferenceAllowed and ((mF > 0 and mdF > 0) or (mF < 0 and mdF < 0) or math.abs(mF) < 1e-10)
sameSignMean = (m1 > 0 and m2 > 0) or (m1 < 0 and m2 < 0)
pooledSD = math.sqrt((sd1*sd1 + sd2*sd2) / 2)
levelDevSD = pooledSD > 1e-10 ? math.abs(m1 - m2) / pooledSD : 0.0
levelOK = levelDevSD <= MAX_DEV_SD
sameSignT = (t1 > 0 and t2 > 0 and tF > 0) or (t1 < 0 and t2 < 0 and tF < 0)
structOK = inferenceAllowed and sameSignMean and levelOK and sameSignT
isMeanRev = vr < 1.0 and vrSig
vrOK = inferenceAllowed and not isMeanRev

// Classification
direction = mF > 0 ? 1 : mF < 0 ? -1 : 0
var string classReason = ""
var int classCode = 0

if not inferenceAllowed
    classCode := 0
    classReason := "n<" + str.tostring(MIN_OBS_STRICT)
else if not powerOK
    classCode := 0
    classReason := "power"
else if not statSig
    classCode := 0
    classReason := "t<" + str.tostring(T_CRIT_95, "#.#")
else if not signTestOK
    classCode := 0
    classReason := "signtest"
else if not meanMedOK
    classCode := 0
    classReason := "median"
else if not structOK
    classCode := 0
    classReason := not sameSignMean ? "sign(m)" : not sameSignT ? "sign(t)" : "level"
else if not vrOK
    classCode := 0
    classReason := "mean-rev"
else if not econSig
    classCode := 1
    classReason := "<" + str.tostring(MIN_DRIFT_ECON * 100, "#") + "%"
else
    classCode := 2
    classReason := ""

strongEvidence = classCode == 2
weakEvidence = classCode == 1
classification = classCode == 2 ? direction * 2 : classCode == 1 ? direction : 0

// Strings
evidenceStr = strongEvidence ? "STRONG" : weakEvidence ? "WEAK" : "NONE"
biasStr = classification == 2 ? "Positive Drift" : classification == -2 ? "Negative Drift" : classification == 1 ? "Positive (weak)" : classification == -1 ? "Negative (weak)" : "No Drift"
regimeStr = isMeanRev ? "Mean-Rev" : "Neutral"

// Visuals
color bgClr = na
if i_bgMode == "On"
    bgClr := classification >= 2 ? cBull : classification <= -2 ? cBear : math.abs(classification) == 1 ? cWeak : na
bgcolor(i_bgMode == "On" ? color.new(bgClr, i_bgInt) : na)

cBarLong  = #22c55e
cBarShort = #ef4444
cBarNeut  = #a3a3a3
color barClr = na
if i_barColor
    barClr := classification >= 2 ? cBarLong : classification <= -2 ? cBarShort : math.abs(classification) == 1 ? (direction > 0 ? cBarLong : cBarShort) : cBarNeut
barcolor(barClr)

// Watermark
var table wm = table.new(f_pos(i_wmPos), 1, 5, bgcolor=color.new(color.black, 100))
if i_showWM and barstate.islast
    evClr = not inferenceAllowed ? cNeut : strongEvidence ? cBull : weakEvidence ? cWeak : cNeut
    wmClr = color.new(i_dark ? color.white : color.black, 30)
    table.cell(wm, 0, 0, syminfo.ticker, text_color=wmClr, text_size=wmSizeT, text_halign=text.align_right)
    table.cell(wm, 0, 1, (inferenceAllowed ? "t=" + str.tostring(tF, "#.##") : "n<" + str.tostring(MIN_OBS_STRICT)) + " | n=" + str.tostring(nF), text_color=wmClr, text_size=wmSizeS, text_halign=text.align_right)
    table.cell(wm, 0, 2, evidenceStr + (classReason != "" ? " (" + classReason + ")" : ""), text_color=evClr, text_size=wmSizeM, text_halign=text.align_right)
    table.cell(wm, 0, 3, biasStr, text_color=evClr, text_size=wmSizeM, text_halign=text.align_right)
    table.cell(wm, 0, 4, isRobust ? "" : "Heuristic (n<20)", text_color=color.new(cWeak, 30), text_size=wmSizeS, text_halign=text.align_right)

// Dashboard
if i_showDash and barstate.islast
    var table d = table.new(f_pos(i_dashPos), 2, 18, border_width=1, bgcolor=color.new(cTblBg, 80))
    hB = color.new(cHdrBg, 20)
    evClr = not inferenceAllowed ? cNeut : strongEvidence ? cBull : weakEvidence ? cWeak : cNeut
    
    table.cell(d, 0, 0, "ADM", text_color=cText, bgcolor=hB, text_size=dashSizeH)
    table.cell(d, 1, 0, "Retrospective", text_color=color.new(cText, 50), bgcolor=hB, text_size=dashSizeF)
    
    table.cell(d, 0, 1, "Classification", text_color=cText, bgcolor=color.new(evClr, 80), text_size=dashSizeH)
    table.cell(d, 1, 1, biasStr, text_color=evClr, bgcolor=color.new(evClr, 80), text_size=dashSizeH)
    
    table.cell(d, 0, 2, "Evidence", text_color=cText, bgcolor=color.new(evClr, 90), text_size=dashSizeD)
    table.cell(d, 1, 2, evidenceStr + (classReason != "" ? " (" + classReason + ")" : ""), text_color=evClr, bgcolor=color.new(evClr, 90), text_size=dashSizeD)
    
    infClr = inferenceAllowed ? cBull : cBear
    table.cell(d, 0, 3, "Inference", text_color=cText, bgcolor=color.new(infClr, 90), text_size=dashSizeD)
    table.cell(d, 1, 3, inferenceAllowed ? (isRobust ? "Allowed" : "Heuristic") : "Blocked", text_color=infClr, bgcolor=color.new(infClr, 90), text_size=dashSizeD)
    
    tClr = statSig ? cBull : cNeut
    table.cell(d, 0, 4, "t(full) n=" + str.tostring(nF), text_color=cText, bgcolor=color.new(tClr, 90), text_size=dashSizeD)
    table.cell(d, 1, 4, str.tostring(tF, "#.##") + (statSig ? "**" : ""), text_color=tClr, bgcolor=color.new(tClr, 90), text_size=dashSizeD)
    
    t1Clr = t1 > 0 ? cBull : cBear
    table.cell(d, 0, 5, "t(H1) n=" + str.tostring(n1), text_color=cText, bgcolor=color.new(t1Clr, 90), text_size=dashSizeD)
    table.cell(d, 1, 5, str.tostring(t1, "#.##"), text_color=t1Clr, bgcolor=color.new(t1Clr, 90), text_size=dashSizeD)
    
    t2Clr = t2 > 0 ? cBull : cBear
    table.cell(d, 0, 6, "t(H2) n=" + str.tostring(n2), text_color=cText, bgcolor=color.new(t2Clr, 90), text_size=dashSizeD)
    table.cell(d, 1, 6, str.tostring(t2, "#.##"), text_color=t2Clr, bgcolor=color.new(t2Clr, 90), text_size=dashSizeD)
    
    table.cell(d, 0, 7, "Power", text_color=cText, bgcolor=color.new(powerOK ? cBull : cBear, 90), text_size=dashSizeD)
    table.cell(d, 1, 7, powerOK ? "OK" : "MDE:" + (na(mde95) ? "N/A" : str.tostring(mde95*100, "#") + "%"), text_color=powerOK ? cBull : cBear, bgcolor=color.new(powerOK ? cBull : cBear, 90), text_size=dashSizeD)
    
    table.cell(d, 0, 8, "Sign Test", text_color=cText, bgcolor=color.new(signTestOK ? cBull : cBear, 90), text_size=dashSizeD)
    table.cell(d, 1, 8, "z=" + str.tostring(zSign, "#.#") + (medianSig ? "*" : ""), text_color=signTestOK ? cBull : cBear, bgcolor=color.new(signTestOK ? cBull : cBear, 90), text_size=dashSizeD)
    
    table.cell(d, 0, 9, "Mean=Median", text_color=cText, bgcolor=color.new(meanMedOK ? cBull : cBear, 90), text_size=dashSizeD)
    table.cell(d, 1, 9, meanMedOK ? "Yes" : "No", text_color=meanMedOK ? cBull : cBear, bgcolor=color.new(meanMedOK ? cBull : cBear, 90), text_size=dashSizeD)
    
    table.cell(d, 0, 10, "Struct(m)", text_color=cText, bgcolor=color.new(sameSignMean and levelOK ? cBull : cBear, 90), text_size=dashSizeD)
    table.cell(d, 1, 10, (sameSignMean ? "+" : "-") + " lv:" + str.tostring(levelDevSD, "#.#") + "sd", text_color=sameSignMean and levelOK ? cBull : cBear, bgcolor=color.new(sameSignMean and levelOK ? cBull : cBear, 90), text_size=dashSizeD)
    
    table.cell(d, 0, 11, "Struct(t)", text_color=cText, bgcolor=color.new(sameSignT ? cBull : cBear, 90), text_size=dashSizeD)
    table.cell(d, 1, 11, sameSignT ? "Consistent" : "Conflict", text_color=sameSignT ? cBull : cBear, bgcolor=color.new(sameSignT ? cBull : cBear, 90), text_size=dashSizeD)
    
    table.cell(d, 0, 12, "VR Test", text_color=cText, bgcolor=color.new(vrOK ? cBull : cBear, 90), text_size=dashSizeD)
    table.cell(d, 1, 12, str.tostring(vr, "#.##") + " z=" + str.tostring(zVR, "#.#") + (vrSig ? "*" : ""), text_color=vrOK ? cBull : cBear, bgcolor=color.new(vrOK ? cBull : cBear, 90), text_size=dashSizeD)
    
    table.cell(d, 0, 13, "Econ. Sig.", text_color=cText, bgcolor=color.new(econSig ? cBull : cWeak, 90), text_size=dashSizeD)
    table.cell(d, 1, 13, econSig ? "Yes" : "<" + str.tostring(MIN_DRIFT_ECON*100, "#") + "%", text_color=econSig ? cBull : cWeak, bgcolor=color.new(econSig ? cBull : cWeak, 90), text_size=dashSizeD)
    
    dClr = annDrift > 0 ? cBull : cBear
    table.cell(d, 0, 14, "Drift (ann.)", text_color=cText, bgcolor=color.new(dClr, 90), text_size=dashSizeD)
    table.cell(d, 1, 14, str.tostring(annDrift * 100, "#.#") + "%", text_color=dClr, bgcolor=color.new(dClr, 90), text_size=dashSizeD)
    
    table.cell(d, 0, 15, "Regime", text_color=cText, bgcolor=color.new(cPrim, 90), text_size=dashSizeD)
    table.cell(d, 1, 15, regimeStr, text_color=cPrim, bgcolor=color.new(cPrim, 90), text_size=dashSizeD)
    
    table.cell(d, 0, 16, "H=" + str.tostring(HORIZON), text_color=color.new(cText, 50), bgcolor=color.new(cNeut, 95), text_size=dashSizeF)
    table.cell(d, 1, 16, "MinN=" + str.tostring(MIN_OBS_STRICT), text_color=color.new(cText, 50), bgcolor=color.new(cNeut, 95), text_size=dashSizeF)
    
    table.cell(d, 0, 17, "**p<.05 *p<.10", text_color=color.new(cText, 50), bgcolor=color.new(cNeut, 95), text_size=dashSizeF)
    table.cell(d, 1, 17, isRobust ? "" : "Heuristic", text_color=color.new(cWeak, 30), bgcolor=color.new(cNeut, 95), text_size=dashSizeF)

// Alerts
changed = classification != classification[1]
if changed and i_alerts
    alert("ADM: " + biasStr + " (" + evidenceStr + ")", alert.freq_once_per_bar)

alertcondition(strongEvidence and direction > 0, "Positive Drift", "Positive Drift detected")
alertcondition(strongEvidence and direction < 0, "Negative Drift", "Negative Drift detected")
alertcondition(weakEvidence, "Weak", "Weak evidence")
alertcondition(not inferenceAllowed, "Blocked", "Inference blocked")
```