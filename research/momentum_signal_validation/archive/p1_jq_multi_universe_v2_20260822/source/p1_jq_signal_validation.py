"""P1 cross-sectional momentum signal validation for JQ Research.

The file is intentionally self-contained: copy it into one JQ Research cell,
review ``DEFAULT_CONFIG``, execute the cell, then run ``RESULTS = run_p1()``.
Only the JQ fetch boundary depends on jqdata; every calculation below can be
verified locally with a synthetic close-price panel.

This is a signal study, not a tradable strategy backtest. A signal dated t uses
prices through close[t]. Its H-day outcome is close[t + H] / close[t] - 1.
That convention contains no future data in signal construction, but close[t]
is only a research reference price and is not assumed to be executable.
"""

try:
    from jqdata import *

    JQDATA_AVAILABLE = True
except ImportError:
    JQDATA_AVAILABLE = False

from math import erfc, sqrt
import builtins as _python_builtins
import datetime
import hashlib
import marshal
import sys
import time

import numpy as np
import pandas as pd


_PYTHON_BUILTIN_NAMES = (
    "ImportError",
    "RuntimeError",
    "TypeError",
    "ValueError",
    "abs",
    "all",
    "any",
    "bool",
    "callable",
    "dict",
    "enumerate",
    "float",
    "getattr",
    "globals",
    "hasattr",
    "int",
    "isinstance",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "print",
    "range",
    "repr",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
)


def restore_python_builtins(namespace=None):
    """Undo Python built-in names shadowed by ``from jqdata import *``."""

    if namespace is None:
        namespace = _python_builtins.globals()
    for name in _PYTHON_BUILTIN_NAMES:
        namespace[name] = _python_builtins.getattr(_python_builtins, name)
    return namespace


restore_python_builtins()


def _resolve_jq_api(name):
    """Resolve a JQ API with Python's globals-then-builtins lookup order."""

    namespace = _python_builtins.globals()
    if name in namespace and namespace[name] is not None:
        return namespace[name]
    return _python_builtins.getattr(_python_builtins, name, None)


def _jq_api_source(name):
    """Describe where a JQ API is visible for runtime diagnostics."""

    namespace = _python_builtins.globals()
    if name in namespace and namespace[name] is not None:
        return "module_globals"
    if _python_builtins.getattr(_python_builtins, name, None) is not None:
        return "python_builtins"
    return "unavailable"


def _build_progress_reporter(total_steps, enabled=True):
    """Return a lightweight JQ-compatible stage progress printer."""

    total_steps = max(int(total_steps), 1)
    completed = [0]
    started_at = time.time()

    def report(message, advance=True):
        if not enabled:
            return
        if advance:
            completed[0] = min(completed[0] + 1, total_steps)
        elapsed_seconds = max(int(time.time() - started_at), 0)
        hours = elapsed_seconds // 3600
        minutes = (elapsed_seconds % 3600) // 60
        seconds = elapsed_seconds % 60
        percentage = int(100.0 * completed[0] / float(total_steps))
        print(
            "[P1 %3d%% | %02d:%02d:%02d] %s"
            % (percentage, hours, minutes, seconds, message)
        )

    return report


BROAD_INDEX_UNIVERSE = {
    "000001.XSHG": "上证指数",
    "399001.XSHE": "深证成指",
    "000016.XSHG": "上证50",
    "000300.XSHG": "沪深300",
    "000905.XSHG": "中证500",
    "000852.XSHG": "中证1000",
    "000985.XSHG": "中证全指",
    "399006.XSHE": "创业板指",
    "399303.XSHE": "国证2000",
    "000688.XSHG": "科创50",
}

STYLE_INDEX_UNIVERSE = {
    "000918.XSHG": "300成长",
    "000919.XSHG": "300价值",
    "000920.XSHG": "300R成长",
    "000921.XSHG": "300R价值",
    "000922.XSHG": "中证红利",
    "000015.XSHG": "上证红利",
    "399372.XSHE": "大盘成长",
    "399373.XSHE": "大盘价值",
    "399374.XSHE": "中盘成长",
    "399375.XSHE": "中盘价值",
    "399376.XSHE": "小盘成长",
    "399377.XSHE": "小盘价值",
}

DEFAULT_UNIVERSES = {
    "broad": {
        "source": "get_price",
        "version": "broad_index_v1",
        "members": BROAD_INDEX_UNIVERSE,
        "min_assets": 6,
    },
    "industry_sw_l1": {
        "source": "sw1_daily_price",
        "version": "all_historical_sw_l1_v1",
        "classification": "sw_l1",
        "members": {},
        "min_assets": 20,
    },
    "style": {
        "source": "get_price",
        "version": "style_index_v1",
        "members": STYLE_INDEX_UNIVERSE,
        "min_assets": 6,
    },
}

DEFAULT_CONFIG = {
    "research_id": "momentum_signal_p1_multi_universe_v2",
    "universe_order": ("broad", "industry_sw_l1", "style"),
    "universes": DEFAULT_UNIVERSES,
    "research_start": "2016-01-01",
    "research_end": "2026-08-20",
    "periods": {
        "development": ("2016-01-01", "2021-12-31"),
        "validation": ("2022-01-01", "2023-12-31"),
        "locked_oos": ("2024-01-01", "2026-08-20"),
    },
    "lookbacks": (10, 15, 20, 25, 30, 40, 60, 90),
    "horizons": (1, 3, 5, 10, 20),
    "signals": ("formation_return", "annualized_slope", "r2", "slope_x_r2"),
    "annualization": 252,
    "group_count": 5,
    "top_ks": (1, 3, 5),
    "primary_signal": "slope_x_r2",
    "primary_lookback": 25,
    "primary_horizon": 5,
}


def build_config(overrides=None):
    """Return an independent validated research configuration."""

    config = dict(DEFAULT_CONFIG)
    config["universes"] = _copy_universes(DEFAULT_CONFIG["universes"])
    config["periods"] = dict(DEFAULT_CONFIG["periods"])
    if overrides:
        for key, value in overrides.items():
            if key == "universes":
                config[key] = _copy_universes(value)
            elif key == "periods":
                config[key] = dict(value)
            else:
                config[key] = value
    _validate_config(config)
    return config


def _validate_config(config):
    required_signals = {
        "formation_return",
        "annualized_slope",
        "r2",
        "slope_x_r2",
    }
    if not config["universes"]:
        raise ValueError("universes must not be empty")
    if set(config["universe_order"]) != set(config["universes"]):
        raise ValueError("universe_order must contain every universe exactly once")
    for universe_group in config["universe_order"]:
        spec = config["universes"][universe_group]
        if spec.get("source") not in ("get_price", "sw1_daily_price", "provided"):
            raise ValueError("unsupported source for %s" % universe_group)
        if spec.get("source") in ("get_price", "provided") and not spec.get(
            "members"
        ):
            raise ValueError("members must not be empty for %s" % universe_group)
        if int(spec.get("min_assets", 0)) < 3:
            raise ValueError("min_assets must be at least 3 for %s" % universe_group)
    if any(int(value) < 2 for value in config["lookbacks"]):
        raise ValueError("lookbacks must be at least 2")
    if any(int(value) < 1 for value in config["horizons"]):
        raise ValueError("horizons must be positive")
    unknown = set(config["signals"]) - required_signals
    if unknown:
        raise ValueError("unknown signals: %s" % sorted(unknown))
    if config["primary_signal"] not in config["signals"]:
        raise ValueError("primary_signal must be included in signals")
    if config["primary_lookback"] not in config["lookbacks"]:
        raise ValueError("primary_lookback must be included in lookbacks")
    if config["primary_horizon"] not in config["horizons"]:
        raise ValueError("primary_horizon must be included in horizons")
    if int(config["group_count"]) < 2:
        raise ValueError("group_count must be at least 2")

    previous_end = None
    for name, bounds in config["periods"].items():
        if len(bounds) != 2:
            raise ValueError("period %s must have start and end" % name)
        start, end = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
        if start > end:
            raise ValueError("period %s starts after it ends" % name)
        if previous_end is not None and start <= previous_end:
            raise ValueError("periods must be chronological and non-overlapping")
        previous_end = end


def fetch_close_panels(
    config=None,
    get_price_func=None,
    get_industries_func=None,
    sw1_price_fetcher=None,
    progress_func=None,
):
    """Fetch every versioned universe and return panels plus member catalogs."""

    config = build_config(config)
    if get_price_func is None and JQDATA_AVAILABLE:
        get_price_func = _resolve_jq_api("get_price")
    if get_industries_func is None and JQDATA_AVAILABLE:
        get_industries_func = _resolve_jq_api("get_industries")
    if sw1_price_fetcher is None and JQDATA_AVAILABLE:
        sw1_price_fetcher = _fetch_sw1_prices_from_jq
    max_lookback = max(config["lookbacks"])
    fetch_start = pd.Timestamp(config["research_start"]) - pd.Timedelta(
        days=max_lookback * 2 + 30
    )
    panels = {}
    catalogs = {}
    for universe_group in config["universe_order"]:
        spec = config["universes"][universe_group]
        if spec["source"] == "get_price":
            if get_price_func is None:
                raise RuntimeError(
                    "jqdata get_price is unavailable; run the fetch in JQ Research"
                )
            members = dict(spec["members"])
            panels[universe_group] = _fetch_index_close_prices(
                members,
                fetch_start,
                config["research_end"],
                get_price_func,
            )
            catalogs[universe_group] = members
        elif spec["source"] == "sw1_daily_price":
            if get_industries_func is None or sw1_price_fetcher is None:
                raise RuntimeError(
                    "JQ sw_l1 APIs are unavailable; run the fetch in JQ Research"
                )
            panel, members = fetch_sw1_close_prices(
                fetch_start,
                config["research_end"],
                get_industries_func,
                sw1_price_fetcher,
            )
            panels[universe_group] = panel
            catalogs[universe_group] = members
        else:
            raise ValueError("provided universes require close= in run_p1")
        if progress_func is not None:
            panel = panels[universe_group]
            progress_func(
                "%s data ready: %d dates x %d assets"
                % (universe_group, len(panel.index), len(panel.columns))
            )
    return panels, catalogs


def _fetch_index_close_prices(members, fetch_start, research_end, get_price_func):
    kwargs = {
        "security": list(members),
        "start_date": fetch_start.strftime("%Y-%m-%d"),
        "end_date": research_end,
        "frequency": "daily",
        "fields": ["close"],
        "skip_paused": False,
        "fq": "pre",
        "panel": False,
        "fill_paused": False,
    }
    try:
        raw = get_price_func(**kwargs)
    except TypeError:
        # Compatibility with a JQ runtime that predates fill_paused.
        kwargs.pop("fill_paused")
        raw = get_price_func(**kwargs)
    return normalize_close_prices(raw, list(members))


def fetch_sw1_close_prices(
    fetch_start,
    research_end,
    get_industries_func,
    sw1_price_fetcher,
):
    """Fetch every historical SW L1 code so taxonomy changes remain explicit."""

    catalog = get_industries_func(name="sw_l1")
    if catalog is None or len(catalog) == 0:
        raise ValueError("get_industries('sw_l1') returned no historical industries")
    members = {}
    for code, row in catalog.iterrows():
        code = str(code)
        if not code.startswith("801"):
            continue
        name = row.get("name", code)
        members[code] = str(name)
    if not members:
        raise ValueError("historical sw_l1 catalog contains no 801xxx codes")
    raw = sw1_price_fetcher(
        list(members),
        pd.Timestamp(fetch_start).strftime("%Y-%m-%d"),
        pd.Timestamp(research_end).strftime("%Y-%m-%d"),
    )
    panel = normalize_close_prices(raw, list(members))
    return panel, members


def _fetch_sw1_prices_from_jq(codes, start_date, end_date):
    """JQ-only SW1_DAILY_PRICE reader, one industry per sub-5000-row query."""

    frames = []
    table = finance.SW1_DAILY_PRICE
    for position, code in enumerate(codes, start=1):
        request = query(table.date, table.code, table.name, table.close).filter(
            table.code == code,
            table.date >= start_date,
            table.date <= end_date,
        ).order_by(table.date)
        frame = finance.run_query(request)
        if len(frame) >= 5000:
            raise ValueError(
                "SW1 query reached the 5000-row boundary for %s; split by date" % code
            )
        frames.append(frame)
        if position % 10 == 0 or position == len(codes):
            print("  sw_l1 prices: %d/%d" % (position, len(codes)))
    if not frames:
        raise ValueError("no SW1 price query was executed")
    result = pd.concat(frames, ignore_index=True)
    if result.empty:
        raise ValueError("SW1_DAILY_PRICE returned no rows")
    duplicated = result.duplicated(["date", "code"])
    if duplicated.any():
        raise ValueError("SW1_DAILY_PRICE returned duplicate date/code rows")
    return result


def _copy_universes(universes):
    copied = {}
    for name, spec in universes.items():
        copied[name] = dict(spec)
        copied[name]["members"] = dict(spec.get("members", {}))
    return copied


def normalize_close_prices(raw, securities=None):
    """Normalize common JQ get_price shapes into a wide close-price panel."""

    if raw is None or len(raw) == 0:
        raise ValueError("get_price returned no rows")
    frame = raw.copy()
    securities = list(securities or [])

    if isinstance(frame.columns, pd.MultiIndex):
        level_zero = frame.columns.get_level_values(0)
        if "close" in level_zero:
            wide = frame["close"].copy()
        else:
            raise ValueError("cannot locate close in MultiIndex columns")
    else:
        flat = frame.reset_index()
        columns = set(flat.columns)
        time_column = next(
            (name for name in ("time", "date", "datetime") if name in columns),
            None,
        )
        if {"code", "close"}.issubset(columns) and time_column is not None:
            wide = flat.pivot_table(
                index=time_column,
                columns="code",
                values="close",
                aggfunc="last",
            )
        elif "close" in frame.columns and len(securities) == 1:
            wide = frame[["close"]].rename(columns={"close": securities[0]})
        elif securities and set(securities).issubset(frame.columns):
            wide = frame[securities].copy()
        else:
            raise ValueError(
                "unsupported get_price result; expected time/code/close long data"
            )

    wide.index = pd.to_datetime(wide.index).normalize()
    wide = wide.groupby(level=0).last().sort_index()
    wide = wide.apply(pd.to_numeric, errors="coerce")
    if securities:
        wide = wide.reindex(columns=securities)
    return wide.replace([np.inf, -np.inf], np.nan)


def compute_signal_panels(close, lookbacks, annualization=252):
    """Build unweighted log-price OLS signals without rounding or filtering."""

    prices = _validate_close_panel(close)
    log_prices = np.log(prices.where(prices > 0.0))
    result = {}
    for lookback in sorted(set(int(value) for value in lookbacks)):
        x = np.arange(lookback, dtype=float)
        centered_x = x - x.mean()
        sxx = float(np.dot(centered_x, centered_x))
        rolling = log_prices.rolling(lookback, min_periods=lookback)
        beta = rolling.apply(
            lambda values: float(np.dot(values, centered_x) / sxx), raw=True
        )
        sum_y = rolling.sum()
        total_ss = log_prices.pow(2).rolling(
            lookback, min_periods=lookback
        ).sum() - sum_y.pow(2) / float(lookback)
        full_window = rolling.count().eq(lookback)
        r2 = beta.pow(2).mul(sxx).div(total_ss.where(total_ss > 1e-14))
        r2 = r2.clip(lower=0.0, upper=1.0)
        r2 = r2.mask(full_window & total_ss.le(1e-14), 0.0).where(full_window)
        annualized_slope = np.exp(
            beta.mul(float(annualization)).clip(lower=-50.0, upper=50.0)
        ).sub(1.0)
        formation_return = prices.div(prices.shift(lookback - 1)).sub(1.0)
        formation_return = formation_return.where(full_window)

        panels = {
            "formation_return": formation_return,
            "annualized_slope": annualized_slope,
            "r2": r2,
            "slope_x_r2": annualized_slope.mul(r2),
        }
        for signal_name, panel in panels.items():
            result[(signal_name, lookback)] = panel.replace(
                [np.inf, -np.inf], np.nan
            )
    return result


def build_forward_returns(close, horizons):
    """Return close[t + H] / close[t] - 1 for every requested horizon."""

    prices = _validate_close_panel(close)
    return {
        int(horizon): prices.shift(-int(horizon)).div(prices).sub(1.0)
        for horizon in sorted(set(int(value) for value in horizons))
    }


def cross_sectional_ic(signal, outcome, min_assets):
    """Calculate daily Spearman Rank IC and secondary Pearson IC."""

    signal, outcome = signal.align(outcome, join="inner", axis=0)
    signal, outcome = signal.align(outcome, join="inner", axis=1)
    valid = signal.notna() & outcome.notna()
    observed = valid.sum(axis=1)
    x = signal.where(valid)
    y = outcome.where(valid)
    rank_ic = x.rank(axis=1, method="average").corrwith(
        y.rank(axis=1, method="average"), axis=1
    )
    pearson_ic = x.corrwith(y, axis=1)
    eligible = observed.ge(int(min_assets))
    return pd.DataFrame(
        {
            "n_assets": observed,
            "rank_ic": rank_ic.where(eligible),
            "pearson_ic": pearson_ic.where(eligible),
        },
        index=signal.index,
    )


def evaluate_ic_grid(signal_panels, forward_returns, config):
    """Evaluate all lookback/horizon cells and return summary plus daily IC."""

    summaries = []
    daily_frames = []
    periods = _period_slices(config)
    for signal_name in config["signals"]:
        for lookback in config["lookbacks"]:
            signal = signal_panels[(signal_name, int(lookback))]
            for horizon in config["horizons"]:
                horizon = int(horizon)
                daily = cross_sectional_ic(
                    signal, forward_returns[horizon], config["min_assets"]
                )
                daily = daily.loc[
                    pd.Timestamp(config["research_start"]) : pd.Timestamp(
                        config["research_end"]
                    )
                ].copy()
                daily.insert(0, "date", daily.index)
                daily.insert(1, "signal", signal_name)
                daily.insert(2, "lookback", int(lookback))
                daily.insert(3, "horizon", horizon)
                daily["period"] = _label_periods(daily["date"], config)
                if (
                    signal_name == config["primary_signal"]
                    and int(lookback) == int(config["primary_lookback"])
                ):
                    daily_frames.append(daily.reset_index(drop=True))

                for period_name, mask in periods.items():
                    period_daily = daily.loc[mask(daily["date"])]
                    for metric in ("rank_ic", "pearson_ic"):
                        summaries.append(
                            {
                                "period": period_name,
                                "signal": signal_name,
                                "lookback": int(lookback),
                                "horizon": horizon,
                                "sample_scheme": "daily_hac",
                                "metric": metric,
                                **summarize_series(
                                    period_daily[metric], hac_lag=horizon - 1
                                ),
                            }
                        )
                        summaries.append(
                            {
                                "period": period_name,
                                "signal": signal_name,
                                "lookback": int(lookback),
                                "horizon": horizon,
                                "sample_scheme": "non_overlap",
                                "metric": metric,
                                **summarize_series(
                                    period_daily[metric].iloc[::horizon], hac_lag=0
                                ),
                            }
                        )

    summary = pd.DataFrame(summaries)
    summary["q_value"] = np.nan
    family = ["period", "sample_scheme", "metric"]
    for _, positions in summary.groupby(family, sort=True).groups.items():
        positions = list(positions)
        summary.loc[positions, "q_value"] = _benjamini_hochberg(
            summary.loc[positions, "p_value"]
        )
    daily_ic = pd.concat(daily_frames, ignore_index=True)
    return summary, daily_ic


def summarize_series(values, hac_lag=0):
    """Describe a series and estimate its mean with Newey-West covariance."""

    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    count = len(series)
    if count == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "icir": np.nan,
            "annualized_icir": np.nan,
            "hac_lag": int(hac_lag),
            "hac_standard_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
        }

    array = series.values.astype(float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else np.nan
    lag = min(max(int(hac_lag), 0), max(count - 1, 0))
    residual = array - mean
    long_run_variance = float(np.dot(residual, residual) / count)
    for offset in range(1, lag + 1):
        covariance = float(
            np.dot(residual[offset:], residual[:-offset]) / count
        )
        weight = 1.0 - offset / float(lag + 1)
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = sqrt(long_run_variance / count) if count > 1 else np.nan
    if standard_error and np.isfinite(standard_error):
        t_stat = mean / standard_error
        p_value = erfc(abs(t_stat) / sqrt(2.0))
        lower = mean - 1.96 * standard_error
        upper = mean + 1.96 * standard_error
    else:
        t_stat = p_value = lower = upper = np.nan
    icir = mean / std if std and np.isfinite(std) else np.nan
    return {
        "n": count,
        "mean": mean,
        "median": float(np.median(array)),
        "std": std,
        "icir": icir,
        "annualized_icir": icir * sqrt(252.0) if np.isfinite(icir) else np.nan,
        "hac_lag": lag,
        "hac_standard_error": standard_error,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci95_lower": lower,
        "ci95_upper": upper,
    }


def build_parameter_plateau(ic_summary):
    """Score neighboring lookbacks so isolated IC peaks remain visible."""

    selected = ic_summary[
        ic_summary["sample_scheme"].eq("daily_hac")
        & ic_summary["metric"].eq("rank_ic")
    ].copy()
    records = []
    keys = ["period", "signal", "horizon"]
    for identity, group in selected.groupby(keys, sort=True):
        group = group.sort_values("lookback").reset_index(drop=True)
        for position, row in group.iterrows():
            neighbors = group.iloc[
                max(0, position - 1) : min(len(group), position + 2)
            ]["mean"].dropna()
            median = neighbors.median() if len(neighbors) else np.nan
            dispersion = neighbors.std(ddof=0) if len(neighbors) else np.nan
            records.append(
                {
                    **dict(zip(keys, identity)),
                    "lookback": int(row["lookback"]),
                    "mean_rank_ic": row["mean"],
                    "neighbor_count": len(neighbors),
                    "neighbor_median_rank_ic": median,
                    "neighbor_dispersion": dispersion,
                    "plateau_score": median - dispersion,
                }
            )
    return pd.DataFrame(records)


def evaluate_primary_groups(signal, forward_returns, config):
    """Evaluate quantile monotonicity for the frozen primary signal/window."""

    daily_records = []
    group_count = int(config["group_count"])
    minimum = max(int(config["min_assets"]), group_count)
    for horizon in config["horizons"]:
        outcome = forward_returns[int(horizon)]
        aligned_signal, aligned_outcome = signal.align(outcome, join="inner", axis=0)
        aligned_signal, aligned_outcome = aligned_signal.align(
            aligned_outcome, join="inner", axis=1
        )
        for date in aligned_signal.index:
            frame = pd.DataFrame(
                {"signal": aligned_signal.loc[date], "outcome": aligned_outcome.loc[date]}
            ).dropna()
            if len(frame) < minimum:
                continue
            ranks = frame["signal"].rank(method="first")
            frame["group"] = pd.qcut(ranks, q=group_count, labels=False) + 1
            for group_number, values in frame.groupby("group")["outcome"]:
                daily_records.append(
                    {
                        "date": date,
                        "horizon": int(horizon),
                        "group": int(group_number),
                        "group_return": values.mean(),
                    }
                )

    daily = pd.DataFrame(daily_records)
    if daily.empty:
        return pd.DataFrame(), pd.DataFrame()
    summaries = []
    diagnostics = []
    periods = _period_slices(config)
    for horizon, horizon_daily in daily.groupby("horizon", sort=True):
        for period_name, mask in periods.items():
            period_daily = horizon_daily.loc[mask(horizon_daily["date"])]
            means = {}
            for group_number in range(1, group_count + 1):
                values = period_daily.loc[
                    period_daily["group"].eq(group_number), "group_return"
                ]
                description = summarize_series(values, hac_lag=int(horizon) - 1)
                means[group_number] = description["mean"]
                summaries.append(
                    {
                        "period": period_name,
                        "horizon": int(horizon),
                        "group": group_number,
                        **description,
                    }
                )
            pivot = period_daily.pivot(
                index="date", columns="group", values="group_return"
            )
            spread = pivot.get(group_count, pd.Series(dtype=float)) - pivot.get(
                1, pd.Series(dtype=float)
            )
            spread_description = summarize_series(
                spread, hac_lag=int(horizon) - 1
            )
            mean_series = pd.Series(means, dtype=float).dropna()
            monotonicity = (
                mean_series.index.to_series().corr(mean_series.rank(), method="pearson")
                if len(mean_series) >= 2
                else np.nan
            )
            diagnostics.append(
                {
                    "period": period_name,
                    "horizon": int(horizon),
                    "group_monotonicity": monotonicity,
                    "top_minus_bottom_mean": spread_description["mean"],
                    "top_minus_bottom_t_stat": spread_description["t_stat"],
                    "top_minus_bottom_p_value": spread_description["p_value"],
                    "top_minus_bottom_n": spread_description["n"],
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(diagnostics)


def evaluate_primary_topk(signal, forward_returns, config):
    """Compare Top1/3/5 selections with the same-date universe mean."""

    records = []
    periods = _period_slices(config)
    for horizon in config["horizons"]:
        outcome = forward_returns[int(horizon)]
        aligned_signal, aligned_outcome = signal.align(outcome, join="inner", axis=0)
        aligned_signal, aligned_outcome = aligned_signal.align(
            aligned_outcome, join="inner", axis=1
        )
        daily_rows = []
        for date in aligned_signal.index:
            frame = pd.DataFrame(
                {"signal": aligned_signal.loc[date], "outcome": aligned_outcome.loc[date]}
            ).dropna()
            if len(frame) < int(config["min_assets"]):
                continue
            frame = frame.sort_values("signal", ascending=False)
            universe_return = frame["outcome"].mean()
            for top_k in config["top_ks"]:
                if int(top_k) > len(frame):
                    continue
                selected_return = frame.head(int(top_k))["outcome"].mean()
                daily_rows.append(
                    {
                        "date": date,
                        "top_k": int(top_k),
                        "selected_return": selected_return,
                        "universe_return": universe_return,
                        "excess_return": selected_return - universe_return,
                    }
                )
        daily = pd.DataFrame(daily_rows)
        for period_name, mask in periods.items():
            period_daily = daily.loc[mask(daily["date"])] if len(daily) else daily
            for top_k in config["top_ks"]:
                sample = period_daily.loc[period_daily["top_k"].eq(int(top_k))]
                selected = summarize_series(
                    sample["selected_return"], hac_lag=int(horizon) - 1
                )
                excess = summarize_series(
                    sample["excess_return"], hac_lag=int(horizon) - 1
                )
                records.append(
                    {
                        "period": period_name,
                        "horizon": int(horizon),
                        "top_k": int(top_k),
                        "n": excess["n"],
                        "selected_mean_return": selected["mean"],
                        "excess_mean_return": excess["mean"],
                        "excess_t_stat": excess["t_stat"],
                        "excess_p_value": excess["p_value"],
                        "excess_hit_rate": (
                            sample["excess_return"].gt(0.0).mean()
                            if len(sample)
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(records)


def evaluate_r2_double_sort(slope, r2, forward_returns, config):
    """Test R2 conditionally in high/low momentum halves, not as a new scalar."""

    cell_records = []
    spread_records = []
    periods = _period_slices(config)
    minimum = max(int(config["min_assets"]), 4)
    for horizon in config["horizons"]:
        outcome = forward_returns[int(horizon)]
        slope_panel, outcome_panel = slope.align(outcome, join="inner", axis=0)
        slope_panel, outcome_panel = slope_panel.align(
            outcome_panel, join="inner", axis=1
        )
        r2_panel = r2.reindex(index=slope_panel.index, columns=slope_panel.columns)
        daily_rows = []
        for date in slope_panel.index:
            frame = pd.DataFrame(
                {
                    "slope": slope_panel.loc[date],
                    "r2": r2_panel.loc[date],
                    "outcome": outcome_panel.loc[date],
                }
            ).dropna()
            if len(frame) < minimum:
                continue
            frame["momentum_half"] = np.where(
                frame["slope"].rank(method="first", pct=True) > 0.5,
                "high",
                "low",
            )
            frame["quality_half"] = np.where(
                frame["r2"].rank(method="first", pct=True) > 0.5,
                "high",
                "low",
            )
            for identity, values in frame.groupby(
                ["momentum_half", "quality_half"]
            )["outcome"]:
                daily_rows.append(
                    {
                        "date": date,
                        "momentum_half": identity[0],
                        "quality_half": identity[1],
                        "cell_return": values.mean(),
                    }
                )
        daily = pd.DataFrame(daily_rows)
        for period_name, mask in periods.items():
            period_daily = daily.loc[mask(daily["date"])] if len(daily) else daily
            for momentum_half in ("low", "high"):
                for quality_half in ("low", "high"):
                    values = period_daily.loc[
                        period_daily["momentum_half"].eq(momentum_half)
                        & period_daily["quality_half"].eq(quality_half),
                        "cell_return",
                    ]
                    description = summarize_series(
                        values, hac_lag=int(horizon) - 1
                    )
                    cell_records.append(
                        {
                            "period": period_name,
                            "horizon": int(horizon),
                            "momentum_half": momentum_half,
                            "quality_half": quality_half,
                            **description,
                        }
                    )
                half = period_daily.loc[
                    period_daily["momentum_half"].eq(momentum_half)
                ]
                pivot = half.pivot(
                    index="date", columns="quality_half", values="cell_return"
                )
                spread = pivot.get("high", pd.Series(dtype=float)) - pivot.get(
                    "low", pd.Series(dtype=float)
                )
                description = summarize_series(
                    spread, hac_lag=int(horizon) - 1
                )
                spread_records.append(
                    {
                        "period": period_name,
                        "horizon": int(horizon),
                        "momentum_half": momentum_half,
                        "high_r2_minus_low_r2": description["mean"],
                        "spread_t_stat": description["t_stat"],
                        "spread_p_value": description["p_value"],
                        "n": description["n"],
                    }
                )
    return pd.DataFrame(cell_records), pd.DataFrame(spread_records)


def build_yearly_primary_ic(daily_ic, config):
    """Summarize primary Rank IC year by year to expose regime dependence."""

    selected = daily_ic[
        daily_ic["signal"].eq(config["primary_signal"])
        & daily_ic["lookback"].eq(int(config["primary_lookback"]))
    ].copy()
    selected["year"] = pd.to_datetime(selected["date"]).dt.year
    records = []
    for (year, horizon), group in selected.groupby(["year", "horizon"], sort=True):
        records.append(
            {
                "year": int(year),
                "horizon": int(horizon),
                **summarize_series(group["rank_ic"], hac_lag=int(horizon) - 1),
            }
        )
    return pd.DataFrame(records)


def build_universe_coverage(close, config):
    """Report missing codes and usable history before interpreting results."""

    records = []
    research_start = pd.Timestamp(config["research_start"])
    research_end = pd.Timestamp(config["research_end"])
    research_dates = close.index.to_series().between(research_start, research_end)
    research_row_count = int(research_dates.sum())
    for code in close.columns:
        values = close[code].dropna()
        research_values = values.loc[
            (values.index >= research_start) & (values.index <= research_end)
        ]
        records.append(
            {
                "code": code,
                "name": config["universe"].get(code, code),
                "observations": len(values),
                "research_observations": len(research_values),
                "has_research_history": bool(len(research_values)),
                "first_date": values.index.min() if len(values) else pd.NaT,
                "last_date": values.index.max() if len(values) else pd.NaT,
                "coverage_ratio": (
                    len(research_values) / float(research_row_count)
                    if research_row_count
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def build_protocol_table(config):
    """Materialize one universe's frozen P1 choices beside numerical outputs."""

    rows = [
        ("research_id", config["research_id"]),
        ("universe_version", config["universe_version"]),
        ("universe_source", config["universe_source"]),
        ("universe_size", len(config["universe"])),
        ("research_range", "%s..%s" % (config["research_start"], config["research_end"])),
        ("signals", ",".join(config["signals"])),
        ("lookbacks", ",".join(map(str, config["lookbacks"]))),
        ("horizons", ",".join(map(str, config["horizons"]))),
        ("annualization", config["annualization"]),
        ("min_assets", config["min_assets"]),
        ("group_count", config["group_count"]),
        ("top_ks", ",".join(map(str, config["top_ks"]))),
        ("primary_cell", "%s/L%d/H%d" % (
            config["primary_signal"],
            config["primary_lookback"],
            config["primary_horizon"],
        )),
        ("signal_timestamp", "close[t]"),
        ("outcome", "close[t+H]/close[t]-1"),
        ("primary_metric", "daily cross-sectional Spearman Rank IC"),
        ("overlap_inference", "Newey-West lag H-1 plus non-overlap sample"),
        ("daily_ic_storage", "frozen primary signal/lookback only"),
        ("tuning_rule", "development proposes; validation decides; locked_oos only confirms"),
        ("excluded_layers", "dynamic pool,timing,stop-loss,filters,execution,costs"),
    ]
    for period_name, bounds in config["periods"].items():
        rows.append(("period_%s" % period_name, "%s..%s" % tuple(bounds)))
    return pd.DataFrame(rows, columns=["item", "value"])


def build_protocol_checks(ic_summary, plateau, group_diagnostics, topk, config):
    """Apply the frozen development/validation gate; OOS stays descriptive."""

    signal = config["primary_signal"]
    lookback = int(config["primary_lookback"])
    horizon = int(config["primary_horizon"])

    def one_value(frame, mask, column):
        values = frame.loc[mask, column].dropna()
        return values.iloc[0] if len(values) else np.nan

    base_mask = (
        ic_summary["signal"].eq(signal)
        & ic_summary["lookback"].eq(lookback)
        & ic_summary["horizon"].eq(horizon)
        & ic_summary["sample_scheme"].eq("daily_hac")
        & ic_summary["metric"].eq("rank_ic")
    )
    development_ic = one_value(
        ic_summary, base_mask & ic_summary["period"].eq("development"), "mean"
    )
    validation_ic = one_value(
        ic_summary, base_mask & ic_summary["period"].eq("validation"), "mean"
    )
    oos_ic = one_value(
        ic_summary, base_mask & ic_summary["period"].eq("locked_oos"), "mean"
    )
    plateau_value = one_value(
        plateau,
        plateau["period"].eq("validation")
        & plateau["signal"].eq(signal)
        & plateau["lookback"].eq(lookback)
        & plateau["horizon"].eq(horizon),
        "neighbor_median_rank_ic",
    )
    group_spread = one_value(
        group_diagnostics,
        group_diagnostics["period"].eq("validation")
        & group_diagnostics["horizon"].eq(horizon),
        "top_minus_bottom_mean",
    )
    top1_excess = one_value(
        topk,
        topk["period"].eq("validation")
        & topk["horizon"].eq(horizon)
        & topk["top_k"].eq(1),
        "excess_mean_return",
    )
    rows = [
        ("development_rank_ic_positive", development_ic, True),
        ("validation_rank_ic_positive", validation_ic, True),
        ("validation_neighbor_plateau_positive", plateau_value, True),
        ("validation_top_minus_bottom_positive", group_spread, True),
        ("validation_top1_excess_positive", top1_excess, True),
        ("locked_oos_rank_ic_positive", oos_ic, False),
    ]
    checks = pd.DataFrame(rows, columns=["check", "value", "is_gate"])
    checks["passed"] = checks["value"].gt(0.0)
    checks["status"] = np.where(
        checks["value"].isna(),
        "insufficient_data",
        np.where(checks["passed"], "pass", "fail"),
    )
    gate = checks.loc[checks["is_gate"], "passed"]
    checks["gate_overall"] = bool(len(gate) and gate.all())
    return checks


def run_p1(
    config=None,
    close=None,
    get_price_func=None,
    get_industries_func=None,
    sw1_price_fetcher=None,
    verbose=True,
):
    """Run P1 independently for broad, complete SW L1, and style universes."""

    config = build_config(config)
    runtime_environment = build_runtime_environment(config)
    progress = None
    if verbose:
        print("P1 runtime environment")
        print(runtime_environment.to_string(index=False))
        total_steps = len(config["universe_order"]) * 8 + 1
        progress = _build_progress_reporter(total_steps)
        progress("started", advance=False)
    if close is None:
        close_panels, catalogs = fetch_close_panels(
            config,
            get_price_func=get_price_func,
            get_industries_func=get_industries_func,
            sw1_price_fetcher=sw1_price_fetcher,
            progress_func=progress,
        )
    else:
        close_panels, catalogs = _normalize_provided_panels(close, config)
        if progress is not None:
            for universe_group in config["universe_order"]:
                panel = close_panels[universe_group]
                progress(
                    "%s provided data ready: %d dates x %d assets"
                    % (universe_group, len(panel.index), len(panel.columns))
                )

    per_universe = {}
    for universe_group in config["universe_order"]:
        spec = config["universes"][universe_group]
        members = catalogs[universe_group]
        group_config = _build_universe_config(config, spec, members)
        per_universe[universe_group] = _run_one_universe(
            group_config,
            close_panels[universe_group],
            universe_group=universe_group,
            progress_func=progress,
        )
    results = _combine_universe_results(per_universe)
    if progress is not None:
        progress("all universe results combined")
    results["runtime_environment"] = runtime_environment
    if verbose:
        print_key_results(results, config)
    return results


def build_runtime_environment(config):
    rows = [
        ("research_id", config["research_id"]),
        ("source_path", "research/momentum_signal_validation/p1_jq_signal_validation.py"),
        ("runtime_logic_hash", build_runtime_logic_hash()),
        ("python_version", sys.version.replace("\n", " ")),
        ("pandas_version", pd.__version__),
        ("numpy_version", np.__version__),
        ("jqdata_imported", bool(JQDATA_AVAILABLE)),
        ("get_price_source", _jq_api_source("get_price")),
        ("get_industries_source", _jq_api_source("get_industries")),
        ("run_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def build_runtime_logic_hash():
    """Hash loaded research functions and frozen universe constants."""

    digest = hashlib.sha256()
    for name in sorted(globals()):
        value = globals()[name]
        if (
            callable(value)
            and hasattr(value, "__code__")
            and getattr(value, "__module__", None) == __name__
        ):
            digest.update(name.encode("utf-8"))
            digest.update(marshal.dumps(value.__code__))
    for value in (BROAD_INDEX_UNIVERSE, STYLE_INDEX_UNIVERSE, DEFAULT_UNIVERSES):
        digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _run_one_universe(
    config,
    close,
    universe_group="universe",
    progress_func=None,
):
    close = normalize_close_prices(close, list(config["universe"]))
    coverage = build_universe_coverage(close, config)
    if coverage["research_observations"].gt(0).sum() < int(config["min_assets"]):
        raise ValueError("fewer than min_assets have close history in the research range")
    if progress_func is not None:
        available = int(coverage["has_research_history"].sum())
        progress_func(
            "%s coverage checked: %d/%d assets have research history"
            % (universe_group, available, len(coverage))
        )

    signals = compute_signal_panels(
        close, config["lookbacks"], annualization=config["annualization"]
    )
    forward = build_forward_returns(close, config["horizons"])
    if progress_func is not None:
        progress_func(
            "%s signals ready: %d panels, %d horizons"
            % (universe_group, len(signals), len(forward))
        )
    ic_summary, daily_ic = evaluate_ic_grid(signals, forward, config)
    plateau = build_parameter_plateau(ic_summary)
    if progress_func is not None:
        grid_cells = (
            len(config["signals"])
            * len(config["lookbacks"])
            * len(config["horizons"])
        )
        progress_func(
            "%s IC grid complete: %d parameter cells"
            % (universe_group, grid_cells)
        )
    primary_key = (config["primary_signal"], int(config["primary_lookback"]))
    group_returns, group_diagnostics = evaluate_primary_groups(
        signals[primary_key], forward, config
    )
    if progress_func is not None:
        progress_func("%s quantile diagnostics complete" % universe_group)
    topk = evaluate_primary_topk(signals[primary_key], forward, config)
    if progress_func is not None:
        progress_func("%s Top-K diagnostics complete" % universe_group)
    primary_lookback = int(config["primary_lookback"])
    double_sort, r2_spreads = evaluate_r2_double_sort(
        signals[("annualized_slope", primary_lookback)],
        signals[("r2", primary_lookback)],
        forward,
        config,
    )
    if progress_func is not None:
        progress_func("%s R2 double-sort complete" % universe_group)
    yearly = build_yearly_primary_ic(daily_ic, config)
    checks = build_protocol_checks(
        ic_summary, plateau, group_diagnostics, topk, config
    )
    if progress_func is not None:
        progress_func("%s protocol checks complete" % universe_group)
    return {
        "protocol": build_protocol_table(config),
        "universe_coverage": coverage,
        "ic_summary": ic_summary,
        "ic_daily": daily_ic,
        "parameter_plateau": plateau,
        "group_returns": group_returns,
        "group_diagnostics": group_diagnostics,
        "topk_summary": topk,
        "r2_double_sort": double_sort,
        "r2_quality_spreads": r2_spreads,
        "yearly_primary_ic": yearly,
        "protocol_checks": checks,
    }


def _normalize_provided_panels(close, config):
    if not isinstance(close, dict):
        if len(config["universe_order"]) != 1:
            raise ValueError("close must map every universe_group to a DataFrame")
        close = {config["universe_order"][0]: close}
    panels = {}
    catalogs = {}
    for universe_group in config["universe_order"]:
        if universe_group not in close:
            raise ValueError("missing provided close panel: %s" % universe_group)
        members = dict(config["universes"][universe_group].get("members", {}))
        if not members:
            members = {
                str(column): str(column) for column in close[universe_group].columns
            }
        panels[universe_group] = normalize_close_prices(
            close[universe_group], list(members)
        )
        catalogs[universe_group] = members
    return panels, catalogs


def _build_universe_config(config, spec, members):
    result = dict(config)
    result["universe"] = dict(members)
    result["min_assets"] = int(spec["min_assets"])
    result["universe_version"] = spec["version"]
    result["universe_source"] = spec["source"]
    return result


def _combine_universe_results(per_universe):
    combined = {}
    table_names = list(next(iter(per_universe.values())).keys())
    for table_name in table_names:
        frames = []
        for universe_group, result in per_universe.items():
            frame = result[table_name].copy()
            frame.insert(0, "universe_group", universe_group)
            frames.append(frame)
        combined[table_name] = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def print_key_results(results, config):
    """Print only the frozen cell and gate; detailed tables stay in RESULTS."""

    horizon = int(config["primary_horizon"])
    selected = results["ic_summary"]
    selected = selected[
        selected["signal"].eq(config["primary_signal"])
        & selected["lookback"].eq(int(config["primary_lookback"]))
        & selected["horizon"].eq(horizon)
        & selected["sample_scheme"].eq("daily_hac")
        & selected["metric"].eq("rank_ic")
    ][
        [
            "universe_group",
            "period",
            "n",
            "mean",
            "annualized_icir",
            "t_stat",
            "p_value",
            "q_value",
        ]
    ]
    print("\nP1 frozen primary cell")
    print(selected.to_string(index=False))
    print("\nP1 protocol checks")
    print(results["protocol_checks"].to_string(index=False))
    print("\nAll detailed outputs are available in RESULTS by table name.")


def export_results(results, prefix="momentum_signal_p1"):
    """Optionally write every result table to CSV in the JQ notebook workspace."""

    written = []
    for name, frame in results.items():
        if isinstance(frame, pd.DataFrame):
            filename = "%s__%s.csv" % (prefix, name)
            frame.to_csv(filename, index=False)
            written.append(filename)
    return written


def _validate_close_panel(close):
    if not isinstance(close, pd.DataFrame) or close.empty:
        raise ValueError("close must be a non-empty DataFrame")
    result = close.copy()
    result.index = pd.to_datetime(result.index).normalize()
    if result.index.has_duplicates:
        raise ValueError("close index must not contain duplicate dates")
    result = result.sort_index().apply(pd.to_numeric, errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def _period_slices(config):
    slices = {
        name: (
            lambda dates, start=pd.Timestamp(bounds[0]), end=pd.Timestamp(bounds[1]):
            pd.to_datetime(dates).between(start, end)
        )
        for name, bounds in config["periods"].items()
    }
    start = pd.Timestamp(config["research_start"])
    end = pd.Timestamp(config["research_end"])
    slices["all"] = lambda dates: pd.to_datetime(dates).between(start, end)
    return slices


def _label_periods(dates, config):
    labels = pd.Series(np.nan, index=dates.index, dtype="object")
    parsed = pd.to_datetime(dates)
    for name, bounds in config["periods"].items():
        labels.loc[parsed.between(pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))] = name
    return labels


def _benjamini_hochberg(values):
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna().clip(lower=0.0, upper=1.0)
    if valid.empty:
        return result
    ordered = valid.sort_values()
    count = len(ordered)
    adjusted = ordered.values * count / np.arange(1, count + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return result
