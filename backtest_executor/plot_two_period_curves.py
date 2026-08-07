"""Plot stitched equity curves for the old and 2026Q2 backtest ranges.

Run this file/cell inside JoinQuant/JQData environment so get_backtest(bt_id)
and get_backtest(bt_id).get_results() are available.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from jqdata import get_backtest
except ImportError:
    get_backtest = None


OLD_MAPPER = "backtest_executor/results/ETF_7star_opt_dynamic/mapper.json"
Q2_MAPPER = "backtest_executor/results/ETF_7star_opt_dynamic/mapper_2026Q2.json"
OUTPUT_FILE = "backtest_executor/results/ETF_7star_opt_dynamic/two_period_top_curves.png"
TOP_N = 5
START_DATE = pd.Timestamp("2023-01-01")
SPLIT_DATE = pd.Timestamp("2026-03-01")
END_DATE = pd.Timestamp("2026-06-30")


def load_done(path):
    path = Path(path)
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"].values()
    return {
        json.dumps(r["params"], sort_keys=True, ensure_ascii=False): r
        for r in runs
        if r.get("status") == "done"
    }


def fetch_curve(bt_id):
    if get_backtest is None:
        raise RuntimeError("get_backtest is unavailable; run this in the JQ environment.")

    rows = get_backtest(bt_id).get_results()
    df = pd.DataFrame(rows)
    if "time" not in df or "returns" not in df:
        raise ValueError(f"Backtest {bt_id} results missing time/returns columns.")
    df["time"] = pd.to_datetime(df["time"])
    df["returns"] = pd.to_numeric(df["returns"], errors="coerce")
    return df.dropna(subset=["time", "returns"]).sort_values("time").set_index("time")["returns"]


def stitch(old_curve, q2_curve):
    if old_curve.index.min() >= SPLIT_DATE:
        raise ValueError(
            "旧区间曲线没有 2026-03-01 之前的数据；请确认 OLD_MAPPER 指向 2023-01-01 ~ 2026-03-01 的 mapper.json。"
        )
    if q2_curve.index.max() <= SPLIT_DATE:
        raise ValueError(
            "Q2 曲线没有 2026-03-01 之后的数据；请确认 Q2_MAPPER 指向 2026-03-01 ~ 2026-06-30 的 mapper_2026Q2.json。"
        )
    old_nav = 1 + old_curve
    q2_nav = old_nav.iloc[-1] * (1 + q2_curve)
    q2_nav = q2_nav[q2_nav.index > old_nav.index[-1]]
    curve = pd.concat([old_nav, q2_nav]) - 1
    return curve[(curve.index >= START_DATE) & (curve.index <= END_DATE)]


def combined_return(old_run, q2_run):
    return (1 + old_run["metrics"]["return"]) * (1 + q2_run["metrics"]["return"]) - 1


def short_label(run):
    p = run["params"]
    s = p["EXECUTION_SCORE_THRESHOLD"]
    r = p["EXECUTION_R2_PARAM"]
    v = p["EXECUTION_VOLUME_PARAM"]
    fm = p["EXECUTION_FLITER_MARKET"]
    r_text = "r-" if not r[0] else f"r{r[1]:g}"
    v_text = "v-" if not v[0] else f"v{v[2]:g}"
    fm_text = "fm" + "".join("T" if x else "F" for x in fm)
    return f"S{s[0]}-{s[1]} {r_text} {v_text} {fm_text}"


def plot_curve(ax, curve, label, color, focus, linestyle="-"):
    old_part = curve[curve.index <= SPLIT_DATE]
    q2_part = curve[curve.index >= SPLIT_DATE]

    if focus == "old":
        ax.plot(old_part.index, 1 + old_part, color=color, linestyle=linestyle, linewidth=2.2, alpha=0.95, label=label)
        ax.plot(q2_part.index, 1 + q2_part, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.25, label="_nolegend_")
    elif focus == "q2":
        ax.plot(old_part.index, 1 + old_part, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.25, label="_nolegend_")
        ax.plot(q2_part.index, 1 + q2_part, color=color, linestyle=linestyle, linewidth=2.2, alpha=0.95, label=label)
    else:
        ax.plot(curve.index, 1 + curve, color=color, linestyle=linestyle, linewidth=1.8, alpha=0.85, label=label)


def finish_axis(ax, title):
    ax.axvline(SPLIT_DATE, color="gray", linestyle="--", linewidth=1, alpha=0.65)
    ax.set_xlim(START_DATE, END_DATE)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Net value (log)")
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=7, loc="upper left")


old = load_done(OLD_MAPPER)
q2 = load_done(Q2_MAPPER)
keys = sorted(set(old) & set(q2))
print(f"OLD_MAPPER: {OLD_MAPPER}, done={len(old)}")
print(f"Q2_MAPPER:  {Q2_MAPPER}, done={len(q2)}")
print(f"matched params: {len(keys)}")
if not keys:
    raise ValueError("两个 mapper 没有匹配的参数组合。")

same_bt_ids = sum(1 for k in keys if old[k].get("bt_id") == q2[k].get("bt_id"))
if same_bt_ids:
    raise ValueError(
        f"旧区间和 Q2 有 {same_bt_ids} 个匹配参数使用了相同 bt_id；"
        "这通常表示 OLD_MAPPER 和 Q2_MAPPER 读到了同一个结果文件。"
    )

old_top = sorted(keys, key=lambda k: old[k]["metrics"]["return"], reverse=True)[:TOP_N]
q2_top = sorted(keys, key=lambda k: q2[k]["metrics"]["return"], reverse=True)[:TOP_N]
combined_top = sorted(keys, key=lambda k: combined_return(old[k], q2[k]), reverse=True)[:TOP_N]

curve_cache = {}


def curve_for(key):
    if key not in curve_cache:
        curve_cache[key] = stitch(fetch_curve(old[key]["bt_id"]), fetch_curve(q2[key]["bt_id"]))
    return curve_cache[key]


fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
colors = plt.get_cmap("tab10").colors

for i, key in enumerate(old_top, 1):
    label = f"#{i} {short_label(old[key])} old={old[key]['metrics']['return']:.1%}, q2={q2[key]['metrics']['return']:.1%}"
    plot_curve(axes[0], curve_for(key), label, colors[(i - 1) % len(colors)], "old")

for i, key in enumerate(q2_top, 1):
    label = f"#{i} {short_label(old[key])} q2={q2[key]['metrics']['return']:.1%}, old={old[key]['metrics']['return']:.1%}"
    plot_curve(axes[0], curve_for(key), label, colors[(i + len(old_top) - 1) % len(colors)], "q2", linestyle="--")
finish_axis(axes[0], f"Old top {len(old_top)} and 2026Q2 top {len(q2_top)}; focused segment is bold")

for i, key in enumerate(combined_top, 1):
    label = f"#{i} {short_label(old[key])} combined={combined_return(old[key], q2[key]):.1%}"
    plot_curve(axes[1], curve_for(key), label, colors[(i - 1) % len(colors)], "all")
finish_axis(axes[1], f"Combined top {len(combined_top)} by compounded return")
axes[1].set_xlabel("Date")

fig.suptitle("ETF_7star_opt_dynamic return curves: old range, 2026Q2, and combined winners", y=0.995)
fig.tight_layout()
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, dpi=160)
plt.show()
print(OUTPUT_FILE)
