# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


ROOT = Path(__file__).resolve().parents[2]
DATA_MID = ROOT / "data" / "第3问" / "中间数据"
DATA_OUT = ROOT / "data" / "第3问" / "结果数据"
OUT_DIR = ROOT / "figure" / "第3问" / "局部新颖PNG"

TARGET_COL = "烧结大烟道外排CO浓度"
STANDARD_CO = 2800.0
BOX_IDS = list(range(1, 19))
FIG_DPI = 320

COLORS = {
    "blue": "#2F6B9A",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "red": "#C44E52",
    "purple": "#7B61A8",
    "gray": "#6E7781",
    "dark": "#263238",
    "light": "#E8EEF2",
}

FIGURE_NOTES = [
    ("新颖_01_300s时滞控制病例册.png", "把控制时刻t、目标时刻t+300s、CO、负压热图和方案B调节指纹放在同一局部病例中。"),
    ("新颖_02_滚动优化收益事件指纹.png", "从最新滚动窗口中抽取预测降幅最大的局部事件，展示逐风箱调节指纹。"),
    ("新颖_03_低排放片段对齐动态指纹.png", "将两个稳定低排放片段按起点对齐，比较CO、主抽风、后段温度和负压模板偏移。"),
    ("新颖_04_最长历史达标片段局部透镜.png", "聚焦最长可信达标片段，展示片段前后CO、负压时空扰动和热工状态。"),
    ("新颖_05_规划候选Pareto前沿与方案定位.png", "用Q50、Q90、总调节量展示候选解多目标结构，并标出方案0/A/B/C位置。"),
    ("新颖_06_达标模板距离门控相图.png", "展示候选解的达标模板距离、低排放工况层概率和Q50/Q90风险之间的关系。"),
    ("新颖_07_当前方案三段协同调节重心.png", "把18个风箱压缩成前段/中段/尾段协同调节，解释方案调整重心。"),
    ("新颖_08_未来300s相空间密度与当前轨迹.png", "用当前CO状态与300s后CO的相空间密度，定位当前控制案例。"),
    ("新颖_09_调压温度联动局部不一致地图.png", "逐风箱展示方案负压调整与第一问温度增量响应的局部不一致。"),
]


def setup_style() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 130,
            "savefig.dpi": FIG_DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.75,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"缺少必要文件: {path}")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require_path(path), encoding="utf-8-sig")


def save_fig(fig: plt.Figure, filename: str) -> Path:
    path = OUT_DIR / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def numeric(series: pd.Series | np.ndarray) -> np.ndarray:
    return pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "是"])


def short_scheme(value: object) -> str:
    text = str(value)
    if "方案0" in text:
        return "方案0基准"
    if "方案A" in text:
        return "方案A Q50"
    if "方案B" in text:
        return "方案B Q90主推"
    if "方案C" in text:
        return "方案C平稳"
    return text.replace("_", " ")


def scheme_order(df: pd.DataFrame, col: str = "方案") -> pd.DataFrame:
    order = {"方案0": 0, "方案A": 1, "方案B": 2, "方案C": 3}
    out = df.copy()
    out["_order"] = out[col].map(lambda x: next((v for k, v in order.items() if k in str(x)), 99))
    return out.sort_values("_order").drop(columns="_order")


def pressure_cols(kind: str = "强度") -> list[str]:
    return [f"{i}#风箱负压{kind if kind else ''}" for i in BOX_IDS]


def clean_pressure_matrix(df: pd.DataFrame, cols: list[str] | None = None) -> np.ndarray:
    if cols is None:
        cols = [f"{i}#风箱负压强度" for i in BOX_IDS]
    arr = df[cols].to_numpy(dtype=float)
    if "强度" not in cols[0]:
        arr = np.abs(arr)
    return arr


def box_delta_matrix(df: pd.DataFrame, target_prefix: str, base_prefix: str) -> np.ndarray:
    target_cols = [f"{i}#{target_prefix}" for i in BOX_IDS]
    base_cols = [f"{i}#{base_prefix}" for i in BOX_IDS]
    return df[target_cols].to_numpy(dtype=float) - df[base_cols].to_numpy(dtype=float)


def find_latest_case(summary: pd.DataFrame) -> tuple[int, int]:
    base = summary[summary["方案"].astype(str).str.contains("方案0")].iloc[0]
    return int(base["控制时刻序列_t"]), int(base["预测目标序列_t+300s"])


def plot_lag_casebook() -> Path:
    cleaned = read_csv(DATA_MID / "清洗后数据_含低排放片段.csv").sort_values("序列")
    summary = scheme_order(read_csv(DATA_OUT / "三类调控方案汇总.csv"))
    combo = scheme_order(read_csv(DATA_OUT / "三类调控方案风箱负压组合.csv"))
    t_seq, target_seq = find_latest_case(summary)
    window = cleaned[(cleaned["序列"] >= t_seq - 120) & (cleaned["序列"] <= target_seq + 80)].copy()
    window["相对控制时刻_s"] = (window["序列"] - t_seq) * 2
    co_col = "第三问原始CO" if "第三问原始CO" in window.columns else TARGET_COL
    pressure = clean_pressure_matrix(window)
    pressure_smooth = gaussian_filter(pressure.T, sigma=(0.55, 1.0), mode="nearest")
    b_combo = combo[combo["方案"].astype(str).str.contains("方案B")].sort_values("风箱编号")
    b_summary = summary[summary["方案"].astype(str).str.contains("方案B")].iloc[0]
    base_summary = summary[summary["方案"].astype(str).str.contains("方案0")].iloc[0]

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(window["相对控制时刻_s"], window[co_col], color=COLORS["dark"], lw=1.5, label="实际CO")
    ax.axhline(STANDARD_CO, color=COLORS["red"], ls="--", lw=1.1, label="2800标准")
    ax.axvline(0, color=COLORS["blue"], lw=1.4, label="控制时刻t")
    ax.axvline(300, color=COLORS["green"], lw=1.4, label="目标时刻t+300s")
    ax.fill_betweenx(ax.get_ylim(), 0, 300, color=COLORS["green"], alpha=0.07)
    ax.set_title("A  控制作用时滞局部病例：t到t+300s")
    ax.set_xlabel("相对控制时刻 (s)")
    ax.set_ylabel("CO (mg/m$^3$)")
    ax.legend(ncol=2, frameon=False, loc="upper left")

    ax = fig.add_subplot(gs[1, 0], sharex=ax)
    extent = [window["相对控制时刻_s"].min(), window["相对控制时刻_s"].max(), 0.5, 18.5]
    im = ax.imshow(pressure_smooth, aspect="auto", origin="lower", extent=extent, cmap="cividis")
    ax.axvline(0, color="white", lw=1.1)
    ax.axvline(300, color="white", lw=1.1)
    ax.set_title("B  局部风箱负压强度时空热图")
    ax.set_xlabel("相对控制时刻 (s)")
    ax.set_ylabel("风箱编号")
    ax.set_yticks([1, 4, 8, 12, 16, 18])
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("负压强度 |KPa|")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(window["相对控制时刻_s"], window["后段风箱温度均值"], color=COLORS["purple"], lw=1.6, label="后段风箱温度均值")
    ax.plot(window["相对控制时刻_s"], window["1#大烟道温度"], color=COLORS["orange"], lw=1.4, label="1#大烟道温度")
    ax.plot(window["相对控制时刻_s"], window["2#大烟道温度"], color=COLORS["sky"], lw=1.4, label="2#大烟道温度")
    ax.axvline(0, color=COLORS["blue"], lw=1.1)
    ax.axvline(300, color=COLORS["green"], lw=1.1)
    ax.set_title("C  热状态在控制窗口内的局部轨迹")
    ax.set_xlabel("相对控制时刻 (s)")
    ax.set_ylabel("温度 (℃)")
    ax.legend(frameon=False, loc="best")

    ax = fig.add_subplot(gs[1, 1])
    boxes = np.array(BOX_IDS)
    delta = b_combo["相对控制时刻实际强度变化"].to_numpy(dtype=float)
    colors = np.where(delta >= 0, COLORS["green"], COLORS["orange"])
    ax.bar(boxes, delta, color=colors, alpha=0.88)
    ax.axhline(0, color=COLORS["dark"], lw=1)
    ax2 = ax.twinx()
    q_labels = ["Q10", "Q50", "Q90"]
    q_delta = [b_summary["ΔQ10"], b_summary["ΔQ50"], b_summary["ΔQ90"]]
    ax2.plot([3, 9, 15], q_delta, marker="D", color=COLORS["red"], lw=1.8, label="方案B分位数变化")
    for x0, y0, label in zip([3, 9, 15], q_delta, q_labels):
        ax2.text(x0, y0, f"{label}\n{y0:.1f}", ha="center", va="bottom" if y0 >= 0 else "top", fontsize=8)
    ax.set_title("D  方案B局部调压指纹与CO分位数变化")
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("Delta P_i |KPa|")
    ax2.set_ylabel("Delta CO分位数 (mg/m$^3$)")
    ax.set_xticks(BOX_IDS)
    return save_fig(fig, FIGURE_NOTES[0][0])


def plot_rolling_event_fingerprints() -> Path:
    traj = read_csv(DATA_OUT / "最新窗口滚动调控轨迹.csv").sort_values("时间_s").reset_index(drop=True)
    traj["窗口内时间_min"] = (traj["时间_s"] - traj["时间_s"].min()) / 60.0
    top = traj.sort_values("模型预测降低量", ascending=False).head(10).sort_values("时间_s").copy()
    delta = box_delta_matrix(top, "滚动目标强度", "控制时刻实际强度")
    vmax = max(0.25, float(np.nanmax(np.abs(delta))))

    fig = plt.figure(figsize=(14.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.25])
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(traj["窗口内时间_min"], traj["模型预测降低量"], color=COLORS["green"], lw=1.7, label="模型预测降低量")
    ax.fill_between(traj["窗口内时间_min"], traj["模型预测降低量"], 0, color=COLORS["green"], alpha=0.18)
    ax.scatter(top["窗口内时间_min"], top["模型预测降低量"], s=58, color=COLORS["orange"], zorder=4, label="Top10局部事件")
    for _, row in top.iterrows():
        ax.text(row["窗口内时间_min"], row["模型预测降低量"] + 8, str(int(row["序列"])), ha="center", fontsize=8)
    ax.axhline(0, color=COLORS["dark"], lw=1)
    ax.set_title("A  滚动窗口中预测降幅最大的局部事件")
    ax.set_xlabel("窗口内时间 (min)")
    ax.set_ylabel("预测降低量 (mg/m$^3$)")
    ax.legend(frameon=False, loc="upper right")

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(delta, aspect="auto", cmap="PuOr_r", norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    ax.set_title("B  Top10事件逐风箱调节指纹（正值：增强负压）")
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("事件序列 / 降低量")
    ax.set_xticks(np.arange(len(BOX_IDS)))
    ax.set_xticklabels(BOX_IDS)
    labels = [f"{int(r['序列'])}\n{r['模型预测降低量']:.0f}" for _, r in top.iterrows()]
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Delta P_i |KPa|")
    return save_fig(fig, FIGURE_NOTES[1][0])


def plot_low_emission_alignment() -> Path:
    cleaned = read_csv(DATA_MID / "清洗后数据_含低排放片段.csv").sort_values("序列")
    intervals = read_csv(DATA_OUT / "低排放工况区间识别.csv")
    co_col = "第三问原始CO" if "第三问原始CO" in cleaned.columns else TARGET_COL
    global_median = cleaned[[f"{i}#风箱负压强度" for i in BOX_IDS]].median().to_numpy(dtype=float)
    rows = []
    pressure_rows = []
    for _, seg in intervals.iterrows():
        start = int(seg["起始序列"])
        stop = int(seg["结束序列"])
        part = cleaned[(cleaned["序列"] >= start - 20) & (cleaned["序列"] <= stop + 20)].copy()
        part["相对片段起点_s"] = (part["序列"] - start) * 2
        part["片段编号"] = int(seg["片段编号"])
        rows.append(part)
        core = cleaned[(cleaned["序列"] >= start) & (cleaned["序列"] <= stop)]
        pressure_rows.append(core[[f"{i}#风箱负压强度" for i in BOX_IDS]].median().to_numpy(dtype=float) - global_median)
    aligned = pd.concat(rows, ignore_index=True)
    pdev = np.vstack(pressure_rows)
    vmax = max(0.2, float(np.nanmax(np.abs(pdev))))

    fig = plt.figure(figsize=(14.2, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    for seg_id, part in aligned.groupby("片段编号"):
        axes[0].plot(part["相对片段起点_s"], part[co_col], lw=1.8, marker="o", ms=3, label=f"片段{seg_id}")
    axes[0].axhline(STANDARD_CO, color=COLORS["red"], ls="--", lw=1.1)
    axes[0].set_title("A  低排放片段CO对齐轨迹")
    axes[0].set_xlabel("相对片段起点 (s)")
    axes[0].set_ylabel("CO (mg/m$^3$)")
    axes[0].legend(frameon=False)

    for seg_id, part in aligned.groupby("片段编号"):
        flue_mean = (part["1#大烟道负压强度"] + part["2#大烟道负压强度"]) / 2
        axes[1].plot(part["相对片段起点_s"], flue_mean, lw=1.7, marker="s", ms=3, label=f"片段{seg_id}")
    axes[1].set_title("B  主抽风负压强度对齐轨迹")
    axes[1].set_xlabel("相对片段起点 (s)")
    axes[1].set_ylabel("大烟道负压强度均值 |KPa|")
    axes[1].legend(frameon=False)

    for seg_id, part in aligned.groupby("片段编号"):
        axes[2].plot(part["相对片段起点_s"], part["后段风箱温度均值"], lw=1.7, marker="^", ms=3, label=f"片段{seg_id}")
    axes[2].set_title("C  后段热状态对齐轨迹")
    axes[2].set_xlabel("相对片段起点 (s)")
    axes[2].set_ylabel("后段风箱温度均值 (℃)")
    axes[2].legend(frameon=False)

    im = axes[3].imshow(pdev, aspect="auto", cmap="PuOr_r", norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    axes[3].set_title("D  片段内负压模板相对全局中位数偏移")
    axes[3].set_xlabel("风箱编号")
    axes[3].set_ylabel("低排放片段")
    axes[3].set_xticks(np.arange(len(BOX_IDS)))
    axes[3].set_xticklabels(BOX_IDS)
    axes[3].set_yticks(np.arange(len(intervals)))
    axes[3].set_yticklabels([f"片段{int(v)}" for v in intervals["片段编号"]])
    axes[3].grid(False)
    cbar = fig.colorbar(im, ax=axes[3], pad=0.01)
    cbar.set_label("相对全局中位数偏移 |KPa|")
    return save_fig(fig, FIGURE_NOTES[2][0])


def plot_longest_compliant_lens() -> Path:
    cleaned = read_csv(DATA_MID / "清洗后数据_含低排放片段.csv").sort_values("序列")
    segments = read_csv(DATA_OUT / "历史达标工况连续片段明细.csv")
    seg = segments.sort_values(["样本数", "持续时间_s"], ascending=False).iloc[0]
    start = int(seg["起始序列"])
    stop = int(seg["结束序列"])
    window = cleaned[(cleaned["序列"] >= start - 55) & (cleaned["序列"] <= stop + 55)].copy()
    window["相对片段起点_s"] = (window["序列"] - start) * 2
    co_col = "第三问原始CO" if "第三问原始CO" in window.columns else TARGET_COL
    pmat = clean_pressure_matrix(window)
    pdev = (pmat - np.nanmedian(pmat, axis=0)).T
    pdev = gaussian_filter(pdev, sigma=(0.4, 0.9), mode="nearest")
    vmax = max(0.25, float(np.nanquantile(np.abs(pdev), 0.98)))

    before = cleaned[(cleaned["序列"] >= start - 55) & (cleaned["序列"] < start)]
    core = cleaned[(cleaned["序列"] >= start) & (cleaned["序列"] <= stop)]
    after = cleaned[(cleaned["序列"] > stop) & (cleaned["序列"] <= stop + 55)]

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(window["相对片段起点_s"], window[co_col], color=COLORS["dark"], lw=1.5)
    ax.axhline(STANDARD_CO, color=COLORS["red"], ls="--", lw=1.1, label="2800标准")
    ax.axvspan(0, (stop - start) * 2, color=COLORS["green"], alpha=0.18, label=f"最长达标片段 {start}-{stop}")
    ax.set_title("A  最长历史达标片段CO局部透镜")
    ax.set_xlabel("相对片段起点 (s)")
    ax.set_ylabel("CO (mg/m$^3$)")
    ax.legend(frameon=False, loc="upper left")

    ax = fig.add_subplot(gs[1, 0], sharex=ax)
    extent = [window["相对片段起点_s"].min(), window["相对片段起点_s"].max(), 0.5, 18.5]
    im = ax.imshow(pdev, aspect="auto", origin="lower", extent=extent, cmap="PuOr_r", norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    ax.axvspan(0, (stop - start) * 2, color="white", alpha=0.16)
    ax.set_title("B  片段前后负压扰动热图（相对本窗口中位数）")
    ax.set_xlabel("相对片段起点 (s)")
    ax.set_ylabel("风箱编号")
    ax.set_yticks([1, 4, 8, 12, 16, 18])
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Delta P |KPa|")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(window["相对片段起点_s"], window["烧结机机速L1设定"], color=COLORS["blue"], lw=1.4, label="机速")
    ax2 = ax.twinx()
    ax2.plot(window["相对片段起点_s"], window["1#大烟道温度"], color=COLORS["orange"], lw=1.35, label="1#大烟道温度")
    ax2.plot(window["相对片段起点_s"], window["2#大烟道温度"], color=COLORS["sky"], lw=1.35, label="2#大烟道温度")
    ax.axvspan(0, (stop - start) * 2, color=COLORS["green"], alpha=0.12)
    ax.set_title("C  片段前后机速与大烟道温度")
    ax.set_xlabel("相对片段起点 (s)")
    ax.set_ylabel("机速")
    ax2.set_ylabel("温度 (℃)")
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, frameon=False, loc="best")

    ax = fig.add_subplot(gs[1, 1])
    boxes = np.array(BOX_IDS)
    for data, label, color, marker in [
        (before, "片段前", COLORS["gray"], "o"),
        (core, "达标片段", COLORS["green"], "s"),
        (after, "片段后", COLORS["orange"], "^"),
    ]:
        vals = data[[f"{i}#风箱负压强度" for i in BOX_IDS]].median().to_numpy(dtype=float)
        ax.plot(boxes, vals, marker=marker, color=color, lw=1.9, label=label)
    ax.set_title("D  片段前-中-后风箱负压空间剖面")
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("负压强度 |KPa|")
    ax.set_xticks(BOX_IDS)
    ax.legend(frameon=False, loc="best")
    return save_fig(fig, FIGURE_NOTES[3][0])


def pareto_front(points: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    data = points.dropna(subset=cols).sort_values(cols).copy()
    keep = []
    best_q90 = np.inf
    best_total = np.inf
    for idx, row in data.iterrows():
        q90 = float(row[cols[1]])
        total = float(row[cols[2]])
        if q90 < best_q90 or total < best_total:
            keep.append(idx)
            best_q90 = min(best_q90, q90)
            best_total = min(best_total, total)
    return data.loc[keep].sort_values(cols[0])


def plot_pareto_candidates() -> Path:
    search = read_csv(DATA_OUT / "主模型_Q50_Q90_概率三层规划搜索记录.csv")
    summary = scheme_order(read_csv(DATA_OUT / "三类调控方案汇总.csv"))
    search["硬筛选"] = bool_series(search["规划约束_Q50是否低于2800"]) & bool_series(search["规划约束_Q90是否不高于基准"]) & bool_series(search["规划约束_Pr是否达标"])
    q50 = numeric(search["Q50_CO"])
    q90 = numeric(search["Q90_CO"])
    total = numeric(search["总调节量"])
    feasible = search["硬筛选"].to_numpy(dtype=bool)
    front = pareto_front(search[search["硬筛选"]].copy(), ["Q50_CO", "Q90_CO", "总调节量"])

    fig = plt.figure(figsize=(14.2, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    sc = ax.scatter(q50[~feasible], q90[~feasible], c=total[~feasible], cmap="Greys", s=22, alpha=0.28, label="未通过三层筛选")
    sc2 = ax.scatter(q50[feasible], q90[feasible], c=total[feasible], cmap="viridis", s=36, alpha=0.86, label="通过三层筛选")
    if not front.empty:
        ax.plot(front["Q50_CO"], front["Q90_CO"], color=COLORS["red"], lw=2.0, label="近似Pareto前沿")
    for _, row in summary.iterrows():
        ax.scatter(row["Q50_CO"], row["Q90_CO"], s=130, marker="*", edgecolor="white", linewidth=0.8, zorder=5)
        ax.text(row["Q50_CO"] + 7, row["Q90_CO"] + 7, short_scheme(row["方案"]), fontsize=9)
    base_q90 = float(summary.loc[summary["方案"].astype(str).str.contains("方案0"), "Q90_CO"].iloc[0])
    ax.axvline(STANDARD_CO, color=COLORS["red"], ls="--", lw=1.1, label="Q50=2800")
    ax.axhline(base_q90, color=COLORS["gray"], ls=":", lw=1.2, label="基准Q90")
    ax.set_title("A  规划候选解：Q50目标-Q90风险-Pareto结构")
    ax.set_xlabel("Q50_CO (mg/m$^3$)")
    ax.set_ylabel("Q90_CO (mg/m$^3$)")
    ax.legend(frameon=False, loc="upper left")
    cbar = fig.colorbar(sc2, ax=ax, pad=0.012)
    cbar.set_label("总调节量")

    ax = fig.add_subplot(gs[0, 1])
    labels = summary["方案"].map(short_scheme).to_list()
    x = np.arange(len(summary))
    width = 0.28
    ax.bar(x - width, summary["Q50_CO"] - STANDARD_CO, width=width, color=COLORS["blue"], alpha=0.82, label="Q50-2800")
    ax.bar(x, summary["Q90_CO"] - base_q90, width=width, color=COLORS["orange"], alpha=0.82, label="Q90-基准Q90")
    ax.bar(x + width, summary["严格达标概率Pr_CO_le_2800"] - summary["规划约束_概率阈值θ"], width=width, color=COLORS["green"], alpha=0.82, label="Pr-阈值")
    ax.axhline(0, color=COLORS["dark"], lw=1)
    ax.set_title("B  方案离三层约束边界的距离")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel("边界差值")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(np.nanmin(total), np.nanmax(total), 18)
    ax.hist(total[~feasible], bins=bins, color=COLORS["gray"], alpha=0.45, label="未通过")
    ax.hist(total[feasible], bins=bins, color=COLORS["green"], alpha=0.68, label="通过")
    ax.set_title("C  总调节量分布：可行候选并不等于大幅调节")
    ax.set_xlabel("总调节量")
    ax.set_ylabel("候选数")
    ax.legend(frameon=False)
    return save_fig(fig, FIGURE_NOTES[4][0])


def nearest_search_rows(summary: pd.DataFrame, search: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, srow in summary.iterrows():
        dist = (search["Q50_CO"] - srow["Q50_CO"]).pow(2) + (search["Q90_CO"] - srow["Q90_CO"]).pow(2)
        hit = search.loc[dist.idxmin()].copy()
        hit["方案"] = srow["方案"]
        rows.append(hit)
    return pd.DataFrame(rows)


def plot_gate_phase_map() -> Path:
    search = read_csv(DATA_OUT / "主模型_Q50_Q90_概率三层规划搜索记录.csv")
    summary = scheme_order(read_csv(DATA_OUT / "三类调控方案汇总.csv"))
    hits = nearest_search_rows(summary, search)
    search["硬筛选"] = bool_series(search["规划约束_Q50是否低于2800"]) & bool_series(search["规划约束_Q90是否不高于基准"]) & bool_series(search["规划约束_Pr是否达标"])

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8), constrained_layout=True)
    ax = axes[0]
    sc = ax.scatter(
        search["整体工况达标模板距离"],
        search["低排放工况层概率Pr_low"],
        c=search["Q50_CO"],
        s=np.clip(search["总调节量"], 0.2, 30) * 6,
        cmap="viridis_r",
        alpha=np.where(search["硬筛选"], 0.82, 0.20),
        edgecolors="none",
    )
    for _, row in hits.iterrows():
        ax.scatter(row["整体工况达标模板距离"], row["低排放工况层概率Pr_low"], marker="*", s=170, edgecolor="white", linewidth=0.8)
        ax.text(row["整体工况达标模板距离"] + 0.02, row["低排放工况层概率Pr_low"], short_scheme(row["方案"]), fontsize=9)
    ax.set_title("A  达标模板距离-低排放层概率门控相图")
    ax.set_xlabel("整体工况达标模板距离")
    ax.set_ylabel("Pr_low")
    cbar = fig.colorbar(sc, ax=ax, pad=0.012)
    cbar.set_label("Q50_CO (mg/m$^3$)")

    ax = axes[1]
    sc = ax.scatter(
        search["历史达标IQR距离"],
        search["Q50_CO"],
        c=search["Q90_CO"],
        s=np.clip(search["总调节量"], 0.2, 30) * 5,
        cmap="plasma",
        alpha=np.where(search["硬筛选"], 0.82, 0.24),
        edgecolors="none",
    )
    for _, row in hits.iterrows():
        ax.scatter(row["历史达标IQR距离"], row["Q50_CO"], marker="*", s=170, edgecolor="white", linewidth=0.8)
        ax.text(row["历史达标IQR距离"] + 0.0004, row["Q50_CO"] + 6, short_scheme(row["方案"]), fontsize=9)
    ax.axhline(STANDARD_CO, color=COLORS["red"], ls="--", lw=1.1)
    ax.set_title("B  历史达标IQR距离与Q50目标")
    ax.set_xlabel("历史达标IQR距离")
    ax.set_ylabel("Q50_CO (mg/m$^3$)")
    cbar = fig.colorbar(sc, ax=ax, pad=0.012)
    cbar.set_label("Q90_CO (mg/m$^3$)")
    return save_fig(fig, FIGURE_NOTES[5][0])


def plot_segment_coordination_center() -> Path:
    combo = scheme_order(read_csv(DATA_OUT / "三类调控方案风箱负压组合.csv"))
    combo = combo[~combo["方案"].astype(str).str.contains("方案0")].copy()
    sections = {
        "前段1-5": list(range(1, 6)),
        "中段6-15": list(range(6, 16)),
        "尾段16-18": list(range(16, 19)),
    }
    rows = []
    for scheme, part in combo.groupby("方案", sort=False):
        deltas = part.sort_values("风箱编号")["相对控制时刻实际强度变化"].to_numpy(dtype=float)
        abs_sum = np.abs(deltas).sum()
        center = float(np.sum(np.abs(deltas) * np.array(BOX_IDS)) / abs_sum) if abs_sum > 0 else np.nan
        row = {"方案": scheme, "方案短名": short_scheme(scheme), "总调节量": abs_sum, "调节重心风箱": center}
        for name, ids in sections.items():
            vals = part[part["风箱编号"].isin(ids)]["相对控制时刻实际强度变化"].to_numpy(dtype=float)
            row[name] = float(vals.sum())
            row[name + "_绝对量"] = float(np.abs(vals).sum())
        rows.append(row)
    sec = pd.DataFrame(rows)
    cum = combo.pivot(index="方案", columns="风箱编号", values="相对控制时刻实际强度变化").loc[sec["方案"], BOX_IDS].cumsum(axis=1)

    fig = plt.figure(figsize=(14.2, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(sec))
    bottom_pos = np.zeros(len(sec))
    bottom_neg = np.zeros(len(sec))
    palette = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    for color, name in zip(palette, sections):
        vals = sec[name].to_numpy(dtype=float)
        pos = np.where(vals > 0, vals, 0)
        neg = np.where(vals < 0, vals, 0)
        ax.bar(x, pos, bottom=bottom_pos, color=color, alpha=0.82, label=name)
        ax.bar(x, neg, bottom=bottom_neg, color=color, alpha=0.46)
        bottom_pos += pos
        bottom_neg += neg
    ax.axhline(0, color=COLORS["dark"], lw=1)
    ax.set_title("A  前/中/尾段签名调节量")
    ax.set_xticks(x)
    ax.set_xticklabels(sec["方案短名"], rotation=10)
    ax.set_ylabel("Delta P强度合计")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(sec["调节重心风箱"], sec["总调节量"], s=sec["总调节量"] * 120 + 90, c=[COLORS["blue"], COLORS["orange"], COLORS["green"]], alpha=0.82)
    for _, row in sec.iterrows():
        ax.text(row["调节重心风箱"] + 0.08, row["总调节量"], row["方案短名"], fontsize=9)
    ax.axvspan(1, 5.5, color=COLORS["blue"], alpha=0.06)
    ax.axvspan(5.5, 15.5, color=COLORS["orange"], alpha=0.06)
    ax.axvspan(15.5, 18.5, color=COLORS["green"], alpha=0.08)
    ax.set_xlim(1, 18.5)
    ax.set_title("B  调节重心风箱与调节强度")
    ax.set_xlabel("按|Delta P|加权的调节重心")
    ax.set_ylabel("总调节量")

    ax = fig.add_subplot(gs[1, :])
    for scheme, row in cum.iterrows():
        ax.plot(BOX_IDS, row.to_numpy(dtype=float), marker="o", lw=2, label=short_scheme(scheme))
    ax.axhline(0, color=COLORS["dark"], lw=1)
    ax.axvline(5.5, color=COLORS["gray"], ls=":", lw=1)
    ax.axvline(15.5, color=COLORS["gray"], ls=":", lw=1)
    ax.set_title("C  沿烧结方向的累计调节量曲线")
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("累计Delta P")
    ax.set_xticks(BOX_IDS)
    ax.legend(ncol=3, frameon=False)
    return save_fig(fig, FIGURE_NOTES[6][0])


def plot_future_phase_density() -> Path:
    future = read_csv(DATA_MID / "未来300s低排放潜力建模数据.csv")
    summary = scheme_order(read_csv(DATA_OUT / "三类调控方案汇总.csv"))
    base = summary[summary["方案"].astype(str).str.contains("方案0")].iloc[0]
    x = future["CO历史浓度__lag0"].to_numpy(dtype=float)
    y = future[TARGET_COL].to_numpy(dtype=float)
    pass_mask = bool_series(future["single_point_pass"]).to_numpy(dtype=bool)
    current_x = float(base["实际CO(t)_mg_m3"])
    current_y = float(base["实际CO(t+300s)_mg_m3"])

    fig = plt.figure(figsize=(14.2, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    hb = ax.hexbin(x, y, gridsize=42, mincnt=1, cmap="cividis", bins="log")
    ax.scatter(x[pass_mask], y[pass_mask], s=16, color=COLORS["orange"], alpha=0.62, label="300s后达标样本")
    ax.scatter([current_x], [current_y], s=150, marker="*", color=COLORS["red"], edgecolor="white", linewidth=0.8, label="当前控制案例")
    ax.axvline(STANDARD_CO, color=COLORS["red"], ls="--", lw=1.0)
    ax.axhline(STANDARD_CO, color=COLORS["red"], ls="--", lw=1.0)
    ax.set_title("A  当前CO状态 vs 300s后CO的相空间密度")
    ax.set_xlabel("控制时刻CO历史浓度 lag0 (mg/m$^3$)")
    ax.set_ylabel("300s后单点CO (mg/m$^3$)")
    ax.legend(frameon=False, loc="upper left")
    cbar = fig.colorbar(hb, ax=ax, pad=0.012)
    cbar.set_label("样本密度(log)")

    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(np.nanpercentile(x, 2), np.nanpercentile(x, 98), 16)
    labels = []
    rates = []
    med_future = []
    centers = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() < 8:
            continue
        centers.append((lo + hi) / 2)
        labels.append(f"{lo:.0f}-{hi:.0f}")
        rates.append(pass_mask[mask].mean() * 100)
        med_future.append(np.nanmedian(y[mask]))
    centers = np.array(centers)
    bars = ax.bar(np.arange(len(rates)), rates, color=COLORS["green"], alpha=0.78, label="300s后达标率")
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(med_future)), med_future, color=COLORS["blue"], marker="o", lw=1.9, label="300s后CO中位数")
    ax2.axhline(STANDARD_CO, color=COLORS["red"], ls="--", lw=1.0)
    if len(centers):
        nearest = int(np.argmin(np.abs(centers - current_x)))
        ax.axvspan(nearest - 0.5, nearest + 0.5, color=COLORS["red"], alpha=0.10, label="当前CO所在箱")
    ax.set_title("B  按当前CO分箱的未来达标率")
    ax.set_xlabel("当前CO分箱")
    ax.set_ylabel("达标率 (%)")
    ax2.set_ylabel("300s后CO中位数")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, frameon=False, loc="upper left")
    return save_fig(fig, FIGURE_NOTES[7][0])


def plot_pressure_temperature_inconsistency() -> Path:
    decision = read_csv(DATA_OUT / "风箱负压调控决策_最新状态.csv").sort_values("风箱编号")
    boxes = decision["风箱编号"].to_numpy(dtype=int)
    dp = decision["相对控制时刻实际强度变化"].to_numpy(dtype=float)
    dt = decision["第一问温度增量"].to_numpy(dtype=float)
    temp_after = decision["增量叠加后温度"].to_numpy(dtype=float)
    norm = TwoSlopeNorm(vcenter=0, vmin=-max(1.0, np.max(np.abs(dt))), vmax=max(1.0, np.max(np.abs(dt))))

    fig = plt.figure(figsize=(14.2, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[:, 0])
    sc = ax.scatter(dp, dt, c=dt, cmap="PuOr_r", norm=norm, s=np.abs(temp_after - np.nanmedian(temp_after)) * 2.2 + 70, edgecolor="white", linewidth=0.8)
    for b, x0, y0 in zip(boxes, dp, dt):
        ax.text(x0 + 0.006, y0, str(b), fontsize=9)
    ax.axhline(0, color=COLORS["dark"], lw=1)
    ax.axvline(0, color=COLORS["dark"], lw=1)
    ax.set_title("A  方案C局部调压-温度增量响应相图")
    ax.set_xlabel("相对控制时刻Delta P_i |KPa|")
    ax.set_ylabel("第一问温度增量Delta T_i (℃)")
    cbar = fig.colorbar(sc, ax=ax, pad=0.012)
    cbar.set_label("Delta T_i (℃)")

    ax = fig.add_subplot(gs[0, 1])
    width = 0.38
    ax.bar(boxes - width / 2, dp, width=width, color=np.where(dp >= 0, COLORS["green"], COLORS["orange"]), alpha=0.82, label="Delta P")
    ax2 = ax.twinx()
    ax2.plot(boxes + width / 2, dt, color=COLORS["purple"], marker="o", lw=1.8, label="Delta T")
    ax.axhline(0, color=COLORS["dark"], lw=1)
    ax2.axhline(0, color=COLORS["gray"], lw=1)
    ax.set_title("B  逐风箱Delta P与Delta T对照")
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("Delta P |KPa|")
    ax2.set_ylabel("Delta T (℃)")
    ax.set_xticks(BOX_IDS)
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, frameon=False, loc="upper left")

    ax = fig.add_subplot(gs[1, 1])
    quadrant = pd.DataFrame(
        {
            "象限": np.select(
                [
                    (dp >= 0) & (dt >= 0),
                    (dp >= 0) & (dt < 0),
                    (dp < 0) & (dt >= 0),
                    (dp < 0) & (dt < 0),
                ],
                ["增强负压+升温", "增强负压+降温", "减弱负压+升温", "减弱负压+降温"],
                default="近零",
            )
        }
    )
    counts = quadrant["象限"].value_counts().reindex(["增强负压+升温", "增强负压+降温", "减弱负压+升温", "减弱负压+降温", "近零"]).fillna(0)
    bars = ax.bar(counts.index, counts.values, color=[COLORS["red"], COLORS["green"], COLORS["orange"], COLORS["blue"], COLORS["gray"]], alpha=0.82)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("C  调压-温度响应象限统计")
    ax.set_ylabel("风箱数")
    ax.tick_params(axis="x", rotation=18)
    return save_fig(fig, FIGURE_NOTES[8][0])


def make_contact_sheet(paths: list[Path]) -> Path:
    ncols = 3
    nrows = math.ceil(len(paths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.9 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, path in zip(axes, paths):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(path.stem, fontsize=10)
        ax.axis("off")
    for ax in axes[len(paths) :]:
        ax.axis("off")
    return save_fig(fig, "contact_sheet.png")


def write_index(paths: list[Path]) -> Path:
    note_map = dict(FIGURE_NOTES)
    lines = [
        "# 第三问局部新颖PNG图表索引",
        "",
        "本组图不重复上一组整体图，重点挖局部病例、滚动事件、低排放片段、候选解结构和调压-温度响应细节，可作为论文亮点图或补充图。",
        "",
        "| 文件 | 用途 |",
        "|---|---|",
    ]
    for path in paths:
        lines.append(f"| `{path.name}` | {note_map.get(path.name, '')} |")
    lines.append("| `contact_sheet.png` | 本组局部新颖图缩略总览。 |")
    out = OUT_DIR / "图表索引.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    setup_style()
    plotters = [
        plot_lag_casebook,
        plot_rolling_event_fingerprints,
        plot_low_emission_alignment,
        plot_longest_compliant_lens,
        plot_pareto_candidates,
        plot_gate_phase_map,
        plot_segment_coordination_center,
        plot_future_phase_density,
        plot_pressure_temperature_inconsistency,
    ]
    paths: list[Path] = []
    for plotter in plotters:
        path = plotter()
        paths.append(path)
        print(f"saved: {path}")
    contact = make_contact_sheet(paths)
    index = write_index(paths)
    print(f"saved: {contact}")
    print(f"saved: {index}")


if __name__ == "__main__":
    main()
