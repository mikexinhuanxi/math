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
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


ROOT = Path(__file__).resolve().parents[2]
DATA_MID = ROOT / "data" / "第3问" / "中间数据"
DATA_OUT = ROOT / "data" / "第3问" / "结果数据"
OUT_DIR = ROOT / "figure" / "第3问" / "可视化PNG"

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
    ("必备_01_负压调节范围与达标工况对照.png", "回答题目要求的各风箱负压调节范围，并把当前、历史达标和低排放片段放在同一坐标中。"),
    ("必备_02_三类调控方案CO分位数与约束.png", "展示方案0/A/B/C在Q10、Q50、Q90、2800标准线和三层规划约束下的差异。"),
    ("必备_03_三类调控方案风箱负压调整.png", "展示各方案目标负压强度和相对控制时刻的逐风箱调整方向。"),
    ("必备_04_工程可实施性与温度风险.png", "说明总调节量、单箱调节、温度增量和工程可实施性评分。"),
    ("必备_05_最新窗口滚动调控轨迹.png", "展示300s滚动窗口内基准预测、优化后预测、实际CO和调节强度。"),
    ("必备_06_模型性能与变量组重要性.png", "说明未来300s调控预测器的误差、达标识别和变量组贡献。"),
    ("必备_07_低排放与历史达标工况证据.png", "展示低排放片段、历史达标样本时间集中性和低排放负压模板。"),
    ("可选_08_3D滚动调控负压平滑曲面.png", "平滑展示最新窗口内滚动目标负压随时间和风箱位置变化。"),
    ("可选_09_3D三类方案负压平滑曲面.png", "平滑展示方案0/A/B/C的空间负压组合差异。"),
    ("可选_10_3D规划搜索响应平滑曲面.png", "展示规划搜索中总调节量、Q90风险和Q50目标之间的响应曲面。"),
    ("可选_11_3D达标模板与推荐方案平滑曲面.png", "把当前工况、历史达标模板、低排放模板和推荐方案放到同一3D曲面比较。"),
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


def finite_series(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def parse_box_id(value: object) -> int | None:
    match = re.search(r"(\d+)#风箱", str(value))
    return int(match.group(1)) if match else None


def short_scheme(value: object) -> str:
    text = str(value)
    if "方案0" in text:
        return "方案0\n基准"
    if "方案A" in text:
        return "方案A\nQ50最优"
    if "方案B" in text:
        return "方案B\nQ90主推"
    if "方案C" in text:
        return "方案C\n工程平稳"
    return text.replace("_", "\n")


def scheme_order(df: pd.DataFrame, col: str = "方案") -> pd.DataFrame:
    order = {"方案0": 0, "方案A": 1, "方案B": 2, "方案C": 3}
    out = df.copy()
    out["_order"] = out[col].map(lambda x: next((v for k, v in order.items() if k in str(x)), 99))
    return out.sort_values("_order").drop(columns="_order")


def bool_value(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "是"}:
        return 1.0
    if text in {"false", "0", "no", "n", "否", "", "nan"}:
        return 0.0
    return np.nan


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.1f}", pad: float = 2.0) -> None:
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        va = "bottom" if height >= 0 else "top"
        y = height + pad if height >= 0 else height - pad
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=8,
            color=COLORS["dark"],
        )


def low_pressure_profile() -> pd.DataFrame:
    path = DATA_OUT / "低排放片段风箱负压典型分布.csv"
    if not path.exists():
        return pd.DataFrame(columns=["风箱编号", "低排放中位数", "低排放P25", "低排放P75"])
    low = read_csv(path)
    if "片段编号" in low.columns:
        low = low[low["片段编号"].astype(str).eq("全部低排放片段")].copy()
    low["风箱编号"] = low["变量"].map(parse_box_id)
    low = low.dropna(subset=["风箱编号"]).copy()
    low["风箱编号"] = low["风箱编号"].astype(int)
    return low.rename(columns={"中位数": "低排放中位数", "P25": "低排放P25", "P75": "低排放P75"})


def smooth_rect_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nx: int = 180,
    ny: int = 140,
    sigma: float = 0.65,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if z.shape != (len(y), len(x)):
        raise ValueError(f"曲面矩阵形状应为({len(y)}, {len(x)}), 实际为{z.shape}")

    if np.isnan(z).any():
        row_means = np.nanmean(z, axis=1)
        for i in range(z.shape[0]):
            mask = np.isnan(z[i])
            z[i, mask] = row_means[i] if np.isfinite(row_means[i]) else np.nanmean(z)

    kx = min(3, max(1, len(y) - 1))
    ky = min(3, max(1, len(x) - 1))
    spline = RectBivariateSpline(y, x, z, kx=kx, ky=ky, s=0)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), nx)
    ys = np.linspace(float(np.min(y)), float(np.max(y)), ny)
    zs = spline(ys, xs)
    if sigma > 0:
        zs = gaussian_filter(zs, sigma=sigma, mode="nearest")
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy, zs


def set_3d_style(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.xaxis.pane.set_facecolor((0.96, 0.98, 1.0, 0.55))
    ax.yaxis.pane.set_facecolor((0.98, 0.98, 0.95, 0.55))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.pane.set_edgecolor("#D7DEE8")
    ax.yaxis.pane.set_edgecolor("#D7DEE8")
    ax.zaxis.pane.set_edgecolor("#D7DEE8")
    ax.tick_params(pad=2)


def plot_pressure_ranges() -> Path:
    ranges = read_csv(DATA_OUT / "风箱负压调节范围.csv")
    compliant = read_csv(DATA_OUT / "历史达标工况风箱负压统计.csv")
    low = low_pressure_profile()

    df = ranges.merge(
        compliant[
            [
                "风箱编号",
                "达标强度P25",
                "达标强度中位数",
                "达标强度P75",
                "控制时刻实际强度",
                "达标中位数较控制时刻变化",
            ]
        ],
        on="风箱编号",
        how="left",
    )
    if not low.empty:
        df = df.merge(low[["风箱编号", "低排放中位数", "低排放P25", "低排放P75"]], on="风箱编号", how="left")
    else:
        df["低排放中位数"] = np.nan

    x = df["风箱编号"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.2), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.vlines(x, df["扩展强度下限_P05"], df["扩展强度上限_P95"], color="#A9B5C3", linewidth=5.8, alpha=0.72, label="主模型范围 P05-P95")
    ax.vlines(x, df["优化强度下限_P10"], df["优化强度上限_P90"], color=COLORS["blue"], linewidth=3.8, alpha=0.86, label="优化范围 P10-P90")
    ax.scatter(x - 0.18, df["控制时刻实际强度"], s=34, color=COLORS["dark"], zorder=4, label="控制时刻实际")
    ax.scatter(x, df["历史强度中位数"], s=30, color=COLORS["gray"], zorder=4, label="历史中位数")
    ax.scatter(x + 0.18, df["达标强度中位数"], s=42, color=COLORS["orange"], zorder=5, label="历史达标中位数")
    if df["低排放中位数"].notna().any():
        ax.scatter(x + 0.32, df["低排放中位数"], s=38, marker="D", color=COLORS["green"], zorder=5, label="稳定低排放中位数")
    ax.set_title("A  各风箱负压强度可调范围与达标工况对照")
    ax.set_ylabel("负压强度 |KPa|")
    ax.set_xlim(0.4, 18.8)
    ax.legend(ncol=5, frameon=False, loc="upper center")

    ax = axes[1]
    width = 0.38
    bars = ax.bar(x - width / 2, df["优化强度范围宽度"], width=width, color=COLORS["sky"], alpha=0.84, label="P10-P90范围宽度")
    annotate_bars(ax, bars, fmt="{:.2f}", pad=0.03)
    ax2 = ax.twinx()
    ax2.plot(x, df["达标中位数较控制时刻变化"], color=COLORS["orange"], marker="o", lw=2, label="达标中位数-当前")
    if df["低排放中位数"].notna().any():
        ax2.plot(x, df["低排放中位数"] - df["控制时刻实际强度"], color=COLORS["green"], marker="s", lw=2, label="低排放中位数-当前")
    ax.axhline(0, color="#A9B5C3", lw=1)
    ax2.axhline(0, color=COLORS["dark"], lw=1, alpha=0.5)
    ax.set_title("B  可调宽度与达标/低排放模板相对当前的位移")
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("P10-P90范围宽度")
    ax2.set_ylabel("相对控制时刻变化 |KPa|")
    ax.set_xticks(x)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, ncol=3, frameon=False, loc="upper left")
    return save_fig(fig, FIGURE_NOTES[0][0])


def plot_scheme_quantiles() -> Path:
    summary = scheme_order(read_csv(DATA_OUT / "三类调控方案汇总.csv"))
    summary["方案短名"] = summary["方案"].map(short_scheme)

    quantile_cols = ["Q10_CO_low_final", "Q50_CO", "Q90_CO"]
    labels = ["Q10低排放潜力", "Q50主目标", "Q90风险上界"]
    colors = [COLORS["green"], COLORS["blue"], COLORS["purple"]]
    x = np.arange(len(summary))
    width = 0.22

    fig = plt.figure(figsize=(16, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 1.05])
    ax = fig.add_subplot(gs[0, 0])
    for idx, (col, label, color) in enumerate(zip(quantile_cols, labels, colors)):
        vals = pd.to_numeric(summary[col], errors="coerce").to_numpy(dtype=float)
        bars = ax.bar(x + (idx - 1) * width, vals, width=width, color=color, alpha=0.88, label=label)
        annotate_bars(ax, bars, fmt="{:.0f}", pad=14)
    baseline_q90 = float(summary.loc[summary["方案"].astype(str).str.contains("方案0"), "Q90_CO"].iloc[0])
    ax.axhline(STANDARD_CO, color=COLORS["red"], ls="--", lw=1.2, label="2800标准")
    ax.axhline(baseline_q90, color=COLORS["gray"], ls=":", lw=1.4, label="基准Q90")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["方案短名"])
    ax.set_ylabel("预测 CO (mg/m$^3$)")
    ax.set_title("A  300s后单点CO分位数规划结果")
    ax.legend(ncol=3, frameon=False, loc="upper center")

    ax = fig.add_subplot(gs[0, 1])
    flag_cols = [
        "规划约束_Q50是否低于2800",
        "规划约束_Q90是否不高于基准",
        "规划约束_Pr是否达标",
        "是否满足严格工程约束",
    ]
    flags = np.array([[bool_value(v) for v in summary[col]] for col in flag_cols], dtype=float)
    ax.imshow(flags, aspect="auto", cmap=mpl.colors.ListedColormap(["#F4B6B6", "#A8DAB5"]), vmin=0, vmax=1)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["方案短名"])
    ax.set_yticks(np.arange(len(flag_cols)))
    ax.set_yticklabels(["Q50<=2800", "Q90<=基准", "Pr>=阈值", "工程可行"])
    for i in range(flags.shape[0]):
        for j in range(flags.shape[1]):
            ax.text(j, i, "是" if flags[i, j] == 1 else "否", ha="center", va="center", fontsize=9, color=COLORS["dark"])
    ax.set_title("B  三层规划筛选与工程约束")
    ax.grid(False)

    ax = fig.add_subplot(gs[0, 2])
    prob_cols = [
        ("近达标概率Pr_CO_le_3200", "Pr near", COLORS["sky"], "o"),
        ("严格达标概率Pr_CO_le_2800", "Pr pass", COLORS["orange"], "s"),
        ("低排放工况层概率Pr_low", "Pr low-layer", COLORS["green"], "^"),
        ("工程可实施性评分", "工程评分", COLORS["purple"], "D"),
    ]
    for col, label, color, marker in prob_cols:
        vals = pd.to_numeric(summary[col], errors="coerce").to_numpy(dtype=float)
        ax.plot(x, vals, marker=marker, lw=2, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["方案短名"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("0-1评分/概率")
    ax.set_title("C  达标倾向与工程评分")
    ax.legend(frameon=False, loc="lower left")
    return save_fig(fig, FIGURE_NOTES[1][0])


def plot_scheme_pressure_adjustments() -> Path:
    combo = scheme_order(read_csv(DATA_OUT / "三类调控方案风箱负压组合.csv"))
    combo["方案短名"] = combo["方案"].map(short_scheme)
    target = combo.pivot(index="方案短名", columns="风箱编号", values="目标强度").reindex(combo["方案短名"].drop_duplicates())
    delta = combo.pivot(index="方案短名", columns="风箱编号", values="相对控制时刻实际强度变化").reindex(target.index)

    boxes = np.array(BOX_IDS)
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 8.4), constrained_layout=True)

    ax = axes[0]
    for idx, scheme in enumerate(target.index):
        color = [COLORS["gray"], COLORS["blue"], COLORS["orange"], COLORS["green"]][idx % 4]
        lw = 1.8 if idx == 0 else 2.2
        marker = "o" if idx != 0 else "."
        ax.plot(boxes, target.loc[scheme, BOX_IDS], marker=marker, lw=lw, color=color, label=scheme.replace("\n", " "))
    ax.set_title("A  方案0/A/B/C的目标负压强度空间组合")
    ax.set_ylabel("目标负压强度 |KPa|")
    ax.set_xticks(boxes)
    ax.legend(ncol=4, frameon=False, loc="upper center")

    ax = axes[1]
    values = delta.loc[:, BOX_IDS].to_numpy(dtype=float)
    vmax = max(0.25, float(np.nanmax(np.abs(values))))
    im = ax.imshow(values, aspect="auto", cmap="PuOr_r", norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    ax.set_title("B  相对控制时刻实际强度的调整量")
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("方案")
    ax.set_xticks(np.arange(len(BOX_IDS)))
    ax.set_xticklabels(BOX_IDS)
    ax.set_yticks(np.arange(len(delta.index)))
    ax.set_yticklabels([s.replace("\n", " ") for s in delta.index])
    cbar = fig.colorbar(im, ax=ax, pad=0.012)
    cbar.set_label("调整量 |KPa|, 正值为增强负压")
    ax.grid(False)
    return save_fig(fig, FIGURE_NOTES[2][0])


def plot_engineering_feasibility() -> Path:
    summary = scheme_order(read_csv(DATA_OUT / "三类调控方案汇总.csv"))
    constraints = read_csv(DATA_OUT / "调控约束参数.csv")
    labels = summary["方案"].map(short_scheme).to_list()
    x = np.arange(len(summary))
    total_budget = float(pd.to_numeric(constraints["总调节预算B"], errors="coerce").dropna().iloc[0])
    max_single_limit = float(pd.to_numeric(constraints["单箱调节上限_Delta_i"], errors="coerce").max())

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.2), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    bars = ax.bar(x, summary["总调节量"], color=COLORS["blue"], alpha=0.86)
    ax.axhline(total_budget, color=COLORS["red"], ls="--", lw=1.2, label=f"总预算 {total_budget:.2f}")
    annotate_bars(ax, bars, fmt="{:.2f}", pad=0.04)
    ax.set_title("A  总调节量")
    ax.set_ylabel("sum |Delta P|")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)

    ax = axes[1]
    bars = ax.bar(x, summary["最大单箱调节量"], color=COLORS["orange"], alpha=0.86)
    ax.axhline(max_single_limit, color=COLORS["red"], ls="--", lw=1.2, label=f"最大单箱上限 {max_single_limit:.2f}")
    annotate_bars(ax, bars, fmt="{:.2f}", pad=0.03)
    ax.set_title("B  最大单箱调节量")
    ax.set_ylabel("max |Delta P_i|")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)

    ax = axes[2]
    width = 0.34
    bars1 = ax.bar(x - width / 2, summary["最大温度增量"], width=width, color=COLORS["purple"], alpha=0.82, label="最大温度增量")
    ax.axhline(60, color=COLORS["green"], ls=":", lw=1.2, label="严格60℃")
    ax.axhline(90, color=COLORS["red"], ls="--", lw=1.2, label="理论90℃")
    annotate_bars(ax, bars1, fmt="{:.1f}", pad=1.1)
    ax2 = ax.twinx()
    ax2.plot(x + width / 2, summary["温度越界数量"], marker="o", color=COLORS["red"], lw=2, label="温度越界数量")
    ax.set_title("C  温度联动风险")
    ax.set_ylabel("最大温度增量 (℃)")
    ax2.set_ylabel("越界风箱数")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, frameon=False, loc="upper left")

    ax = axes[3]
    width = 0.34
    bars = ax.bar(x - width / 2, summary["工程可实施性评分"], width=width, color=COLORS["green"], alpha=0.85, label="工程可实施性评分")
    ax.bar(x + width / 2, summary["负压可控贡献率"], width=width, color=COLORS["sky"], alpha=0.85, label="负压可控贡献率")
    annotate_bars(ax, bars, fmt="{:.2f}", pad=0.02)
    ax.set_ylim(0, 1.05)
    ax.set_title("D  执行友好度与可控贡献")
    ax.set_ylabel("0-1指标")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, loc="upper right")
    return save_fig(fig, FIGURE_NOTES[3][0])


def plot_rolling_control() -> Path:
    traj = read_csv(DATA_OUT / "最新窗口滚动调控轨迹.csv")
    t = pd.to_numeric(traj["时间_s"], errors="coerce").to_numpy(dtype=float) / 60.0
    t = t - np.nanmin(t)

    fig, axes = plt.subplots(3, 1, figsize=(13.8, 8.8), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(t, traj["实际CO"], color=COLORS["dark"], lw=1.4, label="实际CO")
    ax.plot(t, traj["基准模型预测CO"], color=COLORS["gray"], lw=1.7, label="基准预测")
    ax.plot(t, traj["滚动优化后预测CO"], color=COLORS["green"], lw=1.9, label="滚动优化后")
    ax.axhline(STANDARD_CO, color=COLORS["red"], ls="--", lw=1.1, label="2800标准")
    ax.set_title("A  最新300s窗口：基准预测与滚动优化后预测")
    ax.set_ylabel("CO (mg/m$^3$)")
    ax.legend(ncol=4, frameon=False, loc="upper center")

    ax = axes[1]
    decrease = pd.to_numeric(traj["模型预测降低量"], errors="coerce").to_numpy(dtype=float)
    adjust = pd.to_numeric(traj["相对控制时刻实际总调节量"], errors="coerce").to_numpy(dtype=float)
    ax.fill_between(t, decrease, 0, where=decrease >= 0, color=COLORS["green"], alpha=0.24, label="预测降低量")
    ax.plot(t, decrease, color=COLORS["green"], lw=1.6)
    ax.axhline(0, color=COLORS["dark"], lw=1)
    ax.set_ylabel("预测降低量 (mg/m$^3$)")
    ax.set_title("B  降CO收益与总调节量")
    ax2 = ax.twinx()
    ax2.plot(t, adjust, color=COLORS["orange"], lw=1.7, marker="o", ms=3, label="总调节量")
    ax2.set_ylabel("sum |Delta P|")
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, ncol=2, frameon=False, loc="upper center")

    ax = axes[2]
    ax.plot(t, traj["基准达标倾向概率"], color=COLORS["gray"], lw=1.5, label="基准Pr")
    ax.plot(t, traj["滚动目标达标倾向概率"], color=COLORS["purple"], lw=1.8, label="滚动目标Pr")
    ax.plot(t, traj["相对上一目标最大单箱单步调节量"], color=COLORS["blue"], lw=1.45, label="单步最大单箱调节")
    ax.set_title("C  达标倾向与单步平滑性")
    ax.set_xlabel("窗口内时间 (min)")
    ax.set_ylabel("概率 / 调节量")
    ax.set_ylim(bottom=0)
    ax.legend(ncol=3, frameon=False, loc="lower center")
    return save_fig(fig, FIGURE_NOTES[4][0])


def plot_model_evidence() -> Path:
    metrics = read_csv(DATA_OUT / "未来300s低排放潜力模型评价指标.csv")
    importance = read_csv(DATA_OUT / "非线性低排放模型变量组重要性.csv")
    leave_one_path = DATA_OUT / "达标片段留一验证.csv"
    leave_one = read_csv(leave_one_path) if leave_one_path.exists() else pd.DataFrame()

    metrics = metrics.drop_duplicates(subset=["数据集"]).copy()
    dataset_order = ["训练集", "验证集", "测试集"]
    metrics["_order"] = metrics["数据集"].map({v: i for i, v in enumerate(dataset_order)})
    metrics = metrics.sort_values("_order")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.34
    bars1 = ax.bar(x - width / 2, metrics["MAE"], width=width, color=COLORS["sky"], alpha=0.86, label="MAE")
    bars2 = ax.bar(x + width / 2, metrics["RMSE"], width=width, color=COLORS["blue"], alpha=0.86, label="RMSE")
    annotate_bars(ax, bars2, fmt="{:.0f}", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["数据集"])
    ax.set_ylabel("误差 (mg/m$^3$)")
    ax.set_title("A  未来300s调控预测器误差")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(x, metrics["Q50预测CO低于2800比例_%"], marker="o", color=COLORS["green"], lw=2, label="Q50<2800比例")
    ax.plot(x, metrics["Q90预测CO高于3200比例_%"], marker="s", color=COLORS["red"], lw=2, label="Q90>3200比例")
    ax.plot(x, metrics["Pr_low均值"] * 100, marker="^", color=COLORS["purple"], lw=2, label="Pr_low均值x100")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["数据集"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("比例 (%)")
    ax.set_title("B  分位数达标倾向与高风险比例")
    ax.legend(frameon=False)

    ax = axes[2]
    imp = importance.sort_values("正向重要性占比_%", ascending=True)
    bars = ax.barh(imp["变量组"], imp["正向重要性占比_%"], color=COLORS["orange"], alpha=0.86)
    for bar in bars:
        width_val = bar.get_width()
        ax.text(width_val + 0.6, bar.get_y() + bar.get_height() / 2, f"{width_val:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("正向重要性占比 (%)")
    ax.set_title("C  变量组置换重要性")

    ax = axes[3]
    if not leave_one.empty:
        show = leave_one.head(10).copy()
        idx = np.arange(len(show))
        pr_col = "Pr_pass_like均值" if "Pr_pass_like均值" in show.columns else "Pr_strict均值"
        ax.plot(idx, show["Q10低于2800比例_%"], marker="o", color=COLORS["green"], lw=1.8, label="Q10<2800比例")
        ax.plot(idx, show[pr_col] * 100, marker="s", color=COLORS["purple"], lw=1.8, label="Pr_pass均值x100")
        ax.axhline(50, color=COLORS["gray"], ls=":", lw=1.1)
        ax.set_xticks(idx)
        ax.set_xticklabels([f"{int(v)}" for v in show["留一折次"]])
        ax.set_xlabel("达标片段留一折次")
        ax.set_ylabel("比例 / 概率x100")
        ax.legend(frameon=False)
    ax.set_title("D  达标片段留一识别能力")
    return save_fig(fig, FIGURE_NOTES[5][0])


def plot_low_emission_evidence() -> Path:
    cleaned = read_csv(DATA_MID / "清洗后数据_含低排放片段.csv")
    intervals = read_csv(DATA_OUT / "低排放工况区间识别.csv")
    time_dist = read_csv(DATA_OUT / "历史达标工况时间分布.csv")
    low_profile = low_pressure_profile()

    co_col = "第三问原始CO" if "第三问原始CO" in cleaned.columns else TARGET_COL
    cleaned["时间_min"] = pd.to_numeric(cleaned["时间_s"], errors="coerce") / 60.0
    co = pd.to_numeric(cleaned[co_col], errors="coerce")

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(cleaned["时间_min"], co, color=COLORS["dark"], lw=1.15, label="原始CO")
    ax.axhline(STANDARD_CO, color=COLORS["red"], ls="--", lw=1.1, label="2800标准")
    if "低排放CO阈值" in intervals.columns and not intervals.empty:
        ax.axhline(float(intervals["低排放CO阈值"].iloc[0]), color=COLORS["green"], ls=":", lw=1.2, label="低排放阈值")
    for _, row in intervals.iterrows():
        start = float(row["起始时间_s"]) / 60.0
        end = float(row["结束时间_s"]) / 60.0
        ax.axvspan(start, end, color=COLORS["green"], alpha=0.22)
        ax.text((start + end) / 2, np.nanmax(co) * 0.985, f"片段{int(row['片段编号'])}", ha="center", va="top", fontsize=9, color=COLORS["green"])
    pass_mask = co <= STANDARD_CO
    ax.scatter(cleaned.loc[pass_mask, "时间_min"], co[pass_mask], s=16, color=COLORS["orange"], alpha=0.8, label="历史达标点")
    ax.set_title("A  CO时序中的稳定低排放片段与历史达标点")
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("CO (mg/m$^3$)")
    ax.legend(ncol=4, frameon=False, loc="upper center")

    ax = fig.add_subplot(gs[1, 0])
    idx = np.arange(len(time_dist))
    bars = ax.bar(idx, time_dist["达标样本数"], color=COLORS["orange"], alpha=0.86, label="达标样本数")
    annotate_bars(ax, bars, fmt="{:.0f}", pad=0.2)
    ax2 = ax.twinx()
    ax2.plot(idx, time_dist["CO中位数"], color=COLORS["blue"], marker="o", lw=1.8, label="CO中位数")
    ax2.axhline(STANDARD_CO, color=COLORS["red"], ls="--", lw=1.0)
    ax.set_xticks(idx)
    ax.set_xticklabels(time_dist["时间段编号"])
    ax.set_xlabel("等分时间段编号")
    ax.set_ylabel("达标样本数")
    ax2.set_ylabel("CO中位数")
    ax.set_title("B  历史达标样本的时间集中性")
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, frameon=False, loc="upper left")

    ax = fig.add_subplot(gs[1, 1])
    if not low_profile.empty:
        low_profile = low_profile.sort_values("风箱编号")
        x = low_profile["风箱编号"].to_numpy(dtype=float)
        ax.fill_between(x, low_profile["低排放P25"], low_profile["低排放P75"], color=COLORS["green"], alpha=0.22, label="低排放P25-P75")
        ax.plot(x, low_profile["低排放中位数"], marker="o", color=COLORS["green"], lw=2, label="低排放中位数")
        ax.set_xticks(BOX_IDS)
    ax.set_title("C  稳定低排放片段的风箱负压模板")
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("负压强度 |KPa|")
    ax.legend(frameon=False)
    return save_fig(fig, FIGURE_NOTES[6][0])


def plot_3d_rolling_surface() -> Path:
    traj = read_csv(DATA_OUT / "最新窗口滚动调控轨迹.csv")
    target_cols = [f"{i}#滚动目标强度" for i in BOX_IDS]
    actual_cols = [f"{i}#控制时刻实际强度" for i in BOX_IDS]
    target = gaussian_filter(traj[target_cols].to_numpy(dtype=float), sigma=(1.35, 0.72), mode="nearest")
    actual = gaussian_filter(traj[actual_cols].to_numpy(dtype=float), sigma=(1.35, 0.72), mode="nearest")
    y = pd.to_numeric(traj["时间_s"], errors="coerce").to_numpy(dtype=float) / 60.0
    y = y - np.nanmin(y)
    x = np.array(BOX_IDS, dtype=float)
    xx, yy, zz = smooth_rect_surface(x, y, target, nx=180, ny=170, sigma=1.15)

    fig = plt.figure(figsize=(12.5, 8.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=True, alpha=0.94)
    for row_idx in [0, len(y) // 2, len(y) - 1]:
        ax.plot(x, np.full_like(x, y[row_idx]), actual[row_idx], color="#263238", lw=1.05, alpha=0.42)
    ax.set_title("最新窗口滚动目标负压平滑曲面")
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("窗口内时间 (min)")
    ax.set_zlabel("负压强度 |KPa|")
    ax.view_init(elev=27, azim=-132)
    set_3d_style(ax)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.63, pad=0.02)
    cbar.set_label("滚动目标负压强度 |KPa|")
    return save_fig(fig, FIGURE_NOTES[7][0])


def plot_3d_scheme_surface() -> Path:
    combo = scheme_order(read_csv(DATA_OUT / "三类调控方案风箱负压组合.csv"))
    combo["方案短名"] = combo["方案"].map(short_scheme)
    target = combo.pivot(index="方案短名", columns="风箱编号", values="目标强度").reindex(combo["方案短名"].drop_duplicates())
    x = np.array(BOX_IDS, dtype=float)
    y = np.arange(len(target), dtype=float)
    z = target.loc[:, BOX_IDS].to_numpy(dtype=float)
    xx, yy, zz = smooth_rect_surface(x, y, z, nx=180, ny=120, sigma=0.55)

    fig = plt.figure(figsize=(12.5, 8.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, zz, cmap="cividis", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title("方案0/A/B/C负压组合平滑曲面")
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("调控方案")
    ax.set_zlabel("目标负压强度 |KPa|")
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("\n", " ") for s in target.index])
    ax.view_init(elev=28, azim=-124)
    set_3d_style(ax)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.63, pad=0.02)
    cbar.set_label("目标负压强度 |KPa|")
    return save_fig(fig, FIGURE_NOTES[8][0])


def smooth_grid_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    nx: int = 95,
    ny: int = 95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]
    if len(x) < 8:
        raise ValueError("规划搜索记录点数不足，无法绘制3D响应曲面")
    x_lo, x_hi = np.quantile(x, [0.02, 0.98])
    y_lo, y_hi = np.quantile(y, [0.02, 0.98])
    keep = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
    x, y, z = x[keep], y[keep], z[keep]

    x_edges = np.linspace(float(x.min()), float(x.max()), nx + 1)
    y_edges = np.linspace(float(y.min()), float(y.max()), ny + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    zz = np.full((ny, nx), np.nan, dtype=float)
    xi = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    yi = np.clip(np.digitize(y, y_edges) - 1, 0, ny - 1)
    bins: dict[tuple[int, int], list[float]] = {}
    for ix, iy, value in zip(xi, yi, z):
        bins.setdefault((iy, ix), []).append(float(value))
    for (iy, ix), values in bins.items():
        zz[iy, ix] = float(np.median(values))

    xx, yy = np.meshgrid(x_centers, y_centers)
    if np.isnan(zz).any():
        known = np.isfinite(zz)
        nearest = griddata((xx[known], yy[known]), zz[known], (xx, yy), method="nearest")
        zz = np.where(np.isnan(zz), nearest, zz)
    zz = gaussian_filter(zz, sigma=2.8, mode="nearest")
    return xx, yy, zz


def plot_3d_search_surface() -> Path:
    search = read_csv(DATA_OUT / "主模型_Q50_Q90_概率三层规划搜索记录.csv")
    x = pd.to_numeric(search["总调节量"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(search["Q90_CO"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(search["Q50_CO"], errors="coerce").to_numpy(dtype=float)
    xx, yy, zz = smooth_grid_surface(x, y, z)

    fig = plt.figure(figsize=(12.5, 8.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, zz, cmap="plasma_r", linewidth=0, antialiased=True, alpha=0.92)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    sample = np.linspace(0, np.where(finite)[0].size - 1, min(260, np.where(finite)[0].size)).astype(int)
    idx = np.where(finite)[0][sample]
    ax.scatter(x[idx], y[idx], z[idx], s=7, color="#263238", alpha=0.22, depthshade=False)
    ax.set_title("规划搜索响应曲面：总调节量-Q90风险-Q50目标")
    ax.set_xlabel("总调节量 sum |Delta P|")
    ax.set_ylabel("Q90风险上界 (mg/m$^3$)")
    ax.set_zlabel("Q50主目标 (mg/m$^3$)")
    ax.view_init(elev=25, azim=-138)
    set_3d_style(ax)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.63, pad=0.02)
    cbar.set_label("Q50主目标 (mg/m$^3$)")
    return save_fig(fig, FIGURE_NOTES[9][0])


def plot_3d_template_surface() -> Path:
    compliant = read_csv(DATA_OUT / "历史达标工况风箱负压统计.csv").sort_values("风箱编号")
    low = low_pressure_profile().sort_values("风箱编号")
    combo = scheme_order(read_csv(DATA_OUT / "三类调控方案风箱负压组合.csv"))
    combo_b = combo[combo["方案"].astype(str).str.contains("方案B")].sort_values("风箱编号")
    combo_c = combo[combo["方案"].astype(str).str.contains("方案C")].sort_values("风箱编号")

    profiles = [
        ("当前实际", compliant["控制时刻实际强度"].to_numpy(dtype=float)),
        ("历史达标中位数", compliant["达标强度中位数"].to_numpy(dtype=float)),
        ("稳定低排放中位数", low["低排放中位数"].to_numpy(dtype=float) if not low.empty else compliant["达标强度中位数"].to_numpy(dtype=float)),
        ("方案B主推荐", combo_b["目标强度"].to_numpy(dtype=float)),
        ("方案C工程平稳", combo_c["目标强度"].to_numpy(dtype=float)),
    ]
    x = np.array(BOX_IDS, dtype=float)
    y = np.arange(len(profiles), dtype=float)
    z = np.vstack([p[1] for p in profiles])
    xx, yy, zz = smooth_rect_surface(x, y, z, nx=180, ny=130, sigma=0.55)

    fig = plt.figure(figsize=(12.8, 8.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, zz, cmap="YlGnBu", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title("当前-达标模板-低排放模板-推荐方案的平滑曲面对照")
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("工况/方案")
    ax.set_zlabel("负压强度 |KPa|")
    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _ in profiles])
    ax.view_init(elev=27, azim=-126)
    set_3d_style(ax)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.63, pad=0.02)
    cbar.set_label("负压强度 |KPa|")
    return save_fig(fig, FIGURE_NOTES[10][0])


def make_contact_sheet(paths: list[Path]) -> Path:
    images = [p for p in paths if p.exists()]
    if not images:
        raise FileNotFoundError("没有可用于生成总览图的PNG")
    ncols = 3
    nrows = math.ceil(len(images) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.8 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, path in zip(axes, images):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(path.stem, fontsize=10)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")
    return save_fig(fig, "contact_sheet.png")


def write_index(paths: list[Path]) -> Path:
    note_map = dict(FIGURE_NOTES)
    lines = [
        "# 第三问可视化PNG图表索引",
        "",
        "本组图按题目第三问的逻辑组织：先交代负压调节范围，再展示规划方案、工程约束、滚动调控和模型证据；可选图均为平滑3D曲面。",
        "",
        "| 类型 | 文件 | 用途 |",
        "|---|---|---|",
    ]
    for path in paths:
        kind = "必备" if path.name.startswith("必备") else "可选"
        lines.append(f"| {kind} | `{path.name}` | {note_map.get(path.name, '')} |")
    lines.append(f"| 总览 | `contact_sheet.png` | 所有PNG缩略预览。 |")
    out = OUT_DIR / "图表索引.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    setup_style()
    plotters = [
        plot_pressure_ranges,
        plot_scheme_quantiles,
        plot_scheme_pressure_adjustments,
        plot_engineering_feasibility,
        plot_rolling_control,
        plot_model_evidence,
        plot_low_emission_evidence,
        plot_3d_rolling_surface,
        plot_3d_scheme_surface,
        plot_3d_search_surface,
        plot_3d_template_surface,
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
