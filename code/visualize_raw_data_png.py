# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "附件1.原始数据.xlsx"
OUT_DIR = ROOT / "figure" / "原始数据可视化PNG"
TARGET = "烧结大烟道外排CO浓度"
CO_STANDARD = 2800
SAMPLE_SECONDS = 2

COLORS = {
    "blue": "#2F6B9A",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#C44E52",
    "purple": "#7B61A8",
    "gray": "#68717A",
    "dark": "#263238",
}


def setup_style() -> None:
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
            "figure.dpi": 120,
            "savefig.dpi": 330,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def box_no(col: str) -> int:
    return int(re.match(r"^(\d+)#", col).group(1))


def get_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    pressure_cols = sorted(
        [c for c in df.columns if re.match(r"^\d+#风箱负压$", c)], key=box_no
    )
    south_cols = sorted(
        [c for c in df.columns if re.match(r"^\d+#风箱南侧温度$", c)], key=box_no
    )
    north_cols = sorted(
        [c for c in df.columns if re.match(r"^\d+#风箱北侧温度$", c)], key=box_no
    )
    return {
        "pressure": pressure_cols,
        "south": south_cols,
        "north": north_cols,
        "flue_pressure": ["1#大烟道负压", "2#大烟道负压"],
        "flue_temp": ["1#大烟道温度", "2#大烟道温度"],
    }


def build_matrices(df: pd.DataFrame, groups: dict[str, list[str]]) -> dict[str, np.ndarray]:
    pressure = df[groups["pressure"]].to_numpy(float)
    south = df[groups["south"]].to_numpy(float)
    north = df[groups["north"]].to_numpy(float)
    temp = np.nanmean(np.stack([south, north]), axis=0)
    time_min = (df["序列"].to_numpy(float) - 1) * SAMPLE_SECONDS / 60
    return {
        "pressure": pressure,
        "pressure_strength": -pressure,
        "temp": temp,
        "south": south,
        "north": north,
        "time_min": time_min,
        "boxes": np.arange(1, pressure.shape[1] + 1),
    }


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan
    x0, y0 = x[mask], y[mask]
    if np.nanstd(x0) < 1e-12 or np.nanstd(y0) < 1e-12:
        return np.nan
    return float(np.corrcoef(x0, y0)[0, 1])


def lag_corr_matrix(features: np.ndarray, target: np.ndarray, lags: np.ndarray) -> np.ndarray:
    out = np.full((features.shape[1], len(lags)), np.nan)
    for j in range(features.shape[1]):
        x = features[:, j]
        for k, lag in enumerate(lags):
            if lag == 0:
                out[j, k] = corr(x, target)
            else:
                out[j, k] = corr(x[:-lag], target[lag:])
    return out


def paired_lag_corr_matrix(source: np.ndarray, target: np.ndarray, lags: np.ndarray) -> np.ndarray:
    out = np.full((source.shape[1], len(lags)), np.nan)
    for j in range(source.shape[1]):
        x = source[:, j]
        y = target[:, j]
        for k, lag in enumerate(lags):
            if lag == 0:
                out[j, k] = corr(x, y)
            else:
                out[j, k] = corr(x[:-lag], y[lag:])
    return out


def quantile_band(ax: plt.Axes, x: np.ndarray, data: np.ndarray, color: str, label: str) -> None:
    q05, q25, med, q75, q95 = np.nanpercentile(data, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(x, q05, q95, color=color, alpha=0.14, linewidth=0, label="5%-95%")
    ax.fill_between(x, q25, q75, color=color, alpha=0.25, linewidth=0, label="25%-75%")
    ax.plot(x, med, color=color, linewidth=2.2, marker="o", markersize=4, label=label)


def fig_01_overview(df: pd.DataFrame, m: dict[str, np.ndarray], g: dict[str, list[str]]) -> str:
    name = "必备_01_题目变量结构与CO时序总览.png"
    time = m["time_min"]
    co = df[TARGET].to_numpy(float)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6))

    axes[0, 0].plot(time, co, color=COLORS["gray"], alpha=0.35, linewidth=0.8, label="原始CO")
    axes[0, 0].plot(time, rolling(co, 31), color=COLORS["red"], linewidth=2, label="约1分钟滚动均值")
    axes[0, 0].axhline(CO_STANDARD, color=COLORS["dark"], linestyle="--", linewidth=1.2, label="2800 mg/m³标准线")
    axes[0, 0].set_title("CO浓度随时间变化")
    axes[0, 0].set_xlabel("时间/min")
    axes[0, 0].set_ylabel("CO/(mg/m³)")
    axes[0, 0].legend(frameon=False, loc="upper right")

    speed = df["烧结机机速L1设定"].to_numpy(float)
    axes[0, 1].plot(time, speed, color=COLORS["blue"], linewidth=1.5)
    axes[0, 1].set_title("烧结机机速设定")
    axes[0, 1].set_xlabel("时间/min")
    axes[0, 1].set_ylabel("m/min")

    flue_p = -df[g["flue_pressure"]].mean(axis=1).to_numpy(float)
    flue_t = df[g["flue_temp"]].mean(axis=1).to_numpy(float)
    ax = axes[1, 0]
    ax.plot(time, rolling(flue_p, 31), color=COLORS["purple"], linewidth=1.8, label="大烟道负压强度")
    ax.set_xlabel("时间/min")
    ax.set_ylabel("负压强度/(-KPa)")
    ax2 = ax.twinx()
    ax2.plot(time, rolling(flue_t, 31), color=COLORS["orange"], linewidth=1.8, label="大烟道温度")
    ax2.set_ylabel("温度/℃")
    ax.set_title("大烟道抽风与热状态")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")

    labels = ["序列", "机速", "风箱负压", "风箱温度", "大烟道", "CO输出"]
    counts = [1, 1, 18, 36, 4, 1]
    axes[1, 1].bar(labels, counts, color=[COLORS["gray"], COLORS["blue"], COLORS["purple"], COLORS["orange"], COLORS["green"], COLORS["red"]])
    axes[1, 1].set_title("原始字段结构")
    axes[1, 1].set_ylabel("字段数")
    for i, v in enumerate(counts):
        axes[1, 1].text(i, v + 0.6, str(v), ha="center", va="bottom", color=COLORS["dark"])

    fig.suptitle("烧结低碳排放赛题A：原始数据结构与核心输出", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, name)
    return name


def fig_02_co_distribution(df: pd.DataFrame, m: dict[str, np.ndarray]) -> str:
    name = "必备_02_CO浓度分布与排放标准.png"
    co = df[TARGET].to_numpy(float)
    time = m["time_min"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})

    axes[0].hist(co, bins=42, color=COLORS["red"], alpha=0.72, edgecolor="white")
    axes[0].axvline(CO_STANDARD, color=COLORS["dark"], linestyle="--", linewidth=1.4)
    axes[0].axvline(np.nanmedian(co), color=COLORS["orange"], linewidth=1.7, label=f"中位数 {np.nanmedian(co):.0f}")
    axes[0].set_title("CO浓度分布")
    axes[0].set_xlabel("CO/(mg/m³)")
    axes[0].set_ylabel("样本数")
    axes[0].legend(frameon=False)

    sorted_co = np.sort(co[np.isfinite(co)])
    ecdf = np.arange(1, len(sorted_co) + 1) / len(sorted_co)
    axes[1].plot(sorted_co, ecdf, color=COLORS["blue"], linewidth=2)
    axes[1].axvline(CO_STANDARD, color=COLORS["dark"], linestyle="--", linewidth=1.2)
    axes[1].set_title("经验累积分布")
    axes[1].set_xlabel("CO/(mg/m³)")
    axes[1].set_ylabel("累计比例")

    exceed = co > CO_STANDARD
    axes[2].fill_between(time, 0, 1, where=exceed, color=COLORS["red"], alpha=0.35, transform=axes[2].get_xaxis_transform(), label="高于2800")
    axes[2].plot(time, rolling(co, 31), color=COLORS["dark"], linewidth=1.5)
    axes[2].axhline(CO_STANDARD, color=COLORS["dark"], linestyle="--", linewidth=1.0)
    axes[2].set_title(f"超标时间覆盖：{exceed.mean() * 100:.1f}%")
    axes[2].set_xlabel("时间/min")
    axes[2].set_ylabel("CO/(mg/m³)")
    axes[2].legend(frameon=False, loc="upper right")

    fig.tight_layout()
    save(fig, name)
    return name


def fig_03_spacetime_heatmap(m: dict[str, np.ndarray]) -> str:
    name = "必备_03_风箱负压与温度时空热图.png"
    time = m["time_min"]
    boxes = m["boxes"]
    pressure = m["pressure_strength"]
    temp = m["temp"]
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 7.2), sharex=True)

    im0 = axes[0].imshow(
        pressure.T,
        aspect="auto",
        origin="lower",
        extent=[time.min(), time.max(), boxes.min(), boxes.max()],
        cmap="cividis",
        interpolation="bilinear",
        vmin=np.nanpercentile(pressure, 2),
        vmax=np.nanpercentile(pressure, 98),
    )
    axes[0].set_title("18个风箱负压强度时空分布")
    axes[0].set_ylabel("风箱编号")
    fig.colorbar(im0, ax=axes[0], label="负压强度/(-KPa)", pad=0.01)

    im1 = axes[1].imshow(
        temp.T,
        aspect="auto",
        origin="lower",
        extent=[time.min(), time.max(), boxes.min(), boxes.max()],
        cmap="magma",
        interpolation="bilinear",
        vmin=np.nanpercentile(temp, 2),
        vmax=np.nanpercentile(temp, 98),
    )
    axes[1].set_title("南北合并风箱温度时空分布")
    axes[1].set_xlabel("时间/min")
    axes[1].set_ylabel("风箱编号")
    fig.colorbar(im1, ax=axes[1], label="温度/℃", pad=0.01)

    fig.tight_layout()
    save(fig, name)
    return name


def fig_04_profiles(m: dict[str, np.ndarray]) -> str:
    name = "必备_04_风箱负压温度空间剖面.png"
    boxes = m["boxes"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))

    quantile_band(axes[0], boxes, m["pressure_strength"], COLORS["purple"], "中位负压强度")
    axes[0].set_title("各风箱负压调控剖面")
    axes[0].set_xlabel("风箱编号")
    axes[0].set_ylabel("负压强度/(-KPa)")
    axes[0].set_xticks(boxes)
    axes[0].legend(frameon=False)

    quantile_band(axes[1], boxes, m["temp"], COLORS["orange"], "中位合并温度")
    axes[1].set_title("各风箱热状态剖面")
    axes[1].set_xlabel("风箱编号")
    axes[1].set_ylabel("温度/℃")
    axes[1].set_xticks(boxes)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    save(fig, name)
    return name


def fig_05_pressure_temp_relation(m: dict[str, np.ndarray]) -> str:
    name = "必备_05_逐风箱负压温度关系证据.png"
    boxes = m["boxes"]
    pressure = m["pressure_strength"]
    temp = m["temp"]
    lags = np.arange(0, 91, 3)
    lag_corr = paired_lag_corr_matrix(pressure, temp, lags)
    lag0 = lag_corr[:, 0]
    best_idx = np.nanargmax(np.abs(lag_corr), axis=1)
    best = lag_corr[np.arange(len(boxes)), best_idx]
    best_seconds = lags[best_idx] * SAMPLE_SECONDS

    fig = plt.figure(figsize=(13.2, 8.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1], hspace=0.42, wspace=0.34)
    ax0 = fig.add_subplot(gs[0, :2])
    ax1 = fig.add_subplot(gs[0, 2:])

    width = 0.36
    ax0.bar(boxes - width / 2, lag0, width=width, color=COLORS["blue"], label="同步相关")
    ax0.bar(boxes + width / 2, best, width=width, color=COLORS["orange"], label="0-180s内最佳领先相关")
    ax0.axhline(0, color=COLORS["dark"], linewidth=0.8)
    ax0.set_xticks(boxes)
    ax0.set_title("负压强度与对应合并温度的相关性")
    ax0.set_xlabel("风箱编号")
    ax0.set_ylabel("Pearson r")
    ax0.legend(frameon=False)

    ax1.plot(boxes, best_seconds, color=COLORS["green"], marker="o", linewidth=2)
    ax1.set_xticks(boxes)
    ax1.set_title("各风箱负压领先温度的经验时滞")
    ax1.set_xlabel("风箱编号")
    ax1.set_ylabel("最佳领先时间/s")

    reps = [1, 6, 12, 18]
    for i, b in enumerate(reps):
        ax = fig.add_subplot(gs[1, i])
        idx = b - 1
        x = pressure[:, idx]
        y = temp[:, idx]
        mask = np.isfinite(x) & np.isfinite(y)
        coef = np.polyfit(x[mask], y[mask], deg=1)
        xs = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 100)
        ax.scatter(x[::5], y[::5], s=10, alpha=0.28, color=COLORS["purple"], edgecolor="none")
        ax.plot(xs, coef[0] * xs + coef[1], color=COLORS["red"], linewidth=2)
        ax.set_title(f"{b}#风箱")
        ax.set_xlabel("负压强度/(-KPa)")
        if i == 0:
            ax.set_ylabel("合并温度/℃")
        ax.text(0.04, 0.92, f"r={lag0[idx]:.2f}", transform=ax.transAxes, color=COLORS["dark"])

    fig.suptitle("问题1必备：风箱负压与对应温度关系", fontsize=15, fontweight="bold")
    save(fig, name)
    return name


def fig_06_co_lag(df: pd.DataFrame, m: dict[str, np.ndarray]) -> str:
    name = "必备_06_CO影响规律与时滞相关热图.png"
    co = df[TARGET].to_numpy(float)
    boxes = m["boxes"]
    lags = np.arange(0, 151, 5)
    lag_seconds = lags * SAMPLE_SECONDS
    pressure_corr = lag_corr_matrix(m["pressure_strength"], co, lags)
    temp_corr = lag_corr_matrix(m["temp"], co, lags)
    vmax = np.nanmax(np.abs(np.r_[pressure_corr.ravel(), temp_corr.ravel()]))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), gridspec_kw={"width_ratios": [1.2, 1.2, 0.82]})
    im0 = axes[0].imshow(
        pressure_corr,
        aspect="auto",
        origin="lower",
        extent=[lag_seconds.min(), lag_seconds.max(), boxes.min(), boxes.max()],
        cmap="RdBu_r",
        norm=norm,
        interpolation="bilinear",
    )
    axes[0].set_title("风箱负压强度领先CO的相关性")
    axes[0].set_xlabel("变量领先CO时间/s")
    axes[0].set_ylabel("风箱编号")

    im1 = axes[1].imshow(
        temp_corr,
        aspect="auto",
        origin="lower",
        extent=[lag_seconds.min(), lag_seconds.max(), boxes.min(), boxes.max()],
        cmap="RdBu_r",
        norm=norm,
        interpolation="bilinear",
    )
    axes[1].set_title("风箱温度领先CO的相关性")
    axes[1].set_xlabel("变量领先CO时间/s")
    axes[1].set_ylabel("风箱编号")
    fig.colorbar(im1, ax=axes[:2], label="Pearson r", pad=0.012)

    pressure_best = lag_seconds[np.nanargmax(np.abs(pressure_corr), axis=1)]
    temp_best = lag_seconds[np.nanargmax(np.abs(temp_corr), axis=1)]
    axes[2].plot(pressure_best, boxes, color=COLORS["purple"], marker="o", label="负压")
    axes[2].plot(temp_best, boxes, color=COLORS["orange"], marker="s", label="温度")
    axes[2].set_title("最大相关对应时滞")
    axes[2].set_xlabel("领先时间/s")
    axes[2].set_ylabel("风箱编号")
    axes[2].set_yticks(boxes)
    axes[2].legend(frameon=False)

    fig.suptitle("问题2必备：CO预测需考虑空间位置与时滞配准", fontsize=15, fontweight="bold")
    save(fig, name)
    return name


def fig_07_high_low_fingerprint(df: pd.DataFrame, m: dict[str, np.ndarray]) -> str:
    name = "必备_07_高低CO工况风箱空间指纹.png"
    co = df[TARGET].to_numpy(float)
    boxes = m["boxes"]
    low = co <= np.nanpercentile(co, 20)
    high = co >= np.nanpercentile(co, 80)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    axes[0].plot(boxes, np.nanmean(m["pressure_strength"][low], axis=0), color=COLORS["blue"], marker="o", label="低CO 20%")
    axes[0].plot(boxes, np.nanmean(m["pressure_strength"][high], axis=0), color=COLORS["red"], marker="o", label="高CO 20%")
    axes[0].set_title("高低CO样本的负压空间差异")
    axes[0].set_xlabel("风箱编号")
    axes[0].set_ylabel("负压强度/(-KPa)")
    axes[0].set_xticks(boxes)
    axes[0].legend(frameon=False)

    axes[1].plot(boxes, np.nanmean(m["temp"][low], axis=0), color=COLORS["blue"], marker="s", label="低CO 20%")
    axes[1].plot(boxes, np.nanmean(m["temp"][high], axis=0), color=COLORS["red"], marker="s", label="高CO 20%")
    axes[1].set_title("高低CO样本的热状态空间差异")
    axes[1].set_xlabel("风箱编号")
    axes[1].set_ylabel("合并温度/℃")
    axes[1].set_xticks(boxes)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    save(fig, name)
    return name


def fig_08_pressure_ranges(m: dict[str, np.ndarray]) -> str:
    name = "必备_08_风箱负压调节范围.png"
    boxes = m["boxes"]
    strength = m["pressure_strength"]
    stats = np.nanpercentile(strength, [0, 5, 25, 50, 75, 95, 100], axis=0)
    fig, ax = plt.subplots(figsize=(12.4, 4.8))
    ax.vlines(boxes, stats[0], stats[6], color=COLORS["gray"], alpha=0.45, linewidth=2, label="最小-最大")
    ax.vlines(boxes, stats[1], stats[5], color=COLORS["purple"], linewidth=6, alpha=0.45, label="5%-95%建议范围")
    ax.vlines(boxes, stats[2], stats[4], color=COLORS["purple"], linewidth=10, alpha=0.75, label="25%-75%常用区间")
    ax.scatter(boxes, stats[3], color=COLORS["orange"], s=42, zorder=3, label="中位数")
    ax.set_title("问题3必备：由原始数据确定各风箱负压调节范围")
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("负压强度/(-KPa)")
    ax.set_xticks(boxes)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    save(fig, name)
    return name


def fig_09_quality(df: pd.DataFrame, m: dict[str, np.ndarray], g: dict[str, list[str]]) -> str:
    name = "必备_09_原始数据质量与变量分布概览.png"
    groups = {
        "机速": ["烧结机机速L1设定"],
        "风箱负压": g["pressure"],
        "风箱温度": g["south"] + g["north"],
        "大烟道": g["flue_pressure"] + g["flue_temp"],
        "CO": [TARGET],
    }
    missing = [df[cols].isna().mean().mean() * 100 for cols in groups.values()]
    outlier_rates = []
    for cols in groups.values():
        vals = df[cols].to_numpy(float).ravel()
        vals = vals[np.isfinite(vals)]
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        outlier_rates.append(((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).mean() * 100 if iqr > 0 else 0)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), gridspec_kw={"width_ratios": [0.9, 0.9, 1.3]})
    labels = list(groups.keys())
    axes[0].bar(labels, missing, color=COLORS["green"])
    axes[0].set_title("缺失率")
    axes[0].set_ylabel("%")
    axes[0].set_ylim(0, max(1, max(missing) * 1.3))

    axes[1].bar(labels, outlier_rates, color=COLORS["orange"])
    axes[1].set_title("IQR口径异常点比例")
    axes[1].set_ylabel("%")

    scaled = []
    box_labels = []
    for label, cols in groups.items():
        vals = df[cols].to_numpy(float).ravel()
        vals = vals[np.isfinite(vals)]
        vals = vals[:: max(1, len(vals) // 1500)]
        scaled.append((vals - np.median(vals)) / (np.percentile(vals, 75) - np.percentile(vals, 25) + 1e-9))
        box_labels.append(label)
    axes[2].boxplot(scaled, tick_labels=box_labels, showfliers=False, patch_artist=True)
    for patch, color in zip(axes[2].artists, [COLORS["blue"], COLORS["purple"], COLORS["orange"], COLORS["green"], COLORS["red"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    axes[2].set_title("变量组稳健标准化分布")
    axes[2].set_ylabel("(x-中位数)/IQR")
    axes[2].axhline(0, color=COLORS["dark"], linewidth=0.8)

    fig.suptitle("建模前必备：数据完整性与分布尺度检查", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, name)
    return name


def smooth_spacetime_surface(matrix: np.ndarray, time_min: np.ndarray, boxes: np.ndarray, sigma: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z0 = gaussian_filter(matrix, sigma=sigma)
    t0 = np.arange(matrix.shape[0])
    spline = RectBivariateSpline(t0, boxes, z0, kx=3, ky=3)
    t_new = np.linspace(t0.min(), t0.max(), 220)
    b_new = np.linspace(boxes.min(), boxes.max(), 90)
    z = spline(t_new, b_new)
    x, y = np.meshgrid(b_new, np.interp(t_new, t0, time_min))
    return x, y, z


def plot_3d_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, title: str, zlabel: str, cmap: str, name: str) -> str:
    fig = plt.figure(figsize=(11.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.96, rcount=180, ccount=90)
    ax.set_title(title, pad=16, fontsize=14, fontweight="bold")
    ax.set_xlabel("风箱编号", labelpad=8)
    ax.set_ylabel("时间/min", labelpad=8)
    ax.set_zlabel(zlabel, labelpad=8)
    ax.view_init(elev=28, azim=-132)
    ax.xaxis.pane.set_alpha(0.02)
    ax.yaxis.pane.set_alpha(0.02)
    ax.zaxis.pane.set_alpha(0.02)
    fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.08, label=zlabel)
    save(fig, name)
    return name


def fig_10_3d_pressure(m: dict[str, np.ndarray]) -> str:
    name = "可选_10_3D风箱负压时空平滑曲面.png"
    x, y, z = smooth_spacetime_surface(m["pressure_strength"], m["time_min"], m["boxes"], sigma=(8.0, 0.8))
    return plot_3d_surface(x, y, z, "可选3D：风箱负压强度时空平滑曲面", "负压强度/(-KPa)", "viridis", name)


def fig_11_3d_temp(m: dict[str, np.ndarray]) -> str:
    name = "可选_11_3D风箱温度时空平滑曲面.png"
    x, y, z = smooth_spacetime_surface(m["temp"], m["time_min"], m["boxes"], sigma=(9.0, 0.9))
    return plot_3d_surface(x, y, z, "可选3D：南北合并温度时空平滑曲面", "温度/℃", "plasma", name)


def fig_12_3d_co_response(df: pd.DataFrame, m: dict[str, np.ndarray]) -> str:
    name = "可选_12_3D_CO响应平滑曲面.png"
    co = df[TARGET].to_numpy(float)
    pressure_corr = np.array([abs(corr(m["pressure_strength"][:, i], co)) for i in range(18)])
    temp_corr = np.array([abs(corr(m["temp"][:, i], co)) for i in range(18)])
    p_idx = int(np.nanargmax(pressure_corr))
    t_idx = int(np.nanargmax(temp_corr))
    x = m["pressure_strength"][:, p_idx]
    y = m["temp"][:, t_idx]
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(co)
    x, y, z = x[mask], y[mask], co[mask]
    xr = (np.percentile(x, 2), np.percentile(x, 98))
    yr = (np.percentile(y, 2), np.percentile(y, 98))
    z_range = (np.percentile(z, 2), np.percentile(z, 98))
    z_clip = np.clip(z, *z_range)
    stat, x_edge, y_edge, _ = binned_statistic_2d(
        x,
        y,
        z_clip,
        statistic="median",
        bins=56,
        range=[xr, yr],
    )
    xi = (x_edge[:-1] + x_edge[1:]) / 2
    yi = (y_edge[:-1] + y_edge[1:]) / 2
    xx, yy = np.meshgrid(xi, yi)
    zz = stat.T
    valid = np.isfinite(zz)
    fill = griddata((xx[valid], yy[valid]), zz[valid], (xx, yy), method="nearest")
    zz = np.where(valid, zz, fill)
    zz = gaussian_filter(zz, sigma=2.6)

    fig = plt.figure(figsize=(10.8, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, zz, cmap="cividis", linewidth=0, antialiased=True, alpha=0.95)
    ax.scatter(x[::12], y[::12], z_clip[::12], s=6, color=COLORS["dark"], alpha=0.16)
    ax.set_title(f"可选3D：CO响应平滑曲面（{p_idx + 1}#负压强度 × {t_idx + 1}#温度）", pad=16, fontsize=13, fontweight="bold")
    ax.set_xlabel(f"{p_idx + 1}#风箱负压强度/(-KPa)", labelpad=8)
    ax.set_ylabel(f"{t_idx + 1}#风箱合并温度/℃", labelpad=8)
    ax.set_zlabel("CO/(mg/m³)", labelpad=8)
    ax.set_zlim(*z_range)
    ax.view_init(elev=28, azim=-138)
    ax.xaxis.pane.set_alpha(0.02)
    ax.yaxis.pane.set_alpha(0.02)
    ax.zaxis.pane.set_alpha(0.02)
    fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.08, label="CO/(mg/m³)")
    save(fig, name)
    return name


def fig_13_pca(df: pd.DataFrame, m: dict[str, np.ndarray], g: dict[str, list[str]]) -> str:
    name = "可选_13_运行工况PCA嵌入与CO梯度.png"
    cols = ["烧结机机速L1设定"] + g["pressure"] + g["south"] + g["north"] + g["flue_pressure"] + g["flue_temp"]
    x = df[cols].to_numpy(float)
    mask = np.isfinite(x).all(axis=1) & np.isfinite(df[TARGET].to_numpy(float))
    xs = StandardScaler().fit_transform(x[mask])
    pc = PCA(n_components=2, random_state=0).fit_transform(xs)
    co = df.loc[mask, TARGET].to_numpy(float)

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    sc = ax.scatter(pc[:, 0], pc[:, 1], c=co, cmap="cividis", s=18, alpha=0.78, edgecolor="none")
    ax.set_title("可选：多变量运行工况二维嵌入")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, label="CO/(mg/m³)")
    fig.tight_layout()
    save(fig, name)
    return name


def fig_14_flue_phase(df: pd.DataFrame, g: dict[str, list[str]]) -> str:
    name = "可选_14_大烟道状态与CO相图.png"
    co = df[TARGET].to_numpy(float)
    flue_p = -df[g["flue_pressure"]].mean(axis=1).to_numpy(float)
    flue_t = df[g["flue_temp"]].mean(axis=1).to_numpy(float)
    speed = df["烧结机机速L1设定"].to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    h = axes[0].hexbin(flue_p, flue_t, C=co, gridsize=38, cmap="viridis", reduce_C_function=np.nanmean, mincnt=1)
    axes[0].set_title("大烟道负压-温度相图（颜色为平均CO）")
    axes[0].set_xlabel("大烟道负压强度/(-KPa)")
    axes[0].set_ylabel("大烟道温度/℃")
    fig.colorbar(h, ax=axes[0], label="平均CO/(mg/m³)")

    axes[1].scatter(speed, co, c=flue_t, cmap="plasma", s=18, alpha=0.72, edgecolor="none")
    axes[1].axhline(CO_STANDARD, color=COLORS["dark"], linestyle="--", linewidth=1.1)
    axes[1].set_title("机速与CO散点（颜色为大烟道温度）")
    axes[1].set_xlabel("烧结机机速/(m/min)")
    axes[1].set_ylabel("CO/(mg/m³)")
    fig.tight_layout()
    save(fig, name)
    return name


def write_index(files: list[tuple[str, str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 原始数据可视化PNG图表索引",
        "",
        "数据来源：`附件1.原始数据.xlsx`；采样间隔：2 s；风箱温度采用南北两侧均值合并。",
        "",
        "| 文件 | 类型 | 建模用途 |",
        "|---|---|---|",
    ]
    for filename, kind, use in files:
        lines.append(f"| `{filename}` | {kind} | {use} |")
    lines += [
        "",
        "说明：`必备`图对应题目三问的基础证据；`可选`图用于增强论文展示，其中3D图经过高斯平滑与样条/网格插值，避免曲面过度锯齿化。",
    ]
    (OUT_DIR / "图表索引.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_style()
    df = load_raw()
    groups = get_groups(df)
    matrices = build_matrices(df, groups)

    generated: list[tuple[str, str, str]] = []
    tasks = [
        (fig_01_overview, "必备", "全题数据结构、时间跨度和CO输出总览"),
        (fig_02_co_distribution, "必备", "明确CO分布、排放标准线和建模目标难度"),
        (fig_03_spacetime_heatmap, "必备", "展示18个风箱的时空耦合结构"),
        (fig_04_profiles, "必备", "问题1/3所需的空间剖面与负压范围基础"),
        (fig_05_pressure_temp_relation, "必备", "问题1：负压预测对应温度的关系证据"),
        (fig_06_co_lag, "必备", "问题2：CO预测的变量影响与时滞配准证据"),
        (fig_07_high_low_fingerprint, "必备", "问题2/3：高低CO工况差异与调控方向提示"),
        (fig_08_pressure_ranges, "必备", "问题3：由原始数据估计各风箱负压调节范围"),
        (fig_09_quality, "必备", "建模前的数据完整性、异常点和尺度检查"),
        (fig_10_3d_pressure, "可选3D", "平滑展示风箱负压随时间和空间演化"),
        (fig_11_3d_temp, "可选3D", "平滑展示风箱热状态随时间和空间演化"),
        (fig_12_3d_co_response, "可选3D", "平滑展示CO对关键负压/温度变量的响应形态"),
        (fig_13_pca, "可选", "多变量工况聚类与CO梯度的直观展示"),
        (fig_14_flue_phase, "可选", "大烟道状态、机速与CO的辅助相图"),
    ]

    for func, kind, use in tasks:
        if func in {fig_01_overview, fig_09_quality, fig_13_pca}:
            filename = func(df, matrices, groups)
        elif func in {fig_02_co_distribution, fig_06_co_lag, fig_07_high_low_fingerprint, fig_12_3d_co_response}:
            filename = func(df, matrices)
        elif func is fig_14_flue_phase:
            filename = func(df, groups)
        else:
            filename = func(matrices)
        generated.append((filename, kind, use))
        print(filename)

    write_index(generated)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
