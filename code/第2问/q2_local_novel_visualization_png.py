# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[2]
MID_DIR = ROOT / "data" / "第2问" / "中间数据"
RESULT_DIR = ROOT / "data" / "第2问" / "结果数据"
OUT_DIR = ROOT / "figure" / "第2问" / "局部新颖PNG"

TARGET = "烧结大烟道外排CO浓度"
PRED = "监测型模型_预测CO浓度"
RESID = "监测型模型_残差"
STANDARD_CO = 2800
SAMPLE_SECONDS = 2

COLORS = {
    "ink": "#263238",
    "blue": "#2F6B9A",
    "light_blue": "#7AA6C2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#C44E52",
    "purple": "#7B61A8",
    "gray": "#6E7781",
    "pale": "#EEF3F6",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


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
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
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


def windbox_no(text: str) -> int | None:
    match = re.match(r"(\d+)#风箱", str(text))
    return int(match.group(1)) if match else None


def merged_frame() -> pd.DataFrame:
    pred = read_csv(RESULT_DIR / "CO预测结果_全样本.csv")
    aligned = read_csv(MID_DIR / "滞后配准建模数据.csv")
    lag_cols = [c for c in aligned.columns if "__lag" in c or "__rolling" in c]
    return pred.merge(aligned[["序列"] + lag_cols], on="序列", how="left")


def local_window(df: pd.DataFrame, center_index: int, radius: int) -> pd.DataFrame:
    loc = df.index.get_loc(center_index) if center_index in df.index else center_index
    lo = max(0, loc - radius)
    hi = min(len(df), loc + radius + 1)
    return df.iloc[lo:hi].copy()


def zscore(series: pd.Series, ref: pd.Series | None = None) -> pd.Series:
    values = ref if ref is not None else series
    std = float(values.std())
    if not np.isfinite(std) or std == 0:
        return series * 0
    return (series - float(values.mean())) / std


def test_peak_window(radius: int = 90) -> tuple[pd.DataFrame, int]:
    df = merged_frame()
    test = df[df["数据集"] == "测试集"].copy()
    center = int(test[TARGET].idxmax())
    return local_window(df, center, radius), center


def plot_local_peak_lens() -> None:
    df = merged_frame()
    window, center = test_peak_window(radius=100)
    window["相对时间_s"] = window["时间_s"] - df.loc[center, "时间_s"]

    key_features = [
        ("1#大烟道负压强度__lag0", "1#大烟道负压强度"),
        ("烧结机机速L1设定__lag60", "机速(120s滞后)"),
        ("1#风箱平均温度__lag150", "1#温度(300s滞后)"),
        ("9#风箱平均温度__lag150", "9#温度(300s滞后)"),
        ("18#风箱负压强度__lag23", "18#负压(46s滞后)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12.2, 8.8), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(window["相对时间_s"], window[TARGET], color=COLORS["ink"], lw=2.0, label="实际 CO")
    ax.plot(window["相对时间_s"], window[PRED], color=COLORS["orange"], lw=1.7, label="推荐监测模型")
    ax.axhline(STANDARD_CO, color=COLORS["red"], ls="--", lw=1.0, label="2800 mg/m$^3$")
    ax.axvline(0, color=COLORS["gray"], ls=":", lw=1.2)
    ax.set_title("A  测试集最高 CO 峰值的局部放大")
    ax.set_ylabel("CO (mg/m$^3$)")
    ax.legend(ncol=3, frameon=False, loc="upper right")

    ax = axes[1]
    ax.axhline(0, color=COLORS["gray"], lw=1)
    colors = np.where(window[RESID] >= 0, COLORS["red"], COLORS["blue"])
    ax.bar(window["相对时间_s"], window[RESID], width=1.8, color=colors, alpha=0.72)
    ax.axvline(0, color=COLORS["gray"], ls=":", lw=1.2)
    ax.set_title("B  峰值附近预测残差：正值表示高估")
    ax.set_ylabel("残差")

    ax = axes[2]
    ref = df[df["数据集"] == "训练集"]
    offsets = np.linspace(2.4, -2.4, len(key_features))
    for (col, label), offset in zip(key_features, offsets):
        if col not in window:
            continue
        signal = zscore(window[col], ref[col]) + offset
        ax.plot(window["相对时间_s"], signal, lw=1.55, label=label)
    ax.axvline(0, color=COLORS["gray"], ls=":", lw=1.2)
    ax.set_title("C  同一局部窗口内关键滞后变量的标准化轨迹")
    ax.set_xlabel("相对峰值时间 (s)")
    ax.set_ylabel("标准化轨迹(错位显示)")
    ax.set_yticks([])
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save(fig, "新颖_01_最高CO峰值局部透镜.png")


def choose_separated_events(df: pd.DataFrame, score_col: str, count: int, min_gap: int) -> list[int]:
    order = df[score_col].sort_values(ascending=False).index.tolist()
    chosen: list[int] = []
    positions = {idx: pos for pos, idx in enumerate(df.index)}
    for idx in order:
        pos = positions[idx]
        if all(abs(pos - positions[c]) >= min_gap for c in chosen):
            chosen.append(int(idx))
        if len(chosen) == count:
            break
    return chosen


def plot_residual_casebook() -> None:
    df = merged_frame()
    test = df[df["数据集"] == "测试集"].copy()
    test["abs_resid"] = test[RESID].abs()
    centers = choose_separated_events(test, "abs_resid", count=4, min_gap=55)

    fig, axes = plt.subplots(4, 2, figsize=(12, 9.2), constrained_layout=True)
    for row, center in enumerate(centers):
        window = local_window(df, center, 42)
        window["相对时间_s"] = window["时间_s"] - df.loc[center, "时间_s"]

        ax = axes[row, 0]
        ax.plot(window["相对时间_s"], window[TARGET], color=COLORS["ink"], lw=1.7, label="实际")
        ax.plot(window["相对时间_s"], window[PRED], color=COLORS["orange"], lw=1.45, label="预测")
        ax.axvline(0, color=COLORS["gray"], ls=":", lw=1)
        ax.set_ylabel("CO")
        ax.set_title(f"片段 {row + 1}: |残差|={abs(df.loc[center, RESID]):.1f}")
        if row == 0:
            ax.legend(frameon=False, ncol=2, loc="upper right")

        ax = axes[row, 1]
        ax.axhline(0, color=COLORS["gray"], lw=1)
        colors = np.where(window[RESID] >= 0, COLORS["red"], COLORS["blue"])
        ax.bar(window["相对时间_s"], window[RESID], width=1.8, color=colors, alpha=0.76)
        ax.axvline(0, color=COLORS["gray"], ls=":", lw=1)
        ax.set_ylabel("残差")
        ax.set_title("局部误差形态")
    for ax in axes[-1, :]:
        ax.set_xlabel("相对最大误差时刻 (s)")
    save(fig, "新颖_02_最大残差片段病例册.png")


def plot_peak_aligned_dynamics() -> None:
    clean = read_csv(MID_DIR / "清洗后数据.csv")
    pred = read_csv(RESULT_DIR / "CO预测结果_全样本.csv")
    df = clean.merge(pred[["序列", "数据集", PRED]], on="序列", how="left")
    test = df[df["数据集"] == "测试集"].copy()
    y = test[TARGET].to_numpy()
    peaks, props = find_peaks(y, distance=35, prominence=80)
    peak_rows = test.iloc[peaks].nlargest(8, TARGET)
    centers = peak_rows.index.tolist()

    variables = [
        (TARGET, "CO", COLORS["ink"]),
        ("1#大烟道负压强度", "1#大烟道负压强度", COLORS["blue"]),
        ("烧结机机速L1设定", "机速", COLORS["green"]),
        ("1#风箱平均温度", "1#温度", COLORS["red"]),
        ("18#风箱负压强度", "18#负压强度", COLORS["purple"]),
    ]
    radius = 90
    rel = np.arange(-radius, radius + 1) * SAMPLE_SECONDS
    stacks: dict[str, list[np.ndarray]] = {col: [] for col, _, _ in variables}
    co_stack: list[np.ndarray] = []

    for center in centers:
        loc = df.index.get_loc(center)
        if loc - radius < 0 or loc + radius >= len(df):
            continue
        window = df.iloc[loc - radius : loc + radius + 1]
        for col, _, _ in variables:
            stacks[col].append(zscore(window[col], df[df["数据集"] == "训练集"][col]).to_numpy())
        co_stack.append(zscore(window[TARGET], df[df["数据集"] == "训练集"][TARGET]).to_numpy())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    ax = axes[0]
    for col, label, color in variables:
        arr = np.vstack(stacks[col])
        median = np.nanmedian(arr, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        ax.plot(rel, median, color=color, lw=1.8, label=label)
        ax.fill_between(rel, q25, q75, color=color, alpha=0.10, lw=0)
    ax.axvline(0, color=COLORS["gray"], ls=":", lw=1)
    ax.set_title("A  峰值对齐后的中位动态轨迹")
    ax.set_xlabel("相对 CO 峰值时间 (s)")
    ax.set_ylabel("训练集标准化值")
    ax.legend(frameon=False, ncol=2)

    ax = axes[1]
    co_arr = np.vstack(co_stack)
    order = np.argsort(co_arr[:, radius])[::-1]
    im = ax.imshow(co_arr[order], aspect="auto", cmap="magma", extent=[rel.min(), rel.max(), len(order) + 0.5, 0.5])
    ax.axvline(0, color="white", ls=":", lw=1)
    ax.set_title("B  多个局部峰值的 CO 形态对齐热图")
    ax.set_xlabel("相对 CO 峰值时间 (s)")
    ax.set_ylabel("峰值样本编号")
    ax.set_yticks(range(1, len(order) + 1))
    fig.colorbar(im, ax=ax, label="CO 标准化值")
    save(fig, "新颖_03_峰值对齐动态指纹.png")


def plot_local_windbox_state_heatmap() -> None:
    clean = read_csv(MID_DIR / "清洗后数据.csv")
    pred = read_csv(RESULT_DIR / "CO预测结果_全样本.csv")
    df = clean.merge(pred[["序列", "数据集", PRED]], on="序列", how="left")
    test = df[df["数据集"] == "测试集"]
    center = int(test[TARGET].idxmax())
    window = local_window(df, center, 80)
    window["相对时间_s"] = window["时间_s"] - df.loc[center, "时间_s"]
    boxes = range(1, 19)

    temp_cols = [f"{i}#风箱平均温度" for i in boxes]
    pressure_cols = [f"{i}#风箱负压强度" for i in boxes]
    train = df[df["数据集"] == "训练集"]
    temp = ((window[temp_cols] - train[temp_cols].mean()) / train[temp_cols].std()).T
    pressure = ((window[pressure_cols] - train[pressure_cols].mean()) / train[pressure_cols].std()).T

    fig = plt.figure(figsize=(12.4, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 2.2, 2.2])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(window["相对时间_s"], window[TARGET], color=COLORS["ink"], lw=1.8, label="实际 CO")
    ax.plot(window["相对时间_s"], window[PRED], color=COLORS["orange"], lw=1.5, label="预测 CO")
    ax.axvline(0, color=COLORS["gray"], ls=":", lw=1)
    ax.set_title("局部最高 CO 峰值窗口：CO 与风箱状态同步观察")
    ax.set_ylabel("CO")
    ax.legend(frameon=False, ncol=2)

    extent = [window["相对时间_s"].min(), window["相对时间_s"].max(), 18.5, 0.5]
    norm = TwoSlopeNorm(vcenter=0, vmin=-2.5, vmax=2.5)
    ax1 = fig.add_subplot(gs[1, :])
    im1 = ax1.imshow(temp.values, aspect="auto", cmap="RdBu_r", norm=norm, extent=extent)
    ax1.axvline(0, color="black", ls=":", lw=1)
    ax1.set_title("A  风箱平均温度局部标准化热图")
    ax1.set_ylabel("风箱编号")
    ax1.set_yticks([1, 6, 12, 18])
    fig.colorbar(im1, ax=ax1, label="训练集 z-score")

    ax2 = fig.add_subplot(gs[2, :])
    im2 = ax2.imshow(pressure.values, aspect="auto", cmap="PuOr_r", norm=norm, extent=extent)
    ax2.axvline(0, color="black", ls=":", lw=1)
    ax2.set_title("B  风箱负压强度局部标准化热图")
    ax2.set_xlabel("相对峰值时间 (s)")
    ax2.set_ylabel("风箱编号")
    ax2.set_yticks([1, 6, 12, 18])
    fig.colorbar(im2, ax=ax2, label="训练集 z-score")
    save(fig, "新颖_04_局部峰值风箱状态热图.png")


def plot_local_3d_state_surface() -> None:
    clean = read_csv(MID_DIR / "清洗后数据.csv")
    pred = read_csv(RESULT_DIR / "CO预测结果_全样本.csv")
    df = clean.merge(pred[["序列", "数据集"]], on="序列", how="left")
    center = int(df[df["数据集"] == "测试集"][TARGET].idxmax())
    window = local_window(df, center, 90)
    rel = (window["时间_s"] - df.loc[center, "时间_s"]).to_numpy()
    boxes = np.arange(1, 19)
    train = df[df["数据集"] == "训练集"]

    temp_cols = [f"{i}#风箱平均温度" for i in boxes]
    pressure_cols = [f"{i}#风箱负压强度" for i in boxes]
    temp = ((window[temp_cols] - train[temp_cols].mean()) / train[temp_cols].std()).to_numpy()
    pressure = ((window[pressure_cols] - train[pressure_cols].mean()) / train[pressure_cols].std()).to_numpy()
    temp = np.clip(temp, -3.0, 3.0)
    pressure = np.clip(pressure, -3.0, 3.0)
    temp = gaussian_filter(temp, sigma=(2.0, 0.8))
    pressure = gaussian_filter(pressure, sigma=(2.0, 0.8))
    X, Y = np.meshgrid(rel, boxes, indexing="ij")

    fig = plt.figure(figsize=(12.8, 6.4), constrained_layout=True)
    for idx, (data, title, cmap) in enumerate(
        [(temp, "A  局部温度 z-score 曲面", "RdBu_r"), (pressure, "B  局部负压强度 z-score 曲面", "PuOr_r")],
        start=1,
    ):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        norm = TwoSlopeNorm(vcenter=0, vmin=-2.4, vmax=2.4)
        surf = ax.plot_surface(X, Y, data, cmap=cmap, norm=norm, linewidth=0, antialiased=True, rcount=90, ccount=18)
        ax.set_title(title)
        ax.set_xlabel("相对峰值时间 (s)")
        ax.set_ylabel("风箱编号")
        ax.set_zlabel("z-score", labelpad=8)
        ax.set_yticks([1, 6, 12, 18])
        ax.view_init(elev=28, azim=-136)
        fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.08)
    save(fig, "新颖_05_局部峰值3D风箱状态曲面.png")


def plot_phase_portrait() -> None:
    pred = read_csv(RESULT_DIR / "CO预测结果_全样本.csv")
    test = pred[pred["数据集"] == "测试集"].copy()
    smooth_co = test[TARGET].rolling(5, center=True, min_periods=1).mean()
    test["CO变化速率"] = smooth_co.diff(10) / (10 * SAMPLE_SECONDS) * 60
    test["abs_resid"] = test[RESID].abs()
    test = test.dropna(subset=["CO变化速率"])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    ax = axes[0]
    sc = ax.scatter(test[TARGET], test["CO变化速率"], c=test["时间_s"] / 60, cmap="viridis", s=28, alpha=0.78, edgecolor="none")
    ax.axvline(STANDARD_CO, color=COLORS["red"], ls="--", lw=1)
    ax.axhline(0, color=COLORS["gray"], lw=1)
    ax.set_title("A  测试集 CO 运行相图：浓度水平 × 变化速率")
    ax.set_xlabel("CO 浓度 (mg/m$^3$)")
    ax.set_ylabel("CO 变化速率 (mg/m$^3$/min)")
    fig.colorbar(sc, ax=ax, label="时间 (min)")

    ax = axes[1]
    sc = ax.scatter(test[TARGET], test["CO变化速率"], c=test["abs_resid"], cmap="magma_r", s=30, alpha=0.80, edgecolor="none")
    ax.axvline(STANDARD_CO, color=COLORS["red"], ls="--", lw=1)
    ax.axhline(0, color=COLORS["gray"], lw=1)
    ax.set_title("B  相图中的预测困难区")
    ax.set_xlabel("CO 浓度 (mg/m$^3$)")
    ax.set_ylabel("CO 变化速率")
    fig.colorbar(sc, ax=ax, label="|残差|")
    save(fig, "新颖_06_CO运行相图与预测困难区.png")


def plot_lag_pareto_frontier() -> None:
    search = read_csv(MID_DIR / "滞后搜索明细.csv")
    search = search.copy()
    selected = search[search["是否简约规则入选"].astype(bool)]
    best = search.loc[search["稳健目标RMSE"].idxmin()]

    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    sc = ax.scatter(
        search["调整绝对值合计"],
        search["稳健目标RMSE"],
        c=search["非零组调整数"],
        cmap="viridis",
        s=24,
        alpha=0.45,
        edgecolor="none",
    )
    if not selected.empty:
        ax.scatter(selected["调整绝对值合计"], selected["稳健目标RMSE"], s=95, color=COLORS["orange"], edgecolor="white", linewidth=0.9, label="简约规则入选")
    ax.scatter([best["调整绝对值合计"]], [best["稳健目标RMSE"]], s=95, color=COLORS["red"], edgecolor="white", linewidth=0.9, label="理论最优")
    ax.set_title("时滞组合搜索的精度-复杂度局部帕累托图")
    ax.set_xlabel("组时滞调整绝对值合计 (样本)")
    ax.set_ylabel("稳健目标 RMSE")
    ax.legend(frameon=False)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("非零调整组数")
    save(fig, "新颖_07_时滞搜索精度复杂度帕累托.png")


def plot_response_atlas() -> None:
    pdp = read_csv(RESULT_DIR / "关键变量偏依赖曲线.csv")
    aligned = read_csv(MID_DIR / "滞后配准建模数据.csv")
    variables = [
        "1#大烟道负压强度",
        "烧结机机速L1设定",
        "1#风箱平均温度",
        "9#风箱平均温度",
        "17#风箱负压强度",
        "18#风箱负压强度",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.0), constrained_layout=True)
    for ax, var in zip(axes.flat, variables):
        sub = pdp[pdp["变量"] == var].copy()
        if sub.empty:
            ax.axis("off")
            continue
        ax.plot(sub["取值"], sub["相对训练均值CO变化"], color=COLORS["blue"], lw=2)
        ax.axhline(0, color=COLORS["gray"], lw=1)
        feature = str(sub["模型特征"].iloc[0])
        if feature in aligned:
            train_values = aligned[aligned["数据集"] == "训练集"][feature].dropna()
            rug_y = np.full(len(train_values), ax.get_ylim()[0])
            ax.plot(train_values, rug_y, "|", color=COLORS["gray"], alpha=0.18, markersize=7)
            q10, q50, q90 = train_values.quantile([0.1, 0.5, 0.9])
            for q, label in [(q10, "P10"), (q50, "P50"), (q90, "P90")]:
                ax.axvline(q, color=COLORS["orange"], lw=0.8, alpha=0.55)
                ax.text(q, ax.get_ylim()[1], label, ha="center", va="bottom", fontsize=7, color=COLORS["orange"])
        ax.set_title(var)
        ax.set_xlabel("变量取值")
        ax.set_ylabel("相对 CO 变化")
    save(fig, "新颖_08_关键变量局部响应图谱.png")


def plot_high_low_spatial_fingerprint() -> None:
    clean = read_csv(MID_DIR / "清洗后数据.csv")
    pred = read_csv(RESULT_DIR / "CO预测结果_全样本.csv")
    df = clean.merge(pred[["序列", "数据集"]], on="序列", how="left")
    test = df[df["数据集"] == "测试集"].copy()
    q_high = test[TARGET].quantile(0.9)
    q_low = test[TARGET].quantile(0.1)
    high = test[test[TARGET] >= q_high]
    low = test[test[TARGET] <= q_low]
    train = df[df["数据集"] == "训练集"]
    boxes = np.arange(1, 19)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), constrained_layout=True)
    for ax, kind, cols, label in [
        (axes[0], "风箱平均温度", [f"{i}#风箱平均温度" for i in boxes], "温度 z-score"),
        (axes[1], "风箱负压强度", [f"{i}#风箱负压强度" for i in boxes], "负压强度 z-score"),
    ]:
        high_profile = ((high[cols].mean() - train[cols].mean()) / train[cols].std()).to_numpy()
        low_profile = ((low[cols].mean() - train[cols].mean()) / train[cols].std()).to_numpy()
        diff = high_profile - low_profile
        ax.plot(boxes, high_profile, color=COLORS["red"], lw=2, marker="o", label="高 CO 工况(P90+)")
        ax.plot(boxes, low_profile, color=COLORS["blue"], lw=2, marker="s", label="低 CO 工况(P10-)")
        ax.fill_between(boxes, high_profile, low_profile, color=COLORS["orange"], alpha=0.16, label="差异带")
        ax.bar(boxes, diff, color=np.where(diff >= 0, COLORS["red"], COLORS["blue"]), alpha=0.16, width=0.78)
        ax.axhline(0, color=COLORS["gray"], lw=1)
        ax.set_xticks(boxes)
        ax.set_xlabel("风箱编号")
        ax.set_ylabel(label)
        ax.set_title(kind + "空间指纹")
        ax.legend(frameon=False, loc="best")
    save(fig, "新颖_09_高低CO工况风箱空间指纹.png")


def write_index() -> None:
    files = sorted(p.name for p in OUT_DIR.glob("*.png"))
    descriptions = {
        "新颖_01_最高CO峰值局部透镜.png": "最高 CO 峰值附近的预测、残差与关键滞后变量联动。",
        "新颖_02_最大残差片段病例册.png": "测试集中多个最大误差片段的局部对照，适合讨论模型局限。",
        "新颖_03_峰值对齐动态指纹.png": "将多个 CO 峰值按峰顶对齐，观察峰值前后共性动态。",
        "新颖_04_局部峰值风箱状态热图.png": "最高峰局部窗口内 18 个风箱温度/负压状态的细节热图。",
        "新颖_05_局部峰值3D风箱状态曲面.png": "最高峰局部窗口的平滑 3D 风箱状态曲面。",
        "新颖_06_CO运行相图与预测困难区.png": "用 CO 水平与变化速率构成相图，定位预测困难区。",
        "新颖_07_时滞搜索精度复杂度帕累托.png": "展示时滞组合搜索中精度与复杂度的折中。",
        "新颖_08_关键变量局部响应图谱.png": "关键变量偏依赖曲线与训练分位位置，说明局部响应。",
        "新颖_09_高低CO工况风箱空间指纹.png": "高/低 CO 工况下风箱温度与负压强度空间剖面对比。",
    }
    lines = [
        "# 第二问局部新颖PNG图表索引",
        "",
        "本目录聚焦局部片段、峰值事件、误差片段、工况分层和时滞搜索机制。图均为 PNG，可直接用于论文正文或附录。",
        "",
    ]
    for name in files:
        lines.append(f"- `{name}`：{descriptions.get(name, '')}")
    (OUT_DIR / "图表索引.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_local_peak_lens()
    plot_residual_casebook()
    plot_peak_aligned_dynamics()
    plot_local_windbox_state_heatmap()
    plot_local_3d_state_surface()
    plot_phase_portrait()
    plot_lag_pareto_frontier()
    plot_response_atlas()
    plot_high_low_spatial_fingerprint()
    write_index()
    print(f"生成完成：{OUT_DIR}")


if __name__ == "__main__":
    main()
