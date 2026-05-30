# -*- coding: utf-8 -*-
"""Create paper-ready figures for question 2.

The figures in this script are built from the generated data under
``data/第2问``.  They are intentionally separate from the modelling script so
the manuscript figures can be tuned without changing model results.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "第2问" / "结果数据"
MID_DIR = ROOT / "data" / "第2问" / "中间数据"
OUT_DIR = ROOT / "figure" / "第2问" / "论文图"

OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 140,
            "savefig.dpi": 450,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.6,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def windbox_no(value: object) -> int | None:
    match = re.search(r"(\d+)#风箱", str(value))
    return int(match.group(1)) if match else None


def clean_model_name(name: str) -> str:
    mapping = {
        "过程变量模型": "过程变量",
        "非线性过程变量模型": "非线性过程",
        "基线1_CO_lag1持续性模型": "CO lag1",
        "基线2_历史CO_AR模型": "历史CO AR",
        "基线3_过程变量+历史CO监测型模型": "监测融合",
        "残差校正监测型模型_对比": "残差校正",
        "非线性监测型模型": "非线性监测",
    }
    return mapping.get(name, name)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def plot_3d_lag_terrain() -> None:
    lag = read_csv(MID_DIR / "最佳滞后配准.csv")
    selected = lag[lag["变量组"].isin(["风箱负压强度", "风箱平均温度"])].copy()
    selected["风箱编号"] = selected["风箱编号"].astype(int)
    selected["相关强度"] = selected["训练集相关系数"].abs()

    groups = ["风箱负压强度", "风箱平均温度"]
    group_y = {name: idx for idx, name in enumerate(groups)}

    fig = plt.figure(figsize=(7.1, 4.8))
    ax = fig.add_subplot(111, projection="3d")

    for group, color in zip(groups, [OKABE_ITO["blue"], OKABE_ITO["orange"]]):
        data = selected[selected["变量组"] == group].sort_values("风箱编号")
        xs = data["风箱编号"].to_numpy(float)
        ys = np.full_like(xs, group_y[group], dtype=float)
        zs = data["最佳滞后秒数"].to_numpy(float)
        corr = data["训练集相关系数"].to_numpy(float)

        ax.plot(xs, ys, zs, color=color, lw=2.2, label=group)
        sizes = 24 + 650 * np.abs(corr)
        colors = np.where(corr >= 0, OKABE_ITO["vermillion"], OKABE_ITO["blue"])
        ax.scatter(xs, ys, zs, s=sizes, c=colors, edgecolor="white", linewidth=0.5, depthshade=False)

        x_grid = np.vstack([xs, xs])
        y_grid = np.vstack([ys - 0.14, ys + 0.14])
        z_grid = np.vstack([zs, zs])
        ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.13, linewidth=0)

    ax.set_title("风箱位置-变量类型-最佳时滞三维地形", pad=12, fontweight="bold")
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("变量类型")
    ax.set_zlabel("最佳时滞 (s)")
    ax.set_xticks(range(1, 19, 2))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["负压", "温度"])
    ax.set_zlim(0, 930)
    ax.view_init(elev=23, azim=-55)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.96))

    txt = (
        "点大小表示 |训练集相关系数|；\n"
        "红色为正相关，蓝色为负相关。\n"
        "上游风箱达到 900 s 截断上限，\n"
        "下游风箱时滞明显缩短。"
    )
    ax.text2D(0.64, 0.86, txt, transform=ax.transAxes, fontsize=8, color="#37474F")
    save_figure(fig, "Q2_01_三维时滞影响地形")


def plot_spatial_fingerprint() -> None:
    sens = read_csv(DATA_DIR / "过程变量敏感性分析_过程变量模型.csv")
    sens = sens[sens["变量组"].isin(["风箱负压强度", "风箱平均温度"])].copy()
    sens["风箱编号"] = sens["原始字段"].map(windbox_no)
    sens = sens.dropna(subset=["风箱编号"])
    sens["风箱编号"] = sens["风箱编号"].astype(int)

    fig, ax = plt.subplots(figsize=(7.0, 3.45))
    y_map = {"风箱负压强度": 1, "风箱平均温度": 0}
    cmap = mpl.colormaps["RdBu_r"]
    value = sens["增加1个历史标准差后的平均CO变化"].astype(float)
    max_abs = max(1.0, np.nanmax(np.abs(value)))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    for _, row in sens.iterrows():
        x = row["风箱编号"]
        y = y_map[row["变量组"]]
        delta = float(row["增加1个历史标准差后的平均CO变化"])
        lag_s = float(row["最佳滞后秒数"])
        corr = abs(float(row["训练集相关系数"]))
        size = 120 + 1700 * corr
        ax.scatter(
            x,
            y,
            s=size,
            c=[cmap(norm(delta))],
            edgecolor="#263238",
            linewidth=0.45,
            zorder=3,
        )
        ax.vlines(x, y - 0.33, y - 0.33 + 0.66 * lag_s / 900, color="#90A4AE", lw=1.2, zorder=1)

    for x in [6.5, 12.5]:
        ax.axvline(x, color="#B0BEC5", ls="--", lw=0.9)
    ax.text(3.5, -0.58, "前段", ha="center", color="#546E7A")
    ax.text(9.5, -0.58, "中段", ha="center", color="#546E7A")
    ax.text(15.5, -0.58, "后段", ha="center", color="#546E7A")

    ax.set_title("风箱空间影响指纹：敏感性、相关强度与时滞", fontweight="bold")
    ax.set_xlabel("风箱编号")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["平均温度升高", "负压强度增强"])
    ax.set_xlim(0.4, 18.6)
    ax.set_ylim(-0.75, 1.55)
    ax.set_xticks(range(1, 19))
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("增加1个历史标准差后的平均 CO 变化 (mg/m^3)")

    ax.scatter([], [], s=160, c="white", edgecolor="#263238", label="低相关强度")
    ax.scatter([], [], s=420, c="white", edgecolor="#263238", label="高相关强度")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=2)
    save_figure(fig, "Q2_02_风箱空间影响指纹")


def plot_model_performance_bubble() -> None:
    metrics = read_csv(DATA_DIR / "模型评价指标.csv")
    metrics = metrics[metrics["数据集"].isin(["验证集", "测试集"])].copy()
    order = [
        "过程变量模型",
        "非线性过程变量模型",
        "基线1_CO_lag1持续性模型",
        "基线2_历史CO_AR模型",
        "基线3_过程变量+历史CO监测型模型",
        "残差校正监测型模型_对比",
        "非线性监测型模型",
    ]
    metrics["模型短名"] = metrics["模型"].map(clean_model_name)
    metrics["x"] = metrics["模型"].map({m: i for i, m in enumerate(order)})
    metrics["y"] = metrics["数据集"].map({"验证集": 1, "测试集": 0})

    fig, ax = plt.subplots(figsize=(7.15, 3.8))
    norm = mpl.colors.Normalize(vmin=metrics["R2"].min(), vmax=metrics["R2"].max())
    cmap = mpl.colormaps["viridis"]
    sizes = 30 + (metrics["RMSE"] / metrics["RMSE"].max()) ** 2 * 1700

    scatter = ax.scatter(
        metrics["x"],
        metrics["y"],
        s=sizes,
        c=metrics["R2"],
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    for _, row in metrics.iterrows():
        ax.text(row["x"], row["y"], f"{row['RMSE']:.0f}", ha="center", va="center", fontsize=7, color="white")

    ax.set_title("CO 实时预测模型性能矩阵", fontweight="bold")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([clean_model_name(m) for m in order], rotation=25, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["测试集", "验证集"])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(-0.6, 1.6)
    ax.grid(axis="x", alpha=0.14)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("R2")
    ax.text(
        0.02,
        0.96,
        "气泡面积表示 RMSE，数字为 RMSE；颜色越亮 R2 越高",
        transform=ax.transAxes,
        va="top",
        color="#455A64",
    )
    save_figure(fig, "Q2_03_模型性能气泡矩阵")


def plot_prediction_diagnostic() -> None:
    pred = read_csv(DATA_DIR / "CO预测结果_全样本.csv")
    test = pred[pred["数据集"] == "测试集"].copy()
    actual_col = "烧结大烟道外排CO浓度"
    model_col = "监测型模型_预测CO浓度"
    residual_col = "监测型模型_残差"
    threshold = float(read_csv(DATA_DIR / "模型评价指标.csv")["高CO阈值_训练90分位"].dropna().iloc[0])

    fig = plt.figure(figsize=(7.15, 4.75))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.35, 1], height_ratios=[1, 1], wspace=0.32, hspace=0.38)
    ax_ts = fig.add_subplot(gs[:, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, 1])

    ax_ts.plot(test["时间_s"] / 60, test[actual_col], color=OKABE_ITO["black"], lw=1.35, label="实测")
    ax_ts.plot(test["时间_s"] / 60, test[model_col], color=OKABE_ITO["blue"], lw=1.35, label="监测型预测")
    ax_ts.axhline(threshold, color=OKABE_ITO["vermillion"], lw=1.0, ls="--", label="训练90分位")
    ax_ts.fill_between(
        test["时间_s"] / 60,
        test[actual_col],
        test[model_col],
        color=OKABE_ITO["sky"],
        alpha=0.18,
        linewidth=0,
    )
    ax_ts.set_title("测试集 CO 时序预测与误差带", fontweight="bold")
    ax_ts.set_xlabel("时间 (min)")
    ax_ts.set_ylabel("CO浓度 (mg/m^3)")
    ax_ts.legend(frameon=False, loc="upper left")
    ax_ts.spines[["top", "right"]].set_visible(False)
    ax_ts.grid(alpha=0.18)
    panel_label(ax_ts, "A")

    hb = ax_scatter.hexbin(
        test[actual_col],
        test[residual_col],
        gridsize=22,
        cmap="cividis",
        mincnt=1,
        linewidths=0,
    )
    ax_scatter.axhline(0, color="#455A64", lw=0.9)
    ax_scatter.axvline(threshold, color=OKABE_ITO["vermillion"], lw=0.9, ls="--")
    ax_scatter.set_title("残差-浓度密度", fontweight="bold")
    ax_scatter.set_xlabel("实测 CO")
    ax_scatter.set_ylabel("预测残差")
    ax_scatter.spines[["top", "right"]].set_visible(False)
    fig.colorbar(hb, ax=ax_scatter, fraction=0.047, pad=0.02, label="样本数")
    panel_label(ax_scatter, "B")

    residual = test[residual_col].to_numpy(float)
    bins = np.linspace(np.nanpercentile(residual, 1), np.nanpercentile(residual, 99), 26)
    ax_hist.hist(residual, bins=bins, density=True, color=OKABE_ITO["green"], alpha=0.78, edgecolor="white")
    ax_hist.axvline(0, color="#455A64", lw=0.9)
    ax_hist.axvline(np.mean(residual), color=OKABE_ITO["vermillion"], lw=1.0, ls="--", label=f"均值 {np.mean(residual):.1f}")
    ax_hist.set_title("残差分布", fontweight="bold")
    ax_hist.set_xlabel("预测残差 (mg/m^3)")
    ax_hist.set_ylabel("密度")
    ax_hist.legend(frameon=False)
    ax_hist.spines[["top", "right"]].set_visible(False)
    panel_label(ax_hist, "C")
    save_figure(fig, "Q2_04_预测残差诊断")


def plot_lag_gain_and_search_ridges() -> None:
    valid = read_csv(DATA_DIR / "时滞合理性验证.csv")
    search = read_csv(MID_DIR / "滞后搜索明细.csv")
    fields = ["1#风箱负压强度", "7#风箱负压强度", "13#风箱负压强度", "17#风箱负压强度"]
    colors = [OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["orange"], OKABE_ITO["purple"]]

    fig = plt.figure(figsize=(7.15, 4.35))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1.25], wspace=0.32)
    ax_gain = fig.add_subplot(gs[0, 0])
    ax_ridge = fig.add_subplot(gs[0, 1])

    lag0_scene = "无时滞模型_lag0"
    lagged_candidates = [
        "物理初始时滞截断_主模型",
        "物理时滞配准模型",
        "时滞配准模型",
    ]
    lagged_scene = next(scene for scene in lagged_candidates if scene in set(valid["实验场景"]))

    pivot = (
        valid[valid["模型类型"].isin(["过程变量模型", "监测型模型"])]
        .pivot(index="模型类型", columns="实验场景", values="RMSE")
        .reindex(["过程变量模型", "监测型模型"])
    )
    labels = ["过程变量", "监测型"]
    x = np.arange(len(labels))
    width = 0.34
    ax_gain.bar(x - width / 2, pivot[lag0_scene], width, color="#B0BEC5", label="无时滞")
    ax_gain.bar(x + width / 2, pivot[lagged_scene], width, color=OKABE_ITO["blue"], label="时滞配准")
    for i, label in enumerate(pivot.index):
        lag0 = float(pivot.loc[label, lag0_scene])
        lagged = float(pivot.loc[label, lagged_scene])
        improve = (lag0 - lagged) / lag0 * 100
        ax_gain.text(i, max(lag0, lagged) + 13, f"降低 {improve:.1f}%", ha="center", fontsize=8)
    ax_gain.set_title("时滞配准带来的 RMSE 收益", fontweight="bold")
    ax_gain.set_ylabel("测试 RMSE")
    ax_gain.set_xticks(x)
    ax_gain.set_xticklabels(labels)
    ax_gain.legend(frameon=False)
    ax_gain.spines[["top", "right"]].set_visible(False)
    ax_gain.grid(axis="y", alpha=0.18)
    panel_label(ax_gain, "A")

    offsets = np.arange(len(fields))[::-1] * 0.16
    for field, color, offset in zip(fields, colors, offsets):
        data = search[search["字段名"] == field].sort_values("滞后秒数")
        xlag = data["滞后秒数"].to_numpy(float)
        y = data["训练集绝对相关系数"].to_numpy(float)
        if len(y) == 0:
            continue
        y_scaled = y / max(y.max(), 1e-9) * 0.13 + offset
        ax_ridge.fill_between(xlag, offset, y_scaled, color=color, alpha=0.35)
        ax_ridge.plot(xlag, y_scaled, color=color, lw=1.4)
        best = data.iloc[int(np.argmax(y))]
        ax_ridge.scatter(best["滞后秒数"], y_scaled[np.argmax(y)], s=22, color=color, edgecolor="white", zorder=3)
        ax_ridge.text(xlag.max() + 14, offset + 0.035, field.replace("#风箱负压强度", "#负压"), va="center", fontsize=8)

    ax_ridge.set_title("典型风箱滞后搜索相关性山脊", fontweight="bold")
    ax_ridge.set_xlabel("滞后时间 (s)")
    ax_ridge.set_yticks([])
    ax_ridge.set_xlim(-10, max(search[search["字段名"].isin(fields)]["滞后秒数"].max(), 900) + 95)
    ax_ridge.set_ylim(-0.05, offsets.max() + 0.2)
    ax_ridge.spines[["top", "right", "left"]].set_visible(False)
    ax_ridge.grid(axis="x", alpha=0.16)
    ax_ridge.text(0.02, 0.02, "曲线高度为归一化 |相关系数|", transform=ax_ridge.transAxes, color="#546E7A")
    panel_label(ax_ridge, "B")

    save_figure(fig, "Q2_05_时滞配准收益与搜索山脊")


def write_note() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    note = """# 第2问论文图说明

本目录图件依据 `data/第2问/结果数据` 与 `data/第2问/中间数据` 生成，服务于第2问“影响规律 + 实时预测 + 时滞配准”的论文叙事。

## 图件建议

- `Q2_01_三维时滞影响地形`：3D 图，说明风箱位置差异导致最佳时滞由上游到下游递减，并用点大小/颜色表达相关强度和方向。
- `Q2_02_风箱空间影响指纹`：把负压、温度的敏感性、相关强度、时滞压缩到同一张空间图，适合放在“影响规律分析”。
- `Q2_03_模型性能气泡矩阵`：比较验证集/测试集不同模型的 RMSE 和 R2，适合放在“实时预测模型评价”。
- `Q2_04_预测残差诊断`：展示测试集预测时序、残差-浓度关系和残差分布，适合支撑模型可靠性讨论。
- `Q2_05_时滞配准收益与搜索山脊`：直接回答题目提示中的“数据配准会有更好预测精度”，并展示典型风箱的滞后搜索过程。

每张图均保存为 `png`、`pdf`、`svg` 三种格式；论文排版优先使用 `pdf` 或 `svg`，预览和汇报优先使用 `png`。
"""
    (OUT_DIR / "图表说明.md").write_text(note, encoding="utf-8")


def main() -> None:
    setup_style()
    plot_3d_lag_terrain()
    plot_spatial_fingerprint()
    plot_model_performance_bubble()
    plot_prediction_diagnostic()
    plot_lag_gain_and_search_ridges()
    write_note()
    print(f"Saved paper figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
