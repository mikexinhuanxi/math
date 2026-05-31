# -*- coding: utf-8 -*-
"""Publication-quality figures for Question 2.

This script reads the saved result tables from ``data/第2问`` and creates a
separate figure set under ``figure/第2问/论文增强图``. It deliberately does not
retrain models or overwrite the existing "基础图" outputs.
"""

from __future__ import annotations

import math
import logging
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "第2问" / "结果数据"
MID_DIR = ROOT / "data" / "第2问" / "中间数据"
OUT_DIR = ROOT / "figure" / "第2问" / "论文增强图"

PRED_PATH = DATA_DIR / "CO预测结果_全样本.csv"
METRIC_PATH = DATA_DIR / "模型评价指标.csv"
BASELINE_PATH = DATA_DIR / "基线模型比较.csv"
NONLINEAR_PATH = DATA_DIR / "非线性监测模型比较.csv"
LAG_VALIDATE_PATH = DATA_DIR / "时滞合理性验证.csv"
IMPORTANCE_PATH = DATA_DIR / "变量影响强度_过程变量模型.csv"
SENSITIVITY_PATH = DATA_DIR / "过程变量敏感性分析_过程变量模型.csv"
ABLATION_PATH = DATA_DIR / "变量组消融分析.csv"
PDP_PATH = DATA_DIR / "关键变量偏依赖曲线.csv"
BEST_LAG_PATH = MID_DIR / "最佳滞后配准.csv"

TARGET_COL = "烧结大烟道外排CO浓度"
BASELINE3 = "基线3_过程变量+历史CO监测型模型"
VALUE_COL = "增加1个历史标准差后的平均CO变化"


PALETTE = {
    "ink": "#263238",
    "muted": "#607D8B",
    "grid": "#E7EDF2",
    "teal": "#009688",
    "deep_teal": "#00695C",
    "coral": "#EF6C62",
    "amber": "#F2B134",
    "blue": "#2F80ED",
    "violet": "#7E57C2",
    "green": "#43A047",
    "red": "#D64B4B",
    "slate": "#546E7A",
    "purple": "#8E5AAE",
}

MODEL_COLORS = {
    "过程变量": "#7E57C2",
    "基线1": "#90A4AE",
    "基线2": "#2F80ED",
    "基线3": "#009688",
    "残差校正": "#F2B134",
    "非线性": "#EF6C62",
    "非线性监测": "#EF6C62",
}


def setup_style() -> None:
    logging.getLogger("fontTools").setLevel(logging.ERROR)
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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PALETTE["muted"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 140,
            "savefig.dpi": 360,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def ensure_out() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: mpl.figure.Figure, stem: str) -> list[Path]:
    paths: list[Path] = []
    for suffix in ["png", "svg", "pdf"]:
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def clean_name(name: str) -> str:
    name = str(name)
    repl = {
        "过程变量模型_梯度提升树": "过程变量模型",
        "过程变量模型": "过程变量",
        "线性过程变量模型_对比": "线性过程",
        "基线1_CO_lag1持续性模型": "基线1\nCO-lag1",
        "基线2_历史CO_AR模型": "基线2\nAR",
        "基线3_过程变量+历史CO监测型模型": "基线3\n监测型",
        "残差校正监测型模型_对比": "残差校正",
        "非线性监测型模型": "非线性\n集合",
        "非线性监测_LightGBM": "LightGBM",
        "非线性监测_XGBoost": "XGBoost",
        "非线性监测_梯度提升树": "梯度提升树",
        "非线性监测_随机森林": "随机森林",
    }
    return repl.get(name, name.replace("_", "\n"))


def family_color(model: str) -> str:
    model = str(model)
    if "基线3" in model or "监测型模型" == model:
        return MODEL_COLORS["基线3"]
    if "基线2" in model:
        return MODEL_COLORS["基线2"]
    if "基线1" in model:
        return MODEL_COLORS["基线1"]
    if "过程变量" in model and "历史CO" not in model:
        return MODEL_COLORS["过程变量"]
    if "残差" in model:
        return MODEL_COLORS["残差校正"]
    if "非线性" in model:
        return MODEL_COLORS["非线性"]
    return PALETTE["slate"]


def beautify(ax: mpl.axes.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B0BEC5")
    ax.spines["bottom"].set_color("#B0BEC5")
    ax.grid(True, axis=grid_axis, color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
        va="top",
        ha="left",
    )


def add_suptitle(fig: mpl.figure.Figure, title: str, subtitle: str | None = None) -> None:
    fig.suptitle(title, fontsize=17, fontweight="bold", x=0.5, y=0.985)
    if subtitle:
        fig.text(0.5, 0.948, subtitle, ha="center", va="top", fontsize=9.5, color=PALETTE["muted"])


def fmt_num(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.{digits}f}"


def plot_model_dashboard(metrics: pd.DataFrame, baseline: pd.DataFrame) -> list[Path]:
    test_metrics = metrics[metrics["数据集"] == "测试集"].copy()
    selected = [
        "过程变量模型",
        "基线1_CO_lag1持续性模型",
        "基线2_历史CO_AR模型",
        BASELINE3,
        "残差校正监测型模型_对比",
        "非线性监测_梯度提升树",
    ]
    perf = test_metrics[test_metrics["模型"].isin(selected)].copy()
    perf["短名"] = perf["模型"].map(clean_name)
    perf = perf.set_index("模型").loc[selected].reset_index()

    fig = plt.figure(figsize=(13.6, 8.6), constrained_layout=False)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.26)
    add_suptitle(
        fig,
        "第2问核心模型性能总览",
        "测试集指标对比；绿色高亮为推荐的“过程变量 + 历史 CO”监测型模型",
    )

    ax1 = fig.add_subplot(gs[0, 0])
    order = perf.sort_values("RMSE", ascending=False)
    colors = [family_color(m) for m in order["模型"]]
    bars = ax1.barh(order["短名"], order["RMSE"], color=colors, edgecolor="white", linewidth=0.8)
    ax1.set_xlabel("测试 RMSE")
    ax1.set_title("误差规模：RMSE 越低越好", loc="left")
    beautify(ax1)
    for bar, value in zip(bars, order["RMSE"]):
        ax1.text(value + 8, bar.get_y() + bar.get_height() / 2, fmt_num(value, 1), va="center", fontsize=8.5)
    panel_label(ax1, "A")

    ax2 = fig.add_subplot(gs[0, 1])
    order_r2 = perf.sort_values("R2", ascending=True)
    y = np.arange(len(order_r2))
    ax2.hlines(y, xmin=0.0, xmax=order_r2["R2"], color="#CFD8DC", linewidth=5, zorder=1)
    ax2.scatter(order_r2["R2"], y, s=125, color=[family_color(m) for m in order_r2["模型"]], edgecolor="white", linewidth=1.0, zorder=2)
    ax2.set_yticks(y)
    ax2.set_yticklabels(order_r2["短名"])
    ax2.set_xlim(max(0, order_r2["R2"].min() - 0.08), min(1, order_r2["R2"].max() + 0.04))
    ax2.set_xlabel("测试 R²")
    ax2.set_title("解释度：R² 越高越好", loc="left")
    beautify(ax2, "x")
    for xi, yi in zip(order_r2["R2"], y):
        ax2.text(xi + 0.006, yi, f"{xi:.4f}", va="center", fontsize=8.5)
    panel_label(ax2, "B")

    ax3 = fig.add_subplot(gs[1, 0])
    base3 = baseline[baseline["模型"].isin(
        [
            "基线1_CO_lag1持续性模型",
            "基线2_历史CO_AR模型",
            BASELINE3,
        ]
    )].copy()
    data_order = ["训练集", "验证集", "测试集"]
    marker_map = {
        "基线1_CO_lag1持续性模型": "o",
        "基线2_历史CO_AR模型": "s",
        BASELINE3: "D",
    }
    for model, group in base3.groupby("模型", sort=False):
        group = group.set_index("数据集").loc[data_order].reset_index()
        ax3.plot(
            data_order,
            group["RMSE"],
            marker=marker_map.get(model, "o"),
            linewidth=2.3,
            markersize=7,
            color=family_color(model),
            label=clean_name(model).replace("\n", " "),
        )
    ax3.set_ylabel("RMSE")
    ax3.set_title("训练-验证-测试稳定性", loc="left")
    beautify(ax3)
    ax3.legend(ncol=1, frameon=False, loc="upper left")
    panel_label(ax3, "C")

    ax4 = fig.add_subplot(gs[1, 1])
    high = perf.sort_values("高CO_RMSE", ascending=False)
    bars = ax4.barh(high["短名"], high["高CO_RMSE"], color=[family_color(m) for m in high["模型"]], alpha=0.92)
    ax4.set_xlabel("高 CO 样本 RMSE")
    ax4.set_title("高浓度区间表现", loc="left")
    beautify(ax4)
    for bar, value in zip(bars, high["高CO_RMSE"]):
        ax4.text(value + 8, bar.get_y() + bar.get_height() / 2, fmt_num(value, 1), va="center", fontsize=8.5)
    panel_label(ax4, "D")

    return save_fig(fig, "图2E-1_核心模型性能仪表板")


def plot_prediction_series(pred: pd.DataFrame, metrics: pd.DataFrame) -> list[Path]:
    test = pred[pred["数据集"] == "测试集"].copy()
    test["相对时间_min"] = (test["时间_s"] - test["时间_s"].min()) / 60.0

    threshold = float(metrics.loc[(metrics["数据集"] == "测试集") & (metrics["模型"] == BASELINE3), "高CO阈值_训练90分位"].iloc[0])
    cols = [
        ("实际 CO", TARGET_COL, PALETTE["ink"], 2.4),
        ("基线1", "基线1_CO_lag1持续性模型_预测CO浓度", "#90A4AE", 1.6),
        ("基线2", "基线2_历史CO_AR模型_预测CO浓度", PALETTE["blue"], 1.7),
        ("基线3 监测型", "监测型模型_预测CO浓度", PALETTE["teal"], 2.2),
        ("非线性梯度提升树", "非线性监测_梯度提升树_预测CO浓度", PALETTE["coral"], 1.4),
    ]

    fig = plt.figure(figsize=(14.4, 8.1))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.3, 0.72, 0.85], hspace=0.18, figure=fig)
    add_suptitle(fig, "测试集 CO 预测时序与残差诊断", "同一测试区间内比较惯性基线、AR、推荐监测型模型与非线性树模型")

    ax1 = fig.add_subplot(gs[0, 0])
    for label, col, color, lw in cols:
        ax1.plot(test["相对时间_min"], test[col], color=color, linewidth=lw, label=label, alpha=0.95)
    ax1.axhline(threshold, color=PALETTE["amber"], linestyle="--", linewidth=1.2, label="训练90分位阈值")
    ax1.fill_between(test["相对时间_min"], threshold, test[TARGET_COL].max() * 1.03, color=PALETTE["amber"], alpha=0.08)
    ax1.set_ylabel("CO 浓度")
    ax1.set_title("A  实测与预测曲线", loc="left")
    beautify(ax1)
    ax1.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax1.tick_params(labelbottom=False)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    resid = test["监测型模型_残差"].to_numpy(dtype=float)
    x = test["相对时间_min"].to_numpy(dtype=float)
    ax2.axhline(0, color=PALETTE["ink"], linewidth=1.0)
    ax2.fill_between(x, 0, resid, where=resid >= 0, color=PALETTE["red"], alpha=0.34, interpolate=True, label="预测偏高")
    ax2.fill_between(x, 0, resid, where=resid < 0, color=PALETTE["blue"], alpha=0.32, interpolate=True, label="预测偏低")
    ax2.set_ylabel("基线3残差")
    ax2.set_title("B  推荐模型残差带", loc="left")
    beautify(ax2)
    ax2.tick_params(labelbottom=False)

    ax3 = fig.add_subplot(gs[2, 0])
    bins = np.linspace(np.percentile(resid, 1), np.percentile(resid, 99), 35)
    ax3.hist(resid, bins=bins, color=PALETTE["teal"], alpha=0.82, density=True, edgecolor="white", linewidth=0.6)
    ax3.axvline(0, color=PALETTE["ink"], linewidth=1.0)
    ax3.axvline(np.mean(resid), color=PALETTE["coral"], linewidth=1.5, linestyle="--", label=f"均值 {np.mean(resid):.1f}")
    ax3.set_xlabel("残差 = 预测值 - 实测值")
    ax3.set_ylabel("密度")
    ax3.set_title("C  残差分布", loc="left")
    beautify(ax3)
    ax3.legend(frameon=False)

    return save_fig(fig, "图2E-2_测试集CO预测时序残差")


def plot_prediction_hexbin(pred: pd.DataFrame, metrics: pd.DataFrame) -> list[Path]:
    test = pred[pred["数据集"] == "测试集"].copy()
    panels = [
        ("基线1 CO-lag1", "基线1_CO_lag1持续性模型_预测CO浓度", "基线1_CO_lag1持续性模型"),
        ("基线3 监测型", "监测型模型_预测CO浓度", BASELINE3),
        ("非线性梯度提升树", "非线性监测_梯度提升树_预测CO浓度", "非线性监测_梯度提升树"),
        ("过程变量模型", "过程变量模型_预测CO浓度", "过程变量模型"),
    ]
    actual = test[TARGET_COL].astype(float)
    lo = float(min(actual.min(), *(test[col].min() for _, col, _ in panels)))
    hi = float(max(actual.max(), *(test[col].max() for _, col, _ in panels)))
    pad = (hi - lo) * 0.05
    lo -= pad
    hi += pad

    cmap = LinearSegmentedColormap.from_list("paper_density", ["#F7FBFF", "#B3E5FC", "#009688", "#F2B134", "#D64B4B"])

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 10.1), sharex=True, sharey=True)
    add_suptitle(fig, "测试集预测一致性密度散点", "四类模型在同一实测-预测坐标中的集中程度与系统偏差")

    for ax, (title, col, metric_model), label in zip(axes.ravel(), panels, ["A", "B", "C", "D"]):
        pred_y = test[col].astype(float)
        hb = ax.hexbin(actual, pred_y, gridsize=32, cmap=cmap, mincnt=1, linewidths=0.0)
        ax.plot([lo, hi], [lo, hi], color=PALETTE["ink"], linestyle="--", linewidth=1.0)
        coef = np.polyfit(actual, pred_y, 1)
        xs = np.linspace(lo, hi, 80)
        ax.plot(xs, coef[0] * xs + coef[1], color=PALETTE["coral"], linewidth=1.5)
        row = metrics[(metrics["数据集"] == "测试集") & (metrics["模型"] == metric_model)]
        if not row.empty:
            row = row.iloc[0]
            note = f"RMSE={row['RMSE']:.1f}\nR²={row['R2']:.4f}\nMAPE={row['MAPE_%']:.2f}%"
            ax.text(
                0.04,
                0.96,
                note,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(facecolor="white", edgecolor="#CFD8DC", boxstyle="round,pad=0.35", alpha=0.88),
            )
        ax.set_title(f"{label}  {title}", loc="left")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        beautify(ax)

    axes[1, 0].set_xlabel("实测 CO")
    axes[1, 1].set_xlabel("实测 CO")
    axes[0, 0].set_ylabel("预测 CO")
    axes[1, 0].set_ylabel("预测 CO")
    cbar = fig.colorbar(hb, ax=axes.ravel().tolist(), shrink=0.72, pad=0.018)
    cbar.set_label("样本密度")

    return save_fig(fig, "图2E-3_预测一致性密度散点")


def plot_lag_gain_3d(lag_val: pd.DataFrame) -> list[Path]:
    data = lag_val.copy()
    scenario_order = [
        "无时滞模型_lag0",
        "物理初始时滞截断_对比模型",
        "分组时滞组合优化_主模型",
        "单变量局部搜索_对比模型",
    ]
    scenario_labels = ["无时滞", "物理截断", "分组优化", "局部搜索"]
    type_order = ["过程变量模型", "监测型模型", "非线性监测型模型"]
    type_labels = ["过程变量", "监测型", "非线性监测"]
    type_colors = [PALETTE["violet"], PALETTE["teal"], PALETTE["coral"]]

    fig = plt.figure(figsize=(13.5, 7.6))
    add_suptitle(fig, "时滞配准收益的三维比较", "三类模型在四种时滞方案下的测试误差、解释度和平均滞后")

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    dx = dy = 0.56
    for yi, model_type in enumerate(type_order):
        sub = data[data["模型类型"] == model_type].set_index("实验场景")
        for xi, scenario in enumerate(scenario_order):
            row = sub.loc[scenario]
            z = float(row["RMSE"])
            ax.bar3d(xi - dx / 2, yi - dy / 2, 0, dx, dy, z, color=type_colors[yi], alpha=0.82, shade=True, edgecolor="white", linewidth=0.35)
            ax.text(xi, yi, z + 18, f"{z:.0f}", ha="center", va="bottom", fontsize=7.5, zdir=None)
    ax.set_xticks(range(len(scenario_labels)))
    ax.set_xticklabels(scenario_labels, rotation=18, ha="right")
    ax.set_yticks(range(len(type_labels)))
    ax.set_yticklabels(type_labels)
    ax.set_zlabel("测试 RMSE")
    ax.set_title("A  时滞方案 × 模型类型 × RMSE", loc="left", pad=12)
    ax.view_init(elev=23, azim=-52)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))

    ax2 = fig.add_subplot(1, 2, 2)
    mon = data[data["模型类型"] == "监测型模型"].set_index("实验场景").loc[scenario_order].reset_index()
    x = np.arange(len(mon))
    ax2.bar(x, mon["RMSE相对无时滞降低率_%"], color=[PALETTE["slate"], PALETTE["blue"], PALETTE["teal"], PALETTE["amber"]], width=0.56)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_labels)
    ax2.set_ylabel("RMSE 相对无时滞降低率 (%)")
    ax2.set_title("B  监测型模型收益与平均滞后", loc="left")
    beautify(ax2)
    ax2b = ax2.twinx()
    ax2b.plot(x, mon["平均最佳滞后_s"], color=PALETTE["ink"], marker="D", linewidth=2.0, label="平均最佳滞后")
    ax2b.set_ylabel("平均最佳滞后 (s)")
    ax2b.spines["top"].set_visible(False)
    ax2b.spines["right"].set_color("#B0BEC5")
    for xi, gain, lag in zip(x, mon["RMSE相对无时滞降低率_%"], mon["平均最佳滞后_s"]):
        ax2.text(xi, gain + 1.1, f"{gain:.1f}%", ha="center", fontsize=8.5)
        lag_offset = -18 if lag > 180 else 8
        ax2b.text(
            xi + 0.06,
            lag + lag_offset,
            f"{lag:.0f}s",
            fontsize=8.5,
            color=PALETTE["ink"],
            va="top" if lag_offset < 0 else "bottom",
        )
    ax2b.grid(False)

    return save_fig(fig, "图2E-4_时滞配准三维收益")


def windbox_no(value: object) -> int | None:
    text = str(value)
    if "#风箱" not in text:
        return None
    try:
        return int(text.split("#", 1)[0])
    except ValueError:
        return None


def plot_windbox_lag_3d(best_lag: pd.DataFrame) -> list[Path]:
    wb = best_lag[best_lag["变量组"].isin(["风箱负压强度", "风箱平均温度"])].copy()
    wb["风箱编号"] = wb["字段名"].map(windbox_no)
    wb = wb.dropna(subset=["风箱编号"]).copy()
    wb["风箱编号"] = wb["风箱编号"].astype(int)
    wb = wb.sort_values(["变量组", "风箱编号"])

    fig = plt.figure(figsize=(13.2, 7.7))
    add_suptitle(fig, "风箱空间位置、最佳时滞与相关方向", "每个点对应一个风箱变量；高度为最终入模滞后，颜色为训练集相关系数")

    ax = fig.add_subplot(1, 1, 1, projection="3d")
    norm = TwoSlopeNorm(vmin=-0.24, vcenter=0.0, vmax=0.24)
    cmap = LinearSegmentedColormap.from_list("corr_balance", [PALETTE["blue"], "#F7F7F7", PALETTE["red"]])
    group_to_y = {"风箱负压强度": 0, "风箱平均温度": 1}
    group_colors = {"风箱负压强度": PALETTE["blue"], "风箱平均温度": PALETTE["amber"]}

    for group, sub in wb.groupby("变量组"):
        sub = sub.sort_values("风箱编号")
        x = sub["风箱编号"].to_numpy(dtype=float)
        y = np.full(len(sub), group_to_y[group], dtype=float)
        z = sub["最佳滞后秒数"].to_numpy(dtype=float)
        corr = sub["训练集相关系数"].to_numpy(dtype=float)
        ax.plot(x, y, z, color=group_colors[group], linewidth=2.1, alpha=0.68)
        ax.scatter(x, y, z, c=corr, cmap=cmap, norm=norm, s=75 + np.abs(corr) * 380, edgecolor="white", linewidth=0.8, depthshade=False)
        for xi, yi, zi in zip(x, y, z):
            ax.plot([xi, xi], [yi, yi], [0, zi], color="#B0BEC5", linewidth=0.75, alpha=0.62)

    for boundary in [6.5, 12.5]:
        ax.plot([boundary, boundary], [-0.25, 1.25], [0, 0], color=PALETTE["ink"], linestyle="--", linewidth=0.9, alpha=0.55)
    ax.text(3.5, -0.42, 0, "上游", ha="center", fontsize=9)
    ax.text(9.5, -0.42, 0, "中游", ha="center", fontsize=9)
    ax.text(15.5, -0.42, 0, "下游", ha="center", fontsize=9)

    ax.set_xlabel("风箱编号")
    ax.set_ylabel("变量类型")
    ax.set_zlabel("最佳滞后 (s)")
    ax.set_xticks(range(1, 19))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["负压强度", "平均温度"])
    ax.set_zlim(0, max(320, float(wb["最佳滞后秒数"].max()) * 1.08))
    ax.view_init(elev=25, azim=-61)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.06)
    cbar.set_label("训练集相关系数")

    return save_fig(fig, "图2E-5_风箱空间时滞三维剖面")


def plot_importance_sensitivity(importance: pd.DataFrame, sensitivity: pd.DataFrame) -> list[Path]:
    top = importance.sort_values("综合影响强度", ascending=False).head(12).copy()
    sens = sensitivity[["原始字段", "变量组", "最佳滞后秒数", VALUE_COL]].copy()
    merged = top.merge(sens, on="原始字段", how="left", suffixes=("", "_sens"))
    merged[VALUE_COL] = merged[VALUE_COL].fillna(0.0)
    merged["最佳滞后秒数"] = merged["最佳滞后秒数"].fillna(merged["最佳滞后秒数_sens"] if "最佳滞后秒数_sens" in merged else 0)
    merged = merged.iloc[::-1].reset_index(drop=True)

    fig = plt.figure(figsize=(13.2, 8.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.08, 1.22], wspace=0.18, figure=fig)
    add_suptitle(fig, "过程变量影响强度与敏感性双编码", "左侧为置换重要性，右侧为单变量增加 1 个历史标准差后的平均 CO 响应")

    labels = [textwrap.fill(v.replace("__", "\n"), width=14) for v in merged["原始字段"]]
    y = np.arange(len(merged))

    ax1 = fig.add_subplot(gs[0, 0])
    colors = [family_color("过程变量") if "大烟道" in v else (PALETTE["amber"] if "温度" in v else PALETTE["blue"]) for v in merged["原始字段"]]
    ax1.barh(y, merged["综合影响强度"], color=colors, edgecolor="white", linewidth=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("综合影响强度")
    ax1.set_title("A  置换重要性排序", loc="left")
    beautify(ax1)
    for yi, val, lag in zip(y, merged["综合影响强度"], merged["最佳滞后秒数"]):
        ax1.text(val + 0.004, yi, f"{val:.3f} | {lag:.0f}s", va="center", fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    response = merged[VALUE_COL].to_numpy(dtype=float)
    xmin = min(-45, float(np.nanmin(response)) * 1.22)
    xmax = max(60, float(np.nanmax(response)) * 1.22)
    ax2.axvline(0, color=PALETTE["ink"], linewidth=1.1)
    for yi, val, lag, corr in zip(y, response, merged["最佳滞后秒数"], merged["训练集相关系数"]):
        color = PALETTE["red"] if val >= 0 else PALETTE["blue"]
        ax2.hlines(yi, 0, val, color=color, linewidth=3.0, alpha=0.74)
        ax2.scatter(val, yi, s=58 + float(lag) * 0.28, color=color, edgecolor="white", linewidth=0.9, zorder=3)
        if val >= 0:
            text_x = val + 2.5
            text_ha = "left"
            if text_x > xmax - 2:
                text_x = val - 2.5
                text_ha = "right"
        else:
            text_x = val + 2.5
            text_ha = "left"
        ax2.text(text_x, yi, f"{val:+.1f}", ha=text_ha, va="center", fontsize=8.3)
        ax2.text(xmin + 1, yi, f"r={corr:+.2f}", ha="left", va="center", color=PALETTE["muted"], fontsize=7.8)
    ax2.set_xlim(xmin, xmax)
    ax2.tick_params(axis="y", labelleft=False)
    ax2.set_xlabel("平均 CO 响应")
    ax2.set_title("B  扰动响应方向与幅度", loc="left")
    beautify(ax2, "x")

    return save_fig(fig, "图2E-6_变量重要性敏感性双编码")


def plot_windbox_effect_heatmap(sensitivity: pd.DataFrame) -> list[Path]:
    data = sensitivity[sensitivity["变量组"].isin(["风箱负压强度", "风箱平均温度"])].copy()
    data["风箱编号"] = data["原始字段"].map(windbox_no)
    data = data.dropna(subset=["风箱编号"]).copy()
    data["风箱编号"] = data["风箱编号"].astype(int)
    response = data.pivot_table(index="变量组", columns="风箱编号", values=VALUE_COL, aggfunc="mean").reindex(["风箱负压强度", "风箱平均温度"])
    lag = data.pivot_table(index="变量组", columns="风箱编号", values="最佳滞后秒数", aggfunc="mean").reindex(["风箱负压强度", "风箱平均温度"])

    fig = plt.figure(figsize=(14.4, 6.2))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.0, 0.85], hspace=0.23, figure=fig)
    add_suptitle(fig, "风箱变量影响方向热力矩阵", "颜色表示增加 1 个历史标准差后的平均 CO 变化；下方标注对应最佳滞后")

    ax1 = fig.add_subplot(gs[0, 0])
    vmax = max(10.0, float(np.nanmax(np.abs(response.to_numpy()))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("impact", [PALETTE["blue"], "#F9FAFB", PALETTE["red"]])
    im = ax1.imshow(response.to_numpy(), aspect="auto", cmap=cmap, norm=norm)
    ax1.set_xticks(np.arange(18))
    ax1.set_xticklabels([f"{i}#" for i in range(1, 19)])
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["负压强度", "平均温度"])
    ax1.set_title("A  单变量扰动平均响应", loc="left")
    for i in range(response.shape[0]):
        for j in range(response.shape[1]):
            val = response.iloc[i, j]
            if np.isfinite(val):
                ax1.text(j, i, f"{val:+.0f}", ha="center", va="center", fontsize=8, color="white" if abs(val) > vmax * 0.55 else PALETTE["ink"])
    for boundary in [5.5, 11.5]:
        ax1.axvline(boundary, color=PALETTE["ink"], linestyle="--", linewidth=1.0, alpha=0.5)
    ax1.text(2.5, -0.78, "上游", ha="center", va="center", fontsize=9, color=PALETTE["muted"])
    ax1.text(8.5, -0.78, "中游", ha="center", va="center", fontsize=9, color=PALETTE["muted"])
    ax1.text(14.5, -0.78, "下游", ha="center", va="center", fontsize=9, color=PALETTE["muted"])
    cbar = fig.colorbar(im, ax=ax1, shrink=0.88, pad=0.012)
    cbar.set_label("平均 CO 变化")

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    lag_values = lag.to_numpy()
    cmap_lag = LinearSegmentedColormap.from_list("lag", ["#ECEFF1", PALETTE["teal"], PALETTE["amber"]])
    im2 = ax2.imshow(lag_values, aspect="auto", cmap=cmap_lag, vmin=0, vmax=max(300, np.nanmax(lag_values)))
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["负压滞后", "温度滞后"])
    ax2.set_xticks(np.arange(18))
    ax2.set_xticklabels([f"{i}#" for i in range(1, 19)])
    ax2.set_title("B  入模最佳滞后 (s)", loc="left")
    for i in range(lag.shape[0]):
        for j in range(lag.shape[1]):
            val = lag.iloc[i, j]
            if np.isfinite(val):
                ax2.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color=PALETTE["ink"])
    for boundary in [5.5, 11.5]:
        ax2.axvline(boundary, color=PALETTE["ink"], linestyle="--", linewidth=1.0, alpha=0.5)
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.88, pad=0.012)
    cbar2.set_label("秒")

    for ax in [ax1, ax2]:
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    return save_fig(fig, "图2E-7_风箱影响方向热力矩阵")


def plot_ablation_waterfall(ablation: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.4), sharey=False)
    add_suptitle(fig, "变量组消融贡献", "将指定变量组替换为训练集均值后重新预测；RMSE 增加越大，说明该组信息越关键")
    model_order = ["过程变量模型", "监测型模型"]
    titles = ["A  不含历史 CO 的过程变量模型", "B  推荐监测型模型"]

    for ax, model, title in zip(axes, model_order, titles):
        sub = ablation[(ablation["模型"] == model) & (ablation["变量组"] != "完整模型")].copy()
        sub = sub.sort_values("RMSE相对完整模型变化", ascending=True)
        y = np.arange(len(sub))
        vals = sub["RMSE相对完整模型变化"].to_numpy(dtype=float)
        colors = [PALETTE["red"] if v >= 0 else PALETTE["blue"] for v in vals]
        ax.axvline(0, color=PALETTE["ink"], linewidth=1.0)
        ax.barh(y, vals, color=colors, alpha=0.86, edgecolor="white", linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["变量组"])
        ax.set_xlabel("RMSE 相对完整模型变化")
        ax.set_title(title, loc="left")
        beautify(ax, "x")
        max_abs = max(10.0, float(np.nanmax(np.abs(vals))) * 1.18)
        ax.set_xlim(-max_abs, max_abs)
        full = ablation[(ablation["模型"] == model) & (ablation["变量组"] == "完整模型")]
        if not full.empty:
            ax.text(
                0.98,
                0.04,
                f"完整模型 RMSE={float(full.iloc[0]['测试RMSE']):.1f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.5,
                color=PALETTE["muted"],
            )
        for yi, val in zip(y, vals):
            ax.text(val + (max_abs * 0.025 if val >= 0 else -max_abs * 0.025), yi, f"{val:+.1f}", va="center", ha="left" if val >= 0 else "right", fontsize=8.5)

    return save_fig(fig, "图2E-8_变量组消融贡献瀑布")


def plot_partial_dependence(pdp: pd.DataFrame) -> list[Path]:
    variables = list(dict.fromkeys(pdp["变量"].tolist()))
    n = len(variables)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13.2, 4.2 * rows), squeeze=False)
    add_suptitle(fig, "关键变量偏依赖曲线", "纵轴为相对训练均值取值的平均 CO 变化，用于展示非线性边际形态")

    line_colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["amber"], PALETTE["coral"], PALETTE["violet"]]
    for idx, variable in enumerate(variables):
        ax = axes[idx // cols, idx % cols]
        sub = pdp[pdp["变量"] == variable].sort_values("取值")
        x = sub["取值"].to_numpy(dtype=float)
        y = sub["相对训练均值CO变化"].to_numpy(dtype=float)
        color = line_colors[idx % len(line_colors)]
        ax.axhline(0, color=PALETTE["ink"], linewidth=1.0)
        ax.plot(x, y, color=color, linewidth=2.4)
        ax.fill_between(x, 0, y, where=y >= 0, color=PALETTE["red"], alpha=0.18, interpolate=True)
        ax.fill_between(x, 0, y, where=y < 0, color=PALETTE["blue"], alpha=0.16, interpolate=True)
        ax.scatter([x[0], x[-1]], [y[0], y[-1]], color=color, s=28, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_title(f"{chr(65 + idx)}  {variable}", loc="left")
        ax.set_xlabel("变量取值")
        ax.set_ylabel("相对 CO 变化")
        beautify(ax)
        ax.text(0.98, 0.06, f"范围: {x.min():.2f} ~ {x.max():.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=PALETTE["muted"])

    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    return save_fig(fig, "图2E-9_关键变量偏依赖曲线")


def write_catalog(outputs: dict[str, list[Path]]) -> Path:
    lines = [
        "# 第2问论文增强图目录",
        "",
        "所有图均由现有结果表自动生成，未重新训练模型。每张图同时导出 PNG、SVG、PDF 三种格式。",
        "",
    ]
    descriptions = {
        "图2E-1_核心模型性能仪表板": "核心模型测试性能、稳定性和高 CO 区间表现。",
        "图2E-2_测试集CO预测时序残差": "测试集实测/预测曲线与推荐模型残差诊断。",
        "图2E-3_预测一致性密度散点": "同一预测数据的实测-预测密度散点，可替代表格展示拟合质量。",
        "图2E-4_时滞配准三维收益": "时滞方案、模型类型与测试 RMSE 的三维比较。",
        "图2E-5_风箱空间时滞三维剖面": "风箱编号、变量类型、最佳滞后与相关方向的空间剖面。",
        "图2E-6_变量重要性敏感性双编码": "置换重要性与扰动响应方向合并展示。",
        "图2E-7_风箱影响方向热力矩阵": "18 个风箱负压/温度的响应方向与入模滞后。",
        "图2E-8_变量组消融贡献瀑布": "变量组消融后 RMSE 变化，突出历史 CO 与过程变量贡献。",
        "图2E-9_关键变量偏依赖曲线": "关键变量的非线性边际响应形态。",
    }
    for stem, paths in outputs.items():
        rels = [p.relative_to(ROOT).as_posix() for p in paths]
        lines.append(f"## {stem}")
        lines.append(descriptions.get(stem, ""))
        lines.append("")
        for rel in rels:
            lines.append(f"- `{rel}`")
        lines.append("")

    catalog = OUT_DIR / "图表目录.md"
    catalog.write_text("\n".join(lines), encoding="utf-8")
    return catalog


def main() -> None:
    setup_style()
    ensure_out()

    pred = read_csv(PRED_PATH)
    metrics = read_csv(METRIC_PATH)
    baseline = read_csv(BASELINE_PATH)
    lag_val = read_csv(LAG_VALIDATE_PATH)
    importance = read_csv(IMPORTANCE_PATH)
    sensitivity = read_csv(SENSITIVITY_PATH)
    ablation = read_csv(ABLATION_PATH)
    best_lag = read_csv(BEST_LAG_PATH)
    pdp = read_csv(PDP_PATH)

    outputs: dict[str, list[Path]] = {}
    for maker, args, stem in [
        (plot_model_dashboard, (metrics, baseline), "图2E-1_核心模型性能仪表板"),
        (plot_prediction_series, (pred, metrics), "图2E-2_测试集CO预测时序残差"),
        (plot_prediction_hexbin, (pred, metrics), "图2E-3_预测一致性密度散点"),
        (plot_lag_gain_3d, (lag_val,), "图2E-4_时滞配准三维收益"),
        (plot_windbox_lag_3d, (best_lag,), "图2E-5_风箱空间时滞三维剖面"),
        (plot_importance_sensitivity, (importance, sensitivity), "图2E-6_变量重要性敏感性双编码"),
        (plot_windbox_effect_heatmap, (sensitivity,), "图2E-7_风箱影响方向热力矩阵"),
        (plot_ablation_waterfall, (ablation,), "图2E-8_变量组消融贡献瀑布"),
        (plot_partial_dependence, (pdp,), "图2E-9_关键变量偏依赖曲线"),
    ]:
        outputs[stem] = maker(*args)

    catalog = write_catalog(outputs)
    print(f"生成完成：{OUT_DIR}")
    print(f"图表目录：{catalog}")
    print(f"图片数量：{sum(len(v) for v in outputs.values())}")


if __name__ == "__main__":
    main()
