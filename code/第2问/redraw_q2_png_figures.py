# -*- coding: utf-8 -*-
"""Redraw Question 2 figures as paper-style PNG files only.

The visual style is intentionally close to mathematical-modeling papers:
compact panels, white background, light grid lines, restrained blue/orange/
green/purple palette, and data annotations only where they clarify the result.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "第2问" / "结果数据"
MID_DIR = ROOT / "data" / "第2问" / "中间数据"
OUT_DIR = ROOT / "figure" / "第2问" / "论文重绘PNG"

TARGET_COL = "烧结大烟道外排CO浓度"
MAIN_MODEL = "基线3_过程变量+历史CO监测型模型"
SENS_VALUE = "增加1个历史标准差后的平均CO变化"


COLORS = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "teal": "#72B7B2",
    "purple": "#7D3C98",
    "cyan": "#2CB1BC",
    "yellow": "#ECA82C",
    "gray": "#8FA1AD",
    "dark": "#2F3A40",
    "grid": "#E8EEF2",
    "text": "#263238",
}

MODEL_COLOR = {
    "过程变量模型": COLORS["purple"],
    "线性过程变量模型_对比": COLORS["gray"],
    "基线1_CO_lag1持续性模型": COLORS["gray"],
    "基线2_历史CO_AR模型": COLORS["blue"],
    MAIN_MODEL: COLORS["green"],
    "残差校正监测型模型_对比": COLORS["orange"],
    "非线性监测_梯度提升树": COLORS["red"],
    "非线性监测_LightGBM": COLORS["teal"],
    "非线性监测_XGBoost": COLORS["purple"],
    "非线性监测_随机森林": COLORS["cyan"],
}


def setup_style() -> None:
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#B8C4CC",
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.3,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "figure.dpi": 160,
            "savefig.dpi": 420,
        }
    )


def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "pred": pd.read_csv(DATA_DIR / "CO预测结果_全样本.csv", encoding="utf-8-sig"),
        "metric": pd.read_csv(DATA_DIR / "模型评价指标.csv", encoding="utf-8-sig"),
        "baseline": pd.read_csv(DATA_DIR / "基线模型比较.csv", encoding="utf-8-sig"),
        "lag": pd.read_csv(DATA_DIR / "时滞合理性验证.csv", encoding="utf-8-sig"),
        "importance": pd.read_csv(DATA_DIR / "变量影响强度_过程变量模型.csv", encoding="utf-8-sig"),
        "sensitivity": pd.read_csv(DATA_DIR / "过程变量敏感性分析_过程变量模型.csv", encoding="utf-8-sig"),
        "ablation": pd.read_csv(DATA_DIR / "变量组消融分析.csv", encoding="utf-8-sig"),
        "pdp": pd.read_csv(DATA_DIR / "关键变量偏依赖曲线.csv", encoding="utf-8-sig"),
        "best_lag": pd.read_csv(MID_DIR / "最佳滞后配准.csv", encoding="utf-8-sig"),
    }


def ensure_out() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_png(fig: mpl.figure.Figure, name: str) -> Path:
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=420)
    plt.close(fig)
    return path


def style_ax(ax: mpl.axes.Axes, grid: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.grid(True, axis=grid, color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)


def label_panel(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["dark"],
    )


def add_caption(fig: mpl.figure.Figure, text: str) -> None:
    fig.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=8.0, color=COLORS["dark"])


def short_model(name: str) -> str:
    mapping = {
        "过程变量模型": "过程变量",
        "基线1_CO_lag1持续性模型": "CO-lag1",
        "基线2_历史CO_AR模型": "AR",
        MAIN_MODEL: "监测型",
        "残差校正监测型模型_对比": "残差校正",
        "非线性监测_梯度提升树": "GBDT",
        "非线性监测_LightGBM": "LightGBM",
        "非线性监测_XGBoost": "XGBoost",
        "非线性监测_随机森林": "RF",
    }
    return mapping.get(str(name), str(name))


def windbox_no(text: object) -> int | None:
    m = re.search(r"(\d+)#风箱", str(text))
    return int(m.group(1)) if m else None


def fig_model_metrics(t: dict[str, pd.DataFrame]) -> Path:
    metrics = t["metric"]
    baseline = t["baseline"]
    selected = [
        "过程变量模型",
        "基线1_CO_lag1持续性模型",
        "基线2_历史CO_AR模型",
        MAIN_MODEL,
        "残差校正监测型模型_对比",
        "非线性监测_梯度提升树",
    ]
    test = metrics[(metrics["数据集"] == "测试集") & (metrics["模型"].isin(selected))].copy()
    test = test.set_index("模型").loc[selected].reset_index()
    x = np.arange(len(test))

    fig = plt.figure(figsize=(7.4, 5.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    order = test.sort_values("RMSE")
    colors = [MODEL_COLOR.get(m, COLORS["gray"]) for m in order["模型"]]
    ax1.barh(np.arange(len(order)), order["RMSE"], color=colors, height=0.58)
    ax1.set_yticks(np.arange(len(order)))
    ax1.set_yticklabels([short_model(m) for m in order["模型"]])
    ax1.invert_yaxis()
    ax1.set_xlabel("RMSE")
    ax1.set_title("测试误差对比")
    for yi, v in enumerate(order["RMSE"]):
        ax1.text(v + 6, yi, f"{v:.0f}", va="center", fontsize=7.0)
    style_ax(ax1)
    label_panel(ax1, "A")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, test["R2"], color=COLORS["green"], marker="o", linewidth=1.4, markersize=4.2, label="R²")
    ax2.set_xticks(x)
    ax2.set_xticklabels([short_model(m) for m in test["模型"]], rotation=25, ha="right")
    ax2.set_ylabel("R²")
    ax2.set_ylim(max(0, test["R2"].min() - 0.08), 0.96)
    ax2b = ax2.twinx()
    ax2b.plot(x, test["MAPE_%"], color=COLORS["orange"], marker="s", linewidth=1.2, markersize=3.8, label="MAPE")
    ax2b.set_ylabel("MAPE (%)")
    ax2.set_title("解释度与相对误差")
    style_ax(ax2)
    ax2b.spines["top"].set_visible(False)
    ax2b.grid(False)
    lines = ax2.get_lines() + ax2b.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper right")
    label_panel(ax2, "B")

    ax3 = fig.add_subplot(gs[1, :])
    base_models = ["基线1_CO_lag1持续性模型", "基线2_历史CO_AR模型", MAIN_MODEL]
    dataset_order = ["训练集", "验证集", "测试集"]
    for model in base_models:
        sub = baseline[baseline["模型"] == model].set_index("数据集").loc[dataset_order]
        ax3.plot(dataset_order, sub["RMSE"], marker="o", linewidth=1.55, markersize=4.2, color=MODEL_COLOR[model], label=short_model(model))
    ax3.set_ylabel("RMSE")
    ax3.set_title("惯性基线、AR 与推荐监测型模型的分割稳定性")
    ax3.legend(frameon=False, ncol=3, loc="upper left")
    style_ax(ax3)
    label_panel(ax3, "C")

    add_caption(fig, "图2-1  核心模型性能比较。推荐监测型模型在测试 RMSE 和 R² 上综合最优。")
    return save_png(fig, "图2-1_核心模型性能比较.png")


def fig_prediction_series(t: dict[str, pd.DataFrame]) -> Path:
    pred = t["pred"]
    metric = t["metric"]
    test = pred[pred["数据集"] == "测试集"].copy()
    test["t_min"] = (test["时间_s"] - test["时间_s"].min()) / 60.0
    threshold = float(metric[(metric["数据集"] == "测试集") & (metric["模型"] == MAIN_MODEL)]["高CO阈值_训练90分位"].iloc[0])

    fig = plt.figure(figsize=(7.5, 4.9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.35, 0.95], hspace=0.08, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(test["t_min"], test[TARGET_COL], color=COLORS["dark"], linewidth=1.35, label="实测")
    ax1.plot(test["t_min"], test["监测型模型_预测CO浓度"], color=COLORS["green"], linewidth=1.2, label="监测型")
    ax1.plot(test["t_min"], test["基线2_历史CO_AR模型_预测CO浓度"], color=COLORS["blue"], linewidth=0.95, alpha=0.9, label="AR")
    ax1.plot(test["t_min"], test["非线性监测_梯度提升树_预测CO浓度"], color=COLORS["red"], linewidth=0.85, alpha=0.82, label="GBDT")
    ax1.axhline(threshold, color=COLORS["orange"], linestyle="--", linewidth=0.85, label="训练90分位")
    ax1.fill_between(test["t_min"], threshold, test[TARGET_COL].max() * 1.02, color=COLORS["orange"], alpha=0.055)
    ax1.set_ylabel("CO浓度")
    ax1.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax1.set_title("测试集预测曲线")
    style_ax(ax1)
    ax1.tick_params(labelbottom=False)
    label_panel(ax1, "A")

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    resid = test["监测型模型_残差"].to_numpy(dtype=float)
    x = test["t_min"].to_numpy(dtype=float)
    ax2.axhline(0, color=COLORS["dark"], linewidth=0.75)
    ax2.fill_between(x, 0, resid, where=resid >= 0, color=COLORS["red"], alpha=0.25, interpolate=True)
    ax2.fill_between(x, 0, resid, where=resid < 0, color=COLORS["blue"], alpha=0.25, interpolate=True)
    ax2.set_xlabel("测试集相对时间 / min")
    ax2.set_ylabel("残差")
    ax2.set_title("推荐模型残差")
    style_ax(ax2)
    label_panel(ax2, "B")

    add_caption(fig, "图2-2  测试集 CO 实测值、预测值与推荐模型残差。监测型模型能跟随短期波动和高 CO 峰值。")
    return save_png(fig, "图2-2_测试集预测曲线与残差.png")


def fig_prediction_density(t: dict[str, pd.DataFrame]) -> Path:
    pred = t["pred"]
    metric = t["metric"]
    test = pred[pred["数据集"] == "测试集"].copy()
    panels = [
        ("监测型模型", "监测型模型_预测CO浓度", MAIN_MODEL),
        ("过程变量模型", "过程变量模型_预测CO浓度", "过程变量模型"),
    ]
    actual = test[TARGET_COL].astype(float)
    lo = min(actual.min(), *(test[col].min() for _, col, _ in panels))
    hi = max(actual.max(), *(test[col].max() for _, col, _ in panels))
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad
    cmap = LinearSegmentedColormap.from_list("density", ["#F7FBFF", "#CFE8F3", COLORS["teal"], COLORS["orange"], COLORS["red"]])

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.75), sharex=True, sharey=True)
    for ax, (title, col, model), label in zip(axes, panels, ["A", "B"]):
        hb = ax.hexbin(actual, test[col], gridsize=32, cmap=cmap, mincnt=1, linewidths=0)
        ax.plot([lo, hi], [lo, hi], color=COLORS["dark"], linestyle="--", linewidth=0.85)
        coef = np.polyfit(actual, test[col], 1)
        xs = np.linspace(lo, hi, 100)
        ax.plot(xs, coef[0] * xs + coef[1], color=COLORS["red"], linewidth=1.05)
        m = metric[(metric["数据集"] == "测试集") & (metric["模型"] == model)].iloc[0]
        ax.text(0.05, 0.95, f"RMSE={m['RMSE']:.1f}\nR²={m['R2']:.3f}", transform=ax.transAxes, va="top", fontsize=7.2, bbox=dict(facecolor="white", edgecolor="#CDD7DF", pad=2.2, alpha=0.88))
        ax.set_title(title)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        style_ax(ax)
        label_panel(ax, label)
    axes[0].set_ylabel("预测 CO")
    for ax in axes:
        ax.set_xlabel("实测 CO")
    cbar = fig.colorbar(hb, ax=axes.ravel().tolist(), shrink=0.78, pad=0.02)
    cbar.set_label("样本密度", fontsize=8)
    add_caption(fig, "图2-3  实测-预测密度散点。过程变量单独建模无法复原 CO 绝对水平，历史 CO 对短时预测至关重要。")
    return save_png(fig, "图2-3_预测一致性密度散点.png")


def fig_lag_validation(t: dict[str, pd.DataFrame]) -> Path:
    lag = t["lag"]
    scen_order = ["无时滞模型_lag0", "物理初始时滞截断_对比模型", "分组时滞组合优化_主模型", "单变量局部搜索_对比模型"]
    scen_name = ["无时滞", "物理截断", "分组优化", "局部搜索"]
    model_types = ["过程变量模型", "监测型模型", "非线性监测型模型"]
    colors = [COLORS["purple"], COLORS["green"], COLORS["red"]]

    fig = plt.figure(figsize=(7.5, 4.7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.25, 1.0], wspace=0.34, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    width = 0.23
    x = np.arange(len(scen_order))
    for i, (mt, color) in enumerate(zip(model_types, colors)):
        sub = lag[lag["模型类型"] == mt].set_index("实验场景").loc[scen_order]
        ax1.bar(x + (i - 1) * width, sub["RMSE"], width=width, color=color, label=mt.replace("模型", ""), alpha=0.88)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scen_name, rotation=22, ha="right")
    ax1.set_ylabel("测试 RMSE")
    ax1.set_title("不同时滞方案下的模型误差")
    ax1.legend(frameon=False, ncol=1, loc="upper right")
    style_ax(ax1)
    label_panel(ax1, "A")

    ax2 = fig.add_subplot(gs[0, 1])
    mon = lag[lag["模型类型"] == "监测型模型"].set_index("实验场景").loc[scen_order]
    ax2.bar(x, mon["RMSE相对无时滞降低率_%"], color=[COLORS["gray"], COLORS["blue"], COLORS["green"], COLORS["orange"]], width=0.52)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scen_name, rotation=22, ha="right")
    ax2.set_ylabel("RMSE降低率 / %")
    ax2.set_title("监测型模型的时滞收益")
    style_ax(ax2)
    ax2b = ax2.twinx()
    ax2b.plot(x, mon["平均最佳滞后_s"], color=COLORS["dark"], marker="D", linewidth=1.15, markersize=3.8)
    ax2b.set_ylabel("平均滞后 / s")
    ax2b.spines["top"].set_visible(False)
    ax2b.grid(False)
    for xi, g in zip(x, mon["RMSE相对无时滞降低率_%"]):
        ax2.text(xi, g + 1.2, f"{g:.1f}", ha="center", fontsize=7.0)
    label_panel(ax2, "B")

    add_caption(fig, "图2-4  时滞配准验证。物理截断与分组优化将监测型模型 RMSE 降低约 48%。")
    return save_png(fig, "图2-4_时滞配准验证.png")


def fig_windbox_heatmap(t: dict[str, pd.DataFrame]) -> Path:
    sens = t["sensitivity"].copy()
    data = sens[sens["变量组"].isin(["风箱负压强度", "风箱平均温度"])].copy()
    data["风箱编号"] = data["原始字段"].map(windbox_no)
    data = data.dropna(subset=["风箱编号"])
    data["风箱编号"] = data["风箱编号"].astype(int)
    response = data.pivot_table(index="变量组", columns="风箱编号", values=SENS_VALUE, aggfunc="mean").reindex(["风箱负压强度", "风箱平均温度"])
    lag = data.pivot_table(index="变量组", columns="风箱编号", values="最佳滞后秒数", aggfunc="mean").reindex(["风箱负压强度", "风箱平均温度"])

    fig = plt.figure(figsize=(7.5, 3.95))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.35, 0.9], hspace=0.28, figure=fig)
    vmax = float(np.nanmax(np.abs(response.to_numpy())))
    cmap_resp = LinearSegmentedColormap.from_list("resp", [COLORS["blue"], "#F8FAFB", COLORS["red"]])
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(response.to_numpy(), aspect="auto", cmap=cmap_resp, norm=norm)
    ax1.set_title("单变量增加1个历史标准差后的平均 CO 响应")
    ax1.set_xticks(np.arange(18))
    ax1.set_xticklabels([f"{i}#" for i in range(1, 19)])
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["负压强度", "平均温度"])
    for i in range(response.shape[0]):
        for j in range(response.shape[1]):
            val = response.iloc[i, j]
            if np.isfinite(val):
                ax1.text(j, i, f"{val:+.0f}", ha="center", va="center", fontsize=6.3, color="white" if abs(val) > vmax * 0.62 else COLORS["text"])
    for b in [5.5, 11.5]:
        ax1.axvline(b, color=COLORS["dark"], linestyle="--", linewidth=0.8, alpha=0.55)
    ax1.tick_params(length=0)
    for sp in ax1.spines.values():
        sp.set_visible(False)
    cbar = fig.colorbar(im, ax=ax1, pad=0.01, fraction=0.026)
    cbar.set_label("CO变化", fontsize=7.4)
    label_panel(ax1, "A")

    ax2 = fig.add_subplot(gs[1, 0])
    cmap_lag = LinearSegmentedColormap.from_list("lag", ["#F1F5F8", COLORS["teal"], COLORS["orange"]])
    im2 = ax2.imshow(lag.to_numpy(), aspect="auto", cmap=cmap_lag, vmin=0, vmax=300)
    ax2.set_title("最终入模最佳滞后 / s")
    ax2.set_xticks(np.arange(18))
    ax2.set_xticklabels([f"{i}#" for i in range(1, 19)])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["负压", "温度"])
    for i in range(lag.shape[0]):
        for j in range(lag.shape[1]):
            val = lag.iloc[i, j]
            if np.isfinite(val):
                ax2.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6.2)
    for b in [5.5, 11.5]:
        ax2.axvline(b, color=COLORS["dark"], linestyle="--", linewidth=0.8, alpha=0.55)
    ax2.tick_params(length=0)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    cbar2 = fig.colorbar(im2, ax=ax2, pad=0.01, fraction=0.026)
    cbar2.set_label("s", fontsize=7.4)
    label_panel(ax2, "B")

    add_caption(fig, "图2-5  风箱变量影响方向与滞后矩阵。不同风箱位置的负压、温度响应方向并不一致。")
    return save_png(fig, "图2-5_风箱影响方向与滞后矩阵.png")


def fig_importance_sensitivity(t: dict[str, pd.DataFrame]) -> Path:
    imp = t["importance"].sort_values("综合影响强度", ascending=False).head(10).copy()
    sens = t["sensitivity"][["原始字段", SENS_VALUE, "最佳滞后秒数"]].copy()
    df = imp.merge(sens, on="原始字段", how="left", suffixes=("", "_sens")).fillna({SENS_VALUE: 0})
    df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))

    fig = plt.figure(figsize=(7.5, 4.65))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    colors = []
    for v in df["原始字段"]:
        if "大烟道" in v:
            colors.append(COLORS["purple"])
        elif "温度" in v:
            colors.append(COLORS["orange"])
        elif "机速" in v:
            colors.append(COLORS["green"])
        else:
            colors.append(COLORS["blue"])
    ax1.barh(y, df["综合影响强度"], color=colors, height=0.58)
    ax1.set_yticks(y)
    ax1.set_yticklabels(df["原始字段"])
    ax1.set_xlabel("置换重要性")
    ax1.set_title("变量影响强度排序")
    for yi, v in zip(y, df["综合影响强度"]):
        ax1.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=6.8)
    style_ax(ax1)
    label_panel(ax1, "A")

    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    resp = df[SENS_VALUE].to_numpy(dtype=float)
    ax2.axvline(0, color=COLORS["dark"], linewidth=0.75)
    for yi, val, lagv in zip(y, resp, df["最佳滞后秒数"]):
        color = COLORS["red"] if val >= 0 else COLORS["blue"]
        ax2.hlines(yi, 0, val, color=color, linewidth=2.0, alpha=0.82)
        ax2.scatter(val, yi, s=32 + float(lagv) * 0.10, color=color, edgecolor="white", linewidth=0.55, zorder=3)
        ax2.text(val + (1.7 if val >= 0 else -1.7), yi, f"{val:+.1f}", va="center", ha="left" if val >= 0 else "right", fontsize=6.8)
    ax2.tick_params(axis="y", labelleft=False)
    ax2.set_xlabel("平均 CO 响应")
    ax2.set_title("扰动方向与幅度")
    style_ax(ax2, "x")
    label_panel(ax2, "B")

    fig.subplots_adjust(bottom=0.19, wspace=0.30)
    add_caption(fig, "图2-6  变量重要性与敏感性。大烟道负压强度、1#风箱温度和机速是主要解释变量。")
    return save_png(fig, "图2-6_变量重要性与敏感性.png")


def fig_ablation(t: dict[str, pd.DataFrame]) -> Path:
    abl = t["ablation"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.75))
    for ax, model, label in zip(axes, ["过程变量模型", "监测型模型"], ["A", "B"]):
        sub = abl[(abl["模型"] == model) & (abl["变量组"] != "完整模型")].copy()
        sub = sub.sort_values("RMSE相对完整模型变化")
        y = np.arange(len(sub))
        vals = sub["RMSE相对完整模型变化"].to_numpy(dtype=float)
        colors = [COLORS["red"] if v >= 0 else COLORS["blue"] for v in vals]
        ax.axvline(0, color=COLORS["dark"], linewidth=0.75)
        ax.barh(y, vals, color=colors, height=0.58, alpha=0.86)
        pad = max(2.0, float(np.nanmax(np.abs(vals))) * 0.08)
        ax.set_xlim(float(np.nanmin(vals)) - pad, float(np.nanmax(vals)) + pad)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["变量组"])
        ax.set_xlabel("RMSE变化")
        ax.set_title("过程变量模型" if model == "过程变量模型" else "监测型模型")
        style_ax(ax, "x")
        for yi, v in zip(y, vals):
            offset = pad * 0.20
            ax.text(v + offset, yi, f"{v:+.1f}", va="center", ha="left", fontsize=6.8, clip_on=False)
        label_panel(ax, label)
    fig.subplots_adjust(bottom=0.23, left=0.16, right=0.98, wspace=0.48)
    add_caption(fig, "图2-7  变量组消融贡献。监测型模型主要依赖历史 CO，过程变量组用于状态解释与补充。")
    return save_png(fig, "图2-7_变量组消融贡献.png")


def fig_pdp(t: dict[str, pd.DataFrame]) -> Path:
    pdp = t["pdp"].copy()
    vars_ = list(dict.fromkeys(pdp["变量"].tolist()))[:4]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.15), sharey=False)
    palette = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"]]
    for ax, var, color, label in zip(axes.ravel(), vars_, palette, ["A", "B", "C", "D"]):
        sub = pdp[pdp["变量"] == var].sort_values("取值")
        x = sub["取值"].to_numpy(dtype=float)
        y = sub["相对训练均值CO变化"].to_numpy(dtype=float)
        ax.axhline(0, color=COLORS["dark"], linewidth=0.72)
        ax.plot(x, y, color=color, linewidth=1.35)
        ax.fill_between(x, 0, y, color=color, alpha=0.16)
        ax.set_title(var, fontsize=8.5, pad=8)
        if label in ["C", "D"]:
            ax.set_xlabel("取值")
        if label in ["A", "C"]:
            ax.set_ylabel("CO变化")
        style_ax(ax)
        label_panel(ax, label)
    fig.subplots_adjust(bottom=0.16, hspace=0.58, wspace=0.30)
    add_caption(fig, "图2-8  关键变量偏依赖曲线。曲线展示过程变量模型中的非线性边际响应。")
    return save_png(fig, "图2-8_关键变量偏依赖曲线.png")


def fig_windbox_lag_3d(t: dict[str, pd.DataFrame]) -> Path:
    best = t["best_lag"].copy()
    wb = best[best["变量组"].isin(["风箱负压强度", "风箱平均温度"])].copy()
    wb["风箱编号"] = wb["字段名"].map(windbox_no)
    wb = wb.dropna(subset=["风箱编号"])
    wb["风箱编号"] = wb["风箱编号"].astype(int)
    sens = t["sensitivity"].copy()
    sens = sens[sens["变量组"].isin(["风箱负压强度", "风箱平均温度"])][["原始字段", SENS_VALUE]]
    wb = wb.merge(sens, left_on="字段名", right_on="原始字段", how="left")
    wb[SENS_VALUE] = wb[SENS_VALUE].fillna(0.0)

    fig = plt.figure(figsize=(7.5, 4.9))
    ax = fig.add_subplot(111, projection="3d")
    cmap = LinearSegmentedColormap.from_list("resp3d", [COLORS["blue"], "#F8FAFB", COLORS["red"]])
    vmax = max(1.0, float(np.nanmax(np.abs(wb[SENS_VALUE]))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    y_map = {"风箱负压强度": 0, "风箱平均温度": 1}

    for group, sub in wb.groupby("变量组"):
        sub = sub.sort_values("风箱编号")
        xs = sub["风箱编号"].to_numpy(dtype=float)
        ys = np.full_like(xs, y_map[group], dtype=float)
        zs = sub["最佳滞后秒数"].to_numpy(dtype=float)
        colors = cmap(norm(sub[SENS_VALUE].to_numpy(dtype=float)))
        ax.bar3d(xs - 0.25, ys - 0.18, np.zeros_like(zs), 0.5, 0.36, zs, color=colors, alpha=0.88, shade=True, edgecolor="white", linewidth=0.25)
    for b in [6.5, 12.5]:
        ax.plot([b, b], [-0.35, 1.35], [0, 0], color=COLORS["dark"], linestyle="--", linewidth=0.7, alpha=0.55)
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("变量类型")
    ax.set_zlabel("最佳滞后 / s")
    ax.set_xticks(range(1, 19, 2))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["负压", "温度"])
    ax.set_zlim(0, 320)
    ax.set_title("风箱空间位置与最佳滞后")
    ax.view_init(elev=24, azim=-58)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.05)
    cbar.set_label("CO响应", fontsize=7.4)
    add_caption(fig, "图2-9  风箱最佳时滞三维图。柱高表示滞后秒数，颜色表示扰动后 CO 响应方向。")
    return save_png(fig, "图2-9_风箱最佳时滞三维图.png")


def main() -> None:
    setup_style()
    ensure_out()
    t = load_tables()
    makers = [
        fig_model_metrics,
        fig_prediction_series,
        fig_prediction_density,
        fig_lag_validation,
        fig_windbox_heatmap,
        fig_importance_sensitivity,
        fig_ablation,
        fig_pdp,
        fig_windbox_lag_3d,
    ]
    paths = [maker(t) for maker in makers]
    print("已生成PNG：")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
