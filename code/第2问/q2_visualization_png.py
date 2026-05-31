# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "第2问"
MID_DIR = DATA_DIR / "中间数据"
RESULT_DIR = DATA_DIR / "结果数据"
OUT_DIR = ROOT / "figure" / "第2问" / "可视化PNG"

TARGET = "烧结大烟道外排CO浓度"
MONITOR_PRED = "监测型模型_预测CO浓度"
MONITOR_RESID = "监测型模型_残差"
STANDARD_CO = 2800

COLORS = {
    "blue": "#2F6B9A",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#C44E52",
    "purple": "#7B61A8",
    "gray": "#6E7781",
    "dark": "#263238",
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


def short_model_name(name: str) -> str:
    mapping = {
        "过程变量模型": "过程变量",
        "基线1_CO_lag1持续性模型": "CO lag1",
        "基线2_历史CO_AR模型": "历史CO AR",
        "基线3_过程变量+历史CO监测型模型": "推荐监测",
        "残差校正监测型模型_对比": "残差校正",
        "非线性监测_LightGBM": "LightGBM",
        "非线性监测_XGBoost": "XGBoost",
        "非线性监测_梯度提升树": "GBDT",
        "非线性监测_随机森林": "随机森林",
    }
    return mapping.get(name, name)


def windbox_no(text: str) -> int | None:
    match = re.match(r"(\d+)#风箱", str(text))
    return int(match.group(1)) if match else None


def dataset_spans(df: pd.DataFrame) -> list[tuple[float, float, str]]:
    spans: list[tuple[float, float, str]] = []
    if "数据集" not in df:
        return spans
    data = df[["时间_s", "数据集"]].dropna().copy()
    data["时间_min"] = data["时间_s"] / 60
    start = float(data["时间_min"].iloc[0])
    last_label = str(data["数据集"].iloc[0])
    last_time = start
    for _, row in data.iloc[1:].iterrows():
        label = str(row["数据集"])
        current = float(row["时间_min"])
        if label != last_label:
            spans.append((start, last_time, last_label))
            start = current
            last_label = label
        last_time = current
    spans.append((start, last_time, last_label))
    return spans


def plot_co_cleaning_overview() -> None:
    detail = read_csv(MID_DIR / "CO异常修正明细.csv")
    pred = read_csv(RESULT_DIR / "CO预测结果_全样本.csv")
    detail["时间_min"] = detail["时间_s"] / 60

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.2), constrained_layout=True)
    ax = axes[0]
    ax.plot(detail["时间_min"], detail["原始CO"], color="#B7C0C7", lw=1.2, label="原始 CO")
    ax.plot(detail["时间_min"], detail["修正后CO"], color=COLORS["blue"], lw=1.5, label="异常修正后 CO")
    changed = detail[detail["是否修正"].astype(bool)]
    ax.scatter(changed["时间_min"], changed["原始CO"], s=26, color=COLORS["red"], zorder=3, label=f"修正点 n={len(changed)}")
    ax.axhline(STANDARD_CO, color=COLORS["orange"], ls="--", lw=1.1, label="背景标准 2800 mg/m$^3$")
    ax.set_title("A  CO 原始序列、异常修正与排放参考线")
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("CO 浓度 (mg/m$^3$)")
    ax.legend(ncol=4, frameon=False, loc="upper right")

    ax = axes[1]
    bins = np.linspace(detail[["原始CO", "修正后CO"]].min().min(), detail[["原始CO", "修正后CO"]].max().max(), 45)
    ax.hist(detail["原始CO"], bins=bins, color="#B7C0C7", alpha=0.72, label="原始 CO")
    ax.hist(detail["修正后CO"], bins=bins, color=COLORS["green"], alpha=0.55, label="修正后 CO")
    ax.axvline(STANDARD_CO, color=COLORS["orange"], ls="--", lw=1.1)
    ax.set_title("B  CO 分布对比：异常尖峰被拉回局部趋势")
    ax.set_xlabel("CO 浓度 (mg/m$^3$)")
    ax.set_ylabel("样本数")
    ax.legend(frameon=False)

    ax = axes[2]
    merged = detail[["序列", "时间_s", "修正后CO"]].merge(pred[["序列", "数据集"]], on="序列", how="left")
    merged["时间_min"] = merged["时间_s"] / 60
    span_colors = {"训练集": "#E8F2F8", "验证集": "#FEF3D8", "测试集": "#EDE7F6"}
    for start, end, label in dataset_spans(merged.dropna(subset=["数据集"])):
        ax.axvspan(start, end, color=span_colors.get(label, "#F2F2F2"), alpha=0.9, lw=0)
        ax.text((start + end) / 2, merged["修正后CO"].max() * 1.01, label, ha="center", va="bottom", fontsize=9, color=COLORS["dark"])
    ax.plot(merged["时间_min"], merged["修正后CO"], color=COLORS["dark"], lw=1.15)
    ax.set_title("C  建模样本时序划分")
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("修正后 CO (mg/m$^3$)")
    save(fig, "必备_01_CO数据清洗与样本划分.png")


def plot_model_performance() -> None:
    metrics = read_csv(RESULT_DIR / "模型评价指标.csv")
    wanted = [
        "过程变量模型",
        "基线1_CO_lag1持续性模型",
        "基线2_历史CO_AR模型",
        "基线3_过程变量+历史CO监测型模型",
        "残差校正监测型模型_对比",
    ]
    test = metrics[(metrics["数据集"] == "测试集") & (metrics["模型"].isin(wanted))].copy()
    test["模型短名"] = test["模型"].map(short_model_name)
    test["模型短名"] = pd.Categorical(test["模型短名"], [short_model_name(v) for v in wanted], ordered=True)
    test = test.sort_values("模型短名")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    specs = [("RMSE", "RMSE (mg/m$^3$)", COLORS["blue"]), ("MAPE_%", "MAPE (%)", COLORS["green"]), ("R2", "$R^2$", COLORS["purple"])]
    for ax, (col, label, color) in zip(axes, specs):
        bars = ax.bar(test["模型短名"].astype(str), test[col], color=color, alpha=0.86)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=25)
        best_idx = test[col].idxmin() if col != "R2" else test[col].idxmax()
        best_label = short_model_name(test.loc[best_idx, "模型"])
        for bar, value, model_label in zip(bars, test[col], test["模型短名"].astype(str)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3g}", ha="center", va="bottom", fontsize=8)
            if model_label == best_label:
                bar.set_color(COLORS["orange"])
    fig.suptitle("核心模型测试集性能：推荐监测模型在 RMSE / R² 上最优", y=1.04, fontsize=14, fontweight="bold")
    save(fig, "必备_02_核心模型性能比较.png")


def plot_lag_alignment() -> None:
    lag_val = read_csv(RESULT_DIR / "时滞合理性验证.csv")
    lag_val = lag_val[lag_val["模型类型"] == "监测型模型"].copy()
    lag_val["方案短名"] = (
        lag_val["实验场景"]
        .replace(
            {
                "无时滞模型_lag0": "无时滞",
                "物理初始时滞截断": "物理截断",
                "分组时滞组合优化": "分组优化",
                "单变量局部搜索对比": "单变量搜索",
            }
        )
        .astype(str)
    )
    lag = read_csv(MID_DIR / "最佳滞后配准.csv")
    wind = lag[lag["字段名"].str.contains("风箱", na=False)].copy()
    wind["风箱编号"] = wind["字段名"].map(windbox_no)
    wind = wind.dropna(subset=["风箱编号"])
    wind["变量类别"] = np.where(wind["字段名"].str.contains("负压强度"), "负压强度", "平均温度")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    ax = axes[0]
    palette = [COLORS["gray"], COLORS["green"], COLORS["orange"], COLORS["purple"]]
    bars = ax.bar(lag_val["方案短名"], lag_val["RMSE"], color=palette[: len(lag_val)], alpha=0.88)
    for bar, value in zip(bars, lag_val["RMSE"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("测试 RMSE (mg/m$^3$)")
    ax.set_title("A  数据配准带来的预测收益")

    ax = axes[1]
    for label, color, marker in [("负压强度", COLORS["blue"], "o"), ("平均温度", COLORS["red"], "s")]:
        sub = wind[wind["变量类别"] == label].sort_values("风箱编号")
        ax.plot(sub["风箱编号"], sub["最佳滞后秒数"], marker=marker, lw=2, color=color, label=label)
    ax.axhline(300, color=COLORS["gray"], ls="--", lw=1, label="300s 截断窗口")
    ax.set_xticks(range(1, 19))
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("最佳滞后 (s)")
    ax.set_title("B  各风箱变量的空间时滞")
    ax.legend(frameon=False)
    save(fig, "必备_03_时滞配准必要性.png")


def plot_prediction_residual() -> None:
    pred = read_csv(RESULT_DIR / "CO预测结果_全样本.csv")
    test = pred[pred["数据集"] == "测试集"].copy()
    test["时间_min"] = test["时间_s"] / 60

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.4), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(test["时间_min"], test[TARGET], color=COLORS["dark"], lw=1.7, label="实际 CO")
    ax.plot(test["时间_min"], test[MONITOR_PRED], color=COLORS["orange"], lw=1.55, label="推荐监测模型")
    ax.plot(test["时间_min"], test["基线1_CO_lag1持续性模型_预测CO浓度"], color=COLORS["blue"], lw=1.05, alpha=0.75, label="CO lag1")
    ax.set_ylabel("CO (mg/m$^3$)")
    ax.set_title("A  测试集实时预测曲线")
    ax.legend(ncol=3, frameon=False, loc="upper right")

    ax = axes[1]
    ax.axhline(0, color=COLORS["gray"], lw=1)
    ax.fill_between(test["时间_min"], test[MONITOR_RESID], 0, color=COLORS["purple"], alpha=0.22)
    ax.plot(test["时间_min"], test[MONITOR_RESID], color=COLORS["purple"], lw=1.1)
    ax.set_ylabel("预测残差")
    ax.set_title("B  推荐监测模型残差")

    ax = axes[2]
    rolling = test[MONITOR_RESID].pow(2).rolling(30, min_periods=8).mean().pow(0.5)
    ax.plot(test["时间_min"], rolling, color=COLORS["red"], lw=1.5)
    ax.set_title("C  30 点滚动 RMSE")
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("RMSE")
    save(fig, "必备_04_测试集实时预测与残差.png")


def plot_prediction_diagnostics() -> None:
    pred = read_csv(RESULT_DIR / "CO预测结果_全样本.csv")
    test = pred[pred["数据集"] == "测试集"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), constrained_layout=True)
    ax = axes[0]
    hb = ax.hexbin(test[TARGET], test[MONITOR_PRED], gridsize=32, cmap="viridis", mincnt=1)
    mn = min(test[TARGET].min(), test[MONITOR_PRED].min())
    mx = max(test[TARGET].max(), test[MONITOR_PRED].max())
    ax.plot([mn, mx], [mn, mx], color=COLORS["red"], ls="--", lw=1.2)
    ax.set_xlabel("实际 CO")
    ax.set_ylabel("预测 CO")
    ax.set_title("A  实际-预测密度散点")
    fig.colorbar(hb, ax=ax, label="样本密度")

    ax = axes[1]
    ax.hist(test[MONITOR_RESID], bins=32, color=COLORS["blue"], alpha=0.82)
    ax.axvline(0, color=COLORS["red"], ls="--", lw=1)
    ax.set_xlabel("残差 (预测-实际)")
    ax.set_ylabel("样本数")
    ax.set_title("B  残差分布")

    ax = axes[2]
    ax.scatter(test[TARGET], test[MONITOR_RESID], s=18, color=COLORS["purple"], alpha=0.65, edgecolor="none")
    ax.axhline(0, color=COLORS["red"], ls="--", lw=1)
    ax.set_xlabel("实际 CO")
    ax.set_ylabel("残差")
    ax.set_title("C  残差随 CO 水平变化")
    save(fig, "必备_05_实际预测一致性诊断.png")


def windbox_matrix(table: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for kind in ["风箱负压强度", "风箱平均温度"]:
        sub = table[table["变量组"] == kind].copy()
        sub["风箱编号"] = sub["原始字段"].map(windbox_no)
        values = []
        for i in range(1, 19):
            row = sub[sub["风箱编号"] == i]
            values.append(float(row[value_col].iloc[0]) if not row.empty else np.nan)
        rows.append(values)
    return pd.DataFrame(rows, index=["负压强度", "平均温度"], columns=[f"{i}#" for i in range(1, 19)])


def plot_windbox_influence_matrix() -> None:
    sens = read_csv(RESULT_DIR / "过程变量敏感性分析_过程变量模型.csv")
    lag = read_csv(MID_DIR / "最佳滞后配准.csv")
    lag = lag.rename(columns={"字段名": "原始字段"})
    lag["变量组"] = np.where(lag["原始字段"].str.contains("风箱负压强度", na=False), "风箱负压强度", np.where(lag["原始字段"].str.contains("风箱平均温度", na=False), "风箱平均温度", lag.get("变量组", "")))

    response = windbox_matrix(sens, "增加1个历史标准差后的平均CO变化")
    lag_mat = windbox_matrix(lag[lag["变量组"].isin(["风箱负压强度", "风箱平均温度"])], "最佳滞后秒数")
    corr_mat = windbox_matrix(sens, "训练集相关系数")

    fig, axes = plt.subplots(3, 1, figsize=(12, 5.9), constrained_layout=True)
    mats = [
        (response, "A  单变量增加 1 个历史标准差后的平均 CO 响应", "CO 变化", "RdBu_r", TwoSlopeNorm(vcenter=0, vmin=np.nanmin(response.values), vmax=np.nanmax(response.values))),
        (lag_mat, "B  最佳滞后时间", "秒", "viridis", None),
        (corr_mat, "C  训练集滞后相关系数", "相关系数", "RdBu_r", TwoSlopeNorm(vcenter=0, vmin=np.nanmin(corr_mat.values), vmax=np.nanmax(corr_mat.values))),
    ]
    for ax, (mat, title, label, cmap, norm) in zip(axes, mats):
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap, norm=norm)
        ax.set_yticks([0, 1], mat.index)
        ax.set_xticks(range(18), mat.columns)
        ax.set_title(title, loc="left")
        cbar = fig.colorbar(im, ax=ax, shrink=0.86)
        cbar.set_label(label)
        for y in range(mat.shape[0]):
            for x in range(mat.shape[1]):
                value = mat.iloc[y, x]
                if np.isfinite(value):
                    text = f"{value:.0f}" if label in {"CO 变化", "秒"} else f"{value:.2f}"
                    ax.text(x, y, text, ha="center", va="center", fontsize=7, color="white" if abs(value) > np.nanmax(np.abs(mat.values)) * 0.45 else COLORS["dark"])
    save(fig, "必备_06_风箱影响方向与滞后矩阵.png")


def plot_importance_sensitivity() -> None:
    importance = read_csv(RESULT_DIR / "变量影响强度_过程变量模型.csv").head(14).copy()
    sens = read_csv(RESULT_DIR / "过程变量敏感性分析_过程变量模型.csv").copy()
    sens["abs_response"] = sens["增加1个历史标准差后的平均CO变化"].abs()
    sens = sens.nlargest(14, "abs_response").sort_values("增加1个历史标准差后的平均CO变化")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), constrained_layout=True)
    ax = axes[0]
    importance = importance.sort_values("综合影响强度")
    ax.barh(importance["原始字段"], importance["综合影响强度"], color=COLORS["green"], alpha=0.86)
    ax.set_xlabel("置换重要性")
    ax.set_title("A  过程变量模型关键变量重要性")

    ax = axes[1]
    colors = np.where(sens["增加1个历史标准差后的平均CO变化"] >= 0, COLORS["red"], COLORS["blue"])
    ax.barh(sens["原始字段"], sens["增加1个历史标准差后的平均CO变化"], color=colors, alpha=0.86)
    ax.axvline(0, color=COLORS["gray"], lw=1)
    ax.set_xlabel("平均 CO 变化 (mg/m$^3$)")
    ax.set_title("B  敏感性方向：增大变量后的 CO 响应")
    save(fig, "必备_07_变量重要性与敏感性.png")


def plot_ablation_and_nonlinear() -> None:
    ablation = read_csv(RESULT_DIR / "变量组消融分析.csv")
    nonlinear = read_csv(RESULT_DIR / "非线性监测模型比较.csv")
    monitor = ablation[(ablation["模型"] == "监测型模型") & (ablation["变量组"] != "完整模型")].sort_values("RMSE相对完整模型变化")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax = axes[0]
    colors = np.where(monitor["RMSE相对完整模型变化"] >= 0, COLORS["orange"], COLORS["blue"])
    ax.barh(monitor["变量组"], monitor["RMSE相对完整模型变化"], color=colors, alpha=0.88)
    ax.axvline(0, color=COLORS["gray"], lw=1)
    ax.set_xlabel("消融后 RMSE 变化")
    ax.set_title("A  监测型模型变量组消融")

    ax = axes[1]
    nonlinear = nonlinear.copy()
    nonlinear = nonlinear[(nonlinear["数据集"] == "测试集") & (nonlinear["模型"] != "基线3_过程变量+历史CO监测型模型")]
    nonlinear["短名"] = nonlinear["模型"].map(short_model_name)
    ax.bar(nonlinear["短名"], nonlinear["RMSE"], color=COLORS["purple"], alpha=0.84)
    ax.axhline(149.828, color=COLORS["orange"], ls="--", lw=1.2, label="推荐监测模型")
    ax.set_ylabel("测试 RMSE")
    ax.set_title("B  非线性监测模型未超过推荐模型")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False)
    save(fig, "可选_08_消融贡献与非线性模型对比.png")


def plot_3d_lag_search_surface() -> None:
    search = read_csv(MID_DIR / "时滞截断窗口寻优.csv")
    grouped = (
        search.groupby(["1#-6#上游风箱_最大滞后_s", "7#-12#中游风箱_最大滞后_s"], as_index=False)["稳健目标RMSE"]
        .min()
        .rename(columns={"1#-6#上游风箱_最大滞后_s": "up", "7#-12#中游风箱_最大滞后_s": "mid", "稳健目标RMSE": "rmse"})
    )
    x = grouped["up"].to_numpy()
    y = grouped["mid"].to_numpy()
    z = grouped["rmse"].to_numpy()
    xi = np.linspace(x.min(), x.max(), 90)
    yi = np.linspace(y.min(), y.max(), 90)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method="cubic")
    Z_near = griddata((x, y), z, (X, Y), method="nearest")
    Z = np.where(np.isfinite(Z), Z, Z_near)
    Z = gaussian_filter(Z, sigma=1.5)

    best = search.loc[search["稳健目标RMSE"].idxmin()]
    fig = plt.figure(figsize=(8.4, 6.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis_r", linewidth=0, antialiased=True, alpha=0.96, rcount=90, ccount=90)
    ax.scatter(
        [best["1#-6#上游风箱_最大滞后_s"]],
        [best["7#-12#中游风箱_最大滞后_s"]],
        [best["稳健目标RMSE"]],
        color=COLORS["red"],
        s=55,
        depthshade=True,
        label="最优组合",
    )
    ax.set_title("可选 3D：时滞截断窗口寻优平滑地形")
    ax.set_xlabel("上游最大滞后 (s)")
    ax.set_ylabel("中游最大滞后 (s)")
    ax.set_zlabel("稳健目标 RMSE")
    ax.view_init(elev=28, azim=-132)
    ax.legend(frameon=False)
    fig.colorbar(surf, ax=ax, shrink=0.68, pad=0.08, label="稳健目标 RMSE")
    save(fig, "可选_09_3D时滞截断窗口寻优平滑曲面.png")


def plot_3d_time_space_surface() -> None:
    clean = read_csv(MID_DIR / "清洗后数据.csv")
    time = clean["时间_s"].to_numpy() / 60
    step = 10
    time_ds = time[::step]
    boxes = np.arange(1, 19)
    temp = clean[[f"{i}#风箱平均温度" for i in boxes]].to_numpy()[::step]
    pressure = clean[[f"{i}#风箱负压强度" for i in boxes]].to_numpy()[::step]

    temp_smooth = gaussian_filter(temp, sigma=(4.2, 1.15))
    pressure_smooth = gaussian_filter(pressure, sigma=(5.0, 1.2))
    X, Y = np.meshgrid(time_ds, boxes, indexing="ij")

    fig = plt.figure(figsize=(13, 6.4), constrained_layout=True)
    for idx, (data, title, cmap, zlabel) in enumerate(
        [
            (temp_smooth, "A  风箱平均温度时空曲面", "plasma", "温度 (℃)"),
            (pressure_smooth, "B  风箱负压强度时空曲面", "cividis", "负压强度 (KPa)"),
        ],
        start=1,
    ):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        surf = ax.plot_surface(X, Y, data, cmap=cmap, linewidth=0, antialiased=True, rcount=100, ccount=18)
        ax.set_title(title)
        ax.set_xlabel("时间 (min)")
        ax.set_ylabel("风箱编号")
        ax.set_zlabel(zlabel, labelpad=10)
        ax.set_yticks([1, 6, 12, 18])
        ax.view_init(elev=31, azim=-136)
        fig.colorbar(surf, ax=ax, shrink=0.58, pad=0.08)
    save(fig, "可选_10_3D风箱状态时空平滑曲面.png")


def smooth_rbf_surface(points: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = points["风箱编号"].to_numpy(dtype=float)
    y = points["最佳滞后秒数"].to_numpy(dtype=float)
    z = points[value_col].to_numpy(dtype=float)
    xi = np.linspace(1, 18, 130)
    yi = np.linspace(0, max(300, y.max()), 90)
    X, Y = np.meshgrid(xi, yi)
    Z_num = np.zeros_like(X, dtype=float)
    Z_den = np.zeros_like(X, dtype=float)
    for px, py, pz in zip(x, y, z):
        weight = np.exp(-((X - px) ** 2 / (2 * 1.55**2) + (Y - py) ** 2 / (2 * 55.0**2)))
        Z_num += weight * pz
        Z_den += weight
    Z = Z_num / np.maximum(Z_den, 1e-12)
    return X, Y, gaussian_filter(Z, sigma=1.15)


def plot_3d_lag_influence_surface() -> None:
    sens = read_csv(RESULT_DIR / "过程变量敏感性分析_过程变量模型.csv")
    wind = sens[sens["变量组"].isin(["风箱负压强度", "风箱平均温度"])].copy()
    wind["风箱编号"] = wind["原始字段"].map(windbox_no)
    wind = wind.dropna(subset=["风箱编号"])

    fig = plt.figure(figsize=(13, 6.4), constrained_layout=True)
    for idx, (kind, title) in enumerate([("风箱负压强度", "A  负压强度-滞后-CO 响应"), ("风箱平均温度", "B  温度-滞后-CO 响应")], start=1):
        sub = wind[wind["变量组"] == kind].copy()
        X, Y, Z = smooth_rbf_surface(sub, "增加1个历史标准差后的平均CO变化")
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        norm = TwoSlopeNorm(vcenter=0, vmin=np.nanmin(Z), vmax=np.nanmax(Z))
        surf = ax.plot_surface(X, Y, Z, cmap="RdBu_r", norm=norm, linewidth=0, antialiased=True, alpha=0.97)
        ax.scatter(sub["风箱编号"], sub["最佳滞后秒数"], sub["增加1个历史标准差后的平均CO变化"], color=COLORS["dark"], s=16, depthshade=True)
        ax.set_title(title)
        ax.set_xlabel("风箱编号")
        ax.set_ylabel("最佳滞后 (s)")
        ax.set_zlabel("CO 响应")
        ax.set_xticks([1, 6, 12, 18])
        ax.view_init(elev=28, azim=-125)
        fig.colorbar(surf, ax=ax, shrink=0.58, pad=0.08, label="CO 响应")
    save(fig, "可选_11_3D风箱时滞影响平滑曲面.png")


def write_index() -> None:
    files = sorted(p.name for p in OUT_DIR.glob("*.png"))
    lines = [
        "# 第二问可视化PNG图表索引",
        "",
        "这些图基于 `data/第2问/中间数据` 与 `data/第2问/结果数据` 重新绘制，推荐优先使用 `必备_` 开头的图；`可选_` 为增强展示，其中 3D 曲面均做了网格插值/高斯平滑。",
        "",
    ]
    lines.extend(f"- `{name}`" for name in files)
    (OUT_DIR / "图表索引.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_co_cleaning_overview()
    plot_model_performance()
    plot_lag_alignment()
    plot_prediction_residual()
    plot_prediction_diagnostics()
    plot_windbox_influence_matrix()
    plot_importance_sensitivity()
    plot_ablation_and_nonlinear()
    plot_3d_lag_search_surface()
    plot_3d_time_space_surface()
    plot_3d_lag_influence_surface()
    write_index()
    print(f"生成完成：{OUT_DIR}")


if __name__ == "__main__":
    main()
