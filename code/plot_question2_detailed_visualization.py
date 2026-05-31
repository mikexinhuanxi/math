from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
Q2 = "第2问"
DATA_DIR = ROOT / "data" / Q2 / "结果数据"
RESULT_DIR = ROOT / "result" / Q2
OUT_DIR = ROOT / "figure" / Q2 / "数据驱动综合可视化"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUT_DIR / "问题二_详细模型评价与影响规律综合图.png"
SVG_PATH = OUT_DIR / "问题二_详细模型评价与影响规律综合图.svg"


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial",
    "DejaVu Sans",
    "sans-serif",
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 7.4
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.65
plt.rcParams["xtick.major.width"] = 0.55
plt.rcParams["ytick.major.width"] = 0.55
plt.rcParams["legend.frameon"] = False


PALETTE = {
    "blue": "#0F4D92",
    "blue_mid": "#3775BA",
    "teal": "#42949E",
    "orange": "#D88C51",
    "gold": "#C9A227",
    "red": "#B64342",
    "green": "#2E9E44",
    "gray_dark": "#4D4D4D",
    "gray_mid": "#8B8B8B",
    "gray_light": "#D8D8D8",
    "ink": "#272727",
}


MODEL_LABELS = {
    "过程变量模型": "过程变量主模型",
    "线性过程变量模型_对比": "线性过程",
    "非线性过程变量模型": "非线性过程",
    "基线1_CO_lag1持续性模型": "CO持续性",
    "基线2_历史CO_AR模型": "历史CO-AR",
    "基线3_过程变量+历史CO监测型模型": "监测融合",
    "监测型模型": "监测融合",
    "残差校正监测型模型_对比": "残差校正监测",
    "非线性监测型模型": "非线性监测",
    "过程变量_LightGBM": "过程LGBM",
    "过程变量_XGBoost": "过程XGB",
    "过程变量_梯度提升树": "过程GBRT",
    "过程变量_随机森林": "过程RF",
    "非线性监测_LightGBM": "监测LGBM",
    "非线性监测_XGBoost": "监测XGB",
    "非线性监测_梯度提升树": "监测GBRT",
    "非线性监测_随机森林": "监测RF",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def short_model(name: str) -> str:
    return MODEL_LABELS.get(name, name.replace("_", "\n"))


def model_family(name: str) -> str:
    if "非线性监测" in name:
        return "非线性监测"
    if "监测" in name or "历史CO" in name or "lag1" in name:
        return "历史CO/监测"
    if "过程变量" in name:
        return "过程变量"
    return "其他"


FAMILY_COLORS = {
    "过程变量": PALETTE["orange"],
    "历史CO/监测": PALETTE["blue"],
    "非线性监测": PALETTE["teal"],
    "其他": PALETTE["gray_mid"],
}


def family_color(model: str) -> str:
    return FAMILY_COLORS[model_family(model)]


def panel_label(ax, letter: str, x=-0.08, y=1.04) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=PALETTE["ink"],
    )


def soften_axes(ax) -> None:
    ax.grid(axis="x", color="#ECECEC", lw=0.55, zorder=0)
    ax.tick_params(length=2.2, color="#777777", labelcolor=PALETTE["ink"])


def format_feature(name: str, lag=None) -> str:
    label = str(name)
    replacements = {
        "烧结机机速L1设定": "机速",
        "风箱平均温度": "风箱温度",
        "风箱负压强度": "风箱负压",
        "大烟道负压强度": "烟道负压",
        "大烟道温度": "烟道温度",
        "CO历史浓度": "历史CO",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    label = label.replace("__lag", " lag")
    if lag is not None and not pd.isna(lag):
        label = f"{label} ({lag:.0f}s)"
    return label


def quality_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    good_high = {"R2", "趋势准确率_%", "5%误差内比例_%", "10%误差内比例_%"}
    out = pd.DataFrame(index=df.index)
    for col in cols:
        vals = df[col].astype(float)
        span = vals.max() - vals.min()
        if span == 0:
            out[col] = 0.5
        elif col in good_high:
            out[col] = (vals - vals.min()) / span
        else:
            out[col] = 1 - (vals - vals.min()) / span
    return out


def annotate_barh(ax, values, y_positions, fmt="{:.1f}", pad=0.02, color=PALETTE["ink"]):
    xmin, xmax = ax.get_xlim()
    dx = (xmax - xmin) * pad
    for val, y in zip(values, y_positions):
        ha = "left" if val >= 0 else "right"
        ax.text(val + (dx if val >= 0 else -dx), y, fmt.format(val), va="center", ha=ha, fontsize=6.6, color=color)


def build_figure() -> None:
    dual = read_csv(RESULT_DIR / "双口径模型评价对比.csv")
    metrics = read_csv(DATA_DIR / "模型评价指标.csv")
    predictions = read_csv(DATA_DIR / "CO预测结果_全样本.csv")
    proc_importance = read_csv(DATA_DIR / "变量影响强度_过程变量模型.csv")
    monitor_importance = read_csv(DATA_DIR / "变量影响强度_监测型模型.csv")
    ablation = read_csv(DATA_DIR / "变量组消融分析.csv")
    lag_validation = read_csv(DATA_DIR / "时滞合理性验证.csv")
    pdp = read_csv(DATA_DIR / "关键变量偏依赖曲线.csv")

    test_metrics = metrics[metrics["数据集"] == "测试集"].copy()
    test_metrics["短名"] = test_metrics["模型"].map(short_model)
    test_metrics["家族"] = test_metrics["模型"].map(model_family)
    test_predictions = predictions[predictions["数据集"] == "测试集"].copy()

    fig = plt.figure(figsize=(24, 17.8), dpi=150)
    gs = gridspec.GridSpec(
        4,
        4,
        figure=fig,
        height_ratios=[1.25, 1.15, 1.05, 1.05],
        width_ratios=[1.16, 1.02, 1.0, 1.0],
        hspace=0.58,
        wspace=0.42,
    )

    fig.suptitle(
        "问题二：CO实时预测模型评价与过程变量影响规律",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.text(
        0.02,
        0.958,
        "核心结论：异常修正显著提升测试可靠性；实时预测应以历史CO+过程变量监测融合为主，过程变量模型主要用于解释时滞与影响方向。",
        ha="left",
        fontsize=9.2,
        color=PALETTE["gray_dark"],
    )

    # a. Dual-definition RMSE change.
    ax_a = fig.add_subplot(gs[0, 0:2])
    rmse = dual.pivot(index="模型", columns="建模口径", values="测试RMSE")
    rmse = rmse.sort_values("异常修正后建模", ascending=True)
    y = np.arange(len(rmse))
    for i, model in enumerate(rmse.index):
        x_raw = rmse.loc[model, "原始数据建模"]
        x_corr = rmse.loc[model, "异常修正后建模"]
        ax_a.plot([x_raw, x_corr], [i, i], color="#BDBDBD", lw=1.1, zorder=1)
        ax_a.scatter(x_raw, i, s=50, color="#C9C9C9", edgecolor="white", lw=0.6, zorder=2)
        ax_a.scatter(x_corr, i, s=62, color=family_color(model), edgecolor="white", lw=0.7, zorder=3)
        delta = (x_raw - x_corr) / x_raw * 100
        ax_a.text(max(x_raw, x_corr) + 13, i, f"{delta:+.0f}%", va="center", fontsize=6.5, color=PALETTE["gray_dark"])
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([short_model(m) for m in rmse.index])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("测试RMSE（mg/m3，越低越好）")
    ax_a.set_title("双口径对比：异常修正后 RMSE 变化", loc="left", fontsize=10, fontweight="bold")
    ax_a.set_xlim(120, max(rmse.max()) + 80)
    ax_a.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#C9C9C9", label="原始数据建模", markersize=6),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["blue"], label="异常修正后建模", markersize=6),
            Patch(facecolor=PALETTE["orange"], label="过程变量类"),
            Patch(facecolor=PALETTE["teal"], label="非线性监测类"),
        ],
        loc="lower right",
        ncol=2,
        fontsize=6.5,
        handlelength=1.2,
        columnspacing=0.8,
    )
    soften_axes(ax_a)
    panel_label(ax_a, "a")

    # b. Corrected-model performance matrix.
    ax_b = fig.add_subplot(gs[0, 2:4])
    matrix_cols = ["RMSE", "MAPE_%", "R2", "趋势准确率_%", "10%误差内比例_%", "高CO_RMSE"]
    heat_df = test_metrics.sort_values("RMSE", ascending=True).reset_index(drop=True)
    q = quality_matrix(heat_df, matrix_cols)
    cmap = LinearSegmentedColormap.from_list("quality", ["#F6CFCB", "#FFFFFF", "#0F4D92"])
    im = ax_b.imshow(q[matrix_cols].values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    for i in range(len(heat_df)):
        for j, col in enumerate(matrix_cols):
            val = heat_df.loc[i, col]
            txt = f"{val:.2f}" if col == "R2" else (f"{val:.0f}" if "RMSE" in col else f"{val:.1f}")
            ax_b.text(j, i, txt, ha="center", va="center", fontsize=5.8, color=PALETTE["ink"])
    ax_b.set_xticks(np.arange(len(matrix_cols)))
    ax_b.set_xticklabels(["RMSE↓", "MAPE↓", "R2↑", "趋势↑", "10%内↑", "高CO RMSE↓"], rotation=0)
    ax_b.set_yticks(np.arange(len(heat_df)))
    ax_b.set_yticklabels(heat_df["短名"], fontsize=6.1)
    ax_b.set_title("异常修正后测试集：多指标相对质量矩阵", loc="left", fontsize=10, fontweight="bold")
    ax_b.tick_params(length=0)
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.035, pad=0.018)
    cbar.set_label("相对质量", fontsize=6.4)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    panel_label(ax_b, "b", x=-0.11)

    # c. Test-set time series.
    ax_c = fig.add_subplot(gs[1, 0:3])
    x = (test_predictions["时间_s"] - test_predictions["时间_s"].min()) / 60.0
    actual_col = "烧结大烟道外排CO浓度"
    selected_series = [
        ("实际CO", actual_col, PALETTE["ink"], 1.6, 1.0),
        ("监测融合", "监测型模型_预测CO浓度", PALETTE["blue"], 1.15, 0.95),
        ("历史CO-AR", "基线2_历史CO_AR模型_预测CO浓度", PALETTE["gray_dark"], 0.95, 0.82),
        ("过程变量主模型", "过程变量模型_预测CO浓度", PALETTE["orange"], 0.95, 0.85),
        ("监测GBRT", "非线性监测_梯度提升树_预测CO浓度", PALETTE["teal"], 0.9, 0.82),
    ]
    for label, col, color, lw, alpha in selected_series:
        ax_c.plot(x, test_predictions[col], lw=lw, color=color, alpha=alpha, label=label)
    high_threshold = float(test_metrics["高CO阈值_训练90分位"].dropna().iloc[0])
    ax_c.axhline(high_threshold, color=PALETTE["red"], lw=0.8, ls=(0, (4, 2)), alpha=0.75)
    ax_c.fill_between(
        x,
        high_threshold,
        test_predictions[actual_col],
        where=test_predictions[actual_col] >= high_threshold,
        color=PALETTE["red"],
        alpha=0.09,
        linewidth=0,
    )
    ax_c.text(x.max(), high_threshold + 22, "训练90分位高CO阈值", ha="right", va="bottom", fontsize=6.6, color=PALETTE["red"])
    ax_c.set_ylabel("CO浓度（mg/m3）")
    ax_c.set_xlabel("测试集时间（min）")
    ax_c.set_title("测试集预测时序：监测融合跟随真实波动，纯过程变量偏离较大", loc="left", fontsize=10, fontweight="bold")
    ax_c.legend(loc="upper left", ncol=5, fontsize=6.4, handlelength=1.6, columnspacing=0.8)
    soften_axes(ax_c)
    panel_label(ax_c, "c", x=-0.05)

    # d. Absolute residual distributions.
    ax_d = fig.add_subplot(gs[1, 3])
    residual_map = [
        ("监测融合", "监测型模型_残差", PALETTE["blue"]),
        ("历史CO-AR", "基线2_历史CO_AR模型_残差", PALETTE["gray_dark"]),
        ("监测GBRT", "非线性监测_梯度提升树_残差", PALETTE["teal"]),
        ("过程变量", "过程变量模型_残差", PALETTE["orange"]),
        ("非线性过程", "非线性过程变量模型_残差", PALETTE["gold"]),
    ]
    abs_res = [np.abs(test_predictions[col].dropna().values) for _, col, _ in residual_map]
    bplot = ax_d.boxplot(abs_res, vert=False, patch_artist=True, labels=[m[0] for m in residual_map], widths=0.6, showfliers=False)
    for patch, (_, _, color) in zip(bplot["boxes"], residual_map):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor("white")
    for key in ["medians", "whiskers", "caps"]:
        for artist in bplot[key]:
            artist.set_color(PALETTE["ink"])
            artist.set_linewidth(0.75)
    medians = [np.median(v) for v in abs_res]
    for i, med in enumerate(medians, 1):
        ax_d.text(med + 10, i, f"{med:.0f}", va="center", fontsize=6.4, color=PALETTE["ink"])
    ax_d.set_xlabel("|残差|（mg/m3）")
    ax_d.set_title("绝对误差分布", loc="left", fontsize=10, fontweight="bold")
    soften_axes(ax_d)
    panel_label(ax_d, "d", x=-0.18)

    # e. High-CO robustness.
    ax_e = fig.add_subplot(gs[2, 0])
    for fam, part in test_metrics.groupby("家族"):
        ax_e.scatter(
            part["RMSE"],
            part["高CO_RMSE"],
            s=42,
            color=FAMILY_COLORS[fam],
            edgecolor="white",
            lw=0.6,
            label=fam,
            alpha=0.9,
            zorder=3,
        )
    lim_max = max(test_metrics["RMSE"].max(), test_metrics["高CO_RMSE"].max()) + 35
    ax_e.plot([0, lim_max], [0, lim_max], color="#C8C8C8", lw=0.8, ls="--", zorder=1)
    for model in ["基线3_过程变量+历史CO监测型模型", "残差校正监测型模型_对比", "过程变量模型", "非线性监测_梯度提升树"]:
        row = test_metrics[test_metrics["模型"] == model].iloc[0]
        ax_e.text(row["RMSE"] + 8, row["高CO_RMSE"] + 6, short_model(model), fontsize=6.2, color=PALETTE["ink"])
    ax_e.set_xlabel("全测试RMSE")
    ax_e.set_ylabel("高CO RMSE")
    ax_e.set_title("高CO鲁棒性：监测融合同时压低整体与高CO误差", loc="left", fontsize=9.4, fontweight="bold")
    ax_e.legend(loc="lower right", fontsize=6.2)
    ax_e.set_xlim(120, 570)
    ax_e.set_ylim(120, 870)
    soften_axes(ax_e)
    panel_label(ax_e, "e", x=-0.14)

    # f. Tolerance and trend.
    ax_f = fig.add_subplot(gs[2, 1])
    focus_models = [
        "基线3_过程变量+历史CO监测型模型",
        "基线2_历史CO_AR模型",
        "非线性监测_梯度提升树",
        "过程变量模型",
    ]
    focus = test_metrics.set_index("模型").loc[focus_models].reset_index()
    x_pos = np.arange(len(focus))
    width = 0.24
    bars1 = ax_f.bar(x_pos - width, focus["5%误差内比例_%"], width, color="#B4C0E4", label="5%误差内")
    bars2 = ax_f.bar(x_pos, focus["10%误差内比例_%"], width, color=PALETTE["blue_mid"], label="10%误差内")
    bars3 = ax_f.bar(x_pos + width, focus["趋势准确率_%"], width, color=PALETTE["teal"], label="趋势准确率")
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax_f.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.1, f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=5.8)
    ax_f.set_xticks(x_pos)
    ax_f.set_xticklabels([short_model(m) for m in focus["模型"]], rotation=18, ha="right")
    ax_f.set_ylim(0, 105)
    ax_f.set_ylabel("%")
    ax_f.set_title("误差容差与趋势判别能力", loc="left", fontsize=9.4, fontweight="bold")
    ax_f.legend(loc="upper left", fontsize=6.0)
    soften_axes(ax_f)
    panel_label(ax_f, "f", x=-0.14)

    # g. Process-only variable influence.
    ax_g = fig.add_subplot(gs[2, 2])
    top_proc = proc_importance.sort_values("综合影响强度", ascending=False).head(10).iloc[::-1]
    proc_labels = [format_feature(r["原始字段"], r["最佳滞后秒数"]) for _, r in top_proc.iterrows()]
    proc_values = top_proc["综合影响强度"].values
    proc_colors = [PALETTE["orange"] if "负压" in str(v) else (PALETTE["gold"] if "温度" in str(v) else PALETTE["gray_mid"]) for v in top_proc["原始字段"]]
    ax_g.barh(np.arange(len(top_proc)), proc_values, color=proc_colors, edgecolor="white", lw=0.5)
    ax_g.set_yticks(np.arange(len(top_proc)))
    ax_g.set_yticklabels(proc_labels, fontsize=5.9)
    ax_g.set_xlabel("综合影响强度")
    ax_g.set_title("纯过程变量模型：主要过程特征", loc="left", fontsize=9.4, fontweight="bold")
    ax_g.set_xlim(0, proc_values.max() * 1.22)
    annotate_barh(ax_g, proc_values, np.arange(len(top_proc)), fmt="{:.3f}", pad=0.025)
    soften_axes(ax_g)
    panel_label(ax_g, "g", x=-0.14)

    # h. Monitoring-model influence.
    ax_h = fig.add_subplot(gs[2, 3])
    top_mon = monitor_importance.sort_values("综合影响强度", ascending=False).head(10).iloc[::-1]
    mon_labels = [format_feature(r["特征名"]) for _, r in top_mon.iterrows()]
    mon_values = top_mon["综合影响强度"].values
    mon_colors = [PALETTE["blue"] if "CO历史" in str(v) or "历史CO" in str(v) else PALETTE["teal"] for v in top_mon["特征名"]]
    ax_h.barh(np.arange(len(top_mon)), mon_values, color=mon_colors, edgecolor="white", lw=0.5)
    ax_h.set_yticks(np.arange(len(top_mon)))
    ax_h.set_yticklabels(mon_labels, fontsize=5.9)
    ax_h.set_xlabel("综合影响强度")
    ax_h.set_title("监测融合模型：历史CO占主导", loc="left", fontsize=9.4, fontweight="bold")
    ax_h.set_xlim(0, mon_values.max() * 1.17)
    annotate_barh(ax_h, mon_values, np.arange(len(top_mon)), fmt="{:.1f}", pad=0.018)
    soften_axes(ax_h)
    panel_label(ax_h, "h", x=-0.14)

    # i. Ablation by variable group.
    ax_i = fig.add_subplot(gs[3, 0:2])
    abl = ablation[ablation["变量组"] != "完整模型"].copy()
    group_order = ["历史CO", "大烟道负压强度", "大烟道温度", "机速", "风箱负压强度", "风箱平均温度"]
    y_pos = np.arange(len(group_order))
    offsets = {"过程变量模型": -0.17, "监测型模型": 0.17}
    colors = {"过程变量模型": PALETTE["orange"], "监测型模型": PALETTE["blue"]}
    for model, off in offsets.items():
        vals = []
        for group in group_order:
            row = abl[(abl["模型"] == model) & (abl["变量组"] == group)]
            vals.append(float(row["RMSE相对完整模型变化"].iloc[0]) if len(row) else np.nan)
        vals = np.array(vals, dtype=float)
        mask = ~np.isnan(vals)
        ax_i.barh(y_pos[mask] + off, vals[mask], height=0.28, color=colors[model], label=model, edgecolor="white", lw=0.5)
        annotate_barh(ax_i, vals[mask], y_pos[mask] + off, fmt="{:+.1f}", pad=0.01)
    ax_i.axvline(0, color=PALETTE["ink"], lw=0.7)
    ax_i.set_yticks(y_pos)
    ax_i.set_yticklabels(group_order)
    ax_i.set_xlabel("消融后测试RMSE相对完整模型变化（mg/m3）")
    ax_i.set_title("变量组消融：历史CO是监测预测的关键支撑，过程变量贡献分散", loc="left", fontsize=9.6, fontweight="bold")
    ax_i.legend(loc="lower right", fontsize=6.4)
    ax_i.set_xlim(-28, 405)
    soften_axes(ax_i)
    panel_label(ax_i, "i", x=-0.06)

    # j. Lag-validation heatmap.
    ax_j = fig.add_subplot(gs[3, 2])
    scenario_order = ["无时滞模型_lag0", "物理初始时滞截断_对比模型", "分组时滞组合优化_主模型", "单变量局部搜索_对比模型"]
    model_order = ["过程变量模型", "监测型模型", "非线性监测型模型"]
    lag_piv = lag_validation.pivot(index="模型类型", columns="实验场景", values="RMSE相对无时滞降低率_%").loc[model_order, scenario_order]
    rmse_piv = lag_validation.pivot(index="模型类型", columns="实验场景", values="RMSE").loc[model_order, scenario_order]
    norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=50)
    lag_cmap = LinearSegmentedColormap.from_list("lag", [PALETTE["red"], "#FFFFFF", PALETTE["blue"]])
    ax_j.imshow(lag_piv.values, cmap=lag_cmap, norm=norm, aspect="auto")
    for i in range(lag_piv.shape[0]):
        for j in range(lag_piv.shape[1]):
            ax_j.text(j, i, f"{lag_piv.iloc[i, j]:+.1f}%\n{rmse_piv.iloc[i, j]:.0f}", ha="center", va="center", fontsize=6.1, color=PALETTE["ink"])
    ax_j.set_xticks(np.arange(len(scenario_order)))
    ax_j.set_xticklabels(["lag0", "物理截断", "分组优化", "单变量"], rotation=25, ha="right")
    ax_j.set_yticks(np.arange(len(model_order)))
    ax_j.set_yticklabels(["过程变量", "监测融合", "非线性监测"], fontsize=6.6)
    ax_j.set_title("时滞策略验证\n单元格：RMSE降低率 / RMSE", loc="left", fontsize=9.4, fontweight="bold")
    ax_j.tick_params(length=0)
    for spine in ax_j.spines.values():
        spine.set_visible(False)
    panel_label(ax_j, "j", x=-0.18)

    # k. Partial-dependence response.
    ax_k = fig.add_subplot(gs[3, 3])
    pdp_colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["gold"]]
    for color, (var, part) in zip(pdp_colors, pdp.groupby("变量")):
        part = part.sort_values("取值")
        x_norm = (part["取值"] - part["取值"].min()) / (part["取值"].max() - part["取值"].min())
        ax_k.plot(x_norm, part["相对训练均值CO变化"], lw=1.15, color=color, label=format_feature(var))
    ax_k.axhline(0, color=PALETTE["ink"], lw=0.65)
    ax_k.set_xlabel("变量取值分位化位置")
    ax_k.set_ylabel("相对训练均值CO变化")
    ax_k.set_title("关键变量偏依赖：方向与非线性响应", loc="left", fontsize=9.4, fontweight="bold")
    ax_k.legend(loc="best", fontsize=5.9, handlelength=1.2)
    soften_axes(ax_k)
    panel_label(ax_k, "k", x=-0.14)

    fig.text(
        0.02,
        0.018,
        "数据说明：测试集 n=367；高CO阈值为训练集CO 90分位（3989.077 mg/m3）。所有面板由问题二评价、预测明细、变量重要性、消融、时滞验证和偏依赖源数据重新绘制。",
        ha="left",
        fontsize=7.2,
        color=PALETTE["gray_dark"],
    )

    fig.savefig(SVG_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=320, bbox_inches="tight")
    plt.close(fig)

    with Image.open(PNG_PATH) as img:
        arr = np.asarray(img.convert("RGB"))
        if arr.std() < 1:
            raise RuntimeError("Generated PNG appears blank.")
        print(f"PNG: {PNG_PATH}")
        print(f"SVG: {SVG_PATH}")
        print(f"size: {img.size[0]}x{img.size[1]} px")
        print(f"pixel_std: {arr.std():.3f}")


if __name__ == "__main__":
    build_figure()
