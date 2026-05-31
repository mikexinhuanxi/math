from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config


VIS_DIR = config.RESULT_DIR / "problem1_visualizations"
ABLATION_PATH = config.RESULT_DIR / "problem1_history_pressure_ablation.csv"
ABLATION_BY_BOX_PATH = config.RESULT_DIR / "problem1_history_pressure_ablation_by_box.csv"
ABLATION_PRED_PATH = config.RESULT_DIR / "problem1_history_pressure_ablation_predictions.xlsx"
PRESSURE_ONLY_FINAL_PATH = config.RESULT_DIR / "problem1_pressure_only_final.csv"
BEST_EXPERIMENT = "history_pressure_delta_lgbm"

FIG_DPI = 220
BOX_IDS = config.BOX_IDS


EXPERIMENT_LABELS = {
    "time_ordered_random_forest_base": "pressure-only RF",
    "hot_override_rel_lgbm_l1": "pressure-only improved",
    "persistence_lag1": "lag-1 persistence",
    "history_only_direct_lgbm": "history direct",
    "history_pressure_direct_lgbm": "history+pressure direct",
    "history_only_delta_lgbm": "history delta",
    "history_pressure_delta_lgbm": "history+pressure delta",
}


def setup_style() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = VIS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def box_label(box_id: int) -> str:
    return f"{box_id}#"


def short_experiment(name: str) -> str:
    return EXPERIMENT_LABELS.get(name, name)


def read_clean_data() -> pd.DataFrame:
    clean = pd.read_csv(config.CLEAN_DATA_PATH)
    return clean.sort_values(config.SEQ_COL).reset_index(drop=True)


def plot_process_overview(clean: pd.DataFrame) -> Path:
    seq = clean[config.SEQ_COL].to_numpy()
    split_seq = seq[int(len(seq) * (1.0 - config.TEST_SIZE))]
    selected_boxes = [1, 6, 12, 14, 16, 18]
    selected_pressure = [f"{i}#风箱负压" for i in selected_boxes]
    selected_temp = [f"{i}#风箱平均温度" for i in selected_boxes]
    co_col = "烧结大烟道外排CO浓度"

    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    axes[0].plot(seq, clean[config.SPEED_COL], color="#2F6B4F", linewidth=1.2)
    axes[0].set_title("工况时序总览")
    axes[0].set_ylabel("机速")

    for col, box_id in zip(selected_pressure, selected_boxes):
        axes[1].plot(seq, -clean[col], linewidth=0.9, alpha=0.85, label=box_label(box_id))
    axes[1].set_ylabel("负压绝对值 / KPa")
    axes[1].legend(ncol=6, fontsize=9, loc="upper left")

    for col, box_id in zip(selected_temp, selected_boxes):
        axes[2].plot(seq, clean[col], linewidth=0.9, alpha=0.85, label=box_label(box_id))
    axes[2].set_ylabel("平均温度 / ℃")
    axes[2].legend(ncol=6, fontsize=9, loc="upper left")

    if co_col in clean.columns:
        axes[3].plot(seq, clean[co_col], color="#8F3F2B", linewidth=1.0)
        axes[3].set_ylabel("CO / mg/m3")
    else:
        axes[3].plot(seq, -clean[config.PRESSURE_COLS].mean(axis=1), color="#8F3F2B", linewidth=1.0)
        axes[3].set_ylabel("平均负压绝对值")

    for ax in axes:
        ax.axvline(split_seq, color="#222222", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(split_seq, ax.get_ylim()[1], "后20%测试段", va="top", ha="left", fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("序列")
    return save_fig(fig, "01_process_overview.png")


def plot_spatial_profiles(clean: pd.DataFrame) -> Path:
    split = int(len(clean) * (1.0 - config.TEST_SIZE))
    train = clean.iloc[:split]
    test = clean.iloc[split:]
    x = np.array(BOX_IDS)
    pressure_cols = [f"{i}#风箱负压" for i in BOX_IDS]
    temp_cols = [f"{i}#风箱平均温度" for i in BOX_IDS]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    axes[0].plot(x, -train[pressure_cols].mean(), marker="o", linewidth=2, label="前80%")
    axes[0].plot(x, -test[pressure_cols].mean(), marker="s", linewidth=2, label="后20%")
    axes[0].set_title("各风箱平均负压绝对值")
    axes[0].set_xlabel("风箱编号")
    axes[0].set_ylabel("负压绝对值 / KPa")
    axes[0].set_xticks(x)
    axes[0].legend()

    axes[1].plot(x, train[temp_cols].mean(), marker="o", linewidth=2, label="前80%")
    axes[1].plot(x, test[temp_cols].mean(), marker="s", linewidth=2, label="后20%")
    axes[1].set_title("各风箱平均温度空间分布")
    axes[1].set_xlabel("风箱编号")
    axes[1].set_ylabel("平均温度 / ℃")
    axes[1].set_xticks(x)
    axes[1].legend()
    for ax in axes:
        ax.axvspan(15.5, 18.5, color="#E6B655", alpha=0.18, label=None)
        ax.grid(True, alpha=0.25)
    return save_fig(fig, "02_spatial_profiles.png")


def load_model_results() -> pd.DataFrame:
    pressure = pd.read_csv(PRESSURE_ONLY_FINAL_PATH)
    pressure = pressure.assign(result_group="pressure-only")
    ablation = pd.read_csv(ABLATION_PATH)
    ablation = ablation.assign(result_group="history ablation")
    common_cols = ["experiment", "result_group", "mae", "rmse", "r2", "feature_group", "feature_count"]
    for df in [pressure, ablation]:
        for col in common_cols:
            if col not in df.columns:
                df[col] = np.nan
    out = pd.concat([pressure[common_cols], ablation[common_cols]], ignore_index=True)
    out["label"] = out["experiment"].map(short_experiment)
    return out


def plot_model_comparison(results: pd.DataFrame) -> Path:
    order = results.sort_values("mae")["label"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    metrics = [("mae", "MAE 越低越好"), ("rmse", "RMSE 越低越好"), ("r2", "R2 越高越好")]
    palette = {"pressure-only": "#8C6D31", "history ablation": "#2F6B4F"}

    for idx, (ax, (metric, title)) in enumerate(zip(axes, metrics)):
        sns.barplot(
            data=results,
            y="label",
            x=metric,
            hue="result_group",
            order=order,
            dodge=False,
            palette=palette,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("")
        if idx == 0:
            ax.legend(title="", fontsize=8, loc="lower right")
        else:
            ax.legend_.remove()
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
        if metric == "r2":
            ax.axvline(0, color="#222222", linewidth=0.8)
            ax.set_xlim(results["r2"].min() - 0.16, results["r2"].max() + 0.08)

    fig.suptitle("问题一模型结果对比", y=1.01, fontsize=15)
    return save_fig(fig, "03_model_comparison.png")


def plot_by_box_heatmap() -> Path:
    by_box = pd.read_csv(ABLATION_BY_BOX_PATH)
    by_box["box_id"] = by_box["box"].str.extract(r"(\d+)").astype(int)
    by_box["label"] = by_box["experiment"].map(short_experiment)
    summary = pd.read_csv(ABLATION_PATH).sort_values("mae")
    order = [short_experiment(x) for x in summary["experiment"]]
    pivot = by_box.pivot_table(index="label", columns="box_id", values="mae").loc[order, BOX_IDS]

    fig, ax = plt.subplots(figsize=(15, 5.8))
    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "MAE"},
        ax=ax,
    )
    ax.set_title("历史温度消融实验逐风箱 MAE 热力图")
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("")
    return save_fig(fig, "04_ablation_by_box_mae_heatmap.png")


def plot_by_box_lines() -> Path:
    by_box = pd.read_csv(ABLATION_BY_BOX_PATH)
    by_box["box_id"] = by_box["box"].str.extract(r"(\d+)").astype(int)
    keep = ["persistence_lag1", "history_only_delta_lgbm", "history_pressure_delta_lgbm"]
    plot_df = by_box[by_box["experiment"].isin(keep)].copy()
    plot_df["label"] = plot_df["experiment"].map(short_experiment)

    fig, ax = plt.subplots(figsize=(14, 5.2))
    sns.lineplot(data=plot_df, x="box_id", y="mae", hue="label", marker="o", linewidth=2, ax=ax)
    ax.axvspan(11.5, 15.5, color="#C65742", alpha=0.12)
    ax.set_title("关键模型逐风箱误差曲线")
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("MAE")
    ax.set_xticks(BOX_IDS)
    ax.legend(title="")
    ax.grid(True, alpha=0.25)
    return save_fig(fig, "05_key_models_by_box_mae.png")


def read_prediction_sheet(sheet_name: str = BEST_EXPERIMENT) -> pd.DataFrame:
    return pd.read_excel(ABLATION_PRED_PATH, sheet_name=sheet_name)


def plot_best_prediction_tracking(pred: pd.DataFrame) -> Path:
    selected_boxes = [1, 6, 12, 14, 16, 18]
    seq = pred["序列"]
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()
    for ax, box_id in zip(axes, selected_boxes):
        actual = pred[f"{box_id}#实际"]
        forecast = pred[f"{box_id}#预测"]
        ax.plot(seq, actual, color="#253858", linewidth=1.2, label="实际")
        ax.plot(seq, forecast, color="#D65F28", linewidth=1.1, linestyle="--", label="预测")
        ax.fill_between(seq, actual, forecast, color="#D65F28", alpha=0.12, linewidth=0)
        ax.set_title(f"{box_id}#风箱")
        ax.set_ylabel("平均温度 / ℃")
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=2, fontsize=9, loc="upper left")
    axes[-1].set_xlabel("序列")
    axes[-2].set_xlabel("序列")
    fig.suptitle("最优模型后20%样本预测跟踪", y=1.02, fontsize=15)
    return save_fig(fig, "06_best_prediction_tracking.png")


def prediction_long(pred: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for box_id in BOX_IDS:
        frames.append(
            pd.DataFrame(
                {
                    "sequence": pred["序列"],
                    "box_id": box_id,
                    "actual": pred[f"{box_id}#实际"],
                    "predicted": pred[f"{box_id}#预测"],
                    "residual": pred[f"{box_id}#残差_预测减实际"],
                }
            )
        )
    long = pd.concat(frames, ignore_index=True)
    long["abs_residual"] = long["residual"].abs()
    return long


def plot_prediction_diagnostics(pred: pd.DataFrame) -> Path:
    long = prediction_long(pred)
    lim_min = min(long["actual"].min(), long["predicted"].min())
    lim_max = max(long["actual"].max(), long["predicted"].max())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    sns.scatterplot(
        data=long,
        x="actual",
        y="predicted",
        hue="box_id",
        palette="viridis",
        s=10,
        linewidth=0,
        alpha=0.45,
        legend=False,
        ax=axes[0],
    )
    axes[0].plot([lim_min, lim_max], [lim_min, lim_max], color="#A21D2B", linewidth=1.2)
    axes[0].set_title("实际值 vs 预测值")
    axes[0].set_xlabel("实际平均温度 / ℃")
    axes[0].set_ylabel("预测平均温度 / ℃")

    sns.histplot(long["residual"], bins=60, kde=True, color="#557C83", ax=axes[1])
    axes[1].axvline(0, color="#A21D2B", linewidth=1)
    axes[1].set_title("残差分布")
    axes[1].set_xlabel("预测 - 实际 / ℃")
    axes[1].set_ylabel("样本数")

    sns.boxplot(data=long, x="box_id", y="residual", color="#C9D6DF", fliersize=1.5, ax=axes[2])
    axes[2].axhline(0, color="#A21D2B", linewidth=1)
    axes[2].set_title("逐风箱残差箱线图")
    axes[2].set_xlabel("风箱编号")
    axes[2].set_ylabel("预测 - 实际 / ℃")
    return save_fig(fig, "07_best_prediction_diagnostics.png")


def categorize_feature(feature: str) -> str:
    lower = feature.lower()
    if "temp" in lower or "target" in lower:
        return "history temperature"
    if lower.startswith("speed"):
        return "machine speed"
    if any(token in lower for token in ["lag", "delta", "rate", "roll"]):
        return "temporal pressure"
    if any(token in lower for token in ["front", "middle", "back", "long", "short", "all"]):
        return "segment pressure"
    if any(token in lower for token in ["neighbor", "adjacent", "local"]):
        return "spatial pressure"
    if lower.startswith("p_"):
        return "raw pressure"
    return "other"


def plot_feature_importance() -> Path:
    importance = pd.read_csv(config.FEATURE_IMPORTANCE_PATH)
    top = importance.head(25).copy()
    top["feature"] = top["feature"].astype(str)
    top["importance_percent"] = top["importance_ratio"] * 100.0
    top["category"] = top["feature"].map(categorize_feature)

    category = importance.copy()
    category["category"] = category["feature"].astype(str).map(categorize_feature)
    category = category.groupby("category", as_index=False)["importance_ratio"].sum()
    category["importance_percent"] = category["importance_ratio"] * 100.0
    category = category.sort_values("importance_percent", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [1.45, 1.0]})
    sns.barplot(
        data=top.sort_values("importance_percent"),
        y="feature",
        x="importance_percent",
        hue="category",
        dodge=False,
        ax=axes[0],
    )
    axes[0].set_title("Top 25 特征重要性")
    axes[0].set_xlabel("重要性占比 / %")
    axes[0].set_ylabel("")
    axes[0].legend(title="", fontsize=8, loc="lower right")

    sns.barplot(
        data=category,
        y="category",
        x="importance_percent",
        color="#547A8A",
        ax=axes[1],
    )
    axes[1].set_title("特征类别重要性汇总")
    axes[1].set_xlabel("重要性占比 / %")
    axes[1].set_ylabel("")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f%%", fontsize=8, padding=3)
    return save_fig(fig, "08_feature_importance.png")


def plot_dashboard(results: pd.DataFrame, pred: pd.DataFrame) -> Path:
    by_box = pd.read_csv(ABLATION_BY_BOX_PATH)
    by_box["box_id"] = by_box["box"].str.extract(r"(\d+)").astype(int)
    key = by_box[by_box["experiment"].eq(BEST_EXPERIMENT)]
    long = prediction_long(pred)
    importance = pd.read_csv(config.FEATURE_IMPORTANCE_PATH).head(10).copy()
    importance["importance_percent"] = importance["importance_ratio"] * 100.0

    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])

    sns.barplot(data=results.sort_values("mae"), y="label", x="mae", hue="result_group", dodge=False, ax=ax1)
    ax1.set_title("模型 MAE 排名")
    ax1.set_xlabel("MAE")
    ax1.set_ylabel("")
    ax1.legend(title="", fontsize=8)

    sns.lineplot(data=key, x="box_id", y="mae", marker="o", linewidth=2, color="#D65F28", ax=ax2)
    ax2.set_title("最优模型逐风箱 MAE")
    ax2.set_xlabel("风箱编号")
    ax2.set_ylabel("MAE")
    ax2.set_xticks(BOX_IDS)

    sns.scatterplot(data=long, x="actual", y="predicted", s=9, linewidth=0, alpha=0.35, color="#376D7B", ax=ax3)
    lim_min = min(long["actual"].min(), long["predicted"].min())
    lim_max = max(long["actual"].max(), long["predicted"].max())
    ax3.plot([lim_min, lim_max], [lim_min, lim_max], color="#A21D2B", linewidth=1)
    ax3.set_title("最优模型实际值 vs 预测值")
    ax3.set_xlabel("实际平均温度 / ℃")
    ax3.set_ylabel("预测平均温度 / ℃")

    sns.barplot(
        data=importance.sort_values("importance_percent"),
        y="feature",
        x="importance_percent",
        color="#6F8F4E",
        ax=ax4,
    )
    ax4.set_title("Top 10 特征重要性")
    ax4.set_xlabel("重要性占比 / %")
    ax4.set_ylabel("")
    fig.suptitle("问题一结果可视化总览", y=1.01, fontsize=16)
    return save_fig(fig, "00_problem1_dashboard.png")


def write_index(paths: list[Path], results: pd.DataFrame) -> Path:
    best = results.sort_values("mae").iloc[0]
    rel_lines = "\n".join([f"- `{path.name}`" for path in paths])
    text = f"""# 问题一可视化图表索引

## 数据口径
- 题目问题一：探索各风箱负压与对应风箱温度间的关系，并构建基于风箱负压的温度预测模型。
- 图表读取现有导出文件，不重新训练模型。
- 当前综合 MAE 最低的导出结果：`{best["experiment"]}`，MAE = {best["mae"]:.4f}。

## 图表文件
{rel_lines}

## 使用建议
- 论文正文优先使用 `00_problem1_dashboard.png`、`03_model_comparison.png`、`04_ablation_by_box_mae_heatmap.png`、`06_best_prediction_tracking.png`。
- 附录或结果分析可补充 `01_process_overview.png`、`02_spatial_profiles.png`、`07_best_prediction_diagnostics.png`、`08_feature_importance.png`。
"""
    path = VIS_DIR / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    setup_style()
    clean = read_clean_data()
    results = load_model_results()
    pred = read_prediction_sheet(BEST_EXPERIMENT)

    paths = [
        plot_dashboard(results, pred),
        plot_process_overview(clean),
        plot_spatial_profiles(clean),
        plot_model_comparison(results),
        plot_by_box_heatmap(),
        plot_by_box_lines(),
        plot_best_prediction_tracking(pred),
        plot_prediction_diagnostics(pred),
        plot_feature_importance(),
    ]
    index_path = write_index(paths, results)

    print("已生成问题一可视化：")
    for path in paths:
        print(path)
    print(index_path)


if __name__ == "__main__":
    main()
