from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "第一问"
RESULT_DIR = PROJECT_ROOT / "result" / "第一问" / "时间划分数据集+历史温度"
FIG_DIR = PROJECT_ROOT / "figure" / "第一问" / "可视化PNG"

CLEAN_PATH = DATA_DIR / "problem1_clean.csv"
ABLATION_PATH = RESULT_DIR / "problem1_history_pressure_ablation.csv"
ABLATION_BY_BOX_PATH = RESULT_DIR / "problem1_history_pressure_ablation_by_box.csv"
PREDICTION_PATH = RESULT_DIR / "problem1_history_pressure_ablation_predictions.xlsx"
PRESSURE_ONLY_PATH = RESULT_DIR / "problem1_pressure_only_final.csv"
IMPORTANCE_PATH = RESULT_DIR / "problem1_feature_importance.csv"

SEQ_COL = "序列"
SPEED_COL = "烧结机机速L1设定"
BOX_IDS = list(range(1, 19))
PRESSURE_COLS = [f"{i}#风箱负压" for i in BOX_IDS]
TEMP_COLS = [f"{i}#风箱平均温度" for i in BOX_IDS]
BEST_EXPERIMENT = "history_pressure_delta_lgbm"
FIG_DPI = 300

EXPERIMENT_LABELS = {
    "time_ordered_random_forest_base": "仅负压 RF",
    "hot_override_rel_lgbm_l1": "仅负压改进",
    "persistence_lag1": "上一时刻温度",
    "history_only_direct_lgbm": "历史温度-直接",
    "history_pressure_direct_lgbm": "历史温度+负压-直接",
    "history_only_delta_lgbm": "历史温度-增量",
    "history_pressure_delta_lgbm": "历史温度+负压-增量",
}

FIGURE_NOTES = [
    ("必备_01_数据时空总览.png", "说明第一问输入负压、目标温度和时间测试段。"),
    ("必备_02_负压温度空间剖面.png", "说明沿烧结机方向的负压强度和温度梯度。"),
    ("必备_03_负压温度时滞关系.png", "直接回答负压与温度关系及时滞性。"),
    ("必备_04_模型性能对比.png", "支撑最终模型选择，并保留 pressure-only 对照。"),
    ("必备_05_预测跟踪与一致性.png", "展示最优模型实际值、预测值的一致性。"),
    ("必备_06_逐风箱误差与残差诊断.png", "定位不同风箱的预测难度和残差时间段。"),
    ("可选_07_特征重要性结构.png", "解释模型依赖哪些负压、时序、空间派生特征。"),
    ("可选_08_历史温度与负压消融贡献.png", "量化加入负压后的边际改进。"),
    ("可选_09_3D温度时空平滑曲面.png", "平滑展示目标温度的时空形态。"),
    ("可选_10_3D负压时空平滑曲面.png", "平滑展示负压输入的时空形态。"),
    ("可选_11_3D预测残差平滑曲面.png", "平滑展示测试段残差的时空形态。"),
]


def setup_style() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
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
    plt.rcParams["axes.titleweight"] = "bold"


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"缺少必要文件: {path}")
    return path


def save_fig(fig: plt.Figure, filename: str) -> Path:
    path = FIG_DIR / filename
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def experiment_label(name: str) -> str:
    return EXPERIMENT_LABELS.get(str(name), str(name))


def parse_box_id(value: str) -> int:
    match = re.search(r"(\d+)", str(value))
    if not match:
        raise ValueError(f"无法解析风箱编号: {value}")
    return int(match.group(1))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean = pd.read_csv(require_path(CLEAN_PATH)).sort_values(SEQ_COL).reset_index(drop=True)
    pressure_only = pd.read_csv(require_path(PRESSURE_ONLY_PATH))
    ablation = pd.read_csv(require_path(ABLATION_PATH))
    by_box = pd.read_csv(require_path(ABLATION_BY_BOX_PATH))
    pred = pd.read_excel(require_path(PREDICTION_PATH), sheet_name=BEST_EXPERIMENT)
    return clean, pressure_only, ablation, by_box, pred


def prediction_long(pred: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for box_id in BOX_IDS:
        frames.append(
            pd.DataFrame(
                {
                    "序列": pred[SEQ_COL],
                    "风箱": box_id,
                    "实际": pred[f"{box_id}#实际"],
                    "预测": pred[f"{box_id}#预测"],
                    "残差": pred[f"{box_id}#残差_预测减实际"],
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out["绝对残差"] = out["残差"].abs()
    return out


def test_start_sequence(pred: pd.DataFrame, clean: pd.DataFrame) -> float:
    if SEQ_COL in pred.columns and len(pred):
        return float(pred[SEQ_COL].min())
    return float(clean[SEQ_COL].iloc[int(len(clean) * 0.8)])


def plot_data_overview(clean: pd.DataFrame, pred: pd.DataFrame) -> Path:
    seq = clean[SEQ_COL].to_numpy(dtype=float)
    split_seq = test_start_sequence(pred, clean)
    pressure = -clean[PRESSURE_COLS].to_numpy(dtype=float).T
    temp = clean[TEMP_COLS].to_numpy(dtype=float).T

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[0.75, 1.1, 1.1])
    ax_speed = fig.add_subplot(grid[0, 0])
    ax_p = fig.add_subplot(grid[1, 0], sharex=ax_speed)
    ax_t = fig.add_subplot(grid[2, 0], sharex=ax_speed)

    ax_speed.plot(seq, clean[SPEED_COL], color="#2f6b4f", linewidth=1.2)
    ax_speed.set_ylabel("机速")
    ax_speed.set_title("第一问数据时序总览：机速、风箱负压与平均温度")

    extent = [seq.min(), seq.max(), 0.5, 18.5]
    im_p = ax_p.imshow(pressure, aspect="auto", origin="lower", extent=extent, cmap="cividis")
    fig.colorbar(im_p, ax=ax_p, pad=0.01, label="负压绝对值 / KPa")
    ax_p.set_ylabel("风箱编号")

    im_t = ax_t.imshow(temp, aspect="auto", origin="lower", extent=extent, cmap="magma")
    fig.colorbar(im_t, ax=ax_t, pad=0.01, label="平均温度 / ℃")
    ax_t.set_ylabel("风箱编号")
    ax_t.set_xlabel("序列")

    for ax in [ax_speed, ax_p, ax_t]:
        ax.axvline(split_seq, color="#1f2933", linestyle="--", linewidth=1.2)
        ax.grid(False)
    ax_speed.text(split_seq, ax_speed.get_ylim()[1], "后20%测试段", va="top", ha="left", fontsize=10)
    return save_fig(fig, "必备_01_数据时空总览.png")


def plot_spatial_profiles(clean: pd.DataFrame, pred: pd.DataFrame) -> Path:
    split_seq = test_start_sequence(pred, clean)
    train = clean[clean[SEQ_COL] < split_seq]
    test = clean[clean[SEQ_COL] >= split_seq]
    boxes = np.array(BOX_IDS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), constrained_layout=True)
    for ax, cols, title, ylabel, transform in [
        (axes[0], PRESSURE_COLS, "各风箱负压强度空间剖面", "负压绝对值 / KPa", lambda x: -x),
        (axes[1], TEMP_COLS, "各风箱平均温度空间剖面", "平均温度 / ℃", lambda x: x),
    ]:
        train_vals = transform(train[cols])
        test_vals = transform(test[cols])
        train_mean = train_vals.mean(axis=0).to_numpy(dtype=float)
        test_mean = test_vals.mean(axis=0).to_numpy(dtype=float)
        train_q10 = train_vals.quantile(0.10, axis=0).to_numpy(dtype=float)
        train_q90 = train_vals.quantile(0.90, axis=0).to_numpy(dtype=float)
        test_q10 = test_vals.quantile(0.10, axis=0).to_numpy(dtype=float)
        test_q90 = test_vals.quantile(0.90, axis=0).to_numpy(dtype=float)

        ax.fill_between(boxes, train_q10, train_q90, color="#8fb8de", alpha=0.22, linewidth=0)
        ax.fill_between(boxes, test_q10, test_q90, color="#d9915b", alpha=0.20, linewidth=0)
        ax.plot(boxes, train_mean, marker="o", linewidth=2.1, color="#276f9f", label="训练段均值")
        ax.plot(boxes, test_mean, marker="s", linewidth=2.1, color="#b75c2b", label="测试段均值")
        ax.axvspan(15.5, 18.5, color="#e7c75f", alpha=0.14)
        ax.set_title(title)
        ax.set_xlabel("风箱编号")
        ax.set_ylabel(ylabel)
        ax.set_xticks(boxes)
        ax.grid(True, alpha=0.24)
    axes[0].legend(loc="best", fontsize=9)
    axes[1].legend(loc="best", fontsize=9)
    return save_fig(fig, "必备_02_负压温度空间剖面.png")


def compute_lag_relation(clean: pd.DataFrame, max_lag_steps: int = 60) -> tuple[np.ndarray, np.ndarray]:
    pressure = -clean[PRESSURE_COLS].astype(float).reset_index(drop=True)
    temp = clean[TEMP_COLS].astype(float).reset_index(drop=True)
    corr = np.zeros((len(BOX_IDS), len(BOX_IDS)), dtype=float)
    lag_seconds = np.zeros_like(corr)

    for i, p_col in enumerate(PRESSURE_COLS):
        p_series = pressure[p_col]
        for j, t_col in enumerate(TEMP_COLS):
            best_r = np.nan
            best_lag = 0
            best_abs = -np.inf
            for lag in range(max_lag_steps + 1):
                x = p_series.shift(lag)
                y = temp[t_col]
                valid = x.notna() & y.notna()
                if valid.sum() < 30:
                    continue
                r = np.corrcoef(x[valid], y[valid])[0, 1]
                if np.isfinite(r) and abs(r) > best_abs:
                    best_abs = abs(r)
                    best_r = r
                    best_lag = lag
            corr[j, i] = best_r
            lag_seconds[j, i] = best_lag * 2
    return corr, lag_seconds


def plot_lag_relation(clean: pd.DataFrame) -> Path:
    corr, lag_seconds = compute_lag_relation(clean)
    boxes = [str(i) for i in BOX_IDS]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.2), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        norm=norm,
        xticklabels=boxes,
        yticklabels=boxes,
        square=True,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "最佳相关系数"},
        ax=axes[0],
    )
    sns.heatmap(
        lag_seconds,
        cmap="viridis",
        xticklabels=boxes,
        yticklabels=boxes,
        square=True,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "最佳滞后 / s"},
        ax=axes[1],
    )
    for ax in axes:
        for k in range(len(BOX_IDS)):
            ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=False, edgecolor="#111111", linewidth=1.1))
        ax.set_xlabel("负压风箱编号")
        ax.set_ylabel("温度风箱编号")
    axes[0].set_title("负压绝对值 → 平均温度的最佳相关强度")
    axes[1].set_title("达到最佳相关时的滞后时间")
    return save_fig(fig, "必备_03_负压温度时滞关系.png")


def model_results(pressure_only: pd.DataFrame, ablation: pd.DataFrame) -> pd.DataFrame:
    p = pressure_only.copy()
    p["result_group"] = "仅负压"
    a = ablation.copy()
    a["result_group"] = "历史温度消融"
    out = pd.concat([p, a], ignore_index=True, sort=False)
    out["label"] = out["experiment"].map(experiment_label)
    return out


def plot_model_comparison(pressure_only: pd.DataFrame, ablation: pd.DataFrame) -> Path:
    results = model_results(pressure_only, ablation).sort_values("mae")
    order = results["label"].tolist()
    palette = {"仅负压": "#94752d", "历史温度消融": "#2f6b4f"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "mae", "MAE 越低越好"),
        (axes[1], "rmse", "RMSE 越低越好"),
        (axes[2], "r2", "R² 越高越好"),
    ]:
        sns.barplot(data=results, y="label", x=metric, hue="result_group", order=order, dodge=False, palette=palette, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("")
        ax.grid(True, axis="x", alpha=0.25)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    axes[0].legend(title="", fontsize=9, loc="lower right")
    axes[1].legend_.remove()
    axes[2].legend_.remove()
    axes[2].axvline(0, color="#222222", linewidth=0.8)
    return save_fig(fig, "必备_04_模型性能对比.png")


def plot_prediction_quality(pred: pd.DataFrame) -> Path:
    long = prediction_long(pred)
    selected = [1, 6, 12, 14, 16, 18]
    seq = pred[SEQ_COL].to_numpy(dtype=float)
    vmin = min(long["实际"].min(), long["预测"].min())
    vmax = max(long["实际"].max(), long["预测"].max())

    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0])
    ax_hex = fig.add_subplot(grid[0, 0])
    ax_res = fig.add_subplot(grid[0, 1])
    ax_box = fig.add_subplot(grid[0, 2])
    ax_track = fig.add_subplot(grid[1, :])

    hb = ax_hex.hexbin(long["实际"], long["预测"], gridsize=55, mincnt=1, cmap="YlGnBu", linewidths=0)
    fig.colorbar(hb, ax=ax_hex, pad=0.01, label="样本密度")
    ax_hex.plot([vmin, vmax], [vmin, vmax], color="#b51d2a", linewidth=1.2)
    ax_hex.set_title("实际值 vs 预测值密度")
    ax_hex.set_xlabel("实际平均温度 / ℃")
    ax_hex.set_ylabel("预测平均温度 / ℃")

    sns.histplot(long["残差"], bins=70, kde=True, color="#557c83", ax=ax_res)
    ax_res.axvline(0, color="#b51d2a", linewidth=1.1)
    ax_res.set_title("残差分布")
    ax_res.set_xlabel("预测 - 实际 / ℃")
    ax_res.set_ylabel("样本数")

    sns.boxplot(data=long, x="风箱", y="残差", color="#cad7df", fliersize=1.2, ax=ax_box)
    ax_box.axhline(0, color="#b51d2a", linewidth=1.0)
    ax_box.set_title("逐风箱残差箱线图")
    ax_box.set_xlabel("风箱编号")
    ax_box.set_ylabel("预测 - 实际 / ℃")
    ax_box.tick_params(axis="x", labelrotation=0)

    colors = sns.color_palette("tab10", n_colors=len(selected))
    for color, box_id in zip(colors, selected):
        ax_track.plot(seq, pred[f"{box_id}#实际"], color=color, linewidth=1.35, label=f"{box_id}#实际")
        ax_track.plot(seq, pred[f"{box_id}#预测"], color=color, linewidth=1.05, linestyle="--", alpha=0.85, label=f"{box_id}#预测")
    ax_track.set_title("最优模型后20%测试段代表风箱预测跟踪")
    ax_track.set_xlabel("序列")
    ax_track.set_ylabel("平均温度 / ℃")
    ax_track.legend(ncol=6, fontsize=8, loc="upper left")
    ax_track.grid(True, alpha=0.25)
    return save_fig(fig, "必备_05_预测跟踪与一致性.png")


def plot_box_error_diagnostics(by_box: pd.DataFrame, pred: pd.DataFrame) -> Path:
    best = by_box[by_box["experiment"].eq(BEST_EXPERIMENT)].copy()
    best["风箱"] = best["box"].map(parse_box_id)
    best = best.sort_values("风箱")
    residual = np.vstack([pred[f"{box_id}#残差_预测减实际"].to_numpy(dtype=float) for box_id in BOX_IDS]).T
    seq = pred[SEQ_COL].to_numpy(dtype=float)
    lim = float(np.nanpercentile(np.abs(residual), 98))

    fig = plt.figure(figsize=(17, 9.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2])
    ax_top = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[1, 0])

    width = 0.38
    x = np.array(BOX_IDS)
    ax_top.bar(x - width / 2, best["mae"], width=width, color="#487aa1", label="MAE")
    ax_top.bar(x + width / 2, best["rmse"], width=width, color="#d68a47", label="RMSE")
    ax_r2 = ax_top.twinx()
    ax_r2.plot(x, best["r2"], color="#9e2f3a", marker="o", linewidth=2.0, label="R²")
    ax_top.set_title("最优模型逐风箱误差空间指纹")
    ax_top.set_xlabel("风箱编号")
    ax_top.set_ylabel("误差 / ℃")
    ax_r2.set_ylabel("R²")
    ax_top.set_xticks(x)
    ax_top.axvspan(12.5, 14.5, color="#c65742", alpha=0.10)
    ax_top.legend(loc="upper left")
    ax_r2.legend(loc="upper right")

    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    im = ax_heat.imshow(
        residual.T,
        aspect="auto",
        origin="lower",
        extent=[seq.min(), seq.max(), 0.5, 18.5],
        cmap="RdBu_r",
        norm=norm,
    )
    fig.colorbar(im, ax=ax_heat, pad=0.01, label="残差：预测 - 实际 / ℃")
    ax_heat.set_title("测试段残差时空热图")
    ax_heat.set_xlabel("序列")
    ax_heat.set_ylabel("风箱编号")
    ax_heat.set_yticks(BOX_IDS)
    ax_heat.grid(False)
    return save_fig(fig, "必备_06_逐风箱误差与残差诊断.png")


def categorize_feature(feature: str) -> str:
    lower = str(feature).lower()
    if "temp" in lower or "target" in lower:
        return "历史温度"
    if lower.startswith("speed"):
        return "机速"
    if any(token in lower for token in ["lag", "delta", "rate", "roll"]):
        return "负压时序"
    if any(token in lower for token in ["front", "middle", "back", "long", "short", "all"]):
        return "分段负压"
    if any(token in lower for token in ["neighbor", "adjacent", "local"]):
        return "空间邻域"
    if lower.startswith("p_"):
        return "原始负压"
    return "其他"


def plot_feature_importance() -> Path:
    importance = pd.read_csv(require_path(IMPORTANCE_PATH))
    top = importance.head(25).copy()
    top["重要性占比"] = top["importance_ratio"] * 100.0
    top["类别"] = top["feature"].map(categorize_feature)
    grouped = importance.assign(类别=importance["feature"].map(categorize_feature))
    grouped = grouped.groupby("类别", as_index=False)["importance_ratio"].sum()
    grouped["重要性占比"] = grouped["importance_ratio"] * 100.0
    grouped = grouped.sort_values("重要性占比", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [1.45, 1.0]}, constrained_layout=True)
    sns.barplot(data=top.sort_values("重要性占比"), y="feature", x="重要性占比", hue="类别", dodge=False, ax=axes[0])
    axes[0].set_title("Top 25 特征重要性")
    axes[0].set_xlabel("重要性占比 / %")
    axes[0].set_ylabel("")
    axes[0].legend(title="", fontsize=8, loc="lower right")

    sns.barplot(data=grouped, y="类别", x="重要性占比", color="#547a8a", ax=axes[1])
    axes[1].set_title("特征类别重要性汇总")
    axes[1].set_xlabel("重要性占比 / %")
    axes[1].set_ylabel("")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f%%", fontsize=9, padding=3)
    return save_fig(fig, "可选_07_特征重要性结构.png")


def plot_ablation_contribution(ablation: pd.DataFrame, by_box: pd.DataFrame) -> Path:
    plot_df = ablation.copy()
    plot_df["模型"] = plot_df["experiment"].map(experiment_label)
    box_delta = by_box[by_box["experiment"].isin(["history_only_delta_lgbm", BEST_EXPERIMENT])].copy()
    box_delta["风箱"] = box_delta["box"].map(parse_box_id)
    pivot = box_delta.pivot_table(index="风箱", columns="experiment", values="mae")
    pivot["加入负压后的MAE改变量"] = pivot["history_only_delta_lgbm"] - pivot[BEST_EXPERIMENT]
    pivot = pivot.reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), constrained_layout=True)
    sns.barplot(data=plot_df.sort_values("mae"), y="模型", x="mae", color="#4f7f92", ax=axes[0])
    axes[0].set_title("历史温度反馈模型消融")
    axes[0].set_xlabel("MAE / ℃")
    axes[0].set_ylabel("")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.2f", fontsize=9, padding=3)

    colors = np.where(pivot["加入负压后的MAE改变量"] >= 0, "#2f7d57", "#b65a3a")
    axes[1].bar(pivot["风箱"], pivot["加入负压后的MAE改变量"], color=colors)
    axes[1].axhline(0, color="#222222", linewidth=0.9)
    axes[1].set_title("加入负压后的逐风箱 MAE 改变量")
    axes[1].set_xlabel("风箱编号")
    axes[1].set_ylabel("MAE改变量 = 历史温度 - 历史温度+负压 / ℃")
    axes[1].set_xticks(BOX_IDS)
    axes[1].grid(True, axis="y", alpha=0.25)
    return save_fig(fig, "可选_08_历史温度与负压消融贡献.png")


def smooth_surface(
    seq: np.ndarray,
    box_ids: np.ndarray,
    z: np.ndarray,
    seq_points: int = 220,
    box_points: int = 120,
    sigma: tuple[float, float] = (1.5, 1.1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq = np.asarray(seq, dtype=float)
    box_ids = np.asarray(box_ids, dtype=float)
    z = np.asarray(z, dtype=float)
    seq_new = np.linspace(seq.min(), seq.max(), seq_points)
    box_new = np.linspace(box_ids.min(), box_ids.max(), box_points)
    spline = RectBivariateSpline(seq, box_ids, z, kx=3, ky=3, s=0)
    z_new = spline(seq_new, box_new)
    z_smooth = gaussian_filter(z_new, sigma=sigma, mode="nearest")
    x_grid, y_grid = np.meshgrid(seq_new, box_new, indexing="ij")
    return x_grid, y_grid, z_smooth


def style_3d_axis(ax: plt.Axes, title: str, zlabel: str) -> None:
    ax.set_title(title, pad=16)
    ax.set_xlabel("序列", labelpad=8)
    ax.set_ylabel("风箱编号", labelpad=8)
    ax.set_zlabel(zlabel, labelpad=8)
    ax.view_init(elev=28, azim=-132)
    ax.grid(True, alpha=0.22)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0.0))


def plot_3d_surface(
    seq: np.ndarray,
    z: np.ndarray,
    filename: str,
    title: str,
    zlabel: str,
    cmap_name: str,
    diverging: bool = False,
    sigma: tuple[float, float] = (1.5, 1.1),
    clip_percentile: float | None = None,
) -> Path:
    z_for_plot = np.asarray(z, dtype=float)
    if clip_percentile is not None:
        limit = float(np.nanpercentile(np.abs(z_for_plot), clip_percentile))
        z_for_plot = np.clip(z_for_plot, -limit, limit)
    x_grid, y_grid, z_smooth = smooth_surface(seq, np.array(BOX_IDS), z_for_plot, sigma=sigma)
    fig = plt.figure(figsize=(12.5, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    if diverging:
        lim = float(np.nanpercentile(np.abs(z_smooth), 98))
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        cmap_obj = matplotlib.colormaps[cmap_name]
        facecolors = cmap_obj(norm(z_smooth))
        surf = ax.plot_surface(
            x_grid,
            y_grid,
            z_smooth,
            facecolors=facecolors,
            linewidth=0,
            antialiased=True,
            shade=False,
            rcount=220,
            ccount=120,
        )
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array(z_smooth)
        fig.colorbar(mappable, ax=ax, shrink=0.64, pad=0.08, label=zlabel)
    else:
        norm = Normalize(vmin=float(np.nanpercentile(z_smooth, 1)), vmax=float(np.nanpercentile(z_smooth, 99)))
        surf = ax.plot_surface(
            x_grid,
            y_grid,
            z_smooth,
            cmap=cmap_name,
            norm=norm,
            linewidth=0,
            antialiased=True,
            shade=True,
            rcount=220,
            ccount=120,
        )
        fig.colorbar(surf, ax=ax, shrink=0.64, pad=0.08, label=zlabel)
    style_3d_axis(ax, title, zlabel)
    return save_fig(fig, filename)


def plot_3d_figures(clean: pd.DataFrame, pred: pd.DataFrame) -> list[Path]:
    seq = clean[SEQ_COL].to_numpy(dtype=float)
    pressure_abs = -clean[PRESSURE_COLS].to_numpy(dtype=float)
    temp = clean[TEMP_COLS].to_numpy(dtype=float)
    pred_seq = pred[SEQ_COL].to_numpy(dtype=float)
    residual = np.vstack([pred[f"{box_id}#残差_预测减实际"].to_numpy(dtype=float) for box_id in BOX_IDS]).T

    return [
        plot_3d_surface(
            seq,
            temp,
            "可选_09_3D温度时空平滑曲面.png",
            "风箱平均温度时空平滑曲面",
            "平均温度 / ℃",
            "magma",
            diverging=False,
        ),
        plot_3d_surface(
            seq,
            pressure_abs,
            "可选_10_3D负压时空平滑曲面.png",
            "风箱负压强度时空平滑曲面",
            "负压绝对值 / KPa",
            "cividis",
            diverging=False,
        ),
        plot_3d_surface(
            pred_seq,
            residual,
            "可选_11_3D预测残差平滑曲面.png",
            "最优模型测试段残差时空平滑曲面",
            "残差 / ℃",
            "RdBu_r",
            diverging=True,
            sigma=(3.4, 1.8),
            clip_percentile=97.0,
        ),
    ]


def write_index(paths: list[Path], ablation: pd.DataFrame) -> Path:
    best = ablation.sort_values(["mae", "rmse"]).iloc[0]
    file_set = {p.name for p in paths}
    lines = [
        "# 第一问可视化PNG图表索引",
        "",
        "## 建模口径",
        "- 题目第一问要求探索各风箱负压与对应风箱温度间的关系，并建立温度预测模型。",
        "- 南北两侧温度已按题目提示合并为各风箱平均温度。",
        "- 本批图使用现有导出结果，不重新训练模型；主结果采用按时间前80%训练、后20%测试的历史温度+负压增量 LightGBM。",
        f"- 当前最优实验：`{best['experiment']}`，MAE={best['mae']:.4f}，RMSE={best['rmse']:.4f}，R2={best['r2']:.4f}。",
        "",
        "## 正文必备图",
    ]
    for filename, note in FIGURE_NOTES:
        if filename in file_set and filename.startswith("必备"):
            lines.append(f"- `{filename}`：{note}")
    lines.extend(["", "## 可选增强图"])
    for filename, note in FIGURE_NOTES:
        if filename in file_set and filename.startswith("可选"):
            lines.append(f"- `{filename}`：{note}")
    lines.extend(
        [
            "",
            "## 3D平滑说明",
            "- 三维图先在时间轴和风箱位置轴做三次样条插值，再做轻量 Gaussian 平滑；曲面只用于展示连续时空形态，定量读数仍建议看二维热图和逐风箱指标。",
            "- 残差3D图为避免极端尖峰遮挡整体形态，展示前按绝对残差97%分位轻度裁剪；原始残差仍保留在 `必备_06_逐风箱误差与残差诊断.png` 中。",
        ]
    )
    path = FIG_DIR / "图表索引.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    setup_style()
    clean, pressure_only, ablation, by_box, pred = load_inputs()
    paths = [
        plot_data_overview(clean, pred),
        plot_spatial_profiles(clean, pred),
        plot_lag_relation(clean),
        plot_model_comparison(pressure_only, ablation),
        plot_prediction_quality(pred),
        plot_box_error_diagnostics(by_box, pred),
        plot_feature_importance(),
        plot_ablation_contribution(ablation, by_box),
        *plot_3d_figures(clean, pred),
    ]
    index_path = write_index(paths, ablation)
    print("已生成第一问PNG可视化：")
    for path in paths:
        print(path)
    print(index_path)


if __name__ == "__main__":
    main()
