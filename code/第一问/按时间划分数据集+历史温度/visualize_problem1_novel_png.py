from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from scipy.ndimage import gaussian_filter1d


warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "第一问"
RESULT_DIR = PROJECT_ROOT / "result" / "第一问" / "时间划分数据集+历史温度"
FIG_DIR = PROJECT_ROOT / "figure" / "第一问" / "局部新颖PNG"

CLEAN_PATH = DATA_DIR / "problem1_clean.csv"
PREDICTION_PATH = RESULT_DIR / "problem1_history_pressure_ablation_predictions.xlsx"
BY_BOX_PATH = RESULT_DIR / "problem1_history_pressure_ablation_by_box.csv"
BEST_EXPERIMENT = "history_pressure_delta_lgbm"

SEQ_COL = "序列"
SPEED_COL = "烧结机机速L1设定"
BOX_IDS = list(range(1, 19))
BOX_ARRAY = np.array(BOX_IDS, dtype=float)
PRESSURE_COLS = [f"{i}#风箱负压" for i in BOX_IDS]
TEMP_COLS = [f"{i}#风箱平均温度" for i in BOX_IDS]
FIG_DPI = 300
PALETTE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "black": "#111111",
    "gray": "#777777",
}


def setup_style() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="ticks", context="paper", font_scale=1.12)
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


def parse_box_id(value: str) -> int:
    match = re.search(r"(\d+)", str(value))
    if not match:
        raise ValueError(f"无法解析风箱编号: {value}")
    return int(match.group(1))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean = pd.read_csv(require_path(CLEAN_PATH)).sort_values(SEQ_COL).reset_index(drop=True)
    pred = pd.read_excel(require_path(PREDICTION_PATH), sheet_name=BEST_EXPERIMENT)
    by_box = pd.read_csv(require_path(BY_BOX_PATH))
    return clean, pred, by_box


def temp_matrix(clean: pd.DataFrame) -> np.ndarray:
    return clean[TEMP_COLS].to_numpy(dtype=float)


def pressure_abs_matrix(clean: pd.DataFrame) -> np.ndarray:
    return -clean[PRESSURE_COLS].to_numpy(dtype=float)


def residual_matrix(pred: pd.DataFrame) -> np.ndarray:
    return np.vstack([pred[f"{box_id}#残差_预测减实际"].to_numpy(dtype=float) for box_id in BOX_IDS]).T


def state_frame(clean: pd.DataFrame) -> pd.DataFrame:
    seq = clean[SEQ_COL].to_numpy(dtype=float)
    temp = temp_matrix(clean)
    p_abs = pressure_abs_matrix(clean)
    temp_weights = temp - np.nanmin(temp, axis=1, keepdims=True) + 1.0
    thermal_center = (temp_weights * BOX_ARRAY).sum(axis=1) / temp_weights.sum(axis=1)
    suction_center = (p_abs * BOX_ARRAY).sum(axis=1) / p_abs.sum(axis=1)
    delta_temp = np.vstack([np.zeros((1, temp.shape[1])), np.diff(temp, axis=0)])
    delta_pressure = np.vstack([np.zeros((1, p_abs.shape[1])), np.diff(p_abs, axis=0)])

    frame = pd.DataFrame(
        {
            "序列": seq,
            "机速": clean[SPEED_COL].to_numpy(dtype=float),
            "热重心": thermal_center,
            "负压重心": suction_center,
            "峰值温度": temp.max(axis=1),
            "峰值风箱": np.argmax(temp, axis=1) + 1,
            "平均温度": temp.mean(axis=1),
            "前段温度": temp[:, :5].mean(axis=1),
            "中段温度": temp[:, 5:12].mean(axis=1),
            "后段温度": temp[:, 12:].mean(axis=1),
            "温度后前差": temp[:, 12:].mean(axis=1) - temp[:, :5].mean(axis=1),
            "前段负压": p_abs[:, :5].mean(axis=1),
            "中段负压": p_abs[:, 5:12].mean(axis=1),
            "后段负压": p_abs[:, 12:].mean(axis=1),
            "负压后前差": p_abs[:, 12:].mean(axis=1) - p_abs[:, :5].mean(axis=1),
            "温度跃迁能量": np.sqrt((delta_temp**2).mean(axis=1)),
            "负压跃迁能量": np.sqrt((delta_pressure**2).mean(axis=1)),
        }
    )
    return frame


def prediction_state(pred: pd.DataFrame) -> pd.DataFrame:
    residual = residual_matrix(pred)
    out = pd.DataFrame(
        {
            "序列": pred[SEQ_COL].to_numpy(dtype=float),
            "行MAE": np.abs(residual).mean(axis=1),
            "最大绝对残差": np.abs(residual).max(axis=1),
            "最难风箱": np.argmax(np.abs(residual), axis=1) + 1,
        }
    )
    return out


def seq_window(df: pd.DataFrame, center_seq: float, half_width: int) -> pd.DataFrame:
    return df[(df[SEQ_COL] >= center_seq - half_width) & (df[SEQ_COL] <= center_seq + half_width)].copy()


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def plot_thermal_ridge(clean: pd.DataFrame, pred: pd.DataFrame, state: pd.DataFrame) -> Path:
    seq = clean[SEQ_COL].to_numpy(dtype=float)
    temp = temp_matrix(clean).T
    p_abs = pressure_abs_matrix(clean).T
    split_seq = float(pred[SEQ_COL].min())

    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.1, 0.85])
    ax_temp = fig.add_subplot(grid[0, 0])
    ax_press = fig.add_subplot(grid[1, 0], sharex=ax_temp)
    ax_traj = fig.add_subplot(grid[2, 0], sharex=ax_temp)

    extent = [seq.min(), seq.max(), 0.5, 18.5]
    im_temp = ax_temp.imshow(temp, aspect="auto", origin="lower", extent=extent, cmap="magma")
    fig.colorbar(im_temp, ax=ax_temp, pad=0.01, label="平均温度 / ℃")
    ax_temp.plot(seq, state["峰值风箱"], color="#F0E442", linewidth=1.8, label="峰值风箱")
    ax_temp.plot(seq, state["热重心"], color="white", linewidth=1.6, label="热重心")
    ax_temp.set_title("热状态迁移轨迹：峰值风箱、热重心与测试段")
    ax_temp.set_ylabel("风箱编号")
    ax_temp.legend(loc="upper left", ncol=2, frameon=True, fontsize=9)
    add_panel_label(ax_temp, "A")

    im_press = ax_press.imshow(p_abs, aspect="auto", origin="lower", extent=extent, cmap="cividis")
    fig.colorbar(im_press, ax=ax_press, pad=0.01, label="负压绝对值 / KPa")
    ax_press.plot(seq, state["负压重心"], color="white", linewidth=1.6, label="负压重心")
    ax_press.set_title("负压强度分布与抽风重心")
    ax_press.set_ylabel("风箱编号")
    ax_press.legend(loc="upper left", frameon=True, fontsize=9)
    add_panel_label(ax_press, "B")

    ax_traj.plot(seq, state["热重心"], color=PALETTE["red"], linewidth=1.6, label="热重心")
    ax_traj.plot(seq, state["负压重心"], color=PALETTE["blue"], linewidth=1.5, label="负压重心")
    ax_traj.fill_between(seq, state["热重心"], state["负压重心"], color="#999999", alpha=0.14)
    ax_traj.set_title("热重心与负压重心的相对位移")
    ax_traj.set_xlabel("序列")
    ax_traj.set_ylabel("重心位置 / 风箱编号")
    ax_traj.legend(ncol=2, frameon=False, fontsize=9)
    add_panel_label(ax_traj, "C")

    for ax in [ax_temp, ax_press, ax_traj]:
        ax.axvline(split_seq, color="#111111", linestyle="--", linewidth=1.0)
        ax.grid(False)
    return save_fig(fig, "新颖_01_热状态迁移轨迹与负压重心耦合.png")


def plot_residual_casebook(clean: pd.DataFrame, pred: pd.DataFrame, state: pd.DataFrame, pred_state: pd.DataFrame) -> tuple[Path, float]:
    event_seq = float(pred_state.loc[pred_state["行MAE"].idxmax(), "序列"])
    window_pred = seq_window(pred, event_seq, 45)
    window_clean = seq_window(clean, event_seq, 45)
    event_row = pred[pred[SEQ_COL].eq(event_seq)].iloc[0]
    event_resid = np.array([event_row[f"{box_id}#残差_预测减实际"] for box_id in BOX_IDS], dtype=float)
    top_boxes = [int(x) for x in (np.argsort(np.abs(event_resid))[::-1][:4] + 1)]
    residual_local = np.vstack([window_pred[f"{box_id}#残差_预测减实际"].to_numpy(dtype=float) for box_id in BOX_IDS])
    seq = window_pred[SEQ_COL].to_numpy(dtype=float)
    lim = float(np.nanpercentile(np.abs(residual_local), 97))

    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.95], width_ratios=[1.1, 1.0])
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_lines = fig.add_subplot(grid[0, 1])
    ax_press = fig.add_subplot(grid[1, 0])
    ax_bar = fig.add_subplot(grid[1, 1])

    im = ax_heat.imshow(
        residual_local,
        aspect="auto",
        origin="lower",
        extent=[seq.min(), seq.max(), 0.5, 18.5],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
    )
    fig.colorbar(im, ax=ax_heat, pad=0.01, label="残差：预测 - 实际 / ℃")
    ax_heat.axvline(event_seq, color="#111111", linestyle="--", linewidth=1.0)
    ax_heat.set_title(f"最大系统性残差片段：序列 {event_seq:.0f} 附近")
    ax_heat.set_xlabel("序列")
    ax_heat.set_ylabel("风箱编号")
    add_panel_label(ax_heat, "A")

    colors = sns.color_palette("colorblind", n_colors=len(top_boxes))
    for color, box_id in zip(colors, top_boxes):
        ax_lines.plot(window_pred[SEQ_COL], window_pred[f"{box_id}#实际"], color=color, linewidth=1.5, label=f"{box_id}#实际")
        ax_lines.plot(window_pred[SEQ_COL], window_pred[f"{box_id}#预测"], color=color, linewidth=1.2, linestyle="--", alpha=0.85, label=f"{box_id}#预测")
    ax_lines.axvline(event_seq, color="#111111", linestyle="--", linewidth=1.0)
    ax_lines.set_title("最难风箱实际-预测局部跟踪")
    ax_lines.set_xlabel("序列")
    ax_lines.set_ylabel("平均温度 / ℃")
    ax_lines.legend(ncol=2, fontsize=8, frameon=False)
    add_panel_label(ax_lines, "B")

    for color, box_id in zip(colors, top_boxes):
        ax_press.plot(window_clean[SEQ_COL], -window_clean[f"{box_id}#风箱负压"], color=color, linewidth=1.4, label=f"{box_id}#")
    ax_press.axvline(event_seq, color="#111111", linestyle="--", linewidth=1.0)
    ax_press.set_title("同一局部片段的对应风箱负压强度")
    ax_press.set_xlabel("序列")
    ax_press.set_ylabel("负压绝对值 / KPa")
    ax_press.legend(ncol=4, fontsize=8, frameon=False)
    add_panel_label(ax_press, "C")

    bar_colors = np.where(event_resid >= 0, PALETTE["red"], PALETTE["blue"])
    ax_bar.bar(BOX_IDS, event_resid, color=bar_colors, width=0.75)
    ax_bar.axhline(0, color="#111111", linewidth=0.9)
    ax_bar.set_title("事件时刻18个风箱残差剖面")
    ax_bar.set_xlabel("风箱编号")
    ax_bar.set_ylabel("残差 / ℃")
    ax_bar.set_xticks(BOX_IDS)
    add_panel_label(ax_bar, "D")
    sns.despine(fig)
    return save_fig(fig, "新颖_02_最大残差局部病例册.png"), event_seq


def plot_jump_lens(clean: pd.DataFrame, pred: pd.DataFrame, state: pd.DataFrame) -> tuple[Path, float]:
    test_start = float(pred[SEQ_COL].min())
    test_state = state[state[SEQ_COL] >= test_start].copy()
    event_seq = float(test_state.loc[test_state["温度跃迁能量"].idxmax(), SEQ_COL])
    window_clean = seq_window(clean, event_seq, 55)
    window_state = seq_window(state, event_seq, 55)
    seq = window_clean[SEQ_COL].to_numpy(dtype=float)
    temp_local = temp_matrix(window_clean).T
    p_delta = np.diff(pressure_abs_matrix(window_clean), axis=0, prepend=pressure_abs_matrix(window_clean)[[0]]).T
    lim = float(np.nanpercentile(np.abs(p_delta), 98))

    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9])
    ax_temp = fig.add_subplot(grid[0, 0])
    ax_delta = fig.add_subplot(grid[0, 1])
    ax_center = fig.add_subplot(grid[1, 0])
    ax_energy = fig.add_subplot(grid[1, 1])

    extent = [seq.min(), seq.max(), 0.5, 18.5]
    im1 = ax_temp.imshow(temp_local, aspect="auto", origin="lower", extent=extent, cmap="magma")
    fig.colorbar(im1, ax=ax_temp, pad=0.01, label="平均温度 / ℃")
    ax_temp.plot(window_state[SEQ_COL], window_state["峰值风箱"], color="#F0E442", linewidth=1.6, label="峰值风箱")
    ax_temp.axvline(event_seq, color="#111111", linestyle="--", linewidth=1.0)
    ax_temp.set_title(f"最高温度跃迁局部窗口：序列 {event_seq:.0f}")
    ax_temp.set_xlabel("序列")
    ax_temp.set_ylabel("风箱编号")
    ax_temp.legend(fontsize=8, frameon=True, loc="upper left")
    add_panel_label(ax_temp, "A")

    im2 = ax_delta.imshow(
        p_delta,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="PuOr_r",
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
    )
    fig.colorbar(im2, ax=ax_delta, pad=0.01, label="负压强度变化 / KPa")
    ax_delta.axvline(event_seq, color="#111111", linestyle="--", linewidth=1.0)
    ax_delta.set_title("同窗负压变化热图")
    ax_delta.set_xlabel("序列")
    ax_delta.set_ylabel("风箱编号")
    add_panel_label(ax_delta, "B")

    ax_center.plot(window_state[SEQ_COL], window_state["热重心"], color=PALETTE["red"], linewidth=1.7, label="热重心")
    ax_center.plot(window_state[SEQ_COL], window_state["负压重心"], color=PALETTE["blue"], linewidth=1.6, label="负压重心")
    ax_center2 = ax_center.twinx()
    ax_center2.plot(window_state[SEQ_COL], window_state["机速"], color=PALETTE["green"], linewidth=1.2, alpha=0.75, label="机速")
    ax_center.axvline(event_seq, color="#111111", linestyle="--", linewidth=1.0)
    ax_center.set_title("局部重心与机速联动")
    ax_center.set_xlabel("序列")
    ax_center.set_ylabel("重心位置 / 风箱编号")
    ax_center2.set_ylabel("机速")
    lines, labels = ax_center.get_legend_handles_labels()
    lines2, labels2 = ax_center2.get_legend_handles_labels()
    ax_center.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, ncol=3, loc="upper left")
    add_panel_label(ax_center, "C")

    ax_energy.plot(window_state[SEQ_COL], window_state["温度跃迁能量"], color=PALETTE["red"], linewidth=1.6, label="温度跃迁能量")
    ax_energy.plot(window_state[SEQ_COL], window_state["负压跃迁能量"], color=PALETTE["blue"], linewidth=1.4, label="负压跃迁能量")
    ax_energy.axvline(event_seq, color="#111111", linestyle="--", linewidth=1.0)
    ax_energy.set_title("多变量跃迁能量")
    ax_energy.set_xlabel("序列")
    ax_energy.set_ylabel("RMS变化量")
    ax_energy.legend(frameon=False, fontsize=8)
    add_panel_label(ax_energy, "D")
    sns.despine(fig)
    return save_fig(fig, "新颖_03_最高温跃迁局部放大镜.png"), event_seq


def plot_phase_hysteresis(clean: pd.DataFrame, by_box: pd.DataFrame, event_seq: float) -> Path:
    best = by_box[by_box["experiment"].eq(BEST_EXPERIMENT)].copy()
    best["风箱"] = best["box"].map(parse_box_id)
    selected = best.sort_values("mae", ascending=False)["风箱"].head(4).astype(int).tolist()
    local = seq_window(clean, event_seq, 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()
    for ax, box_id in zip(axes, selected):
        x_all = -clean[f"{box_id}#风箱负压"].to_numpy(dtype=float)
        y_all = clean[f"{box_id}#风箱平均温度"].to_numpy(dtype=float)
        ax.hexbin(x_all, y_all, gridsize=34, mincnt=1, cmap="Greys", alpha=0.35, linewidths=0)
        x = -local[f"{box_id}#风箱负压"].to_numpy(dtype=float)
        y = local[f"{box_id}#风箱平均温度"].to_numpy(dtype=float)
        seq = local[SEQ_COL].to_numpy(dtype=float)
        ax.plot(x, y, color=PALETTE["orange"], linewidth=1.3, alpha=0.85)
        sc = ax.scatter(x, y, c=seq, cmap="viridis", s=22, edgecolor="white", linewidth=0.2, zorder=3)
        step = max(4, len(local) // 9)
        for k in range(0, len(local) - step, step):
            ax.annotate("", xy=(x[k + step], y[k + step]), xytext=(x[k], y[k]), arrowprops=dict(arrowstyle="->", color=PALETTE["orange"], lw=0.9, alpha=0.75))
        ax.set_title(f"{box_id}#风箱局部负压-温度相图")
        ax.set_xlabel("负压绝对值 / KPa")
        ax.set_ylabel("平均温度 / ℃")
        add_panel_label(ax, chr(ord("A") + list(axes).index(ax)))
    fig.colorbar(sc, ax=axes.tolist(), shrink=0.78, pad=0.01, label="序列")
    sns.despine(fig)
    return save_fig(fig, "新颖_04_局部相图_负压温度滞回轨迹.png")


def plot_difficulty_phase_map(state: pd.DataFrame, pred_state: pd.DataFrame, residual_event_seq: float, jump_event_seq: float) -> Path:
    test = state.merge(pred_state, on=SEQ_COL, how="inner")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)

    sc1 = axes[0].scatter(
        test["负压重心"],
        test["热重心"],
        c=test["行MAE"],
        s=28 + test["最大绝对残差"] * 2.4,
        cmap="magma",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.25,
    )
    axes[0].plot([1, 18], [1, 18], color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].scatter(
        test.loc[test[SEQ_COL].eq(residual_event_seq), "负压重心"],
        test.loc[test[SEQ_COL].eq(residual_event_seq), "热重心"],
        marker="*",
        s=260,
        color=PALETTE["sky"],
        edgecolor=PALETTE["black"],
        linewidth=0.8,
        label="最大残差片段",
    )
    axes[0].scatter(
        test.loc[test[SEQ_COL].eq(jump_event_seq), "负压重心"],
        test.loc[test[SEQ_COL].eq(jump_event_seq), "热重心"],
        marker="P",
        s=180,
        color=PALETTE["green"],
        edgecolor=PALETTE["black"],
        linewidth=0.8,
        label="最高跃迁片段",
    )
    axes[0].set_title("测试段困难工况：热重心-负压重心相空间")
    axes[0].set_xlabel("负压重心 / 风箱编号")
    axes[0].set_ylabel("热重心 / 风箱编号")
    axes[0].legend(frameon=False, fontsize=9)
    add_panel_label(axes[0], "A")

    sc2 = axes[1].scatter(
        test["负压后前差"],
        test["温度后前差"],
        c=test["行MAE"],
        s=34,
        cmap="magma",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.25,
    )
    axes[1].axhline(0, color="#555555", linewidth=0.9)
    axes[1].axvline(0, color="#555555", linewidth=0.9)
    axes[1].scatter(
        test.loc[test[SEQ_COL].eq(residual_event_seq), "负压后前差"],
        test.loc[test[SEQ_COL].eq(residual_event_seq), "温度后前差"],
        marker="*",
        s=260,
        color=PALETTE["sky"],
        edgecolor=PALETTE["black"],
        linewidth=0.8,
    )
    axes[1].scatter(
        test.loc[test[SEQ_COL].eq(jump_event_seq), "负压后前差"],
        test.loc[test[SEQ_COL].eq(jump_event_seq), "温度后前差"],
        marker="P",
        s=180,
        color=PALETTE["green"],
        edgecolor=PALETTE["black"],
        linewidth=0.8,
    )
    axes[1].set_title("前后段梯度相空间")
    axes[1].set_xlabel("后段-前段负压强度 / KPa")
    axes[1].set_ylabel("后段-前段平均温度 / ℃")
    add_panel_label(axes[1], "B")

    fig.colorbar(sc1, ax=axes, shrink=0.78, pad=0.01, label="测试段行MAE / ℃")
    sns.despine(fig)
    return save_fig(fig, "新颖_05_残差困难区相空间地图.png")


def compute_lag_corr(clean: pd.DataFrame, max_lag_steps: int = 60) -> tuple[np.ndarray, np.ndarray]:
    pressure = pd.DataFrame(pressure_abs_matrix(clean), columns=BOX_IDS)
    temp = pd.DataFrame(temp_matrix(clean), columns=BOX_IDS)
    corr = np.zeros((18, 18), dtype=float)
    lag = np.zeros((18, 18), dtype=float)
    for i, box_p in enumerate(BOX_IDS):
        p_series = pressure[box_p]
        for j, box_t in enumerate(BOX_IDS):
            best_r = 0.0
            best_abs = -np.inf
            best_lag = 0
            for lag_step in range(max_lag_steps + 1):
                x = p_series.shift(lag_step)
                y = temp[box_t]
                valid = x.notna() & y.notna()
                if valid.sum() < 40:
                    continue
                r = np.corrcoef(x[valid], y[valid])[0, 1]
                if np.isfinite(r) and abs(r) > best_abs:
                    best_abs = abs(r)
                    best_r = r
                    best_lag = lag_step
            corr[j, i] = best_r
            lag[j, i] = best_lag * 2.0
    return corr, lag


def draw_arc(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: str, width: float, alpha: float) -> None:
    ctrl_y = 0.5 + 0.28 * min(1.0, abs(x1 - x0) / 9.0)
    path = MplPath([(x0, y0), ((x0 + x1) / 2.0, ctrl_y), (x1, y1)], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    patch = PathPatch(path, facecolor="none", edgecolor=color, lw=width, alpha=alpha, capstyle="round")
    ax.add_patch(patch)


def plot_coupling_network(clean: pd.DataFrame) -> Path:
    corr, lag = compute_lag_corr(clean)
    links = []
    for j, temp_box in enumerate(BOX_IDS):
        for i, pressure_box in enumerate(BOX_IDS):
            if abs(temp_box - pressure_box) > 5:
                continue
            links.append(
                {
                    "pressure_box": pressure_box,
                    "temp_box": temp_box,
                    "corr": corr[j, i],
                    "lag": lag[j, i],
                    "score": abs(corr[j, i]),
                }
            )
    link_df = pd.DataFrame(links).sort_values("score", ascending=False).head(24)

    fig, ax = plt.subplots(figsize=(17, 6.6), constrained_layout=True)
    for box in BOX_IDS:
        ax.scatter(box, 0, s=150, color="#dce8f2", edgecolor="#385f7d", zorder=3)
        ax.scatter(box, 1, s=150, color="#f2e3d5", edgecolor="#9b5834", zorder=3)
        ax.text(box, -0.10, str(box), ha="center", va="top", fontsize=8)
        ax.text(box, 1.10, str(box), ha="center", va="bottom", fontsize=8)
    for idx, row in enumerate(link_df.itertuples(index=False)):
        color = PALETTE["red"] if row.corr > 0 else PALETTE["blue"]
        width = 0.7 + 4.2 * row.score
        draw_arc(ax, row.pressure_box, 0, row.temp_box, 1, color, width, 0.26 + 0.55 * row.score)
        if idx < 8:
            mid_x = (row.pressure_box + row.temp_box) / 2.0
            ax.text(mid_x, 0.52 + 0.04 * (idx % 3), f"{row.lag:.0f}s", color=color, ha="center", va="center", fontsize=8)

    ax.text(0.35, 0, "负压输入", ha="right", va="center", fontsize=11, fontweight="bold", color="#385f7d")
    ax.text(0.35, 1, "温度响应", ha="right", va="center", fontsize=11, fontweight="bold", color="#9b5834")
    ax.plot([14.1, 15.0], [1.25, 1.25], color=PALETTE["red"], linewidth=3.0, alpha=0.75)
    ax.text(15.1, 1.25, "正相关", va="center", fontsize=9, color=PALETTE["red"])
    ax.plot([16.0, 16.9], [1.25, 1.25], color=PALETTE["blue"], linewidth=3.0, alpha=0.75)
    ax.text(17.0, 1.25, "负相关", va="center", fontsize=9, color=PALETTE["blue"])
    ax.set_title("局部负压-温度强耦合通道网络（线宽表示相关强度，颜色表示方向）")
    ax.set_xlim(0.2, 18.8)
    ax.set_ylim(-0.24, 1.32)
    ax.axis("off")
    return save_fig(fig, "新颖_06_负压温度强耦合通道网络.png")


def zscore_fingerprint(state: pd.DataFrame, test_start: float, residual_event_seq: float, jump_event_seq: float) -> Path:
    variables = [
        "机速",
        "前段负压",
        "中段负压",
        "后段负压",
        "负压重心",
        "前段温度",
        "中段温度",
        "后段温度",
        "峰值温度",
        "热重心",
        "温度后前差",
        "温度跃迁能量",
        "负压跃迁能量",
    ]
    train = state[state[SEQ_COL] < test_start]
    test = state[state[SEQ_COL] >= test_start]
    residual_window = seq_window(state, residual_event_seq, 45)
    jump_window = seq_window(state, jump_event_seq, 45)
    baseline_mean = train[variables].mean()
    baseline_std = train[variables].std(ddof=0).replace(0, np.nan)
    comparison = pd.DataFrame(
        {
            "训练均值": train[variables].mean(),
            "测试均值": test[variables].mean(),
            "最大残差局部": residual_window[variables].mean(),
            "最高跃迁局部": jump_window[variables].mean(),
        }
    )
    z = comparison.sub(baseline_mean, axis=0).div(baseline_std, axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10.5, 8.2), constrained_layout=True)
    lim = float(np.nanpercentile(np.abs(z.to_numpy()), 95))
    sns.heatmap(
        z,
        cmap="RdBu_r",
        center=0,
        vmin=-lim,
        vmax=lim,
        annot=True,
        fmt=".1f",
        linewidths=0.45,
        linecolor="white",
        cbar_kws={"label": "相对训练段的标准化偏移 / σ"},
        ax=ax,
    )
    ax.set_title("局部工况指纹：测试段与关键片段相对训练段的偏移")
    ax.set_xlabel("")
    ax.set_ylabel("")
    return save_fig(fig, "新颖_07_关键片段工况指纹热图.png")


def rolling_rms(values: np.ndarray, window: int) -> np.ndarray:
    delta = np.vstack([np.zeros((1, values.shape[1])), np.diff(values, axis=0)])
    energy = np.sqrt((delta**2).mean(axis=1))
    return pd.Series(energy).rolling(window=window, min_periods=1, center=True).mean().to_numpy(dtype=float)


def plot_multiscale_energy(clean: pd.DataFrame, pred: pd.DataFrame, residual_event_seq: float, jump_event_seq: float) -> Path:
    seq = clean[SEQ_COL].to_numpy(dtype=float)
    temp = temp_matrix(clean)
    p_abs = pressure_abs_matrix(clean)
    windows = [3, 5, 15, 30, 60, 120]
    labels = [f"{w * 2}s" for w in windows]
    temp_energy = np.vstack([rolling_rms(temp, w) for w in windows])
    pressure_energy = np.vstack([rolling_rms(p_abs, w) for w in windows])

    temp_energy = np.apply_along_axis(lambda x: gaussian_filter1d(x, sigma=2.0), axis=1, arr=temp_energy)
    pressure_energy = np.apply_along_axis(lambda x: gaussian_filter1d(x, sigma=2.0), axis=1, arr=pressure_energy)

    fig, axes = plt.subplots(2, 1, figsize=(17, 7.8), sharex=True, constrained_layout=True)
    for ax, data, title, cmap in [
        (axes[0], temp_energy, "温度变化多尺度能量", "magma"),
        (axes[1], pressure_energy, "负压变化多尺度能量", "cividis"),
    ]:
        im = ax.imshow(data, aspect="auto", origin="lower", extent=[seq.min(), seq.max(), -0.5, len(windows) - 0.5], cmap=cmap)
        fig.colorbar(im, ax=ax, pad=0.01, label="滚动RMS变化量")
        ax.axvline(float(pred[SEQ_COL].min()), color="#111111", linestyle="--", linewidth=1.0, label="测试段起点")
        ax.axvline(residual_event_seq, color=PALETTE["sky"], linestyle="-", linewidth=1.2, label="最大残差")
        ax.axvline(jump_event_seq, color=PALETTE["green"], linestyle="-", linewidth=1.2, label="最高跃迁")
        ax.set_yticks(range(len(windows)))
        ax.set_yticklabels(labels)
        ax.set_ylabel("平滑窗口")
        ax.set_title(title)
        ax.legend(loc="upper right", ncol=3, fontsize=8, frameon=True)
        ax.grid(False)
    axes[-1].set_xlabel("序列")
    return save_fig(fig, "新颖_08_多尺度波动能量图.png")


def write_index(paths: list[Path], residual_event_seq: float, jump_event_seq: float) -> Path:
    lines = [
        "# 第一问局部新颖PNG图表索引",
        "",
        "## 设计思路",
        "- 上一批图主要覆盖整体时空概览、模型比较和基础诊断；本批图转向局部、派生状态和机制解释。",
        "- 图表均读取现有数据与预测结果，不重新训练模型，可直接作为论文正文或附录的 PNG 图。",
        f"- 最大系统性残差片段中心序列：{residual_event_seq:.0f}。",
        f"- 测试段最高温度跃迁片段中心序列：{jump_event_seq:.0f}。",
        "",
        "## 图表清单",
    ]
    notes = {
        "新颖_01_热状态迁移轨迹与负压重心耦合.png": "把温度峰值位置、热重心、负压重心放在同一时空框架下，解释燃烧带迁移和抽风分布。",
        "新颖_02_最大残差局部病例册.png": "放大最大系统性残差窗口，联合展示残差热图、实际-预测、局部负压和事件剖面。",
        "新颖_03_最高温跃迁局部放大镜.png": "聚焦测试段最大温度突变，观察温度跃迁是否伴随负压变化和重心漂移。",
        "新颖_04_局部相图_负压温度滞回轨迹.png": "用局部轨迹展示同一负压水平下温度可能因时间状态不同而出现滞回。",
        "新颖_05_残差困难区相空间地图.png": "把行MAE投影到热重心-负压重心和前后段梯度相空间，识别模型难点工况。",
        "新颖_06_负压温度强耦合通道网络.png": "把全矩阵关系压缩成强耦合通道网络，更适合论文中讲关键路径。",
        "新颖_07_关键片段工况指纹热图.png": "比较训练均值、测试均值、最大残差片段、最高跃迁片段的标准化偏移。",
        "新颖_08_多尺度波动能量图.png": "从不同时间窗口观察温度和负压变化能量，辅助说明局部事件不是单点噪声。",
    }
    for path in paths:
        lines.append(f"- `{path.name}`：{notes.get(path.name, '')}")
    out = FIG_DIR / "图表索引.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    setup_style()
    clean, pred, by_box = load_inputs()
    state = state_frame(clean)
    pred_state = prediction_state(pred)
    residual_event_path, residual_event_seq = plot_residual_casebook(clean, pred, state, pred_state)
    jump_path, jump_event_seq = plot_jump_lens(clean, pred, state)
    paths = [
        plot_thermal_ridge(clean, pred, state),
        residual_event_path,
        jump_path,
        plot_phase_hysteresis(clean, by_box, residual_event_seq),
        plot_difficulty_phase_map(state, pred_state, residual_event_seq, jump_event_seq),
        plot_coupling_network(clean),
        zscore_fingerprint(state, float(pred[SEQ_COL].min()), residual_event_seq, jump_event_seq),
        plot_multiscale_energy(clean, pred, residual_event_seq, jump_event_seq),
    ]
    index_path = write_index(paths, residual_event_seq, jump_event_seq)
    print("已生成第一问局部新颖PNG：")
    for path in paths:
        print(path)
    print(index_path)


if __name__ == "__main__":
    main()
