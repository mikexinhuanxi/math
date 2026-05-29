# -*- coding: utf-8 -*-
"""
第1问：基于风箱负压预测对应风箱温度。

运行位置：项目根目录或本脚本所在目录均可。
依赖：pandas、numpy、openpyxl、Pillow。
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
SOURCE_XLSX = ROOT / "附件1.原始数据.xlsx"
SOURCE_MD = ROOT / "题目分析.md"
SOURCE_PDF = ROOT / "赛题A.pdf"

RAW_DIR = ROOT / "data" / "第1问" / "原始数据"
MID_DIR = ROOT / "data" / "第1问" / "中间数据"
OUT_DIR = ROOT / "data" / "第1问" / "结果数据"
CODE_DIR = ROOT / "code" / "第1问"
FIG_DIR = ROOT / "figure" / "第1问" / "基础图"
RESULT_DIR = ROOT / "result" / "第1问"

BOX_IDS = list(range(1, 19))
LAGS = [0, 1, 3, 5, 10]
SAMPLE_INTERVAL_SECONDS = 2
LAMBDA_GRID = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
RANDOM_SEED = 20260529
SPLIT_BLOCK_SIZE = 60
FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
]


@dataclass
class ModelResult:
    name: str
    feature_names: list[str]
    target_names: list[str]
    best_lambda: float
    x_mean: np.ndarray
    x_std: np.ndarray
    intercept: np.ndarray
    coef: np.ndarray
    val_pred: np.ndarray
    test_pred: np.ndarray
    all_pred: np.ndarray


def ensure_dirs() -> None:
    for path in [RAW_DIR, MID_DIR, OUT_DIR, CODE_DIR, FIG_DIR, RESULT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def copy_source_files() -> None:
    for source in [SOURCE_XLSX, SOURCE_MD, SOURCE_PDF]:
        if source.exists():
            shutil.copy2(source, RAW_DIR / source.name)


def read_workbook() -> pd.DataFrame:
    if not SOURCE_XLSX.exists():
        raise FileNotFoundError(f"未找到原始数据：{SOURCE_XLSX}")
    df = pd.read_excel(SOURCE_XLSX, sheet_name="co_levels_comparison_final（1）", header=1)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def pressure_col(i: int) -> str:
    return f"{i}#风箱负压"


def south_temp_col(i: int) -> str:
    return f"{i}#风箱南侧温度"


def north_temp_col(i: int) -> str:
    return f"{i}#风箱北侧温度"


def merged_temp_col(i: int) -> str:
    return f"{i}#风箱合并温度"


def clean_and_merge(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["序列"] + [pressure_col(i) for i in BOX_IDS]
    required += [south_temp_col(i) for i in BOX_IDS] + [north_temp_col(i) for i in BOX_IDS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"附件缺少第1问所需字段：{missing}")

    selected = df[required].copy()
    for col in required:
        selected[col] = pd.to_numeric(selected[col], errors="coerce")

    before_rows = len(selected)
    selected = selected.dropna(subset=required).copy()
    after_rows = len(selected)
    selected["time_s"] = (selected["序列"] - selected["序列"].iloc[0]) * SAMPLE_INTERVAL_SECONDS

    for i in BOX_IDS:
        s_col = south_temp_col(i)
        n_col = north_temp_col(i)
        selected[merged_temp_col(i)] = selected[[s_col, n_col]].mean(axis=1)
        selected[f"{i}#风箱南北温差"] = selected[s_col] - selected[n_col]

    pressure_cols = [pressure_col(i) for i in BOX_IDS]
    south_cols = [south_temp_col(i) for i in BOX_IDS]
    north_cols = [north_temp_col(i) for i in BOX_IDS]
    merged_cols = [merged_temp_col(i) for i in BOX_IDS]
    diff_cols = [f"{i}#风箱南北温差" for i in BOX_IDS]
    clean_cols = ["序列", "time_s"] + pressure_cols + south_cols + north_cols + merged_cols + diff_cols
    clean_df = selected[clean_cols].sort_values("序列").reset_index(drop=True)

    mapping_rows = []
    for i in BOX_IDS:
        mapping_rows.append(
            {
                "风箱编号": i,
                "负压字段": pressure_col(i),
                "南侧温度字段": south_temp_col(i),
                "北侧温度字段": north_temp_col(i),
                "合并温度字段": merged_temp_col(i),
                "合并规则": "南侧温度与北侧温度的算术平均值",
                "单位": "℃",
            }
        )
    mapping_df = pd.DataFrame(mapping_rows)

    metadata = {
        "source_file": str(SOURCE_XLSX),
        "sheet_name": "co_levels_comparison_final（1）",
        "raw_rows_after_header": int(before_rows),
        "clean_rows": int(after_rows),
        "dropped_rows_due_to_missing": int(before_rows - after_rows),
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "temperature_merge_rule": "T_merge_i = (T_south_i + T_north_i) / 2",
        "pressure_fields": pressure_cols,
        "merged_temperature_fields": merged_cols,
    }
    (MID_DIR / "q1_cleaning_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return clean_df, mapping_df


def build_features(clean_df: pd.DataFrame, lags: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pressure_cols = [pressure_col(i) for i in BOX_IDS]
    target_cols = [merged_temp_col(i) for i in BOX_IDS]
    feature_parts: dict[str, pd.Series] = {}

    for lag in lags:
        for col in pressure_cols:
            feature_parts[f"{col}_lag{lag}"] = clean_df[col].shift(lag)

    for col in pressure_cols:
        feature_parts[f"{col}_diff1"] = clean_df[col] - clean_df[col].shift(1)
        feature_parts[f"{col}_roll5"] = clean_df[col].rolling(window=5, min_periods=5).mean()
        feature_parts[f"{col}_sq"] = clean_df[col] ** 2

    for lag in [0, 3, 10]:
        lagged = clean_df[pressure_cols].shift(lag)
        feature_parts[f"全风箱负压均值_lag{lag}"] = lagged.mean(axis=1)
        feature_parts[f"全风箱负压标准差_lag{lag}"] = lagged.std(axis=1)
        feature_parts[f"前段风箱负压均值_lag{lag}"] = lagged[[pressure_col(i) for i in range(1, 7)]].mean(axis=1)
        feature_parts[f"中段风箱负压均值_lag{lag}"] = lagged[[pressure_col(i) for i in range(7, 13)]].mean(axis=1)
        feature_parts[f"后段风箱负压均值_lag{lag}"] = lagged[[pressure_col(i) for i in range(13, 19)]].mean(axis=1)

    features = pd.DataFrame(feature_parts)
    full = pd.concat([clean_df[["序列", "time_s"]], features, clean_df[target_cols]], axis=1)
    usable = full.dropna().reset_index(drop=True)
    feature_df = usable[["序列", "time_s"] + list(features.columns)].copy()
    target_df = usable[["序列", "time_s"] + target_cols].copy()
    return feature_df, target_df


def split_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    block_ids = np.arange(math.ceil(n / SPLIT_BLOCK_SIZE))
    rng.shuffle(block_ids)
    train_block_count = int(len(block_ids) * 0.70)
    val_block_count = int(len(block_ids) * 0.15)
    train_blocks = set(block_ids[:train_block_count])
    val_blocks = set(block_ids[train_block_count : train_block_count + val_block_count])

    labels = np.full(n, "test", dtype=object)
    for idx in range(n):
        block = idx // SPLIT_BLOCK_SIZE
        if block in train_blocks:
            labels[idx] = "train"
        elif block in val_blocks:
            labels[idx] = "val"
    train_idx = np.where(labels == "train")[0]
    val_idx = np.where(labels == "val")[0]
    test_idx = np.where(labels == "test")[0]
    return train_idx, val_idx, test_idx, labels


def chronological_split_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    idx = np.arange(n)
    labels = np.full(n, "test", dtype=object)
    labels[:train_end] = "train"
    labels[train_end:val_end] = "val"
    return idx[:train_end], idx[train_end:val_end], idx[val_end:], labels


def standardize_x(x_train: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0, ddof=0)
    x_std[x_std == 0] = 1.0
    return (x - x_mean) / x_std, x_mean, x_std


def ridge_fit(x_train_std: np.ndarray, y_train: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    x_aug = np.column_stack([np.ones(len(x_train_std)), x_train_std])
    penalty = np.eye(x_aug.shape[1]) * lam
    penalty[0, 0] = 0.0
    lhs = x_aug.T @ x_aug + penalty
    rhs = x_aug.T @ y_train
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
    return beta[0], beta[1:]


def predict_with_model(x_std: np.ndarray, intercept: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return intercept + x_std @ coef


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err**2) / denom) if denom > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def per_target_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(target_names):
        m = metric_dict(y_true[:, j], y_pred[:, j])
        rows.append(
            {
                "风箱编号": j + 1,
                "目标字段": name,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "R2": m["R2"],
            }
        )
    return pd.DataFrame(rows)


def fit_model(
    name: str,
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_names: list[str],
    target_names: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> ModelResult:
    x = feature_df[feature_names].to_numpy(dtype=float)
    y = target_df[target_names].to_numpy(dtype=float)
    x_train = x[train_idx]
    x_train_std_all, x_mean, x_std = standardize_x(x_train, x)
    y_train = y[train_idx]
    y_val = y[val_idx]

    best_lambda = LAMBDA_GRID[0]
    best_val_rmse = float("inf")
    best_intercept = None
    best_coef = None

    for lam in LAMBDA_GRID:
        intercept, coef = ridge_fit(x_train_std_all[train_idx], y_train, lam)
        val_pred = predict_with_model(x_train_std_all[val_idx], intercept, coef)
        rmse = metric_dict(y_val, val_pred)["RMSE"]
        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_lambda = lam
            best_intercept = intercept
            best_coef = coef

    assert best_intercept is not None and best_coef is not None
    all_pred = predict_with_model(x_train_std_all, best_intercept, best_coef)
    return ModelResult(
        name=name,
        feature_names=feature_names,
        target_names=target_names,
        best_lambda=float(best_lambda),
        x_mean=x_mean,
        x_std=x_std,
        intercept=best_intercept,
        coef=best_coef,
        val_pred=all_pred[val_idx],
        test_pred=all_pred[test_idx],
        all_pred=all_pred,
    )


def fit_single_pressure_baseline(
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_names: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    y = target_df[target_names].to_numpy(dtype=float)
    pred = np.zeros_like(y)
    coefs = []
    for j, i in enumerate(BOX_IDS):
        x_one = feature_df[[f"{pressure_col(i)}_lag0"]].to_numpy(dtype=float)
        x_aug = np.column_stack([np.ones(len(x_one)), x_one])
        beta = np.linalg.pinv(x_aug[train_idx].T @ x_aug[train_idx]) @ x_aug[train_idx].T @ y[train_idx, j]
        pred[:, j] = x_aug @ beta
        coefs.append({"风箱编号": i, "截距": float(beta[0]), "系数": float(beta[1])})
    return {
        "all_pred": pred,
        "val_pred": pred[val_idx],
        "test_pred": pred[test_idx],
        "coef_df": pd.DataFrame(coefs),
    }


def relationship_table(clean_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    max_lag = 20
    for i in BOX_IDS:
        p = clean_df[pressure_col(i)].to_numpy(dtype=float)
        t = clean_df[merged_temp_col(i)].to_numpy(dtype=float)
        current_corr = float(np.corrcoef(p, t)[0, 1])
        best_lag = 0
        best_corr = current_corr
        for lag in range(max_lag + 1):
            if lag == 0:
                p_lag = p
                t_now = t
            else:
                p_lag = p[:-lag]
                t_now = t[lag:]
            corr = float(np.corrcoef(p_lag, t_now)[0, 1])
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        x = np.column_stack([np.ones(len(p)), p])
        beta = np.linalg.pinv(x.T @ x) @ x.T @ t
        rows.append(
            {
                "风箱编号": i,
                "负压字段": pressure_col(i),
                "合并温度字段": merged_temp_col(i),
                "当前相关系数": current_corr,
                "最大绝对相关滞后_样本": int(best_lag),
                "最大绝对相关滞后_秒": int(best_lag * SAMPLE_INTERVAL_SECONDS),
                "该滞后相关系数": best_corr,
                "单变量线性截距": float(beta[0]),
                "单变量线性斜率": float(beta[1]),
            }
        )
    return pd.DataFrame(rows)


def save_model_outputs(
    model: ModelResult,
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    split_labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    comparison_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = target_df[model.target_names].to_numpy(dtype=float)

    pred_rows = target_df[["序列", "time_s"]].copy()
    pred_rows["split"] = split_labels
    for j, name in enumerate(model.target_names):
        pred_rows[f"{name}_真实值"] = y[:, j]
        pred_rows[f"{name}_预测值"] = model.all_pred[:, j]
        pred_rows[f"{name}_残差"] = model.all_pred[:, j] - y[:, j]
    pred_rows.to_csv(OUT_DIR / "q1_predictions_all.csv", index=False, encoding="utf-8-sig")
    pred_rows[pred_rows["split"] == "test"].to_csv(
        OUT_DIR / "q1_predictions_test.csv", index=False, encoding="utf-8-sig"
    )

    test_metrics = per_target_metrics(y[test_idx], model.test_pred, model.target_names)
    val_metrics = per_target_metrics(y[val_idx], model.val_pred, model.target_names)
    train_pred = model.all_pred[train_idx]
    train_metrics = per_target_metrics(y[train_idx], train_pred, model.target_names)

    for split_name, df_metrics in [
        ("train", train_metrics),
        ("val", val_metrics),
        ("test", test_metrics),
    ]:
        df_metrics.insert(0, "数据集", split_name)
    metrics_all = pd.concat([train_metrics, val_metrics, test_metrics], ignore_index=True)
    metrics_all.to_csv(OUT_DIR / "q1_metrics_per_box.csv", index=False, encoding="utf-8-sig")

    overall_rows = []
    for split_name, idx, pred in [
        ("train", train_idx, train_pred),
        ("val", val_idx, model.val_pred),
        ("test", test_idx, model.test_pred),
    ]:
        m = metric_dict(y[idx], pred)
        overall_rows.append({"数据集": split_name, **m})
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(OUT_DIR / "q1_overall_metrics.csv", index=False, encoding="utf-8-sig")

    coef_df = pd.DataFrame(model.coef, index=model.feature_names, columns=model.target_names)
    coef_df.insert(0, "feature", coef_df.index)
    intercept_row = pd.DataFrame(
        [["intercept", *model.intercept.tolist()]],
        columns=["feature", *model.target_names],
    )
    pd.concat([intercept_row, coef_df.reset_index(drop=True)], ignore_index=True).to_csv(
        OUT_DIR / "q1_model_coefficients_standardized.csv", index=False, encoding="utf-8-sig"
    )

    np.savez_compressed(
        OUT_DIR / "q1_ridge_model_parameters.npz",
        feature_names=np.array(model.feature_names, dtype=object),
        target_names=np.array(model.target_names, dtype=object),
        x_mean=model.x_mean,
        x_std=model.x_std,
        intercept=model.intercept,
        coef=model.coef,
        best_lambda=np.array([model.best_lambda]),
        lags=np.array(LAGS),
    )
    schema = {
        "model": model.name,
        "formula": "Y_hat = intercept + ((X - x_mean) / x_std) @ coef",
        "selected_lambda": model.best_lambda,
        "feature_count": len(model.feature_names),
        "target_count": len(model.target_names),
        "feature_names": model.feature_names,
        "target_names": model.target_names,
        "lags_samples": LAGS,
        "lags_seconds": [lag * SAMPLE_INTERVAL_SECONDS for lag in LAGS],
        "requires_last_seconds": max(LAGS) * SAMPLE_INTERVAL_SECONDS,
        "split": {
            "method": "continuous block random split",
            "block_size_samples": SPLIT_BLOCK_SIZE,
            "random_seed": RANDOM_SEED,
            "train_rows": int(len(train_idx)),
            "val_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
        },
    }
    (OUT_DIR / "q1_model_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    comparison_df.to_csv(OUT_DIR / "q1_model_comparison.csv", index=False, encoding="utf-8-sig")
    return metrics_all, overall_df, pred_rows


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: str = "#3A3A3A") -> None:
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=color, width=2)
    draw.line((x0, y0, x0, y1), fill=color, width=2)


def scale_points(
    xs: np.ndarray,
    ys: np.ndarray,
    box: tuple[int, int, int, int],
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box
    xmin, xmax = xlim if xlim is not None else (float(np.nanmin(xs)), float(np.nanmax(xs)))
    ymin, ymax = ylim if ylim is not None else (float(np.nanmin(ys)), float(np.nanmax(ys)))
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1
    pts = []
    for x, y in zip(xs, ys):
        px = x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)
        py = y1 - (y - ymin) / (ymax - ymin) * (y1 - y0)
        pts.append((px, py))
    return pts


def draw_line_chart(
    path: Path,
    title: str,
    xs: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    y_label: str,
    y_lim: tuple[float, float] | None = None,
) -> None:
    img = Image.new("RGB", (1300, 760), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    font_title = get_font(34)
    font = get_font(22)
    small = get_font(18)
    draw.text((70, 35), title, font=font_title, fill="#111827")
    plot = (95, 115, 1225, 650)
    draw_axes(draw, plot)
    all_y = np.concatenate([np.asarray(v, dtype=float) for _, v, _ in series])
    ylim = y_lim if y_lim is not None else (float(np.nanmin(all_y)), float(np.nanmax(all_y)))
    if ylim[0] == ylim[1]:
        ylim = (ylim[0] - 1, ylim[1] + 1)
    for name, ys, color in series:
        pts = scale_points(xs, np.asarray(ys, dtype=float), plot, (float(xs.min()), float(xs.max())), ylim)
        draw.line(pts, fill=color, width=4)
        for p in pts:
            draw.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=color)
    for i in range(1, 19):
        px = plot[0] + (i - 1) / 17 * (plot[2] - plot[0])
        if i % 2 == 0 or i in [1, 18]:
            draw.text((px - 8, plot[3] + 12), str(i), font=small, fill="#374151")
    for tick in np.linspace(ylim[0], ylim[1], 5):
        py = plot[3] - (tick - ylim[0]) / (ylim[1] - ylim[0]) * (plot[3] - plot[1])
        draw.line((plot[0] - 6, py, plot[0], py), fill="#3A3A3A", width=2)
        draw.text((18, py - 10), f"{tick:.2f}", font=small, fill="#374151")
    draw.text((580, 700), "风箱编号", font=font, fill="#111827")
    draw.text((20, 80), y_label, font=font, fill="#111827")
    legend_x = 870
    for k, (name, _, color) in enumerate(series):
        y = 44 + k * 34
        draw.line((legend_x, y + 12, legend_x + 42, y + 12), fill=color, width=5)
        draw.text((legend_x + 52, y), name, font=small, fill="#111827")
    img.save(path)


def draw_bar_chart(path: Path, title: str, labels: list[int], values: np.ndarray, y_label: str) -> None:
    img = Image.new("RGB", (1300, 760), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    font_title = get_font(34)
    font = get_font(22)
    small = get_font(18)
    draw.text((70, 35), title, font=font_title, fill="#111827")
    plot = (95, 115, 1225, 650)
    draw_axes(draw, plot)
    vmax = max(float(np.max(values)) * 1.12, 1.0)
    bar_w = (plot[2] - plot[0]) / len(labels) * 0.68
    gap = (plot[2] - plot[0]) / len(labels)
    for k, (label, val) in enumerate(zip(labels, values)):
        cx = plot[0] + gap * (k + 0.5)
        h = float(val) / vmax * (plot[3] - plot[1])
        x0 = cx - bar_w / 2
        x1 = cx + bar_w / 2
        y0 = plot[3] - h
        draw.rectangle((x0, y0, x1, plot[3]), fill="#2563EB")
        if label % 2 == 0 or label in [1, 18]:
            draw.text((cx - 8, plot[3] + 12), str(label), font=small, fill="#374151")
    for tick in np.linspace(0, vmax, 5):
        py = plot[3] - tick / vmax * (plot[3] - plot[1])
        draw.line((plot[0] - 6, py, plot[0], py), fill="#3A3A3A", width=2)
        draw.text((25, py - 10), f"{tick:.1f}", font=small, fill="#374151")
    draw.text((580, 700), "风箱编号", font=font, fill="#111827")
    draw.text((20, 80), y_label, font=font, fill="#111827")
    img.save(path)


def draw_scatter(path: Path, title: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    rng = np.random.default_rng(42)
    a = y_true.reshape(-1)
    b = y_pred.reshape(-1)
    n = min(5000, len(a))
    idx = rng.choice(len(a), size=n, replace=False)
    a = a[idx]
    b = b[idx]
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    pad = (hi - lo) * 0.05
    lo -= pad
    hi += pad
    img = Image.new("RGB", (900, 860), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    font_title = get_font(30)
    font = get_font(20)
    small = get_font(16)
    draw.text((55, 35), title, font=font_title, fill="#111827")
    plot = (95, 110, 820, 750)
    draw_axes(draw, plot)
    diag = scale_points(np.array([lo, hi]), np.array([lo, hi]), plot, (lo, hi), (lo, hi))
    draw.line(diag, fill="#DC2626", width=3)
    pts = scale_points(a, b, plot, (lo, hi), (lo, hi))
    for p in pts:
        draw.ellipse((p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1), fill="#1D4ED8")
    for tick in np.linspace(lo, hi, 6):
        px = plot[0] + (tick - lo) / (hi - lo) * (plot[2] - plot[0])
        py = plot[3] - (tick - lo) / (hi - lo) * (plot[3] - plot[1])
        draw.line((px, plot[3], px, plot[3] + 5), fill="#3A3A3A", width=2)
        draw.text((px - 25, plot[3] + 12), f"{tick:.0f}", font=small, fill="#374151")
        draw.line((plot[0] - 5, py, plot[0], py), fill="#3A3A3A", width=2)
        draw.text((28, py - 8), f"{tick:.0f}", font=small, fill="#374151")
    draw.text((365, 805), "真实温度(℃)", font=font, fill="#111827")
    draw.text((18, 78), "预测温度(℃)", font=font, fill="#111827")
    draw.text((620, 120), "红线: y=x", font=small, fill="#DC2626")
    img.save(path)


def draw_timeseries(path: Path, title: str, test_sequences: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    selected = [1, 10, 18]
    n = min(240, len(test_sequences))
    img = Image.new("RGB", (1400, 1020), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    font_title = get_font(34)
    font = get_font(21)
    small = get_font(17)
    draw.text((70, 35), title, font=font_title, fill="#111827")
    for row, box_id in enumerate(selected):
        top = 120 + row * 285
        plot = (105, top, 1305, top + 210)
        draw_axes(draw, plot)
        j = box_id - 1
        xs = test_sequences[:n]
        actual = y_true[:n, j]
        pred = y_pred[:n, j]
        ymin = float(min(actual.min(), pred.min()))
        ymax = float(max(actual.max(), pred.max()))
        pad = max((ymax - ymin) * 0.08, 1.0)
        ylim = (ymin - pad, ymax + pad)
        pts_a = scale_points(xs, actual, plot, (float(xs.min()), float(xs.max())), ylim)
        pts_p = scale_points(xs, pred, plot, (float(xs.min()), float(xs.max())), ylim)
        draw.line(pts_a, fill="#1D4ED8", width=3)
        draw.line(pts_p, fill="#DC2626", width=3)
        draw.text((115, top - 30), f"{box_id}#风箱合并温度", font=font, fill="#111827")
        for tick in np.linspace(ylim[0], ylim[1], 4):
            py = plot[3] - (tick - ylim[0]) / (ylim[1] - ylim[0]) * (plot[3] - plot[1])
            draw.line((plot[0] - 6, py, plot[0], py), fill="#3A3A3A", width=2)
            draw.text((35, py - 9), f"{tick:.0f}", font=small, fill="#374151")
        draw.text((1210, top - 30), "蓝:真实  红:预测", font=small, fill="#111827")
    draw.text((625, 975), "测试集序列编号", font=font, fill="#111827")
    img.save(path)


def generate_figures(
    relationship_df: pd.DataFrame,
    test_metrics: pd.DataFrame,
    target_df: pd.DataFrame,
    model: ModelResult,
    test_idx: np.ndarray,
) -> dict[str, str]:
    fig_paths = {
        "correlation": FIG_DIR / "q1_pressure_temperature_correlation.png",
        "rmse": FIG_DIR / "q1_test_rmse_by_box.png",
        "scatter": FIG_DIR / "q1_pred_vs_actual_scatter.png",
        "timeseries": FIG_DIR / "q1_test_timeseries_selected.png",
    }

    xs = np.array(BOX_IDS)
    draw_line_chart(
        fig_paths["correlation"],
        "各风箱负压与合并温度相关系数",
        xs,
        [
            ("当前相关", relationship_df["当前相关系数"].to_numpy(dtype=float), "#1D4ED8"),
            ("最优滞后相关", relationship_df["该滞后相关系数"].to_numpy(dtype=float), "#DC2626"),
        ],
        "相关系数",
        (-1.0, 1.0),
    )

    test_only = test_metrics[test_metrics["数据集"] == "test"].sort_values("风箱编号")
    draw_bar_chart(
        fig_paths["rmse"],
        "测试集各风箱温度预测 RMSE",
        BOX_IDS,
        test_only["RMSE"].to_numpy(dtype=float),
        "RMSE(℃)",
    )

    target_names = model.target_names
    y = target_df[target_names].to_numpy(dtype=float)
    draw_scatter(
        fig_paths["scatter"],
        "测试集预测值与真实值散点图",
        y[test_idx],
        model.test_pred,
    )
    draw_timeseries(
        fig_paths["timeseries"],
        "测试集代表风箱温度真实值与预测值",
        target_df["序列"].to_numpy(dtype=float)[test_idx],
        y[test_idx],
        model.test_pred,
    )
    return {k: str(v.relative_to(ROOT)) for k, v in fig_paths.items()}


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    headers = [str(c) for c in df.columns]
    rows: list[list[str]] = []
    for _, row in df.iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                cells.append(format(float(value), floatfmt))
            elif isinstance(value, (int, np.integer)):
                cells.append(str(int(value)))
            else:
                cells.append(str(value))
        rows.append(cells)
    widths = [len(h) for h in headers]
    for cells in rows:
        widths = [max(widths[i], len(cells[i])) for i in range(len(headers))]
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("---".ljust(widths[i]) for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for cells in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])


def make_comparison(
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    dynamic_model: ModelResult,
    static_model: ModelResult,
    single_baseline: dict[str, np.ndarray],
) -> pd.DataFrame:
    mean_pred = np.repeat(y[train_idx].mean(axis=0, keepdims=True), len(y), axis=0)
    rows = []
    configs = [
        ("训练均值基准", mean_pred),
        ("对应风箱单变量线性", single_baseline["all_pred"]),
        ("全风箱当前负压岭回归", static_model.all_pred),
        ("全风箱当前+滞后负压岭回归", dynamic_model.all_pred),
    ]
    for name, pred in configs:
        for split_name, idx in [("val", val_idx), ("test", test_idx)]:
            rows.append({"模型": name, "数据集": split_name, **metric_dict(y[idx], pred[idx])})
    return pd.DataFrame(rows)


def write_reports(
    clean_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    relationship_df: pd.DataFrame,
    metrics_all: pd.DataFrame,
    overall_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    chronological_overall_df: pd.DataFrame,
    model: ModelResult,
    figure_paths: dict[str, str],
) -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    test_overall = overall_df[overall_df["数据集"] == "test"].iloc[0]
    val_overall = overall_df[overall_df["数据集"] == "val"].iloc[0]
    chrono_test = chronological_overall_df[chronological_overall_df["数据集"] == "chronological_test"].iloc[0]
    test_metrics = metrics_all[metrics_all["数据集"] == "test"].copy()
    best_row = test_metrics.sort_values("RMSE").iloc[0]
    worst_row = test_metrics.sort_values("RMSE", ascending=False).iloc[0]
    avg_current_corr = relationship_df["当前相关系数"].abs().mean()
    avg_best_corr = relationship_df["该滞后相关系数"].abs().mean()
    median_best_lag_s = relationship_df["最大绝对相关滞后_秒"].median()

    file_rows = []
    for base in [RAW_DIR, MID_DIR, OUT_DIR, CODE_DIR, FIG_DIR, RESULT_DIR]:
        for file in sorted(base.glob("*")):
            if file.is_file() and file.name != "第1问.md":
                file_rows.append(str(file.relative_to(ROOT)))
    file_list_text = "\n".join([f"- `{row}`" for row in file_rows])

    comp_text = df_to_markdown(comparison_df, floatfmt=".4f")
    top_metrics_text = df_to_markdown(test_metrics[["风箱编号", "目标字段", "MAE", "RMSE", "R2"]], floatfmt=".4f")

    report = f"""# 第1问

## 1. 本问任务理解

第1问要求探索 18 个风箱“负压”与对应风箱“温度”之间的关系，并建立只使用风箱负压信息预测对应风箱温度的模型。题面提示南北两侧温度可以合并，因此本问将每个风箱的南侧、北侧温度取算术平均，形成 `{merged_temp_col(1)}` 至 `{merged_temp_col(18)}` 这 18 个统一温度目标。

本问不预测 CO 浓度，也不做负压调控决策。

## 2. 与其他问题的关系

第1问没有对第2问或第3问的硬依赖，只依赖 `附件1.原始数据.xlsx`。本问可向第2问传递统一后的风箱温度字段口径、温度预测结果和模型接口；第2问是否采用这些结果由第2问自行决定。

## 3. 数据说明

- 原始数据：
  - `data/第1问/原始数据/附件1.原始数据.xlsx`
  - `data/第1问/原始数据/赛题A.pdf`
  - `data/第1问/原始数据/题目分析.md`
  - `data/第1问/原始数据/q1_raw_extracted.csv`
- 中间数据：
  - `data/第1问/中间数据/q1_cleaned_wide.csv`
  - `data/第1问/中间数据/q1_temperature_field_mapping.csv`
  - `data/第1问/中间数据/q1_model_features.csv`
  - `data/第1问/中间数据/q1_model_targets.csv`
  - `data/第1问/中间数据/q1_train_val_test_split.csv`
  - `data/第1问/中间数据/q1_chronological_stress_split.csv`
  - `data/第1问/中间数据/q1_pressure_temperature_relationship.csv`
- 结果数据：
  - `data/第1问/结果数据/q1_predictions_all.csv`
  - `data/第1问/结果数据/q1_predictions_test.csv`
  - `data/第1问/结果数据/q1_metrics_per_box.csv`
  - `data/第1问/结果数据/q1_overall_metrics.csv`
  - `data/第1问/结果数据/q1_model_comparison.csv`
  - `data/第1问/结果数据/q1_chronological_stress_overall_metrics.csv`
  - `data/第1问/结果数据/q1_chronological_stress_metrics_per_box.csv`
  - `data/第1问/结果数据/q1_model_schema.json`
  - `data/第1问/结果数据/q1_ridge_model_parameters.npz`
  - `data/第1问/结果数据/q1_run_summary.json`
- 字段说明：
  - 输入字段为 `1#风箱负压` 至 `18#风箱负压`，并使用这些负压字段的当前值、滞后值、差分、滚动均值、平方项和分段均值构造模型特征。
  - 输出字段为 `1#风箱合并温度` 至 `18#风箱合并温度`。
  - 合并规则为：`i#风箱合并温度 = (i#风箱南侧温度 + i#风箱北侧温度) / 2`。
- 数据清洗说明：
  - Excel 第 1 行为样本输入/输出分组，第 2 行为字段名，读取时以第 2 行作为表头。
  - 第1问所需字段均转换为数值型；本数据在所需字段中无缺失值，未删除有效观测。
  - 未对极值做截尾或替换，保留原始工况波动；仅新增时间秒、合并温度、南北温差和负压派生特征。
  - 使用最大 10 个样本滞后，即 20 s 历史负压，因此建模样本从第 11 条可形成完整特征。

## 4. 方法与模型

先用相关分析描述一一对应关系：对每个风箱计算当前负压与合并温度的皮尔逊相关系数，并在 0 至 20 个样本滞后内寻找绝对相关最大的滞后。平均当前绝对相关系数为 `{avg_current_corr:.3f}`，平均最优滞后绝对相关系数为 `{avg_best_corr:.3f}`，最优滞后的中位数为 `{median_best_lag_s:.0f}` s，说明负压与温度存在一定动态对应关系。

预测模型采用多输出岭回归：

`T_hat(t) = b + ((X(t) - mean_X) / std_X) * B`

其中 `X(t)` 只由 18 个风箱负压构造，包括当前值、1/3/5/10 个样本滞后值、一阶差分、5 点滚动均值、平方项以及前/中/后段负压均值等特征。模型正则系数在 `{LAMBDA_GRID}` 中用验证集选择，最终选择 `lambda = {model.best_lambda:g}`。

## 5. 计算过程

1. 读取 `附件1.原始数据.xlsx` 的 `co_levels_comparison_final（1）` 工作表。
2. 提取 `序列`、18 个风箱负压字段、36 个南北侧温度字段。
3. 对每个风箱合并南北温度，生成 18 个预测目标。
4. 生成由负压构造的当前与滞后特征，保留可完整形成 20 s 历史窗口的样本。
5. 按连续时间块随机划分训练集、验证集、测试集，比例约为 70%/15%/15%，使三个集合覆盖不同工况区间，同时避免单点完全随机打乱。
6. 在训练集拟合模型，在验证集选择正则系数，在测试集给出最终误差。
7. 额外做一次“前 70% 训练、后 15% 测试”的时间后段外推压力测试，用于判断跨工况外推风险。
8. 保存预测值、残差、评价指标、模型参数和基础可视化。

## 6. 结果

模型比较如下：

{comp_text}

最终模型测试集整体结果：MAE = `{test_overall["MAE"]:.3f}` ℃，RMSE = `{test_overall["RMSE"]:.3f}` ℃，R2 = `{test_overall["R2"]:.3f}`。验证集整体结果：MAE = `{val_overall["MAE"]:.3f}` ℃，RMSE = `{val_overall["RMSE"]:.3f}` ℃，R2 = `{val_overall["R2"]:.3f}`。

需要注意，整体 R2 会受到不同风箱温度水平差异较大的影响，因此分风箱 RMSE 和分风箱 R2 更适合判断单个风箱的预测效果。本问将整体指标用于比较模型，把分风箱指标用于定位误差来源。

时间后段外推压力测试的测试集结果为：MAE = `{chrono_test["MAE"]:.3f}` ℃，RMSE = `{chrono_test["RMSE"]:.3f}` ℃，R2 = `{chrono_test["R2"]:.3f}`。该结果低于连续块随机测试，说明单纯依靠历史前段数据外推到后段工况并不稳定，后续在线使用时应滚动更新或覆盖更多工况样本。

测试集分风箱指标如下：

{top_metrics_text}

测试集中 RMSE 最小的是 `{int(best_row["风箱编号"])}`# 风箱，RMSE = `{best_row["RMSE"]:.3f}` ℃；RMSE 最大的是 `{int(worst_row["风箱编号"])}`# 风箱，RMSE = `{worst_row["RMSE"]:.3f}` ℃。整体看，当前+滞后负压特征明显优于训练均值基准和单变量线性模型，说明仅看对应风箱当前负压不足以刻画温度变化，风箱间耦合和短时历史信息对温度预测有帮助。

## 7. 基础可视化

| 图编号 | 文件路径 | 图的含义 |
|---|---|---|
| 图1 | `{figure_paths["correlation"]}` | 各风箱负压与合并温度的当前相关和最优滞后相关 |
| 图2 | `{figure_paths["rmse"]}` | 测试集各风箱温度预测 RMSE |
| 图3 | `{figure_paths["scatter"]}` | 测试集所有风箱预测温度与真实温度散点对比 |
| 图4 | `{figure_paths["timeseries"]}` | 1#、10#、18# 风箱测试集真实值与预测值时序对比 |

## 8. 本问生成的文件清单

{file_list_text}

## 9. 可传递给后续问题的信息

- 可用数据：
  - `data/第1问/中间数据/q1_temperature_field_mapping.csv`：南北温度合并规则和字段映射。
  - `data/第1问/中间数据/q1_cleaned_wide.csv`：含 18 个负压、36 个原始温度、18 个合并温度和南北温差的宽表。
  - `data/第1问/结果数据/q1_predictions_all.csv`：全样本可用区间的合并温度预测值和残差。
  - `data/第1问/结果数据/q1_metrics_per_box.csv`：每个风箱的训练/验证/测试指标。
- 可用参数：
  - `data/第1问/结果数据/q1_model_schema.json`：模型输入特征、输出字段、滞后窗口和调用公式。
  - `data/第1问/结果数据/q1_model_coefficients_standardized.csv`：标准化特征空间中的模型系数。
  - `data/第1问/结果数据/q1_ridge_model_parameters.npz`：可直接读取的 numpy 模型参数。
- 可用模型：
  - 多输出岭回归温度预测模型，输入为最近 20 s 内的 18 个风箱负压序列，输出为 18 个风箱合并温度预测值。
  - 后续问题若只需要统一温度口径，可直接采用 `i#风箱合并温度 = (南侧 + 北侧) / 2`，不必调用预测模型。

生成时间：{generated_at}
"""
    (RESULT_DIR / "第1问.md").write_text(report, encoding="utf-8")
    (RESULT_DIR / "第1问_结果摘要.md").write_text(report, encoding="utf-8")

    interface = f"""# 风箱温度字段说明

生成时间：{generated_at}

负责Agent：第1问子Agent

适用问题：第1问；第2问可选择读取本文件统一风箱温度口径。

## 字段口径

每个风箱的南侧温度与北侧温度合并为一个温度字段：

`i#风箱合并温度 = (i#风箱南侧温度 + i#风箱北侧温度) / 2`

单位为 ℃，其中 `i = 1, 2, ..., 18`。

## 字段映射文件

完整字段映射见：

- `data/第1问/中间数据/q1_temperature_field_mapping.csv`

## 可读取结果

- 清洗宽表：`data/第1问/中间数据/q1_cleaned_wide.csv`
- 全样本预测：`data/第1问/结果数据/q1_predictions_all.csv`
- 模型接口：`data/第1问/结果数据/q1_model_schema.json`
- 模型参数：`data/第1问/结果数据/q1_ridge_model_parameters.npz`

## 注意

本问模型只使用风箱负压相关特征预测风箱温度，不使用 CO 浓度、大烟道参数或未来信息。若第2问使用真实温度作为 CO 预测输入，建议优先读取 `q1_cleaned_wide.csv` 中的合并温度字段；若需要在线估计温度，可读取本问预测结果或模型参数。
"""
    (RESULT_DIR / "风箱温度字段说明.md").write_text(interface, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    copy_source_files()

    raw_df = read_workbook()
    raw_df.to_csv(RAW_DIR / "q1_raw_extracted.csv", index=False, encoding="utf-8-sig")

    clean_df, mapping_df = clean_and_merge(raw_df)
    clean_df.to_csv(MID_DIR / "q1_cleaned_wide.csv", index=False, encoding="utf-8-sig")
    mapping_df.to_csv(MID_DIR / "q1_temperature_field_mapping.csv", index=False, encoding="utf-8-sig")

    feature_df, target_df = build_features(clean_df, LAGS)
    feature_df.to_csv(MID_DIR / "q1_model_features.csv", index=False, encoding="utf-8-sig")
    target_df.to_csv(MID_DIR / "q1_model_targets.csv", index=False, encoding="utf-8-sig")

    train_idx, val_idx, test_idx, split_labels = split_indices(len(feature_df))
    split_df = feature_df[["序列", "time_s"]].copy()
    split_df["split"] = split_labels
    split_df.to_csv(MID_DIR / "q1_train_val_test_split.csv", index=False, encoding="utf-8-sig")

    relationship_df = relationship_table(clean_df)
    relationship_df.to_csv(MID_DIR / "q1_pressure_temperature_relationship.csv", index=False, encoding="utf-8-sig")

    target_names = [merged_temp_col(i) for i in BOX_IDS]
    dynamic_features = [c for c in feature_df.columns if c not in ["序列", "time_s"]]
    static_features = [f"{pressure_col(i)}_lag0" for i in BOX_IDS]

    dynamic_model = fit_model(
        "current_lagged_pressure_ridge",
        feature_df,
        target_df,
        dynamic_features,
        target_names,
        train_idx,
        val_idx,
        test_idx,
    )
    static_model = fit_model(
        "current_pressure_ridge",
        feature_df,
        target_df,
        static_features,
        target_names,
        train_idx,
        val_idx,
        test_idx,
    )
    single_baseline = fit_single_pressure_baseline(feature_df, target_df, target_names, train_idx, val_idx, test_idx)
    single_baseline["coef_df"].to_csv(
        OUT_DIR / "q1_single_pressure_linear_coefficients.csv", index=False, encoding="utf-8-sig"
    )

    y = target_df[target_names].to_numpy(dtype=float)
    comparison_df = make_comparison(y, train_idx, val_idx, test_idx, dynamic_model, static_model, single_baseline)

    chrono_train_idx, chrono_val_idx, chrono_test_idx, chrono_labels = chronological_split_indices(len(feature_df))
    chrono_model = fit_model(
        "chronological_current_lagged_pressure_ridge",
        feature_df,
        target_df,
        dynamic_features,
        target_names,
        chrono_train_idx,
        chrono_val_idx,
        chrono_test_idx,
    )
    chrono_overall_rows = []
    for split_name, idx, pred in [
        ("chronological_train", chrono_train_idx, chrono_model.all_pred[chrono_train_idx]),
        ("chronological_val", chrono_val_idx, chrono_model.val_pred),
        ("chronological_test", chrono_test_idx, chrono_model.test_pred),
    ]:
        chrono_overall_rows.append({"数据集": split_name, **metric_dict(y[idx], pred)})
    chronological_overall_df = pd.DataFrame(chrono_overall_rows)
    chronological_overall_df.to_csv(
        OUT_DIR / "q1_chronological_stress_overall_metrics.csv", index=False, encoding="utf-8-sig"
    )
    chrono_per_box = per_target_metrics(y[chrono_test_idx], chrono_model.test_pred, target_names)
    chrono_per_box.insert(0, "数据集", "chronological_test")
    chrono_per_box.to_csv(
        OUT_DIR / "q1_chronological_stress_metrics_per_box.csv", index=False, encoding="utf-8-sig"
    )
    chrono_split_df = feature_df[["序列", "time_s"]].copy()
    chrono_split_df["split"] = chrono_labels
    chrono_split_df.to_csv(
        MID_DIR / "q1_chronological_stress_split.csv", index=False, encoding="utf-8-sig"
    )

    metrics_all, overall_df, _ = save_model_outputs(
        dynamic_model,
        feature_df,
        target_df,
        split_labels,
        train_idx,
        val_idx,
        test_idx,
        comparison_df,
    )
    summary = {
        "rows_raw": int(len(raw_df)),
        "rows_clean": int(len(clean_df)),
        "rows_model": int(len(feature_df)),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "best_lambda": dynamic_model.best_lambda,
        "test_overall": overall_df[overall_df["数据集"] == "test"].iloc[0].to_dict(),
        "chronological_stress_test_overall": chronological_overall_df[
            chronological_overall_df["数据集"] == "chronological_test"
        ].iloc[0].to_dict(),
        "generated_files_root": str(ROOT),
    }
    (OUT_DIR / "q1_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    figure_paths = generate_figures(relationship_df, metrics_all, target_df, dynamic_model, test_idx)
    write_reports(
        clean_df,
        mapping_df,
        relationship_df,
        metrics_all,
        overall_df,
        comparison_df,
        chronological_overall_df,
        dynamic_model,
        figure_paths,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
