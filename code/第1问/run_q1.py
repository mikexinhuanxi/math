# -*- coding: utf-8 -*-
"""
第1问：风箱温度精准预测增强版。

本脚本保留“仅风箱负压”的题意基线，同时增加精度优先的在线增强模型：
当前/历史负压 + 当前/历史机速 + 严格过去的合并温度。

核心防泄露约束：
- 最终 15% 时间段只作为最终测试集，不参与模型族或参数选择。
- 增强模型的温度特征只使用 t-1 及更早的温度。
- 不使用 CO 浓度、大烟道负压、大烟道温度。
"""

from __future__ import annotations

import json
import math
import os
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
SOURCE_XLSX = ROOT / "附件1.原始数据.xlsx"
SOURCE_MD = ROOT / "题目分析.md"
SOURCE_PDF = ROOT / "赛题A.pdf"

RAW_DIR = ROOT / "data" / "第1问" / "原始数据"
MID_DIR = ROOT / "data" / "第1问" / "中间数据"
OUT_DIR = ROOT / "data" / "第1问" / "结果数据"
FIG_DIR = ROOT / "figure" / "第1问" / "基础图"
RESULT_DIR = ROOT / "result" / "第1问"

BOX_IDS = list(range(1, 19))
SAMPLE_INTERVAL_SECONDS = 2
RANDOM_SEED = 20260529

PRESSURE_LAGS = [0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60]
SPEED_LAGS = [0, 1, 2, 3, 5, 10, 20, 30, 60]
TEMP_HISTORY_LAGS = [1, 2, 3, 5, 10, 20, 30, 60]
ROLL_MEAN_WINDOWS = [5, 15, 30]
ROLL_STD_WINDOWS = [15, 30]
SEGMENT_LAGS = [0, 10, 30, 60]
LAG_RELATION_LAGS = list(range(0, 61, 2))
SPEED_COL = "烧结机机速L1设定"

FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
]


@dataclass
class Candidate:
    name: str
    family: str
    estimator: Any
    complexity_rank: int


@dataclass
class FittedModel:
    group: str
    group_label: str
    candidate: Candidate
    feature_names: list[str]
    target_names: list[str]
    estimator: Any
    all_pred: np.ndarray


def pressure_col(i: int) -> str:
    return f"{i}#风箱负压"


def south_temp_col(i: int) -> str:
    return f"{i}#风箱南侧温度"


def north_temp_col(i: int) -> str:
    return f"{i}#风箱北侧温度"


def merged_temp_col(i: int) -> str:
    return f"{i}#风箱合并温度"


def ensure_dirs() -> None:
    for path in [RAW_DIR, MID_DIR, OUT_DIR, FIG_DIR, RESULT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def cleanup_stale_outputs() -> None:
    """Remove older ambiguous outputs that could be mistaken for the new final model."""
    stale = [
        OUT_DIR / "q1_model_coefficients_standardized.csv",
        OUT_DIR / "q1_ridge_model_parameters.npz",
        OUT_DIR / "q1_extra_trees_model.joblib",
        OUT_DIR / "q1_extra_trees_model_details.csv",
        OUT_DIR / "q1_model_comparison.csv",
        OUT_DIR / "q1_single_pressure_linear_coefficients.csv",
        OUT_DIR / "q1_chronological_stress_overall_metrics.csv",
        OUT_DIR / "q1_chronological_stress_metrics_per_box.csv",
        MID_DIR / "q1_chronological_stress_split.csv",
        FIG_DIR / "q1_pred_vs_actual_scatter.png",
        FIG_DIR / "q1_pressure_temperature_correlation.png",
        FIG_DIR / "q1_test_rmse_by_box.png",
    ]
    for path in stale:
        if path.exists():
            path.unlink()


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


def clean_and_merge(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pressure_cols = [pressure_col(i) for i in BOX_IDS]
    south_cols = [south_temp_col(i) for i in BOX_IDS]
    north_cols = [north_temp_col(i) for i in BOX_IDS]
    required = ["序列", SPEED_COL] + pressure_cols + south_cols + north_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"附件缺少第1问所需字段：{missing}")

    selected = df[required].copy()
    for col in required:
        selected[col] = pd.to_numeric(selected[col], errors="coerce")

    before_rows = len(selected)
    selected = selected.dropna(subset=required).copy()
    selected = selected.sort_values("序列").reset_index(drop=True)
    selected["time_s"] = (selected["序列"] - selected["序列"].iloc[0]) * SAMPLE_INTERVAL_SECONDS

    mapping_rows = []
    diff_rows = []
    for i in BOX_IDS:
        s_col = south_temp_col(i)
        n_col = north_temp_col(i)
        m_col = merged_temp_col(i)
        d_col = f"{i}#风箱南北温差"
        selected[m_col] = selected[[s_col, n_col]].mean(axis=1)
        selected[d_col] = selected[s_col] - selected[n_col]
        mapping_rows.append(
            {
                "风箱编号": i,
                "负压字段": pressure_col(i),
                "南侧温度字段": s_col,
                "北侧温度字段": n_col,
                "合并温度字段": m_col,
                "合并规则": "南侧温度与北侧温度的算术平均值",
                "单位": "℃",
            }
        )
        diff = selected[d_col]
        diff_rows.append(
            {
                "风箱编号": i,
                "南北温差均值": float(diff.mean()),
                "南北温差绝对值均值": float(diff.abs().mean()),
                "南北温差绝对值中位数": float(diff.abs().median()),
                "南北温差绝对值95分位": float(diff.abs().quantile(0.95)),
                "南北温差绝对值最大值": float(diff.abs().max()),
            }
        )

    merged_cols = [merged_temp_col(i) for i in BOX_IDS]
    diff_cols = [f"{i}#风箱南北温差" for i in BOX_IDS]
    clean_cols = ["序列", "time_s", SPEED_COL] + pressure_cols + south_cols + north_cols + merged_cols + diff_cols
    clean_df = selected[clean_cols].copy()
    mapping_df = pd.DataFrame(mapping_rows)
    diff_df = pd.DataFrame(diff_rows)

    metadata = {
        "source_file": str(SOURCE_XLSX),
        "sheet_name": "co_levels_comparison_final（1）",
        "raw_rows_after_header": int(before_rows),
        "clean_rows": int(len(clean_df)),
        "dropped_rows_due_to_missing": int(before_rows - len(clean_df)),
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "temperature_merge_rule": "T_merge_i = (T_south_i + T_north_i) / 2",
        "pressure_fields": pressure_cols,
        "speed_field": SPEED_COL,
        "merged_temperature_fields": merged_cols,
    }
    (MID_DIR / "q1_cleaning_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return clean_df, mapping_df, diff_df


def build_features(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    pressure_cols = [pressure_col(i) for i in BOX_IDS]
    target_cols = [merged_temp_col(i) for i in BOX_IDS]
    feature_parts: dict[str, pd.Series] = {}
    groups = {"M0_pressure_only": [], "M1_pressure_speed": [], "M2_enhanced_online": []}

    def add_feature(name: str, series: pd.Series, feature_groups: list[str]) -> None:
        feature_parts[name] = series
        for group in feature_groups:
            groups[group].append(name)

    pressure_groups = ["M0_pressure_only", "M1_pressure_speed", "M2_enhanced_online"]
    speed_groups = ["M1_pressure_speed", "M2_enhanced_online"]
    temp_groups = ["M2_enhanced_online"]

    for lag in PRESSURE_LAGS:
        for col in pressure_cols:
            add_feature(f"{col}_lag{lag}", clean_df[col].shift(lag), pressure_groups)

    for col in pressure_cols:
        add_feature(f"{col}_diff1", clean_df[col] - clean_df[col].shift(1), pressure_groups)
        add_feature(f"{col}_diff5", clean_df[col] - clean_df[col].shift(5), pressure_groups)
        add_feature(f"{col}_diff10", clean_df[col] - clean_df[col].shift(10), pressure_groups)
        for window in ROLL_MEAN_WINDOWS:
            add_feature(
                f"{col}_roll{window}_mean",
                clean_df[col].rolling(window=window, min_periods=window).mean(),
                pressure_groups,
            )
        for window in ROLL_STD_WINDOWS:
            add_feature(
                f"{col}_roll{window}_std",
                clean_df[col].rolling(window=window, min_periods=window).std(),
                pressure_groups,
            )
        add_feature(f"{col}_sq", clean_df[col] ** 2, pressure_groups)

    for lag in SEGMENT_LAGS:
        lagged = clean_df[pressure_cols].shift(lag)
        add_feature(f"全风箱负压均值_lag{lag}", lagged.mean(axis=1), pressure_groups)
        add_feature(f"全风箱负压标准差_lag{lag}", lagged.std(axis=1), pressure_groups)
        add_feature(
            f"前段风箱负压均值_lag{lag}",
            lagged[[pressure_col(i) for i in range(1, 7)]].mean(axis=1),
            pressure_groups,
        )
        add_feature(
            f"中段风箱负压均值_lag{lag}",
            lagged[[pressure_col(i) for i in range(7, 13)]].mean(axis=1),
            pressure_groups,
        )
        add_feature(
            f"后段风箱负压均值_lag{lag}",
            lagged[[pressure_col(i) for i in range(13, 19)]].mean(axis=1),
            pressure_groups,
        )

    for lag in SPEED_LAGS:
        add_feature(f"{SPEED_COL}_lag{lag}", clean_df[SPEED_COL].shift(lag), speed_groups)
    add_feature(f"{SPEED_COL}_diff1", clean_df[SPEED_COL] - clean_df[SPEED_COL].shift(1), speed_groups)
    add_feature(f"{SPEED_COL}_diff5", clean_df[SPEED_COL] - clean_df[SPEED_COL].shift(5), speed_groups)
    for window in ROLL_MEAN_WINDOWS:
        add_feature(
            f"{SPEED_COL}_roll{window}_mean",
            clean_df[SPEED_COL].rolling(window=window, min_periods=window).mean(),
            speed_groups,
        )

    for lag in TEMP_HISTORY_LAGS:
        for col in target_cols:
            add_feature(f"{col}_hist_lag{lag}", clean_df[col].shift(lag), temp_groups)
    shifted_targets = clean_df[target_cols].shift(1)
    for col in target_cols:
        for window in ROLL_MEAN_WINDOWS:
            add_feature(
                f"{col}_hist_roll{window}_mean",
                shifted_targets[col].rolling(window=window, min_periods=window).mean(),
                temp_groups,
            )
        for window in ROLL_STD_WINDOWS:
            add_feature(
                f"{col}_hist_roll{window}_std",
                shifted_targets[col].rolling(window=window, min_periods=window).std(),
                temp_groups,
            )

    features = pd.DataFrame(feature_parts)
    full = pd.concat([clean_df[["序列", "time_s"]], features, clean_df[target_cols]], axis=1)
    usable = full.dropna().reset_index(drop=True)
    feature_names = list(feature_parts.keys())
    feature_df = usable[["序列", "time_s"] + feature_names].copy()
    target_df = usable[["序列", "time_s"] + target_cols].copy()
    return feature_df, target_df, groups


def split_indices(n: int) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    idx = np.arange(n)
    dev_end = int(n * 0.85)
    splits = {
        "dev_train": idx[:dev_end],
        "final_test": idx[dev_end:],
    }
    folds = []
    for fold_id, (train_pct, val_pct) in enumerate([(0.55, 0.65), (0.65, 0.75), (0.75, 0.85)], start=1):
        train_end = int(n * train_pct)
        val_end = int(n * val_pct)
        folds.append(
            {
                "fold": fold_id,
                "train_idx": idx[:train_end],
                "val_idx": idx[train_end:val_end],
                "train_range": f"0-{int(train_pct * 100)}%",
                "val_range": f"{int(train_pct * 100)}-{int(val_pct * 100)}%",
            }
        )
    return splits, folds


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
        std = float(np.std(y_true[:, j], ddof=0))
        temp_range = float(np.max(y_true[:, j]) - np.min(y_true[:, j]))
        rows.append(
            {
                "风箱编号": j + 1,
                "目标字段": name,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "R2": m["R2"],
                "测试标准差": std,
                "测试范围": temp_range,
                "RMSE_over_std": float(m["RMSE"] / std) if std > 0 else float("nan"),
                "MAE_over_range": float(m["MAE"] / temp_range) if temp_range > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def macro_metric_dict(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> dict[str, float]:
    per_box = per_target_metrics(y_true, y_pred, target_names)
    pooled = metric_dict(y_true, y_pred)
    return {
        "MAE_macro": float(per_box["MAE"].mean()),
        "RMSE_macro": float(per_box["RMSE"].mean()),
        "R2_macro_mean": float(per_box["R2"].mean()),
        "R2_macro_median": float(per_box["R2"].median()),
        "R2_negative_boxes": int((per_box["R2"] < -1e-12).sum()),
        "RMSE_over_std_macro": float(per_box["RMSE_over_std"].mean()),
        "MAE_pooled": pooled["MAE"],
        "RMSE_pooled": pooled["RMSE"],
        "R2_pooled": pooled["R2"],
    }


def candidates() -> list[Candidate]:
    return [
        Candidate(
            name="Ridge_alpha10",
            family="Ridge",
            estimator=make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            complexity_rank=1,
        ),
        Candidate(
            name="ElasticNet_multitask_alpha005_l1_02",
            family="ElasticNet",
            estimator=make_pipeline(
                StandardScaler(),
                MultiTaskElasticNet(
                    alpha=0.05,
                    l1_ratio=0.2,
                    max_iter=5000,
                    tol=1e-3,
                    random_state=RANDOM_SEED,
                ),
            ),
            complexity_rank=2,
        ),
        Candidate(
            name="ExtraTrees_constrained",
            family="ExtraTrees",
            estimator=ExtraTreesRegressor(
                n_estimators=80,
                max_depth=8,
                min_samples_leaf=8,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            complexity_rank=3,
        ),
        Candidate(
            name="HistGradientBoosting",
            family="HistGradientBoosting",
            estimator=MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    max_iter=60,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=0.1,
                    random_state=RANDOM_SEED,
                )
            ),
            complexity_rank=4,
        ),
    ]


def fit_predict(estimator: Any, x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray) -> tuple[Any, np.ndarray]:
    model = clone(estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    pred = model.predict(x_pred)
    return model, np.asarray(pred, dtype=float)


def evaluate_candidates(
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    folds: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_names = [merged_temp_col(i) for i in BOX_IDS]
    y = target_df[target_names].to_numpy(dtype=float)
    fold_rows: list[dict[str, Any]] = []
    group_labels = {
        "M0_pressure_only": "仅负压历史",
        "M1_pressure_speed": "负压历史+机速历史",
        "M2_enhanced_online": "负压历史+机速历史+过去温度",
    }

    for group, feature_names in feature_groups.items():
        x = feature_df[feature_names].to_numpy(dtype=float)
        for cand in candidates():
            for fold in folds:
                train_idx = fold["train_idx"]
                val_idx = fold["val_idx"]
                _, pred = fit_predict(cand.estimator, x[train_idx], y[train_idx], x[val_idx])
                metrics = macro_metric_dict(y[val_idx], pred, target_names)
                fold_rows.append(
                    {
                        "模型组": group,
                        "模型组说明": group_labels[group],
                        "候选模型": cand.name,
                        "模型族": cand.family,
                        "复杂度排序": cand.complexity_rank,
                        "折号": fold["fold"],
                        "训练区间": fold["train_range"],
                        "验证区间": fold["val_range"],
                        **metrics,
                    }
                )

    fold_df = pd.DataFrame(fold_rows)
    agg = (
        fold_df.groupby(["模型组", "模型组说明", "候选模型", "模型族", "复杂度排序"], as_index=False)
        .agg(
            rolling_MAE_macro=("MAE_macro", "mean"),
            rolling_RMSE_macro=("RMSE_macro", "mean"),
            rolling_R2_macro_mean=("R2_macro_mean", "mean"),
            rolling_R2_macro_median=("R2_macro_median", "mean"),
            rolling_R2_negative_boxes=("R2_negative_boxes", "mean"),
            rolling_RMSE_over_std_macro=("RMSE_over_std_macro", "mean"),
        )
        .sort_values(["模型组", "rolling_RMSE_macro", "rolling_R2_negative_boxes", "复杂度排序"])
        .reset_index(drop=True)
    )
    return fold_df, agg


def choose_best(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, group_df in summary_df.groupby("模型组", sort=False):
        best_rmse = float(group_df["rolling_RMSE_macro"].min())
        tied = group_df[group_df["rolling_RMSE_macro"] <= best_rmse * 1.02].copy()
        tied = tied.sort_values(["rolling_R2_negative_boxes", "复杂度排序", "rolling_RMSE_macro"])
        row = tied.iloc[0].to_dict()
        row["组内入选"] = True
        rows.append(row)
    best_by_group = pd.DataFrame(rows)
    overall_rmse = float(best_by_group["rolling_RMSE_macro"].min())
    tied_overall = best_by_group[best_by_group["rolling_RMSE_macro"] <= overall_rmse * 1.02].copy()
    tied_overall = tied_overall.sort_values(["rolling_R2_negative_boxes", "复杂度排序", "rolling_RMSE_macro"])
    selected = tied_overall.iloc[0]
    best_by_group["最终推荐"] = (
        (best_by_group["模型组"] == selected["模型组"]) & (best_by_group["候选模型"] == selected["候选模型"])
    )
    return best_by_group


def candidate_by_name(name: str) -> Candidate:
    for cand in candidates():
        if cand.name == name:
            return cand
    raise KeyError(f"未知候选模型：{name}")


def train_group_best_models(
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    best_by_group: pd.DataFrame,
    dev_idx: np.ndarray,
) -> dict[str, FittedModel]:
    target_names = [merged_temp_col(i) for i in BOX_IDS]
    y = target_df[target_names].to_numpy(dtype=float)
    fitted: dict[str, FittedModel] = {}
    for _, row in best_by_group.iterrows():
        group = str(row["模型组"])
        cand = candidate_by_name(str(row["候选模型"]))
        feature_names = feature_groups[group]
        x = feature_df[feature_names].to_numpy(dtype=float)
        model, all_pred = fit_predict(cand.estimator, x[dev_idx], y[dev_idx], x)
        fitted[group] = FittedModel(
            group=group,
            group_label=str(row["模型组说明"]),
            candidate=cand,
            feature_names=feature_names,
            target_names=target_names,
            estimator=model,
            all_pred=all_pred,
        )
    return fitted


def evaluate_all_candidates_on_final_test(
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    dev_idx: np.ndarray,
    final_test_idx: np.ndarray,
) -> pd.DataFrame:
    """Post-hoc final-test diagnostics for every candidate; not used for model selection."""
    target_names = [merged_temp_col(i) for i in BOX_IDS]
    y = target_df[target_names].to_numpy(dtype=float)
    group_labels = {
        "M0_pressure_only": "仅负压历史",
        "M1_pressure_speed": "负压历史+机速历史",
        "M2_enhanced_online": "负压历史+机速历史+过去温度",
    }
    rows = []
    for group, feature_names in feature_groups.items():
        if group != "M2_enhanced_online":
            continue
        x = feature_df[feature_names].to_numpy(dtype=float)
        for cand in candidates():
            _, pred = fit_predict(cand.estimator, x[dev_idx], y[dev_idx], x[final_test_idx])
            rows.append(
                {
                    "模型组": group,
                    "模型组说明": group_labels[group],
                    "候选模型": cand.name,
                    "模型族": cand.family,
                    "说明": "最终测试集后验诊断，不参与选模",
                    **macro_metric_dict(y[final_test_idx], pred, target_names),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["模型组", "RMSE_macro", "R2_negative_boxes"], ascending=[True, True, True]
    )


def relationship_table(clean_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_id in BOX_IDS:
        t = clean_df[merged_temp_col(target_id)].to_numpy(dtype=float)
        for source_id in BOX_IDS:
            p = clean_df[pressure_col(source_id)].to_numpy(dtype=float)
            best_lag = 0
            best_corr = float(np.corrcoef(p, t)[0, 1])
            for lag in LAG_RELATION_LAGS:
                if lag == 0:
                    p_lag = p
                    t_now = t
                else:
                    p_lag = p[:-lag]
                    t_now = t[lag:]
                corr = float(np.corrcoef(p_lag, t_now)[0, 1])
                if np.isfinite(corr) and abs(corr) > abs(best_corr):
                    best_lag = lag
                    best_corr = corr
            rows.append(
                {
                    "目标风箱编号": target_id,
                    "源负压风箱编号": source_id,
                    "目标字段": merged_temp_col(target_id),
                    "源负压字段": pressure_col(source_id),
                    "最佳滞后_样本": int(best_lag),
                    "最佳滞后_秒": int(best_lag * SAMPLE_INTERVAL_SECONDS),
                    "最佳滞后相关系数": best_corr,
                    "最佳滞后绝对相关": abs(best_corr),
                }
            )
    return pd.DataFrame(rows)


def save_model_outputs(
    clean_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    split_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    best_by_group: pd.DataFrame,
    all_candidate_final_df: pd.DataFrame,
    fitted: dict[str, FittedModel],
    final_group: str,
    final_test_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_names = [merged_temp_col(i) for i in BOX_IDS]
    y = target_df[target_names].to_numpy(dtype=float)

    final_rows = []
    metrics_frames = []
    normalized_frames = []
    for group, model in fitted.items():
        test_pred = model.all_pred[final_test_idx]
        test_metrics = macro_metric_dict(y[final_test_idx], test_pred, target_names)
        final_rows.append(
            {
                "模型组": group,
                "模型组说明": model.group_label,
                "候选模型": model.candidate.name,
                "模型族": model.candidate.family,
                "数据集": "final_test",
                **test_metrics,
            }
        )
        per_box = per_target_metrics(y[final_test_idx], test_pred, target_names)
        per_box.insert(0, "模型组", group)
        per_box.insert(1, "模型组说明", model.group_label)
        per_box.insert(2, "候选模型", model.candidate.name)
        metrics_frames.append(per_box)
        normalized_frames.append(
            per_box[
                [
                    "模型组",
                    "模型组说明",
                    "候选模型",
                    "风箱编号",
                    "目标字段",
                    "RMSE",
                    "测试标准差",
                    "测试范围",
                    "RMSE_over_std",
                    "MAE_over_range",
                    "R2",
                ]
            ]
        )

    final_summary = pd.DataFrame(final_rows)
    final_per_box = pd.concat(metrics_frames, ignore_index=True)
    normalized_df = pd.concat(normalized_frames, ignore_index=True)

    final_model = fitted[final_group]
    pred_rows = target_df[["序列", "time_s"]].copy()
    pred_rows = pred_rows.merge(split_df[["序列", "split", "prediction_type"]], on="序列", how="left")
    for j, name in enumerate(target_names):
        pred_rows[f"{name}_真实值"] = y[:, j]
        pred_rows[f"{name}_预测值"] = final_model.all_pred[:, j]
        pred_rows[f"{name}_残差"] = final_model.all_pred[:, j] - y[:, j]

    final_test_predictions = pred_rows[pred_rows["split"] == "final_test"].copy()

    folds_df.to_csv(OUT_DIR / "q1_rolling_validation_metrics.csv", index=False, encoding="utf-8-sig")
    rolling_summary.to_csv(OUT_DIR / "q1_rolling_validation_summary.csv", index=False, encoding="utf-8-sig")
    best_by_group.to_csv(OUT_DIR / "q1_selected_models_by_group.csv", index=False, encoding="utf-8-sig")
    all_candidate_final_df.to_csv(
        OUT_DIR / "q1_all_candidate_final_test_comparison.csv", index=False, encoding="utf-8-sig"
    )
    final_summary.to_csv(OUT_DIR / "q1_enhanced_model_comparison.csv", index=False, encoding="utf-8-sig")
    final_per_box.to_csv(OUT_DIR / "q1_final_test_metrics_per_box.csv", index=False, encoding="utf-8-sig")
    normalized_df.to_csv(OUT_DIR / "q1_normalized_error_per_box.csv", index=False, encoding="utf-8-sig")
    pred_rows.to_csv(OUT_DIR / "q1_predictions_all.csv", index=False, encoding="utf-8-sig")
    final_test_predictions.to_csv(OUT_DIR / "q1_predictions_test.csv", index=False, encoding="utf-8-sig")

    final_only_per_box = final_per_box[final_per_box["模型组"] == final_group].copy()
    final_only_per_box.insert(0, "数据集", "final_test")
    final_only_per_box.to_csv(OUT_DIR / "q1_metrics_per_box.csv", index=False, encoding="utf-8-sig")
    final_summary[final_summary["模型组"] == final_group].to_csv(
        OUT_DIR / "q1_overall_metrics.csv", index=False, encoding="utf-8-sig"
    )
    final_summary.to_csv(OUT_DIR / "q1_model_macro_summary.csv", index=False, encoding="utf-8-sig")

    joblib.dump(
        {
            "model_group": final_model.group,
            "model_group_label": final_model.group_label,
            "candidate": final_model.candidate.name,
            "family": final_model.candidate.family,
            "feature_names": final_model.feature_names,
            "target_names": final_model.target_names,
            "estimator": final_model.estimator,
            "leakage_guard": "temperature history features use t-1 and earlier only",
        },
        OUT_DIR / "q1_final_enhanced_model.joblib",
    )
    if "M0_pressure_only" in fitted:
        base = fitted["M0_pressure_only"]
        joblib.dump(
            {
                "model_group": base.group,
                "model_group_label": base.group_label,
                "candidate": base.candidate.name,
                "family": base.candidate.family,
                "feature_names": base.feature_names,
                "target_names": base.target_names,
                "estimator": base.estimator,
            },
            OUT_DIR / "q1_pressure_only_baseline_model.joblib",
        )
    ridge_rows = rolling_summary[rolling_summary["模型族"] == "Ridge"].copy()
    ridge_rows.to_csv(OUT_DIR / "q1_ridge_baseline_validation_summary.csv", index=False, encoding="utf-8-sig")

    schema = {
        "final_model_group": final_model.group,
        "final_model_group_label": final_model.group_label,
        "final_candidate": final_model.candidate.name,
        "final_family": final_model.candidate.family,
        "target_names": target_names,
        "feature_count": len(final_model.feature_names),
        "feature_names": final_model.feature_names,
        "feature_groups": {k: {"feature_count": len(v), "feature_names": v} for k, v in fitted_feature_groups.items()},
        "pressure_lags_samples": PRESSURE_LAGS,
        "speed_lags_samples": SPEED_LAGS,
        "temperature_history_lags_samples": TEMP_HISTORY_LAGS,
        "temperature_history_lags_seconds": [lag * SAMPLE_INTERVAL_SECONDS for lag in TEMP_HISTORY_LAGS],
        "leakage_controls": [
            "final 15% time block is never used for model selection",
            "M2 temperature features use only t-1 and earlier merged temperatures",
            "CO concentration and main flue variables are excluded from question 1",
            "q1_predictions_all.csv contains in-sample predictions on the development segment",
            "R2_pooled is auxiliary and not a primary accuracy conclusion",
        ],
        "split": {
            "method": "rolling validation inside first 85%, final 15% held out once",
            "development_rows": int(len(split_df[split_df["split"] == "dev_train"])),
            "final_test_rows": int(len(split_df[split_df["split"] == "final_test"])),
            "rolling_folds": [
                {"fold": int(row["折号"]), "train_range": row["训练区间"], "val_range": row["验证区间"]}
                for _, row in folds_df[["折号", "训练区间", "验证区间"]].drop_duplicates().iterrows()
            ],
        },
        "model_artifact": str((OUT_DIR / "q1_final_enhanced_model.joblib").relative_to(ROOT)),
        "pressure_only_baseline_artifact": str((OUT_DIR / "q1_pressure_only_baseline_model.joblib").relative_to(ROOT)),
    }
    (OUT_DIR / "q1_model_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_summary, final_per_box, normalized_df, pred_rows


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def color_interp(value: float, min_value: float, max_value: float, colors: tuple[tuple[int, int, int], tuple[int, int, int]]) -> tuple[int, int, int]:
    if max_value <= min_value:
        ratio = 0.5
    else:
        ratio = max(0.0, min(1.0, (value - min_value) / (max_value - min_value)))
    return tuple(int(colors[0][i] + ratio * (colors[1][i] - colors[0][i])) for i in range(3))


def draw_bar_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output: Path,
    color: tuple[int, int, int] = (37, 99, 235),
) -> None:
    width, height = 1280, 760
    margin_l, margin_r, margin_t, margin_b = 110, 70, 110, 110
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(34)
    label_font = load_font(22)
    tick_font = load_font(18)
    draw.text((margin_l, 35), title, fill="#111827", font=title_font)
    draw.text((20, margin_t - 35), ylabel, fill="#111827", font=label_font)
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    y_max = float(data[y_col].max()) * 1.15
    y_max = y_max if y_max > 0 else 1.0
    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill="#111827", width=2)
    draw.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill="#111827", width=2)
    for k in range(5):
        y_val = y_max * k / 4
        y = margin_t + plot_h - int(plot_h * y_val / y_max)
        draw.line((margin_l - 6, y, margin_l, y), fill="#111827", width=2)
        draw.text((20, y - 12), f"{y_val:.1f}", fill="#374151", font=tick_font)
    n = len(data)
    slot = plot_w / max(n, 1)
    bar_w = slot * 0.58
    for idx, row in data.reset_index(drop=True).iterrows():
        val = float(row[y_col])
        x0 = margin_l + idx * slot + (slot - bar_w) / 2
        x1 = x0 + bar_w
        y1 = margin_t + plot_h
        y0 = y1 - plot_h * val / y_max
        draw.rectangle((x0, y0, x1, y1), fill=color)
        label = str(row[x_col]).replace("M0_pressure_only", "M0").replace("M1_pressure_speed", "M1").replace("M2_enhanced_online", "M2")
        draw.text((x0 - 8, y1 + 15), label, fill="#374151", font=tick_font)
        draw.text((x0 - 2, y0 - 28), f"{val:.2f}", fill="#111827", font=tick_font)
    img.save(output)


def generate_lag_heatmap(relationship_df: pd.DataFrame, output: Path) -> None:
    corr = relationship_df.pivot(index="目标风箱编号", columns="源负压风箱编号", values="最佳滞后绝对相关")
    lag = relationship_df.pivot(index="目标风箱编号", columns="源负压风箱编号", values="最佳滞后_秒")
    width, height = 1180, 1040
    margin_l, margin_t = 130, 120
    cell = 43
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(32)
    tick_font = load_font(15)
    small_font = load_font(12)
    draw.text((margin_l, 40), "负压-温度最佳滞后相关热图", fill="#111827", font=title_font)
    draw.text((margin_l, 80), "颜色: 绝对相关强度；数字: 最佳滞后秒数", fill="#374151", font=tick_font)
    for j, source in enumerate(BOX_IDS):
        x = margin_l + j * cell
        draw.text((x + 12, margin_t - 28), str(source), fill="#111827", font=tick_font)
    for i, target in enumerate(BOX_IDS):
        y = margin_t + i * cell
        draw.text((margin_l - 46, y + 13), str(target), fill="#111827", font=tick_font)
        for j, source in enumerate(BOX_IDS):
            value = float(corr.loc[target, source])
            lag_s = int(lag.loc[target, source])
            color = color_interp(value, 0.0, 0.8, ((239, 246, 255), (29, 78, 216)))
            x = margin_l + j * cell
            draw.rectangle((x, y, x + cell - 2, y + cell - 2), fill=color, outline="#E5E7EB")
            draw.text((x + 9, y + 14), str(lag_s), fill="#111827" if value < 0.55 else "white", font=small_font)
    draw.text((margin_l + 19 * cell, margin_t + 2 * cell), "源负压风箱", fill="#111827", font=tick_font)
    draw.text((30, margin_t + 8 * cell), "目标温度风箱", fill="#111827", font=tick_font)
    img.save(output)


def generate_residual_heatmap(pred_rows: pd.DataFrame, target_names: list[str], output: Path) -> None:
    test = pred_rows[pred_rows["split"] == "final_test"].reset_index(drop=True)
    residual = np.vstack([test[f"{name}_残差"].to_numpy(dtype=float) for name in target_names])
    abs_max = float(np.nanpercentile(np.abs(residual), 95))
    abs_max = abs_max if abs_max > 0 else 1.0
    max_cols = 180
    step = max(1, math.ceil(residual.shape[1] / max_cols))
    residual_ds = residual[:, ::step]
    width, height = 1280, 720
    margin_l, margin_t = 110, 115
    cell_w = max(4, int((width - margin_l - 80) / residual_ds.shape[1]))
    cell_h = 24
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(32)
    tick_font = load_font(16)
    draw.text((margin_l, 38), "最终测试集全风箱残差热图", fill="#111827", font=title_font)
    draw.text((margin_l, 78), "红色为高估，蓝色为低估；颜色按残差95%绝对分位截断", fill="#374151", font=tick_font)
    for i, box_id in enumerate(BOX_IDS):
        y = margin_t + i * cell_h
        draw.text((margin_l - 48, y + 3), str(box_id), fill="#111827", font=tick_font)
        for j in range(residual_ds.shape[1]):
            val = float(residual_ds[i, j])
            if val >= 0:
                color = color_interp(min(val, abs_max), 0, abs_max, ((254, 242, 242), (220, 38, 38)))
            else:
                color = color_interp(min(abs(val), abs_max), 0, abs_max, ((239, 246, 255), (37, 99, 235)))
            x = margin_l + j * cell_w
            draw.rectangle((x, y, x + cell_w, y + cell_h - 1), fill=color)
    draw.text((margin_l, margin_t + 19 * cell_h), "测试集时间顺序", fill="#111827", font=tick_font)
    draw.text((28, margin_t + 7 * cell_h), "风箱编号", fill="#111827", font=tick_font)
    img.save(output)


def generate_timeseries_plot(pred_rows: pd.DataFrame, output: Path) -> None:
    test = pred_rows[pred_rows["split"] == "final_test"].reset_index(drop=True)
    boxes = [1, 10, 18]
    width, height = 1280, 920
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(32)
    label_font = load_font(18)
    draw.text((70, 34), "最终测试集代表风箱真实值与预测值", fill="#111827", font=title_font)
    draw.text((1010, 78), "蓝:真实  红:预测", fill="#111827", font=label_font)
    panel_h = 230
    for panel, box_id in enumerate(boxes):
        top = 115 + panel * 260
        left, right = 105, 1215
        bottom = top + panel_h
        name = merged_temp_col(box_id)
        actual = test[f"{name}_真实值"].to_numpy(dtype=float)
        pred = test[f"{name}_预测值"].to_numpy(dtype=float)
        y_min = float(min(actual.min(), pred.min()))
        y_max = float(max(actual.max(), pred.max()))
        pad = (y_max - y_min) * 0.12 if y_max > y_min else 1.0
        y_min -= pad
        y_max += pad
        draw.line((left, top, left, bottom), fill="#111827", width=2)
        draw.line((left, bottom, right, bottom), fill="#111827", width=2)
        draw.text((left + 10, top - 28), name, fill="#111827", font=label_font)
        draw.text((36, top - 7), f"{y_max:.0f}", fill="#374151", font=label_font)
        draw.text((36, bottom - 15), f"{y_min:.0f}", fill="#374151", font=label_font)
        if len(actual) > 1:
            step = max(1, math.ceil(len(actual) / 260))
            xs = np.linspace(left, right, len(actual[::step]))
            for series, color in [(actual[::step], "#2563EB"), (pred[::step], "#DC2626")]:
                points = []
                for x, val in zip(xs, series):
                    y = bottom - (val - y_min) / (y_max - y_min) * (bottom - top)
                    points.append((float(x), float(y)))
                if len(points) >= 2:
                    draw.line(points, fill=color, width=2)
    draw.text((590, 872), "最终测试集序列顺序", fill="#111827", font=label_font)
    img.save(output)


def generate_figures(
    final_summary: pd.DataFrame,
    all_candidate_final_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    relationship_df: pd.DataFrame,
    pred_rows: pd.DataFrame,
    target_names: list[str],
) -> dict[str, str]:
    figure_paths = {
        "ablation": FIG_DIR / "q1_model_ablation_rmse.png",
        "candidate": FIG_DIR / "q1_m2_candidate_model_rmse.png",
        "south_north": FIG_DIR / "q1_south_north_temperature_difference.png",
        "lag_heatmap": FIG_DIR / "q1_pressure_temperature_lag_heatmap.png",
        "residual_heatmap": FIG_DIR / "q1_residual_heatmap_all_boxes.png",
        "timeseries": FIG_DIR / "q1_test_timeseries_selected.png",
    }
    draw_bar_chart(
        final_summary.sort_values("模型组"),
        "模型组",
        "RMSE_macro",
        "最终测试集模型消融 RMSE_macro",
        "RMSE_macro(℃)",
        figure_paths["ablation"],
        color=(37, 99, 235),
    )
    m2_candidates = all_candidate_final_df[all_candidate_final_df["模型组"] == "M2_enhanced_online"].copy()
    draw_bar_chart(
        m2_candidates,
        "模型族",
        "RMSE_macro",
        "增强模型不同算法最终测试 RMSE_macro",
        "RMSE_macro(℃)",
        figure_paths["candidate"],
        color=(124, 58, 237),
    )
    draw_bar_chart(
        diff_df,
        "风箱编号",
        "南北温差绝对值95分位",
        "各风箱南北温差绝对值 95 分位",
        "温差(℃)",
        figure_paths["south_north"],
        color=(5, 150, 105),
    )
    generate_lag_heatmap(relationship_df, figure_paths["lag_heatmap"])
    generate_residual_heatmap(pred_rows, target_names, figure_paths["residual_heatmap"])
    generate_timeseries_plot(pred_rows, figure_paths["timeseries"])
    return {k: str(v.relative_to(ROOT)) for k, v in figure_paths.items()}


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(format(float(value), floatfmt))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep] + rows)


def write_reports(
    clean_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    relationship_df: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    best_by_group: pd.DataFrame,
    all_candidate_final_df: pd.DataFrame,
    final_summary: pd.DataFrame,
    final_per_box: pd.DataFrame,
    final_group: str,
    figure_paths: dict[str, str],
) -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_row = final_summary[final_summary["模型组"] == final_group].iloc[0]
    m0_row = final_summary[final_summary["模型组"] == "M0_pressure_only"].iloc[0]
    m2_row = final_summary[final_summary["模型组"] == "M2_enhanced_online"].iloc[0]
    rmse_improve = (m0_row["RMSE_macro"] - m2_row["RMSE_macro"]) / m0_row["RMSE_macro"] * 100
    neg_improve = int(m0_row["R2_negative_boxes"] - m2_row["R2_negative_boxes"])
    final_box = final_per_box[final_per_box["模型组"] == final_group].copy()
    worst = final_box.sort_values("RMSE", ascending=False).iloc[0]
    best = final_box.sort_values("RMSE").iloc[0]

    comp_cols = [
        "模型组",
        "模型组说明",
        "候选模型",
        "模型族",
        "RMSE_macro",
        "MAE_macro",
        "R2_macro_mean",
        "R2_macro_median",
        "R2_negative_boxes",
        "RMSE_over_std_macro",
        "R2_pooled",
    ]
    final_text = df_to_markdown(final_summary[comp_cols], floatfmt=".4f")
    m2_candidate_text = df_to_markdown(
        all_candidate_final_df[all_candidate_final_df["模型组"] == "M2_enhanced_online"][
            [
                "候选模型",
                "模型族",
                "RMSE_macro",
                "MAE_macro",
                "R2_macro_mean",
                "R2_macro_median",
                "R2_negative_boxes",
                "RMSE_over_std_macro",
            ]
        ],
        floatfmt=".4f",
    )
    selected_text = df_to_markdown(
        best_by_group[
            [
                "模型组",
                "模型组说明",
                "候选模型",
                "模型族",
                "rolling_RMSE_macro",
                "rolling_R2_negative_boxes",
                "最终推荐",
            ]
        ],
        floatfmt=".4f",
    )
    per_box_text = df_to_markdown(
        final_box[
            [
                "风箱编号",
                "目标字段",
                "MAE",
                "RMSE",
                "R2",
                "RMSE_over_std",
                "MAE_over_range",
            ]
        ],
        floatfmt=".4f",
    )
    diff_text = df_to_markdown(diff_df.round(4), floatfmt=".4f")
    relation_summary = (
        relationship_df.sort_values("最佳滞后绝对相关", ascending=False)
        .head(8)[["目标风箱编号", "源负压风箱编号", "最佳滞后_秒", "最佳滞后相关系数"]]
    )
    relation_text = df_to_markdown(relation_summary, floatfmt=".4f")

    file_rows = []
    for base in [RAW_DIR, MID_DIR, OUT_DIR, FIG_DIR, RESULT_DIR]:
        for file in sorted(base.glob("*")):
            if file.is_file() and file.name not in {"第1问.md", "第1问_结果摘要.md"}:
                file_rows.append(str(file.relative_to(ROOT)))
    file_list_text = "\n".join(f"- `{row}`" for row in file_rows)

    report = f"""# 第1问

## 1. 本问任务理解

第1问要求探索 18 个风箱负压与对应风箱温度之间的关系，并构建基于风箱负压预测对应温度的模型。为追求题目所说的“精准预测”，本版采用双路线：

- `M0_pressure_only`：严格只用风箱负压及其历史派生特征，作为题意基线和关系解释模型。
- `M2_enhanced_online`：在负压历史基础上加入烧结机机速历史和严格过去的合并温度历史，用于在线短时精准预测。

增强模型不使用 CO 浓度、大烟道负压或大烟道温度；温度历史特征只使用 `t-1` 及更早信息，不使用当前或未来温度。

## 2. 数据口径与防泄露设计

- 合并温度：`i#风箱合并温度 = (i#风箱南侧温度 + i#风箱北侧温度) / 2`，`i = 1,...,18`。
- 主划分：前 85% 样本作为开发集，后 15% 时间段作为最终测试集。
- 选模方式：只在开发集内做 3 折滚动时间验证，最终测试集只评估一次。
- 主评价：分风箱 MAE/RMSE/R2 的宏平均、负 R2 风箱数量、`RMSE/测试标准差`。`R2_pooled` 仅作辅助，不作为精准预测主结论。
- 注意：`q1_predictions_all.csv` 的开发集预测由开发集重训模型生成，属于 in-sample 预测；最终测试段才是独立评估。

清洗后共有 `{len(clean_df)}` 条原始有效样本，构造完整历史特征后用于建模的样本数见 `q1_model_features.csv`。

## 3. 南北温度合并风险

题面允许南北温度合并，但部分风箱南北差异很大。南北温差统计如下：

{diff_text}

因此，平均合并是统一建模口径，不代表南北温场完全一致；后续解释预测误差时应考虑横向温差带来的信息损失。

## 4. 时滞关系分析

本版补充了 18×18 的“源负压风箱 -> 目标温度风箱”滞后相关分析，在 0 至 120 s 内寻找绝对相关最大的滞后。绝对相关最强的若干关系如下：

{relation_text}

完整矩阵见 `data/第1问/中间数据/q1_pressure_temperature_lag_matrix.csv` 和图 `figure/第1问/基础图/q1_pressure_temperature_lag_heatmap.png`。该结果说明负压与温度存在跨风箱、跨时滞的动态关系，但相关强度并不足以支撑仅负压模型稳定精准预测全部风箱。

## 5. 滚动验证选模

开发集内使用 3 折滚动时间验证：`0-55% -> 55-65%`、`0-65% -> 65-75%`、`0-75% -> 75-85%`。每组特征比较 Ridge、ElasticNet、受约束 ExtraTrees、HistGradientBoosting。组内入选模型如下：

{selected_text}

选择规则为：先比较滚动验证 `RMSE_macro`；2% 以内视为接近，再优先选择负 R2 风箱数更少者；仍接近时选择更简单模型。

## 6. 最终测试结果

最终测试集模型消融结果如下：

{final_text}

为检查“换别的模型是否更好”，本版还把增强特征组 `M2_enhanced_online` 的四类候选算法全部用开发集重训，并在最终测试集上做后验诊断。该表不参与选模，只用于比较不同模型效果：

{m2_candidate_text}

本次最终推荐模型为 `{final_row["模型组"]}`（{final_row["模型组说明"]}，{final_row["候选模型"]}）。最终测试集：

- `MAE_macro = {final_row["MAE_macro"]:.3f}` ℃
- `RMSE_macro = {final_row["RMSE_macro"]:.3f}` ℃
- `R2_macro_mean = {final_row["R2_macro_mean"]:.3f}`
- `R2_macro_median = {final_row["R2_macro_median"]:.3f}`
- 负 R2 风箱数 = `{int(final_row["R2_negative_boxes"])}`
- `RMSE/测试标准差` 宏平均 = `{final_row["RMSE_over_std_macro"]:.3f}`

相对严格负压基线，增强在线模型 `M2_enhanced_online` 的最终测试 `RMSE_macro` 改善 `{rmse_improve:.1f}%`，负 R2 风箱数减少 `{neg_improve}` 个。若目标是题面中的“精准预测”，应采用增强在线模型；若目标是严格回答“负压与温度关系”，则以 `M0_pressure_only` 作为边界说明。

最终模型分风箱指标如下：

{per_box_text}

测试集中 RMSE 最小的是 `{int(best["风箱编号"])}`# 风箱，RMSE = `{best["RMSE"]:.3f}` ℃；RMSE 最大的是 `{int(worst["风箱编号"])}`# 风箱，RMSE = `{worst["RMSE"]:.3f}` ℃。

## 7. 基础可视化

| 图编号 | 文件路径 | 图的含义 |
|---|---|---|
| 图1 | `{figure_paths["ablation"]}` | 最终测试集三组特征消融 RMSE 对比 |
| 图2 | `{figure_paths["candidate"]}` | 增强特征组不同算法最终测试 RMSE 对比 |
| 图3 | `{figure_paths["lag_heatmap"]}` | 18×18 负压-温度最佳滞后关系热图 |
| 图4 | `{figure_paths["residual_heatmap"]}` | 最终测试集全风箱残差热图 |
| 图5 | `{figure_paths["south_north"]}` | 南北温差绝对值 95 分位 |
| 图6 | `{figure_paths["timeseries"]}` | 1#、10#、18# 风箱最终测试集真实值与预测值 |

## 8. 可传递给后续问题的信息

- `data/第1问/中间数据/q1_cleaned_wide.csv`：统一风箱温度口径和原始负压/机速数据。
- `data/第1问/结果数据/q1_model_schema.json`：最终增强模型输入特征、滞后窗口、防泄露说明和模型文件。
- `data/第1问/结果数据/q1_final_enhanced_model.joblib`：最终增强在线模型。
- `data/第1问/结果数据/q1_pressure_only_baseline_model.joblib`：严格负压基线模型。
- `data/第1问/结果数据/q1_predictions_test.csv`：最终测试集预测值与残差。

## 9. 本问生成的文件清单

{file_list_text}

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

## 模型口径

- 严格题意基线：`data/第1问/结果数据/q1_pressure_only_baseline_model.joblib`
- 精度优先增强模型：`data/第1问/结果数据/q1_final_enhanced_model.joblib`
- 增强模型可使用当前/历史负压、当前/历史机速、严格过去温度；不使用 CO 或大烟道变量。

## 注意

若第2问使用真实温度作为 CO 预测输入，建议优先读取 `q1_cleaned_wide.csv` 中的合并温度字段。若需要在线估计温度，读取增强模型；但开发集预测为 in-sample，评估精度应以最终测试集结果为准。
"""
    (RESULT_DIR / "风箱温度字段说明.md").write_text(interface, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    cleanup_stale_outputs()
    copy_source_files()

    raw_df = read_workbook()
    raw_df.to_csv(RAW_DIR / "q1_raw_extracted.csv", index=False, encoding="utf-8-sig")

    clean_df, mapping_df, diff_df = clean_and_merge(raw_df)
    clean_df.to_csv(MID_DIR / "q1_cleaned_wide.csv", index=False, encoding="utf-8-sig")
    mapping_df.to_csv(MID_DIR / "q1_temperature_field_mapping.csv", index=False, encoding="utf-8-sig")
    diff_df.to_csv(OUT_DIR / "q1_south_north_diff_summary.csv", index=False, encoding="utf-8-sig")

    feature_df, target_df, feature_groups = build_features(clean_df)
    feature_df.to_csv(MID_DIR / "q1_model_features.csv", index=False, encoding="utf-8-sig")
    target_df.to_csv(MID_DIR / "q1_model_targets.csv", index=False, encoding="utf-8-sig")
    (MID_DIR / "q1_feature_groups.json").write_text(
        json.dumps({k: {"feature_count": len(v), "feature_names": v} for k, v in feature_groups.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    splits, folds = split_indices(len(feature_df))
    split_df = feature_df[["序列", "time_s"]].copy()
    split_df["split"] = "dev_train"
    split_df.loc[splits["final_test"], "split"] = "final_test"
    split_df["prediction_type"] = np.where(split_df["split"] == "final_test", "held_out_final_test", "development_in_sample")
    split_df.to_csv(MID_DIR / "q1_train_val_test_split.csv", index=False, encoding="utf-8-sig")

    relationship_df = relationship_table(clean_df)
    relationship_df.to_csv(MID_DIR / "q1_pressure_temperature_lag_matrix.csv", index=False, encoding="utf-8-sig")
    # Keep the older filename for downstream compatibility, now as the 18x18 lag matrix.
    relationship_df.to_csv(MID_DIR / "q1_pressure_temperature_relationship.csv", index=False, encoding="utf-8-sig")

    folds_df, rolling_summary = evaluate_candidates(feature_df, target_df, feature_groups, folds)
    best_by_group = choose_best(rolling_summary)
    all_candidate_final_df = evaluate_all_candidates_on_final_test(
        feature_df, target_df, feature_groups, splits["dev_train"], splits["final_test"]
    )
    fitted = train_group_best_models(feature_df, target_df, feature_groups, best_by_group, splits["dev_train"])
    final_group = str(best_by_group[best_by_group["最终推荐"]].iloc[0]["模型组"])

    global fitted_feature_groups
    fitted_feature_groups = feature_groups
    final_summary, final_per_box, normalized_df, pred_rows = save_model_outputs(
        clean_df,
        feature_df,
        target_df,
        split_df,
        folds_df,
        rolling_summary,
        best_by_group,
        all_candidate_final_df,
        fitted,
        final_group,
        splits["final_test"],
    )

    target_names = [merged_temp_col(i) for i in BOX_IDS]
    figure_paths = generate_figures(final_summary, all_candidate_final_df, diff_df, relationship_df, pred_rows, target_names)
    write_reports(
        clean_df,
        diff_df,
        relationship_df,
        rolling_summary,
        best_by_group,
        all_candidate_final_df,
        final_summary,
        final_per_box,
        final_group,
        figure_paths,
    )

    summary = {
        "rows_raw": int(len(raw_df)),
        "rows_clean": int(len(clean_df)),
        "rows_model": int(len(feature_df)),
        "development_rows": int(len(splits["dev_train"])),
        "final_test_rows": int(len(splits["final_test"])),
        "final_model_group": final_group,
        "final_model": fitted[final_group].candidate.name,
        "validation_method": "3-fold rolling validation inside first 85%, final 15% held out",
        "final_test": final_summary[final_summary["模型组"] == final_group].iloc[0].to_dict(),
        "generated_files_root": str(ROOT),
    }
    (OUT_DIR / "q1_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
