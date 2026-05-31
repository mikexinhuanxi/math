# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib


ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "第3问" / "原始数据"
DATA_MID = ROOT / "data" / "第3问" / "中间数据"
DATA_OUT = ROOT / "data" / "第3问" / "结果数据"
FIG_DIR = ROOT / "figure" / "第3问" / "基础图"
RESULT_DIR = ROOT / "result" / "第3问"

Q1_RESULT_DIR = ROOT / "result" / "第一问" / "时间划分数据集+历史温度"
Q1_DATA_DIR = ROOT / "data" / "第一问"

Q2_RESULT_DIR = ROOT / "result" / "第2问"
Q2_MODEL_DIR = Q2_RESULT_DIR / "异常修正后建模"
Q2_DATA_MID = ROOT / "data" / "第2问" / "异常修正后建模" / "中间数据"
Q2_CO_CORRECTION_DETAIL = Q2_DATA_MID / "CO异常修正明细.csv"

SOURCE_XLSX_CANDIDATES = [
    ROOT / "附件1.原始数据.xlsx",
    ROOT / "data" / "第2问" / "原始数据" / "附件1.原始数据.xlsx",
    ROOT / "data" / "第3问" / "原始数据" / "附件1.原始数据.xlsx",
]
SOURCE_PDF_CANDIDATES = [
    ROOT / "赛题A.pdf",
    ROOT / "data" / "第2问" / "原始数据" / "赛题A.pdf",
    ROOT / "data" / "第3问" / "原始数据" / "赛题A.pdf",
]
SOURCE_ANALYSIS_CANDIDATES = [
    ROOT / "题目分析.md",
    ROOT / "data" / "第2问" / "原始数据" / "题目分析.md",
    ROOT / "data" / "第3问" / "原始数据" / "题目分析.md",
]

TARGET_COL = "烧结大烟道外排CO浓度"
SPEED_COL = "烧结机机速L1设定"
SAMPLE_INTERVAL_S = 2
EMISSION_STANDARD = 2800.0
NEAR_STANDARD = 3200.0
Q10_ALPHA = 0.10
Q05_ALPHA = 0.05
TAIL_NEIGHBOR_CO = 3600.0
THEORY_TEMP_DELTA_LIMIT = 90.0
ENGINEERING_TEMP_DELTA_LIMIT = 35.0
RANGE_LOW_Q = 0.10
RANGE_HIGH_Q = 0.90
MAIN_RANGE_LOW_Q = 0.05
MAIN_RANGE_HIGH_Q = 0.95
BOX_IDS = list(range(1, 19))
CO_HISTORY_PREFIX = "CO历史浓度__"
SINGLE_ADJUST_FRACTION = 0.35
SINGLE_ADJUST_MIN = 0.10
SINGLE_ADJUST_MAX = 1.20
TOTAL_BUDGET_FRACTION = 0.45
Q2_CORRECTED_CO_COL = "第二问异常修正后CO"
Q2_CO_REPAIRED_FLAG_COL = "第二问CO是否修正"
Q2_CO_SPIKE_FLAG_COL = "第二问CO局部尖峰异常"
Q2_CO_GLOBAL_FLAG_COL = "第二问CO全局界限异常"
ISOLATED_LOW_FLAG_COL = "CO孤立低谷异常"
TRUSTED_STANDARD_COL = "可信达标样本"
LOW_EMISSION_SEGMENT_COL = "低排放片段ID"
LOW_EMISSION_MIN_DURATION_S = 30
LOW_EMISSION_QUANTILE = 0.15
STABILITY_WINDOW_S = 30
REGIME_COUNT = 3
CONTROL_EFFECT_HORIZON_S = 300
ROLLING_CONTROL_WINDOW_S = 300
LOW_POTENTIAL_BASELINE_CO = 3200.0
LOW_POTENTIAL_MAX = 900.0


def ensure_dirs() -> None:
    for path in [DATA_RAW, DATA_MID, DATA_OUT, FIG_DIR, RESULT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def copy_if_exists(src: Path | None, dst: Path) -> bool:
    if src is not None and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            if dst.exists():
                return True
            shutil.copy2(src, dst)
        return True
    return False


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}_tmp_{os.getpid()}{path.suffix}")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    try:
        os.replace(tmp, path)
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_new{path.suffix}")
        os.replace(tmp, fallback)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}_tmp_{os.getpid()}{path.suffix}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    try:
        os.replace(tmp, path)
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_new{path.suffix}")
        os.replace(tmp, fallback)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_original_excel() -> pd.DataFrame:
    source_xlsx = first_existing(SOURCE_XLSX_CANDIDATES)
    if source_xlsx is None:
        raise FileNotFoundError("未找到附件1.原始数据.xlsx")
    copy_if_exists(source_xlsx, DATA_RAW / "附件1.原始数据.xlsx")
    copy_if_exists(first_existing(SOURCE_PDF_CANDIDATES), DATA_RAW / "赛题A.pdf")
    copy_if_exists(first_existing(SOURCE_ANALYSIS_CANDIDATES), DATA_RAW / "题目分析.md")
    raw = pd.read_excel(source_xlsx, sheet_name=0, header=1, engine="openpyxl")
    raw = raw.dropna(how="all").copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    try:
        save_csv(raw, DATA_RAW / "附件1_字段化原始数据.csv")
    except PermissionError:
        pass
    return raw


def clean_raw_data(raw: pd.DataFrame) -> pd.DataFrame:
    cleaned = raw.copy()
    for col in cleaned.columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    if "序列" not in cleaned.columns:
        raise KeyError("原始数据中未找到字段：序列")

    missing = cleaned.isna().sum().rename("缺失值数量").reset_index()
    missing.columns = ["字段", "缺失值数量"]
    save_csv(missing, DATA_MID / "缺失值处理统计.csv")

    cleaned = cleaned.dropna(subset=["序列"]).copy()
    cleaned["序列"] = cleaned["序列"].astype(int)
    cleaned = cleaned.sort_values("序列").drop_duplicates("序列", keep="first")
    cleaned = cleaned.interpolate(method="linear", limit_direction="both").ffill().bfill()
    cleaned["时间_s"] = (cleaned["序列"] - 1) * SAMPLE_INTERVAL_S

    for i in BOX_IDS:
        p_col = f"{i}#风箱负压"
        s_col = f"{i}#风箱南侧温度"
        n_col = f"{i}#风箱北侧温度"
        if p_col in cleaned.columns:
            cleaned[f"{i}#风箱负压强度"] = cleaned[p_col].abs()
        if s_col in cleaned.columns and n_col in cleaned.columns:
            cleaned[f"{i}#风箱平均温度"] = cleaned[[s_col, n_col]].mean(axis=1)
            cleaned[f"{i}#风箱合并温度"] = cleaned[f"{i}#风箱平均温度"]
    for i in [1, 2]:
        col = f"{i}#大烟道负压"
        if col in cleaned.columns:
            cleaned[f"{i}#大烟道负压强度"] = cleaned[col].abs()

    save_csv(cleaned, DATA_MID / "清洗后数据.csv")
    return cleaned


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    text = series.fillna(False).astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y", "是"])


def trusted_standard_mask(df: pd.DataFrame) -> pd.Series:
    if TRUSTED_STANDARD_COL in df.columns:
        return bool_series(df[TRUSTED_STANDARD_COL])
    return pd.to_numeric(df[TARGET_COL], errors="coerce") <= EMISSION_STANDARD


def apply_q2_corrected_co(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Use Q2 abnormal-corrected CO as Q3 target and tag reliable standard samples."""
    df = cleaned.copy()
    df["第三问原始CO"] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    if Q2_CO_CORRECTION_DETAIL.exists():
        detail = pd.read_csv(Q2_CO_CORRECTION_DETAIL, encoding="utf-8-sig")
        detail = detail.rename(
            columns={
                "原始CO": "第二问修正前CO",
                "修正后CO": Q2_CORRECTED_CO_COL,
                "是否修正": Q2_CO_REPAIRED_FLAG_COL,
                "局部尖峰异常": Q2_CO_SPIKE_FLAG_COL,
                "全局界限异常": Q2_CO_GLOBAL_FLAG_COL,
            }
        )
        keep_cols = [
            "序列",
            "第二问修正前CO",
            Q2_CORRECTED_CO_COL,
            "修正差值",
            Q2_CO_REPAIRED_FLAG_COL,
            Q2_CO_SPIKE_FLAG_COL,
            Q2_CO_GLOBAL_FLAG_COL,
        ]
        df = df.merge(detail[[c for c in keep_cols if c in detail.columns]], on="序列", how="left")
    else:
        df[Q2_CORRECTED_CO_COL] = df["第三问原始CO"]
        df["第二问修正前CO"] = df["第三问原始CO"]
        df["修正差值"] = 0.0
        df[Q2_CO_REPAIRED_FLAG_COL] = False
        df[Q2_CO_SPIKE_FLAG_COL] = False
        df[Q2_CO_GLOBAL_FLAG_COL] = False

    df[Q2_CORRECTED_CO_COL] = pd.to_numeric(df[Q2_CORRECTED_CO_COL], errors="coerce").fillna(df["第三问原始CO"])
    df[TARGET_COL] = df[Q2_CORRECTED_CO_COL]

    for col in [Q2_CO_REPAIRED_FLAG_COL, Q2_CO_SPIKE_FLAG_COL, Q2_CO_GLOBAL_FLAG_COL]:
        if col not in df.columns:
            df[col] = False
        df[col] = bool_series(df[col])

    standard = df[TARGET_COL] <= EMISSION_STANDARD
    run_id = standard.ne(standard.shift(fill_value=False)).cumsum()
    standard_run_len = standard.groupby(run_id).transform("sum")
    isolated_low = standard & (standard_run_len <= 1)
    abnormal = df[Q2_CO_REPAIRED_FLAG_COL] | df[Q2_CO_SPIKE_FLAG_COL] | df[Q2_CO_GLOBAL_FLAG_COL] | isolated_low

    df["CO达标样本_修正后小于等于2800"] = standard
    df[ISOLATED_LOW_FLAG_COL] = isolated_low
    df["CO达标异常剔除标记"] = abnormal
    df[TRUSTED_STANDARD_COL] = standard & ~abnormal

    summary = pd.DataFrame(
        [
            {
                "样本数": int(len(df)),
                "修正后CO小于等于2800样本数": int(standard.sum()),
                "第二问已修正CO点数": int(df[Q2_CO_REPAIRED_FLAG_COL].sum()),
                "第二问局部尖峰异常点数": int(df[Q2_CO_SPIKE_FLAG_COL].sum()),
                "第二问全局界限异常点数": int(df[Q2_CO_GLOBAL_FLAG_COL].sum()),
                "孤立低谷达标点数": int(isolated_low.sum()),
                "可信达标样本数": int(df[TRUSTED_STANDARD_COL].sum()),
                "可信达标比例_%": float(df[TRUSTED_STANDARD_COL].mean() * 100),
                "CO口径": "第三问目标列已替换为第二问异常修正后的CO；达标=修正后CO<=2800且非修正异常点、非孤立低谷。",
            }
        ]
    )
    save_csv(summary, DATA_OUT / "第二问修正CO可信达标样本摘要.csv")
    save_csv(df[["序列", "时间_s", "第三问原始CO", Q2_CORRECTED_CO_COL, "修正差值", Q2_CO_REPAIRED_FLAG_COL, Q2_CO_SPIKE_FLAG_COL, Q2_CO_GLOBAL_FLAG_COL, ISOLATED_LOW_FLAG_COL, TRUSTED_STANDARD_COL]], DATA_OUT / "第二问修正CO可信达标样本明细.csv")
    save_csv(df, DATA_MID / "清洗后数据.csv")
    save_csv(df, DATA_MID / "清洗后数据_第二问CO异常修正口径.csv")
    return df


def apply_q2_corrected_co_to_aligned(aligned: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    cols = ["序列", Q2_CORRECTED_CO_COL, Q2_CO_REPAIRED_FLAG_COL, Q2_CO_SPIKE_FLAG_COL, Q2_CO_GLOBAL_FLAG_COL, ISOLATED_LOW_FLAG_COL, TRUSTED_STANDARD_COL]
    addon = cleaned[[c for c in cols if c in cleaned.columns]].copy()
    out = aligned.drop(columns=[c for c in cols if c in aligned.columns and c != "序列"], errors="ignore").merge(addon, on="序列", how="left")
    out[Q2_CORRECTED_CO_COL] = pd.to_numeric(out[Q2_CORRECTED_CO_COL], errors="coerce").fillna(pd.to_numeric(out[TARGET_COL], errors="coerce"))
    out[TARGET_COL] = out[Q2_CORRECTED_CO_COL]
    if TRUSTED_STANDARD_COL not in out.columns:
        out[TRUSTED_STANDARD_COL] = out[TARGET_COL] <= EMISSION_STANDARD
    out[TRUSTED_STANDARD_COL] = bool_series(out[TRUSTED_STANDARD_COL])
    save_csv(out, DATA_MID / "滞后配准建模数据_第三问修正CO口径.csv")
    return out


def make_future_prediction_dataset(cleaned: pd.DataFrame) -> pd.DataFrame:
    df = cleaned.sort_values("序列").reset_index(drop=True).copy()
    horizon = max(1, CONTROL_EFFECT_HORIZON_S // SAMPLE_INTERVAL_S)
    co = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    trusted = trusted_standard_mask(df).to_numpy(dtype=bool)
    rows: list[dict[str, object]] = []
    co_lags = [0, 1, 2, 5, 10, 30, 60, 120, 150]
    start = max(max(co_lags), 15)
    end = len(df) - horizon
    for pos in range(start, end):
        future_pos = pos + horizon
        row_now = df.iloc[pos]
        row_future = df.iloc[future_pos]
        target_co = float(co[future_pos])
        item: dict[str, object] = {
            "序列": int(row_future["序列"]),
            "时间_s": float(row_future["时间_s"]),
            "控制序列": int(row_now["序列"]),
            "控制时间_s": float(row_now["时间_s"]),
            "预测提前量_s": int(CONTROL_EFFECT_HORIZON_S),
            TARGET_COL: target_co,
            Q2_CORRECTED_CO_COL: target_co,
            TRUSTED_STANDARD_COL: bool(trusted[future_pos]),
            "未来CO是否低于2800": bool(target_co <= EMISSION_STANDARD),
            "低排放潜力目标": float(np.clip(LOW_POTENTIAL_BASELINE_CO - target_co, 0.0, LOW_POTENTIAL_MAX)),
        }
        for lag in co_lags:
            item[f"{CO_HISTORY_PREFIX}lag{lag}"] = float(co[pos - lag])
        item["CO状态__rolling30_mean"] = float(np.mean(co[max(0, pos - 29) : pos + 1]))
        item["CO状态__rolling120_mean"] = float(np.mean(co[max(0, pos - 119) : pos + 1]))
        item[f"{SPEED_COL}__lag{CONTROL_EFFECT_HORIZON_S}"] = float(row_now[SPEED_COL])
        item[f"{SPEED_COL}__delta30"] = float(row_now[SPEED_COL] - df.iloc[max(0, pos - 15)][SPEED_COL])
        for i in BOX_IDS:
            p_col = f"{i}#风箱负压强度"
            t_col = f"{i}#风箱平均温度"
            item[f"{p_col}__lag{CONTROL_EFFECT_HORIZON_S}"] = float(row_now[p_col])
            item[f"{p_col}调节量__相对30s前"] = float(row_now[p_col] - df.iloc[max(0, pos - 15)][p_col])
            item[f"{p_col}调节量__相对120s均值"] = float(row_now[p_col] - df[p_col].iloc[max(0, pos - 59) : pos + 1].mean())
            item[f"{t_col}__lag{CONTROL_EFFECT_HORIZON_S}"] = float(row_now[t_col])
        for col in ["1#大烟道负压强度", "2#大烟道负压强度", "1#大烟道温度", "2#大烟道温度"]:
            if col in df.columns:
                item[f"{col}__lag{CONTROL_EFFECT_HORIZON_S}"] = float(row_now[col])
        rows.append(item)

    future = pd.DataFrame(rows)
    if future.empty:
        raise ValueError("未来300s预测数据集为空，请检查样本长度和控制时滞。")

    non_standard = ~trusted_standard_mask(future)
    order = future.loc[non_standard].sort_values("序列").index.to_list()
    n = len(order)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    future["数据集"] = "训练集"
    future.loc[order[train_end:val_end], "数据集"] = "验证集"
    future.loc[order[val_end:], "数据集"] = "测试集"
    future.loc[trusted_standard_mask(future), "数据集"] = "训练集"

    save_csv(future, DATA_MID / "未来300s低排放潜力建模数据.csv")
    split_summary = (
        future.groupby("数据集")
        .agg(
            样本数=("序列", "size"),
            可信达标样本数=(TRUSTED_STANDARD_COL, lambda s: int(bool_series(s).sum())),
            未来CO均值=(TARGET_COL, "mean"),
            未来CO中位数=(TARGET_COL, "median"),
        )
        .reset_index()
    )
    split_summary["可信达标比例_%"] = split_summary["可信达标样本数"] / split_summary["样本数"] * 100
    save_csv(split_summary, DATA_OUT / "未来300s建模数据划分摘要.csv")
    return future


def make_pressure_ranges(cleaned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in BOX_IDS:
        strength_col = f"{i}#风箱负压强度"
        raw_col = f"{i}#风箱负压"
        if strength_col not in cleaned.columns and raw_col in cleaned.columns:
            cleaned[strength_col] = cleaned[raw_col].abs()
        s = cleaned[strength_col].dropna().astype(float)
        raw = cleaned[raw_col].dropna().astype(float)
        rows.append(
            {
                "风箱编号": i,
                "原始负压字段": raw_col,
                "负压强度字段": strength_col,
                "样本数": int(s.size),
                "历史强度最小值": float(s.min()),
                "历史强度P01": float(s.quantile(0.01)),
                "扩展强度下限_P05": float(s.quantile(0.05)),
                "优化强度下限_P10": float(s.quantile(RANGE_LOW_Q)),
                "历史强度中位数": float(s.median()),
                "历史强度均值": float(s.mean()),
                "优化强度上限_P90": float(s.quantile(RANGE_HIGH_Q)),
                "扩展强度上限_P95": float(s.quantile(0.95)),
                "历史强度P99": float(s.quantile(0.99)),
                "历史强度最大值": float(s.max()),
                "历史强度标准差": float(s.std(ddof=0)),
                "优化强度范围宽度": float(s.quantile(RANGE_HIGH_Q) - s.quantile(RANGE_LOW_Q)),
                "优化原始负压下限_更负": -float(s.quantile(RANGE_HIGH_Q)),
                "优化原始负压上限_靠近0": -float(s.quantile(RANGE_LOW_Q)),
                "历史原始负压最小值": float(raw.min()),
                "历史原始负压最大值": float(raw.max()),
            }
        )
    ranges = pd.DataFrame(rows)
    save_csv(ranges, DATA_OUT / "风箱负压调节范围.csv")
    return ranges


def condition_quantile_row(part: pd.DataFrame, segment_label: object, var_name: str, col: str) -> dict[str, object]:
    s = pd.to_numeric(part[col], errors="coerce").dropna()
    if s.empty:
        return {
            "片段编号": segment_label,
            "变量": var_name,
            "字段": col,
            "样本数": 0,
            "P10": np.nan,
            "P25": np.nan,
            "中位数": np.nan,
            "P75": np.nan,
            "P90": np.nan,
            "均值": np.nan,
            "最小值": np.nan,
            "最大值": np.nan,
        }
    return {
        "片段编号": segment_label,
        "变量": var_name,
        "字段": col,
        "样本数": int(s.size),
        "P10": float(s.quantile(0.10)),
        "P25": float(s.quantile(0.25)),
        "中位数": float(s.median()),
        "P75": float(s.quantile(0.75)),
        "P90": float(s.quantile(0.90)),
        "均值": float(s.mean()),
        "最小值": float(s.min()),
        "最大值": float(s.max()),
    }


def identify_low_emission_conditions(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    df = cleaned.sort_values("序列").copy()
    back_temp_cols = [f"{i}#风箱平均温度" for i in range(13, 19) if f"{i}#风箱平均温度" in df.columns]
    if back_temp_cols:
        df["后段风箱温度均值"] = df[back_temp_cols].mean(axis=1)
    pressure_cols = [f"{i}#风箱负压强度" for i in BOX_IDS if f"{i}#风箱负压强度" in df.columns]
    if pressure_cols:
        df["风箱负压强度均值"] = df[pressure_cols].mean(axis=1)

    stability_cols = [c for c in ["1#大烟道负压强度", "2#大烟道负压强度", "1#大烟道温度", "2#大烟道温度", "后段风箱温度均值"] if c in df.columns]
    window = max(3, STABILITY_WINDOW_S // SAMPLE_INTERVAL_S)
    stability_parts = []
    for col in stability_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        rolling_std = s.rolling(window=window, min_periods=max(3, window // 3), center=True).std(ddof=0)
        scale = float(s.quantile(0.75) - s.quantile(0.25))
        if scale <= 1e-9 or not np.isfinite(scale):
            scale = float(s.std(ddof=0))
        if scale <= 1e-9 or not np.isfinite(scale):
            scale = 1.0
        stability_parts.append(rolling_std.fillna(rolling_std.median()).fillna(0.0) / scale)
    df["低排放稳定性评分"] = pd.concat(stability_parts, axis=1).mean(axis=1) if stability_parts else 0.0

    abnormal_cols = [Q2_CO_REPAIRED_FLAG_COL, Q2_CO_SPIKE_FLAG_COL, Q2_CO_GLOBAL_FLAG_COL, ISOLATED_LOW_FLAG_COL]
    abnormal_mask = pd.Series(False, index=df.index)
    for col in abnormal_cols:
        if col in df.columns:
            abnormal_mask = abnormal_mask | bool_series(df[col])
    valid = ~abnormal_mask
    co = pd.to_numeric(df[TARGET_COL], errors="coerce")
    low_threshold = float(max(EMISSION_STANDARD, co.loc[valid].quantile(LOW_EMISSION_QUANTILE))) if valid.any() else float(max(EMISSION_STANDARD, co.quantile(LOW_EMISSION_QUANTILE)))
    low_candidate = valid & (co <= low_threshold)
    stable_threshold = float(df.loc[low_candidate, "低排放稳定性评分"].quantile(0.60)) if low_candidate.any() else float(df["低排放稳定性评分"].quantile(0.60))
    stable_candidate = low_candidate & (df["低排放稳定性评分"] <= stable_threshold)
    min_samples = max(1, math.ceil(LOW_EMISSION_MIN_DURATION_S / SAMPLE_INTERVAL_S))

    df["低排放CO阈值"] = low_threshold
    df["低排放候选"] = low_candidate
    df["低排放稳定候选"] = stable_candidate
    df[LOW_EMISSION_SEGMENT_COL] = np.nan

    run_marker = stable_candidate.ne(stable_candidate.shift(fill_value=False)).cumsum()
    interval_rows: list[dict[str, object]] = []
    segment_id = 0
    for _, group in df.loc[stable_candidate].groupby(run_marker[stable_candidate]):
        if len(group) < min_samples:
            continue
        segment_id += 1
        df.loc[group.index, LOW_EMISSION_SEGMENT_COL] = segment_id
        interval_rows.append(
            {
                "片段编号": segment_id,
                "起始序列": int(group["序列"].iloc[0]),
                "结束序列": int(group["序列"].iloc[-1]),
                "样本数": int(len(group)),
                "持续时间_s": int(len(group) * SAMPLE_INTERVAL_S),
                "起始时间_s": float(group["时间_s"].iloc[0]),
                "结束时间_s": float(group["时间_s"].iloc[-1]),
                "CO均值": float(group[TARGET_COL].mean()),
                "CO中位数": float(group[TARGET_COL].median()),
                "CO最小值": float(group[TARGET_COL].min()),
                "CO最大值": float(group[TARGET_COL].max()),
                "可信达标样本数": int(trusted_standard_mask(group).sum()),
                "低排放稳定性评分中位数": float(group["低排放稳定性评分"].median()),
                "低排放CO阈值": low_threshold,
                "稳定性评分阈值": stable_threshold,
            }
        )

    intervals = pd.DataFrame(interval_rows)
    low_df = df[df[LOW_EMISSION_SEGMENT_COL].notna()].copy()

    pressure_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    if not low_df.empty:
        segment_groups: list[tuple[object, pd.DataFrame]] = [("全部低排放片段", low_df)]
        segment_groups.extend([(int(k), g) for k, g in low_df.groupby(LOW_EMISSION_SEGMENT_COL)])
        for label, part in segment_groups:
            for i in BOX_IDS:
                col = f"{i}#风箱负压强度"
                if col in part.columns:
                    row = condition_quantile_row(part, label, f"{i}#风箱负压强度", col)
                    row["典型原始负压中位数"] = -row["中位数"] if pd.notna(row["中位数"]) else np.nan
                    pressure_rows.append(row)
            key_specs = [
                ("烧结机机速区间", SPEED_COL),
                ("1#大烟道负压强度区间", "1#大烟道负压强度"),
                ("2#大烟道负压强度区间", "2#大烟道负压强度"),
                ("1#大烟道原始负压区间", "1#大烟道负压"),
                ("2#大烟道原始负压区间", "2#大烟道负压"),
                ("1#大烟道温度区间", "1#大烟道温度"),
                ("2#大烟道温度区间", "2#大烟道温度"),
                ("后段风箱温度区间", "后段风箱温度均值"),
            ]
            for name, col in key_specs:
                if col in part.columns:
                    key_rows.append(condition_quantile_row(part, label, name, col))

    pressure_profile = pd.DataFrame(pressure_rows)
    key_ranges = pd.DataFrame(key_rows)
    summary = pd.DataFrame(
        [
            {
                "全样本数": int(len(df)),
                "低排放CO阈值": low_threshold,
                "稳定性评分阈值": stable_threshold,
                "低排放候选样本数": int(low_candidate.sum()),
                "稳定低排放候选样本数": int(stable_candidate.sum()),
                "低排放片段数": int(len(intervals)),
                "低排放片段样本数": int(len(low_df)),
                "最长低排放片段持续时间_s": int(intervals["持续时间_s"].max()) if not intervals.empty else 0,
                "识别口径": f"修正CO<=max(2800,P{int(LOW_EMISSION_QUANTILE*100)})，剔除异常点，且大烟道/后段温度状态稳定，连续不少于{LOW_EMISSION_MIN_DURATION_S}s。",
            }
        ]
    )

    save_csv(df, DATA_MID / "清洗后数据_含低排放片段.csv")
    save_csv(summary, DATA_OUT / "低排放工况识别摘要.csv")
    save_csv(intervals, DATA_OUT / "低排放工况区间识别.csv")
    save_csv(pressure_profile, DATA_OUT / "低排放片段风箱负压典型分布.csv")
    save_csv(key_ranges, DATA_OUT / "低排放片段关键工况区间.csv")
    return df, {"summary": summary, "intervals": intervals, "pressure_profile": pressure_profile, "key_ranges": key_ranges}


def make_adjustment_constraints(ranges: pd.DataFrame, cleaned: pd.DataFrame) -> dict[str, object]:
    widths = ranges["优化强度范围宽度"].to_numpy(dtype=float)
    single_delta = np.clip(widths * SINGLE_ADJUST_FRACTION, SINGLE_ADJUST_MIN, SINGLE_ADJUST_MAX)
    adjacent_rows = []
    adjacent_limits = np.zeros(len(BOX_IDS) - 1, dtype=float)
    for i in range(2, 19):
        diff = (cleaned[f"{i}#风箱负压强度"] - cleaned[f"{i-1}#风箱负压强度"]).abs().astype(float)
        limit = float(diff.quantile(0.90))
        adjacent_limits[i - 2] = limit
        adjacent_rows.append(
            {
                "相邻约束编号": i,
                "左风箱": i - 1,
                "右风箱": i,
                "历史相邻强度差P90_d_i": limit,
                "历史相邻强度差P95": float(diff.quantile(0.95)),
                "历史相邻强度差最大值": float(diff.max()),
            }
        )
    total_budget = float(single_delta.sum() * TOTAL_BUDGET_FRACTION)
    table = pd.DataFrame(
        {
            "风箱编号": BOX_IDS,
            "优化强度下限_P10": ranges["优化强度下限_P10"].to_numpy(dtype=float),
            "优化强度上限_P90": ranges["优化强度上限_P90"].to_numpy(dtype=float),
            "单箱调节上限_Delta_i": single_delta,
            "Delta_i口径": f"clip({SINGLE_ADJUST_FRACTION:.2f}*(P90-P10), {SINGLE_ADJUST_MIN}, {SINGLE_ADJUST_MAX})",
            "总调节预算B": total_budget,
            "B口径": f"{TOTAL_BUDGET_FRACTION:.2f}*sum(Delta_i)",
        }
    )
    save_csv(table, DATA_OUT / "调控约束参数.csv")
    save_csv(pd.DataFrame(adjacent_rows), DATA_OUT / "相邻风箱约束参数.csv")
    return {"single_delta": single_delta, "adjacent_limits": adjacent_limits, "total_budget": total_budget}


def load_dependencies() -> dict[str, Path]:
    deps = {
        "第1问报告": Q1_RESULT_DIR / "problem1_summary.md",
        "第1问模型文件": Q1_DATA_DIR / "最佳模型文件.pkl",
        "第1问清洗数据": Q1_DATA_DIR / "problem1_clean.csv",
        "第1问特征矩阵": Q1_DATA_DIR / "problem1_features_pressure_only.csv",
        "第1问特征索引": Q1_DATA_DIR / "problem1_feature_index.csv",
        "第2问监测型模型": Q2_MODEL_DIR / "CO预测模型_监测型模型.json",
        "第2问模型接口说明": Q2_MODEL_DIR / "CO预测模型接口说明.md",
        "第2问输入输出字段清单": Q2_MODEL_DIR / "输入输出字段清单.md",
        "第2问数据配准说明": Q2_MODEL_DIR / "数据配准说明.md",
        "第2问滞后配准建模数据": Q2_DATA_MID / "滞后配准建模数据.csv",
        "第2问最佳滞后配准": Q2_DATA_MID / "最佳滞后配准.csv",
    }
    missing = [name for name, path in deps.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"第3问依赖文件缺失：{missing}")
    dep_table = pd.DataFrame([{"依赖文件": k, "路径": str(v)} for k, v in deps.items()])
    save_csv(dep_table, DATA_MID / "依赖文件清单.csv")
    save_csv(dep_table, DATA_MID / "第2问依赖文件清单.csv")
    shutil.copy2(deps["第2问最佳滞后配准"], DATA_MID / "第2问最佳滞后配准_副本.csv")
    return deps


def augment_features(z: np.ndarray, degree: int) -> np.ndarray:
    if degree == 1:
        return z
    return np.concatenate([z, z * z], axis=1)


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err * err))),
        "MAPE_%": float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-9)) * 100),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "平均误差": float(np.mean(err)),
    }


@dataclass
class LinearCOModel:
    name: str
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    intercept: float
    linear_coef: np.ndarray
    square_coef: np.ndarray
    degree: int
    alpha: float
    validation_rmse: float
    extra: dict

    @classmethod
    def from_json(cls, path: Path) -> "LinearCOModel":
        raw = load_json(path)
        feature_names = list(raw["特征字段"])
        linear_coef = np.array(raw["线性系数"], dtype=float)
        square_coef = np.array(raw.get("平方项系数", np.zeros_like(linear_coef)), dtype=float)
        if len(square_coef) != len(linear_coef):
            square_coef = np.zeros_like(linear_coef)
        return cls(
            name=str(raw.get("模型名称", path.stem)),
            feature_names=feature_names,
            mean=np.array(raw["标准化均值"], dtype=float),
            scale=np.array(raw["标准化标准差"], dtype=float),
            intercept=float(raw["截距"]),
            linear_coef=linear_coef,
            square_coef=square_coef,
            degree=int(raw.get("实际采用特征次数", 1)),
            alpha=float(raw.get("岭回归alpha", 0.0)),
            validation_rmse=float(raw.get("验证集RMSE", np.nan)),
            extra=raw,
        )

    def ridge_predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.feature_names].to_numpy(dtype=float)
        scale = self.scale.copy()
        scale[scale == 0] = 1.0
        z = (x - self.mean) / scale
        pred = np.full(len(frame), self.intercept, dtype=float)
        pred += z @ self.linear_coef
        if self.degree == 2:
            pred += (z * z) @ self.square_coef
        return pred

    def predict_monitor(self, frame: pd.DataFrame) -> np.ndarray:
        pred = self.ridge_predict(frame)
        blend = self.extra.get("融合说明")
        if blend:
            ridge_weight = float(blend["岭回归预测权重"])
            lag_weight = float(blend["CO_lag1基线权重"])
            pred = ridge_weight * pred + lag_weight * frame["CO历史浓度__lag1"].to_numpy(dtype=float)
        return pred

    def to_json(self, metrics_table: pd.DataFrame | None = None) -> dict:
        payload = {
            "模型名称": self.name,
            "模型类型": "重新训练的无历史CO标准化线性/二次特征岭回归",
            "目标字段": TARGET_COL,
            "特征字段": self.feature_names,
            "标准化均值": self.mean.tolist(),
            "标准化标准差": self.scale.tolist(),
            "岭回归alpha": self.alpha,
            "实际采用特征次数": self.degree,
            "截距": self.intercept,
            "线性系数": self.linear_coef.tolist(),
            "平方项系数": self.square_coef.tolist(),
            "验证集RMSE": self.validation_rmse,
            "训练说明": "使用第二问异常修正后建模的滞后配准数据重新训练，特征集合剔除全部历史CO项，避免历史CO抢占过程变量解释能力。",
        }
        if metrics_table is not None:
            payload["模型评价指标"] = metrics_table.to_dict(orient="records")
        return payload


@dataclass
class NonlinearCOModel:
    name: str
    feature_names: list[str]
    regressor: object
    classifier: object
    extra: dict

    def ridge_predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.feature_names].to_numpy(dtype=float)
        return np.asarray(self.regressor.predict(x), dtype=float)

    def predict_standard_probability(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.feature_names].to_numpy(dtype=float)
        if self.classifier is None:
            return np.zeros(len(frame), dtype=float)
        return np.asarray(self.classifier.predict_proba(x)[:, 1], dtype=float)

    @property
    def degree(self) -> str:
        return str(self.extra.get("特征形式", "非线性树模型"))

    @property
    def alpha(self) -> str:
        return str(self.extra.get("正则化", "内置"))

    @property
    def validation_rmse(self) -> float:
        return float(self.extra.get("验证集RMSE", np.nan))


@dataclass
class ConstantProbabilityClassifier:
    probability: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        p = float(np.clip(self.probability, 0.0, 1.0))
        return np.column_stack([np.full(n, 1.0 - p), np.full(n, p)])


@dataclass
class LayeredCOModel:
    name: str
    feature_names: list[str]
    regime_feature_names: list[str]
    regime_scaler: StandardScaler
    regime_clusterer: object | None
    global_regressor: object
    global_classifier: object
    regime_regressors: dict[int, object]
    regime_classifiers: dict[int, object]
    regime_profiles: pd.DataFrame
    extra: dict
    global_near_classifier: object | None = None
    regime_near_classifiers: dict[int, object] | None = None
    global_q05_regressor: object | None = None
    tail_q10_regressor: object | None = None
    tail_q05_regressor: object | None = None

    def assign_regime(self, frame: pd.DataFrame) -> np.ndarray:
        x_regime = frame[self.regime_feature_names].to_numpy(dtype=float)
        z = self.regime_scaler.transform(x_regime)
        if "达标模板中心_标准化空间" in self.extra and "达标模板距离阈值" in self.extra:
            center = np.asarray(self.extra["达标模板中心_标准化空间"], dtype=float)
            thresholds = np.asarray(self.extra["达标模板距离阈值"], dtype=float)
            dist = np.sqrt(np.mean((z - center) ** 2, axis=1))
            labels = np.full(len(frame), 2, dtype=int)
            labels[dist <= thresholds[1]] = 1
            labels[dist <= thresholds[0]] = 0
            return labels
        if self.regime_clusterer is None:
            return np.zeros(len(frame), dtype=int)
        return np.asarray(self.regime_clusterer.predict(z), dtype=int)

    def _predict_regressor(self, frame: pd.DataFrame, regressor: object, regime_regressors: dict[int, object] | None = None) -> np.ndarray:
        x = frame[self.feature_names].to_numpy(dtype=float)
        pred = np.asarray(regressor.predict(x), dtype=float)
        if regime_regressors:
            regimes = self.assign_regime(frame)
            for regime_id, local_regressor in regime_regressors.items():
                mask = regimes == int(regime_id)
                if mask.any():
                    pred[mask] = np.asarray(local_regressor.predict(x[mask]), dtype=float)
        return pred

    def predict_low_quantile(self, frame: pd.DataFrame, alpha: float = Q10_ALPHA) -> np.ndarray:
        if alpha <= Q05_ALPHA + 1e-8:
            global_regressor = self.global_q05_regressor if self.global_q05_regressor is not None else self.global_regressor
            tail_regressor = self.tail_q05_regressor if self.tail_q05_regressor is not None else global_regressor
        else:
            global_regressor = self.global_regressor
            tail_regressor = self.tail_q10_regressor if self.tail_q10_regressor is not None else global_regressor
        global_pred = self._predict_regressor(frame, global_regressor, self.regime_regressors if alpha > Q05_ALPHA + 1e-8 else None)
        tail_pred = self._predict_regressor(frame, tail_regressor, None)
        blend = float(self.extra.get("尾部融合权重", 0.5))
        return np.clip((1.0 - blend) * global_pred + blend * tail_pred, 0.0, None)

    def ridge_predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.extra.get("回归目标") in {"Q10/Q05达标潜力", "低分位CO"}:
            return self.predict_low_quantile(frame, Q10_ALPHA)
        pred = self._predict_regressor(frame, self.global_regressor, self.regime_regressors)
        if self.extra.get("回归目标") == "低排放潜力":
            pred = np.clip(pred, 0.0, float(self.extra.get("低排放潜力上限", LOW_POTENTIAL_MAX)))
            pred = float(self.extra.get("低排放潜力基准CO", LOW_POTENTIAL_BASELINE_CO)) - pred
        return pred

    def _predict_probability(self, frame: pd.DataFrame, global_classifier: object, regime_classifiers: dict[int, object] | None) -> np.ndarray:
        x = frame[self.feature_names].to_numpy(dtype=float)
        regimes = self.assign_regime(frame)
        prob = np.asarray(global_classifier.predict_proba(x)[:, 1], dtype=float)
        for regime_id, classifier in (regime_classifiers or {}).items():
            mask = regimes == int(regime_id)
            if mask.any():
                prob[mask] = np.asarray(classifier.predict_proba(x[mask])[:, 1], dtype=float)
        return prob

    def predict_near_probability(self, frame: pd.DataFrame) -> np.ndarray:
        classifier = self.global_near_classifier if self.global_near_classifier is not None else self.global_classifier
        return self._predict_probability(frame, classifier, self.regime_near_classifiers)

    def predict_strict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        return self._predict_probability(frame, self.global_classifier, self.regime_classifiers)

    def predict_standard_probability(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_strict_probability(frame)

    @property
    def degree(self) -> str:
        return str(self.extra.get("特征形式", "工况分层非线性树模型"))

    @property
    def alpha(self) -> str:
        return str(self.extra.get("正则化", "内置"))

    @property
    def validation_rmse(self) -> float:
        return float(self.extra.get("验证集RMSE", np.nan))


def fit_no_co_control_model(aligned: pd.DataFrame, monitor_model: LinearCOModel) -> tuple[LinearCOModel, pd.DataFrame, pd.DataFrame]:
    feature_cols = [name for name in monitor_model.feature_names if not name.startswith(CO_HISTORY_PREFIX)]
    missing = [name for name in feature_cols if name not in aligned.columns]
    if missing:
        raise KeyError(f"第二问配准数据缺少无CO调控模型特征：{missing[:10]}")

    train_mask = aligned["数据集"] == "训练集"
    val_mask = aligned["数据集"] == "验证集"
    x = aligned[feature_cols].to_numpy(dtype=float)
    y = aligned[TARGET_COL].to_numpy(dtype=float)
    x_train = x[train_mask]
    y_train = y[train_mask]
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    z = (x - mu) / sigma

    alphas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    candidates = []
    for degree in [1, 2]:
        z_aug = augment_features(z, degree)
        design = np.concatenate([np.ones((len(z_aug), 1)), z_aug], axis=1)
        xtr = design[train_mask]
        ytr = y[train_mask]
        xval = design[val_mask]
        yval = y[val_mask]
        best_for_degree = None
        for alpha in alphas:
            penalty = np.eye(xtr.shape[1])
            penalty[0, 0] = 0.0
            lhs = xtr.T @ xtr + alpha * penalty
            rhs = xtr.T @ ytr
            try:
                coef = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                coef = np.linalg.pinv(lhs) @ rhs
            pred_val = xval @ coef
            rmse = float(np.sqrt(np.mean((pred_val - yval) ** 2)))
            if best_for_degree is None or rmse < best_for_degree["rmse"]:
                best_for_degree = {"degree": degree, "alpha": alpha, "rmse": rmse, "coef": coef, "design": design}
        candidates.append(best_for_degree)

    linear_candidate = [item for item in candidates if item["degree"] == 1][0]
    best_candidate = min(candidates, key=lambda item: item["rmse"])
    if best_candidate["degree"] == 2 and best_candidate["rmse"] > 0.97 * linear_candidate["rmse"]:
        best_candidate = linear_candidate

    coef = best_candidate["coef"]
    degree = int(best_candidate["degree"])
    linear_coef = coef[1 : 1 + len(feature_cols)]
    square_coef = coef[1 + len(feature_cols) :] if degree == 2 else np.zeros(len(feature_cols), dtype=float)
    pred_all = best_candidate["design"] @ coef

    metrics_rows = []
    for split_name in ["训练集", "验证集", "测试集"]:
        mask = aligned["数据集"] == split_name
        if not mask.any():
            continue
        row = {
            "模型": "无历史CO调控模型",
            "数据集": split_name,
            "样本数": int(mask.sum()),
            **regression_metrics(y[mask], pred_all[mask]),
        }
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)
    save_csv(metrics, DATA_OUT / "无历史CO调控模型评价指标.csv")

    importance_rows = []
    for feature, c1, c2 in zip(feature_cols, linear_coef, square_coef):
        importance_rows.append(
            {
                "模型": "无历史CO调控模型",
                "特征名": feature,
                "标准化线性系数": float(c1),
                "标准化平方项系数": float(c2),
                "综合影响强度": float(math.sqrt(c1 * c1 + c2 * c2)),
                "线性影响方向": "正向" if c1 >= 0 else "负向",
            }
        )
    importance = pd.DataFrame(importance_rows).sort_values("综合影响强度", ascending=False)
    save_csv(importance, DATA_OUT / "变量影响强度_无历史CO调控模型.csv")

    model = LinearCOModel(
        name="无历史CO调控模型",
        feature_names=feature_cols,
        mean=mu,
        scale=sigma,
        intercept=float(coef[0]),
        linear_coef=linear_coef,
        square_coef=square_coef,
        degree=degree,
        alpha=float(best_candidate["alpha"]),
        validation_rmse=float(best_candidate["rmse"]),
        extra={},
    )
    payload = model.to_json(metrics)
    save_json(payload, DATA_MID / "CO调控模型_无历史CO.json")
    save_json(payload, RESULT_DIR / "CO调控模型_无历史CO.json")
    return model, metrics, importance


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    y_bin = np.asarray(y_true, dtype=bool)
    out = {
        "达标样本数": int(y_bin.sum()),
        "预测达标均值概率": float(np.mean(prob)),
        "真实达标样本平均概率": float(np.mean(prob[y_bin])) if y_bin.any() else np.nan,
        "真实未达标样本平均概率": float(np.mean(prob[~y_bin])) if (~y_bin).any() else np.nan,
    }
    if len(np.unique(y_bin)) == 2:
        out["达标分类AUC"] = float(roc_auc_score(y_bin, prob))
        out["达标分类AP"] = float(average_precision_score(y_bin, prob))
    else:
        out["达标分类AUC"] = np.nan
        out["达标分类AP"] = np.nan
    return out


def fit_nonlinear_control_model(aligned: pd.DataFrame, monitor_model: LinearCOModel) -> tuple[NonlinearCOModel, pd.DataFrame, pd.DataFrame]:
    feature_cols = [name for name in monitor_model.feature_names if not name.startswith(CO_HISTORY_PREFIX)]
    missing = [name for name in feature_cols if name not in aligned.columns]
    if missing:
        raise KeyError(f"第二问配准数据缺少无CO调控模型特征：{missing[:10]}")

    x = aligned[feature_cols].to_numpy(dtype=float)
    y = aligned[TARGET_COL].to_numpy(dtype=float)
    y_standard = y <= EMISSION_STANDARD
    train_mask = aligned["数据集"] == "训练集"

    sample_weight = np.ones(len(aligned), dtype=float)
    sample_weight[y_standard] = 6.0
    sample_weight[(y > EMISSION_STANDARD) & (y <= EMISSION_STANDARD + 400)] = 2.5

    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=260,
        learning_rate=0.045,
        max_leaf_nodes=15,
        min_samples_leaf=18,
        l2_regularization=0.08,
        random_state=42,
    )
    regressor.fit(x[train_mask], y[train_mask], sample_weight=sample_weight[train_mask])

    classifier = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=16,
        l2_regularization=0.08,
        random_state=42,
    )
    classifier.fit(x[train_mask], y_standard[train_mask], sample_weight=sample_weight[train_mask])

    rows = []
    for split_name in ["训练集", "验证集", "测试集"]:
        mask = aligned["数据集"] == split_name
        if not mask.any():
            continue
        pred = np.asarray(regressor.predict(x[mask]), dtype=float)
        prob = np.asarray(classifier.predict_proba(x[mask])[:, 1], dtype=float)
        low_mask = y_standard[mask]
        row = {
            "模型": "非线性无历史CO低排放调控模型",
            "数据集": split_name,
            "样本数": int(mask.sum()),
            **regression_metrics(y[mask], pred),
            "MAE": float(mean_absolute_error(y[mask], pred)),
            "达标样本MAE": float(mean_absolute_error(y[mask][low_mask], pred[low_mask])) if low_mask.any() else np.nan,
            "预测CO低于2800比例_%": float(np.mean(pred <= EMISSION_STANDARD) * 100),
            "真实CO低于2800比例_%": float(np.mean(low_mask) * 100),
            **classification_metrics(np.asarray(low_mask, dtype=bool), prob),
        }
        rows.append(row)
    metrics = pd.DataFrame(rows)
    save_csv(metrics, DATA_OUT / "非线性低排放调控模型评价指标.csv")

    importance_rows = []
    baseline_mask = aligned["数据集"] == "验证集"
    if not baseline_mask.any():
        baseline_mask = train_mask
    x_eval = x[baseline_mask]
    y_eval = y[baseline_mask]
    baseline_rmse = float(np.sqrt(np.mean((regressor.predict(x_eval) - y_eval) ** 2)))
    rng = np.random.default_rng(42)
    max_features = min(41, len(feature_cols))
    for idx, feature in enumerate(feature_cols[:max_features]):
        x_perm = x_eval.copy()
        rng.shuffle(x_perm[:, idx])
        rmse = float(np.sqrt(np.mean((regressor.predict(x_perm) - y_eval) ** 2)))
        importance_rows.append(
            {
                "模型": "非线性无历史CO低排放调控模型",
                "特征名": feature,
                "置换RMSE增量": rmse - baseline_rmse,
            }
        )
    importance = pd.DataFrame(importance_rows).sort_values("置换RMSE增量", ascending=False)
    save_csv(importance, DATA_OUT / "变量影响强度_非线性低排放调控模型.csv")

    extra = {
        "模型类型": "HistGradientBoostingRegressor + HistGradientBoostingClassifier",
        "特征形式": "非线性树模型",
        "达标样本权重": 6.0,
        "近达标样本权重": 2.5,
        "验证集RMSE": float(metrics.loc[metrics["数据集"] == "验证集", "RMSE"].iloc[0]) if (metrics["数据集"] == "验证集").any() else np.nan,
        "说明": "历史CO<=2800样本通过样本权重进入回归/分类训练；分类器输出达标倾向概率，主模型不单独依赖回归值判定达标。",
    }
    model = NonlinearCOModel(
        name="非线性无历史CO低排放调控模型",
        feature_names=feature_cols,
        regressor=regressor,
        classifier=classifier,
        extra=extra,
    )
    save_json(
        {
            "模型名称": model.name,
            "模型类型": extra["模型类型"],
            "目标字段": TARGET_COL,
            "特征字段": feature_cols,
            "训练权重说明": extra,
            "模型评价指标": metrics.to_dict(orient="records"),
        },
        DATA_MID / "CO调控模型_非线性低排放.json",
    )
    save_json(
        {
            "模型名称": model.name,
            "模型类型": extra["模型类型"],
            "目标字段": TARGET_COL,
            "特征字段": feature_cols,
            "训练权重说明": extra,
            "模型评价指标": metrics.to_dict(orient="records"),
        },
        RESULT_DIR / "CO调控模型_非线性低排放.json",
    )
    joblib.dump({"regressor": regressor, "classifier": classifier, "feature_names": feature_cols, "extra": extra}, DATA_MID / "CO调控模型_非线性低排放.joblib")
    return model, metrics, importance


def select_regime_feature_names(feature_cols: list[str]) -> list[str]:
    selected: list[str] = []
    for name in feature_cols:
        if name.startswith(SPEED_COL):
            selected.append(name)
        elif "大烟道负压强度" in name or "大烟道温度" in name:
            selected.append(name)
        else:
            m = re.match(r"^(\d+)#风箱平均温度__lag\d+$", name)
            if m and int(m.group(1)) >= 13:
                selected.append(name)
    if len(selected) < 4:
        selected = feature_cols[: min(len(feature_cols), 12)]
    return selected


def make_hgb_regressor(random_state: int, min_samples_leaf: int = 10) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=240,
        learning_rate=0.045,
        max_leaf_nodes=15,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=0.08,
        random_state=random_state,
    )


def make_quantile_regressor(alpha: float, random_state: int, min_samples_leaf: int = 6) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=260,
        learning_rate=0.035,
        max_depth=3,
        min_samples_leaf=min_samples_leaf,
        subsample=0.85,
        random_state=random_state,
    )


def make_hgb_classifier(y_bin: np.ndarray, probability: float, random_state: int) -> object:
    if len(np.unique(y_bin.astype(bool))) < 2:
        return ConstantProbabilityClassifier(probability=probability)
    return HistGradientBoostingClassifier(
        max_iter=210,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=12,
        l2_regularization=0.08,
        random_state=random_state,
    )


def make_regime_profiles(aligned: pd.DataFrame, regime_labels: np.ndarray, regime_feature_names: list[str]) -> pd.DataFrame:
    df = aligned.copy()
    df["工况层"] = regime_labels.astype(int)
    df["_可信达标"] = trusted_standard_mask(df).to_numpy(dtype=bool)
    back_temp_features = [c for c in regime_feature_names if re.match(r"^1[3-8]#风箱平均温度__lag\d+$", c)]
    speed_feature = next((c for c in regime_feature_names if c.startswith(SPEED_COL)), "")
    specs = [
        ("烧结机机速中位数", speed_feature),
        ("1#大烟道负压强度中位数", next((c for c in regime_feature_names if "1#大烟道负压强度" in c), "")),
        ("2#大烟道负压强度中位数", next((c for c in regime_feature_names if "2#大烟道负压强度" in c), "")),
        ("1#大烟道温度中位数", next((c for c in regime_feature_names if "1#大烟道温度" in c), "")),
        ("2#大烟道温度中位数", next((c for c in regime_feature_names if "2#大烟道温度" in c), "")),
    ]
    rows = []
    co_medians: dict[int, float] = {}
    for regime_id, group in df.groupby("工况层"):
        co_medians[int(regime_id)] = float(group[TARGET_COL].median())
    ordered = sorted(co_medians, key=lambda rid: co_medians[rid])
    name_map = {}
    for rank, regime_id in enumerate(ordered):
        if rank == 0:
            name_map[regime_id] = "低CO工况层"
        elif rank == len(ordered) - 1:
            name_map[regime_id] = "高CO工况层"
        else:
            name_map[regime_id] = "中间CO工况层"

    for regime_id, group in df.groupby("工况层"):
        row: dict[str, object] = {
            "工况层": int(regime_id),
            "工况层名称": name_map.get(int(regime_id), f"工况层{regime_id}"),
            "样本数": int(len(group)),
            "训练集样本数": int((group["数据集"] == "训练集").sum()) if "数据集" in group.columns else int(len(group)),
            "可信达标样本数": int(group["_可信达标"].sum()),
            "可信达标比例_%": float(group["_可信达标"].mean() * 100),
            "CO均值": float(group[TARGET_COL].mean()),
            "CO中位数": float(group[TARGET_COL].median()),
            "CO_P10": float(group[TARGET_COL].quantile(0.10)),
            "CO_P90": float(group[TARGET_COL].quantile(0.90)),
        }
        for out_name, col in specs:
            row[out_name] = float(group[col].median()) if col and col in group.columns else np.nan
        if back_temp_features:
            row["后段风箱温度中位数"] = float(group[back_temp_features].mean(axis=1).median())
        rows.append(row)
    profiles = pd.DataFrame(rows).sort_values("CO中位数").reset_index(drop=True)
    save_csv(profiles, DATA_OUT / "工况分层画像.csv")
    return profiles


def regime_name_map_for_template_layers() -> dict[int, str]:
    return {
        0: "可信达标模板邻域",
        1: "达标模板过渡层",
        2: "远离达标模板层",
    }


def fit_layered_control_model(aligned: pd.DataFrame, monitor_model: LinearCOModel) -> tuple[LayeredCOModel, pd.DataFrame, pd.DataFrame]:
    feature_cols = [name for name in monitor_model.feature_names if not name.startswith(CO_HISTORY_PREFIX)]
    missing = [name for name in feature_cols if name not in aligned.columns]
    if missing:
        raise KeyError(f"第二问配准数据缺少分层调控模型特征：{missing[:10]}")

    x = aligned[feature_cols].to_numpy(dtype=float)
    y = aligned[TARGET_COL].to_numpy(dtype=float)
    y_standard = trusted_standard_mask(aligned).to_numpy(dtype=bool)
    train_mask = (aligned["数据集"] == "训练集").to_numpy(dtype=bool)

    sample_weight = np.ones(len(aligned), dtype=float)
    sample_weight[y_standard] = 30.0
    near_standard = (~y_standard) & (y <= EMISSION_STANDARD + 400)
    sample_weight[near_standard] = 5.0

    regime_features = select_regime_feature_names(feature_cols)
    scaler = StandardScaler()
    scaler.fit(aligned.loc[train_mask, regime_features].to_numpy(dtype=float))
    z_all = scaler.transform(aligned[regime_features].to_numpy(dtype=float))
    if y_standard.any():
        trusted_center = z_all[y_standard].mean(axis=0)
        trusted_dist = np.sqrt(np.mean((z_all[y_standard] - trusted_center) ** 2, axis=1))
    else:
        trusted_center = z_all[train_mask].mean(axis=0)
        trusted_dist = np.sqrt(np.mean((z_all[train_mask] - trusted_center) ** 2, axis=1))
    dist_all = np.sqrt(np.mean((z_all - trusted_center) ** 2, axis=1))
    train_dist = dist_all[train_mask]
    threshold_near = float(np.quantile(train_dist, 0.20))
    threshold_mid = float(max(threshold_near + 1e-6, np.quantile(train_dist, 0.60)))
    regime_labels = np.full(len(aligned), 2, dtype=int)
    regime_labels[dist_all <= threshold_mid] = 1
    regime_labels[dist_all <= threshold_near] = 0
    regime_labels[y_standard] = 0
    n_clusters = 3

    global_regressor = make_hgb_regressor(42, min_samples_leaf=8)
    global_regressor.fit(x[train_mask], y[train_mask], sample_weight=sample_weight[train_mask])
    global_classifier = make_hgb_classifier(y_standard[train_mask], float(y_standard[train_mask].mean()), 43)
    if isinstance(global_classifier, ConstantProbabilityClassifier):
        pass
    else:
        global_classifier.fit(x[train_mask], y_standard[train_mask], sample_weight=sample_weight[train_mask])

    regime_regressors: dict[int, object] = {}
    regime_classifiers: dict[int, object] = {}
    min_regime_train_rows = 35
    for regime_id in range(n_clusters):
        mask = train_mask & (regime_labels == regime_id)
        if int(mask.sum()) < min_regime_train_rows:
            continue
        regressor = make_hgb_regressor(100 + regime_id, min_samples_leaf=6 if regime_id == 0 else 8)
        regressor.fit(x[mask], y[mask], sample_weight=sample_weight[mask])
        classifier = make_hgb_classifier(y_standard[mask], float(y_standard[mask].mean()), 200 + regime_id)
        if not isinstance(classifier, ConstantProbabilityClassifier):
            classifier.fit(x[mask], y_standard[mask], sample_weight=sample_weight[mask])
        regime_regressors[int(regime_id)] = regressor
        regime_classifiers[int(regime_id)] = classifier

    regime_profiles = make_regime_profiles(aligned, regime_labels, regime_features)
    layer_names = regime_name_map_for_template_layers()
    if not regime_profiles.empty:
        regime_profiles["工况层名称"] = regime_profiles["工况层"].map(layer_names).fillna(regime_profiles["工况层名称"])
        regime_profiles = regime_profiles.sort_values("工况层").reset_index(drop=True)
        save_csv(regime_profiles, DATA_OUT / "工况分层画像.csv")
    extra = {
        "模型类型": "可信达标模板邻域分层 + 各层HistGradientBoostingRegressor/Classifier",
        "特征形式": "达标模板邻域分层非线性树模型",
        "工况分层特征": regime_features,
        "工况层数量": int(n_clusters),
        "达标模板中心_标准化空间": trusted_center.tolist(),
        "达标模板距离阈值": [threshold_near, threshold_mid],
        "可信达标样本权重": 30.0,
        "近达标样本权重": 5.0,
        "说明": "不再使用KMeans命名达标层；先以42条可信达标样本形成达标模板中心，再按工况特征到达标模板的距离划分邻域/过渡/远离三层，每层单独训练调控响应模型。",
    }
    model = LayeredCOModel(
        name="分层低排放调控响应模型",
        feature_names=feature_cols,
        regime_feature_names=regime_features,
        regime_scaler=scaler,
        regime_clusterer=None,
        global_regressor=global_regressor,
        global_classifier=global_classifier,
        regime_regressors=regime_regressors,
        regime_classifiers=regime_classifiers,
        regime_profiles=regime_profiles,
        extra=extra,
    )

    pred_all = model.ridge_predict(aligned)
    prob_all = model.predict_standard_probability(aligned)
    rows = []
    for split_name in ["训练集", "验证集", "测试集"]:
        mask = (aligned["数据集"] == split_name).to_numpy(dtype=bool)
        if not mask.any():
            continue
        low_mask = y_standard[mask]
        row = {
            "模型": "分层低排放调控响应模型",
            "数据集": split_name,
            "样本数": int(mask.sum()),
            **regression_metrics(y[mask], pred_all[mask]),
            "MAE": float(mean_absolute_error(y[mask], pred_all[mask])),
            "可信达标样本MAE": float(mean_absolute_error(y[mask][low_mask], pred_all[mask][low_mask])) if low_mask.any() else np.nan,
            "预测CO低于2800比例_%": float(np.mean(pred_all[mask] <= EMISSION_STANDARD) * 100),
            "真实可信达标比例_%": float(np.mean(low_mask) * 100),
            **classification_metrics(np.asarray(low_mask, dtype=bool), prob_all[mask]),
        }
        rows.append(row)
    metrics = pd.DataFrame(rows)
    if (metrics["数据集"] == "验证集").any():
        extra["验证集RMSE"] = float(metrics.loc[metrics["数据集"] == "验证集", "RMSE"].iloc[0])
    else:
        extra["验证集RMSE"] = np.nan
    save_csv(metrics, DATA_OUT / "分层低排放调控模型评价指标.csv")
    save_csv(metrics, DATA_OUT / "非线性低排放调控模型评价指标.csv")

    baseline_mask = (aligned["数据集"] == "验证集").to_numpy(dtype=bool)
    if not baseline_mask.any():
        baseline_mask = train_mask
    eval_frame = aligned.loc[baseline_mask, feature_cols].copy()
    y_eval = y[baseline_mask]
    baseline_rmse = float(np.sqrt(np.mean((model.ridge_predict(eval_frame) - y_eval) ** 2)))
    importance_rows = []
    rng = np.random.default_rng(42)
    max_features = min(41, len(feature_cols))
    for feature in feature_cols[:max_features]:
        perm_frame = eval_frame.copy()
        values = perm_frame[feature].to_numpy(dtype=float).copy()
        rng.shuffle(values)
        perm_frame[feature] = values
        rmse = float(np.sqrt(np.mean((model.ridge_predict(perm_frame) - y_eval) ** 2)))
        importance_rows.append(
            {
                "模型": "分层低排放调控响应模型",
                "特征名": feature,
                "置换RMSE增量": rmse - baseline_rmse,
            }
        )
    importance = pd.DataFrame(importance_rows).sort_values("置换RMSE增量", ascending=False)
    save_csv(importance, DATA_OUT / "变量影响强度_分层低排放调控模型.csv")
    save_csv(importance, DATA_OUT / "变量影响强度_非线性低排放调控模型.csv")

    payload = {
        "模型名称": model.name,
        "模型类型": extra["模型类型"],
        "目标字段": TARGET_COL,
        "特征字段": feature_cols,
        "工况分层特征": regime_features,
        "训练权重说明": extra,
        "工况分层画像": regime_profiles.to_dict(orient="records"),
        "模型评价指标": metrics.to_dict(orient="records"),
    }
    save_json(payload, DATA_MID / "CO调控模型_分层低排放.json")
    save_json(payload, RESULT_DIR / "CO调控模型_分层低排放.json")
    save_json(payload, DATA_MID / "CO调控模型_非线性低排放.json")
    save_json(payload, RESULT_DIR / "CO调控模型_非线性低排放.json")
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_cols,
            "regime_feature_names": regime_features,
            "extra": extra,
        },
        DATA_MID / "CO调控模型_分层低排放.joblib",
    )
    return model, metrics, importance


def fit_future_tail_control_model(future: pd.DataFrame) -> tuple[LayeredCOModel, pd.DataFrame, pd.DataFrame]:
    meta_cols = {
        "序列",
        "时间_s",
        "控制序列",
        "控制时间_s",
        "预测提前量_s",
        TARGET_COL,
        Q2_CORRECTED_CO_COL,
        TRUSTED_STANDARD_COL,
        "未来CO是否低于2800",
        "低排放潜力目标",
        "数据集",
    }
    feature_cols = [c for c in future.columns if c not in meta_cols]
    x = future[feature_cols].to_numpy(dtype=float)
    y_future = future[TARGET_COL].to_numpy(dtype=float)
    y_strict = y_future <= EMISSION_STANDARD
    y_near = y_future <= NEAR_STANDARD
    y_trusted = trusted_standard_mask(future).to_numpy(dtype=bool)
    train_mask = (future["数据集"] == "训练集").to_numpy(dtype=bool)

    sample_weight = np.ones(len(future), dtype=float)
    sample_weight[y_near] = 6.0
    sample_weight[y_strict] = 18.0

    regime_features = select_regime_feature_names(feature_cols)
    scaler = StandardScaler()
    scaler.fit(future.loc[train_mask, regime_features].to_numpy(dtype=float))
    z_all = scaler.transform(future[regime_features].to_numpy(dtype=float))
    trusted_train = train_mask & y_trusted
    if trusted_train.any():
        trusted_center = z_all[trusted_train].mean(axis=0)
    elif (train_mask & y_strict).any():
        trusted_center = z_all[train_mask & y_strict].mean(axis=0)
    else:
        trusted_center = z_all[train_mask].mean(axis=0)
    dist_all = np.sqrt(np.mean((z_all - trusted_center) ** 2, axis=1))
    train_dist = dist_all[train_mask]
    threshold_near = float(np.quantile(train_dist, 0.20))
    threshold_mid = float(max(threshold_near + 1e-6, np.quantile(train_dist, 0.60)))
    regime_labels = np.full(len(future), 2, dtype=int)
    regime_labels[dist_all <= threshold_mid] = 1
    regime_labels[dist_all <= threshold_near] = 0
    regime_labels[y_strict | y_trusted] = 0

    tail_core = y_future <= NEAR_STANDARD
    if (train_mask & tail_core).any():
        tail_dist_threshold = float(np.quantile(dist_all[train_mask & tail_core], 0.90))
    else:
        tail_dist_threshold = threshold_near
    tail_neighbor = (dist_all <= max(tail_dist_threshold, threshold_near)) & (y_future <= TAIL_NEIGHBOR_CO)
    tail_mask = tail_core | tail_neighbor
    tail_train_mask = train_mask & tail_mask
    if int(tail_train_mask.sum()) < 80:
        fallback_threshold = float(np.quantile(y_future[train_mask], 0.45))
        tail_mask = y_future <= max(fallback_threshold, NEAR_STANDARD)
        tail_train_mask = train_mask & tail_mask

    tail_weight = np.ones(len(future), dtype=float)
    tail_weight[y_near] = 8.0
    tail_weight[y_strict] = 30.0
    tail_weight[tail_neighbor & ~tail_core] = 2.0

    global_regressor = make_quantile_regressor(Q10_ALPHA, 52, min_samples_leaf=6)
    global_regressor.fit(x[train_mask], y_future[train_mask], sample_weight=sample_weight[train_mask])
    global_q05_regressor = make_quantile_regressor(Q05_ALPHA, 53, min_samples_leaf=6)
    global_q05_regressor.fit(x[train_mask], y_future[train_mask], sample_weight=sample_weight[train_mask])
    tail_q10_regressor = make_quantile_regressor(Q10_ALPHA, 152, min_samples_leaf=4)
    tail_q10_regressor.fit(x[tail_train_mask], y_future[tail_train_mask], sample_weight=tail_weight[tail_train_mask])
    tail_q05_regressor = make_quantile_regressor(Q05_ALPHA, 153, min_samples_leaf=4)
    tail_q05_regressor.fit(x[tail_train_mask], y_future[tail_train_mask], sample_weight=tail_weight[tail_train_mask])

    classifier_weight = np.ones(len(future), dtype=float)
    classifier_weight[y_near] = 5.0
    classifier_weight[y_strict] = 20.0
    global_near_classifier = make_hgb_classifier(y_near[train_mask], float(y_near[train_mask].mean()), 54)
    if not isinstance(global_near_classifier, ConstantProbabilityClassifier):
        global_near_classifier.fit(x[train_mask], y_near[train_mask], sample_weight=classifier_weight[train_mask])
    global_classifier = make_hgb_classifier(y_strict[train_mask], float(y_strict[train_mask].mean()), 55)
    if not isinstance(global_classifier, ConstantProbabilityClassifier):
        global_classifier.fit(x[train_mask], y_strict[train_mask], sample_weight=classifier_weight[train_mask])

    regime_regressors: dict[int, object] = {}
    regime_classifiers: dict[int, object] = {}
    regime_near_classifiers: dict[int, object] = {}
    for regime_id in range(3):
        mask = train_mask & (regime_labels == regime_id)
        if int(mask.sum()) < 30:
            continue
        classifier_near = make_hgb_classifier(y_near[mask], float(y_near[mask].mean()), 250 + regime_id)
        if not isinstance(classifier_near, ConstantProbabilityClassifier):
            classifier_near.fit(x[mask], y_near[mask], sample_weight=classifier_weight[mask])
            regime_near_classifiers[int(regime_id)] = classifier_near
        classifier = make_hgb_classifier(y_strict[mask], float(y_strict[mask].mean()), 260 + regime_id)
        if not isinstance(classifier, ConstantProbabilityClassifier):
            classifier.fit(x[mask], y_strict[mask], sample_weight=classifier_weight[mask])
            regime_classifiers[int(regime_id)] = classifier

    regime_profiles = make_regime_profiles(future, regime_labels, regime_features)
    layer_names = regime_name_map_for_template_layers()
    if not regime_profiles.empty:
        regime_profiles["工况层名称"] = regime_profiles["工况层"].map(layer_names).fillna(regime_profiles["工况层名称"])
        regime_profiles = regime_profiles.sort_values("工况层").reset_index(drop=True)
        save_csv(regime_profiles, DATA_OUT / "工况分层画像.csv")

    extra = {
        "模型类型": "未来300s Q10/Q05达标潜力模型 + 全局/尾部融合 + 两级达标分类器",
        "特征形式": "当前CO状态+当前过程变量+负压调节量",
        "回归目标": "Q10/Q05达标潜力",
        "主优化低分位": "Q10_CO",
        "理论极限低分位": "Q05_CO",
        "Q10全局尾部融合公式": "CO_low_final=0.5*Q10_global+0.5*Q10_tail",
        "Q05全局尾部融合公式": "CO_low_extreme=0.5*Q05_global+0.5*Q05_tail",
        "尾部融合权重": 0.5,
        "预测提前量_s": CONTROL_EFFECT_HORIZON_S,
        "工况分层特征": regime_features,
        "达标模板中心_标准化空间": trusted_center.tolist(),
        "达标模板距离阈值": [threshold_near, threshold_mid],
        "尾部局部模型训练样本数": int(tail_train_mask.sum()),
        "尾部核心样本_CO<=3200": int((train_mask & tail_core).sum()),
        "尾部邻域样本数": int((train_mask & tail_neighbor & ~tail_core).sum()),
        "全局权重_CO<=2800": 18.0,
        "全局权重_2800到3200": 6.0,
        "尾部权重_CO<=2800": 30.0,
        "尾部权重_2800到3200": 8.0,
        "两级分类器": "Pr_near=Pr(CO<=3200), Pr_strict=Pr(CO<=2800)",
        "说明": "保留历史CO作为状态变量，使用当前过程变量和负压调节量预测未来300s低分位CO；全局Q10/Q05模型用全部样本训练，尾部Q10/Q05模型只用CO<=3200及其达标模板邻域样本训练，并对CO<=2800加高权重。",
    }
    model = LayeredCOModel(
        name="未来300s Q10/Q05达标潜力调控模型",
        feature_names=feature_cols,
        regime_feature_names=regime_features,
        regime_scaler=scaler,
        regime_clusterer=None,
        global_regressor=global_regressor,
        global_classifier=global_classifier,
        regime_regressors=regime_regressors,
        regime_classifiers=regime_classifiers,
        regime_profiles=regime_profiles,
        extra=extra,
        global_near_classifier=global_near_classifier,
        regime_near_classifiers=regime_near_classifiers,
        global_q05_regressor=global_q05_regressor,
        tail_q10_regressor=tail_q10_regressor,
        tail_q05_regressor=tail_q05_regressor,
    )

    pred_all = model.ridge_predict(future)
    pred_q05_all = model.predict_low_quantile(future, Q05_ALPHA)
    prob_near_all = model.predict_near_probability(future)
    prob_strict_all = model.predict_standard_probability(future)
    rows = []
    for split_name in ["训练集", "验证集", "测试集"]:
        mask = (future["数据集"] == split_name).to_numpy(dtype=bool)
        if not mask.any():
            continue
        low_mask = y_strict[mask]
        strict_metrics = classification_metrics(np.asarray(low_mask, dtype=bool), prob_strict_all[mask])
        near_metrics = classification_metrics(np.asarray(y_near[mask], dtype=bool), prob_near_all[mask])
        rows.append(
            {
                "模型": "未来300s Q10/Q05达标潜力调控模型",
                "数据集": split_name,
                "样本数": int(mask.sum()),
                **regression_metrics(y_future[mask], pred_all[mask]),
                "MAE": float(mean_absolute_error(y_future[mask], pred_all[mask])),
                "Q05_MAE": float(mean_absolute_error(y_future[mask], pred_q05_all[mask])),
                "严格达标样本Q10_MAE": float(mean_absolute_error(y_future[mask][low_mask], pred_all[mask][low_mask])) if low_mask.any() else np.nan,
                "Q10预测CO低于2800比例_%": float(np.mean(pred_all[mask] <= EMISSION_STANDARD) * 100),
                "Q05预测CO低于2800比例_%": float(np.mean(pred_q05_all[mask] <= EMISSION_STANDARD) * 100),
                "真实严格达标比例_%": float(np.mean(low_mask) * 100),
                "真实近达标比例_%": float(np.mean(y_near[mask]) * 100),
                **strict_metrics,
                **{f"近达标_{k}": v for k, v in near_metrics.items()},
            }
        )
    metrics = pd.DataFrame(rows)
    extra["验证集RMSE"] = float(metrics.loc[metrics["数据集"] == "验证集", "RMSE"].iloc[0]) if (metrics["数据集"] == "验证集").any() else np.nan
    save_csv(metrics, DATA_OUT / "未来300s低排放潜力模型评价指标.csv")
    save_csv(metrics, DATA_OUT / "分层低排放调控模型评价指标.csv")
    save_csv(metrics, DATA_OUT / "非线性低排放调控模型评价指标.csv")

    baseline_mask = (future["数据集"] == "验证集").to_numpy(dtype=bool)
    if not baseline_mask.any():
        baseline_mask = train_mask
    eval_frame = future.loc[baseline_mask, feature_cols].copy()
    y_eval = y_future[baseline_mask]
    baseline_rmse = float(np.sqrt(np.mean((model.ridge_predict(eval_frame) - y_eval) ** 2)))
    rng = np.random.default_rng(42)
    importance_rows = []
    for feature in feature_cols[: min(60, len(feature_cols))]:
        perm_frame = eval_frame.copy()
        values = perm_frame[feature].to_numpy(dtype=float).copy()
        rng.shuffle(values)
        perm_frame[feature] = values
        rmse = float(np.sqrt(np.mean((model.ridge_predict(perm_frame) - y_eval) ** 2)))
        importance_rows.append({"模型": "未来300s低排放潜力调控模型", "特征名": feature, "置换RMSE增量": rmse - baseline_rmse})
    importance = pd.DataFrame(importance_rows).sort_values("置换RMSE增量", ascending=False)
    if not importance.empty:
        importance["置换RMSE增量"] = pd.to_numeric(importance["置换RMSE增量"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        importance = importance.sort_values("置换RMSE增量", ascending=False)
    save_csv(importance, DATA_OUT / "变量影响强度_未来300s低排放潜力模型.csv")
    save_csv(importance, DATA_OUT / "变量影响强度_非线性低排放调控模型.csv")

    payload = {
        "模型名称": model.name,
        "模型类型": extra["模型类型"],
        "目标字段": "未来300s后CO",
        "回归目标": "Q10/Q05达标潜力",
        "特征字段": feature_cols,
        "训练权重说明": extra,
        "工况分层画像": regime_profiles.to_dict(orient="records"),
        "模型评价指标": metrics.to_dict(orient="records"),
    }
    save_json(payload, DATA_MID / "CO调控模型_未来300s低排放潜力.json")
    save_json(payload, RESULT_DIR / "CO调控模型_未来300s低排放潜力.json")
    save_json(payload, DATA_MID / "CO调控模型_分层低排放.json")
    save_json(payload, RESULT_DIR / "CO调控模型_分层低排放.json")
    save_json(payload, DATA_MID / "CO调控模型_非线性低排放.json")
    save_json(payload, RESULT_DIR / "CO调控模型_非线性低排放.json")
    joblib.dump({"model": model, "feature_names": feature_cols, "extra": extra}, DATA_MID / "CO调控模型_未来300s低排放潜力.joblib")
    return model, metrics, importance


def audit_trusted_standard_predictions(aligned: pd.DataFrame, control_model: object) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    mask = trusted_standard_mask(aligned)
    audit_source = aligned.loc[mask].copy()
    if audit_source.empty:
        detail = pd.DataFrame()
        summary = pd.DataFrame(
            [
                {
                    "可信达标样本数": 0,
                    "预测低于2800样本数": 0,
                    "预测3000以上样本数": 0,
                    "预测低于2800比例_%": np.nan,
                    "预测3000以上比例_%": np.nan,
                    "平均真实CO": np.nan,
                    "平均预测CO": np.nan,
                    "平均达标概率": np.nan,
                    "达标概率是否失效": True,
                    "诊断结论": "无可信达标样本，无法校验调控模型是否识别达标区域。",
                }
            ]
        )
        save_csv(detail, DATA_OUT / "可信达标样本_调控模型预测复核.csv")
        save_csv(summary, DATA_OUT / "可信达标样本_调控模型预测复核摘要.csv")
        return detail, summary, {"probability_effective": False, "model_erases_standard_region": True}

    pred = control_model.ridge_predict(audit_source)
    pred_q05 = control_model.predict_low_quantile(audit_source, Q05_ALPHA) if hasattr(control_model, "predict_low_quantile") else pred
    prob_near = control_model.predict_near_probability(audit_source) if hasattr(control_model, "predict_near_probability") else np.zeros(len(audit_source))
    prob = control_model.predict_standard_probability(audit_source) if hasattr(control_model, "predict_standard_probability") else np.zeros(len(audit_source))
    regimes = control_model.assign_regime(audit_source) if hasattr(control_model, "assign_regime") else np.full(len(audit_source), -1, dtype=int)
    regime_names = {}
    if hasattr(control_model, "regime_profiles") and not control_model.regime_profiles.empty:
        regime_names = control_model.regime_profiles.set_index("工况层")["工况层名称"].to_dict()
    detail = audit_source[["序列", "时间_s", "数据集", TARGET_COL]].copy()
    detail = detail.rename(columns={TARGET_COL: "真实修正CO"})
    detail["Q10_CO_low_final"] = pred
    detail["Q05_CO_low_extreme"] = pred_q05
    detail["调控模型预测CO"] = pred
    detail["近达标概率Pr_CO_le_3200"] = prob_near
    detail["严格达标概率Pr_CO_le_2800"] = prob
    detail["分类器达标概率"] = prob
    detail["预测误差_预测减真实"] = detail["Q10_CO_low_final"] - detail["真实修正CO"]
    detail["预测是否低于2800"] = detail["Q10_CO_low_final"] <= EMISSION_STANDARD
    detail["Q05预测是否低于2800"] = detail["Q05_CO_low_extreme"] <= EMISSION_STANDARD
    detail["预测是否3000以上"] = detail["Q10_CO_low_final"] >= 3000
    detail["所属工况层"] = regimes
    detail["所属工况层名称"] = [regime_names.get(int(r), f"工况层{int(r)}") for r in regimes]
    detail = detail.sort_values(["真实修正CO", "序列"]).reset_index(drop=True)

    predicted_under = int(detail["预测是否低于2800"].sum())
    predicted_above_3000 = int(detail["预测是否3000以上"].sum())
    probability_effective = bool(predicted_under / len(detail) >= 0.20 and float(np.nanmean(prob)) >= 0.10)
    erases = bool(predicted_above_3000 / len(detail) >= 0.80 or predicted_under == 0)
    conclusion = (
        "调控模型对真实达标样本仍有可用识别能力。"
        if probability_effective
        else "调控模型未能有效识别真实达标样本，达标概率项不进入优化目标，只保留为诊断输出。"
    )
    summary = pd.DataFrame(
        [
            {
                "可信达标样本数": int(len(detail)),
                "预测低于2800样本数": predicted_under,
                "预测3000以上样本数": predicted_above_3000,
                "预测低于2800比例_%": float(predicted_under / len(detail) * 100),
                "预测3000以上比例_%": float(predicted_above_3000 / len(detail) * 100),
                "平均真实CO": float(detail["真实修正CO"].mean()),
                "平均预测CO": float(detail["Q10_CO_low_final"].mean()),
                "预测CO中位数": float(detail["Q10_CO_low_final"].median()),
                "平均Q05_CO": float(detail["Q05_CO_low_extreme"].mean()),
                "平均近达标概率": float(np.nanmean(prob_near)),
                "平均预测误差": float(detail["预测误差_预测减真实"].mean()),
                "平均达标概率": float(np.nanmean(prob)),
                "达标概率是否失效": bool(not probability_effective),
                "是否抹平真实达标区域": erases,
                "诊断结论": conclusion,
            }
        ]
    )
    save_csv(detail, DATA_OUT / "可信达标样本_调控模型预测复核.csv")
    save_csv(summary, DATA_OUT / "可信达标样本_调控模型预测复核摘要.csv")
    return detail, summary, {"probability_effective": probability_effective, "model_erases_standard_region": erases}


@dataclass
class Q1TemperatureLinker:
    feature_names: list[str]
    target_names: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    target_history: np.ndarray
    target_p01: np.ndarray
    target_p99: np.ndarray
    fit_rows: int
    model: object | None = None
    source_note: str = ""

    @classmethod
    def fit_from_q1_outputs(cls, deps: dict[str, Path]) -> "Q1TemperatureLinker":
        if "第1问模型文件" in deps:
            feature_df = pd.read_csv(deps["第1问特征矩阵"], encoding="utf-8-sig")
            feature_index = pd.read_csv(deps["第1问特征索引"], encoding="utf-8-sig")
            feature_names = list(feature_df.columns)
            target_names = [f"box_{i:02d}_avg_temp" for i in BOX_IDS]
            clean = pd.read_csv(deps["第1问清洗数据"], encoding="utf-8-sig")
            target_cols = []
            for name in target_names:
                m = re.match(r"^box_(\d{2})_avg_temp$", name)
                if not m:
                    raise ValueError(f"第1问当前模型目标字段无法映射：{name}")
                target_cols.append(f"{int(m.group(1))}#风箱平均温度")
            clean_by_seq = clean.set_index("序列")
            aligned_targets = clean_by_seq.loc[feature_index["序列"].astype(int), target_cols].reset_index(drop=True)
            x = feature_df[feature_names].to_numpy(dtype=float)
            y = aligned_targets.to_numpy(dtype=float)
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            model.fit(x, y)
            scaler = model.named_steps["standardscaler"]
            ridge = model.named_steps["ridge"]
            y_full = clean[target_cols].to_numpy(dtype=float)
            return cls(
                feature_names=feature_names,
                target_names=target_names,
                scaler_mean=np.asarray(scaler.mean_, dtype=float),
                scaler_scale=np.asarray(scaler.scale_, dtype=float),
                coef=np.asarray(ridge.coef_, dtype=float),
                intercept=np.asarray(ridge.intercept_, dtype=float),
                target_history=y_full,
                target_p01=np.quantile(y_full, 0.01, axis=0),
                target_p99=np.quantile(y_full, 0.99, axis=0),
                fit_rows=int(len(feature_df)),
                source_note="当前第一问pressure-only特征 + Ridge_alpha10温度增量兼容层",
            )

        schema = load_json(deps["第1问模型schema"])
        feature_names = schema["feature_names"]
        target_names = schema["target_names"]
        feature_df = pd.read_csv(deps["第1问模型特征"], encoding="utf-8-sig")
        target_df = pd.read_csv(deps["第1问模型目标"], encoding="utf-8-sig")
        x = feature_df[feature_names].to_numpy(dtype=float)
        y = target_df[target_names].to_numpy(dtype=float)
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x, y)
        scaler = model.named_steps["standardscaler"]
        ridge = model.named_steps["ridge"]
        return cls(
            feature_names=feature_names,
            target_names=target_names,
            scaler_mean=np.asarray(scaler.mean_, dtype=float),
            scaler_scale=np.asarray(scaler.scale_, dtype=float),
            coef=np.asarray(ridge.coef_, dtype=float),
            intercept=np.asarray(ridge.intercept_, dtype=float),
            target_history=y,
            target_p01=np.quantile(y, 0.01, axis=0),
            target_p99=np.quantile(y, 0.99, axis=0),
            fit_rows=int(len(feature_df)),
            source_note="M2_enhanced_online + Ridge_alpha10",
        )

    def _linked_feature_vector(self, intensity: np.ndarray, speed: float) -> tuple[np.ndarray, list[str]]:
        raw_pressure = -np.asarray(intensity, dtype=float)
        values = np.zeros(len(self.feature_names), dtype=float)
        unhandled: list[str] = []
        abs_pressure = np.asarray(intensity, dtype=float)
        positions = np.arange(1, 19, dtype=float)
        lengths = np.array([4.0 if i <= 15 else 2.0 for i in BOX_IDS], dtype=float)
        segments = {
            "front": np.arange(0, 5),
            "middle": np.arange(5, 12),
            "back": np.arange(12, 18),
            "long": np.arange(0, 15),
            "short": np.arange(15, 18),
        }

        def box_value(box: int) -> float:
            return float(raw_pressure[box - 1])

        def current_q1_value(name: str) -> float | None:
            if name == "speed":
                return float(speed)

            m = re.match(r"^p_(\d{2})$", name)
            if m:
                return box_value(int(m.group(1)))
            m = re.match(r"^p_(\d{2})_abs_suction$", name)
            if m:
                return float(abs_pressure[int(m.group(1)) - 1])
            m = re.match(r"^p_adjacent_diff_(\d{2})_(\d{2})$", name)
            if m:
                left = int(m.group(1)) - 1
                right = int(m.group(2)) - 1
                return float(raw_pressure[left] - raw_pressure[right])
            m = re.match(r"^p_adjacent_absdiff_(\d{2})_(\d{2})$", name)
            if m:
                left = int(m.group(1)) - 1
                right = int(m.group(2)) - 1
                return float(abs(raw_pressure[left] - raw_pressure[right]))
            m = re.match(r"^p_(\d{2})_(neighbor_mean|local_mean|local_range|minus_neighbor_mean)$", name)
            if m:
                box = int(m.group(1))
                kind = m.group(2)
                neighbors = [j for j in [box - 1, box + 1] if 1 <= j <= 18]
                local = [j for j in [box - 1, box, box + 1] if 1 <= j <= 18]
                neighbor_vals = raw_pressure[np.array(neighbors) - 1]
                local_vals = raw_pressure[np.array(local) - 1]
                if kind == "neighbor_mean":
                    return float(neighbor_vals.mean())
                if kind == "local_mean":
                    return float(local_vals.mean())
                if kind == "local_range":
                    return float(local_vals.max() - local_vals.min())
                return float(box_value(box) - neighbor_vals.mean())

            if name == "p_all_mean":
                return float(raw_pressure.mean())
            if name == "p_all_std":
                return float(np.std(raw_pressure, ddof=1))
            if name == "p_all_min":
                return float(raw_pressure.min())
            if name == "p_all_max":
                return float(raw_pressure.max())
            if name == "p_all_range":
                return float(raw_pressure.max() - raw_pressure.min())
            if name == "p_abs_all_mean":
                return float(abs_pressure.mean())
            if name == "p_abs_length_weighted_mean":
                return float((abs_pressure * lengths).sum() / lengths.sum())
            if name == "p_abs_position_weighted_mean":
                return float((abs_pressure * positions).sum() / positions.sum())
            if name == "p_abs_center_of_suction":
                denom = float(abs_pressure.sum())
                return float((abs_pressure * positions).sum() / denom) if denom > 0 else float(positions.mean())

            m = re.match(r"^(front|middle|back|long|short)_p_(mean|std|min|max|range|abs_mean|abs_length_weighted_mean)$", name)
            if m:
                seg = segments[m.group(1)]
                kind = m.group(2)
                raw_vals = raw_pressure[seg]
                abs_vals = abs_pressure[seg]
                seg_lengths = lengths[seg]
                if kind == "mean":
                    return float(raw_vals.mean())
                if kind == "std":
                    return float(np.std(raw_vals, ddof=1))
                if kind == "min":
                    return float(raw_vals.min())
                if kind == "max":
                    return float(raw_vals.max())
                if kind == "range":
                    return float(raw_vals.max() - raw_vals.min())
                if kind == "abs_mean":
                    return float(abs_vals.mean())
                return float((abs_vals * seg_lengths).sum() / seg_lengths.sum())

            segment_mean = {
                "front": float(raw_pressure[segments["front"]].mean()),
                "middle": float(raw_pressure[segments["middle"]].mean()),
                "back": float(raw_pressure[segments["back"]].mean()),
                "long": float(raw_pressure[segments["long"]].mean()),
                "short": float(raw_pressure[segments["short"]].mean()),
            }
            if name == "front_minus_middle_p_mean":
                return segment_mean["front"] - segment_mean["middle"]
            if name == "middle_minus_back_p_mean":
                return segment_mean["middle"] - segment_mean["back"]
            if name == "front_minus_back_p_mean":
                return segment_mean["front"] - segment_mean["back"]
            if name == "long_minus_short_p_mean":
                return segment_mean["long"] - segment_mean["short"]

            if re.match(r"^speed_lag_\d+$", name) or re.match(r"^speed_roll\d+_mean$", name):
                return float(speed)
            if re.match(r"^speed_(delta|rate)_\d+$", name) or re.match(r"^speed_roll\d+_std$", name):
                return 0.0
            m = re.match(r"^p_(\d{2})_lag_\d+$", name)
            if m:
                return box_value(int(m.group(1)))
            m = re.match(r"^p_(\d{2})_(delta|rate)_\d+$", name)
            if m:
                return 0.0
            m = re.match(r"^p_(\d{2})_roll\d+_(mean|std|range)$", name)
            if m:
                return box_value(int(m.group(1))) if m.group(2) == "mean" else 0.0
            return None

        def pressure_std(vals: np.ndarray) -> float:
            return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        for idx, name in enumerate(self.feature_names):
            current_value = current_q1_value(name)
            if current_value is not None:
                values[idx] = current_value
                continue

            m = re.match(r"^(\d+)#风箱负压_lag\d+$", name)
            if m:
                values[idx] = box_value(int(m.group(1)))
                continue
            m = re.match(r"^(\d+)#风箱负压_diff(?:1|5|10)$", name)
            if m:
                values[idx] = 0.0
                continue
            m = re.match(r"^(\d+)#风箱负压_roll\d+_mean$", name)
            if m:
                values[idx] = box_value(int(m.group(1)))
                continue
            m = re.match(r"^(\d+)#风箱负压_roll\d+_std$", name)
            if m:
                values[idx] = 0.0
                continue
            m = re.match(r"^(\d+)#风箱负压_sq$", name)
            if m:
                values[idx] = box_value(int(m.group(1))) ** 2
                continue

            if name.startswith("全风箱负压均值_lag"):
                values[idx] = float(raw_pressure.mean())
                continue
            if name.startswith("全风箱负压标准差_lag"):
                values[idx] = pressure_std(raw_pressure)
                continue
            if name.startswith("前段风箱负压均值_lag"):
                values[idx] = float(raw_pressure[0:6].mean())
                continue
            if name.startswith("中段风箱负压均值_lag"):
                values[idx] = float(raw_pressure[6:12].mean())
                continue
            if name.startswith("后段风箱负压均值_lag"):
                values[idx] = float(raw_pressure[12:18].mean())
                continue

            if re.match(rf"^{re.escape(SPEED_COL)}_lag\d+$", name):
                values[idx] = float(speed)
                continue
            if re.match(rf"^{re.escape(SPEED_COL)}_diff(?:1|5)$", name):
                values[idx] = 0.0
                continue
            if re.match(rf"^{re.escape(SPEED_COL)}_roll\d+_mean$", name):
                values[idx] = float(speed)
                continue

            m = re.match(r"^(\d+)#风箱合并温度_hist_lag\d+$", name)
            if m:
                box = int(m.group(1)) - 1
                lag = int(re.search(r"lag(\d+)$", name).group(1))
                values[idx] = float(self.target_history[-lag, box])
                continue
            m = re.match(r"^(\d+)#风箱合并温度_hist_roll\d+_mean$", name)
            if m:
                box = int(m.group(1)) - 1
                window = int(re.search(r"roll(\d+)_mean$", name).group(1))
                values[idx] = float(self.target_history[-window:, box].mean())
                continue
            m = re.match(r"^(\d+)#风箱合并温度_hist_roll\d+_std$", name)
            if m:
                box = int(m.group(1)) - 1
                window = int(re.search(r"roll(\d+)_std$", name).group(1))
                values[idx] = float(np.std(self.target_history[-window:, box], ddof=1))
                continue
            unhandled.append(name)

        return values, unhandled

    def predict_linked_temperature(self, intensity: np.ndarray, speed: float) -> tuple[np.ndarray, list[str], int]:
        values, unhandled = self._linked_feature_vector(intensity, speed)
        if self.model is not None:
            frame = pd.DataFrame([values], columns=self.feature_names)
            raw_temp = np.asarray(self.model.predict(frame), dtype=float).reshape(-1)
            clipped = np.clip(raw_temp, self.target_p01, self.target_p99)
            clipped_count = int(np.sum(np.abs(clipped - raw_temp) > 1e-9))
            return clipped, unhandled, clipped_count
        scale = self.scaler_scale.copy()
        scale[scale == 0] = 1.0
        z = (values - self.scaler_mean) / scale
        raw_temp = self.intercept + self.coef @ z
        clipped = np.clip(raw_temp, self.target_p01, self.target_p99)
        clipped_count = int(np.sum(np.abs(clipped - raw_temp) > 1e-9))
        return clipped, unhandled, clipped_count


def pressure_feature_map(model: LinearCOModel) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for name in model.feature_names:
        m = re.match(r"^(\d+)#风箱负压强度__lag\d+$", name)
        if m:
            mapping[int(m.group(1))] = name
    return mapping


def pressure_adjustment_feature_map(model: object) -> dict[int, list[str]]:
    mapping: dict[int, list[str]] = {}
    for name in model.feature_names:
        m = re.match(r"^(\d+)#风箱负压强度调节量__", name)
        if m:
            mapping.setdefault(int(m.group(1)), []).append(name)
    return mapping


def temperature_feature_map(model: LinearCOModel) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for name in model.feature_names:
        m = re.match(r"^(\d+)#风箱平均温度__lag\d+$", name)
        if m:
            mapping[int(m.group(1))] = name
    return mapping


def apply_plan_to_frame(frame: pd.DataFrame, intensity: np.ndarray, temperature: np.ndarray, model: LinearCOModel) -> pd.DataFrame:
    planned = frame.copy()
    p_map = pressure_feature_map(model)
    adj_map = pressure_adjustment_feature_map(model)
    base_pressure = {box_id: planned[feature].astype(float).copy() for box_id, feature in p_map.items() if feature in planned.columns}
    for box_id, features in adj_map.items():
        if box_id in base_pressure:
            for feature in features:
                planned[feature] = float(intensity[box_id - 1]) - base_pressure[box_id]
    for box_id, feature in p_map.items():
        planned[feature] = float(intensity[box_id - 1])
    for box_id, feature in temperature_feature_map(model).items():
        planned[feature] = float(temperature[box_id - 1])
    return planned


def speed_from_row(row: pd.Series) -> float:
    for key in [f"{SPEED_COL}__lag0", f"{SPEED_COL}__lag{CONTROL_EFFECT_HORIZON_S}", SPEED_COL]:
        if key in row.index:
            return float(row[key])
    for key in row.index:
        if str(key).startswith(SPEED_COL):
            return float(row[key])
    return 0.0


def latest_intensity_from_cleaned(cleaned: pd.DataFrame) -> np.ndarray:
    latest = cleaned.loc[cleaned["序列"].idxmax()]
    return np.array([float(latest[f"{i}#风箱负压强度"]) for i in BOX_IDS], dtype=float)


def current_intensity_from_cleaned_sequence(cleaned_by_sequence: pd.DataFrame, sequence: float | int) -> np.ndarray:
    seq = int(sequence)
    if seq in cleaned_by_sequence.index:
        row = cleaned_by_sequence.loc[seq]
    else:
        nearest_pos = int(np.argmin(np.abs(cleaned_by_sequence.index.to_numpy(dtype=int) - seq)))
        row = cleaned_by_sequence.iloc[nearest_pos]
    return np.array([float(row[f"{i}#风箱负压强度"]) for i in BOX_IDS], dtype=float)


def lagged_intensity_from_row(row: pd.Series, model: LinearCOModel) -> np.ndarray:
    mapping = pressure_feature_map(model)
    return np.array([float(row[mapping[i]]) for i in BOX_IDS], dtype=float)


def lagged_temperature_from_row(row: pd.Series, model: LinearCOModel) -> np.ndarray:
    mapping = temperature_feature_map(model)
    return np.array([float(row[mapping[i]]) for i in BOX_IDS], dtype=float)


def incremental_temperature_for_plan(
    row: pd.Series,
    intensity: np.ndarray,
    base_intensity: np.ndarray,
    model: LinearCOModel,
    q1_linker: Q1TemperatureLinker,
    speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, list[str]]:
    q2_baseline_temp = lagged_temperature_from_row(row, model)
    q1_baseline_temp, unhandled_base, clip_base = q1_linker.predict_linked_temperature(np.asarray(base_intensity, dtype=float), speed)
    q1_plan_temp, unhandled_plan, clip_plan = q1_linker.predict_linked_temperature(np.asarray(intensity, dtype=float), speed)
    temp_delta = q1_plan_temp - q1_baseline_temp
    scenario_temp = q2_baseline_temp + temp_delta
    unhandled = sorted(set(unhandled_base) | set(unhandled_plan))
    return scenario_temp, q1_baseline_temp, q1_plan_temp, temp_delta, max(clip_base, clip_plan), unhandled


def objective_factory(
    base_frame: pd.DataFrame,
    control_model: LinearCOModel,
    q1_linker: Q1TemperatureLinker,
    speed: float,
    base_intensity: np.ndarray,
) -> tuple[callable, dict[str, object]]:
    cache: dict[tuple[float, ...], tuple[float, np.ndarray]] = {}
    base_row = base_frame.iloc[0]
    q2_baseline_temp = lagged_temperature_from_row(base_row, control_model)
    base_control_frame = apply_plan_to_frame(base_frame, np.asarray(base_intensity, dtype=float), q2_baseline_temp, control_model)
    base_pred = float(control_model.ridge_predict(base_control_frame)[0])
    diagnostics: dict[str, object] = {
        "evaluated_candidates": 0,
        "max_temperature_clip_count": 0,
        "unhandled_q1_features": [],
        "baseline_control_model_output": base_pred,
    }

    def objective(intensity: np.ndarray) -> float:
        key = tuple(np.round(np.asarray(intensity, dtype=float), 8))
        if key not in cache:
            temp, _, _, temp_delta, clipped_count, unhandled = incremental_temperature_for_plan(
                base_row,
                np.asarray(intensity, dtype=float),
                np.asarray(base_intensity, dtype=float),
                control_model,
                q1_linker,
                speed,
            )
            if unhandled:
                diagnostics["unhandled_q1_features"] = sorted(set(diagnostics["unhandled_q1_features"]) | set(unhandled))
            diagnostics["max_temperature_clip_count"] = max(int(diagnostics["max_temperature_clip_count"]), clipped_count)
            planned = apply_plan_to_frame(base_frame, np.asarray(intensity, dtype=float), temp, control_model)
            pred = float(control_model.ridge_predict(planned)[0])
            cache[key] = (pred - base_pred, temp)
            diagnostics["evaluated_candidates"] += 1
        return cache[key][0]

    return objective, diagnostics


def direct_prediction_factory(
    base_frame: pd.DataFrame,
    control_model: LinearCOModel,
    q1_linker: Q1TemperatureLinker,
    speed: float,
    reference_intensity: np.ndarray,
) -> tuple[callable, dict[str, object]]:
    cache: dict[tuple[float, ...], tuple[float, np.ndarray]] = {}
    base_row = base_frame.iloc[0]
    diagnostics: dict[str, object] = {
        "unique_plan_evaluations": 0,
        "max_temperature_clip_count": 0,
        "unhandled_q1_features": [],
    }

    def predict_co(intensity: np.ndarray) -> float:
        key = tuple(np.round(np.asarray(intensity, dtype=float), 8))
        if key not in cache:
            temp, _, _, temp_delta, clipped_count, unhandled = incremental_temperature_for_plan(
                base_row,
                np.asarray(intensity, dtype=float),
                np.asarray(reference_intensity, dtype=float),
                control_model,
                q1_linker,
                speed,
            )
            if unhandled:
                diagnostics["unhandled_q1_features"] = sorted(set(diagnostics["unhandled_q1_features"]) | set(unhandled))
            diagnostics["max_temperature_clip_count"] = max(int(diagnostics["max_temperature_clip_count"]), clipped_count)
            planned = apply_plan_to_frame(base_frame, np.asarray(intensity, dtype=float), temp, control_model)
            pred = float(control_model.ridge_predict(planned)[0])
            cache[key] = (pred, temp)
            diagnostics["unique_plan_evaluations"] += 1
        return cache[key][0]

    return predict_co, diagnostics


def scenario_evaluator_factory(
    base_frame: pd.DataFrame,
    control_model: object,
    q1_linker: Q1TemperatureLinker,
    speed: float,
    reference_intensity: np.ndarray,
) -> tuple[callable, dict[str, object]]:
    cache: dict[tuple[float, ...], dict[str, object]] = {}
    base_row = base_frame.iloc[0]
    diagnostics: dict[str, object] = {
        "unique_plan_evaluations": 0,
        "max_temperature_clip_count": 0,
        "unhandled_q1_features": [],
    }

    def evaluate(intensity: np.ndarray) -> dict[str, object]:
        key = tuple(np.round(np.asarray(intensity, dtype=float), 8))
        if key not in cache:
            temp, _, _, temp_delta, clipped_count, unhandled = incremental_temperature_for_plan(
                base_row,
                np.asarray(intensity, dtype=float),
                np.asarray(reference_intensity, dtype=float),
                control_model,
                q1_linker,
                speed,
            )
            if unhandled:
                diagnostics["unhandled_q1_features"] = sorted(set(diagnostics["unhandled_q1_features"]) | set(unhandled))
            diagnostics["max_temperature_clip_count"] = max(int(diagnostics["max_temperature_clip_count"]), clipped_count)
            planned = apply_plan_to_frame(base_frame, np.asarray(intensity, dtype=float), temp, control_model)
            q10_pred = float(control_model.ridge_predict(planned)[0])
            q05_pred = float(control_model.predict_low_quantile(planned, Q05_ALPHA)[0]) if hasattr(control_model, "predict_low_quantile") else q10_pred
            q05_pred = min(q05_pred, q10_pred)
            if hasattr(control_model, "predict_near_probability"):
                prob_near = float(control_model.predict_near_probability(planned)[0])
            else:
                prob_near = float(np.clip((NEAR_STANDARD + 400 - q10_pred) / 800, 0.0, 1.0))
            if hasattr(control_model, "predict_standard_probability"):
                prob = float(control_model.predict_standard_probability(planned)[0])
            else:
                prob = float(np.clip((EMISSION_STANDARD + 400 - q10_pred) / 800, 0.0, 1.0))
            cache[key] = {
                "pred_co": q10_pred,
                "q10_co": q10_pred,
                "q05_co": q05_pred,
                "near_probability": prob_near,
                "strict_probability": prob,
                "standard_probability": prob,
                "temperature": temp,
                "temperature_delta": temp_delta,
            }
            diagnostics["unique_plan_evaluations"] += 1
        return cache[key]

    return evaluate, diagnostics


def main_allowed_bounds(ranges: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lows = ranges["扩展强度下限_P05"].to_numpy(dtype=float)
    highs = ranges["扩展强度上限_P95"].to_numpy(dtype=float)
    return lows, highs


def is_in_bounds(x: np.ndarray, lows: np.ndarray, highs: np.ndarray, eps: float = 1e-8) -> bool:
    x = np.asarray(x, dtype=float)
    return bool(np.all(x >= lows - eps) and np.all(x <= highs + eps))


def normalized_adjustment(x: np.ndarray, reference: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> float:
    widths = np.maximum(highs - lows, 1e-6)
    return float(np.sum(np.abs(np.asarray(x, dtype=float) - np.asarray(reference, dtype=float)) / widths))


def interval_distance(x: np.ndarray, low: np.ndarray, high: np.ndarray, scale: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    below = np.maximum(low - x, 0.0)
    above = np.maximum(x - high, 0.0)
    return float(np.mean((below + above) / np.maximum(scale, 1e-6)))


def boundary_pressure_penalty(x: np.ndarray, lows: np.ndarray, highs: np.ndarray, margin: float = 0.08) -> float:
    rel = (np.asarray(x, dtype=float) - lows) / np.maximum(highs - lows, 1e-6)
    near_low = np.maximum(margin - rel, 0.0) / margin
    near_high = np.maximum(rel - (1.0 - margin), 0.0) / margin
    return float(np.mean(near_low * near_low + near_high * near_high))


def temperature_violation_score(temp_delta: np.ndarray, mode: str = "theory") -> float:
    limit = THEORY_TEMP_DELTA_LIMIT if mode == "theory" else ENGINEERING_TEMP_DELTA_LIMIT
    delta = np.asarray(temp_delta, dtype=float)
    excess = np.maximum(np.abs(delta) - limit, 0.0) / max(limit, 1e-6)
    return float(np.mean(excess * excess))


def pressure_smoothness(x: np.ndarray, reference: np.ndarray, widths: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    reference = np.asarray(reference, dtype=float)
    scale = np.maximum((widths[:-1] + widths[1:]) / 2.0, 1e-6)
    return float(np.mean(np.abs(np.diff(x) - np.diff(reference)) / scale))


def low_emission_seed_points(
    cleaned: pd.DataFrame,
    ranges: pd.DataFrame,
    latest_actual: np.ndarray,
    historical_stats: pd.DataFrame,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    lows, highs = main_allowed_bounds(ranges)
    lows = np.minimum(lows, np.asarray(latest_actual, dtype=float))
    highs = np.maximum(highs, np.asarray(latest_actual, dtype=float))
    seed_rows: list[dict[str, object]] = []
    seeds: list[np.ndarray] = []

    def add_seed(label: str, values: np.ndarray, source_count: int, source_co: float | None = None) -> None:
        clipped = np.clip(np.asarray(values, dtype=float), lows, highs)
        key = tuple(np.round(clipped, 6))
        if any(tuple(np.round(existing, 6)) == key for existing in seeds):
            return
        seeds.append(clipped)
        seed_rows.append(
            {
                "种子名称": label,
                "来源样本数": int(source_count),
                "来源CO": np.nan if source_co is None else float(source_co),
                "总强度": float(np.sum(clipped)),
                "相对当前总调节量": float(np.sum(np.abs(clipped - latest_actual))),
            }
        )

    hist_low = historical_stats["达标强度P25"].to_numpy(dtype=float)
    hist_median = historical_stats["达标强度中位数"].to_numpy(dtype=float)
    hist_high = historical_stats["达标强度P75"].to_numpy(dtype=float)
    add_seed("可信达标P25模板", hist_low, int(historical_stats["达标样本数"].max()))
    add_seed("可信达标中位模板", hist_median, int(historical_stats["达标样本数"].max()))
    add_seed("可信达标P75模板", hist_high, int(historical_stats["达标样本数"].max()))

    pressure_cols = [f"{i}#风箱负压强度" for i in BOX_IDS]
    candidates = cleaned.dropna(subset=[TARGET_COL] + pressure_cols).copy()
    candidate_defs = [
        ("严格达标样本均值种子", candidates[pd.to_numeric(candidates[TARGET_COL], errors="coerce") <= EMISSION_STANDARD]),
        ("近达标样本均值种子", candidates[pd.to_numeric(candidates[TARGET_COL], errors="coerce") <= NEAR_STANDARD]),
        ("最低CO前12条均值种子", candidates.nsmallest(min(12, len(candidates)), TARGET_COL)),
    ]
    for label, part in candidate_defs:
        if part.empty:
            continue
        add_seed(label, part[pressure_cols].mean(axis=0).to_numpy(dtype=float), len(part), float(part[TARGET_COL].mean()))

    low_rows = candidates.nsmallest(min(6, len(candidates)), TARGET_COL)
    for _, row in low_rows.iterrows():
        add_seed(f"低CO历史样本种子_序列{int(row['序列'])}", row[pressure_cols].to_numpy(dtype=float), 1, float(row[TARGET_COL]))

    seed_df = pd.DataFrame(seed_rows)
    save_csv(seed_df, DATA_MID / "低排放优化初始种子.csv")
    save_csv(seed_df, DATA_OUT / "低排放优化初始种子.csv")
    return seeds, seed_df


def make_historical_compliant_profile(cleaned: pd.DataFrame, latest_actual: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    compliant_mask = trusted_standard_mask(cleaned)
    compliant = cleaned[compliant_mask].copy()
    source = compliant if not compliant.empty else cleaned.copy()
    profile = np.array([float(source[f"{i}#风箱负压强度"].median()) for i in BOX_IDS], dtype=float)
    rows = []
    for i in BOX_IDS:
        s = source[f"{i}#风箱负压强度"].astype(float)
        all_s = cleaned[f"{i}#风箱负压强度"].astype(float)
        rows.append(
            {
                "风箱编号": i,
                "统计样本": "第二问修正CO<=2800且非异常孤立点" if not compliant.empty else "全历史样本_无可信达标样本兜底",
                "达标样本数": int(len(compliant)),
                "全样本数": int(len(cleaned)),
                "达标样本比例_%": float(len(compliant) / len(cleaned) * 100),
                "达标强度P05": float(s.quantile(0.05)),
                "达标强度P25": float(s.quantile(0.25)),
                "达标强度中位数": float(s.median()),
                "达标强度均值": float(s.mean()),
                "达标强度P75": float(s.quantile(0.75)),
                "达标强度P95": float(s.quantile(0.95)),
                "全样本强度中位数": float(all_s.median()),
                "控制时刻实际强度": float(latest_actual[i - 1]),
                "达标中位数较控制时刻变化": float(s.median() - latest_actual[i - 1]),
                "达标中位原始负压": -float(s.median()),
            }
        )
    stats = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "全样本数": int(len(cleaned)),
                "历史达标样本数": int(len(compliant)),
                "修正后CO小于等于2800样本数": int((cleaned[TARGET_COL] <= EMISSION_STANDARD).sum()),
                "历史达标比例_%": float(len(compliant) / len(cleaned) * 100),
                "达标样本CO均值": float(compliant[TARGET_COL].mean()) if not compliant.empty else np.nan,
                "达标样本CO中位数": float(compliant[TARGET_COL].median()) if not compliant.empty else np.nan,
                "达标样本CO最小值": float(compliant[TARGET_COL].min()) if not compliant.empty else np.nan,
                "达标样本CO最大值": float(compliant[TARGET_COL].max()) if not compliant.empty else np.nan,
            }
        ]
    )
    save_csv(stats, DATA_OUT / "历史达标工况风箱负压统计.csv")
    save_csv(summary, DATA_OUT / "历史达标工况摘要.csv")
    return stats, summary, profile


def make_compliant_condition_diagnostics(cleaned: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = cleaned.sort_values("序列").copy()
    compliant_mask = trusted_standard_mask(df)
    df["_可信达标"] = compliant_mask.to_numpy(dtype=bool)
    compliant_count = int(compliant_mask.sum())

    if all(f"{i}#风箱负压强度" in df.columns for i in BOX_IDS):
        df["风箱负压强度均值"] = df[[f"{i}#风箱负压强度" for i in BOX_IDS]].mean(axis=1)
    if all(f"{i}#风箱平均温度" in df.columns for i in range(12, 19)):
        df["风箱后段温度均值"] = df[[f"{i}#风箱平均温度" for i in range(12, 19)]].mean(axis=1)

    run_rows: list[dict[str, object]] = []
    if compliant_count > 0:
        run_marker = compliant_mask.ne(compliant_mask.shift(fill_value=False)).cumsum()
        for run_no, (_, group) in enumerate(df.loc[compliant_mask].groupby(run_marker[compliant_mask]), start=1):
            row = {
                "连续片段编号": run_no,
                "起始序列": int(group["序列"].iloc[0]),
                "结束序列": int(group["序列"].iloc[-1]),
                "样本数": int(len(group)),
                "持续时间_s": int(len(group) * SAMPLE_INTERVAL_S),
                "起始时间_s": float(group["时间_s"].iloc[0]),
                "结束时间_s": float(group["时间_s"].iloc[-1]),
                "CO均值": float(group[TARGET_COL].mean()),
                "CO中位数": float(group[TARGET_COL].median()),
                "CO最小值": float(group[TARGET_COL].min()),
                "CO最大值": float(group[TARGET_COL].max()),
                "机速中位数": float(group[SPEED_COL].median()),
            }
            for col in ["1#大烟道负压强度", "2#大烟道负压强度", "1#大烟道温度", "2#大烟道温度", "风箱负压强度均值", "风箱后段温度均值"]:
                if col in group.columns:
                    row[f"{col}中位数"] = float(group[col].median())
            run_rows.append(row)
    runs = pd.DataFrame(run_rows)
    if not runs.empty:
        runs = runs.sort_values(["样本数", "起始序列"], ascending=[False, True]).reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
                "全样本数": int(len(df)),
                "达标样本数": compliant_count,
                "达标比例_%": float(compliant_count / len(df) * 100) if len(df) else np.nan,
                "连续达标片段数": int(len(runs)),
                "单点片段数": int((runs["样本数"] == 1).sum()) if not runs.empty else 0,
                "片段样本数中位数": float(runs["样本数"].median()) if not runs.empty else 0.0,
                "最长连续样本数": int(runs["样本数"].max()) if not runs.empty else 0,
                "最长连续持续时间_s": int(runs["持续时间_s"].max()) if not runs.empty else 0,
                "持续不少于30s片段数": int((runs["持续时间_s"] >= 30).sum()) if not runs.empty else 0,
                "持续不少于60s片段数": int((runs["持续时间_s"] >= 60).sum()) if not runs.empty else 0,
            }
        ]
    )

    compare_cols = [
        SPEED_COL,
        "1#大烟道负压强度",
        "2#大烟道负压强度",
        "1#大烟道温度",
        "2#大烟道温度",
        "风箱负压强度均值",
        "风箱后段温度均值",
    ]
    compare_rows: list[dict[str, object]] = []
    for col in [c for c in compare_cols if c in df.columns]:
        ok = df.loc[compliant_mask, col].astype(float)
        bad = df.loc[~compliant_mask, col].astype(float)
        if ok.empty or bad.empty:
            continue
        pooled_std = math.sqrt((float(ok.var(ddof=0)) + float(bad.var(ddof=0))) / 2.0)
        compare_rows.append(
            {
                "变量": col,
                "达标中位数": float(ok.median()),
                "未达标中位数": float(bad.median()),
                "差值_达标减未达标": float(ok.median() - bad.median()),
                "达标P25": float(ok.quantile(0.25)),
                "达标P75": float(ok.quantile(0.75)),
                "未达标P25": float(bad.quantile(0.25)),
                "未达标P75": float(bad.quantile(0.75)),
                "达标均值": float(ok.mean()),
                "未达标均值": float(bad.mean()),
                "标准化均值差": float((ok.mean() - bad.mean()) / pooled_std) if pooled_std > 0 else np.nan,
            }
        )
    comparison = pd.DataFrame(compare_rows)

    time_distribution = pd.DataFrame()
    if len(df) > 0:
        binned = df.copy()
        binned["时间段编号"] = pd.qcut(binned["序列"], 10, labels=False, duplicates="drop") + 1
        time_distribution = (
            binned.groupby("时间段编号")
            .agg(
                起始序列=("序列", "min"),
                结束序列=("序列", "max"),
                样本数=("序列", "size"),
                达标样本数=("_可信达标", lambda s: int(s.sum())),
                CO中位数=(TARGET_COL, "median"),
            )
            .reset_index()
        )
        time_distribution["达标比例_%"] = time_distribution["达标样本数"] / time_distribution["样本数"] * 100
        time_distribution["占全部达标样本_%"] = (
            time_distribution["达标样本数"] / compliant_count * 100 if compliant_count > 0 else np.nan
        )

    save_csv(summary, DATA_OUT / "历史达标工况连续性诊断.csv")
    save_csv(runs, DATA_OUT / "历史达标工况连续片段明细.csv")
    save_csv(comparison, DATA_OUT / "历史达标工况关键变量对比.csv")
    save_csv(time_distribution, DATA_OUT / "历史达标工况时间分布.csv")
    return {
        "summary": summary,
        "runs": runs,
        "comparison": comparison,
        "time_distribution": time_distribution,
    }


def make_importance_group_summary(importance: pd.DataFrame) -> pd.DataFrame:
    grouped = importance.copy()
    if "置换RMSE增量" in grouped.columns:
        grouped["置换RMSE增量"] = pd.to_numeric(grouped["置换RMSE增量"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def group_feature(name: str) -> str:
        if "大烟道温度" in name:
            return "大烟道温度"
        if "大烟道负压" in name:
            return "大烟道负压"
        if "风箱负压" in name:
            return "风箱负压"
        if "风箱平均温度" in name or "风箱温度" in name or "合并温度" in name:
            return "风箱温度"
        if "机速" in name:
            return "机速"
        return "其他"

    grouped["变量组"] = grouped["特征名"].map(group_feature)
    grouped["正向RMSE增量"] = grouped["置换RMSE增量"].clip(lower=0)
    summary = (
        grouped.groupby("变量组")
        .agg(
            特征数=("特征名", "size"),
            正向特征数=("正向RMSE增量", lambda s: int((s > 0).sum())),
            RMSE增量合计=("置换RMSE增量", "sum"),
            正向RMSE增量合计=("正向RMSE增量", "sum"),
            平均RMSE增量=("置换RMSE增量", "mean"),
        )
        .reset_index()
    )
    positive_total = float(summary["正向RMSE增量合计"].sum())
    summary["正向重要性占比_%"] = summary["正向RMSE增量合计"] / positive_total * 100 if positive_total > 0 else np.nan
    summary = summary.sort_values("正向RMSE增量合计", ascending=False).reset_index(drop=True)
    save_csv(summary, DATA_OUT / "非线性低排放模型变量组重要性.csv")
    return summary


def optimize_global_standard_model(
    ranges: pd.DataFrame,
    evaluate_plan: callable,
    latest_actual: np.ndarray,
    historical_profile: np.ndarray,
    historical_stats: pd.DataFrame,
    probability_weight: float,
    model_co_weight: float,
    template_weight: float,
    seed_points: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, float, np.ndarray, float, pd.DataFrame, dict[str, object]]:
    lows, highs = main_allowed_bounds(ranges)
    lows = np.minimum(lows, np.asarray(latest_actual, dtype=float))
    highs = np.maximum(highs, np.asarray(latest_actual, dtype=float))
    bounds = list(zip(lows, highs))
    widths = np.maximum(highs - lows, 1e-6)
    center = (lows + highs) / 2
    hist_low = historical_stats["达标强度P25"].to_numpy(dtype=float)
    hist_high = historical_stats["达标强度P75"].to_numpy(dtype=float)
    hist_median = historical_stats["达标强度中位数"].to_numpy(dtype=float)
    starts_raw = [
        np.clip(latest_actual, lows, highs),
        np.clip(historical_profile, lows, highs),
        np.clip(hist_low, lows, highs),
        np.clip(hist_median, lows, highs),
        np.clip(hist_high, lows, highs),
    ]
    if seed_points:
        starts_raw.extend([np.clip(np.asarray(seed, dtype=float), lows, highs) for seed in seed_points[:6]])
    starts_raw.append(center)
    starts: list[np.ndarray] = []
    for start in starts_raw:
        key = tuple(np.round(start, 6))
        if not any(tuple(np.round(item, 6)) == key for item in starts):
            starts.append(start)
    starts = starts[:6]
    search_rows: list[dict[str, object]] = []

    def score_components(x: np.ndarray, target_quantile: str) -> dict[str, float]:
        scenario = evaluate_plan(np.asarray(x, dtype=float))
        q10 = float(scenario.get("q10_co", scenario["pred_co"]))
        q05 = float(scenario.get("q05_co", q10))
        q05 = min(q05, q10)
        pred = q05 if target_quantile == "Q05" else q10
        near_prob = float(scenario.get("near_probability", np.nan))
        prob = float(scenario.get("strict_probability", scenario.get("standard_probability", np.nan)))
        hist_penalty = interval_distance(x, hist_low, hist_high, widths)
        boundary_penalty = boundary_pressure_penalty(x, lows, highs)
        adjust_penalty = normalized_adjustment(x, latest_actual, lows, highs) / len(BOX_IDS)
        smooth_penalty = pressure_smoothness(x, latest_actual, widths)
        temp_penalty = temperature_violation_score(np.asarray(scenario.get("temperature_delta", np.zeros(len(BOX_IDS))), dtype=float), mode="theory")
        gap = max(0.0, (pred - EMISSION_STANDARD) / 500.0)
        near_shortfall = max(0.45 - near_prob, 0.0) if np.isfinite(near_prob) else 0.0
        score = (
            1000.0 * gap * gap
            + pred / 500.0
            + 18.0 * temp_penalty
            + 2.0 * smooth_penalty
            + 0.9 * adjust_penalty
            + template_weight * hist_penalty
            + 0.45 * boundary_penalty
            + probability_weight * near_shortfall * near_shortfall
        )
        return {
            "低排放目标评分": float(score),
            "目标低分位": target_quantile,
            "目标低分位CO_mg_m3": pred,
            "模型预测CO_mg_m3": pred,
            "Q10_CO_low_final": q10,
            "Q05_CO_low_extreme": q05,
            "近达标概率Pr_CO_le_3200": near_prob,
            "严格达标概率Pr_CO_le_2800": prob,
            "达标倾向概率": prob,
            "历史达标IQR距离": hist_penalty,
            "达标缺口gap": float(gap),
            "温度宽约束违规": temp_penalty,
            "负压平滑惩罚": smooth_penalty,
            "近达标概率引导不足": float(near_shortfall),
            "边界惩罚": boundary_penalty,
            "当前工况标准化调节量": normalized_adjustment(x, latest_actual, lows, highs),
            "CO模型项权重": model_co_weight,
            "达标概率项权重": probability_weight,
            "可信达标模板距离权重": template_weight,
        }

    candidates: list[tuple[float, np.ndarray, dict[str, float]]] = []

    for target_quantile in ["Q10", "Q05"]:
        for start_id, start in enumerate(starts, start=1):
            result = minimize(
                lambda x, tq=target_quantile: score_components(np.clip(np.asarray(x, dtype=float), lows, highs), tq)["低排放目标评分"],
                np.asarray(start, dtype=float),
                method="Powell",
                bounds=bounds,
                options={"maxiter": 8, "xtol": 1e-4, "ftol": 1e-5, "disp": False},
            )
            x = np.clip(np.asarray(result.x if result.x is not None else start, dtype=float), lows, highs)
            comp = score_components(x, target_quantile)
            candidates.append((comp["低排放目标评分"], x.copy(), comp))
            search_rows.append(
                {
                    "模型": f"主模型_{target_quantile}_达标缺口优先_Powell",
                    "起点": start_id,
                    "优化成功": bool(result.success),
                    "优化消息": str(result.message),
                    "迭代次数": int(getattr(result, "nit", 0)),
                    "函数评估数": int(getattr(result, "nfev", 0)),
                    **comp,
                }
            )

        for start_id, start in enumerate(starts, start=1):
            comp0 = score_components(np.asarray(start, dtype=float), target_quantile)
            candidates.append((comp0["低排放目标评分"], np.asarray(start, dtype=float).copy(), comp0))
            search_rows.append(
                {
                    "模型": f"主模型_{target_quantile}_低排放种子直接评估",
                    "起点": start_id,
                    "优化成功": True,
                    "优化消息": "低排放候选种子直接纳入候选集",
                    "迭代次数": 0,
                    "函数评估数": 1,
                    **comp0,
                }
            )

    if not candidates:
        raise RuntimeError("主模型未找到允许范围内的负压组合")

    q10_candidates = [item for item in candidates if item[2]["目标低分位"] == "Q10"]
    q05_candidates = [item for item in candidates if item[2]["目标低分位"] == "Q05"]

    def priority_key(item: tuple[float, np.ndarray, dict[str, float]]) -> tuple[float, float, float, float, float, float]:
        comp = item[2]
        return (
            float(comp["达标缺口gap"]),
            float(comp["目标低分位CO_mg_m3"]),
            float(comp["温度宽约束违规"]),
            float(comp["负压平滑惩罚"]),
            float(comp["当前工况标准化调节量"]),
            float(comp["历史达标IQR距离"]),
        )

    best_q10 = min(q10_candidates, key=priority_key)
    if float(best_q10[2]["Q10_CO_low_final"]) <= EMISSION_STANDARD:
        scheme_a = best_q10
        scheme_a_note = "Q10预测已低于2800，作为理论达标优先方案"
    else:
        scheme_a = min(q05_candidates, key=priority_key)
        scheme_a_note = "Q10未压到2800，改用Q05作为理论低排放极限方案"
    scheme_b = min(q10_candidates, key=lambda item: item[0])

    final_plan = scheme_a[1].copy()
    final_comp = scheme_a[2]
    final_co = float(final_comp["目标低分位CO_mg_m3"])
    min_plan = scheme_b[1].copy()
    min_co = float(scheme_b[2]["Q10_CO_low_final"])

    search_df = pd.DataFrame(search_rows)
    save_csv(search_df, DATA_MID / "主模型_全局达标优化搜索记录.csv")
    save_csv(search_df, DATA_MID / "主模型_非线性低排放优化搜索记录.csv")
    save_csv(search_df, DATA_OUT / "主模型_Q10_Q05达标潜力优化搜索记录.csv")

    def scheme_payload(name: str, item: tuple[float, np.ndarray, dict[str, float]], note: str) -> dict[str, object]:
        comp = item[2]
        return {
            "方案": name,
            "说明": note,
            "目标低分位": comp["目标低分位"],
            "目标低分位CO_mg_m3": float(comp["目标低分位CO_mg_m3"]),
            "Q10_CO_low_final": float(comp["Q10_CO_low_final"]),
            "Q05_CO_low_extreme": float(comp["Q05_CO_low_extreme"]),
            "近达标概率Pr_CO_le_3200": float(comp["近达标概率Pr_CO_le_3200"]),
            "严格达标概率Pr_CO_le_2800": float(comp["严格达标概率Pr_CO_le_2800"]),
            "达标缺口gap": float(comp["达标缺口gap"]),
            "温度宽约束违规": float(comp["温度宽约束违规"]),
            "负压平滑惩罚": float(comp["负压平滑惩罚"]),
            "当前工况标准化调节量": float(comp["当前工况标准化调节量"]),
            "历史达标IQR距离": float(comp["历史达标IQR距离"]),
            "目标强度": {str(i): float(item[1][i - 1]) for i in BOX_IDS},
            "目标原始负压": {str(i): -float(item[1][i - 1]) for i in BOX_IDS},
        }

    scheme_a_payload = scheme_payload("方案A_理论达标优先方案", scheme_a, scheme_a_note)
    scheme_b_payload = scheme_payload("方案B_折中低排放方案", scheme_b, "Q10达标缺口目标下综合评分最低，调节量与模板距离相对折中")
    diagnostics = {
        "main_range": "P05-P95",
        "model_role": "Q10/Q05达标潜力评分，不单独承担现场达标判定",
        "target_feasible": bool(final_co <= EMISSION_STANDARD),
        "target_success": bool(final_co <= EMISSION_STANDARD),
        "min_co": float(min_co),
        "score_best_co": float(scheme_b[2]["Q10_CO_low_final"]),
        "final_selection_rule": "先Q10达标缺口优先；若Q10未低于2800，则采用Q05理论极限方案",
        "final_co": float(final_co),
        "final_adjustment_norm": float(final_comp["当前工况标准化调节量"]),
        "final_near_probability": float(final_comp["近达标概率Pr_CO_le_3200"]),
        "final_standard_probability": float(final_comp["严格达标概率Pr_CO_le_2800"]),
        "final_historical_distance": float(final_comp["历史达标IQR距离"]),
        "probability_weight": float(probability_weight),
        "model_co_weight": float(model_co_weight),
        "template_weight": float(template_weight),
        "probability_term_enabled": bool(probability_weight > 0),
        "low_emission_seed_count": int(max(0, len(starts) - 6)),
        "scheme_a": scheme_a_payload,
        "scheme_b": scheme_b_payload,
        "status": (
            "方案A理论低分位已低于2800_需实测复核"
            if final_co <= EMISSION_STANDARD
            else "方案A理论低分位仍未低于2800_只能作为低排放潜力方向"
        ),
    }
    return final_plan, float(final_co), min_plan, float(min_co), search_df, diagnostics


def constrained_bounds_and_limits(
    ranges: pd.DataFrame,
    constraints: dict[str, object],
    base_intensity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    range_lows = ranges["优化强度下限_P10"].to_numpy(dtype=float)
    range_highs = ranges["优化强度上限_P90"].to_numpy(dtype=float)
    single_delta = np.asarray(constraints["single_delta"], dtype=float)
    base = np.asarray(base_intensity, dtype=float)
    lows = np.maximum(range_lows, base - single_delta)
    highs = np.minimum(range_highs, base + single_delta)
    lows = np.minimum(lows, base)
    highs = np.maximum(highs, base)
    fallback = base.copy()
    bad = lows > highs
    lows[bad] = fallback[bad]
    highs[bad] = fallback[bad]
    adjacent_limits = np.asarray(constraints["adjacent_limits"], dtype=float).copy()
    for i in range(1, len(BOX_IDS)):
        adjacent_limits[i - 1] = max(adjacent_limits[i - 1], abs(float(base_intensity[i] - base_intensity[i - 1])))
    total_budget = max(float(constraints["total_budget"]), float(np.sum(np.abs(fallback - base_intensity))))
    return lows, highs, single_delta, adjacent_limits, total_budget


def is_feasible_plan(
    x: np.ndarray,
    base_intensity: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    adjacent_limits: np.ndarray,
    total_budget: float,
    eps: float = 1e-8,
) -> bool:
    x = np.asarray(x, dtype=float)
    if np.any(x < lows - eps) or np.any(x > highs + eps):
        return False
    if np.sum(np.abs(x - base_intensity)) > total_budget + eps:
        return False
    if np.any(np.abs(np.diff(x)) > adjacent_limits + eps):
        return False
    return True


def max_constraint_violation(
    x: np.ndarray,
    base_intensity: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    adjacent_limits: np.ndarray,
    total_budget: float,
) -> float:
    x = np.asarray(x, dtype=float)
    violations = [
        float(np.max(lows - x)),
        float(np.max(x - highs)),
        float(np.sum(np.abs(x - base_intensity)) - total_budget),
        float(np.max(np.abs(np.diff(x)) - adjacent_limits)),
    ]
    return max(0.0, max(violations))


def feasible_initial_point(
    start: np.ndarray,
    base_intensity: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    adjacent_limits: np.ndarray,
    total_budget: float,
) -> np.ndarray:
    x = np.clip(np.asarray(start, dtype=float), lows, highs)
    if is_feasible_plan(x, base_intensity, lows, highs, adjacent_limits, total_budget):
        return x

    fallback = np.clip(np.asarray(base_intensity, dtype=float), lows, highs)
    if is_feasible_plan(fallback, base_intensity, lows, highs, adjacent_limits, total_budget):
        for weight in np.linspace(0.75, 0.0, 4):
            trial = fallback + weight * (x - fallback)
            if is_feasible_plan(trial, base_intensity, lows, highs, adjacent_limits, total_budget):
                return trial
        return fallback

    return x


def coordinate_refine_pressure_plan(
    lows: np.ndarray,
    highs: np.ndarray,
    start_points: list[np.ndarray],
    objective: callable,
    base_intensity: np.ndarray,
    adjacent_limits: np.ndarray,
    total_budget: float,
    grid_points: int,
    max_rounds: int,
    method_name: str,
    start_offset: int = 0,
) -> tuple[np.ndarray | None, float, list[dict[str, object]]]:
    grids = [np.linspace(lows[i], highs[i], grid_points) for i in range(len(BOX_IDS))]
    search_rows: list[dict[str, object]] = []
    best_global: tuple[float, np.ndarray] | None = None

    for start_id, start in enumerate(start_points, start=1):
        x = feasible_initial_point(start, base_intensity, lows, highs, adjacent_limits, total_budget)
        if not is_feasible_plan(x, base_intensity, lows, highs, adjacent_limits, total_budget):
            continue
        value = float(objective(x))
        search_rows.append(
            {
                "求解方法": method_name,
                "起点": start_offset + start_id,
                "轮次": 0,
                "风箱编号": 0,
                "候选数": 1,
                "当前最优模型增量CO": value,
                "最大约束违规量": max_constraint_violation(x, base_intensity, lows, highs, adjacent_limits, total_budget),
            }
        )
        for round_id in range(1, max_rounds + 1):
            improved = False
            for j in range(len(BOX_IDS)):
                candidates = np.unique(np.r_[grids[j], x[j], lows[j], highs[j], (lows[j] + highs[j]) / 2])
                best_j = x[j]
                best_val = value
                for cand in candidates:
                    trial = x.copy()
                    trial[j] = cand
                    if not is_feasible_plan(trial, base_intensity, lows, highs, adjacent_limits, total_budget):
                        continue
                    trial_val = float(objective(trial))
                    if trial_val < best_val - 1e-10:
                        best_val = trial_val
                        best_j = float(cand)
                if best_val < value - 1e-10:
                    improved = True
                    x[j] = best_j
                    value = best_val
                search_rows.append(
                    {
                        "求解方法": method_name,
                        "起点": start_offset + start_id,
                        "轮次": round_id,
                        "风箱编号": j + 1,
                        "候选数": int(len(candidates)),
                        "当前最优模型增量CO": value,
                        "最大约束违规量": max_constraint_violation(x, base_intensity, lows, highs, adjacent_limits, total_budget),
                    }
                )
            if not improved:
                break
        if best_global is None or value < best_global[0]:
            best_global = (value, x.copy())

    if best_global is None:
        return None, float("inf"), search_rows
    return best_global[1], float(best_global[0]), search_rows


def optimize_pressure_plan(
    ranges: pd.DataFrame,
    start_points: list[np.ndarray],
    objective: callable,
    constraints: dict[str, object],
    base_intensity: np.ndarray,
    grid_points: int = 21,
    max_rounds: int = 6,
    save_search: bool = True,
    use_slsqp: bool = True,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    lows, highs, _, adjacent_limits, total_budget = constrained_bounds_and_limits(ranges, constraints, base_intensity)
    search_rows: list[dict[str, object]] = []
    best_global: tuple[float, np.ndarray] | None = None

    if use_slsqp:
        n_box = len(BOX_IDS)
        bounds = list(zip(lows, highs)) + [(0.0, total_budget) for _ in range(n_box)]
        scipy_constraints = [
            {"type": "ineq", "fun": lambda y: total_budget - np.sum(np.asarray(y, dtype=float)[n_box:])},
            {"type": "ineq", "fun": lambda y: np.asarray(y, dtype=float)[n_box:] - (np.asarray(y, dtype=float)[:n_box] - base_intensity)},
            {"type": "ineq", "fun": lambda y: np.asarray(y, dtype=float)[n_box:] + (np.asarray(y, dtype=float)[:n_box] - base_intensity)},
            {"type": "ineq", "fun": lambda y: adjacent_limits - np.diff(np.asarray(y, dtype=float)[:n_box])},
            {"type": "ineq", "fun": lambda y: adjacent_limits + np.diff(np.asarray(y, dtype=float)[:n_box])},
        ]

        for start_id, start in enumerate(start_points, start=1):
            x0 = feasible_initial_point(start, base_intensity, lows, highs, adjacent_limits, total_budget)
            u0 = np.abs(x0 - base_intensity)
            if float(np.sum(u0)) > total_budget:
                u0 = u0 * (total_budget / max(float(np.sum(u0)), 1e-12))
            y0 = np.r_[x0, u0]
            start_value = float(objective(x0))
            result = minimize(
                lambda y: float(objective(np.asarray(y, dtype=float)[:n_box])),
                y0,
                method="SLSQP",
                bounds=bounds,
                constraints=scipy_constraints,
                options={"maxiter": 300, "ftol": 1e-9, "disp": False, "eps": 1e-4},
            )
            x = np.clip(np.asarray(result.x[:n_box] if result.x is not None else x0, dtype=float), lows, highs)
            violation = max_constraint_violation(x, base_intensity, lows, highs, adjacent_limits, total_budget)
            feasible = is_feasible_plan(x, base_intensity, lows, highs, adjacent_limits, total_budget, eps=1e-5)
            value = float(objective(x)) if feasible else start_value
            if feasible and (best_global is None or value < best_global[0]):
                best_global = (value, x.copy())
            search_rows.append(
                {
                    "求解方法": "SLSQP连续优化",
                    "起点": start_id,
                    "轮次": int(getattr(result, "nit", 0)),
                    "风箱编号": 0,
                    "候选数": np.nan,
                    "函数评估数": int(getattr(result, "nfev", 0)),
                    "SLSQP成功": bool(result.success),
                    "SLSQP消息": str(result.message),
                    "当前最优模型增量CO": value,
                    "最大约束违规量": violation,
                }
            )

        if best_global is not None:
            refined_x, refined_value, refine_rows = coordinate_refine_pressure_plan(
                lows,
                highs,
                [best_global[1]],
                objective,
                base_intensity,
                adjacent_limits,
                total_budget,
                max(3, min(grid_points, 11)),
                min(max_rounds, 2),
                "SLSQP后坐标精修",
                start_offset=len(start_points),
            )
            search_rows.extend(refine_rows)
            if refined_x is not None and refined_value < best_global[0] - 1e-10:
                best_global = (refined_value, refined_x.copy())

    if best_global is None:
        fallback_x, fallback_value, fallback_rows = coordinate_refine_pressure_plan(
            lows,
            highs,
            start_points,
            objective,
            base_intensity,
            adjacent_limits,
            total_budget,
            grid_points,
            max_rounds,
            "坐标局部求解" if not use_slsqp else "坐标搜索兜底",
        )
        search_rows.extend(fallback_rows)
        if fallback_x is not None:
            best_global = (fallback_value, fallback_x.copy())

    if best_global is None:
        keep = np.asarray(base_intensity, dtype=float).copy()
        value = float(objective(keep))
        search_rows.append(
            {
                "求解方法": "无可行调节_保持当前",
                "起点": 0,
                "轮次": 0,
                "风箱编号": 0,
                "候选数": 1,
                "当前最优模型增量CO": value,
                "最大约束违规量": max_constraint_violation(keep, base_intensity, lows, highs, adjacent_limits, total_budget),
            }
        )
        best_global = (value, keep)

    search_df = pd.DataFrame(search_rows)
    if save_search:
        save_csv(search_df, DATA_MID / "优化搜索记录.csv")
    return best_global[1], float(best_global[0]), search_df


def make_marginal_table(
    base_frame: pd.DataFrame,
    ranges: pd.DataFrame,
    constraints: dict[str, object],
    base_intensity: np.ndarray,
    control_model: LinearCOModel,
    q1_linker: Q1TemperatureLinker,
    speed: float,
) -> pd.DataFrame:
    rows = []
    base_temp = lagged_temperature_from_row(base_frame.iloc[0], control_model)
    base_frame_for_model = apply_plan_to_frame(base_frame, base_intensity, base_temp, control_model)
    base_pred = float(control_model.ridge_predict(base_frame_for_model)[0])
    lows, highs, _, adjacent_limits, total_budget = constrained_bounds_and_limits(ranges, constraints, base_intensity)
    for i in BOX_IDS:
        low = float(lows[i - 1])
        high = float(highs[i - 1])
        trial_low = base_intensity.copy()
        trial_high = base_intensity.copy()
        trial_low[i - 1] = low
        trial_high[i - 1] = high
        low_feasible = is_feasible_plan(trial_low, base_intensity, lows, highs, adjacent_limits, total_budget)
        high_feasible = is_feasible_plan(trial_high, base_intensity, lows, highs, adjacent_limits, total_budget)
        temp_low, _, q1_low, delta_low, _, _ = incremental_temperature_for_plan(
            base_frame.iloc[0], trial_low, base_intensity, control_model, q1_linker, speed
        )
        temp_high, _, q1_high, delta_high, _, _ = incremental_temperature_for_plan(
            base_frame.iloc[0], trial_high, base_intensity, control_model, q1_linker, speed
        )
        pred_low = float(control_model.ridge_predict(apply_plan_to_frame(base_frame, trial_low, temp_low, control_model))[0]) if low_feasible else np.nan
        pred_high = float(control_model.ridge_predict(apply_plan_to_frame(base_frame, trial_high, temp_high, control_model))[0]) if high_feasible else np.nan
        delta = pred_high - pred_low if np.isfinite(pred_low) and np.isfinite(pred_high) else np.nan
        if np.isfinite(delta):
            suggested_boundary = "P90高强度" if pred_high < pred_low else "P10低强度"
            direction = "负压强度增大会降低CO" if delta < 0 else "负压强度增大会升高CO"
            unit_delta = delta / (high - low) if high > low else 0.0
        else:
            suggested_boundary = "受耦合约束限制"
            direction = "需联合调整判断"
            unit_delta = np.nan
        rows.append(
            {
                "风箱编号": i,
                "调控模型输出_基准": base_pred,
                "当前滞后温度": float(base_temp[i - 1]),
                "P10强度": low,
                "P90强度": high,
                "P10第一问方案温度": float(q1_low[i - 1]),
                "P90第一问方案温度": float(q1_high[i - 1]),
                "P10温度增量": float(delta_low[i - 1]),
                "P90温度增量": float(delta_high[i - 1]),
                "P10增量叠加后温度": float(temp_low[i - 1]),
                "P90增量叠加后温度": float(temp_high[i - 1]),
                "P10单箱边界可行": bool(low_feasible),
                "P90单箱边界可行": bool(high_feasible),
                "调控模型输出_仅该风箱P10": pred_low,
                "调控模型输出_仅该风箱P90": pred_high,
                "P90较P10_CO变化": delta,
                "单位强度边际_CO变化": unit_delta,
                "调控模型建议边界": suggested_boundary,
                "影响方向": direction,
            }
        )
    out = pd.DataFrame(rows)
    save_csv(out, DATA_MID / "模型风箱负压边际系数.csv")
    save_csv(out, DATA_OUT / "风箱调控边际贡献_最新状态.csv")
    return out


def make_decision_table(
    ranges: pd.DataFrame,
    cleaned: pd.DataFrame,
    latest_q2_row: pd.Series,
    constraints: dict[str, object],
    control_model: LinearCOModel,
    best_intensity: np.ndarray,
    scenario_temperature: np.ndarray,
    q1_baseline_temperature: np.ndarray,
    q1_plan_temperature: np.ndarray,
    temperature_delta: np.ndarray,
    marginal: pd.DataFrame,
) -> pd.DataFrame:
    control_seq = int(latest_q2_row.get("控制序列", cleaned["序列"].max()))
    latest_actual = current_intensity_from_cleaned_sequence(cleaned.set_index("序列", drop=False), control_seq)
    lagged = lagged_intensity_from_row(latest_q2_row, control_model)
    lagged_temp = lagged_temperature_from_row(latest_q2_row, control_model)
    lows, highs, single_delta, adjacent_limits, total_budget = constrained_bounds_and_limits(ranges, constraints, latest_actual)
    total_adjust = float(np.sum(np.abs(best_intensity - latest_actual)))
    rows = []
    marginal_map = marginal.set_index("风箱编号").to_dict(orient="index")
    for i in BOX_IDS:
        r = ranges[ranges["风箱编号"] == i].iloc[0].to_dict()
        target = float(best_intensity[i - 1])
        current_latest = float(latest_actual[i - 1])
        current_lagged = float(lagged[i - 1])
        delta_latest = target - current_latest
        if abs(delta_latest) <= 1e-8:
            direction = "保持"
        elif delta_latest > 0:
            direction = "原始负压更负"
        else:
            direction = "原始负压靠近0"
        m = marginal_map[i]
        rows.append(
            {
                **r,
                "控制时刻实际强度": current_latest,
                "第2问当前滞后强度": current_lagged,
                "约束后强度下限": float(lows[i - 1]),
                "约束后强度上限": float(highs[i - 1]),
                "单箱调节上限_Delta_i": float(single_delta[i - 1]),
                "最优目标强度": target,
                "最优目标原始负压": -target,
                "相对控制时刻实际强度变化": delta_latest,
                "相对第2问滞后强度变化": target - current_lagged,
                "总调节预算B": total_budget,
                "本方案总调节量": total_adjust,
                "调整方向": direction,
                "第2问当前滞后温度": float(lagged_temp[i - 1]),
                "第一问基准预测温度": float(q1_baseline_temperature[i - 1]),
                "第一问方案预测温度": float(q1_plan_temperature[i - 1]),
                "第一问温度增量": float(temperature_delta[i - 1]),
                "增量叠加后温度": float(scenario_temperature[i - 1]),
                "绝对替换偏差": float(q1_plan_temperature[i - 1] - scenario_temperature[i - 1]),
                "单位强度边际_CO变化": m["单位强度边际_CO变化"],
                "P90较P10_CO变化": m["P90较P10_CO变化"],
            }
        )
    decision = pd.DataFrame(rows)
    save_csv(decision, DATA_OUT / "风箱负压调控决策_最新状态.csv")
    save_csv(decision, DATA_OUT / "负压温度联动调控决策_最新状态.csv")
    save_csv(
        decision[["风箱编号", "最优目标原始负压", "最优目标强度", "增量叠加后温度", "调整方向"]],
        DATA_OUT / "负压温度联动调控决策_最新状态_简表.csv",
    )
    save_csv(
        decision[["风箱编号", "第2问当前滞后温度", "第一问基准预测温度", "第一问方案预测温度", "第一问温度增量", "增量叠加后温度", "绝对替换偏差"]],
        DATA_OUT / "最优方案联动温度_最新状态.csv",
    )
    temp_note = decision[["风箱编号", "第2问当前滞后温度", "第一问基准预测温度", "第一问方案预测温度", "第一问温度增量", "增量叠加后温度", "绝对替换偏差"]].copy()
    temp_note["本问温度处理"] = "采用第一问温度增量叠加到第二问配准温度；不做第一问绝对温度替换"
    save_csv(temp_note, DATA_OUT / "非调控温度保持_最新状态.csv")
    return decision


def make_global_decision_table(
    ranges: pd.DataFrame,
    cleaned: pd.DataFrame,
    latest_q2_row: pd.Series,
    control_model: LinearCOModel,
    best_intensity: np.ndarray,
    scenario_temperature: np.ndarray,
    q1_baseline_temperature: np.ndarray,
    q1_plan_temperature: np.ndarray,
    temperature_delta: np.ndarray,
    historical_stats: pd.DataFrame,
) -> pd.DataFrame:
    control_seq = int(latest_q2_row.get("控制序列", cleaned["序列"].max()))
    latest_actual = current_intensity_from_cleaned_sequence(cleaned.set_index("序列", drop=False), control_seq)
    lagged = lagged_intensity_from_row(latest_q2_row, control_model)
    lagged_temp = lagged_temperature_from_row(latest_q2_row, control_model)
    lows, highs = main_allowed_bounds(ranges)
    lows = np.minimum(lows, latest_actual)
    highs = np.maximum(highs, latest_actual)
    total_adjust = float(np.sum(np.abs(best_intensity - latest_actual)))
    hist_map = historical_stats.set_index("风箱编号").to_dict(orient="index")
    rows = []
    for i in BOX_IDS:
        r = ranges[ranges["风箱编号"] == i].iloc[0].to_dict()
        target = float(best_intensity[i - 1])
        current_latest = float(latest_actual[i - 1])
        delta_latest = target - current_latest
        if abs(delta_latest) <= 1e-8:
            direction = "保持"
        elif delta_latest > 0:
            direction = "原始负压更负"
        else:
            direction = "原始负压靠近0"
        hist = hist_map[i]
        rows.append(
            {
                **r,
                "主模型允许强度下限_P05": float(lows[i - 1]),
                "主模型允许强度上限_P95": float(highs[i - 1]),
                "控制时刻实际强度": current_latest,
                "第2问当前滞后强度": float(lagged[i - 1]),
                "历史达标强度P25": hist["达标强度P25"],
                "历史达标强度中位数": hist["达标强度中位数"],
                "历史达标强度P75": hist["达标强度P75"],
                "最优目标强度": target,
                "最优目标原始负压": -target,
                "相对控制时刻实际强度变化": delta_latest,
                "相对第2问滞后强度变化": target - float(lagged[i - 1]),
                "本方案总调节量": total_adjust,
                "调整方向": direction,
                "第2问当前滞后温度": float(lagged_temp[i - 1]),
                "第一问基准预测温度": float(q1_baseline_temperature[i - 1]),
                "第一问方案预测温度": float(q1_plan_temperature[i - 1]),
                "第一问温度增量": float(temperature_delta[i - 1]),
                "增量叠加后温度": float(scenario_temperature[i - 1]),
                "绝对替换偏差": float(q1_plan_temperature[i - 1] - scenario_temperature[i - 1]),
            }
        )
    decision = pd.DataFrame(rows)
    save_csv(decision, DATA_OUT / "主模型_全局达标负压组合.csv")
    save_csv(decision, DATA_OUT / "主模型_非线性低排放负压组合.csv")
    save_csv(
        decision[["风箱编号", "主模型允许强度下限_P05", "主模型允许强度上限_P95", "历史达标强度中位数", "控制时刻实际强度", "最优目标强度", "最优目标原始负压", "调整方向"]],
        DATA_OUT / "主模型_全局达标负压组合_简表.csv",
    )
    save_csv(
        decision[["风箱编号", "主模型允许强度下限_P05", "主模型允许强度上限_P95", "历史达标强度中位数", "控制时刻实际强度", "最优目标强度", "最优目标原始负压", "调整方向"]],
        DATA_OUT / "主模型_非线性低排放负压组合_简表.csv",
    )
    save_csv(
        decision[["风箱编号", "第2问当前滞后温度", "第一问基准预测温度", "第一问方案预测温度", "第一问温度增量", "增量叠加后温度", "绝对替换偏差"]],
        DATA_OUT / "主模型_全局达标联动温度.csv",
    )
    return decision


def make_global_comparison(
    cleaned: pd.DataFrame,
    latest_q2: pd.DataFrame,
    ranges: pd.DataFrame,
    control_model: LinearCOModel,
    evaluate_plan: callable,
    historical_stats: pd.DataFrame,
    latest_actual: np.ndarray,
    best_intensity: np.ndarray,
    best_co: float,
    min_intensity: np.ndarray,
    min_co: float,
    diagnostics: dict[str, object],
) -> pd.DataFrame:
    latest_observed = float(latest_q2.iloc[0][TARGET_COL]) if TARGET_COL in latest_q2.columns else float(cleaned.loc[cleaned["序列"].idxmax(), TARGET_COL])
    base_eval = evaluate_plan(latest_actual)
    min_eval = evaluate_plan(min_intensity)
    best_eval = evaluate_plan(best_intensity)
    base_direct = float(base_eval.get("q10_co", base_eval["pred_co"]))
    lows, highs = main_allowed_bounds(ranges)
    lows = np.minimum(lows, latest_actual)
    highs = np.maximum(highs, latest_actual)
    widths = np.maximum(highs - lows, 1e-6)
    hist_low = historical_stats["达标强度P25"].to_numpy(dtype=float)
    hist_high = historical_stats["达标强度P75"].to_numpy(dtype=float)
    scheme_a = diagnostics.get("scheme_a", {})
    scheme_b = diagnostics.get("scheme_b", {})

    def row_for_plan(name: str, role: str, eval_data: dict[str, object], intensity: np.ndarray, target_co: float, note: str) -> dict[str, object]:
        q10 = float(eval_data.get("q10_co", eval_data["pred_co"]))
        q05 = min(float(eval_data.get("q05_co", q10)), q10)
        near_prob = float(eval_data.get("near_probability", np.nan))
        strict_prob = float(eval_data.get("strict_probability", eval_data.get("standard_probability", np.nan)))
        return {
            "场景": name,
            "模型角色": role,
            "实际观测CO_mg_m3": np.nan,
            "非线性模型预测CO_mg_m3": target_co,
            "Q10_CO_low_final": q10,
            "Q05_CO_low_extreme": q05,
            "近达标概率Pr_CO_le_3200": near_prob,
            "严格达标概率Pr_CO_le_2800": strict_prob,
            "达标倾向概率": strict_prob,
            "相对2800": target_co - EMISSION_STANDARD,
            "模型是否低于2800_仅参考": target_co <= EMISSION_STANDARD,
            "较当前模型预测降低量": base_direct - target_co,
            "标准化调节量": normalized_adjustment(intensity, latest_actual, lows, highs),
            "历史达标IQR距离": interval_distance(intensity, hist_low, hist_high, widths),
            "说明": note,
        }

    rows = [
        {
            "场景": "控制时刻基准工况",
            "模型角色": "基准",
            "实际观测CO_mg_m3": latest_observed,
            "非线性模型预测CO_mg_m3": base_direct,
            "Q10_CO_low_final": float(base_eval.get("q10_co", base_direct)),
            "Q05_CO_low_extreme": min(float(base_eval.get("q05_co", base_direct)), base_direct),
            "近达标概率Pr_CO_le_3200": float(base_eval.get("near_probability", np.nan)),
            "严格达标概率Pr_CO_le_2800": float(base_eval.get("strict_probability", base_eval.get("standard_probability", np.nan))),
            "达标倾向概率": float(base_eval.get("strict_probability", base_eval.get("standard_probability", np.nan))),
            "相对2800": base_direct - EMISSION_STANDARD,
            "模型是否低于2800_仅参考": base_direct <= EMISSION_STANDARD,
            "较当前模型预测降低量": 0.0,
            "标准化调节量": 0.0,
            "历史达标IQR距离": interval_distance(latest_actual, hist_low, hist_high, widths),
            "说明": "模型口径下的基准比较，不作为达标判定；Q10为主优化低分位",
        },
    ]
    rows.append(
        row_for_plan(
            "方案A_理论达标优先方案",
            "理论低排放潜力",
            best_eval,
            best_intensity,
            best_co,
            str(scheme_a.get("说明", diagnostics["status"])),
        )
    )
    rows.append(
        row_for_plan(
            "方案B_折中低排放方案",
            "综合评分折中",
            min_eval,
            min_intensity,
            min_co,
            str(scheme_b.get("说明", "Q10达标缺口目标下综合评分最低")),
        )
    )
    comp = pd.DataFrame(rows)
    save_csv(comp, DATA_OUT / "主模型_非线性低排放评分对比.csv")
    save_csv(comp, DATA_OUT / "主模型_CO直接预测对比.csv")
    save_csv(comp, DATA_OUT / "方案A_B理论与折中低排放对比.csv")
    return comp


def make_latest_comparison(
    cleaned: pd.DataFrame,
    latest_q2: pd.DataFrame,
    q1_linker: Q1TemperatureLinker,
    control_model: LinearCOModel,
    base_intensity: np.ndarray,
    best_intensity: np.ndarray,
    scenario_temperature: np.ndarray,
    best_control_delta: float,
    speed: float,
) -> pd.DataFrame:
    base = latest_q2.copy()
    latest_actual_co = float(cleaned.loc[cleaned["序列"].idxmax(), TARGET_COL])
    q2_baseline_temp = lagged_temperature_from_row(base.iloc[0], control_model)
    base_control_frame = apply_plan_to_frame(base, base_intensity, q2_baseline_temp, control_model)
    control_before = float(control_model.ridge_predict(base_control_frame)[0])
    best_control_frame = apply_plan_to_frame(base, best_intensity, scenario_temperature, control_model)
    control_after = float(control_model.ridge_predict(best_control_frame)[0])
    model_delta = control_after - control_before
    rows = [
        {
            "场景": "辅助模型_控制时刻基准工况",
            "口径": "调控模型直接预测",
            "实际观测CO_mg_m3": latest_actual_co,
            "调控模型直接预测CO_mg_m3": control_before,
            "模型增量_mg_m3": 0.0,
            "相对2800": control_before - EMISSION_STANDARD,
            "是否达标": control_before <= EMISSION_STANDARD,
            "模型预测降低量": 0.0,
            "说明": "当前工况平稳调控的模型预测基准",
        },
        {
            "场景": "辅助模型_小幅平稳优化",
            "口径": "调控模型直接预测",
            "实际观测CO_mg_m3": np.nan,
            "调控模型直接预测CO_mg_m3": control_after,
            "模型增量_mg_m3": model_delta,
            "相对2800": control_after - EMISSION_STANDARD,
            "是否达标": control_after <= EMISSION_STANDARD,
            "模型预测降低量": control_before - control_after,
            "说明": "只允许当前工况附近的小幅平稳调整，因此仅作为辅助说明",
        },
    ]
    if abs(model_delta - best_control_delta) > 1e-6:
        raise ValueError(f"优化增量与复算增量不一致：{best_control_delta} vs {model_delta}")
    comp = pd.DataFrame(rows)
    save_csv(comp, DATA_OUT / "辅助模型_当前工况平稳调控CO直接预测.csv")
    save_csv(comp, DATA_OUT / "优化前后CO对比_最新状态.csv")
    return comp


def make_three_scheme_outputs(
    main_comp: pd.DataFrame,
    aux_decision: pd.DataFrame,
    aux_comp: pd.DataFrame,
    main_diagnostics: dict[str, object],
    aux_diagnostics: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for scenario_name, plan_type in [
        ("方案A_理论达标优先方案", "理论达标优先"),
        ("方案B_折中低排放方案", "折中低排放"),
    ]:
        row = main_comp[main_comp["场景"] == scenario_name].iloc[0]
        rows.append(
            {
                "方案": scenario_name,
                "方案类型": plan_type,
                "目标": "Q10低于2800；若Q10未达标则采用Q05理论极限" if "方案A" in scenario_name else "Q10达标缺口目标下综合评分最低",
                "CO口径": "Q10/Q05低分位CO",
                "预测CO_mg_m3": float(row["非线性模型预测CO_mg_m3"]),
                "Q10_CO_low_final": float(row["Q10_CO_low_final"]),
                "Q05_CO_low_extreme": float(row["Q05_CO_low_extreme"]),
                "近达标概率Pr_CO_le_3200": float(row["近达标概率Pr_CO_le_3200"]),
                "严格达标概率Pr_CO_le_2800": float(row["严格达标概率Pr_CO_le_2800"]),
                "相对2800": float(row["相对2800"]),
                "标准化调节量": float(row["标准化调节量"]),
                "历史达标IQR距离": float(row["历史达标IQR距离"]),
                "温度/工程约束口径": "宽温度约束；代表低排放潜力，不直接作为工程执行承诺",
                "说明": row["说明"],
            }
        )

    aux_after = aux_comp[aux_comp["场景"] == "辅助模型_小幅平稳优化"].iloc[0]
    strict_temp_ok = bool((aux_decision["第一问温度增量"].abs() <= ENGINEERING_TEMP_DELTA_LIMIT + 1e-8).all())
    rows.append(
        {
            "方案": "方案C_工程平稳方案",
            "方案类型": "工程平稳",
            "目标": "小幅调节、温度可行、相邻平滑",
            "CO口径": "Q10低分位CO/辅助平稳预测",
            "预测CO_mg_m3": float(aux_after["调控模型直接预测CO_mg_m3"]),
            "Q10_CO_low_final": float(aux_after["调控模型直接预测CO_mg_m3"]),
            "Q05_CO_low_extreme": np.nan,
            "近达标概率Pr_CO_le_3200": np.nan,
            "严格达标概率Pr_CO_le_2800": np.nan,
            "相对2800": float(aux_after["相对2800"]),
            "标准化调节量": float(aux_decision["本方案总调节量"].iloc[0]),
            "历史达标IQR距离": np.nan,
            "温度/工程约束口径": f"严格温度增量<={ENGINEERING_TEMP_DELTA_LIMIT:g}，并执行单箱、总调节量、相邻风箱约束",
            "说明": "工程平稳推荐方案；可能仍高于2800，但更接近现场执行",
            "温度严格可行": strict_temp_ok,
            "压力约束最大违规量": float(aux_diagnostics.get("best_constraint_violation", np.nan)),
        }
    )
    summary = pd.DataFrame(rows)
    save_csv(summary, DATA_OUT / "三类调控方案汇总.csv")
    save_csv(summary, RESULT_DIR / "三类调控方案汇总.csv")

    pressure_rows: list[dict[str, object]] = []
    scheme_payloads = {
        "方案A_理论达标优先方案": main_diagnostics.get("scheme_a", {}),
        "方案B_折中低排放方案": main_diagnostics.get("scheme_b", {}),
    }
    for scheme_name, payload in scheme_payloads.items():
        targets = payload.get("目标强度", {})
        raw_targets = payload.get("目标原始负压", {})
        for i in BOX_IDS:
            pressure_rows.append(
                {
                    "方案": scheme_name,
                    "风箱编号": i,
                    "目标强度": float(targets.get(str(i), np.nan)),
                    "目标原始负压": float(raw_targets.get(str(i), np.nan)),
                    "方案定位": payload.get("说明", ""),
                }
            )
    for _, row in aux_decision.iterrows():
        pressure_rows.append(
            {
                "方案": "方案C_工程平稳方案",
                "风箱编号": int(row["风箱编号"]),
                "目标强度": float(row["最优目标强度"]),
                "目标原始负压": float(row["最优目标原始负压"]),
                "控制时刻实际强度": float(row["控制时刻实际强度"]),
                "相对控制时刻实际强度变化": float(row["相对控制时刻实际强度变化"]),
                "调整方向": row["调整方向"],
                "方案定位": "严格工程约束下的小幅平稳推荐",
            }
        )
    pressure = pd.DataFrame(pressure_rows)
    save_csv(pressure, DATA_OUT / "三类调控方案风箱负压组合.csv")
    save_csv(pressure, RESULT_DIR / "三类调控方案风箱负压组合.csv")
    return summary, pressure


def evaluate_plan_over_samples(
    aligned: pd.DataFrame,
    cleaned_by_sequence: pd.DataFrame,
    ranges: pd.DataFrame,
    constraints: dict[str, object],
    q1_linker: Q1TemperatureLinker,
    control_model: LinearCOModel,
    median_start: np.ndarray,
) -> pd.DataFrame:
    eval_source = aligned[aligned["数据集"] == "测试集"].copy()
    if eval_source.empty:
        eval_source = aligned.copy()
    rows = []
    horizon_samples = max(1, CONTROL_EFFECT_HORIZON_S // SAMPLE_INTERVAL_S)
    for _, row in eval_source.iterrows():
        speed = speed_from_row(row)
        control_sequence = int(row["序列"]) - horizon_samples
        p0 = current_intensity_from_cleaned_sequence(cleaned_by_sequence, control_sequence)
        one = eval_source.loc[[row.name]].copy()
        objective, diag = objective_factory(one, control_model, q1_linker, speed, p0)
        starts = [p0]
        best_p, best_delta, _ = optimize_pressure_plan(
            ranges,
            starts,
            objective,
            constraints,
            p0,
            grid_points=3,
            max_rounds=1,
            save_search=False,
            use_slsqp=False,
        )
        scenario_temp, _, _, temp_delta, _, _ = incremental_temperature_for_plan(row, best_p, p0, control_model, q1_linker, speed)
        q2_temp = lagged_temperature_from_row(row, control_model)
        base_frame = apply_plan_to_frame(one, p0, q2_temp, control_model)
        plan_frame = apply_plan_to_frame(one, best_p, scenario_temp, control_model)
        base_output = float(control_model.ridge_predict(base_frame)[0])
        plan_output = float(control_model.ridge_predict(plan_frame)[0])
        model_delta = plan_output - base_output
        base_co = float(row[TARGET_COL])
        rows.append(
            {
                "序列": row["序列"],
                "预测CO目标序列": int(row["序列"]),
                "控制基准序列_约": int(control_sequence),
                "控制作用时滞_s": int(CONTROL_EFFECT_HORIZON_S),
                "时间_s": row["时间_s"],
                "数据集": row["数据集"],
                TARGET_COL: row[TARGET_COL],
                "实际CO": base_co,
                "基准模型预测CO": base_output,
                "优化后模型预测CO": plan_output,
                "调控模型增量": model_delta,
                "模型预测降低量": base_output - plan_output,
                "实际CO是否低于2800": base_co <= EMISSION_STANDARD,
                "基准预测是否低于2800": base_output <= EMISSION_STANDARD,
                "优化后预测是否低于2800": plan_output <= EMISSION_STANDARD,
                "控制时刻实际总强度": float(np.sum(p0)),
                "最优总强度": float(np.sum(best_p)),
                "总调节量": float(np.sum(np.abs(best_p - p0))),
                "平均温度增量": float(np.mean(temp_delta)),
                "目标函数评估次数": int(diag["evaluated_candidates"]),
                **{f"{i}#控制时刻实际强度": float(p0[i - 1]) for i in BOX_IDS},
                **{f"{i}#最优目标强度": float(best_p[i - 1]) for i in BOX_IDS},
            }
        )
    out = pd.DataFrame(rows)
    save_csv(out, DATA_OUT / "调控策略评估_全样本.csv")
    save_csv(out, DATA_OUT / "测试集逐样本滚动优化评估.csv")
    return out


def summarize_eval(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = [name for name in ["训练集", "验证集", "测试集"] if name in set(eval_df["数据集"])]
    for name in order + ["全样本"]:
        part = eval_df if name == "全样本" else eval_df[eval_df["数据集"] == name]
        if part.empty:
            continue
        rows.append(
            {
                "评估样本": name,
                "样本数": int(len(part)),
                "实际CO均值": float(part["实际CO"].mean()),
                "基准预测CO均值": float(part["基准模型预测CO"].mean()),
                "优化后预测CO均值": float(part["优化后模型预测CO"].mean()),
                "平均模型增量": float(part["调控模型增量"].mean()),
                "平均模型预测降低量": float(part["模型预测降低量"].mean()),
                "实际CO低于2800比例_%": float(part["实际CO是否低于2800"].mean() * 100),
                "基准预测低于2800比例_%": float(part["基准预测是否低于2800"].mean() * 100),
                "优化后预测低于2800比例_%": float(part["优化后预测是否低于2800"].mean() * 100),
            }
        )
    summary = pd.DataFrame(rows)
    save_csv(summary, DATA_OUT / "调控策略汇总.csv")
    return summary


def generate_rolling_control_trajectory(
    aligned: pd.DataFrame,
    cleaned_by_sequence: pd.DataFrame,
    ranges: pd.DataFrame,
    constraints: dict[str, object],
    q1_linker: Q1TemperatureLinker,
    control_model: object,
    median_start: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_samples = max(1, ROLLING_CONTROL_WINDOW_S // SAMPLE_INTERVAL_S)
    source = aligned.sort_values("序列").tail(window_samples).copy()
    rows: list[dict[str, object]] = []
    previous_target: np.ndarray | None = None
    horizon_samples = max(1, CONTROL_EFFECT_HORIZON_S // SAMPLE_INTERVAL_S)

    for step, (_, row) in enumerate(source.iterrows(), start=1):
        one = source.loc[[row.name]].copy()
        speed = speed_from_row(row)
        target_sequence = int(row["序列"])
        control_sequence = target_sequence - horizon_samples
        current_actual = current_intensity_from_cleaned_sequence(cleaned_by_sequence, control_sequence)
        previous_reference = previous_target.copy() if previous_target is not None else current_actual.copy()
        objective, diag = objective_factory(one, control_model, q1_linker, speed, current_actual)

        def rolling_objective(intensity: np.ndarray) -> float:
            raw_value = float(objective(intensity))
            smooth_penalty = 0.03 * normalized_adjustment(np.asarray(intensity, dtype=float), previous_reference, *constrained_bounds_and_limits(ranges, constraints, current_actual)[:2]) / len(BOX_IDS)
            return raw_value + smooth_penalty

        starts = [current_actual, previous_reference]
        best_p, best_delta, _ = optimize_pressure_plan(
            ranges,
            starts,
            rolling_objective,
            constraints,
            current_actual,
            grid_points=3,
            max_rounds=1,
            save_search=False,
            use_slsqp=False,
        )
        scenario_temp, _, _, temp_delta, _, _ = incremental_temperature_for_plan(row, best_p, current_actual, control_model, q1_linker, speed)
        q2_temp = lagged_temperature_from_row(row, control_model)
        base_frame = apply_plan_to_frame(one, current_actual, q2_temp, control_model)
        plan_frame = apply_plan_to_frame(one, best_p, scenario_temp, control_model)
        base_pred = float(control_model.ridge_predict(base_frame)[0])
        plan_pred = float(control_model.ridge_predict(plan_frame)[0])
        if hasattr(control_model, "predict_standard_probability"):
            base_prob = float(control_model.predict_standard_probability(base_frame)[0])
            plan_prob = float(control_model.predict_standard_probability(plan_frame)[0])
            base_near_prob = float(control_model.predict_near_probability(base_frame)[0]) if hasattr(control_model, "predict_near_probability") else np.nan
            plan_near_prob = float(control_model.predict_near_probability(plan_frame)[0]) if hasattr(control_model, "predict_near_probability") else np.nan
        else:
            base_prob = float(np.clip((EMISSION_STANDARD + 400 - base_pred) / 800, 0.0, 1.0))
            plan_prob = float(np.clip((EMISSION_STANDARD + 400 - plan_pred) / 800, 0.0, 1.0))
            base_near_prob = float(np.clip((NEAR_STANDARD + 400 - base_pred) / 800, 0.0, 1.0))
            plan_near_prob = float(np.clip((NEAR_STANDARD + 400 - plan_pred) / 800, 0.0, 1.0))
        single_step_adjust = float(np.sum(np.abs(best_p - previous_reference)))
        current_adjust = float(np.sum(np.abs(best_p - current_actual)))
        previous_target = best_p.copy()
        rows.append(
            {
                "滚动步": step,
                "序列": int(row["序列"]),
                "预测CO目标序列": target_sequence,
                "控制基准序列_约": control_sequence,
                "控制作用时滞_s": int(CONTROL_EFFECT_HORIZON_S),
                "时间_s": float(row["时间_s"]),
                "数据集": row.get("数据集", ""),
                "实际CO": float(row[TARGET_COL]),
                "基准模型预测CO": base_pred,
                "滚动优化后预测CO": plan_pred,
                "模型预测降低量": base_pred - plan_pred,
                "调控模型增量": plan_pred - base_pred,
                "基准近达标概率Pr_CO_le_3200": base_near_prob,
                "滚动目标近达标概率Pr_CO_le_3200": plan_near_prob,
                "基准严格达标概率Pr_CO_le_2800": base_prob,
                "滚动目标严格达标概率Pr_CO_le_2800": plan_prob,
                "基准达标倾向概率": base_prob,
                "滚动目标达标倾向概率": plan_prob,
                "相对控制时刻实际总调节量": current_adjust,
                "相对上一目标单步调节量": single_step_adjust,
                "平均温度增量": float(np.mean(temp_delta)),
                "目标函数评估次数": int(diag["evaluated_candidates"]),
                "优化口径": "滚动调控：以控制时刻实际负压为约束基准，以上一时刻目标负压作平滑参考",
                **{f"{i}#控制时刻实际强度": float(current_actual[i - 1]) for i in BOX_IDS},
                **{f"{i}#滚动目标强度": float(best_p[i - 1]) for i in BOX_IDS},
            }
        )

    trajectory = pd.DataFrame(rows)
    if trajectory.empty:
        summary = pd.DataFrame()
    else:
        summary = pd.DataFrame(
            [
                {
                    "滚动窗口样本数": int(len(trajectory)),
                    "滚动窗口持续时间_s": int(len(trajectory) * SAMPLE_INTERVAL_S),
                    "基准预测CO均值": float(trajectory["基准模型预测CO"].mean()),
                    "滚动优化预测CO均值": float(trajectory["滚动优化后预测CO"].mean()),
                    "平均模型预测降低量": float(trajectory["模型预测降低量"].mean()),
                    "最终步基准预测CO": float(trajectory["基准模型预测CO"].iloc[-1]),
                    "最终步滚动优化预测CO": float(trajectory["滚动优化后预测CO"].iloc[-1]),
                    "最终步模型预测降低量": float(trajectory["模型预测降低量"].iloc[-1]),
                    "平均达标倾向概率变化": float((trajectory["滚动目标达标倾向概率"] - trajectory["基准达标倾向概率"]).mean()),
                    "平均单步调节量": float(trajectory["相对上一目标单步调节量"].mean()),
                    "最大单步调节量": float(trajectory["相对上一目标单步调节量"].max()),
                    "平均相对控制时刻实际调节量": float(trajectory["相对控制时刻实际总调节量"].mean()),
                }
            ]
        )
    save_csv(trajectory, DATA_OUT / "最新窗口滚动调控轨迹.csv")
    save_csv(summary, DATA_OUT / "最新窗口滚动调控策略汇总.csv")
    return trajectory, summary


def make_figures(ranges: pd.DataFrame, decision: pd.DataFrame, comp: pd.DataFrame, eval_df: pd.DataFrame, marginal: pd.DataFrame) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140

    x = ranges["风箱编号"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.vlines(x, ranges["优化强度下限_P10"], ranges["优化强度上限_P90"], color="#2563EB", linewidth=5, alpha=0.75, label="辅助范围 P10-P90")
    ax.vlines(x + 0.12, ranges["扩展强度下限_P05"], ranges["扩展强度上限_P95"], color="#94A3B8", linewidth=2, alpha=0.8, label="主模型范围 P05-P95")
    ax.scatter(x, ranges["历史强度均值"], color="#DC2626", s=26, label="历史均值", zorder=3)
    ax.set_xticks(x)
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("负压强度 |KPa|")
    ax.set_title("各风箱负压强度调节范围")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "图3-1_各风箱负压强度调节范围.svg")
    fig.savefig(FIG_DIR / "图3-1_各风箱负压强度可调范围.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    width = 0.36
    ax.bar(x - width / 2, decision["控制时刻实际强度"], width, label="控制时刻实际强度", color="#64748B")
    ax.bar(x + width / 2, decision["最优目标强度"], width, label="主模型最优目标强度", color="#0F766E")
    ax.set_xticks(x)
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("负压强度 |KPa|")
    ax.set_title("主模型全局达标优化负压组合")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "图3-2_调控模型最优负压组合.svg")
    fig.savefig(FIG_DIR / "图3-2_负压温度联动最优调控方案.svg")
    fig.savefig(FIG_DIR / "图3-2_最新状态负压调控决策.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(x, decision["第2问当前滞后温度"], marker="o", label="第2问当前滞后温度", color="#64748B")
    ax.plot(x, decision["增量叠加后温度"], marker="o", label="增量叠加后温度", color="#DC2626")
    ax.set_xticks(x)
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("温度 (℃)")
    ax.set_title("最优负压方案下的温度增量叠加")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "图3-3_最优方案温度联动变化.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    colors = np.where(marginal["P90较P10_CO变化"].to_numpy() < 0, "#0F766E", "#DC2626")
    ax.bar(x, marginal["P90较P10_CO变化"], color=colors)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xticks(x)
    ax.set_xlabel("风箱编号")
    ax.set_ylabel("P90较P10的调控型CO变化 (mg/m^3)")
    ax.set_title("无历史CO调控模型的负压-温度联动边际方向")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "图3-4_风箱调控边际贡献.svg")
    fig.savefig(FIG_DIR / "图3-4_调控模型负压边际方向.svg")
    fig.savefig(FIG_DIR / "图3-3_调控模型负压边际方向.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    plot_comp = comp.copy()
    labels = plot_comp["场景"].str.replace("_", "\n", regex=False).to_list()
    co_col = "非线性模型预测CO_mg_m3" if "非线性模型预测CO_mg_m3" in plot_comp.columns else "调控模型直接预测CO_mg_m3"
    vals = plot_comp[co_col].to_numpy()
    colors = ["#64748B", "#2563EB", "#0F766E"]
    ax.bar(np.arange(len(vals)), vals, color=colors[: len(vals)])
    ax.axhline(EMISSION_STANDARD, color="#DC2626", linestyle="--", linewidth=1.2, label="2800 mg/m^3")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("CO 浓度 (mg/m^3)")
    ax.set_title("主模型非线性低排放评分方案对比")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "图3-4_主模型直接预测CO对比.svg")
    fig.savefig(FIG_DIR / "图3-4_优化前后CO与2800对比.svg")
    plt.close(fig)

    test = eval_df[eval_df["数据集"] == "测试集"].copy()
    if test.empty:
        test = eval_df.copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.boxplot(
        [test["基准模型预测CO"], test["优化后模型预测CO"]],
        labels=["基准预测", "优化后预测"],
        patch_artist=True,
    )
    ax.axhline(EMISSION_STANDARD, color="#DC2626", linestyle="--", linewidth=1.2, label="2800 mg/m^3")
    ax.set_ylabel("CO (mg/m^3)")
    ax.set_title("辅助模型测试集直接预测CO分布")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "图3-5_辅助模型测试集直接预测分布.svg")
    fig.savefig(FIG_DIR / "图3-5_测试集优化前后CO分布.svg")
    fig.savefig(FIG_DIR / "图3-3_调控前后CO预测分布.svg")
    plt.close(fig)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None, float_digits: int = 3) -> str:
    show = df.copy()
    if max_rows is not None:
        show = show.head(max_rows)
    for col in show.select_dtypes(include=[np.number]).columns:
        show[col] = show[col].map(lambda x: f"{x:.{float_digits}f}" if pd.notna(x) else "")
    try:
        return show.to_markdown(index=False)
    except ImportError:
        return show.to_csv(index=False, sep="\t")


def make_schema(control_model: LinearCOModel, q1_linker: Q1TemperatureLinker) -> None:
    text = f"""# 调控决策字段说明

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

负责Agent：第3问子Agent

## 关键口径

- 本问重新训练 `未来300s Q10/Q05达标潜力调控模型`，保留 `CO历史浓度__*` 作为短期状态变量，不再完全删除历史 CO。
- 调控模型的训练数据仍沿用第二问异常修正后建模的 `滞后配准建模数据.csv`，因此风箱负压、风箱温度、机速、大烟道变量的时滞口径与第二问一致。
- 主模型改用未来300s低分位达标潜力评分：输入为当前 CO 状态、当前过程变量和负压调节量，输出为 `Q10_CO(P)`；若 Q10 无法低于 2800，则输出 `Q05_CO(P)` 作为理论低排放极限。
- 低排放预测采用全局/尾部融合：`CO_low_final=0.5*Q10_global+0.5*Q10_tail`；尾部模型只使用 `CO<=3200` 及其邻域样本，并对 `CO<=2800` 加高权重。
- 分类器为两级：`Pr_near=Pr(CO<=3200)` 用于优化引导，`Pr_strict=Pr(CO<=2800)` 用于严格达标诊断。
- 温度采用增量叠加口径：先用第一问当前温度模型 `{q1_linker.source_note}` 分别预测基准负压和方案负压下的温度，取 `ΔT`，再令 `T_scenario = T_第二问当前配准温度 + ΔT`。
- 第三问负压基准使用控制时刻实际负压强度 `P_i^0`，不是预测目标时刻负压，也不是第二问中的滞后负压强度；第二问滞后强度只用于说明配准输入来源。
- 代码输出三类方案：方案A为理论达标优先，方案B为折中低排放，方案C为工程平稳；方案C保留单箱调节上限、总调节预算、相邻风箱差值和严格温度增量约束。
- 历史达标样本重定义为第二问修正CO<=2800且非修正异常点、非孤立低谷；这些样本单独统计风箱负压分布，并通过样本加权、达标倾向分类器和历史达标 IQR 距离进入主模型。
- 低排放工况先按 CO较低、连续不少于30s、温度/大烟道状态稳定识别，再提取风箱负压、机速、大烟道负压/温度和后段风箱温度区间。
- 历史达标样本同时做连续性、机速、大烟道负压/温度、时间分布和变量组重要性诊断；若达标样本对应整体工况切换，则不能只套用达标样本的风箱负压中位数模板。
- 本问取消使用完整监测型模型作为调控复核指标；非线性模型输出只作为调控评分，不作为现场达标承诺。
- 原始负压为负值，模型输入为 `负压强度=abs(原始负压)`；目标原始负压为 `-最优目标强度`。
- 主模型静态评分范围为各风箱历史负压强度 P05-P95；最终调控输出采用最新窗口滚动轨迹，辅助平稳调控模型使用 P10-P90 与当前附近小幅约束。

## 模型规模

- 无历史 CO 调控模型特征数：{len(control_model.feature_names)}
- 第一问温度联动模型特征数：{len(q1_linker.feature_names)}
- 第一问温度联动模型训练样本数：{q1_linker.fit_rows}

## 时滞解释

第1问温度模型使用机速、风箱负压及其滞后/滚动派生特征；第2问 CO 模型在最新口径下使用最长 300s 的风箱过程变量空间时滞。由于两者绝对温度口径并不完全等价，本问只使用第一问给出的温度变化量，不直接用第一问绝对温度替换第二问温度特征。
"""
    (RESULT_DIR / "调控决策字段说明.md").write_text(text, encoding="utf-8")


def make_report(
    main_decision: pd.DataFrame,
    main_comp: pd.DataFrame,
    aux_decision: pd.DataFrame,
    aux_comp: pd.DataFrame,
    summary: pd.DataFrame,
    historical_stats: pd.DataFrame,
    historical_summary: pd.DataFrame,
    compliant_condition_diagnostics: dict[str, pd.DataFrame],
    low_emission: dict[str, pd.DataFrame],
    regime_profiles: pd.DataFrame,
    trusted_audit_summary: pd.DataFrame,
    trusted_audit: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    importance_group_summary: pd.DataFrame,
    control_metrics: pd.DataFrame,
    control_model: LinearCOModel,
    q1_linker: Q1TemperatureLinker,
    main_diagnostics: dict[str, object],
    aux_diagnostics: dict[str, object],
) -> None:
    main_final = main_comp[main_comp["场景"] == "方案A_理论达标优先方案"].iloc[0]
    main_min = main_comp[main_comp["场景"] == "方案B_折中低排放方案"].iloc[0]
    main_base = main_comp[main_comp["场景"] == "控制时刻基准工况"].iloc[0]
    aux_after = aux_comp[aux_comp["场景"] == "辅助模型_小幅平稳优化"].iloc[0]
    test_summary = summary[summary["评估样本"] == "测试集"].iloc[0] if "测试集" in set(summary["评估样本"]) else summary.iloc[-1]
    test_metric = control_metrics[control_metrics["数据集"] == "测试集"].iloc[0]
    hist_row = historical_summary.iloc[0]
    model_type = getattr(control_model, "extra", {}).get("模型类型", "非线性调控模型")
    model_weight_note = getattr(control_model, "extra", {}).get("说明", "")
    compliant_diag_row = compliant_condition_diagnostics["summary"].iloc[0]
    compliant_runs = compliant_condition_diagnostics["runs"].copy()
    compliant_compare = compliant_condition_diagnostics["comparison"].copy()
    compliant_time = compliant_condition_diagnostics["time_distribution"].copy()
    low_summary = low_emission.get("summary", pd.DataFrame())
    low_intervals = low_emission.get("intervals", pd.DataFrame()).copy()
    low_key_ranges = low_emission.get("key_ranges", pd.DataFrame()).copy()
    low_pressure = low_emission.get("pressure_profile", pd.DataFrame()).copy()
    low_summary_row = low_summary.iloc[0] if not low_summary.empty else pd.Series(
        {"低排放片段数": 0, "低排放CO阈值": np.nan, "低排放片段样本数": 0, "最长低排放片段持续时间_s": 0}
    )
    low_interval_view = low_intervals.head(8) if not low_intervals.empty else low_intervals
    low_key_view = low_key_ranges[low_key_ranges["片段编号"] == "全部低排放片段"] if not low_key_ranges.empty and "片段编号" in low_key_ranges.columns else low_key_ranges
    low_pressure_view = low_pressure[low_pressure["片段编号"] == "全部低排放片段"] if not low_pressure.empty and "片段编号" in low_pressure.columns else low_pressure
    low_pressure_view = low_pressure_view[["变量", "P10", "P25", "中位数", "P75", "P90", "典型原始负压中位数"]] if not low_pressure_view.empty else low_pressure_view
    regime_view = regime_profiles[
        [
            "工况层",
            "工况层名称",
            "样本数",
            "可信达标样本数",
            "可信达标比例_%",
            "CO中位数",
            "烧结机机速中位数",
            "1#大烟道负压强度中位数",
            "2#大烟道负压强度中位数",
            "1#大烟道温度中位数",
            "2#大烟道温度中位数",
            "后段风箱温度中位数",
        ]
    ] if not regime_profiles.empty else regime_profiles
    rolling_row = rolling_summary.iloc[0] if not rolling_summary.empty else pd.Series(
        {
            "滚动窗口样本数": 0,
            "滚动窗口持续时间_s": 0,
            "基准预测CO均值": np.nan,
            "滚动优化预测CO均值": np.nan,
            "平均模型预测降低量": np.nan,
            "最终步模型预测降低量": np.nan,
            "平均单步调节量": np.nan,
            "最大单步调节量": np.nan,
        }
    )
    audit_row = trusted_audit_summary.iloc[0] if not trusted_audit_summary.empty else pd.Series(
        {
            "可信达标样本数": 0,
            "预测低于2800样本数": 0,
            "预测3000以上样本数": 0,
            "预测低于2800比例_%": np.nan,
            "预测3000以上比例_%": np.nan,
            "平均真实CO": np.nan,
            "平均预测CO": np.nan,
            "平均达标概率": np.nan,
            "达标概率是否失效": True,
            "诊断结论": "未生成可信达标样本复核。",
        }
    )
    audit_view_cols = [
        "序列",
        "真实修正CO",
        "Q10_CO_low_final",
        "Q05_CO_low_extreme",
        "近达标概率Pr_CO_le_3200",
        "严格达标概率Pr_CO_le_2800",
        "预测误差_预测减真实",
        "预测是否低于2800",
        "Q05预测是否低于2800",
        "预测是否3000以上",
        "所属工况层名称",
    ]
    audit_view = trusted_audit[[c for c in audit_view_cols if c in trusted_audit.columns]].head(12) if not trusted_audit.empty else trusted_audit
    run_view_cols = [
        "连续片段编号",
        "起始序列",
        "结束序列",
        "样本数",
        "持续时间_s",
        "CO中位数",
        "机速中位数",
        "1#大烟道负压强度中位数",
        "2#大烟道负压强度中位数",
        "1#大烟道温度中位数",
        "2#大烟道温度中位数",
    ]
    run_view = compliant_runs[[c for c in run_view_cols if c in compliant_runs.columns]].head(8) if not compliant_runs.empty else compliant_runs
    compare_view_cols = [
        "变量",
        "达标中位数",
        "未达标中位数",
        "差值_达标减未达标",
        "达标P25",
        "达标P75",
        "未达标P25",
        "未达标P75",
        "标准化均值差",
    ]
    compare_view = compliant_compare[[c for c in compare_view_cols if c in compliant_compare.columns]]
    time_view = compliant_time[
        ["时间段编号", "起始序列", "结束序列", "样本数", "达标样本数", "达标比例_%", "占全部达标样本_%", "CO中位数"]
    ]
    importance_view = importance_group_summary[
        ["变量组", "特征数", "正向特征数", "RMSE增量合计", "正向RMSE增量合计", "正向重要性占比_%"]
    ]

    compare_lookup = compliant_compare.set_index("变量") if not compliant_compare.empty else pd.DataFrame()

    def diff_of(var_name: str) -> float:
        if compare_lookup.empty or var_name not in compare_lookup.index:
            return np.nan
        return float(compare_lookup.loc[var_name, "差值_达标减未达标"])

    def metric_of(var_name: str, metric_name: str) -> float:
        if compare_lookup.empty or var_name not in compare_lookup.index or metric_name not in compare_lookup.columns:
            return np.nan
        return float(compare_lookup.loc[var_name, metric_name])

    imp_lookup = importance_group_summary.set_index("变量组") if not importance_group_summary.empty else pd.DataFrame()

    def share_of(group_name: str) -> float:
        if imp_lookup.empty or group_name not in imp_lookup.index:
            return 0.0
        value = float(imp_lookup.loc[group_name, "正向重要性占比_%"])
        return value if np.isfinite(value) else 0.0

    pressure_share = share_of("风箱负压")
    flue_temp_share = share_of("大烟道温度")
    thermal_share = share_of("大烟道温度") + share_of("风箱温度")
    late_time_count = int(time_view.loc[time_view["时间段编号"].isin([9, 10]), "达标样本数"].sum()) if not time_view.empty else 0
    late_time_share = (
        late_time_count / float(compliant_diag_row["达标样本数"]) * 100 if float(compliant_diag_row["达标样本数"]) > 0 else np.nan
    )
    fifth_time_count = int(time_view.loc[time_view["时间段编号"] == 5, "达标样本数"].sum()) if not time_view.empty else 0
    fifth_time_share = (
        fifth_time_count / float(compliant_diag_row["达标样本数"]) * 100 if float(compliant_diag_row["达标样本数"]) > 0 else np.nan
    )

    main_view = main_decision[
        [
            "风箱编号",
            "主模型允许强度下限_P05",
            "主模型允许强度上限_P95",
            "历史达标强度中位数",
            "控制时刻实际强度",
            "最优目标强度",
            "最优目标原始负压",
            "调整方向",
            "第一问温度增量",
            "增量叠加后温度",
        ]
    ]
    aux_view = aux_decision[
        [
            "风箱编号",
            "约束后强度下限",
            "约束后强度上限",
            "控制时刻实际强度",
            "最优目标强度",
            "最优目标原始负压",
            "调整方向",
        ]
    ]
    hist_view = historical_stats[["风箱编号", "达标强度P25", "达标强度中位数", "达标强度P75", "控制时刻实际强度", "达标中位数较控制时刻变化"]]

    files = [
        ("线性无历史CO调控模型", "result/第3问/CO调控模型_无历史CO.json"),
        ("非线性低排放调控模型", "result/第3问/CO调控模型_非线性低排放.json"),
        ("非线性模型评价", "data/第3问/结果数据/非线性低排放调控模型评价指标.csv"),
        ("历史达标工况负压统计", "data/第3问/结果数据/历史达标工况风箱负压统计.csv"),
        ("历史达标连续性诊断", "data/第3问/结果数据/历史达标工况连续性诊断.csv"),
        ("历史达标连续片段明细", "data/第3问/结果数据/历史达标工况连续片段明细.csv"),
        ("历史达标关键变量对比", "data/第3问/结果数据/历史达标工况关键变量对比.csv"),
        ("历史达标时间分布", "data/第3问/结果数据/历史达标工况时间分布.csv"),
        ("第二问修正CO可信达标明细", "data/第3问/结果数据/第二问修正CO可信达标样本明细.csv"),
        ("低排放工况区间", "data/第3问/结果数据/低排放工况区间识别.csv"),
        ("低排放片段风箱负压分布", "data/第3问/结果数据/低排放片段风箱负压典型分布.csv"),
        ("低排放片段关键工况区间", "data/第3问/结果数据/低排放片段关键工况区间.csv"),
        ("工况分层画像", "data/第3问/结果数据/工况分层画像.csv"),
        ("未来300s建模数据划分摘要", "data/第3问/结果数据/未来300s建模数据划分摘要.csv"),
        ("未来300s低排放潜力模型评价", "data/第3问/结果数据/未来300s低排放潜力模型评价指标.csv"),
        ("未来300s低排放潜力模型", "result/第3问/CO调控模型_未来300s低排放潜力.json"),
        ("可信达标样本模型复核明细", "data/第3问/结果数据/可信达标样本_调控模型预测复核.csv"),
        ("可信达标样本模型复核摘要", "data/第3问/结果数据/可信达标样本_调控模型预测复核摘要.csv"),
        ("分层调控模型", "result/第3问/CO调控模型_分层低排放.json"),
        ("最新窗口滚动调控轨迹", "data/第3问/结果数据/最新窗口滚动调控轨迹.csv"),
        ("最新窗口滚动调控汇总", "data/第3问/结果数据/最新窗口滚动调控策略汇总.csv"),
        ("非线性模型变量组重要性", "data/第3问/结果数据/非线性低排放模型变量组重要性.csv"),
        ("主模型非线性低排放负压组合", "data/第3问/结果数据/主模型_非线性低排放负压组合.csv"),
        ("主模型非线性低排放评分对比", "data/第3问/结果数据/主模型_非线性低排放评分对比.csv"),
        ("主模型搜索记录", "data/第3问/中间数据/主模型_非线性低排放优化搜索记录.csv"),
        ("低排放优化初始种子", "data/第3问/结果数据/低排放优化初始种子.csv"),
        ("三类调控方案汇总", "data/第3问/结果数据/三类调控方案汇总.csv"),
        ("三类调控方案风箱负压组合", "data/第3问/结果数据/三类调控方案风箱负压组合.csv"),
        ("辅助模型平稳调控方案", "data/第3问/结果数据/风箱负压调控决策_最新状态.csv"),
        ("辅助模型CO直接预测", "data/第3问/结果数据/辅助模型_当前工况平稳调控CO直接预测.csv"),
        ("辅助模型测试集评估", "data/第3问/结果数据/测试集逐样本滚动优化评估.csv"),
        ("评估汇总", "data/第3问/结果数据/调控策略汇总.csv"),
        ("代码", "code/第3问/question3_control_optimization.py"),
        ("字段说明", "result/第3问/调控决策字段说明.md"),
    ]
    file_lines = "\n".join([f"- `{path}`：{desc}" for desc, path in files])

    report = f"""# 第3问

## 1. 本问任务理解

第3问的目标不是继续预测当前 CO，也不是用普通均值回归直接承诺达标，而是构造“当前状态 -> 未来 300s 低分位 CO”的短期调控模型。主优化量改为 `CO_low(P)=Q10_CO(P)`；若 Q10 仍无法压到 2800，则用 `Q05_CO(P)` 作为理论低排放极限方案。考虑到达标样本集中在尾部，本版采用全局低分位模型、低排放尾部局部模型和两级达标概率诊断。

- 主模型：未来 300s `Q10/Q05` 达标潜力模型。全局模型用全部样本训练，尾部模型只用 `CO<=3200` 及其达标模板邻域样本训练，并对 `CO<=2800` 加高权重；最终 `Q10` 低排放预测采用 `0.5*Q10_global + 0.5*Q10_tail`。
- 两级分类器：`Pr_near=Pr(CO<=3200)` 用来引导优化先进入低排放区；`Pr_strict=Pr(CO<=2800)` 只作为严格达标诊断。
- 辅助模型：工程平稳推荐方案。保留单箱调节上限、总调节预算、相邻风箱差值和严格温度增量筛选，用来给出更接近现场可执行的小幅调节方案。

本问不再用 `当前 CO + ΔCO` 作为最终判断。模型输出只作为优化评分的一部分，方案是否真实达标需结合现场复核：

```text
gap(P) = max(0, (CO_low(P)-2800)/500)
Score(P)=1000*gap(P)^2 + CO_low(P)/500
       + w_temp*temperature_violation(P)
       + w_smooth*smooth(P)
       + w_adjust*adjustment(P)
       + w_template*distance_to_target_IQR(P)
Classifier(P) = [Pr(CO<=3200), Pr(CO<=2800)]
```

## 2. 可信达标样本

原始历史样本共有 `{int(hist_row["全样本数"])}` 条，其中第二问修正后 CO 不高于 2800 的样本 `{int(hist_row.get("修正后CO小于等于2800样本数", hist_row["历史达标样本数"]))}` 条；剔除第二问已修正异常点、局部尖峰/跌落点和孤立低谷后，可信达标样本为 `{int(hist_row["历史达标样本数"])}` 条，占 `{hist_row["历史达标比例_%"]:.2f}%`。这些样本不再只是参考：训练时可信达标样本权重提高，另训练达标倾向分类器；优化时还加入“偏离历史达标工况 P25-P75 区间”的惩罚。各风箱可信达标工况负压分布如下：

{markdown_table(hist_view, float_digits=3)}

## 3. 稳定低排放工况区间

低排放工况不是简单的单点 CO 较低，而是同时满足“修正 CO 较低、持续时间较长、温度/大烟道状态稳定”。本问用修正 CO 的低分位阈值与 2800 取较高者作为低 CO 阈值，本次阈值为 `{low_summary_row["低排放CO阈值"]:.3f}` mg/m^3；再用大烟道负压、大烟道温度和后段风箱温度的 30 s 滚动波动筛选稳定点，最后保留连续不少于 `{LOW_EMISSION_MIN_DURATION_S}` s 的片段。最终识别低排放片段 `{int(low_summary_row["低排放片段数"])}` 个，覆盖 `{int(low_summary_row["低排放片段样本数"])}` 条样本，最长持续 `{int(low_summary_row["最长低排放片段持续时间_s"])}` s。该模板只用于解释“较低 CO 且稳定”的工况背景，不作为达标优化目标；达标优化的模板只来自可信 CO<=2800 样本。

低排放片段如下：

{markdown_table(low_interval_view, float_digits=3)}

从低排放片段中提取的关键工况区间如下：

{markdown_table(low_key_view, float_digits=3)}

18 个风箱负压的典型分布如下：

{markdown_table(low_pressure_view, float_digits=3)}

## 4. 达标样本稳定性与整体工况诊断

仅统计达标样本的风箱负压中位数是不够的，因为这 `{int(hist_row["历史达标样本数"])}` 条样本可能不是负压单变量作用，而是机速、主抽风、大烟道温度和后段热状态共同变化后的结果。为此，本文增加连续性、关键变量、时间分布和变量组重要性四类检查。

从连续性看，达标样本被拆成 `{int(compliant_diag_row["连续达标片段数"])}` 个连续片段，其中单点片段 `{int(compliant_diag_row["单点片段数"])}` 个；片段样本数中位数为 `{compliant_diag_row["片段样本数中位数"]:.1f}`，最长连续达标只有 `{int(compliant_diag_row["最长连续样本数"])}` 条，即 `{int(compliant_diag_row["最长连续持续时间_s"])}` s，没有出现持续不少于 60 s 的稳定达标段。因此这些样本更像短时达标片段或阶段性工况切换信号，不能直接等同于可长期保持的稳定达标窗口。

达标样本连续片段如下：

{markdown_table(run_view, float_digits=3)}

达标与未达标关键工况变量对比如下。机速中位数同为 `{metric_of(SPEED_COL, "达标中位数"):.3f}`，但达标样本机速 P25 为 `{metric_of(SPEED_COL, "达标P25"):.3f}`，低于未达标样本 P25 `{metric_of(SPEED_COL, "未达标P25"):.3f}`，说明部分达标片段伴随偏低机速。注意负压强度越小表示原始负压越接近 0；本批达标样本的大烟道负压强度和风箱平均负压强度整体更低，而 1#、2#大烟道温度中位数分别高 `{diff_of("1#大烟道温度"):.3f}` ℃、`{diff_of("2#大烟道温度"):.3f}` ℃，说明达标样本不是“主抽风更强 + 大烟道温度更低”的单一模式。

{markdown_table(compare_view, float_digits=3)}

达标样本也有明显时间集中性。第 9-10 个时间段合计贡献 `{late_time_count}` 条达标样本，占全部达标样本约 `{late_time_share:.2f}%`；第 5 个时间段还有 `{fifth_time_count}` 条，占约 `{fifth_time_share:.2f}%`。这说明达标样本并非均匀散布在全历史中，而是集中在少数阶段。

{markdown_table(time_view, float_digits=3)}

从模型关联看，非线性调控模型的正向置换重要性中，风箱负压组占 `{pressure_share:.2f}%`，热状态组（大烟道温度 + 风箱温度）合计约 `{thermal_share:.2f}%`，其中大烟道温度组单独占 `{flue_temp_share:.2f}%`。这只能说明相关性，不能当作因果证明；但足以说明现有数据不支持“CO 降低主要由风箱负压单变量造成”的解释，至少需要把大烟道温度、主抽风状态和机速作为联合工况条件。也因此，主模型即使靠近历史达标负压 IQR，仍只得到 `{main_final["非线性模型预测CO_mg_m3"]:.3f}` mg/m^3、达标倾向概率 `{main_final["达标倾向概率"]:.3f}`，不能承诺达标。

{markdown_table(importance_view, float_digits=3)}

## 5. 分层调控响应模型

调控模型重新构造未来预测数据集：每一行用控制时刻 `t` 的 CO 状态、过程变量和负压调节量，预测 `t+300s` 的 CO。历史 CO 不再删除，而是作为短期状态变量进入模型；训练目标从普通均值 CO 改为低分位 CO。全局模型使用全部样本训练 `Q10/Q05`，保证整体稳定；尾部模型只使用 `CO<=3200` 及其达标模板邻域样本训练，并对 `CO<=2800` 加高权重。最终低排放预测为：

```text
X(t) = CO状态(t) + 过程变量(t) + 负压调节量(t)
CO_low_final = 0.5 * Q10_global + 0.5 * Q10_tail
CO_low_extreme = 0.5 * Q05_global + 0.5 * Q05_tail
Pr_near = Pr(CO(t+300)<=3200 | X(t))
Pr_strict = Pr(CO(t+300)<=2800 | X(t))
```

本次主模型为 `{model_type}`，特征数为 `{len(control_model.feature_names)}`。测试集 Q10 口径 RMSE 为 `{test_metric["RMSE"]:.3f}` mg/m^3，R2 为 `{test_metric["R2"]:.3f}`；严格达标样本在测试集中占 `{test_metric.get("真实严格达标比例_%", test_metric.get("真实可信达标比例_%", np.nan)):.2f}%`。这说明模型仍不足以直接替代现场实测判定，因此本文只把它作为低排放潜力评分模型。{model_weight_note}

工况分层画像如下：

{markdown_table(regime_view, float_digits=3)}

为了检查“模型是否把真实达标区域抹平”，本文单独复核可信达标样本：真实 CO、`Q10/Q05` 低分位 CO 和两级分类概率逐条输出。结果显示，可信达标样本中 Q10 低于 2800 的比例为 `{audit_row["预测低于2800比例_%"]:.2f}%`，Q10 预测为 3000 以上的比例为 `{audit_row["预测3000以上比例_%"]:.2f}%`，平均真实 CO 为 `{audit_row["平均真实CO"]:.3f}`，平均 Q10 为 `{audit_row["平均预测CO"]:.3f}`，平均 `Pr_near` 为 `{audit_row.get("平均近达标概率", np.nan):.6f}`，平均 `Pr_strict` 为 `{audit_row["平均达标概率"]:.6f}`。诊断结论：{audit_row["诊断结论"]}

可信达标样本模型复核表前 12 条如下：

{markdown_table(audit_view, float_digits=3)}

温度联动采用第一问当前温度模型 `{q1_linker.source_note}` 的增量口径：

```text
ΔT = T_q1(方案负压) - T_q1(控制时刻实际负压P0)
T_scenario = T_q2_current + ΔT
```

## 6. 主模型：分层低排放优化与滚动调控

主模型的决策变量为 18 个风箱负压强度 `P=(P1,...,P18)`。静态评分部分使用 P05-P95 范围寻找低排放倾向组合，但最终执行建议改为滚动调控：在最新 `{int(rolling_row["滚动窗口持续时间_s"])}` s 窗口内逐点求解，后一时刻以上一时刻目标负压作为调节参考，避免一次性跳到静态组合。滚动窗口取 `{ROLLING_CONTROL_WINDOW_S}` s，是为了匹配第二问风箱负压/温度接近 300 s 的主导时滞，而不是用 30 s 短窗即时评估。

第二问配准方向是“过去风箱负压/温度 -> 当前 CO”。第三问调控方向必须解释为“当前调风箱负压 -> 约 `{CONTROL_EFFECT_HORIZON_S}` s 后 CO”。因此滚动回测中，每个目标 CO 时刻都回推约 `{CONTROL_EFFECT_HORIZON_S}` s 作为控制时刻，把该控制时刻的负压作为基准，再把调控方案写入未来 CO 模型所需的滞后风箱特征。

本版不再让严格达标概率全程为 0 后直接退出，而是拆成两级分类器：`Pr_near` 先引导方案进入 `CO<=3200` 的低排放区，`Pr_strict` 只用于诊断是否达到 `CO<=2800`。主目标函数按“达标缺口优先”排序，CO 缺口的平方惩罚远大于温度、平滑、调节量和模板距离项：

```text
gap(P)=max(0, (CO_low(P)-2800)/500)
Score(P)=1000*gap(P)^2
          + CO_low(P)/500
          + w_temp*temperature_violation(P)
          + w_smooth*smooth(P)
          + w_adjust*adjustment(P,P0)
          + w_template*distance(P, 可信达标模板IQR)
```

本次主模型状态为 `{main_diagnostics["status"]}`。控制时刻基准工况下 Q10 预测未来 CO 为 `{main_base["Q10_CO_low_final"]:.3f}` mg/m^3，`Pr_near` 为 `{main_base["近达标概率Pr_CO_le_3200"]:.3f}`、`Pr_strict` 为 `{main_base["严格达标概率Pr_CO_le_2800"]:.3f}`；方案 A 的目标低分位 CO 为 `{main_final["非线性模型预测CO_mg_m3"]:.3f}` mg/m^3，Q10 为 `{main_final["Q10_CO_low_final"]:.3f}`，Q05 为 `{main_final["Q05_CO_low_extreme"]:.3f}`，历史达标 IQR 距离为 `{main_final["历史达标IQR距离"]:.3f}`。这些数值只表示模型评分下的低排放倾向，不作为现场达标承诺。

主模型对比如下：

{markdown_table(main_comp[["场景", "模型角色", "实际观测CO_mg_m3", "非线性模型预测CO_mg_m3", "Q10_CO_low_final", "Q05_CO_low_extreme", "近达标概率Pr_CO_le_3200", "严格达标概率Pr_CO_le_2800", "相对2800", "模型是否低于2800_仅参考", "较当前模型预测降低量", "标准化调节量", "历史达标IQR距离", "说明"]], float_digits=3)}

最终输出三类方案，而不是只给单一推荐：

- 方案 A：理论达标优先方案。目标是 `Q10` 预测低于 2800；若 Q10 未达标，则采用 `Q05` 作为理论低排放极限。当前方案 A 的目标低分位 CO 为 `{main_final["非线性模型预测CO_mg_m3"]:.3f}` mg/m^3。
- 方案 B：折中低排放方案。目标是 Q10 达标缺口目标下综合评分最低，兼顾温度、平滑、调节量和历史达标模板距离。当前方案 B 的 Q10 为 `{main_min["Q10_CO_low_final"]:.3f}` mg/m^3。
- 方案 C：工程平稳方案。目标是小幅调节、温度可行和相邻平滑，严格执行工程约束；它可能仍高于 2800，但更接近现场执行。当前方案 C 的预测 CO 为 `{aux_after["调控模型直接预测CO_mg_m3"]:.3f}` mg/m^3。

主模型推荐负压组合如下：

{markdown_table(main_view, float_digits=3)}

最新窗口滚动调控摘要如下。滚动窗口 `{int(rolling_row["滚动窗口样本数"])}` 个样本，基准预测 CO 均值为 `{rolling_row["基准预测CO均值"]:.3f}` mg/m^3，滚动优化后均值为 `{rolling_row["滚动优化预测CO均值"]:.3f}` mg/m^3，平均模型预测降低 `{rolling_row["平均模型预测降低量"]:.3f}` mg/m^3；平均单步调节量为 `{rolling_row["平均单步调节量"]:.3f}`，最大单步调节量为 `{rolling_row["最大单步调节量"]:.3f}`。

{markdown_table(rolling_summary, float_digits=3)}

## 7. 辅助模型：当前工况平稳调控

辅助模型保留工程平稳约束：

```text
|P_i - P_i^0| <= Delta_i
|P_i - P_{{i-1}}| <= d_i
sum_i |P_i - P_i^0| <= B
|ΔT_i| <= {ENGINEERING_TEMP_DELTA_LIMIT:g}
```

其中 `Delta_i` 按历史 P10-P90 宽度的 35% 估计，并限制在 `[0.10, 1.20]`；`d_i` 取相邻风箱历史强度差 P90；`B=0.45*sum(Delta_i)`。温度约束在方案 A/B 中只作为宽约束惩罚，不提前压死低排放搜索；在方案 C 中作为严格筛选条件。

辅助模型下，当前工况小幅平稳优化后的直接预测 CO 为 `{aux_after["调控模型直接预测CO_mg_m3"]:.3f}` mg/m^3，模型预测降低 `{aux_after["模型预测降低量"]:.3f}` mg/m^3。这说明如果只允许当前附近小幅平稳调整，降 CO 空间有限。

{markdown_table(aux_comp[["场景", "口径", "实际观测CO_mg_m3", "调控模型直接预测CO_mg_m3", "模型增量_mg_m3", "相对2800", "是否达标", "模型预测降低量"]], float_digits=3)}

辅助模型的平稳调控组合如下：

{markdown_table(aux_view, float_digits=3)}

测试集逐样本平稳调控评估只用于说明辅助模型方向稳定性。测试集基准预测 CO 均值为 `{test_summary["基准预测CO均值"]:.3f}` mg/m^3，平稳优化后预测 CO 均值为 `{test_summary["优化后预测CO均值"]:.3f}` mg/m^3，平均模型预测降低 `{test_summary["平均模型预测降低量"]:.3f}` mg/m^3。

## 8. 基础可视化

| 图编号 | 文件路径 | 图的含义 |
|---|---|---|
| 图3-1 | `figure/第3问/基础图/图3-1_各风箱负压强度调节范围.svg` | P10-P90 与 P05-P95 允许范围参考 |
| 图3-2 | `figure/第3问/基础图/图3-2_调控模型最优负压组合.svg` | 控制时刻实际强度与主模型目标强度对比 |
| 图3-3 | `figure/第3问/基础图/图3-3_最优方案温度联动变化.svg` | 主模型方案温度增量叠加 |
| 图3-4 | `figure/第3问/基础图/图3-4_主模型直接预测CO对比.svg` | 当前、CO 最小方案和最终推荐方案的非线性模型预测 CO 对比 |
| 图3-5 | `figure/第3问/基础图/图3-5_辅助模型测试集直接预测分布.svg` | 辅助模型测试集直接预测 CO 分布 |

## 9. 本问生成的文件清单

{file_lines}

## 10. 补充说明

- 主模型保留当前 CO 和历史 CO 作为短期状态变量，但不再用 `当前CO + ΔCO` 判断是否达标；模型直接学习 `CO(t+300s)` 的 `Q10/Q05` 低分位达标潜力。
- 历史达标样本改为第二问修正 CO≤2800 且非异常、非孤立低谷；可信达标样本只有 `{int(hist_row["历史达标样本数"])}` 条且连续性不足，最长连续达标 `{int(compliant_diag_row["最长连续持续时间_s"])}` s；达标样本还集中在少数时间段，不能把达标样本风箱负压中位数当成充分调控模板。
- 低排放工况先按“CO较低 + 至少{LOW_EMISSION_MIN_DURATION_S}s连续 + 大烟道/后段温度状态稳定”识别，再提取 18 个风箱负压、机速、大烟道负压/温度和后段风箱温度区间；它只作为辅助参考，不作为达标优化目标。
- 调控响应模型已改为全局低分位模型 + 低排放尾部局部模型；`CO_low_final=0.5*Q10_global+0.5*Q10_tail`。
- 可信达标样本已逐条复核模型预测；达标概率分类器拆成 `Pr_near=Pr(CO<=3200)` 和 `Pr_strict=Pr(CO<=2800)`，其中 `Pr_near` 用于低排放区引导，`Pr_strict` 用于达标诊断。
- 优化输出已由单次静态组合扩展为最新 `{ROLLING_CONTROL_WINDOW_S}` s 窗口滚动调控轨迹，并按 `{CONTROL_EFFECT_HORIZON_S}` s 控制作用时滞回推控制时刻。
- 当前附近调节、单箱调节上限、总调节预算、相邻风箱差值和严格温度增量约束只放在方案 C 工程平稳模型中；方案 A/B 使用宽温度约束搜索低排放潜力。
- 第一问温度只提供增量 `ΔT`，避免第一问绝对温度与第二问配准温度时间口径不一致造成基准漂移。
- 主模型范围为 P05-P95，状态为 `{main_diagnostics["status"]}`；主模型唯一方案评估次数为 `{main_diagnostics.get("unique_plan_evaluations", 0)}`。

## 更新记录

- {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}，第3问子Agent：按用户复核意见修正，达标模板仅来自可信 CO<=2800 样本，低排放模板降为辅助参考，分层由 KMeans 改为可信达标模板邻域分层，并新增可信达标样本模型预测复核与 300s 时滞滚动调控。
"""
    (RESULT_DIR / "第3问.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    raw = read_original_excel()
    cleaned = clean_raw_data(raw)
    cleaned = apply_q2_corrected_co(cleaned)
    cleaned, low_emission = identify_low_emission_conditions(cleaned)
    ranges = make_pressure_ranges(cleaned)
    constraints = make_adjustment_constraints(ranges, cleaned)
    cleaned_by_sequence = cleaned.set_index("序列", drop=False)
    deps = load_dependencies()

    aligned = pd.read_csv(deps["第2问滞后配准建模数据"], encoding="utf-8-sig")
    aligned = apply_q2_corrected_co_to_aligned(aligned, cleaned)
    future_aligned = make_future_prediction_dataset(cleaned)
    latest_q2 = future_aligned[future_aligned["序列"] == future_aligned["序列"].max()].copy()
    if latest_q2.empty:
        raise ValueError("未来300s建模数据中未找到最新样本")

    monitor_model = LinearCOModel.from_json(deps["第2问监测型模型"])
    control_model, control_metrics, control_importance = fit_future_tail_control_model(future_aligned)
    trusted_audit, trusted_audit_summary, trusted_audit_flags = audit_trusted_standard_predictions(future_aligned, control_model)
    q1_linker = Q1TemperatureLinker.fit_from_q1_outputs(deps)

    latest_clean = cleaned.loc[cleaned["序列"].idxmax()]
    latest_speed = speed_from_row(latest_q2.iloc[0])
    latest_actual = current_intensity_from_cleaned_sequence(cleaned_by_sequence, int(latest_q2.iloc[0].get("控制序列", cleaned["序列"].max())))
    historical_stats, historical_summary, historical_profile = make_historical_compliant_profile(cleaned, latest_actual)
    compliant_condition_diagnostics = make_compliant_condition_diagnostics(cleaned)
    importance_group_summary = make_importance_group_summary(control_importance)
    low_seed_points, low_seed_table = low_emission_seed_points(cleaned, ranges, latest_actual, historical_stats)

    evaluate_main_plan, main_pred_diag = scenario_evaluator_factory(latest_q2, control_model, q1_linker, latest_speed, latest_actual)
    probability_weight = 1.6
    model_co_weight = 0.65
    template_weight = 3.0
    global_best_intensity, global_best_co, global_min_intensity, global_min_co, global_search, main_diagnostics = optimize_global_standard_model(
        ranges,
        evaluate_main_plan,
        latest_actual,
        historical_profile,
        historical_stats,
        probability_weight,
        model_co_weight,
        template_weight,
        low_seed_points,
    )
    main_diagnostics.update(main_pred_diag)
    if main_diagnostics["unhandled_q1_features"]:
        raise ValueError(f"第1问特征未处理：{main_diagnostics['unhandled_q1_features'][:10]}")
    global_scenario_temperature, global_q1_base_temp, global_q1_plan_temp, global_temp_delta, global_clip, global_unhandled = (
        incremental_temperature_for_plan(latest_q2.iloc[0], global_best_intensity, latest_actual, control_model, q1_linker, latest_speed)
    )
    main_diagnostics["max_temperature_clip_count"] = max(int(main_diagnostics["max_temperature_clip_count"]), global_clip)
    main_diagnostics["unhandled_q1_features"] = sorted(set(main_diagnostics["unhandled_q1_features"]) | set(global_unhandled))
    global_decision = make_global_decision_table(
        ranges,
        cleaned,
        latest_q2.iloc[0],
        control_model,
        global_best_intensity,
        global_scenario_temperature,
        global_q1_base_temp,
        global_q1_plan_temp,
        global_temp_delta,
        historical_stats,
    )
    global_comp = make_global_comparison(
        cleaned,
        latest_q2,
        ranges,
        control_model,
        evaluate_main_plan,
        historical_stats,
        latest_actual,
        global_best_intensity,
        global_best_co,
        global_min_intensity,
        global_min_co,
        main_diagnostics,
    )

    latest_lagged = lagged_intensity_from_row(latest_q2.iloc[0], control_model)
    median_start = ranges["历史强度中位数"].to_numpy(dtype=float)
    lower_start = ranges["优化强度下限_P10"].to_numpy(dtype=float)
    upper_start = ranges["优化强度上限_P90"].to_numpy(dtype=float)
    center_start = (lower_start + upper_start) / 2
    start_points = [latest_actual, latest_lagged, median_start, lower_start, upper_start, center_start]

    objective, aux_diagnostics = objective_factory(latest_q2, control_model, q1_linker, latest_speed, latest_actual)
    aux_best_intensity, aux_best_delta, search_record = optimize_pressure_plan(ranges, start_points, objective, constraints, latest_actual)
    lows, highs, _, adjacent_limits, total_budget = constrained_bounds_and_limits(ranges, constraints, latest_actual)
    slsqp_record = search_record[search_record["求解方法"] == "SLSQP连续优化"] if "求解方法" in search_record.columns else pd.DataFrame()
    aux_diagnostics["optimizer"] = "当前工况平稳调控：SLSQP连续优化 + 坐标局部精修"
    aux_diagnostics["slsqp_start_count"] = int(len(slsqp_record))
    aux_diagnostics["slsqp_success_count"] = int(slsqp_record["SLSQP成功"].fillna(False).sum()) if "SLSQP成功" in slsqp_record.columns else 0
    aux_diagnostics["slsqp_total_nfev"] = int(pd.to_numeric(slsqp_record.get("函数评估数", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    aux_diagnostics["best_constraint_violation"] = max_constraint_violation(aux_best_intensity, latest_actual, lows, highs, adjacent_limits, total_budget)
    aux_scenario_temperature, aux_q1_baseline_temperature, aux_q1_plan_temperature, aux_temperature_delta, clipped_count, unhandled = (
        incremental_temperature_for_plan(latest_q2.iloc[0], aux_best_intensity, latest_actual, control_model, q1_linker, latest_speed)
    )
    aux_diagnostics["unhandled_q1_features"] = sorted(set(aux_diagnostics["unhandled_q1_features"]) | set(unhandled))
    aux_diagnostics["max_temperature_clip_count"] = max(int(aux_diagnostics["max_temperature_clip_count"]), clipped_count)
    if aux_diagnostics["unhandled_q1_features"]:
        raise ValueError(f"第1问特征未处理：{aux_diagnostics['unhandled_q1_features'][:10]}")
    if np.any(np.abs(aux_temperature_delta) > ENGINEERING_TEMP_DELTA_LIMIT + 1e-8):
        aux_diagnostics["temperature_screening"] = "工程平稳方案触发严格温度筛选，回退到控制时刻实际负压"
        aux_best_intensity = latest_actual.copy()
        aux_best_delta = float(objective(aux_best_intensity))
        aux_scenario_temperature, aux_q1_baseline_temperature, aux_q1_plan_temperature, aux_temperature_delta, clipped_count, unhandled = (
            incremental_temperature_for_plan(latest_q2.iloc[0], aux_best_intensity, latest_actual, control_model, q1_linker, latest_speed)
        )
        aux_diagnostics["max_temperature_clip_count"] = max(int(aux_diagnostics["max_temperature_clip_count"]), clipped_count)
    else:
        aux_diagnostics["temperature_screening"] = "工程平稳方案满足严格温度增量筛选"

    marginal = make_marginal_table(latest_q2, ranges, constraints, latest_actual, control_model, q1_linker, latest_speed)
    aux_decision = make_decision_table(
        ranges,
        cleaned,
        latest_q2.iloc[0],
        constraints,
        control_model,
        aux_best_intensity,
        aux_scenario_temperature,
        aux_q1_baseline_temperature,
        aux_q1_plan_temperature,
        aux_temperature_delta,
        marginal,
    )
    aux_comp = make_latest_comparison(
        cleaned,
        latest_q2,
        q1_linker,
        control_model,
        latest_actual,
        aux_best_intensity,
        aux_scenario_temperature,
        aux_best_delta,
        latest_speed,
    )
    scheme_summary, scheme_pressure = make_three_scheme_outputs(global_comp, aux_decision, aux_comp, main_diagnostics, aux_diagnostics)
    eval_df = evaluate_plan_over_samples(future_aligned, cleaned_by_sequence, ranges, constraints, q1_linker, control_model, median_start)
    summary = summarize_eval(eval_df)
    rolling_trajectory, rolling_summary = generate_rolling_control_trajectory(
        future_aligned,
        cleaned_by_sequence,
        ranges,
        constraints,
        q1_linker,
        control_model,
        median_start,
    )

    params = {
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "优化目标": "主模型按工况分层训练调控响应模型，并在最新窗口执行滚动调控；辅助模型保留当前工况附近平稳约束",
        "调控模型构造": "重新构造未来300s预测数据集，保留历史CO状态变量，输入当前过程变量与负压调节量，输出未来300s低排放潜力",
        "温度联动": "第一问只提供ΔT=T_q1(方案)-T_q1(基准)，再叠加到第二问当前配准温度，不做绝对温度替换",
        "情景估计": "不用当前CO+ΔCO作为最终判断；非线性模型预测CO、达标概率和历史达标距离共同构成低排放评分，不能单独替代实测达标判定",
        "压力口径": "原始负压为负值；模型输入使用负压强度abs(原始负压)",
        "主模型优化范围": "每个风箱历史负压强度P05-P95",
        "辅助模型优化范围": "当前工况附近小幅平稳约束叠加P10-P90",
        "低排放工况识别": low_emission["summary"].to_dict(orient="records"),
        "工况分层画像": control_model.regime_profiles.to_dict(orient="records") if hasattr(control_model, "regime_profiles") else [],
        "可信达标样本调控模型复核": trusted_audit_summary.to_dict(orient="records"),
        "滚动调控摘要": rolling_summary.to_dict(orient="records"),
        "三类调控方案汇总": scheme_summary.to_dict(orient="records"),
        "第1问重构模型": q1_linker.source_note,
        "第1问训练样本数": q1_linker.fit_rows,
        "第2问监测型模型": str(deps["第2问监测型模型"]),
        "未来300s低排放潜力模型特征数": len(control_model.feature_names),
        "未来300s低排放潜力模型验证RMSE": control_model.validation_rmse,
        "可信达标样本数": int(historical_summary.iloc[0]["历史达标样本数"]),
        "修正后CO小于等于2800样本数": int(historical_summary.iloc[0].get("修正后CO小于等于2800样本数", historical_summary.iloc[0]["历史达标样本数"])),
        "低排放片段数": int(low_emission["summary"].iloc[0]["低排放片段数"]),
        "最长低排放片段持续时间_s": int(low_emission["summary"].iloc[0]["最长低排放片段持续时间_s"]),
        "主模型状态": main_diagnostics["status"],
        "达标概率项是否启用": bool(main_diagnostics.get("probability_term_enabled", False)),
        "达标概率项权重": float(main_diagnostics.get("probability_weight", 0.0)),
        "CO模型项权重": float(main_diagnostics.get("model_co_weight", np.nan)),
        "可信达标模板距离权重": float(main_diagnostics.get("template_weight", np.nan)),
        "主模型最终非线性预测CO": float(global_best_co),
        "主模型CO最小非线性预测CO": float(global_min_co),
        "主模型最终达标倾向概率": float(main_diagnostics.get("final_standard_probability", np.nan)),
        "主模型最终历史达标IQR距离": float(main_diagnostics.get("final_historical_distance", np.nan)),
        "辅助模型最优模型增量CO": float(aux_best_delta),
        "达标连续片段数": int(compliant_condition_diagnostics["summary"].iloc[0]["连续达标片段数"]),
        "最长连续达标持续时间_s": int(compliant_condition_diagnostics["summary"].iloc[0]["最长连续持续时间_s"]),
        "主模型最优目标强度": {str(i): float(global_best_intensity[i - 1]) for i in BOX_IDS},
        "主模型最优目标原始负压": {str(i): -float(global_best_intensity[i - 1]) for i in BOX_IDS},
        "辅助模型最优目标强度": {str(i): float(aux_best_intensity[i - 1]) for i in BOX_IDS},
        "主模型第一问温度增量": {str(i): float(global_temp_delta[i - 1]) for i in BOX_IDS},
        "主模型第二问增量叠加后温度": {str(i): float(global_scenario_temperature[i - 1]) for i in BOX_IDS},
        "主模型诊断": main_diagnostics,
        "辅助模型诊断": aux_diagnostics,
    }
    save_json(params, DATA_MID / "负压温度CO联动模型参数.json")
    save_json(params, DATA_MID / "监测型衍生调控模型参数.json")
    save_json(params, DATA_MID / "规划模型参数.json")
    save_json(
        {
            "样本数_原始": int(len(raw)),
            "样本数_清洗后": int(len(cleaned)),
            "样本数_第2问建模": int(len(aligned)),
            "样本数_未来300s建模": int(len(future_aligned)),
            "未来300s低排放潜力模型测试RMSE": float(control_metrics.loc[control_metrics["数据集"] == "测试集", "RMSE"].iloc[0]),
            "历史达标样本数": int(historical_summary.iloc[0]["历史达标样本数"]),
            "修正后CO小于等于2800样本数": int(historical_summary.iloc[0].get("修正后CO小于等于2800样本数", historical_summary.iloc[0]["历史达标样本数"])),
            "低排放片段数": int(low_emission["summary"].iloc[0]["低排放片段数"]),
            "最长低排放片段持续时间_s": int(low_emission["summary"].iloc[0]["最长低排放片段持续时间_s"]),
            "可信达标样本预测低于2800比例_%": float(trusted_audit_summary.iloc[0]["预测低于2800比例_%"]),
            "可信达标样本预测3000以上比例_%": float(trusted_audit_summary.iloc[0]["预测3000以上比例_%"]),
            "达标概率项是否启用": bool(main_diagnostics.get("probability_term_enabled", False)),
            "主模型状态": main_diagnostics["status"],
            "主模型最终非线性预测CO": float(global_best_co),
            "主模型最终模型值是否低于2800_仅参考": bool(global_best_co <= EMISSION_STANDARD),
            "主模型距2800_仅参考": float(global_best_co - EMISSION_STANDARD),
            "主模型最终达标倾向概率": float(main_diagnostics.get("final_standard_probability", np.nan)),
            "主模型最终历史达标IQR距离": float(main_diagnostics.get("final_historical_distance", np.nan)),
            "主模型CO最小非线性预测CO": float(global_min_co),
            "辅助模型最优模型增量CO": float(aux_best_delta),
            "三类调控方案汇总": scheme_summary.to_dict(orient="records"),
            "滚动窗口平均预测降低量": float(rolling_summary.iloc[0]["平均模型预测降低量"]) if not rolling_summary.empty else np.nan,
            "达标连续片段数": int(compliant_condition_diagnostics["summary"].iloc[0]["连续达标片段数"]),
            "最长连续达标持续时间_s": int(compliant_condition_diagnostics["summary"].iloc[0]["最长连续持续时间_s"]),
        },
        DATA_OUT / "第3问运行摘要.json",
    )

    make_figures(ranges, global_decision, global_comp, eval_df, marginal)
    make_schema(control_model, q1_linker)
    make_report(
        global_decision,
        global_comp,
        aux_decision,
        aux_comp,
        summary,
        historical_stats,
        historical_summary,
        compliant_condition_diagnostics,
        low_emission,
        control_model.regime_profiles if hasattr(control_model, "regime_profiles") else pd.DataFrame(),
        trusted_audit_summary,
        trusted_audit,
        rolling_summary,
        importance_group_summary,
        control_metrics,
        control_model,
        q1_linker,
        main_diagnostics,
        aux_diagnostics,
    )


if __name__ == "__main__":
    main()
