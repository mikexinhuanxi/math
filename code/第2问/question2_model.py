# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn import __version__ as SKLEARN_VERSION
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
except Exception:  # pragma: no cover - optional runtime dependency
    SKLEARN_VERSION = ""
    GradientBoostingRegressor = None
    RandomForestRegressor = None

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional runtime dependency
    lgb = None
    LGBMRegressor = None

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional runtime dependency
    xgb = None
    XGBRegressor = None


ROOT = Path(__file__).resolve().parents[2]
XLSX_PATH = ROOT / "附件1.原始数据.xlsx"
PDF_PATH = ROOT / "赛题A.pdf"
ANALYSIS_PATH = ROOT / "题目分析.md"

RAW_DIR = ROOT / "data" / "第2问" / "原始数据"
MID_DIR = ROOT / "data" / "第2问" / "中间数据"
RESULT_DATA_DIR = ROOT / "data" / "第2问" / "结果数据"
CODE_DIR = ROOT / "code" / "第2问"
FIG_ROOT = ROOT / "figure" / "第2问"
SHARED_FIG_DIR = FIG_ROOT / "共用图"
FIG_DIR = SHARED_FIG_DIR
RESULT_DIR = ROOT / "result" / "第2问"

SAMPLE_INTERVAL_SECONDS = 2
MAX_LAG_SAMPLES = 450
MAX_LAG_SECONDS = MAX_LAG_SAMPLES * SAMPLE_INTERVAL_SECONDS
CO_HISTORY_MAX_LAG_SAMPLES = 150
LOCAL_LAG_MIN_RADIUS_SECONDS = 60
LOCAL_LAG_MAX_RADIUS_SECONDS = 300
TARGET_COL = "烧结大烟道外排CO浓度"
SPEED_COL = "烧结机机速L1设定"
CO_HISTORY_LAGS = sorted({1, 2, 5, 10, 30, 60, 120, CO_HISTORY_MAX_LAG_SAMPLES})
WIND_BOX_CAP_GROUPS = [
    ("1#-6#上游风箱", 1, 6),
    ("7#-12#中游风箱", 7, 12),
    ("13#-18#下游风箱", 13, 18),
]
WIND_BOX_LAG_CAP_CANDIDATES = {
    "1#-6#上游风箱": [150, 240, 300, 360, 450],
    "7#-12#中游风箱": [90, 150, 210, 300],
    "13#-18#下游风箱": [30, 60, 90, 150],
}
SPEED_MAX_LAG_SAMPLES = 150
DUCT_MAX_LAG_SAMPLES = 60
RAW_VARIANT_NAME = "原始数据建模"
CORRECTED_VARIANT_NAME = "异常修正后建模"
BASELINE3_MODEL_NAME = "基线3_过程变量+历史CO监测型模型"
GROUP_LAG_DELTA_CANDIDATES = [-60, 0, 60]
GROUP_LAG_SIMPLICITY_TOLERANCE = 0.01
LAG_SELECTION_FOLDS = [
    ("前段验证窗口", 0.55, 0.70),
    ("后段验证窗口", 0.70, 0.85),
]
LAG_SELECTION_STD_PENALTY = 0.25


def ensure_dirs() -> None:
    for folder in [RAW_DIR, MID_DIR, RESULT_DATA_DIR, CODE_DIR, FIG_DIR, SHARED_FIG_DIR, RESULT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def set_variant_output_dirs(variant_name: str) -> None:
    global MID_DIR, RESULT_DATA_DIR, FIG_DIR, RESULT_DIR
    MID_DIR = ROOT / "data" / "第2问" / variant_name / "中间数据"
    RESULT_DATA_DIR = ROOT / "data" / "第2问" / variant_name / "结果数据"
    FIG_DIR = ROOT / "figure" / "第2问" / variant_name / "基础图"
    RESULT_DIR = ROOT / "result" / "第2问" / variant_name
    ensure_dirs()


def copy_sources() -> None:
    for src in [XLSX_PATH, PDF_PATH, ANALYSIS_PATH]:
        if src.exists():
            shutil.copy2(src, RAW_DIR / src.name)


def load_source_workbook() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw_matrix = pd.read_excel(XLSX_PATH, sheet_name=0, header=None, engine="openpyxl")
    group_row = raw_matrix.iloc[0].copy()
    header_row = raw_matrix.iloc[1].astype(str).copy()
    data = raw_matrix.iloc[2:].copy()
    data.columns = header_row
    data.reset_index(drop=True, inplace=True)

    source_header = pd.DataFrame(
        {
            "列序号": np.arange(1, len(header_row) + 1),
            "分组说明": group_row.values,
            "字段名": header_row.values,
        }
    )
    raw_csv = RAW_DIR / "附件1_字段化原始数据.csv"
    data.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    source_header.to_csv(RAW_DIR / "附件1_原始表头说明.csv", index=False, encoding="utf-8-sig")
    return data, source_header, header_row


def clean_data(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    numeric = raw_data.copy()
    for col in numeric.columns:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")

    missing_before = numeric.isna().sum()
    n_before = len(numeric)
    numeric = numeric.dropna(subset=["序列", TARGET_COL]).copy()
    numeric = numeric.drop_duplicates(subset=["序列"], keep="first")
    numeric = numeric.sort_values("序列").reset_index(drop=True)

    numeric_interp = numeric.interpolate(method="linear", limit_direction="both")
    numeric_interp = numeric_interp.ffill().bfill()
    missing_after = numeric_interp.isna().sum()

    numeric_interp.insert(1, "时间_s", (numeric_interp["序列"] - numeric_interp["序列"].iloc[0]) * SAMPLE_INTERVAL_SECONDS)

    for i in range(1, 19):
        south = f"{i}#风箱南侧温度"
        north = f"{i}#风箱北侧温度"
        avg = f"{i}#风箱平均温度"
        numeric_interp[avg] = numeric_interp[[south, north]].mean(axis=1)
        pressure = f"{i}#风箱负压"
        pressure_strength = f"{i}#风箱负压强度"
        numeric_interp[pressure_strength] = numeric_interp[pressure].abs()

    for i in range(1, 3):
        pressure = f"{i}#大烟道负压"
        pressure_strength = f"{i}#大烟道负压强度"
        numeric_interp[pressure_strength] = numeric_interp[pressure].abs()

    report = {
        "原始有效数据行数": int(n_before),
        "清洗后数据行数": int(len(numeric_interp)),
        "删除序列或目标缺失行数": int(n_before - len(numeric.dropna(subset=["序列", TARGET_COL]))),
        "重复序列删除后行数": int(len(numeric)),
        "清洗后剩余缺失总数": int(numeric_interp.isna().sum().sum()),
        "采样间隔_s": SAMPLE_INTERVAL_SECONDS,
        "温度合并规则": "每个风箱的南侧温度与北侧温度逐行算术平均，形成 i#风箱平均温度。",
        "负压方向统一规则": "原始负压保留不改；建模输入使用负压绝对值，形成 i#风箱负压强度、i#大烟道负压强度。强度越大表示抽风/负压越强。",
    }

    missing_table = pd.DataFrame(
        {
            "字段名": numeric.columns,
            "清洗前缺失数": [int(missing_before.get(col, 0)) for col in numeric.columns],
            "清洗后缺失数": [int(missing_after.get(col, 0)) for col in numeric.columns],
        }
    )
    numeric_interp.to_csv(MID_DIR / "清洗后数据.csv", index=False, encoding="utf-8-sig")
    missing_table.to_csv(MID_DIR / "缺失值处理统计.csv", index=False, encoding="utf-8-sig")
    with open(MID_DIR / "数据清洗报告.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return numeric_interp, missing_table, report


def correct_co_outliers(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    corrected = clean_df.copy()
    y = corrected[TARGET_COL].astype(float)

    q1 = float(y.quantile(0.25))
    q3 = float(y.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 4.0 * iqr
    upper_bound = q3 + 4.0 * iqr
    global_mask = (y < lower_bound) | (y > upper_bound)

    rolling_median = y.rolling(31, center=True, min_periods=7).median()
    local_dev = (y - rolling_median).abs()
    local_mad = local_dev.rolling(31, center=True, min_periods=7).median() * 1.4826
    local_floor = max(500.0, 3.0 * iqr)
    local_threshold = np.maximum(6.0 * local_mad.fillna(float(local_mad.median())), local_floor)
    prev_jump = (y - y.shift(1)).abs()
    next_jump = (y - y.shift(-1)).abs()
    local_spike_mask = (local_dev > local_threshold) & (prev_jump > 0.5 * local_threshold) & (next_jump > 0.5 * local_threshold)

    outlier_mask = (global_mask | local_spike_mask).fillna(False)
    original = y.copy()
    corrected_series = y.mask(outlier_mask)
    corrected_series = corrected_series.interpolate(method="linear", limit_direction="both").ffill().bfill()
    corrected[TARGET_COL] = corrected_series

    detail = pd.DataFrame(
        {
            "序列": corrected["序列"],
            "时间_s": corrected["时间_s"],
            "原始CO": original,
            "修正后CO": corrected_series,
            "修正差值": corrected_series - original,
            "是否修正": outlier_mask.astype(bool),
            "全局界限异常": global_mask.astype(bool),
            "局部尖峰异常": local_spike_mask.astype(bool),
            "全局下界": lower_bound,
            "全局上界": upper_bound,
        }
    )
    detail.to_csv(MID_DIR / "CO异常修正明细.csv", index=False, encoding="utf-8-sig")

    changed = detail[detail["是否修正"]].copy()
    report = {
        "启用": True,
        "处理方式": "不删除样本；将判定为异常的 CO 点置空后，用相邻正常点线性插值修正。",
        "判定规则": "全局 IQR 界限（Q1-4IQR, Q3+4IQR）或局部 Hampel 型孤立尖峰/跌落。",
        "滚动窗口样本数": 31,
        "全局下界": lower_bound,
        "全局上界": upper_bound,
        "修正点数": int(outlier_mask.sum()),
        "样本总数": int(len(corrected)),
        "修正比例_%": float(outlier_mask.mean() * 100),
        "最大绝对修正量": float(changed["修正差值"].abs().max()) if not changed.empty else 0.0,
        "修正点序列": [int(v) for v in changed["序列"].tolist()],
    }
    with open(MID_DIR / "CO异常修正报告.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return corrected, report


def write_no_co_outlier_correction_report(clean_df: pd.DataFrame) -> dict:
    report = {
        "启用": False,
        "处理方式": "原始数据建模口径，不修正 CO 异常点，也不删除样本。",
        "修正点数": 0,
        "样本总数": int(len(clean_df)),
        "修正比例_%": 0.0,
    }
    pd.DataFrame(
        {
            "序列": clean_df["序列"],
            "时间_s": clean_df["时间_s"],
            "原始CO": clean_df[TARGET_COL],
            "修正后CO": clean_df[TARGET_COL],
            "修正差值": 0.0,
            "是否修正": False,
        }
    ).to_csv(MID_DIR / "CO异常修正明细.csv", index=False, encoding="utf-8-sig")
    with open(MID_DIR / "CO异常修正报告.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def build_field_table(columns: pd.Index) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col == "序列":
            kind, unit, source = "时间索引", "编号，间隔2s", "原始字段"
        elif col == "时间_s":
            kind, unit, source = "时间索引", "s", "由序列换算"
        elif col == SPEED_COL:
            kind, unit, source = "控制变量", "m/min", "原始字段"
        elif "风箱负压强度" in col:
            kind, unit, source = "控制变量（方向统一）", "KPa", "由原始风箱负压取绝对值"
        elif "风箱负压" in col:
            kind, unit, source = "控制变量", "KPa", "原始字段，负值表示负压"
        elif "风箱平均温度" in col:
            kind, unit, source = "状态变量", "℃", "南北侧温度平均"
        elif "风箱" in col and "温度" in col:
            kind, unit, source = "状态变量", "℃", "原始字段"
        elif "大烟道负压强度" in col:
            kind, unit, source = "控制变量（方向统一）", "KPa", "由原始大烟道负压取绝对值"
        elif "大烟道负压" in col:
            kind, unit, source = "控制变量", "KPa", "原始字段，负值表示负压"
        elif "大烟道温度" in col:
            kind, unit, source = "状态变量", "℃", "原始字段"
        elif col == TARGET_COL:
            kind, unit, source = "预测目标", "mg/m^3", "原始字段"
        else:
            kind, unit, source = "其他", "", "原始字段或派生字段"
        rows.append({"字段名": col, "类型": kind, "单位/口径": unit, "来源说明": source})
    field_table = pd.DataFrame(rows)
    field_table.to_csv(MID_DIR / "字段清单.csv", index=False, encoding="utf-8-sig")
    return field_table


def get_feature_groups() -> dict[str, list[str]]:
    return {
        "机速": [SPEED_COL],
        "风箱负压强度": [f"{i}#风箱负压强度" for i in range(1, 19)],
        "风箱平均温度": [f"{i}#风箱平均温度" for i in range(1, 19)],
        "大烟道负压强度": ["1#大烟道负压强度", "2#大烟道负压强度"],
        "大烟道温度": ["1#大烟道温度", "2#大烟道温度"],
    }


def lagged_corr(x: np.ndarray, y: np.ndarray, lag: int, end: int) -> float:
    if lag == 0:
        xx = x[:end]
        yy = y[:end]
    else:
        xx = x[: end - lag]
        yy = y[lag:end]
    mask = np.isfinite(xx) & np.isfinite(yy)
    if mask.sum() < 10:
        return 0.0
    xx = xx[mask]
    yy = yy[mask]
    sx = xx.std()
    sy = yy.std()
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.corrcoef(xx, yy)[0, 1])


def lagged_univariate_rmse(x: np.ndarray, y: np.ndarray, lag: int, end: int) -> float:
    if lag == 0:
        xx = x[:end]
        yy = y[:end]
    else:
        xx = x[: end - lag]
        yy = y[lag:end]
    mask = np.isfinite(xx) & np.isfinite(yy)
    if mask.sum() < 10:
        return float("inf")
    xx = xx[mask]
    yy = yy[mask]
    x_std = float(xx.std())
    if x_std <= 1e-12:
        pred = np.full_like(yy, float(yy.mean()), dtype=float)
    else:
        design = np.column_stack([np.ones(len(xx)), xx])
        coef = np.linalg.pinv(design.T @ design) @ (design.T @ yy)
        pred = design @ coef
    return float(np.sqrt(np.mean((pred - yy) ** 2)))


def windbox_geometry() -> tuple[dict[int, dict], float]:
    lengths = {i: 4.0 for i in range(1, 16)}
    lengths.update({i: 2.0 for i in range(16, 19)})
    start = 0.0
    rows = {}
    for i in range(1, 19):
        length = lengths[i]
        center = start + length / 2
        rows[i] = {"长度_m": length, "中心位置_m": center}
        start += length
    return rows, start


def windbox_no_from_feature(feature: str) -> int | None:
    if "#风箱" not in feature:
        return None
    try:
        return int(feature.split("#", 1)[0])
    except ValueError:
        return None


def windbox_cap_group_name_from_no(box_no: int | None) -> str | None:
    if box_no is None:
        return None
    for group_name, start, end in WIND_BOX_CAP_GROUPS:
        if start <= box_no <= end:
            return group_name
    return None


def default_windbox_lag_cap_config() -> dict[str, int]:
    return {group_name: max(candidates) for group_name, candidates in WIND_BOX_LAG_CAP_CANDIDATES.items()}


def max_lag_for_feature(feature: str, cap_config: dict[str, int] | None = None) -> tuple[int, str]:
    box_no = windbox_no_from_feature(feature)
    if box_no is not None:
        group_name = windbox_cap_group_name_from_no(box_no)
        if group_name is not None:
            config = cap_config or default_windbox_lag_cap_config()
            samples = int(config[group_name])
            return samples, f"{group_name}候选寻优截断窗口，最大滞后{samples * SAMPLE_INTERVAL_SECONDS}s"
    if feature == SPEED_COL:
        return SPEED_MAX_LAG_SAMPLES, "机速全线设定量，最大滞后300s"
    if "大烟道" in feature:
        return DUCT_MAX_LAG_SAMPLES, "大烟道变量靠近检测位置，最大滞后120s"
    return SPEED_MAX_LAG_SAMPLES, "其他过程变量，最大滞后300s"


def make_physical_lag_priors(
    clean_df: pd.DataFrame,
    process_features: list[str],
    train_end: int,
    max_lag_samples: int = MAX_LAG_SAMPLES,
) -> pd.DataFrame:
    geometry, detector_position = windbox_geometry()
    train_speed = float(clean_df[SPEED_COL].iloc[:train_end].mean())
    speed_m_per_s = train_speed / 60 if train_speed > 0 else np.nan
    groups = get_feature_groups()
    feature_to_group = {feature: group for group, features in groups.items() for feature in features}
    rows = []

    for feature in process_features:
        group = feature_to_group.get(feature, "")
        box_no = windbox_no_from_feature(feature)
        feature_max_lag_samples, cap_rule = max_lag_for_feature(feature)
        feature_max_lag_samples = min(feature_max_lag_samples, max_lag_samples)
        length = np.nan
        center = np.nan
        distance = 0.0
        initial_seconds = 0.0
        source = "非风箱变量，初始时滞设为0，仅在局部窗口内微调"

        if box_no is not None:
            length = geometry[box_no]["长度_m"]
            center = geometry[box_no]["中心位置_m"]
            distance = detector_position - center
            if np.isfinite(speed_m_per_s) and speed_m_per_s > 0:
                initial_seconds = distance / speed_m_per_s
            source = "按风箱中心位置到18#风箱末端检测参考点的距离/训练集平均机速估计"
        elif feature == SPEED_COL:
            source = "机速为全线设定量，无单独空间位置，初始时滞设为0"
        elif "大烟道" in feature:
            source = "大烟道变量靠近汇集/检测位置，初始时滞设为0"

        initial_samples = int(round(initial_seconds / SAMPLE_INTERVAL_SECONDS))
        radius_seconds = max(
            LOCAL_LAG_MIN_RADIUS_SECONDS,
            min(LOCAL_LAG_MAX_RADIUS_SECONDS, initial_seconds * 0.15),
        )
        radius_samples = int(round(radius_seconds / SAMPLE_INTERVAL_SECONDS))
        max_allowed = min(max(train_end - 20, 0), feature_max_lag_samples)
        capped_initial_samples = max(0, min(initial_samples, max_allowed))
        search_low = max(0, capped_initial_samples - radius_samples)
        search_high = min(max_allowed, capped_initial_samples + radius_samples)
        if search_low > search_high:
            search_low = max(0, min(capped_initial_samples, max_allowed))
            search_high = search_low

        rows.append(
            {
                "字段名": feature,
                "变量组": group,
                "风箱编号": box_no if box_no is not None else "",
                "风箱长度_m": length,
                "风箱中心位置_m": center,
                "检测参考位置_m": detector_position,
                "到检测参考点距离_m": distance,
                "训练集平均机速_m_min": train_speed,
                "物理初始时滞_s": initial_samples * SAMPLE_INTERVAL_SECONDS,
                "物理初始滞后样本数": initial_samples,
                "窗口截断后初始时滞_s": capped_initial_samples * SAMPLE_INTERVAL_SECONDS,
                "窗口截断后初始滞后样本数": capped_initial_samples,
                "分段最大滞后_s": feature_max_lag_samples * SAMPLE_INTERVAL_SECONDS,
                "分段最大滞后样本数": feature_max_lag_samples,
                "分段最大滞后规则": cap_rule,
                "局部搜索半径_s": radius_samples * SAMPLE_INTERVAL_SECONDS,
                "搜索下限样本数": search_low,
                "搜索上限样本数": search_high,
                "搜索下限_s": search_low * SAMPLE_INTERVAL_SECONDS,
                "搜索上限_s": search_high * SAMPLE_INTERVAL_SECONDS,
                "初始时滞来源": source,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(MID_DIR / "风箱位置初始时滞.csv", index=False, encoding="utf-8-sig")
    return table


def make_lag_cap_candidate_configs() -> list[dict]:
    rows = []
    upstream_name, middle_name, downstream_name = [item[0] for item in WIND_BOX_CAP_GROUPS]
    combo_id = 1
    for upstream in WIND_BOX_LAG_CAP_CANDIDATES[upstream_name]:
        for middle in WIND_BOX_LAG_CAP_CANDIDATES[middle_name]:
            for downstream in WIND_BOX_LAG_CAP_CANDIDATES[downstream_name]:
                if not (upstream >= middle >= downstream):
                    continue
                cap_config = {
                    upstream_name: int(upstream),
                    middle_name: int(middle),
                    downstream_name: int(downstream),
                }
                rows.append(
                    {
                        "截断组合编号": combo_id,
                        "截断窗口配置": cap_config,
                        "截断窗口配置_s": {key: value * SAMPLE_INTERVAL_SECONDS for key, value in cap_config.items()},
                    }
                )
                combo_id += 1
    return rows


def apply_lag_cap_config_to_prior(
    lag_prior_table: pd.DataFrame,
    cap_config: dict[str, int],
    train_end: int,
    max_lag_samples: int = MAX_LAG_SAMPLES,
) -> pd.DataFrame:
    table = lag_prior_table.copy()
    max_allowed_global = min(max(train_end - 20, 0), max_lag_samples)

    for idx, row in table.iterrows():
        feature = str(row["字段名"])
        box_no = windbox_no_from_feature(feature)
        cap_group = windbox_cap_group_name_from_no(box_no)
        if cap_group is not None:
            feature_max_lag_samples = int(cap_config[cap_group])
            cap_rule = f"{cap_group}截断窗口由多验证窗口寻优选定，最大滞后{feature_max_lag_samples * SAMPLE_INTERVAL_SECONDS}s"
        else:
            feature_max_lag_samples, cap_rule = max_lag_for_feature(feature)
            cap_group = "非风箱变量"

        feature_max_lag_samples = min(feature_max_lag_samples, max_lag_samples)
        max_allowed = min(max_allowed_global, feature_max_lag_samples)
        raw_lag = int(row.get("物理初始滞后样本数", 0))
        capped_initial_samples = max(0, min(raw_lag, max_allowed))
        initial_seconds = raw_lag * SAMPLE_INTERVAL_SECONDS
        radius_seconds = max(
            LOCAL_LAG_MIN_RADIUS_SECONDS,
            min(LOCAL_LAG_MAX_RADIUS_SECONDS, initial_seconds * 0.15),
        )
        radius_samples = int(round(radius_seconds / SAMPLE_INTERVAL_SECONDS))
        search_low = max(0, capped_initial_samples - radius_samples)
        search_high = min(max_allowed, capped_initial_samples + radius_samples)
        if search_low > search_high:
            search_low = max(0, min(capped_initial_samples, max_allowed))
            search_high = search_low

        table.loc[idx, "截断窗口优化组"] = cap_group
        table.loc[idx, "分段最大滞后样本数"] = int(feature_max_lag_samples)
        table.loc[idx, "分段最大滞后_s"] = int(feature_max_lag_samples) * SAMPLE_INTERVAL_SECONDS
        table.loc[idx, "分段最大滞后规则"] = cap_rule
        table.loc[idx, "窗口截断后初始滞后样本数"] = int(capped_initial_samples)
        table.loc[idx, "窗口截断后初始时滞_s"] = int(capped_initial_samples) * SAMPLE_INTERVAL_SECONDS
        table.loc[idx, "搜索下限样本数"] = int(search_low)
        table.loc[idx, "搜索上限样本数"] = int(search_high)
        table.loc[idx, "搜索下限_s"] = int(search_low) * SAMPLE_INTERVAL_SECONDS
        table.loc[idx, "搜索上限_s"] = int(search_high) * SAMPLE_INTERVAL_SECONDS
    return table


def select_lags(
    clean_df: pd.DataFrame,
    process_features: list[str],
    train_end: int,
    max_lag_samples: int = MAX_LAG_SAMPLES,
    detail_output: Path | None = None,
    best_output: Path | None = None,
    lag_prior_table: pd.DataFrame | None = None,
    alignment_check_output: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = clean_df[TARGET_COL].to_numpy(dtype=float)
    detail_rows = []
    best_rows = []
    groups = get_feature_groups()
    feature_to_group = {feature: group for group, features in groups.items() for feature in features}
    prior_lookup = {}
    if lag_prior_table is not None:
        prior_lookup = lag_prior_table.set_index("字段名").to_dict(orient="index")

    for feature in process_features:
        x = clean_df[feature].to_numpy(dtype=float)
        prior = prior_lookup.get(feature, {})
        feature_max_lag_samples = int(prior.get("分段最大滞后样本数", max_lag_for_feature(feature)[0]))
        feature_max_lag_samples = min(feature_max_lag_samples, max_lag_samples)
        search_low = int(prior.get("搜索下限样本数", 0))
        search_high = int(prior.get("搜索上限样本数", max_lag_samples))
        search_low = max(0, search_low)
        max_allowed = min(max(train_end - 20, 0), feature_max_lag_samples)
        search_low = min(search_low, max_allowed)
        search_high = min(max_allowed, max(search_low, search_high))
        scores = []
        for lag in range(search_low, search_high + 1):
            corr = lagged_corr(x, y, lag, train_end)
            rmse = lagged_univariate_rmse(x, y, lag, train_end)
            scores.append({"lag": lag, "corr": corr, "rmse": rmse})
            detail_rows.append(
                {
                    "字段名": feature,
                    "变量组": feature_to_group.get(feature, ""),
                    "物理初始滞后样本数": prior.get("物理初始滞后样本数", ""),
                    "物理初始时滞_s": prior.get("物理初始时滞_s", ""),
                    "分段最大滞后样本数": feature_max_lag_samples,
                    "分段最大滞后_s": feature_max_lag_samples * SAMPLE_INTERVAL_SECONDS,
                    "搜索下限样本数": search_low,
                    "搜索上限样本数": search_high,
                    "滞后样本数": lag,
                    "滞后秒数": lag * SAMPLE_INTERVAL_SECONDS,
                    "训练集相关系数": corr,
                    "训练集绝对相关系数": abs(corr),
                    "训练集单变量RMSE": rmse,
                }
            )
        best_score = min(scores, key=lambda item: item["rmse"])
        best_lag = int(best_score["lag"])
        best_corr = float(best_score["corr"])
        best_rmse = float(best_score["rmse"])
        zero_rmse = lagged_univariate_rmse(x, y, 0, train_end)
        initial_lag = int(prior.get("窗口截断后初始滞后样本数", prior.get("物理初始滞后样本数", best_lag)))
        initial_lag = max(0, min(max_allowed, initial_lag))
        initial_rmse = lagged_univariate_rmse(x, y, initial_lag, train_end)
        best_rows.append(
            {
                "字段名": feature,
                "变量组": feature_to_group.get(feature, ""),
                "风箱编号": prior.get("风箱编号", ""),
                "到检测参考点距离_m": prior.get("到检测参考点距离_m", ""),
                "训练集平均机速_m_min": prior.get("训练集平均机速_m_min", ""),
                "物理初始滞后样本数": prior.get("物理初始滞后样本数", ""),
                "物理初始时滞_s": prior.get("物理初始时滞_s", ""),
                "窗口截断后初始滞后样本数": prior.get("窗口截断后初始滞后样本数", initial_lag),
                "窗口截断后初始时滞_s": prior.get("窗口截断后初始时滞_s", initial_lag * SAMPLE_INTERVAL_SECONDS),
                "分段最大滞后样本数": feature_max_lag_samples,
                "分段最大滞后_s": feature_max_lag_samples * SAMPLE_INTERVAL_SECONDS,
                "分段最大滞后规则": prior.get("分段最大滞后规则", max_lag_for_feature(feature)[1]),
                "搜索下限样本数": search_low,
                "搜索上限样本数": search_high,
                "搜索下限_s": search_low * SAMPLE_INTERVAL_SECONDS,
                "搜索上限_s": search_high * SAMPLE_INTERVAL_SECONDS,
                "最佳滞后样本数": best_lag,
                "最佳滞后秒数": best_lag * SAMPLE_INTERVAL_SECONDS,
                "训练集相关系数": best_corr,
                "零滞后单变量RMSE": zero_rmse,
                "物理初始滞后单变量RMSE": initial_rmse,
                "最佳滞后单变量RMSE": best_rmse,
                "RMSE较零滞后降低率_%": (zero_rmse - best_rmse) / zero_rmse * 100 if np.isfinite(zero_rmse) and zero_rmse else np.nan,
                "RMSE较物理初始降低率_%": (initial_rmse - best_rmse) / initial_rmse * 100 if np.isfinite(initial_rmse) and initial_rmse else np.nan,
                "影响方向": "正相关" if best_corr >= 0 else "负相关",
                "时滞选择准则": "物理初始时滞附近局部搜索，但候选滞后不超过变量分段最大窗口" if lag_prior_table is not None else "给定范围内训练集单变量RMSE最小",
                "最大搜索滞后样本数": feature_max_lag_samples,
                "最大搜索滞后秒数": feature_max_lag_samples * SAMPLE_INTERVAL_SECONDS,
            }
        )

    detail = pd.DataFrame(detail_rows)
    best = pd.DataFrame(best_rows)
    if detail_output is not None:
        detail.to_csv(detail_output, index=False, encoding="utf-8-sig")
    if best_output is not None:
        best.to_csv(best_output, index=False, encoding="utf-8-sig")
    if alignment_check_output is not None:
        best.to_csv(alignment_check_output, index=False, encoding="utf-8-sig")
    return detail, best


def make_no_lag_table(clean_df: pd.DataFrame, process_features: list[str], train_end: int) -> pd.DataFrame:
    groups = get_feature_groups()
    feature_to_group = {feature: group for group, features in groups.items() for feature in features}
    y = clean_df[TARGET_COL].to_numpy(dtype=float)
    rows = []
    for feature in process_features:
        corr = lagged_corr(clean_df[feature].to_numpy(dtype=float), y, 0, train_end)
        rows.append(
            {
                "字段名": feature,
                "变量组": feature_to_group.get(feature, ""),
                "最佳滞后样本数": 0,
                "最佳滞后秒数": 0,
                "训练集相关系数": corr,
                "影响方向": "正相关" if corr >= 0 else "负相关",
                "最大搜索滞后样本数": 0,
                "最大搜索滞后秒数": 0,
            }
        )
    return pd.DataFrame(rows)


def make_physical_initial_lag_table(
    clean_df: pd.DataFrame,
    process_features: list[str],
    train_end: int,
    lag_prior_table: pd.DataFrame,
    max_lag_samples: int = MAX_LAG_SAMPLES,
) -> pd.DataFrame:
    groups = get_feature_groups()
    feature_to_group = {feature: group for group, features in groups.items() for feature in features}
    prior_lookup = lag_prior_table.set_index("字段名").to_dict(orient="index")
    y = clean_df[TARGET_COL].to_numpy(dtype=float)
    max_allowed = min(max(train_end - 20, 0), max_lag_samples)
    rows = []

    for feature in process_features:
        prior = prior_lookup.get(feature, {})
        feature_max_lag_samples = int(prior.get("分段最大滞后样本数", max_lag_for_feature(feature)[0]))
        feature_max_lag_samples = min(feature_max_lag_samples, max_lag_samples)
        max_allowed_feature = min(max_allowed, feature_max_lag_samples)
        raw_lag = int(prior.get("物理初始滞后样本数", 0))
        lag = int(prior.get("窗口截断后初始滞后样本数", raw_lag))
        lag = max(0, min(max_allowed_feature, lag))
        x = clean_df[feature].to_numpy(dtype=float)
        corr = lagged_corr(x, y, lag, train_end)
        zero_rmse = lagged_univariate_rmse(x, y, 0, train_end)
        initial_rmse = lagged_univariate_rmse(x, y, lag, train_end)
        rows.append(
            {
                "字段名": feature,
                "变量组": feature_to_group.get(feature, ""),
                "风箱编号": prior.get("风箱编号", ""),
                "到检测参考点距离_m": prior.get("到检测参考点距离_m", ""),
                "训练集平均机速_m_min": prior.get("训练集平均机速_m_min", ""),
                "未截断物理初始滞后样本数": raw_lag,
                "未截断物理初始时滞_s": raw_lag * SAMPLE_INTERVAL_SECONDS,
                "物理初始滞后样本数": lag,
                "物理初始时滞_s": lag * SAMPLE_INTERVAL_SECONDS,
                "窗口截断后初始滞后样本数": lag,
                "窗口截断后初始时滞_s": lag * SAMPLE_INTERVAL_SECONDS,
                "分段最大滞后样本数": feature_max_lag_samples,
                "分段最大滞后_s": feature_max_lag_samples * SAMPLE_INTERVAL_SECONDS,
                "分段最大滞后规则": prior.get("分段最大滞后规则", max_lag_for_feature(feature)[1]),
                "搜索下限样本数": lag,
                "搜索上限样本数": lag,
                "搜索下限_s": lag * SAMPLE_INTERVAL_SECONDS,
                "搜索上限_s": lag * SAMPLE_INTERVAL_SECONDS,
                "最佳滞后样本数": lag,
                "最佳滞后秒数": lag * SAMPLE_INTERVAL_SECONDS,
                "训练集相关系数": corr,
                "零滞后单变量RMSE": zero_rmse,
                "物理初始滞后单变量RMSE": initial_rmse,
                "最佳滞后单变量RMSE": initial_rmse,
                "RMSE较零滞后降低率_%": (zero_rmse - initial_rmse) / zero_rmse * 100 if np.isfinite(zero_rmse) and zero_rmse else np.nan,
                "RMSE较物理初始降低率_%": 0.0,
                "影响方向": "正相关" if corr >= 0 else "负相关",
                "时滞选择准则": "物理初始时滞经变量分段最大窗口截断，不做局部修正",
                "最大搜索滞后样本数": feature_max_lag_samples,
                "最大搜索滞后秒数": feature_max_lag_samples * SAMPLE_INTERVAL_SECONDS,
                "截断窗口优化组": prior.get("截断窗口优化组", ""),
                "截断窗口组合编号": prior.get("截断窗口组合编号", ""),
                "截断窗口选择准则": prior.get("截断窗口选择准则", ""),
                "截断窗口平均验证RMSE": prior.get("截断窗口平均验证RMSE", ""),
                "截断窗口验证RMSE标准差": prior.get("截断窗口验证RMSE标准差", ""),
                "截断窗口稳健目标RMSE": prior.get("截断窗口稳健目标RMSE", ""),
                "截断窗口配置": prior.get("截断窗口配置", ""),
                "截断窗口配置_s": prior.get("截断窗口配置_s", ""),
            }
        )
    return pd.DataFrame(rows)


def lag_search_group_name(feature: str) -> str:
    box_no = windbox_no_from_feature(feature)
    if box_no is not None:
        if 1 <= box_no <= 6:
            position = "1#-6#上游风箱"
        elif 7 <= box_no <= 12:
            position = "7#-12#中游风箱"
        else:
            position = "13#-18#下游风箱"
        if "负压强度" in feature:
            return f"{position}_负压强度"
        if "平均温度" in feature:
            return f"{position}_平均温度"
        return f"{position}_其他"
    if feature == SPEED_COL:
        return "机速"
    if "大烟道负压强度" in feature:
        return "大烟道负压强度"
    if "大烟道温度" in feature:
        return "大烟道温度"
    return "其他过程变量"


def group_candidate_lag_map(base_lag_table: pd.DataFrame, group_name: str, delta_samples: int) -> dict[str, int]:
    subset = base_lag_table[base_lag_table["时滞优化组"] == group_name]
    lag_map = {}
    for _, row in subset.iterrows():
        max_allowed = int(row["分段最大滞后样本数"])
        base_lag = int(row["窗口截断后初始滞后样本数"])
        lag = max(0, min(max_allowed, base_lag + int(delta_samples)))
        lag_map[str(row["字段名"])] = int(lag)
    return lag_map


def make_grouped_lag_candidate_table(base_lag_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name in sorted(base_lag_table["时滞优化组"].unique()):
        for delta in GROUP_LAG_DELTA_CANDIDATES:
            lag_map = group_candidate_lag_map(base_lag_table, group_name, delta)
            if not lag_map:
                continue
            lags = list(lag_map.values())
            row = {
                "时滞优化组": group_name,
                "候选调整样本数": int(delta),
                "候选调整秒数": int(delta) * SAMPLE_INTERVAL_SECONDS,
                "组内字段数": len(lag_map),
                "组内最小候选滞后样本数": int(min(lags)),
                "组内最大候选滞后样本数": int(max(lags)),
                "组内最小候选滞后_s": int(min(lags)) * SAMPLE_INTERVAL_SECONDS,
                "组内最大候选滞后_s": int(max(lags)) * SAMPLE_INTERVAL_SECONDS,
                "候选字段滞后": json.dumps({k: v * SAMPLE_INTERVAL_SECONDS for k, v in lag_map.items()}, ensure_ascii=False),
            }
            rows.append(row)
    table = pd.DataFrame(rows)
    table = table.drop_duplicates(subset=["时滞优化组", "候选字段滞后"]).reset_index(drop=True)
    return table


def apply_group_lag_choice(base_lag_table: pd.DataFrame, choice: dict[str, int]) -> pd.DataFrame:
    selected = base_lag_table.copy()
    selected["时滞选择准则"] = "分组时滞组合优化：以完整监测型模型多验证窗口稳健RMSE最小为目标"
    selected["已选组滞后调整样本数"] = 0
    selected["已选组滞后调整_s"] = 0
    selected["优化前物理截断滞后样本数"] = selected["窗口截断后初始滞后样本数"].astype(int)
    selected["优化前物理截断时滞_s"] = selected["窗口截断后初始时滞_s"]

    for group_name, delta in choice.items():
        mask = selected["时滞优化组"] == group_name
        selected.loc[mask, "已选组滞后调整样本数"] = int(delta)
        selected.loc[mask, "已选组滞后调整_s"] = int(delta) * SAMPLE_INTERVAL_SECONDS
        for idx, row in selected[mask].iterrows():
            max_allowed = int(row["分段最大滞后样本数"])
            base_lag = int(row["窗口截断后初始滞后样本数"])
            lag = max(0, min(max_allowed, base_lag + int(delta)))
            selected.loc[idx, "最佳滞后样本数"] = int(lag)
            selected.loc[idx, "最佳滞后秒数"] = int(lag) * SAMPLE_INTERVAL_SECONDS
            selected.loc[idx, "搜索下限样本数"] = int(max(0, min(max_allowed, base_lag + min(GROUP_LAG_DELTA_CANDIDATES))))
            selected.loc[idx, "搜索上限样本数"] = int(max(0, min(max_allowed, base_lag + max(GROUP_LAG_DELTA_CANDIDATES))))
            selected.loc[idx, "搜索下限_s"] = int(selected.loc[idx, "搜索下限样本数"]) * SAMPLE_INTERVAL_SECONDS
            selected.loc[idx, "搜索上限_s"] = int(selected.loc[idx, "搜索上限样本数"]) * SAMPLE_INTERVAL_SECONDS
    return selected


def split_for_lag_selection_fold(
    aligned: pd.DataFrame,
    train_ratio: float,
    val_end_ratio: float,
    source_length: int | None = None,
) -> pd.DataFrame:
    if source_length is None:
        source_length = int(aligned["原始行号"].max()) + 1
    train_end = int(source_length * train_ratio)
    val_end = int(source_length * val_end_ratio)
    target_rows = aligned["原始行号"].to_numpy(dtype=int)
    split = np.where(
        target_rows < train_end,
        "训练集",
        np.where((target_rows >= train_end) & (target_rows < val_end), "验证集", "调参未用"),
    )
    folded = aligned.copy()
    folded["数据集"] = split
    return folded


def monitor_validation_rmse_for_lag_table(clean_df: pd.DataFrame, lag_table: pd.DataFrame) -> tuple[float, dict]:
    aligned, process_lagged_features, co_features = make_aligned_dataset(clean_df, lag_table, output_path=None)
    monitor_features = process_lagged_features + co_features
    source_length = int(clean_df["原始行号"].max()) + 1 if "原始行号" in clean_df else len(clean_df)
    fold_rows = []

    for fold_name, train_ratio, val_end_ratio in LAG_SELECTION_FOLDS:
        folded = split_for_lag_selection_fold(aligned, train_ratio, val_end_ratio, source_length=source_length)
        if (folded["数据集"] == "训练集").sum() < 50 or (folded["数据集"] == "验证集").sum() < 20:
            continue
        _, pred_monitor_ridge, monitor_info = fit_ridge_model(folded, monitor_features, f"分组时滞监测型模型_{fold_name}")
        _, blend_info = blend_monitor_prediction(folded, pred_monitor_ridge)
        fold_rows.append(
            {
                "验证窗口": fold_name,
                "训练截止比例": train_ratio,
                "验证截止比例": val_end_ratio,
                "训练样本数": int((folded["数据集"] == "训练集").sum()),
                "验证样本数": int((folded["数据集"] == "验证集").sum()),
                "监测型岭回归验证RMSE": float(monitor_info["validation_rmse"]),
                "监测型融合验证RMSE": float(blend_info["validation_rmse"]),
                "监测型岭回归权重": float(blend_info["ridge_weight"]),
                "监测型CO_lag1权重": float(blend_info["lag1_weight"]),
            }
        )

    if not fold_rows:
        raise RuntimeError("时滞稳健选择没有可用的验证窗口。")

    fold_table = pd.DataFrame(fold_rows)
    fold_rmse = fold_table["监测型融合验证RMSE"].to_numpy(dtype=float)
    mean_rmse = float(np.mean(fold_rmse))
    std_rmse = float(np.std(fold_rmse, ddof=0))
    objective_rmse = mean_rmse + LAG_SELECTION_STD_PENALTY * std_rmse
    return objective_rmse, {
        "样本数": int(len(aligned)),
        "调参验证窗口数": int(len(fold_rows)),
        "平均验证RMSE": mean_rmse,
        "验证RMSE标准差": std_rmse,
        "最差验证RMSE": float(np.max(fold_rmse)),
        "稳健目标RMSE": objective_rmse,
        "稳健目标标准差惩罚系数": LAG_SELECTION_STD_PENALTY,
        "验证窗口明细": json.dumps(fold_rows, ensure_ascii=False),
        "监测型岭回归验证RMSE": float(fold_table["监测型岭回归验证RMSE"].mean()),
        "监测型融合验证RMSE": mean_rmse,
        "监测型岭回归权重": float(fold_table["监测型岭回归权重"].mean()),
        "监测型CO_lag1权重": float(fold_table["监测型CO_lag1权重"].mean()),
    }


def select_lag_cap_config_by_monitor_cv(
    clean_df: pd.DataFrame,
    process_features: list[str],
    train_end: int,
    lag_prior_table: pd.DataFrame,
    search_output: Path | None = None,
    best_prior_output: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    best = None

    for item in make_lag_cap_candidate_configs():
        candidate_prior = apply_lag_cap_config_to_prior(lag_prior_table, item["截断窗口配置"], train_end)
        candidate_lag_table = make_physical_initial_lag_table(
            clean_df,
            process_features,
            train_end,
            candidate_prior,
            max_lag_samples=MAX_LAG_SAMPLES,
        )
        objective_rmse, info = monitor_validation_rmse_for_lag_table(clean_df, candidate_lag_table)
        row = {
            "截断组合编号": int(item["截断组合编号"]),
            "稳健目标RMSE": objective_rmse,
            "平均验证RMSE": info["平均验证RMSE"],
            "验证RMSE标准差": info["验证RMSE标准差"],
            "最差验证RMSE": info["最差验证RMSE"],
            "调参验证窗口数": info["调参验证窗口数"],
            "截断窗口配置": json.dumps(item["截断窗口配置"], ensure_ascii=False),
            "截断窗口配置_s": json.dumps(item["截断窗口配置_s"], ensure_ascii=False),
            "验证窗口明细": info["验证窗口明细"],
        }
        for group_name, samples in item["截断窗口配置"].items():
            row[f"{group_name}_最大滞后样本数"] = int(samples)
            row[f"{group_name}_最大滞后_s"] = int(samples) * SAMPLE_INTERVAL_SECONDS
        rows.append(row)
        if best is None or objective_rmse < best["objective_rmse"]:
            best = {
                "objective_rmse": objective_rmse,
                "info": info,
                "item": item,
                "prior": candidate_prior,
            }

    search_table = pd.DataFrame(rows).sort_values("稳健目标RMSE").reset_index(drop=True)
    search_table["按稳健目标RMSE排序"] = np.arange(1, len(search_table) + 1)
    if search_output is not None:
        search_table.to_csv(search_output, index=False, encoding="utf-8-sig")

    selected_prior = best["prior"].copy()
    selected_prior["截断窗口组合编号"] = int(best["item"]["截断组合编号"])
    selected_prior["截断窗口选择准则"] = "完整监测型模型多验证窗口平均RMSE+稳定性惩罚最小"
    selected_prior["截断窗口平均验证RMSE"] = float(best["info"]["平均验证RMSE"])
    selected_prior["截断窗口验证RMSE标准差"] = float(best["info"]["验证RMSE标准差"])
    selected_prior["截断窗口稳健目标RMSE"] = float(best["objective_rmse"])
    selected_prior["截断窗口配置"] = json.dumps(best["item"]["截断窗口配置"], ensure_ascii=False)
    selected_prior["截断窗口配置_s"] = json.dumps(best["item"]["截断窗口配置_s"], ensure_ascii=False)
    selected_prior["截断窗口验证窗口明细"] = best["info"]["验证窗口明细"]
    if best_prior_output is not None:
        selected_prior.to_csv(best_prior_output, index=False, encoding="utf-8-sig")
    return selected_prior, search_table


def select_grouped_lags_by_monitor_validation(
    clean_df: pd.DataFrame,
    base_lag_table: pd.DataFrame,
    candidate_output: Path | None = None,
    combo_output: Path | None = None,
    best_output: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = base_lag_table.copy()
    working["时滞优化组"] = working["字段名"].map(lag_search_group_name)
    candidate_table = make_grouped_lag_candidate_table(working)
    if candidate_output is not None:
        candidate_table.to_csv(candidate_output, index=False, encoding="utf-8-sig")

    group_candidates = []
    for group_name in sorted(working["时滞优化组"].unique()):
        candidates = candidate_table[candidate_table["时滞优化组"] == group_name]["候选调整样本数"].astype(int).tolist()
        candidates = sorted(set(candidates))
        group_candidates.append((group_name, candidates))

    combo_rows = []
    combo_lookup = {}
    import itertools

    for combo_id, deltas in enumerate(itertools.product(*[item[1] for item in group_candidates]), start=1):
        choice = {group_name: int(delta) for (group_name, _), delta in zip(group_candidates, deltas)}
        candidate_lag_table = apply_group_lag_choice(working, choice)
        validation_rmse, info = monitor_validation_rmse_for_lag_table(clean_df, candidate_lag_table)
        nonzero_adjustments = int(sum(1 for delta in choice.values() if int(delta) != 0))
        abs_adjustment_sum = int(sum(abs(int(delta)) for delta in choice.values()))
        row = {
            "组合编号": combo_id,
            "稳健目标RMSE": validation_rmse,
            "验证RMSE": info["平均验证RMSE"],
            "组数": len(choice),
            "非零组调整数": nonzero_adjustments,
            "调整绝对值合计": abs_adjustment_sum,
            "组合": json.dumps(choice, ensure_ascii=False),
        }
        row.update(info)
        for group_name, delta in choice.items():
            row[f"{group_name}_调整样本数"] = int(delta)
            row[f"{group_name}_调整_s"] = int(delta) * SAMPLE_INTERVAL_SECONDS
        combo_rows.append(row)
        combo_lookup[combo_id] = {"validation_rmse": validation_rmse, "choice": choice, "info": info, "combo_id": combo_id}

    combo_table = pd.DataFrame(combo_rows).sort_values("稳健目标RMSE").reset_index(drop=True)
    combo_table["按稳健目标RMSE排序"] = np.arange(1, len(combo_table) + 1)
    best_objective = float(combo_table["稳健目标RMSE"].min())
    eligible = combo_table[combo_table["稳健目标RMSE"] <= best_objective * (1 + GROUP_LAG_SIMPLICITY_TOLERANCE)].copy()
    chosen_row = eligible.sort_values(["非零组调整数", "调整绝对值合计", "稳健目标RMSE"]).iloc[0]
    best = combo_lookup[int(chosen_row["组合编号"])]
    combo_table["是否简约规则入选"] = combo_table["组合编号"] == int(chosen_row["组合编号"])
    combo_table["简约入选容忍率_%"] = GROUP_LAG_SIMPLICITY_TOLERANCE * 100
    if combo_output is not None:
        combo_table.to_csv(combo_output, index=False, encoding="utf-8-sig")

    selected = apply_group_lag_choice(working, best["choice"])
    selected["分组组合编号"] = int(best["combo_id"])
    selected["分组组合验证RMSE"] = float(best["info"]["平均验证RMSE"])
    selected["分组组合稳健目标RMSE"] = float(best["validation_rmse"])
    selected["分组组合验证RMSE标准差"] = float(best["info"]["验证RMSE标准差"])
    selected["分组组合最差验证RMSE"] = float(best["info"]["最差验证RMSE"])
    selected["分组组合"] = json.dumps(best["choice"], ensure_ascii=False)
    selected["分组组合选择规则"] = f"先取稳健目标RMSE最优值{GROUP_LAG_SIMPLICITY_TOLERANCE * 100:.1f}%以内的组合，再优先选择非零调整组数和调整幅度最小的简约组合"
    selected["分组组合理论最优稳健目标RMSE"] = best_objective
    selected["分组组合简约容忍率_%"] = GROUP_LAG_SIMPLICITY_TOLERANCE * 100
    for key, value in best["info"].items():
        selected[f"分组组合{key}"] = value
    y = clean_df[TARGET_COL].to_numpy(dtype=float)
    for idx, row in selected.iterrows():
        x = clean_df[row["字段名"]].to_numpy(dtype=float)
        lag = int(row["最佳滞后样本数"])
        corr = lagged_corr(x, y, lag, int(len(clean_df) * 0.70))
        selected.loc[idx, "训练集相关系数"] = corr
        selected.loc[idx, "影响方向"] = "正相关" if corr >= 0 else "负相关"
    if best_output is not None:
        selected.to_csv(best_output, index=False, encoding="utf-8-sig")
    return selected, candidate_table, combo_table


def make_aligned_dataset(
    clean_df: pd.DataFrame,
    lag_table: pd.DataFrame,
    output_path: Path | None = MID_DIR / "滞后配准建模数据.csv",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    aligned = clean_df[["序列", "时间_s", TARGET_COL]].copy()
    aligned["原始行号"] = np.arange(len(clean_df))
    process_lagged_features = []

    for _, row in lag_table.iterrows():
        feature = row["字段名"]
        lag = int(row["最佳滞后样本数"])
        new_name = f"{feature}__lag{lag}"
        aligned[new_name] = clean_df[feature].shift(lag)
        process_lagged_features.append(new_name)

    co_features = []
    for lag in CO_HISTORY_LAGS:
        col = f"CO历史浓度__lag{lag}"
        aligned[col] = clean_df[TARGET_COL].shift(lag)
        co_features.append(col)
    aligned["CO历史浓度__rolling30_lag1"] = clean_df[TARGET_COL].shift(1).rolling(30, min_periods=3).mean()
    aligned["CO历史浓度__rolling120_lag1"] = clean_df[TARGET_COL].shift(1).rolling(120, min_periods=10).mean()
    co_features.extend(["CO历史浓度__rolling30_lag1", "CO历史浓度__rolling120_lag1"])

    aligned = aligned.dropna().reset_index(drop=True)
    if output_path is not None:
        aligned.to_csv(output_path, index=False, encoding="utf-8-sig")
    return aligned, process_lagged_features, co_features


def split_by_time(aligned: pd.DataFrame, source_length: int | None = None) -> pd.DataFrame:
    if source_length is None:
        source_length = int(aligned["原始行号"].max()) + 1
    train_end = int(source_length * 0.70)
    val_end = int(source_length * 0.85)
    target_rows = aligned["原始行号"].to_numpy(dtype=int)
    split = np.where(target_rows < train_end, "训练集", np.where(target_rows < val_end, "验证集", "测试集"))
    aligned = aligned.copy()
    aligned["数据集"] = split
    return aligned


def augment_features(x: np.ndarray, degree: int) -> np.ndarray:
    if degree == 1:
        return x
    return np.concatenate([x, x * x], axis=1)


def fit_ridge_model(aligned: pd.DataFrame, feature_cols: list[str], model_name: str) -> tuple[dict, pd.Series, dict]:
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

    alphas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
    candidates = []

    for degree in [1, 2]:
        z_aug = augment_features(z, degree)
        x_design = np.concatenate([np.ones((len(z_aug), 1)), z_aug], axis=1)
        xtr = x_design[train_mask]
        ytr = y[train_mask]
        xval = x_design[val_mask]
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
                best_for_degree = {
                    "degree": degree,
                    "alpha": alpha,
                    "rmse": rmse,
                    "coef": coef,
                    "x_design": x_design,
                }
        candidates.append(best_for_degree)

    linear_candidate = [item for item in candidates if item["degree"] == 1][0]
    best_candidate = min(candidates, key=lambda item: item["rmse"])
    # Prefer the simpler linear model unless the quadratic terms reduce validation RMSE by at least 3%.
    if best_candidate["degree"] == 2 and best_candidate["rmse"] > 0.97 * linear_candidate["rmse"]:
        best_candidate = linear_candidate

    best_alpha = best_candidate["alpha"]
    best_coef = best_candidate["coef"]
    best_degree = best_candidate["degree"]
    best_rmse = best_candidate["rmse"]
    x_design = best_candidate["x_design"]

    pred = pd.Series(x_design @ best_coef, index=aligned.index, name=f"{model_name}_预测CO浓度")
    feature_importance = []
    linear_coef = best_coef[1 : 1 + len(feature_cols)]
    if best_degree == 2:
        square_coef = best_coef[1 + len(feature_cols) :]
    else:
        square_coef = np.zeros(len(feature_cols))
    for feature, c1, c2 in zip(feature_cols, linear_coef, square_coef):
        feature_importance.append(
            {
                "模型": model_name,
                "特征名": feature,
                "标准化线性系数": float(c1),
                "标准化平方项系数": float(c2),
                "综合影响强度": float(math.sqrt(c1 * c1 + c2 * c2)),
                "线性影响方向": "正向" if c1 >= 0 else "负向",
            }
        )

    model = {
        "模型名称": model_name,
        "模型类型": "标准化线性/二次特征岭回归",
        "实际采用特征次数": best_degree,
        "目标字段": TARGET_COL,
        "采样间隔_s": SAMPLE_INTERVAL_SECONDS,
        "最大滞后_s": MAX_LAG_SECONDS,
        "特征字段": feature_cols,
        "标准化均值": mu.tolist(),
        "标准化标准差": sigma.tolist(),
        "岭回归alpha": best_alpha,
        "截距": float(best_coef[0]),
        "线性系数": linear_coef.tolist(),
        "平方项系数": square_coef.tolist(),
        "公式说明": "z=(x-mean)/std; 若实际采用特征次数为1，则 y=intercept+sum(linear_coef*z)；若为2，则再加 sum(square_coef*z^2)",
    }
    return model, pred, {"model_name": model_name, "validation_rmse": best_rmse, "importance": feature_importance}


def fit_residual_ridge_model(
    aligned: pd.DataFrame,
    feature_cols: list[str],
    baseline_col: str,
    model_name: str,
) -> tuple[dict, pd.Series, dict]:
    train_mask = aligned["数据集"] == "训练集"
    val_mask = aligned["数据集"] == "验证集"

    x = aligned[feature_cols].to_numpy(dtype=float)
    y = aligned[TARGET_COL].to_numpy(dtype=float)
    baseline = aligned[baseline_col].to_numpy(dtype=float)
    residual = y - baseline

    x_train = x[train_mask]
    residual_train = residual[train_mask]
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    z = (x - mu) / sigma

    alphas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
    candidates = []

    for degree in [1, 2]:
        z_aug = augment_features(z, degree)
        x_design = np.concatenate([np.ones((len(z_aug), 1)), z_aug], axis=1)
        xtr = x_design[train_mask]
        rtr = residual_train
        xval = x_design[val_mask]
        yval = y[val_mask]
        base_val = baseline[val_mask]

        best_for_degree = None
        for alpha in alphas:
            penalty = np.eye(xtr.shape[1])
            penalty[0, 0] = 0.0
            lhs = xtr.T @ xtr + alpha * penalty
            rhs = xtr.T @ rtr
            try:
                coef = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                coef = np.linalg.pinv(lhs) @ rhs
            pred_val = base_val + xval @ coef
            rmse = float(np.sqrt(np.mean((pred_val - yval) ** 2)))
            if best_for_degree is None or rmse < best_for_degree["rmse"]:
                best_for_degree = {
                    "degree": degree,
                    "alpha": alpha,
                    "rmse": rmse,
                    "coef": coef,
                    "x_design": x_design,
                }
        candidates.append(best_for_degree)

    linear_candidate = [item for item in candidates if item["degree"] == 1][0]
    best_candidate = min(candidates, key=lambda item: item["rmse"])
    if best_candidate["degree"] == 2 and best_candidate["rmse"] > 0.97 * linear_candidate["rmse"]:
        best_candidate = linear_candidate

    best_alpha = best_candidate["alpha"]
    best_coef = best_candidate["coef"]
    best_degree = best_candidate["degree"]
    best_rmse = best_candidate["rmse"]
    x_design = best_candidate["x_design"]

    residual_pred = x_design @ best_coef
    pred = pd.Series(baseline + residual_pred, index=aligned.index, name=f"{model_name}_预测CO浓度")

    linear_coef = best_coef[1 : 1 + len(feature_cols)]
    if best_degree == 2:
        square_coef = best_coef[1 + len(feature_cols) :]
    else:
        square_coef = np.zeros(len(feature_cols))

    feature_importance = []
    for feature, c1, c2 in zip(feature_cols, linear_coef, square_coef):
        feature_importance.append(
            {
                "模型": model_name,
                "特征名": feature,
                "标准化线性系数": float(c1),
                "标准化平方项系数": float(c2),
                "综合影响强度": float(math.sqrt(c1 * c1 + c2 * c2)),
                "线性影响方向": "正向" if c1 >= 0 else "负向",
            }
        )

    model = {
        "模型名称": model_name,
        "模型类型": "CO_lag1残差校正岭回归",
        "目标字段": TARGET_COL,
        "基线字段": baseline_col,
        "残差定义": f"{TARGET_COL} - {baseline_col}",
        "采样间隔_s": SAMPLE_INTERVAL_SECONDS,
        "特征字段": feature_cols,
        "标准化均值": mu.tolist(),
        "标准化标准差": sigma.tolist(),
        "岭回归alpha": best_alpha,
        "实际采用特征次数": best_degree,
        "截距": float(best_coef[0]),
        "线性系数": linear_coef.tolist(),
        "平方项系数": square_coef.tolist(),
        "公式说明": "baseline=CO_lag1; residual_pred=intercept+sum(linear_coef*z)[+sum(square_coef*z^2)]; CO_pred=baseline+residual_pred",
    }
    return model, pred, {"model_name": model_name, "validation_rmse": best_rmse, "importance": feature_importance}


def blend_monitor_prediction(aligned: pd.DataFrame, ridge_pred: pd.Series) -> tuple[pd.Series, dict]:
    val_mask = aligned["数据集"] == "验证集"
    y_val = aligned.loc[val_mask, TARGET_COL].to_numpy(dtype=float)
    lag1 = aligned["CO历史浓度__lag1"].to_numpy(dtype=float)
    ridge = ridge_pred.to_numpy(dtype=float)
    best = None
    for weight in np.linspace(0, 1, 101):
        pred_val = weight * ridge[val_mask] + (1 - weight) * lag1[val_mask]
        rmse = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))
        if best is None or rmse < best["validation_rmse"]:
            best = {"ridge_weight": float(weight), "lag1_weight": float(1 - weight), "validation_rmse": rmse}
    blended = best["ridge_weight"] * ridge + best["lag1_weight"] * lag1
    return pd.Series(blended, index=aligned.index, name="监测型模型_预测CO浓度"), best


def tree_leaf(value: float) -> dict:
    return {"leaf": True, "value": float(value)}


def predict_tree(tree: dict, x: np.ndarray) -> float:
    node = tree
    while not node.get("leaf", False):
        if x[node["feature"]] <= node["threshold"]:
            node = node["left"]
        else:
            node = node["right"]
    return float(node["value"])


def predict_forest(forest: dict, x: np.ndarray) -> np.ndarray:
    preds = np.zeros(len(x), dtype=float)
    for tree in forest["trees"]:
        preds += np.array([predict_tree(tree, row) for row in x], dtype=float)
    return preds / max(len(forest["trees"]), 1)


def build_regression_tree(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    depth: int,
    max_depth: int,
    min_samples_leaf: int,
    max_features: int,
    split_candidates: int,
) -> dict:
    if depth >= max_depth or len(y) < 2 * min_samples_leaf or float(np.var(y)) < 1e-9:
        return tree_leaf(float(np.mean(y)))

    n_features = x.shape[1]
    feature_count = min(max_features, n_features)
    feature_ids = rng.choice(n_features, size=feature_count, replace=False)
    total_sum = float(np.sum(y))
    total_sq = float(np.sum(y * y))
    best = None

    for feature in feature_ids:
        values = x[:, feature]
        if float(np.nanmax(values) - np.nanmin(values)) <= 1e-12:
            continue
        quantiles = np.linspace(0.1, 0.9, split_candidates)
        thresholds = np.unique(np.quantile(values, quantiles))
        for threshold in thresholds:
            left_mask = values <= threshold
            n_left = int(left_mask.sum())
            n_right = len(y) - n_left
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            y_left = y[left_mask]
            sum_left = float(np.sum(y_left))
            sq_left = float(np.sum(y_left * y_left))
            sum_right = total_sum - sum_left
            sq_right = total_sq - sq_left
            sse_left = sq_left - sum_left * sum_left / n_left
            sse_right = sq_right - sum_right * sum_right / n_right
            sse = sse_left + sse_right
            if best is None or sse < best["sse"]:
                best = {
                    "sse": sse,
                    "feature": int(feature),
                    "threshold": float(threshold),
                    "left_mask": left_mask,
                }

    if best is None:
        return tree_leaf(float(np.mean(y)))

    left_mask = best["left_mask"]
    right_mask = ~left_mask
    return {
        "leaf": False,
        "feature": best["feature"],
        "threshold": best["threshold"],
        "left": build_regression_tree(
            x[left_mask],
            y[left_mask],
            rng,
            depth + 1,
            max_depth,
            min_samples_leaf,
            max_features,
            split_candidates,
        ),
        "right": build_regression_tree(
            x[right_mask],
            y[right_mask],
            rng,
            depth + 1,
            max_depth,
            min_samples_leaf,
            max_features,
            split_candidates,
        ),
    }


def fit_random_forest_model(
    aligned: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    seed: int = 20260529,
) -> tuple[dict, pd.Series, dict]:
    train_mask = aligned["数据集"] == "训练集"
    val_mask = aligned["数据集"] == "验证集"
    x_all = aligned[feature_cols].to_numpy(dtype=float)
    y_all = aligned[TARGET_COL].to_numpy(dtype=float)
    x_train = x_all[train_mask]
    y_train = y_all[train_mask]

    configs = [
        {"n_estimators": 50, "max_depth": 3, "min_samples_leaf": 24, "max_features": max(3, int(math.sqrt(len(feature_cols)))), "split_candidates": 9},
        {"n_estimators": 70, "max_depth": 4, "min_samples_leaf": 18, "max_features": max(4, int(math.sqrt(len(feature_cols))) + 1), "split_candidates": 11},
        {"n_estimators": 90, "max_depth": 5, "min_samples_leaf": 14, "max_features": max(5, int(math.sqrt(len(feature_cols))) + 2), "split_candidates": 11},
    ]
    best = None

    for config_id, config in enumerate(configs):
        rng = np.random.default_rng(seed + config_id * 1009)
        trees = []
        n_train = len(x_train)
        for _ in range(config["n_estimators"]):
            sample_ids = rng.integers(0, n_train, size=n_train)
            tree = build_regression_tree(
                x_train[sample_ids],
                y_train[sample_ids],
                rng,
                depth=0,
                max_depth=config["max_depth"],
                min_samples_leaf=config["min_samples_leaf"],
                max_features=config["max_features"],
                split_candidates=config["split_candidates"],
            )
            trees.append(tree)
        forest = {"model_name": model_name, "feature_cols": feature_cols, "config": config, "trees": trees}
        pred_val = predict_forest(forest, x_all[val_mask])
        rmse = float(np.sqrt(np.mean((pred_val - y_all[val_mask]) ** 2)))
        if best is None or rmse < best["validation_rmse"]:
            best = {"forest": forest, "validation_rmse": rmse, "config": config}

    pred = pd.Series(predict_forest(best["forest"], x_all), index=aligned.index, name=f"{model_name}_预测CO浓度")
    model = {
        "模型名称": model_name,
        "模型类型": "纯numpy随机森林回归",
        "目标字段": TARGET_COL,
        "特征字段": feature_cols,
        "参数": best["config"],
        "验证RMSE": best["validation_rmse"],
        "说明": "模型由 question2_model.py 训练；为避免模型文件过大，JSON保存参数摘要，完整复现以代码和输入数据为准。",
    }
    return model, pred, {"model_name": model_name, "validation_rmse": best["validation_rmse"], "forest": best["forest"]}


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def monitor_nonlinear_specs() -> list[dict]:
    specs = []
    if LGBMRegressor is not None:
        specs.append(
            {
                "模型名称": "非线性监测_LightGBM",
                "算法": "LightGBM LGBMRegressor",
                "包版本": getattr(lgb, "__version__", ""),
                "factory": lambda params, seed: LGBMRegressor(
                    objective="regression",
                    random_state=seed,
                    n_jobs=2,
                    verbosity=-1,
                    **params,
                ),
                "参数候选": [
                    {"n_estimators": 120, "learning_rate": 0.04, "num_leaves": 15, "max_depth": 4, "min_child_samples": 20, "colsample_bytree": 0.9, "reg_lambda": 1.0},
                    {"n_estimators": 180, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 5, "min_child_samples": 15, "colsample_bytree": 0.85, "reg_lambda": 3.0},
                    {"n_estimators": 100, "learning_rate": 0.06, "num_leaves": 11, "max_depth": 3, "min_child_samples": 30, "colsample_bytree": 0.95, "reg_lambda": 5.0},
                ],
            }
        )
    if XGBRegressor is not None:
        specs.append(
            {
                "模型名称": "非线性监测_XGBoost",
                "算法": "XGBoost XGBRegressor",
                "包版本": getattr(xgb, "__version__", ""),
                "factory": lambda params, seed: XGBRegressor(
                    objective="reg:squarederror",
                    random_state=seed,
                    n_jobs=2,
                    tree_method="hist",
                    verbosity=0,
                    **params,
                ),
                "参数候选": [
                    {"n_estimators": 120, "learning_rate": 0.04, "max_depth": 3, "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 3.0},
                    {"n_estimators": 180, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 8, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 6.0},
                    {"n_estimators": 90, "learning_rate": 0.07, "max_depth": 2, "min_child_weight": 10, "subsample": 0.95, "colsample_bytree": 0.95, "reg_lambda": 8.0},
                ],
            }
        )
    if GradientBoostingRegressor is not None:
        specs.append(
            {
                "模型名称": "非线性监测_梯度提升树",
                "算法": "scikit-learn GradientBoostingRegressor",
                "包版本": SKLEARN_VERSION,
                "factory": lambda params, seed: GradientBoostingRegressor(random_state=seed, **params),
                "参数候选": [
                    {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 10, "subsample": 0.9},
                    {"n_estimators": 180, "learning_rate": 0.03, "max_depth": 3, "min_samples_leaf": 12, "subsample": 0.85},
                    {"n_estimators": 80, "learning_rate": 0.08, "max_depth": 2, "min_samples_leaf": 16, "subsample": 0.95},
                ],
            }
        )
    if RandomForestRegressor is not None:
        specs.append(
            {
                "模型名称": "非线性监测_随机森林",
                "算法": "scikit-learn RandomForestRegressor",
                "包版本": SKLEARN_VERSION,
                "factory": lambda params, seed: RandomForestRegressor(random_state=seed, n_jobs=2, **params),
                "参数候选": [
                    {"n_estimators": 220, "max_depth": 6, "min_samples_leaf": 6, "max_features": "sqrt"},
                    {"n_estimators": 260, "max_depth": 8, "min_samples_leaf": 4, "max_features": 0.65},
                    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 8, "max_features": "sqrt"},
                ],
            }
        )
    for spec in specs:
        spec["模型类型"] = "过程变量+历史CO非线性监测模型"
    return specs


def process_nonlinear_specs() -> list[dict]:
    specs = []
    if LGBMRegressor is not None:
        specs.append(
            {
                "模型名称": "过程变量_LightGBM",
                "算法": "LightGBM LGBMRegressor",
                "包版本": getattr(lgb, "__version__", ""),
                "factory": lambda params, seed: LGBMRegressor(
                    objective="regression",
                    random_state=seed,
                    n_jobs=2,
                    verbosity=-1,
                    **params,
                ),
                "参数候选": [
                    {"n_estimators": 120, "learning_rate": 0.04, "num_leaves": 11, "max_depth": 3, "min_child_samples": 24, "colsample_bytree": 0.9, "reg_lambda": 3.0},
                    {"n_estimators": 160, "learning_rate": 0.03, "num_leaves": 15, "max_depth": 4, "min_child_samples": 20, "colsample_bytree": 0.85, "reg_lambda": 5.0},
                    {"n_estimators": 90, "learning_rate": 0.06, "num_leaves": 9, "max_depth": 3, "min_child_samples": 30, "colsample_bytree": 0.95, "reg_lambda": 8.0},
                ],
            }
        )
    if XGBRegressor is not None:
        specs.append(
            {
                "模型名称": "过程变量_XGBoost",
                "算法": "XGBoost XGBRegressor",
                "包版本": getattr(xgb, "__version__", ""),
                "factory": lambda params, seed: XGBRegressor(
                    objective="reg:squarederror",
                    random_state=seed,
                    n_jobs=2,
                    tree_method="hist",
                    verbosity=0,
                    **params,
                ),
                "参数候选": [
                    {"n_estimators": 100, "learning_rate": 0.04, "max_depth": 2, "min_child_weight": 8, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 5.0},
                    {"n_estimators": 150, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 10, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 8.0},
                    {"n_estimators": 80, "learning_rate": 0.07, "max_depth": 2, "min_child_weight": 12, "subsample": 0.95, "colsample_bytree": 0.95, "reg_lambda": 10.0},
                ],
            }
        )
    if GradientBoostingRegressor is not None:
        specs.append(
            {
                "模型名称": "过程变量_梯度提升树",
                "算法": "scikit-learn GradientBoostingRegressor",
                "包版本": SKLEARN_VERSION,
                "factory": lambda params, seed: GradientBoostingRegressor(random_state=seed, **params),
                "参数候选": [
                    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 12, "subsample": 0.9},
                    {"n_estimators": 150, "learning_rate": 0.03, "max_depth": 3, "min_samples_leaf": 14, "subsample": 0.85},
                    {"n_estimators": 80, "learning_rate": 0.08, "max_depth": 2, "min_samples_leaf": 18, "subsample": 0.95},
                ],
            }
        )
    if RandomForestRegressor is not None:
        specs.append(
            {
                "模型名称": "过程变量_随机森林",
                "算法": "scikit-learn RandomForestRegressor",
                "包版本": SKLEARN_VERSION,
                "factory": lambda params, seed: RandomForestRegressor(random_state=seed, n_jobs=2, **params),
                "参数候选": [
                    {"n_estimators": 220, "max_depth": 5, "min_samples_leaf": 8, "max_features": "sqrt"},
                    {"n_estimators": 260, "max_depth": 7, "min_samples_leaf": 6, "max_features": 0.7},
                    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 10, "max_features": "sqrt"},
                ],
            }
        )
    for spec in specs:
        spec["模型类型"] = "不含历史CO的非线性过程变量回归模型"
    return specs


def fit_sklearn_like_regressor(
    aligned: pd.DataFrame,
    feature_cols: list[str],
    spec: dict,
    seed: int,
) -> tuple[dict, pd.Series, dict]:
    train_mask = aligned["数据集"] == "训练集"
    val_mask = aligned["数据集"] == "验证集"
    x_all = aligned[feature_cols].to_numpy(dtype=float)
    y_all = aligned[TARGET_COL].to_numpy(dtype=float)
    x_train = x_all[train_mask]
    y_train = y_all[train_mask]
    x_val = x_all[val_mask]
    y_val = y_all[val_mask]

    candidates = []
    failures = []
    best = None
    for config_id, params in enumerate(spec["参数候选"]):
        try:
            estimator = spec["factory"](params, seed + config_id * 1009)
            estimator.fit(x_train, y_train)
            pred_val = estimator.predict(x_val)
            rmse = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))
            candidate = {"参数": json_ready(params), "验证RMSE": rmse}
            candidates.append(candidate)
            if best is None or rmse < best["验证RMSE"]:
                best = {"estimator": estimator, "参数": dict(params), "验证RMSE": rmse}
        except Exception as exc:
            failures.append({"参数": json_ready(params), "错误": f"{type(exc).__name__}: {exc}"})

    if best is None:
        raise RuntimeError(f"{spec['模型名称']} 训练失败：{failures}")

    estimator = best["estimator"]
    pred = pd.Series(estimator.predict(x_all), index=aligned.index, name=f"{spec['模型名称']}_预测CO浓度")
    feature_importance = getattr(estimator, "feature_importances_", None)
    if feature_importance is not None:
        importance_rows = [
            {"模型特征": feature, "重要性": float(value)}
            for feature, value in sorted(zip(feature_cols, feature_importance), key=lambda item: item[1], reverse=True)
        ]
    else:
        importance_rows = []

    model = {
        "模型名称": spec["模型名称"],
        "模型类型": spec.get("模型类型", "非线性回归模型"),
        "算法": spec["算法"],
        "包版本": spec["包版本"],
        "目标字段": TARGET_COL,
        "采样间隔_s": SAMPLE_INTERVAL_SECONDS,
        "特征字段": feature_cols,
        "参数候选": json_ready(spec["参数候选"]),
        "候选验证RMSE": candidates,
        "最佳参数": json_ready(best["参数"]),
        "验证RMSE": best["验证RMSE"],
        "训练样本数": int(train_mask.sum()),
        "验证样本数": int(val_mask.sum()),
        "特征重要性": importance_rows,
        "训练失败候选": failures,
        "说明": "使用过程变量滞后特征与历史CO特征训练；JSON保存参数摘要，完整模型可由代码和输入数据复现。",
    }
    info = {
        "model_name": spec["模型名称"],
        "validation_rmse": best["验证RMSE"],
        "estimator": estimator,
        "feature_cols": feature_cols,
        "model": model,
    }
    return model, pred, info


def fit_nonlinear_monitor_suite(
    aligned: pd.DataFrame,
    feature_cols: list[str],
    seed: int = 20260531,
) -> tuple[dict[str, dict], dict[str, pd.Series], dict[str, dict]]:
    models = {}
    predictions = {}
    infos = {}
    for idx, spec in enumerate(monitor_nonlinear_specs()):
        model, pred, info = fit_sklearn_like_regressor(aligned, feature_cols, spec, seed + idx * 10007)
        models[spec["模型名称"]] = model
        predictions[spec["模型名称"]] = pred
        infos[spec["模型名称"]] = info
    return models, predictions, infos


def fit_nonlinear_process_suite(
    aligned: pd.DataFrame,
    feature_cols: list[str],
    seed: int = 20260531,
) -> tuple[dict[str, dict], dict[str, pd.Series], dict[str, dict]]:
    models = {}
    predictions = {}
    infos = {}
    for idx, spec in enumerate(process_nonlinear_specs()):
        model, pred, info = fit_sklearn_like_regressor(aligned, feature_cols, spec, seed + idx * 10007)
        models[spec["模型名称"]] = model
        predictions[spec["模型名称"]] = pred
        infos[spec["模型名称"]] = info
    return models, predictions, infos


def select_best_nonlinear_model(
    models: dict[str, dict],
    predictions: dict[str, pd.Series],
    infos: dict[str, dict],
    final_model_name: str,
) -> tuple[dict, pd.Series, dict, str]:
    if not infos:
        raise RuntimeError(f"{final_model_name} 没有可用的非线性候选模型。")
    best_name = min(infos, key=lambda name: infos[name]["validation_rmse"])
    model = dict(models[best_name])
    model["原候选模型名称"] = best_name
    model["模型名称"] = final_model_name
    model["选择依据"] = "在训练集拟合候选非线性模型，并按验证集 RMSE 最小选择为正式模型。"
    model["候选模型验证RMSE"] = {name: info["validation_rmse"] for name, info in infos.items()}
    pred = predictions[best_name].rename(f"{final_model_name}_预测CO浓度")
    info = dict(infos[best_name])
    info["selected_model_name"] = best_name
    info["model_name"] = final_model_name
    info["model"] = model
    return model, pred, info, best_name


def make_sklearn_frame_predictor(info: dict):
    feature_cols = info["feature_cols"]
    estimator = info["estimator"]
    return lambda frame: estimator.predict(frame[feature_cols].to_numpy(dtype=float))


def make_generic_permutation_importance(
    aligned: pd.DataFrame,
    feature_cols: list[str],
    predict_func,
    model_name: str,
    lag_table: pd.DataFrame,
    output_filename: str,
    seed: int = 20260529,
) -> pd.DataFrame:
    test = aligned[aligned["数据集"] == "测试集"].copy()
    x_test = test[feature_cols].to_numpy(dtype=float)
    y_test = test[TARGET_COL].to_numpy(dtype=float)
    base_pred = predict_func(x_test)
    base_rmse = float(np.sqrt(np.mean((base_pred - y_test) ** 2)))
    rng = np.random.default_rng(seed)
    lag_seconds = dict(zip(lag_table["字段名"], lag_table["最佳滞后秒数"]))
    corr_map = dict(zip(lag_table["字段名"], lag_table["训练集相关系数"]))
    rows = []

    for idx, feature in enumerate(feature_cols):
        x_perm = x_test.copy()
        x_perm[:, idx] = rng.permutation(x_perm[:, idx])
        pred_perm = predict_func(x_perm)
        rmse = float(np.sqrt(np.mean((pred_perm - y_test) ** 2)))
        base = base_feature_name(feature)
        rows.append(
            {
                "模型": model_name,
                "变量组": feature_group_name(feature),
                "原始字段": base,
                "模型特征": feature,
                "最佳滞后秒数": lag_seconds.get(base, np.nan),
                "训练集相关系数": corr_map.get(base, np.nan),
                "基准RMSE": base_rmse,
                "置换后RMSE": rmse,
                "RMSE增加量": rmse - base_rmse,
                "RMSE增加率_%": (rmse - base_rmse) / base_rmse * 100 if base_rmse else np.nan,
            }
        )

    table = pd.DataFrame(rows).sort_values("RMSE增加量", ascending=False)
    table.to_csv(RESULT_DATA_DIR / output_filename, index=False, encoding="utf-8-sig")
    return table


def make_permutation_importance(
    aligned: pd.DataFrame,
    forest: dict,
    lag_table: pd.DataFrame,
    output_filename: str,
    seed: int = 20260529,
) -> pd.DataFrame:
    test = aligned[aligned["数据集"] == "测试集"].copy()
    feature_cols = forest["feature_cols"]
    x_test = test[feature_cols].to_numpy(dtype=float)
    y_test = test[TARGET_COL].to_numpy(dtype=float)
    base_pred = predict_forest(forest, x_test)
    base_rmse = float(np.sqrt(np.mean((base_pred - y_test) ** 2)))
    rng = np.random.default_rng(seed)
    lag_seconds = dict(zip(lag_table["字段名"], lag_table["最佳滞后秒数"]))
    corr_map = dict(zip(lag_table["字段名"], lag_table["训练集相关系数"]))
    rows = []

    for idx, feature in enumerate(feature_cols):
        x_perm = x_test.copy()
        x_perm[:, idx] = rng.permutation(x_perm[:, idx])
        pred_perm = predict_forest(forest, x_perm)
        rmse = float(np.sqrt(np.mean((pred_perm - y_test) ** 2)))
        base = base_feature_name(feature)
        rows.append(
            {
                "模型": forest["model_name"],
                "变量组": feature_group_name(feature),
                "原始字段": base,
                "模型特征": feature,
                "最佳滞后秒数": lag_seconds.get(base, np.nan),
                "训练集相关系数": corr_map.get(base, np.nan),
                "基准RMSE": base_rmse,
                "置换后RMSE": rmse,
                "RMSE增加量": rmse - base_rmse,
                "RMSE增加率_%": (rmse - base_rmse) / base_rmse * 100 if base_rmse else np.nan,
            }
        )

    table = pd.DataFrame(rows).sort_values("RMSE增加量", ascending=False)
    table.to_csv(RESULT_DATA_DIR / output_filename, index=False, encoding="utf-8-sig")
    return table


def metrics(y_true: pd.Series, y_pred: pd.Series, high_threshold: float | None = None) -> dict:
    y_true = y_true.to_numpy(dtype=float)
    y_pred = y_pred.to_numpy(dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(err) / denom) * 100)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    within_5 = float(np.mean(np.abs(err) <= 0.05 * np.abs(y_true)) * 100)
    within_10 = float(np.mean(np.abs(err) <= 0.10 * np.abs(y_true)) * 100)
    bias = float(np.mean(err))

    true_trend = np.sign(np.diff(y_true))
    pred_trend = np.sign(np.diff(y_pred))
    trend_mask = true_trend != 0
    if trend_mask.sum() > 0:
        trend_accuracy = float(np.mean(true_trend[trend_mask] == pred_trend[trend_mask]) * 100)
    else:
        trend_accuracy = float("nan")

    high_count = 0
    high_mae = float("nan")
    high_rmse = float("nan")
    high_mape = float("nan")
    if high_threshold is not None and np.isfinite(high_threshold):
        high_mask = y_true >= high_threshold
        high_count = int(high_mask.sum())
        if high_count > 0:
            high_err = err[high_mask]
            high_y = y_true[high_mask]
            high_mae = float(np.mean(np.abs(high_err)))
            high_rmse = float(np.sqrt(np.mean(high_err**2)))
            high_denom = np.where(np.abs(high_y) < 1e-9, np.nan, np.abs(high_y))
            high_mape = float(np.nanmean(np.abs(high_err) / high_denom) * 100)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_%": mape,
        "R2": r2,
        "平均误差": bias,
        "趋势准确率_%": trend_accuracy,
        "5%误差内比例_%": within_5,
        "10%误差内比例_%": within_10,
        "高CO阈值_训练90分位": float(high_threshold) if high_threshold is not None else float("nan"),
        "高CO样本数": high_count,
        "高CO_MAE": high_mae,
        "高CO_RMSE": high_rmse,
        "高CO_MAPE_%": high_mape,
    }


def evaluate_models(predictions: pd.DataFrame, pred_cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    high_threshold = float(predictions.loc[predictions["数据集"] == "训练集", TARGET_COL].quantile(0.90))
    for split in ["训练集", "验证集", "测试集"]:
        subset = predictions[predictions["数据集"] == split]
        for model_label, pred_col in pred_cols.items():
            item = metrics(subset[TARGET_COL], subset[pred_col], high_threshold=high_threshold)
            item.update({"数据集": split, "模型": model_label, "样本数": int(len(subset))})
            rows.append(item)
    table = pd.DataFrame(rows)
    table = table[
        [
            "模型",
            "数据集",
            "样本数",
            "MAE",
            "RMSE",
            "MAPE_%",
            "R2",
            "平均误差",
            "趋势准确率_%",
            "5%误差内比例_%",
            "10%误差内比例_%",
            "高CO阈值_训练90分位",
            "高CO样本数",
            "高CO_MAE",
            "高CO_RMSE",
            "高CO_MAPE_%",
        ]
    ]
    table.to_csv(RESULT_DATA_DIR / "模型评价指标.csv", index=False, encoding="utf-8-sig")
    return table


def make_baseline_comparison(metrics_table: pd.DataFrame) -> pd.DataFrame:
    requested_order = [
        "基线1_CO_lag1持续性模型",
        "基线2_历史CO_AR模型",
        BASELINE3_MODEL_NAME,
    ]
    rows = []
    for split in ["训练集", "验证集", "测试集"]:
        subset = metrics_table[(metrics_table["数据集"] == split) & (metrics_table["模型"].isin(requested_order))].copy()
        if subset.empty:
            continue
        base = subset[subset["模型"] == "基线1_CO_lag1持续性模型"].iloc[0]
        for _, row in subset.iterrows():
            rows.append(
                {
                    "数据集": split,
                    "模型": row["模型"],
                    "样本数": int(row["样本数"]),
                    "MAE": row["MAE"],
                    "RMSE": row["RMSE"],
                    "MAPE_%": row["MAPE_%"],
                    "R2": row["R2"],
                    "趋势准确率_%": row["趋势准确率_%"],
                    "高CO阈值_训练90分位": row["高CO阈值_训练90分位"],
                    "高CO样本数": int(row["高CO样本数"]),
                    "高CO_MAE": row["高CO_MAE"],
                    "高CO_RMSE": row["高CO_RMSE"],
                    "高CO_MAPE_%": row["高CO_MAPE_%"],
                    "RMSE相对基线1变化": row["RMSE"] - base["RMSE"],
                    "RMSE相对基线1降低率_%": (base["RMSE"] - row["RMSE"]) / base["RMSE"] * 100 if base["RMSE"] else np.nan,
                    "MAE相对基线1降低率_%": (base["MAE"] - row["MAE"]) / base["MAE"] * 100 if base["MAE"] else np.nan,
                    "R2相对基线1提升": row["R2"] - base["R2"],
                }
            )
    table = pd.DataFrame(rows)
    table["模型排序"] = table["模型"].map({name: i for i, name in enumerate(requested_order)})
    table = table.sort_values(["数据集", "模型排序"]).drop(columns=["模型排序"])
    table.to_csv(RESULT_DATA_DIR / "基线模型比较.csv", index=False, encoding="utf-8-sig")
    return table


def make_nonlinear_monitor_comparison(metrics_table: pd.DataFrame, nonlinear_model_names: list[str]) -> pd.DataFrame:
    requested_order = [BASELINE3_MODEL_NAME] + nonlinear_model_names
    rows = []
    for split in ["训练集", "验证集", "测试集"]:
        subset = metrics_table[(metrics_table["数据集"] == split) & (metrics_table["模型"].isin(requested_order))].copy()
        if subset.empty or BASELINE3_MODEL_NAME not in set(subset["模型"]):
            continue
        base = subset[subset["模型"] == BASELINE3_MODEL_NAME].iloc[0]
        for _, row in subset.iterrows():
            rows.append(
                {
                    "数据集": split,
                    "模型": row["模型"],
                    "样本数": int(row["样本数"]),
                    "MAE": row["MAE"],
                    "RMSE": row["RMSE"],
                    "MAPE_%": row["MAPE_%"],
                    "R2": row["R2"],
                    "趋势准确率_%": row["趋势准确率_%"],
                    "高CO样本数": int(row["高CO样本数"]),
                    "高CO_RMSE": row["高CO_RMSE"],
                    "RMSE相对基线3变化": row["RMSE"] - base["RMSE"],
                    "RMSE相对基线3降低率_%": (base["RMSE"] - row["RMSE"]) / base["RMSE"] * 100 if base["RMSE"] else np.nan,
                    "MAE相对基线3降低率_%": (base["MAE"] - row["MAE"]) / base["MAE"] * 100 if base["MAE"] else np.nan,
                    "R2相对基线3提升": row["R2"] - base["R2"],
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["模型排序"] = table["模型"].map({name: i for i, name in enumerate(requested_order)})
        table = table.sort_values(["数据集", "模型排序"]).drop(columns=["模型排序"])
    table.to_csv(RESULT_DATA_DIR / "非线性监测模型比较.csv", index=False, encoding="utf-8-sig")
    return table


def make_process_model_comparison(metrics_table: pd.DataFrame, candidate_model_names: list[str]) -> pd.DataFrame:
    requested_order = ["线性过程变量模型_对比"] + candidate_model_names + ["过程变量模型"]
    requested_order = list(dict.fromkeys(requested_order))
    rows = []
    for split in ["训练集", "验证集", "测试集"]:
        subset = metrics_table[(metrics_table["数据集"] == split) & (metrics_table["模型"].isin(requested_order))].copy()
        if subset.empty:
            continue
        if "线性过程变量模型_对比" in set(subset["模型"]):
            base = subset[subset["模型"] == "线性过程变量模型_对比"].iloc[0]
        else:
            base = subset.iloc[0]
        for _, row in subset.iterrows():
            rows.append(
                {
                    "数据集": split,
                    "模型": row["模型"],
                    "样本数": int(row["样本数"]),
                    "MAE": row["MAE"],
                    "RMSE": row["RMSE"],
                    "MAPE_%": row["MAPE_%"],
                    "R2": row["R2"],
                    "趋势准确率_%": row["趋势准确率_%"],
                    "高CO样本数": int(row["高CO样本数"]),
                    "高CO_RMSE": row["高CO_RMSE"],
                    "RMSE相对线性过程变量模型变化": row["RMSE"] - base["RMSE"],
                    "RMSE相对线性过程变量模型降低率_%": (base["RMSE"] - row["RMSE"]) / base["RMSE"] * 100 if base["RMSE"] else np.nan,
                    "R2相对线性过程变量模型提升": row["R2"] - base["R2"],
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["模型排序"] = table["模型"].map({name: i for i, name in enumerate(requested_order)})
        table = table.sort_values(["数据集", "模型排序"]).drop(columns=["模型排序"])
    table.to_csv(RESULT_DATA_DIR / "过程变量模型比较.csv", index=False, encoding="utf-8-sig")
    return table


def base_feature_name(feature: str) -> str:
    if feature.startswith("CO历史浓度"):
        return feature
    if "__lag" in feature:
        return feature.split("__lag")[0]
    return feature


def feature_group_name(feature: str) -> str:
    if feature.startswith("CO历史浓度"):
        return "历史CO"
    base = base_feature_name(feature)
    for group, fields in get_feature_groups().items():
        if base in fields:
            return group
    return "其他"


def predict_from_model(frame: pd.DataFrame, model: dict) -> np.ndarray:
    feature_cols = model["特征字段"]
    x = frame[feature_cols].to_numpy(dtype=float)
    mu = np.array(model["标准化均值"], dtype=float)
    sigma = np.array(model["标准化标准差"], dtype=float)
    z = (x - mu) / sigma
    pred = np.full(len(frame), float(model["截距"]), dtype=float)
    pred += z @ np.array(model["线性系数"], dtype=float)
    if int(model.get("实际采用特征次数", 1)) == 2:
        pred += (z * z) @ np.array(model["平方项系数"], dtype=float)

    if "基线字段" in model:
        pred = frame[model["基线字段"]].to_numpy(dtype=float) + pred

    blend = model.get("融合说明")
    if blend:
        ridge_weight = float(blend["岭回归预测权重"])
        lag1_weight = float(blend["CO_lag1基线权重"])
        pred = ridge_weight * pred + lag1_weight * frame["CO历史浓度__lag1"].to_numpy(dtype=float)
    return pred


def make_group_ablation(aligned: pd.DataFrame, models: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model_name, model_entry in models.items():
        if "model" in model_entry and "predict_func" in model_entry:
            model = model_entry["model"]
            predict_func = model_entry["predict_func"]
        else:
            model = model_entry
            predict_func = lambda frame, current_model=model: predict_from_model(frame, current_model)
        train = aligned[aligned["数据集"] == "训练集"]
        test = aligned[aligned["数据集"] == "测试集"]
        feature_cols = model["特征字段"]
        full_pred = pd.Series(predict_func(test), index=test.index)
        full_metrics = metrics(test[TARGET_COL], full_pred)
        groups = sorted({feature_group_name(feature) for feature in feature_cols})
        train_means = train[feature_cols].mean()

        rows.append(
            {
                "模型": model_name,
                "变量组": "完整模型",
                "消融字段数": 0,
                "测试RMSE": full_metrics["RMSE"],
                "测试MAE": full_metrics["MAE"],
                "测试R2": full_metrics["R2"],
                "RMSE相对完整模型变化": 0.0,
                "说明": "完整模型基准",
            }
        )

        for group in groups:
            cols = [feature for feature in feature_cols if feature_group_name(feature) == group]
            if not cols:
                continue
            ablated = test.copy()
            for col in cols:
                ablated[col] = train_means[col]
            pred = pd.Series(predict_func(ablated), index=test.index)
            item = metrics(test[TARGET_COL], pred)
            rows.append(
                {
                    "模型": model_name,
                    "变量组": group,
                    "消融字段数": len(cols),
                    "测试RMSE": item["RMSE"],
                    "测试MAE": item["MAE"],
                    "测试R2": item["R2"],
                    "RMSE相对完整模型变化": item["RMSE"] - full_metrics["RMSE"],
                    "说明": "将该组输入替换为训练集均值后重新预测；RMSE增加越多，说明该组对该模型越重要。",
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(RESULT_DATA_DIR / "变量组消融分析.csv", index=False, encoding="utf-8-sig")
    return table


def make_sensitivity_analysis(aligned: pd.DataFrame, model: dict, lag_table: pd.DataFrame, output_filename: str, predict_func=None) -> pd.DataFrame:
    train = aligned[aligned["数据集"] == "训练集"]
    test = aligned[aligned["数据集"] == "测试集"]
    if predict_func is None:
        predict_func = lambda frame: predict_from_model(frame, model)
    base_pred = predict_func(test)
    lag_seconds = dict(zip(lag_table["字段名"], lag_table["最佳滞后秒数"]))
    corr_map = dict(zip(lag_table["字段名"], lag_table["训练集相关系数"]))
    rows = []

    for feature in model["特征字段"]:
        group = feature_group_name(feature)
        if group == "历史CO":
            continue
        std = float(train[feature].std())
        if not np.isfinite(std) or std == 0:
            continue
        plus = test.copy()
        minus = test.copy()
        plus[feature] = plus[feature] + std
        minus[feature] = minus[feature] - std
        plus_pred = predict_func(plus)
        minus_pred = predict_func(minus)
        base = base_feature_name(feature)
        rows.append(
            {
                "模型": model["模型名称"],
                "变量组": group,
                "原始字段": base,
                "模型特征": feature,
                "最佳滞后秒数": lag_seconds.get(base, np.nan),
                "训练集相关系数": corr_map.get(base, np.nan),
                "训练集标准差": std,
                "增加1个历史标准差后的平均CO变化": float(np.mean(plus_pred - base_pred)),
                "减少1个历史标准差后的平均CO变化": float(np.mean(minus_pred - base_pred)),
                "中心差分敏感度_mg每原单位": float(np.mean((plus_pred - minus_pred) / (2 * std))),
            }
        )

    table = pd.DataFrame(rows).sort_values("增加1个历史标准差后的平均CO变化", key=lambda s: np.abs(s), ascending=False)
    table.to_csv(RESULT_DATA_DIR / output_filename, index=False, encoding="utf-8-sig")
    return table


def run_lag_validation_scenario(clean_df: pd.DataFrame, lag_table: pd.DataFrame, scenario_name: str) -> list[dict]:
    aligned, process_lagged_features, co_features = make_aligned_dataset(clean_df, lag_table, output_path=None)
    aligned = split_by_time(aligned)
    monitor_features = process_lagged_features + co_features

    process_suite_models, process_suite_predictions, process_suite_infos = fit_nonlinear_process_suite(
        aligned,
        process_lagged_features,
        seed=20260531 + len(scenario_name),
    )
    _, pred_process, process_info, _ = select_best_nonlinear_model(
        process_suite_models,
        process_suite_predictions,
        process_suite_infos,
        f"{scenario_name}_过程变量模型",
    )
    monitor_ridge_model, pred_monitor_ridge, _ = fit_ridge_model(
        aligned,
        monitor_features,
        f"{scenario_name}_监测型模型",
    )
    pred_monitor, monitor_blend_info = blend_monitor_prediction(aligned, pred_monitor_ridge)
    nonlinear_monitor_suite_models, nonlinear_monitor_suite_predictions, nonlinear_monitor_suite_infos = fit_nonlinear_monitor_suite(
        aligned,
        monitor_features,
        seed=20260601 + len(scenario_name),
    )
    _, pred_nonlinear_monitor, nonlinear_monitor_info, _ = select_best_nonlinear_model(
        nonlinear_monitor_suite_models,
        nonlinear_monitor_suite_predictions,
        nonlinear_monitor_suite_infos,
        f"{scenario_name}_非线性监测型模型",
    )

    test = aligned[aligned["数据集"] == "测试集"].copy()
    pred_process_test = pd.Series(pred_process.loc[test.index], index=test.index)
    pred_monitor_test = pd.Series(pred_monitor.loc[test.index], index=test.index)
    pred_nonlinear_monitor_test = pd.Series(pred_nonlinear_monitor.loc[test.index], index=test.index)
    rows = []
    model_predictions = [
        ("过程变量模型", pred_process_test),
        ("监测型模型", pred_monitor_test),
        ("非线性监测型模型", pred_nonlinear_monitor_test),
    ]
    for model_type, pred in model_predictions:
        item = metrics(test[TARGET_COL], pred)
        rows.append(
            {
                "实验场景": scenario_name,
                "模型类型": model_type,
                "样本数": int(len(test)),
                "MAE": item["MAE"],
                "RMSE": item["RMSE"],
                "MAPE_%": item["MAPE_%"],
                "R2": item["R2"],
                "平均最佳滞后_s": float(lag_table["最佳滞后秒数"].mean()),
                "最大已选滞后_s": float(lag_table["最佳滞后秒数"].max()),
                "非零滞后变量数": int((lag_table["最佳滞后秒数"] > 0).sum()),
                "监测型岭回归权重": monitor_blend_info["ridge_weight"] if model_type == "监测型模型" else np.nan,
                "监测型CO_lag1权重": monitor_blend_info["lag1_weight"] if model_type == "监测型模型" else np.nan,
                "过程变量候选模型": process_info["selected_model_name"] if model_type == "过程变量模型" else "",
                "非线性监测候选模型": nonlinear_monitor_info["selected_model_name"] if model_type == "非线性监测型模型" else "",
                "输入变量口径": "过程变量" if model_type == "过程变量模型" else "过程变量+历史CO",
            }
        )
    return rows


def make_lag_validation_experiments(
    clean_df: pd.DataFrame,
    process_features: list[str],
    main_lag_table: pd.DataFrame,
    physical_lag_table: pd.DataFrame,
    local_search_lag_table: pd.DataFrame,
    lag_prior_table: pd.DataFrame,
    train_end: int,
) -> pd.DataFrame:
    no_lag_table = make_no_lag_table(clean_df, process_features, train_end)
    no_lag_table.to_csv(MID_DIR / "无时滞配准表.csv", index=False, encoding="utf-8-sig")

    rows = []
    rows.extend(run_lag_validation_scenario(clean_df, no_lag_table, "无时滞模型_lag0"))
    rows.extend(run_lag_validation_scenario(clean_df, physical_lag_table, "物理初始时滞截断_对比模型"))
    rows.extend(run_lag_validation_scenario(clean_df, main_lag_table, "分组时滞组合优化_主模型"))
    rows.extend(run_lag_validation_scenario(clean_df, local_search_lag_table, "单变量局部搜索_对比模型"))

    table = pd.DataFrame(rows)
    for model_type in table["模型类型"].unique():
        mask = table["模型类型"] == model_type
        base = table[mask & (table["实验场景"] == "无时滞模型_lag0")].iloc[0]
        table.loc[mask, "RMSE相对无时滞降低率_%"] = (base["RMSE"] - table.loc[mask, "RMSE"]) / base["RMSE"] * 100
        table.loc[mask, "R2相对无时滞提升"] = table.loc[mask, "R2"] - base["R2"]

    table.to_csv(RESULT_DATA_DIR / "时滞合理性验证.csv", index=False, encoding="utf-8-sig")
    nonlinear_table = table[table["模型类型"] == "非线性监测型模型"].copy()
    nonlinear_table.to_csv(RESULT_DATA_DIR / "非线性过程变量历史CO实验.csv", index=False, encoding="utf-8-sig")
    return table


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Microsoft YaHei,sans-serif;font-size:13px;fill:#263238}.title{font-size:18px;font-weight:700}.small{font-size:11px;fill:#546E7A}.axis{stroke:#607D8B;stroke-width:1}.grid{stroke:#ECEFF1;stroke-width:1}.line1{fill:none;stroke:#1565C0;stroke-width:2}.line2{fill:none;stroke:#EF6C00;stroke-width:2}.line3{fill:none;stroke:#2E7D32;stroke-width:2}.bar{fill:#26A69A}.dot1{fill:#1565C0;opacity:.68}.dot2{fill:#EF6C00;opacity:.62}</style>',
    ]


def scale(values: np.ndarray, lo: float, hi: float, out_lo: float, out_hi: float) -> np.ndarray:
    if hi == lo:
        return np.full_like(values, (out_lo + out_hi) / 2, dtype=float)
    return out_lo + (values - lo) / (hi - lo) * (out_hi - out_lo)


def make_line_plot(path: Path, df: pd.DataFrame, title: str) -> None:
    width, height = 1050, 520
    left, right, top, bottom = 72, 32, 56, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    data = df.copy()
    max_points = 360
    if len(data) > max_points:
        idx = np.linspace(0, len(data) - 1, max_points).astype(int)
        data = data.iloc[idx]
    x = np.arange(len(data), dtype=float)
    cols = [TARGET_COL, "过程变量模型_预测CO浓度", "监测型模型_预测CO浓度"]
    y_all = np.concatenate([data[c].to_numpy(dtype=float) for c in cols])
    y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    pad = (y_max - y_min) * 0.08 or 1
    y_min -= pad
    y_max += pad
    xs = scale(x, 0, max(len(data) - 1, 1), left, left + plot_w)

    parts = svg_header(width, height)
    parts.append(f'<text x="{left}" y="30" class="title">{title}</text>')
    for k in range(6):
        yy = top + plot_h * k / 5
        val = y_max - (y_max - y_min) * k / 5
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="12" y="{yy + 4:.1f}" class="small">{val:.0f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')

    for cls, col in zip(["line1", "line2", "line3"], cols):
        ys = scale(data[col].to_numpy(dtype=float), y_min, y_max, top + plot_h, top)
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
        parts.append(f'<polyline points="{pts}" class="{cls}"/>')
    parts.append(f'<text x="{left}" y="{height - 25}" class="small">Sequence in test period</text>')
    parts.append(f'<text x="{width - 420}" y="30" class="small">blue=actual  orange=process-only  green=monitoring</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_scatter_plot(path: Path, df: pd.DataFrame, title: str) -> None:
    width, height = 620, 560
    left, right, top, bottom = 72, 44, 56, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    data = df.copy()
    if len(data) > 600:
        idx = np.linspace(0, len(data) - 1, 600).astype(int)
        data = data.iloc[idx]
    cols = [TARGET_COL, "过程变量模型_预测CO浓度", "监测型模型_预测CO浓度"]
    v = np.concatenate([data[c].to_numpy(dtype=float) for c in cols])
    v_min, v_max = float(np.nanmin(v)), float(np.nanmax(v))
    pad = (v_max - v_min) * 0.08 or 1
    v_min -= pad
    v_max += pad

    parts = svg_header(width, height)
    parts.append(f'<text x="{left}" y="30" class="title">{title}</text>')
    for k in range(6):
        xx = left + plot_w * k / 5
        yy = top + plot_h * k / 5
        val = v_min + (v_max - v_min) * k / 5
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top + plot_h}" class="grid"/>')
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{xx - 18:.1f}" y="{top + plot_h + 20}" class="small">{val:.0f}</text>')
        parts.append(f'<text x="12" y="{top + plot_h - plot_h * k / 5 + 4:.1f}" class="small">{val:.0f}</text>')
    diag_x1, diag_y1 = left, top + plot_h
    diag_x2, diag_y2 = left + plot_w, top
    parts.append(f'<line x1="{diag_x1}" y1="{diag_y1}" x2="{diag_x2}" y2="{diag_y2}" stroke="#90A4AE" stroke-width="1.4" stroke-dasharray="5 5"/>')
    x_actual = scale(data[TARGET_COL].to_numpy(dtype=float), v_min, v_max, left, left + plot_w)
    for pred_col, cls in [("过程变量模型_预测CO浓度", "dot2"), ("监测型模型_预测CO浓度", "dot1")]:
        y_pred = scale(data[pred_col].to_numpy(dtype=float), v_min, v_max, top + plot_h, top)
        for x, y in zip(x_actual, y_pred):
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" class="{cls}"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<text x="{left + plot_w - 190}" y="{top + plot_h + 42}" class="small">Actual CO</text>')
    parts.append(f'<text x="{width - 270}" y="30" class="small">orange=process-only blue=monitoring</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_bar_plot(path: Path, lag_table: pd.DataFrame, title: str) -> None:
    width, height = 1120, 620
    left, right, top, bottom = 82, 34, 58, 150
    plot_w = width - left - right
    plot_h = height - top - bottom
    data = lag_table.copy().reset_index(drop=True)
    y_max = max(float(data["最佳滞后秒数"].max()), 1.0)
    bar_w = plot_w / len(data) * 0.72
    parts = svg_header(width, height)
    parts.append(f'<text x="{left}" y="30" class="title">{title}</text>')
    for k in range(6):
        yy = top + plot_h * k / 5
        val = y_max - y_max * k / 5
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="20" y="{yy + 4:.1f}" class="small">{val:.0f}s</text>')
    for i, row in data.iterrows():
        cx = left + plot_w * (i + 0.5) / len(data)
        h = plot_h * float(row["最佳滞后秒数"]) / y_max
        x = cx - bar_w / 2
        y = top + plot_h - h
        color = {
            "机速": "#8E24AA",
            "风箱负压强度": "#26A69A",
            "风箱平均温度": "#EF6C00",
            "大烟道负压强度": "#1565C0",
            "大烟道温度": "#C62828",
        }.get(row["变量组"], "#78909C")
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        if i % 2 == 0:
            label = row["字段名"].replace("风箱", "").replace("平均温度", "T").replace("负压", "P")
            parts.append(f'<text x="{cx - 22:.1f}" y="{height - 102}" transform="rotate(55 {cx - 22:.1f},{height - 102})" class="small">{label}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<text x="{width - 620}" y="30" class="small">physical-prior local search; bars show selected lag</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_importance_plot(path: Path, importance: pd.DataFrame, title: str) -> None:
    width, height = 880, 620
    left, right, top, bottom = 290, 44, 56, 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    data = importance.head(18).iloc[::-1].reset_index(drop=True)
    max_v = max(float(data["综合影响强度"].max()), 1.0)
    row_h = plot_h / len(data)
    parts = svg_header(width, height)
    parts.append(f'<text x="{left}" y="30" class="title">{title}</text>')
    for i, row in data.iterrows():
        y = top + i * row_h + row_h * 0.18
        h = row_h * 0.62
        w = plot_w * float(row["综合影响强度"]) / max_v
        color = "#1565C0" if row["线性影响方向"] == "正向" else "#EF6C00"
        label = str(row["原始字段"]).replace("__lag", " lag")
        if len(label) > 28:
            label = label[:27] + "..."
        parts.append(f'<text x="12" y="{y + h * 0.72:.1f}" class="small">{label}</text>')
        parts.append(f'<rect x="{left}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{color}" opacity="0.84"/>')
        parts.append(f'<text x="{left + w + 6:.1f}" y="{y + h * 0.72:.1f}" class="small">{row["综合影响强度"]:.1f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<text x="{width - 250}" y="30" class="small">blue=positive orange=negative</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def interpolate_color(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> str:
    t = max(0.0, min(1.0, float(t)))
    r = round(c1[0] + (c2[0] - c1[0]) * t)
    g = round(c1[1] + (c2[1] - c1[1]) * t)
    b = round(c1[2] + (c2[2] - c1[2]) * t)
    return f"#{r:02X}{g:02X}{b:02X}"


def diverging_color(value: float, max_abs: float) -> str:
    if not np.isfinite(value) or max_abs <= 0:
        return "#ECEFF1"
    t = min(abs(float(value)) / max_abs, 1.0)
    if value >= 0:
        return interpolate_color((255, 245, 238), (198, 40, 40), t)
    return interpolate_color((239, 247, 255), (21, 101, 192), t)


def make_windbox_lag_position_plot(path: Path, lag_table: pd.DataFrame, title: str) -> None:
    width, height = 980, 560
    left, right, top, bottom = 82, 48, 62, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    data = lag_table[lag_table["变量组"].isin(["风箱负压强度", "风箱平均温度"])].copy()
    data["风箱编号_num"] = pd.to_numeric(data["风箱编号"], errors="coerce")
    data["距离_m"] = pd.to_numeric(data["到检测参考点距离_m"], errors="coerce")
    data["滞后_s"] = pd.to_numeric(data["最佳滞后秒数"], errors="coerce")
    data = data.dropna(subset=["风箱编号_num", "距离_m", "滞后_s"])
    x_min, x_max = 0.0, max(float(data["距离_m"].max()), 1.0)
    y_min, y_max = 0.0, max(float(data["滞后_s"].max()), 1.0)
    y_max = math.ceil(y_max / 100) * 100

    parts = svg_header(width, height)
    parts.append(f'<text x="{left}" y="32" class="title">{title}</text>')
    parts.append(f'<text x="{left}" y="52" class="small">横轴为风箱中心到检测参考点距离；距离越大，物理传播滞后通常越长</text>')
    for k in range(6):
        yy = top + plot_h * k / 5
        val = y_max - y_max * k / 5
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="18" y="{yy + 4:.1f}" class="small">{val:.0f}s</text>')
        xx = left + plot_w * k / 5
        xval = x_min + (x_max - x_min) * k / 5
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top + plot_h}" class="grid"/>')
        parts.append(f'<text x="{xx - 14:.1f}" y="{top + plot_h + 22}" class="small">{xval:.0f}m</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')

    for group, color in [("风箱负压强度", "#26A69A"), ("风箱平均温度", "#EF6C00")]:
        sub = data[data["变量组"] == group].sort_values("距离_m")
        xs = scale(sub["距离_m"].to_numpy(dtype=float), x_min, x_max, left, left + plot_w)
        ys = scale(sub["滞后_s"].to_numpy(dtype=float), y_min, y_max, top + plot_h, top)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for x, y, box_no in zip(xs, ys, sub["风箱编号_num"].astype(int)):
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}" opacity="0.9"/>')
            if box_no in {1, 6, 12, 18}:
                parts.append(f'<text x="{x + 5:.1f}" y="{y - 7:.1f}" class="small">{box_no}#</text>')
    parts.append(f'<text x="{left + plot_w - 230}" y="{top + 18}" class="small" fill="#26A69A">绿色：风箱负压强度</text>')
    parts.append(f'<text x="{left + plot_w - 230}" y="{top + 38}" class="small" fill="#EF6C00">橙色：风箱平均温度</text>')
    parts.append(f'<text x="{left + plot_w - 132}" y="{height - 18}" class="small">距离检测点/m</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_windbox_direction_heatmap(path: Path, sensitivity: pd.DataFrame, group_name: str, title: str) -> None:
    width, height = 1050, 260
    left, top = 72, 76
    cell_w, cell_h = 48, 54
    data = sensitivity[sensitivity["变量组"] == group_name].copy()
    data["风箱编号"] = data["原始字段"].map(windbox_no_from_feature)
    value_col = "增加1个历史标准差后的平均CO变化"
    values = dict(zip(data["风箱编号"], data[value_col]))
    max_abs = max([abs(float(v)) for v in values.values() if np.isfinite(v)] + [1.0])

    parts = svg_header(width, height)
    parts.append(f'<text x="{left}" y="32" class="title">{title}</text>')
    parts.append(f'<text x="{left}" y="54" class="small">数值为该变量升高 1 个训练集标准差后，过程变量模型预测 CO 的平均变化；红色升高，蓝色降低</text>')
    for i in range(1, 19):
        x = left + (i - 1) * cell_w
        value = float(values.get(i, np.nan))
        color = diverging_color(value, max_abs)
        parts.append(f'<rect x="{x:.1f}" y="{top}" width="{cell_w - 4}" height="{cell_h}" fill="{color}" stroke="#CFD8DC"/>')
        parts.append(f'<text x="{x + 13:.1f}" y="{top - 10}" class="small">{i}#</text>')
        label = "" if not np.isfinite(value) else f"{value:.0f}"
        parts.append(f'<text x="{x + 10:.1f}" y="{top + 32}" class="small">{label}</text>')
    legend_x = left
    legend_y = top + cell_h + 42
    for k in range(13):
        t = k / 12
        val = -max_abs + 2 * max_abs * t
        parts.append(f'<rect x="{legend_x + k * 24}" y="{legend_y}" width="24" height="12" fill="{diverging_color(val, max_abs)}"/>')
    parts.append(f'<text x="{legend_x}" y="{legend_y + 30}" class="small">降低 CO</text>')
    parts.append(f'<text x="{legend_x + 132}" y="{legend_y + 30}" class="small">0</text>')
    parts.append(f'<text x="{legend_x + 246}" y="{legend_y + 30}" class="small">升高 CO</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_permutation_importance_plot(path: Path, permutation_table: pd.DataFrame, title: str) -> None:
    width, height = 960, 650
    left, right, top, bottom = 300, 70, 62, 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    data = permutation_table.head(18).iloc[::-1].reset_index(drop=True)
    max_v = max(float(data["RMSE增加量"].max()), 1e-9)
    row_h = plot_h / max(len(data), 1)
    colors = {
        "机速": "#8E24AA",
        "风箱负压强度": "#26A69A",
        "风箱平均温度": "#EF6C00",
        "大烟道负压强度": "#1565C0",
        "大烟道温度": "#C62828",
    }

    parts = svg_header(width, height)
    parts.append(f'<text x="{left}" y="32" class="title">{title}</text>')
    parts.append(f'<text x="{left}" y="52" class="small">基于正式过程变量模型：置换某变量后 RMSE 增加越多，说明该变量越重要</text>')
    for i, row in data.iterrows():
        y = top + i * row_h + row_h * 0.18
        h = row_h * 0.62
        value = max(float(row["RMSE增加量"]), 0.0)
        w = plot_w * value / max_v
        color = colors.get(row["变量组"], "#78909C")
        label = str(row["原始字段"])
        if len(label) > 26:
            label = label[:25] + "..."
        parts.append(f'<text x="18" y="{y + h * 0.72:.1f}" class="small">{label}</text>')
        parts.append(f'<rect x="{left}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{color}" opacity="0.86"/>')
        parts.append(f'<text x="{left + w + 6:.1f}" y="{y + h * 0.72:.1f}" class="small">+{value:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<text x="{width - 190}" y="{height - 18}" class="small">RMSE increase</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_group_ablation_plot(path: Path, ablation_table: pd.DataFrame, title: str) -> None:
    width, height = 820, 420
    left, right, top, bottom = 210, 70, 62, 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    data = ablation_table[(ablation_table["模型"] == "过程变量模型") & (ablation_table["变量组"] != "完整模型")].copy()
    data = data.sort_values("RMSE相对完整模型变化").reset_index(drop=True)
    values = data["RMSE相对完整模型变化"].to_numpy(dtype=float)
    max_abs = max(float(np.nanmax(np.abs(values))), 1.0)
    row_h = plot_h / max(len(data), 1)
    zero_x = left + plot_w / 2

    parts = svg_header(width, height)
    parts.append(f'<text x="{left}" y="32" class="title">{title}</text>')
    parts.append(f'<text x="{left}" y="52" class="small">将整组变量替换为训练集均值；RMSE 增加为正表示该组对过程变量模型贡献更大</text>')
    parts.append(f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{top + plot_h}" stroke="#455A64" stroke-width="1.2"/>')
    for i, row in data.iterrows():
        y = top + i * row_h + row_h * 0.22
        h = row_h * 0.56
        value = float(row["RMSE相对完整模型变化"])
        w = (plot_w / 2) * abs(value) / max_abs
        x = zero_x if value >= 0 else zero_x - w
        color = "#C62828" if value >= 0 else "#1565C0"
        parts.append(f'<text x="18" y="{y + h * 0.72:.1f}" class="small">{row["变量组"]}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{color}" opacity="0.84"/>')
        tx = x + w + 6 if value >= 0 else x - 58
        parts.append(f'<text x="{tx:.1f}" y="{y + h * 0.72:.1f}" class="small">{value:+.1f}</text>')
    parts.append(f'<text x="{left}" y="{height - 18}" class="small">蓝色：消融后 RMSE 降低或无贡献；红色：消融后 RMSE 增加</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_sensitivity_plot(path: Path, sensitivity: pd.DataFrame, title: str) -> None:
    width, height = 980, 720
    left, right, top, bottom = 320, 70, 64, 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    data = sensitivity.copy()
    data["响应强度"] = data[
        ["增加1个历史标准差后的平均CO变化", "减少1个历史标准差后的平均CO变化"]
    ].abs().max(axis=1)
    data = data.sort_values("响应强度", ascending=False).head(12).iloc[::-1].reset_index(drop=True)
    max_abs = max(
        float(
            np.nanmax(
                np.abs(
                    data[
                        ["增加1个历史标准差后的平均CO变化", "减少1个历史标准差后的平均CO变化"]
                    ].to_numpy(dtype=float)
                )
            )
        ),
        1.0,
    )
    row_h = plot_h / max(len(data), 1)
    zero_x = left + plot_w / 2

    parts = svg_header(width, height)
    parts.append(f'<text x="{left}" y="32" class="title">{title}</text>')
    parts.append(f'<text x="{left}" y="52" class="small">深色条：变量升高 1σ；浅色条：变量降低 1σ；横向位置表示 CO 预测变化</text>')
    parts.append(f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{top + plot_h}" stroke="#455A64" stroke-width="1.2"/>')
    for i, row in data.iterrows():
        y0 = top + i * row_h + row_h * 0.16
        label = str(row["原始字段"])
        if len(label) > 28:
            label = label[:27] + "..."
        parts.append(f'<text x="18" y="{y0 + row_h * 0.52:.1f}" class="small">{label}</text>')
        for j, (col, opacity) in enumerate(
            [
                ("增加1个历史标准差后的平均CO变化", 0.88),
                ("减少1个历史标准差后的平均CO变化", 0.42),
            ]
        ):
            value = float(row[col])
            w = (plot_w / 2) * abs(value) / max_abs
            x = zero_x if value >= 0 else zero_x - w
            y = y0 + j * row_h * 0.34
            h = row_h * 0.24
            color = "#C62828" if value >= 0 else "#1565C0"
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{color}" opacity="{opacity}"/>')
            tx = x + w + 5 if value >= 0 else x - 56
            if j == 0:
                parts.append(f'<text x="{tx:.1f}" y="{y + h * 0.85:.1f}" class="small">{value:+.0f}</text>')
    parts.append(f'<text x="{left}" y="{height - 18}" class="small">蓝色表示 CO 下降，红色表示 CO 上升；浅色条给出变量降低时的反向响应</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def save_models(
    process_model: dict,
    linear_process_model: dict,
    process_suite_models: dict[str, dict],
    ar_model: dict,
    monitor_model: dict,
    residual_monitor_model: dict,
    nonlinear_process_model: dict,
    nonlinear_monitor_model: dict,
    nonlinear_monitor_suite_models: dict[str, dict],
    lag_table: pd.DataFrame,
) -> None:
    payload_process = dict(process_model)
    payload_process["滞后配准表"] = lag_table.to_dict(orient="records")
    payload_linear_process = dict(linear_process_model)
    payload_linear_process["滞后配准表"] = lag_table.to_dict(orient="records")
    payload_process_suite = {}
    for model_name, model in process_suite_models.items():
        payload = dict(model)
        payload["滞后配准表"] = lag_table.to_dict(orient="records")
        payload_process_suite[model_name] = payload
    payload_ar = dict(ar_model)
    payload_ar["滞后配准表"] = []
    payload_monitor = dict(monitor_model)
    payload_monitor["滞后配准表"] = lag_table.to_dict(orient="records")
    payload_residual_monitor = dict(residual_monitor_model)
    payload_residual_monitor["滞后配准表"] = lag_table.to_dict(orient="records")
    payload_nonlinear_process = dict(nonlinear_process_model)
    payload_nonlinear_process["滞后配准表"] = lag_table.to_dict(orient="records")
    payload_nonlinear_monitor = dict(nonlinear_monitor_model)
    payload_nonlinear_monitor["滞后配准表"] = lag_table.to_dict(orient="records")
    payload_nonlinear_monitor_suite = {}
    for model_name, model in nonlinear_monitor_suite_models.items():
        payload = dict(model)
        payload["滞后配准表"] = lag_table.to_dict(orient="records")
        payload_nonlinear_monitor_suite[model_name] = payload
    with open(RESULT_DIR / "CO预测模型_过程变量模型.json", "w", encoding="utf-8") as f:
        json.dump(payload_process, f, ensure_ascii=False, indent=2)
    with open(RESULT_DIR / "CO预测模型_线性过程变量模型_对比.json", "w", encoding="utf-8") as f:
        json.dump(payload_linear_process, f, ensure_ascii=False, indent=2)
    for model_name, payload in payload_process_suite.items():
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        with open(RESULT_DIR / f"CO预测模型_{safe_name}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(RESULT_DIR / "CO预测模型_过程变量模型集合.json", "w", encoding="utf-8") as f:
        json.dump(payload_process_suite, f, ensure_ascii=False, indent=2)
    with open(RESULT_DIR / "CO预测模型_历史CO_AR模型.json", "w", encoding="utf-8") as f:
        json.dump(payload_ar, f, ensure_ascii=False, indent=2)
    with open(RESULT_DIR / "CO预测模型_监测型模型.json", "w", encoding="utf-8") as f:
        json.dump(payload_monitor, f, ensure_ascii=False, indent=2)
    with open(RESULT_DIR / "CO预测模型_残差校正监测型模型_对比.json", "w", encoding="utf-8") as f:
        json.dump(payload_residual_monitor, f, ensure_ascii=False, indent=2)
    with open(RESULT_DIR / "CO预测模型_非线性过程变量模型.json", "w", encoding="utf-8") as f:
        json.dump(payload_nonlinear_process, f, ensure_ascii=False, indent=2)
    with open(RESULT_DIR / "CO预测模型_非线性监测型模型.json", "w", encoding="utf-8") as f:
        json.dump(payload_nonlinear_monitor, f, ensure_ascii=False, indent=2)
    for model_name, payload in payload_nonlinear_monitor_suite.items():
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        with open(RESULT_DIR / f"CO预测模型_{safe_name}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(RESULT_DIR / "CO预测模型_非线性监测模型集合.json", "w", encoding="utf-8") as f:
        json.dump(payload_nonlinear_monitor_suite, f, ensure_ascii=False, indent=2)


def summarize_lag(lag_table: pd.DataFrame) -> pd.DataFrame:
    summary = (
        lag_table.groupby("变量组")
        .agg(
            变量数=("字段名", "count"),
            平均最佳滞后_s=("最佳滞后秒数", "mean"),
            中位最佳滞后_s=("最佳滞后秒数", "median"),
            最大最佳滞后_s=("最佳滞后秒数", "max"),
            平均绝对相关=("训练集相关系数", lambda s: float(np.mean(np.abs(s)))),
        )
        .reset_index()
    )
    summary.to_csv(RESULT_DATA_DIR / "变量组滞后摘要.csv", index=False, encoding="utf-8-sig")
    return summary


def dataframe_to_markdown(df: pd.DataFrame, float_digits: int = 4) -> str:
    def fmt(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def make_markdown_tables(metrics_table: pd.DataFrame, lag_summary: pd.DataFrame, top_importance: pd.DataFrame) -> tuple[str, str, str]:
    test_metrics = metrics_table[metrics_table["数据集"] == "测试集"].copy()
    metric_cols = [
        "模型",
        "样本数",
        "MAE",
        "RMSE",
        "MAPE_%",
        "R2",
        "趋势准确率_%",
        "高CO样本数",
        "高CO_RMSE",
        "高CO_MAPE_%",
    ]
    metric_md = dataframe_to_markdown(test_metrics[metric_cols], float_digits=4)
    lag_md = dataframe_to_markdown(lag_summary, float_digits=2)
    imp_cols = ["原始字段", "最佳滞后秒数", "训练集相关系数", "综合影响强度", "线性影响方向"]
    imp_md = dataframe_to_markdown(top_importance[imp_cols].head(12), float_digits=4)
    return metric_md, lag_md, imp_md


def strip_lag_name(name: str) -> str:
    return name.split("__lag")[0].replace("__rolling30_lag1", "").replace("__rolling120_lag1", "")


def write_docs(
    metrics_table: pd.DataFrame,
    process_model_comparison: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    nonlinear_monitor_comparison: pd.DataFrame,
    lag_validation_table: pd.DataFrame,
    lag_table: pd.DataFrame,
    lag_summary: pd.DataFrame,
    importance_process: pd.DataFrame,
    predictions: pd.DataFrame,
    cleaning_report: dict,
    process_model: dict,
    monitor_model: dict,
    ablation_table: pd.DataFrame,
    sensitivity_process_table: pd.DataFrame,
    sensitivity_monitor_table: pd.DataFrame,
    nonlinear_permutation_table: pd.DataFrame,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    variant_name = cleaning_report.get("建模口径", "单口径建模")
    outlier_report = cleaning_report.get("CO异常修正", {"启用": False, "修正点数": 0, "修正比例_%": 0.0})
    if outlier_report.get("启用"):
        outlier_note = (
            f"- CO异常修正：启用；共修正 {outlier_report.get('修正点数', 0)} 个 CO 点，"
            f"占 {outlier_report.get('修正比例_%', 0.0):.3f}%；不删除样本，采用相邻正常点线性插值修正。"
        )
    else:
        outlier_note = "- CO异常修正：未启用；本口径保留原始 CO 序列，不删除也不修正样本。"
    metric_md, lag_md, imp_md = make_markdown_tables(metrics_table, lag_summary, importance_process)
    process_compare_test = process_model_comparison[process_model_comparison["数据集"] == "测试集"].copy()
    process_compare_cols = [
        "模型",
        "RMSE",
        "MAPE_%",
        "R2",
        "RMSE相对线性过程变量模型变化",
        "RMSE相对线性过程变量模型降低率_%",
        "R2相对线性过程变量模型提升",
    ]
    process_compare_md = dataframe_to_markdown(process_compare_test[process_compare_cols], float_digits=4)
    baseline_test = baseline_comparison[baseline_comparison["数据集"] == "测试集"].copy()
    baseline_cols = [
        "模型",
        "RMSE相对基线1变化",
        "RMSE相对基线1降低率_%",
        "MAE相对基线1降低率_%",
        "R2相对基线1提升",
    ]
    baseline_md = dataframe_to_markdown(baseline_test[baseline_cols], float_digits=4)
    nonlinear_monitor_test_compare = nonlinear_monitor_comparison[nonlinear_monitor_comparison["数据集"] == "测试集"].copy()
    nonlinear_monitor_cols = [
        "模型",
        "RMSE",
        "MAPE_%",
        "R2",
        "RMSE相对基线3变化",
        "RMSE相对基线3降低率_%",
        "R2相对基线3提升",
    ]
    nonlinear_monitor_md = dataframe_to_markdown(nonlinear_monitor_test_compare[nonlinear_monitor_cols], float_digits=4)
    lag_validation_md = dataframe_to_markdown(
        lag_validation_table[
            [
                "实验场景",
                "模型类型",
                "样本数",
                "RMSE",
                "MAPE_%",
                "R2",
                "平均最佳滞后_s",
                "最大已选滞后_s",
                "RMSE相对无时滞降低率_%",
            ]
        ],
        float_digits=4,
    )
    nonlinear_lag_validation_table = lag_validation_table[lag_validation_table["模型类型"] == "非线性监测型模型"].copy()
    nonlinear_lag_validation_md = dataframe_to_markdown(
        nonlinear_lag_validation_table[
            [
                "实验场景",
                "样本数",
                "RMSE",
                "MAPE_%",
                "R2",
                "平均最佳滞后_s",
                "最大已选滞后_s",
                "非线性监测候选模型",
                "RMSE相对无时滞降低率_%",
            ]
        ],
        float_digits=4,
    )
    main_monitor_row = lag_validation_table[
        (lag_validation_table["实验场景"] == "分组时滞组合优化_主模型") & (lag_validation_table["模型类型"] == "监测型模型")
    ].iloc[0]
    main_nonlinear_lag_row = lag_validation_table[
        (lag_validation_table["实验场景"] == "分组时滞组合优化_主模型") & (lag_validation_table["模型类型"] == "非线性监测型模型")
    ].iloc[0]
    physical_monitor_row = lag_validation_table[
        (lag_validation_table["实验场景"] == "物理初始时滞截断_对比模型") & (lag_validation_table["模型类型"] == "监测型模型")
    ].iloc[0]
    local_search_monitor_row = lag_validation_table[
        (lag_validation_table["实验场景"] == "单变量局部搜索_对比模型") & (lag_validation_table["模型类型"] == "监测型模型")
    ].iloc[0]
    no_lag_monitor_row = lag_validation_table[
        (lag_validation_table["实验场景"] == "无时滞模型_lag0") & (lag_validation_table["模型类型"] == "监测型模型")
    ].iloc[0]
    test_rows = int((predictions["数据集"] == "测试集").sum())
    aligned_rows = len(predictions)
    max_selected_lag = int(lag_table["最佳滞后秒数"].max())
    mean_selected_lag = float(lag_table["最佳滞后秒数"].mean())
    grouped_lag_validation_rmse = float(lag_table["分组组合验证RMSE"].iloc[0]) if "分组组合验证RMSE" in lag_table else float("nan")
    grouped_lag_objective_rmse = float(lag_table["分组组合稳健目标RMSE"].iloc[0]) if "分组组合稳健目标RMSE" in lag_table else float("nan")
    grouped_lag_rmse_std = float(lag_table["分组组合验证RMSE标准差"].iloc[0]) if "分组组合验证RMSE标准差" in lag_table else float("nan")
    selected_lag_combo = str(lag_table["分组组合"].iloc[0]) if "分组组合" in lag_table else ""
    process_test = metrics_table[(metrics_table["模型"] == "过程变量模型") & (metrics_table["数据集"] == "测试集")].iloc[0]
    linear_process_test = metrics_table[
        (metrics_table["模型"] == "线性过程变量模型_对比") & (metrics_table["数据集"] == "测试集")
    ].iloc[0]
    nonlinear_process_test = metrics_table[
        (metrics_table["模型"] == "非线性过程变量模型") & (metrics_table["数据集"] == "测试集")
    ].iloc[0]
    nonlinear_monitor_test = metrics_table[
        (metrics_table["模型"] == "非线性监测型模型") & (metrics_table["数据集"] == "测试集")
    ].iloc[0]
    monitor_test = metrics_table[
        (metrics_table["模型"] == BASELINE3_MODEL_NAME) & (metrics_table["数据集"] == "测试集")
    ].iloc[0]
    selected_process_model_name = process_model.get("原候选模型名称", process_model.get("算法", ""))
    nonlinear_candidate_rows = nonlinear_monitor_test_compare[nonlinear_monitor_test_compare["模型"] != BASELINE3_MODEL_NAME]
    if nonlinear_candidate_rows.empty:
        best_nonlinear_monitor = nonlinear_monitor_test
    else:
        best_nonlinear_monitor = nonlinear_candidate_rows.loc[nonlinear_candidate_rows["RMSE"].idxmin()]
    ablation_cols = ["变量组", "测试RMSE", "测试R2", "RMSE相对完整模型变化"]
    process_ablation = ablation_table[ablation_table["模型"] == "过程变量模型"][ablation_cols].copy()
    process_ablation_md = dataframe_to_markdown(process_ablation, float_digits=4)
    sens_cols = ["变量组", "原始字段", "最佳滞后秒数", "训练集相关系数", "增加1个历史标准差后的平均CO变化"]
    sensitivity_process_md = dataframe_to_markdown(sensitivity_process_table[sens_cols].head(12), float_digits=4)
    sensitivity_monitor_md = dataframe_to_markdown(sensitivity_monitor_table[sens_cols].head(12), float_digits=4)
    perm_cols = ["变量组", "原始字段", "最佳滞后秒数", "基准RMSE", "置换后RMSE", "RMSE增加量", "RMSE增加率_%"]
    nonlinear_perm_md = dataframe_to_markdown(nonlinear_permutation_table[perm_cols].head(12), float_digits=4)
    lag_cap_parts = []
    for group_name, _, _ in WIND_BOX_CAP_GROUPS:
        subset = lag_table[lag_table["截断窗口优化组"] == group_name] if "截断窗口优化组" in lag_table else pd.DataFrame()
        if subset.empty:
            seconds = default_windbox_lag_cap_config()[group_name] * SAMPLE_INTERVAL_SECONDS
        else:
            seconds = int(subset["分段最大滞后_s"].iloc[0])
        lag_cap_parts.append(f"{group_name}最大{seconds}s")
    lag_cap_parts.extend(["大烟道变量最大120s", "机速最大300s"])
    lag_cap_md = "，".join(lag_cap_parts)
    lag_cap_objective = float(lag_table["截断窗口稳健目标RMSE"].iloc[0]) if "截断窗口稳健目标RMSE" in lag_table else float("nan")
    lag_cap_mean_rmse = float(lag_table["截断窗口平均验证RMSE"].iloc[0]) if "截断窗口平均验证RMSE" in lag_table else float("nan")
    co_history_fields = [f"CO历史浓度__lag{lag}" for lag in CO_HISTORY_LAGS] + [
        "CO历史浓度__rolling30_lag1",
        "CO历史浓度__rolling120_lag1",
    ]
    co_history_fields_md = "、".join(f"`{field}`" for field in co_history_fields)

    variant_fig_path = f"figure/第2问/{variant_name}/基础图"
    shared_fig_path = "figure/第2问/共用图"
    fig_table = f"""| 图编号 | 文件路径 | 图的含义 |
|---|---|---|
| 图2-1 | `{variant_fig_path}/图2-1_测试集CO实际与预测.svg` | 测试集实际 CO 与两类模型预测曲线对比 |
| 图2-2 | `{variant_fig_path}/图2-2_测试集预测散点.svg` | 测试集预测值与实际值的一致性 |
| 图2-3 | `{shared_fig_path}/图2-3_风箱位置与最佳时滞.svg` | 风箱到检测点距离与最佳时滞的关系，用于说明不同位置风箱对 CO 的滞后影响不同 |
| 图2-4 | `{variant_fig_path}/图2-4_非线性过程变量重要性排序.svg` | 基于正式过程变量模型 permutation importance 的变量重要性排序 |
| 图2-5 | `{variant_fig_path}/图2-5_风箱负压强度影响方向热力图.svg` | 各风箱负压强度升高时 CO 预测变化方向和强度 |
| 图2-6 | `{variant_fig_path}/图2-6_风箱温度影响方向热力图.svg` | 各风箱平均温度升高时 CO 预测变化方向和强度 |
| 图2-7 | `{variant_fig_path}/图2-7_变量组消融贡献.svg` | 风箱负压、风箱温度、大烟道负压、大烟道温度等变量组的整体贡献 |
| 图2-8 | `{variant_fig_path}/图2-8_敏感性分析_CO响应.svg` | 单个变量升高或降低 1 个历史标准差时，CO 预测值的响应方向和幅度 |"""

    files_md = """- `data/第2问/原始数据/附件1.原始数据.xlsx`：原始 Excel 只读副本。
- `data/第2问/原始数据/赛题A.pdf`：题面 PDF 只读副本。
- `data/第2问/原始数据/题目分析.md`：任务分析文件副本。
- `data/第2问/原始数据/附件1_字段化原始数据.csv`：按第2行字段名展开后的原始数据。
- `data/第2问/中间数据/清洗后数据.csv`：数值化、去重、插值并增加平均温度后的数据。
- `data/第2问/中间数据/风箱位置初始时滞_候选上限.csv`：按风箱位置、风箱长度和训练集平均机速得到的未寻优初始时滞候选上限表。
- `data/第2问/中间数据/时滞截断窗口寻优.csv`：上游、中游、下游风箱最大时滞截断窗口的多验证窗口寻优结果。
- `data/第2问/中间数据/最优时滞截断窗口初始时滞.csv`：采用最优截断窗口后的物理初始时滞表。
- `data/第2问/中间数据/风箱位置初始时滞.csv`：按风箱位置、风箱长度和训练集平均机速得到的物理初始时滞，并记录寻优后的分段最大滞后窗口。
- `data/第2问/中间数据/分组时滞候选集合.csv`：按风箱空间位置和变量类型划分后的分组候选滞后集合。
- `data/第2问/中间数据/分组时滞组合搜索.csv`：完整监测型模型在各分组时滞组合上的验证集 RMSE 搜索明细。
- `data/第2问/中间数据/滞后搜索明细.csv`：与 `分组时滞组合搜索.csv` 同步的组合搜索明细。
- `data/第2问/中间数据/最佳滞后配准.csv`：主配准方案，即多验证窗口稳健目标 RMSE 最小的分组时滞组合展开到每个过程变量后的结果。
- `data/第2问/中间数据/单变量局部搜索滞后配准_对比.csv`：旧单变量局部搜索得到的对比配准表，不作为主模型输入。
- `data/第2问/中间数据/时滞配准检查.csv`：分组时滞组合搜索的多验证窗口稳健目标 RMSE 排序明细。
- `data/第2问/中间数据/滞后配准建模数据.csv`：模型训练用的配准数据。
- `data/第2问/结果数据/CO预测结果_全样本.csv`：训练、验证、测试全样本预测结果。
- `data/第2问/结果数据/模型评价指标.csv`：MAE、RMSE、MAPE、R2 等评价指标。
- `data/第2问/结果数据/过程变量模型比较.csv`：线性过程变量模型与 LightGBM、XGBoost、梯度提升树、随机森林等非线性过程变量模型的对比。
- `data/第2问/结果数据/基线模型比较.csv`：CO_lag1、历史 CO AR、过程变量+历史 CO 监测型模型的统一对比。
- `data/第2问/结果数据/非线性监测模型比较.csv`：LightGBM、XGBoost、梯度提升树、随机森林等过程变量+历史 CO 非线性监测模型与基线3的对比。
- `data/第2问/结果数据/非线性过程变量历史CO实验.csv`：在各时滞配准场景下，用过程变量+历史 CO 作为输入训练非线性监测模型的补充实验。
- `data/第2问/结果数据/时滞合理性验证.csv`：无时滞、物理初始时滞截断、分组时滞组合优化主模型、单变量局部搜索的统一对照。
- `data/第2问/结果数据/变量影响强度_过程变量模型.csv`：过程变量模型影响强度排序。
- `data/第2问/结果数据/过程变量模型_permutation_importance.csv`：正式过程变量模型的置换重要性解释。
- `data/第2问/结果数据/非线性过程变量模型_permutation_importance.csv`：非线性过程变量模型的置换重要性解释。
- `data/第2问/结果数据/非线性监测型模型_permutation_importance.csv`：非线性监测型模型的置换重要性解释。
- `data/第2问/结果数据/非线性监测模型_permutation_importance_汇总.csv`：多类非线性监测模型的置换重要性汇总。
- `data/第2问/结果数据/变量组消融分析.csv`：各变量组替换为训练集均值后的预测性能变化；影响规律以过程变量模型消融为主，监测型消融仅作在线预测诊断。
- `data/第2问/结果数据/过程变量敏感性分析_过程变量模型.csv`：基于纯过程变量模型的扰动响应，作为影响规律主要参考。
- `data/第2问/结果数据/过程变量敏感性分析_监测型模型.csv`：历史 CO 状态固定下，监测型模型对过程变量扰动的局部响应。
- `result/第2问/CO预测模型_过程变量模型.json`：不含历史 CO 的正式非线性过程变量模型参数。
- `result/第2问/CO预测模型_线性过程变量模型_对比.json`：旧线性/岭回归过程变量模型参数，仅作为对比。
- `result/第2问/CO预测模型_过程变量_LightGBM.json`、`CO预测模型_过程变量_XGBoost.json`、`CO预测模型_过程变量_梯度提升树.json`、`CO预测模型_过程变量_随机森林.json`：非线性过程变量候选模型参数摘要。
- `result/第2问/CO预测模型_过程变量模型集合.json`：非线性过程变量候选模型集合摘要。
- `result/第2问/CO预测模型_历史CO_AR模型.json`：只用历史 CO 的 AR 模型参数。
- `result/第2问/CO预测模型_监测型模型.json`：用于在线监测预测的含历史 CO 模型参数。
- `result/第2问/CO预测模型_残差校正监测型模型_对比.json`：残差校正诊断对比模型参数，不作为主模型。
- `result/第2问/CO预测模型_非线性过程变量模型.json`：非线性过程变量模型参数摘要。
- `result/第2问/CO预测模型_非线性监测型模型.json`：非线性监测型模型参数摘要。
- `result/第2问/CO预测模型_非线性监测_LightGBM.json`、`CO预测模型_非线性监测_XGBoost.json`、`CO预测模型_非线性监测_梯度提升树.json`、`CO预测模型_非线性监测_随机森林.json`：新增过程变量+历史 CO 非线性监测模型参数摘要。
- `result/第2问/CO预测模型_非线性监测模型集合.json`：新增非线性监测模型集合摘要。
- `result/第2问/CO预测模型接口说明.md`、`输入输出字段清单.md`、`数据配准说明.md`：后续问题接口文件。
- `code/第2问/question2_model.py`：本问全部计算代码。"""

    main_md = f"""# 第2问（{variant_name}）

## 1. 本问任务理解

第2问要求探索各风箱负压、风箱温度、1#与2#大烟道负压和温度对 `烧结大烟道外排CO浓度` 的影响规律，并建立可用于实时预测的数学模型。题面特别强调风箱空间位置与检测点距离不同，因此本问把“时间滞后配准”作为建模前置步骤。为避免原始负压负值导致解释方向反转，本问保留原始负压字段，但建模统一使用 `负压强度=abs(原始负压)`，强度越大表示抽风/负压越强。

本问建立并比较了三类实时预测模型：

- **基线1：CO_lag1 持续性模型**，属于朴素持续性预测，直接用上一采样点 CO 作为当前预测。
- **基线2：只用历史 CO 的 AR 模型**，属于多滞后 CO 自回归线性回归，使用多阶历史 CO 滞后项和滚动均值。
- **基线3：过程变量 + 历史 CO 的监测型融合模型**，属于时滞配准过程变量与历史 CO 的岭回归加权融合模型，用历史 CO 与经过滞后配准的机速、风箱负压强度、风箱平均温度、大烟道负压强度和温度直接预测 CO 绝对浓度，再在验证集上与 `CO_lag1` 持续性基线加权融合。

另保留一个不含历史 CO 的**过程变量模型**，用于解释过程变量影响和无历史 CO 场景下的调控敏感性分析。该模型不再采用岭回归，而是在 LightGBM、XGBoost、梯度提升树和随机森林等非线性候选模型中按验证集 RMSE 选择。
为检验基线3是否被线性/岭回归形式限制，本问另用同一套“过程变量 + 历史 CO”输入训练 LightGBM、XGBoost、梯度提升树和随机森林等非线性监测模型，并统一与基线3比较。

## 2. 与其他问题的关系

第2问可直接基于 `附件1.原始数据.xlsx` 完成，对第1问没有硬依赖。本次没有读取或采用第1问结果，而是按题面提示自行将每个风箱南北两侧温度合并为 `i#风箱平均温度`。

第2问是第3问的硬依赖。本问已输出 `result/第2问/CO预测模型_监测型模型.json`、`result/第2问/CO预测模型_过程变量模型.json`、`result/第2问/CO预测模型接口说明.md`、`result/第2问/输入输出字段清单.md` 和 `result/第2问/数据配准说明.md`，供第3问确定负压调控决策时使用。

## 3. 数据说明

- 原始数据：`data/第2问/原始数据/附件1.原始数据.xlsx`、`data/第2问/原始数据/附件1_字段化原始数据.csv`
- 中间数据：`data/第2问/中间数据/清洗后数据.csv`、`data/第2问/中间数据/风箱位置初始时滞.csv`、`data/第2问/中间数据/时滞截断窗口寻优.csv`、`data/第2问/中间数据/分组时滞候选集合.csv`、`data/第2问/中间数据/分组时滞组合搜索.csv`、`data/第2问/中间数据/最佳滞后配准.csv`、`data/第2问/中间数据/时滞配准检查.csv`、`data/第2问/中间数据/滞后配准建模数据.csv`
- 结果数据：`data/第2问/结果数据/CO预测结果_全样本.csv`、`data/第2问/结果数据/模型评价指标.csv`、`data/第2问/结果数据/变量影响强度_过程变量模型.csv`
- 字段说明：输入过程变量包括 `烧结机机速L1设定`、`1#风箱负压强度` 至 `18#风箱负压强度`、`1#风箱平均温度` 至 `18#风箱平均温度`、`1#大烟道负压强度`、`2#大烟道负压强度`、`1#大烟道温度`、`2#大烟道温度`；输出字段为 `烧结大烟道外排CO浓度`。其中 `负压强度=abs(原始负压)`。
- 数据清洗说明：原始有效样本 {cleaning_report["原始有效数据行数"]} 条，清洗后 {cleaning_report["清洗后数据行数"]} 条；数值字段统一转换为数值型，按 `序列` 排序并去重，缺失值采用线性插值并前后向补齐。每个风箱温度用南侧与北侧逐行算术平均合并。原始负压字段保留为负值，建模字段统一取绝对值形成 `负压强度`，防止解释“增大/减小”时方向反转。
{outlier_note}

## 4. 方法与模型

1. 按 2s 采样间隔将 `序列` 换算为 `时间_s`。
2. 根据风箱长度计算 18 个风箱的中心位置。题面给出 `1#-15#` 风箱长度方向按 4m 计，`16#-18#` 按 2m 计；检测参考点取 18# 风箱末端位置。对每个风箱变量，先用 `距离/训练集平均机速` 估计物理初始时滞。
3. 不再人工固定 `900s/600s/300s` 截断窗口，而是为上游、中游、下游风箱分别构造最大滞后候选集合，并保持上游窗口不短于中游、中游不短于下游。每个候选窗口先截断物理初始时滞，再用完整监测型模型做多验证窗口评价，按“平均验证 RMSE + {LAG_SELECTION_STD_PENALTY}×验证 RMSE 标准差”最小选择截断窗口。
4. 本口径最终采用的分段最大滞后窗口为：{lag_cap_md}。若物理初始时滞超过对应寻优窗口，则先截断到该窗口上限，形成物理初始时滞截断结果。
5. 在物理初始时滞截断结果基础上，按风箱空间位置和变量类型划分滞后组；风箱变量分为上游/中游/下游负压强度组和上游/中游/下游平均温度组，机速、大烟道负压强度、大烟道温度分别成组。
6. 对每个滞后组，在对应变量允许的最大滞后窗口内构造候选滞后集合，并枚举分组时滞组合；同样用多个时间验证窗口评价完整监测型模型。为避免只追逐微小验证集收益而过拟合，先取稳健目标 RMSE 最优值 {GROUP_LAG_SIMPLICITY_TOLERANCE * 100:.1f}% 以内的组合，再优先选择非零调整组数和调整幅度最小的简约组合，展开为 `最佳滞后配准.csv`。旧单变量局部搜索只作为对比分析，不作为主模型输入。
7. 先基于完整时间序列构造 `变量__lagN` 和历史 CO 滞后特征；每一行仍以当前 CO 所在的目标时刻为样本时刻。
8. 再按原始目标时刻划分训练集、验证集、测试集，比例约为 70%/15%/15%，避免先删行再切分导致测试时段前移。时滞窗口和分组组合寻优只使用测试集之前的多个验证窗口，最终测试集不参与任何时滞选择。
9. 建立三类对照预测：`CO_lag1` 持续性模型、只用历史 CO 的 AR 模型、过程变量 + 历史 CO 的监测型融合模型。
10. AR 模型直接拟合 CO；监测型主模型也直接拟合 CO 绝对浓度，再由验证集选择权重，将岭回归预测与 `CO_lag1` 持续性基线加权融合。
11. 原“预测 `CO真实-CO_lag1`”的残差校正模型仅保留为诊断对照，不作为主模型；原因是过程变量与该短时残差的相关性过弱，容易出现过校正。
12. 过程变量模型不再使用岭回归，而是基于同一套时滞配准过程变量训练 LightGBM、XGBoost、梯度提升树和随机森林，并按验证集 RMSE 选择正式过程变量模型；旧岭回归过程变量模型仅作为 `线性过程变量模型_对比` 保留。
13. 同时基于同一套“过程变量 + 历史 CO”输入训练 LightGBM、XGBoost、梯度提升树和随机森林等非线性监测模型，并与基线3统一比较。
14. 评价指标除 MAE、RMSE、MAPE、R2 外，补充趋势准确率和高 CO 段误差。趋势准确率按相邻样本真实变化方向与预测变化方向是否一致计算；高 CO 阈值取训练集 CO 浓度 90% 分位数。
15. 对纯过程变量模型做变量组消融：将某一组过程变量替换为训练集均值后重新预测，观察测试 RMSE 的变化；监测型模型消融仅作为在线预测诊断，用于说明历史 CO 状态对预测精度的贡献，不作为影响规律主依据。
16. 对过程变量做敏感性分析：在测试集上将单个过程变量增加/减少一个训练集历史标准差，计算平均 CO 预测变化。主要采用过程变量模型解释影响规律，同时补充监测型模型在历史 CO 状态固定时的局部响应。
17. 对时滞配准做合理性验证：比较 `无时滞模型_lag0`、`物理初始时滞截断_对比模型`、`分组时滞组合优化_主模型` 和 `单变量局部搜索_对比模型`，评价同一测试集上的 RMSE、MAPE 与 R2；同时补充一组“过程变量 + 历史 CO”的非线性监测实验，检验非线性模型在不同配准口径下的泛化表现。

## 5. 计算过程

- 时滞选择方式：先由风箱位置距离和训练集平均机速计算物理初始时滞；上游、中游、下游风箱的最大截断窗口由候选集合寻优得到；随后按风箱空间位置和变量类型划分滞后组，在各组允许窗口内枚举候选组合，并以完整监测型模型多验证窗口稳健 RMSE 最小选择主配准方案。单变量局部搜索只作为对比分析。
- 寻优后的分段最大窗口：{lag_cap_md}；截断窗口平均验证 RMSE 为 {lag_cap_mean_rmse:.3f}，稳健目标 RMSE 为 {lag_cap_objective:.3f}。
- 最终进入配准建模数据的样本数：{aligned_rows} 条，其中测试集 {test_rows} 条。
- 各变量组最佳滞后摘要：

{lag_md}

- 过程变量模型影响强度最高的变量：

{imp_md}

## 6. 结果

测试集评价指标如下：

{metric_md}

三类实时预测模型相对 `CO_lag1` 基线的增益如下，避免与上表重复列出同一批绝对指标：

{baseline_md}

其中 `高CO阈值_训练90分位` 已写入 `模型评价指标.csv`，所有高 CO 段误差均只用训练集阈值划分，避免用测试集分布反向定义阈值。

时滞配准合理性验证如下：

{lag_validation_md}

其中，新增的“过程变量 + 历史 CO”非线性补充实验如下。每个场景均在 LightGBM、XGBoost、梯度提升树和随机森林中按验证集 RMSE 选择非线性监测候选模型，再只在测试集上评价：

{nonlinear_lag_validation_md}

配准检查显示，按风箱中心到检测参考点距离和训练集平均机速计算的原始物理初始时滞较长；为避免人工给定截断窗口造成主观性，本问先对上游、中游、下游风箱最大截断窗口做候选寻优，选定窗口为：{lag_cap_md}。随后再按空间位置和变量类型做分组组合优化。最终选中的分组组合为 `{selected_lag_combo}`，多验证窗口平均融合 RMSE 为 {grouped_lag_validation_rmse:.3f}，标准差为 {grouped_lag_rmse_std:.3f}，稳健目标 RMSE 为 {grouped_lag_objective_rmse:.3f}；主模型平均最佳滞后约 {main_monitor_row["平均最佳滞后_s"]:.1f}s，最大约 {main_monitor_row["最大已选滞后_s"]:.1f}s。测试集上，分组时滞组合优化监测型 RMSE 为 {main_monitor_row["RMSE"]:.3f}，物理初始时滞截断对比模型为 {physical_monitor_row["RMSE"]:.3f}，单变量局部搜索对比模型为 {local_search_monitor_row["RMSE"]:.3f}，无时滞监测模型为 {no_lag_monitor_row["RMSE"]:.3f}。

在分组时滞组合优化口径下，补充实验选中的非线性监测候选模型为 `{main_nonlinear_lag_row["非线性监测候选模型"]}`，测试集 RMSE 为 {main_nonlinear_lag_row["RMSE"]:.3f}。该实验使用的输入为同一套滞后配准过程变量和历史 CO 特征，目标仍是当前 CO 浓度。

从结果看，监测型融合模型直接拟合 CO 绝对浓度，并通过验证集权重吸收 `CO_lag1` 的惯性基线，适合在线预测当前/短时 CO 浓度，也适合第3问在“历史 CO 状态保持给定”的条件下评估负压调整；过程变量模型不依赖目标历史值，适合做无历史 CO 场景的敏感性分析。正式过程变量模型由验证集选择为 `{selected_process_model_name}`，测试集 RMSE 为 {process_test["RMSE"]:.3f} mg/m^3，MAPE 为 {process_test["MAPE_%"]:.3f}%；旧线性过程变量模型对比口径测试集 RMSE 为 {linear_process_test["RMSE"]:.3f} mg/m^3。监测型融合模型测试集 RMSE 为 {monitor_test["RMSE"]:.3f} mg/m^3，MAPE 为 {monitor_test["MAPE_%"]:.3f}%。

过程变量模型候选与旧线性过程变量模型对比如下：

{process_compare_md}

新增的过程变量 + 历史 CO 非线性监测模型与基线3对比如下：

{nonlinear_monitor_md}

非线性对照显示，过程变量到 CO 的映射存在非线性结构，但不同非线性模型在测试时段上的泛化能力差异较大；因此正式过程变量模型只按验证集选择，不用测试集反向挑模型。过程变量 + 历史 CO 口径下，测试集 RMSE 最低的非线性监测模型为 `{best_nonlinear_monitor["模型"]}`，RMSE 为 {best_nonlinear_monitor["RMSE"]:.3f} mg/m^3；基线3监测型融合模型测试集 RMSE 为 {monitor_test["RMSE"]:.3f} mg/m^3。若非线性模型相对基线3的降低率为负，说明在当前样本规模与时间切分下非线性模型存在过拟合或泛化不足。

过程变量模型测试集 R2 偏低，说明仅靠本附件中的压力、温度和机速很难在最后一段时间序列上独立复原 CO 的绝对水平；这不应作为第2问实时预测模型的主结论。但第2问要回答的是过程变量对 CO 浓度的影响规律，因此影响规律的主证据应来自正式非线性过程变量模型、物理滞后配准、过程变量模型消融和敏感性分析。监测型模型只用于完成“实时预测”任务，其消融结果不作为影响规律主依据。

过程变量模型变量组消融结果（影响规律参考）：

{process_ablation_md}

监测型模型的变量组消融包含历史 CO 输入，主要用于诊断在线预测时 `CO_lag1` 等历史状态的贡献。由于它会把历史 CO 的惯性记忆与过程变量作用混在一起，本报告不把监测型消融作为“过程变量影响规律”的主证据；完整诊断表保存在 `data/第2问/结果数据/变量组消融分析.csv`。

过程变量模型中敏感性较高的字段：

{sensitivity_process_md}

监测型模型中过程变量局部响应较高的字段：

{sensitivity_monitor_md}

监测型敏感性分析是在 `CO_lag1` 及其他历史 CO 状态固定的条件下得到的局部响应。由于监测型融合模型同时包含历史 CO 与过程变量，其扰动响应适合解释当前状态下的边际变化；影响规律的主解释仍以纯过程变量模型敏感性、物理滞后配准、非线性过程变量置换重要性和过程变量模型消融为主。

正式过程变量模型的 permutation importance 结果如下：

{nonlinear_perm_md}

置换重要性含义为：在测试集上随机打乱某个输入字段，观察 RMSE 增加量。该指标不依赖线性系数，适合作为非线性模型下的解释补充。

滞后配准显示，主方案变量最佳滞后平均约 {mean_selected_lag:.1f}s，最大被选滞后为 {max_selected_lag}s，且不超过对应变量的分段最大窗口。结果说明不能简单地把所有过程变量与 CO 同时刻对齐，风箱负压强度、风箱温度和大烟道状态均存在不同程度的时滞影响。

## 7. 基础可视化

{fig_table}

## 8. 本问生成的文件清单

{files_md}

## 9. 可传递给后续问题的信息

- 可用数据：`data/第2问/结果数据/CO预测结果_全样本.csv`、`data/第2问/中间数据/最佳滞后配准.csv`、`data/第2问/中间数据/滞后配准建模数据.csv`
- 可用参数：每个过程变量的最佳滞后、非线性过程变量模型的入选算法、最佳参数和特征重要性，见 `result/第2问/CO预测模型_过程变量模型.json`
- 可用模型：第3问做状态条件调控时建议使用 `监测型模型` 并固定历史 CO 滞后项；做无历史 CO 的过程变量敏感性分析时使用 `过程变量模型`

## 更新记录

- {now}，第2问子Agent：完成数据清洗、滞后配准、双模型训练、结果数据、基础图和接口文件生成。
"""

    interface_md = f"""# CO预测模型接口说明

生成时间：{now}

负责Agent：第2问子Agent

读取源文件：`题目分析.md`、`赛题A.pdf`、`附件1.原始数据.xlsx`

适用问题：第2问、第3问

## 推荐给第3问调用的模型

若第3问是在某一运行状态下寻找风箱负压调控策略，推荐调用 `result/第2问/CO预测模型_监测型模型.json`，并将历史 CO 滞后项作为当前状态常量，只改变候选调控方案对应的风箱负压强度输入。若第3问给出的是原始负压设定值，应先取绝对值转换为负压强度。

若第3问需要不依赖历史 CO 的纯过程变量模型，可调用 `result/第2问/CO预测模型_过程变量模型.json`。该模型只使用可观测或可调控的过程变量，已改为从 LightGBM、XGBoost、梯度提升树和随机森林候选中按验证集 RMSE 选择；本口径入选候选为 `{selected_process_model_name}`。测试集预测精度低于监测型模型，更适合做敏感性与方向性参考。

其中 `x_i` 是按照 `滞后配准表` 取到的输入特征。例如 `5#风箱负压强度__lag37` 表示预测当前 CO 时，应使用 74s 前的 `5#风箱负压强度`，即对应原始 `5#风箱负压` 的绝对值。

## 监测模型

`result/第2问/CO预测模型_监测型模型.json` 加入了历史 CO 滞后项，直接拟合 CO 绝对浓度，并在验证集上选择与 `CO_lag1` 持续性基线的融合权重：

```text
监测型预测 = w * f(历史CO, 过程变量) + (1 - w) * CO_lag1
```

该模型测试集精度更高，适合在线预测和误差监测。但它把历史 CO 作为输入，不适合作为第3问单独优化风箱负压时的唯一依据；做调控解释时应同时参考过程变量模型和敏感性分析。

## 输入输出

- 输入：见 `result/第2问/输入输出字段清单.md`。
- 输出：当前时刻 `烧结大烟道外排CO浓度` 的预测值，单位 mg/m^3。
- 时间滞后口径：主模型先按风箱位置距离和训练集平均机速计算物理初始时滞；上游、中游、下游风箱最大截断窗口由多验证窗口寻优得到，再按寻优窗口截断；随后按风箱空间位置和变量类型划分滞后组，并以完整监测型模型多验证窗口稳健目标 RMSE 最小选择分组时滞组合。模型 JSON 中保留每个变量实际采用的主配准滞后；单变量局部搜索结果仅作为对比分析，不写入主模型。
- 数据口径：风箱温度采用南北侧算术平均，即 `i#风箱平均温度=(i#风箱南侧温度+i#风箱北侧温度)/2`。

## 调用注意

1. 第3问若改变某一时刻风箱负压，需要先将原始负压设定转换为 `负压强度=abs(负压)`，再按模型 JSON 中对应 `__lagN` 的要求，把调控后的负压强度放到预测时刻前 `N` 个采样点的位置。
2. 若只做单点静态近似，可先用同一调控方案填充各相关滞后位置，但需在第3问中声明这是静态近似。
3. 输入变量应落在附件1历史数据范围附近，超出范围属于外推，可靠性会下降。
"""

    fields_md = f"""# 输入输出字段清单

生成时间：{now}

负责Agent：第2问子Agent

## 输出字段

| 字段 | 单位 | 含义 |
|---|---|---|
| `烧结大烟道外排CO浓度` | mg/m^3 | 模型预测目标 |

## 过程变量模型输入字段

| 字段组 | 字段 |
|---|---|
| 机速 | `烧结机机速L1设定` |
| 风箱负压强度 | `1#风箱负压强度` 至 `18#风箱负压强度`，由原始负压取绝对值得到，强度越大表示抽风/负压越强 |
| 风箱温度 | `1#风箱平均温度` 至 `18#风箱平均温度`，由南北侧温度平均得到 |
| 大烟道负压强度 | `1#大烟道负压强度`、`2#大烟道负压强度`，由原始负压取绝对值得到 |
| 大烟道温度 | `1#大烟道温度`、`2#大烟道温度` |

每个字段实际调用时应使用 `data/第2问/中间数据/最佳滞后配准.csv` 中给出的最佳滞后。

## 监测型模型额外输入

{co_history_fields_md}。
"""

    alignment_md = f"""# 数据配准说明

生成时间：{now}

负责Agent：第2问子Agent

## 配准原则

题面说明各风箱位置不同且指标存在规律性时滞。本问将空间位置差异先转化为物理初始时滞：按题面风箱长度计算每个风箱中心位置，以 18# 风箱末端作为检测参考位置，使用 `距离/训练集平均机速` 得到各风箱变量的初始时滞。随后不再人工固定 `900s/600s/300s`，而是为上游、中游、下游风箱设置候选截断窗口，并用完整监测型模型多验证窗口稳健目标 RMSE 寻优。寻优后的分段最大滞后窗口为：{lag_cap_md}。若物理初始时滞超过分段窗口，则先截断到该分段上限。在此基础上，按风箱空间位置和变量类型划分滞后组，在各组允许的最大滞后窗口内构造候选滞后集合，并以完整监测型模型多验证窗口稳健目标 RMSE 最小选择整体最优分组时滞组合。单变量局部搜索结果只作为对比分析，不作为主模型输入。

## 配准结果摘要

{lag_md}

主配准表见 `data/第2问/中间数据/最佳滞后配准.csv`，截断窗口寻优见 `data/第2问/中间数据/时滞截断窗口寻优.csv`，初始时滞见 `data/第2问/中间数据/风箱位置初始时滞.csv`，分组候选集合见 `data/第2问/中间数据/分组时滞候选集合.csv`，组合搜索明细见 `data/第2问/中间数据/分组时滞组合搜索.csv` 和 `data/第2问/中间数据/滞后搜索明细.csv`，单变量局部搜索对比表见 `data/第2问/中间数据/单变量局部搜索滞后配准_对比.csv`，配准检查见 `data/第2问/中间数据/时滞配准检查.csv`。

## 避免信息泄漏

物理初始时滞中的平均机速只使用训练集计算；单变量局部搜索也只使用训练集且仅作对比。截断窗口寻优和分组时滞组合优化均只使用测试集之前的多个时间验证窗口，评价目标为平均验证 RMSE 加稳定性惩罚，不接触最终测试集。全部滞后特征先在完整时间序列上按历史方向构造，再按当前 CO 的原始目标时刻划分训练集、验证集和测试集。模型参数由训练集拟合，岭参数与融合权重由验证集选择，测试集只用于最终评价。所有 `__lagN` 特征都表示使用历史时刻变量，不使用未来信息。
"""

    summary_md = f"""# 第2问_结果摘要（{variant_name}）

生成时间：{now}

本问完成了 CO 浓度实时预测建模与过程变量影响规律分析。主模型采用“物理初始时滞 + 截断窗口寻优 + 分组时滞组合稳健优化”的配准方式：先由风箱位置与机速给出物理初始时滞，再用多验证窗口寻优上游、中游、下游风箱最大截断窗口，之后按风箱空间位置和变量类型划分滞后组，以完整监测型模型多验证窗口稳健目标 RMSE 最小选择整体组合。过程变量模型已从 LightGBM、XGBoost、梯度提升树和随机森林候选中按验证集 RMSE 选择，旧岭回归过程变量模型仅作为对比保留。由于过程变量模型在测试集上 R2 偏低，本问不把它作为实时预测主模型，而是用监测型模型完成预测；同时补充 LightGBM、XGBoost、梯度提升树和随机森林等过程变量 + 历史 CO 非线性监测模型，与基线3统一对比。影响规律判断则主要依据正式过程变量模型、物理滞后配准、过程变量模型消融和敏感性分析。

{outlier_note}

测试集主要结果：

{metric_md}

第3问建议优先读取：

- `result/第2问/CO预测模型_监测型模型.json`
- `result/第2问/CO预测模型_过程变量模型.json`
- `result/第2问/CO预测模型接口说明.md`
- `result/第2问/输入输出字段清单.md`
- `result/第2问/数据配准说明.md`
"""

    (ROOT / "第2问.md").write_text(main_md, encoding="utf-8")
    (RESULT_DIR / "第2问.md").write_text(main_md, encoding="utf-8")
    (RESULT_DIR / "第2问_结果摘要.md").write_text(summary_md, encoding="utf-8")
    (RESULT_DIR / "CO预测模型接口说明.md").write_text(interface_md, encoding="utf-8")
    (RESULT_DIR / "输入输出字段清单.md").write_text(fields_md, encoding="utf-8")
    (RESULT_DIR / "数据配准说明.md").write_text(alignment_md, encoding="utf-8")


def run_modeling_variant(raw_data: pd.DataFrame, variant_name: str, enable_co_outlier_correction: bool) -> dict:
    set_variant_output_dirs(variant_name)

    clean_df, missing_table, cleaning_report = clean_data(raw_data)
    cleaning_report["建模口径"] = variant_name
    if enable_co_outlier_correction:
        clean_df, outlier_report = correct_co_outliers(clean_df)
        cleaning_report["CO异常修正"] = outlier_report
        clean_df.to_csv(MID_DIR / "清洗后数据.csv", index=False, encoding="utf-8-sig")
    else:
        outlier_report = write_no_co_outlier_correction_report(clean_df)
        cleaning_report["CO异常修正"] = outlier_report
    with open(MID_DIR / "数据清洗报告.json", "w", encoding="utf-8") as f:
        json.dump(cleaning_report, f, ensure_ascii=False, indent=2)

    field_table = build_field_table(clean_df.columns)

    groups = get_feature_groups()
    process_features = [feature for features in groups.values() for feature in features]

    n_clean = len(clean_df)
    train_end_for_lag = int(n_clean * 0.70)
    lag_prior_table = make_physical_lag_priors(clean_df, process_features, train_end_for_lag)
    lag_prior_table.to_csv(MID_DIR / "风箱位置初始时滞_候选上限.csv", index=False, encoding="utf-8-sig")
    lag_prior_table, lag_cap_search_table = select_lag_cap_config_by_monitor_cv(
        clean_df,
        process_features,
        train_end_for_lag,
        lag_prior_table,
        search_output=MID_DIR / "时滞截断窗口寻优.csv",
        best_prior_output=MID_DIR / "最优时滞截断窗口初始时滞.csv",
    )
    lag_prior_table.to_csv(MID_DIR / "风箱位置初始时滞.csv", index=False, encoding="utf-8-sig")
    lag_detail, local_search_lag_table = select_lags(
        clean_df,
        process_features,
        train_end_for_lag,
        max_lag_samples=MAX_LAG_SAMPLES,
        detail_output=MID_DIR / "单变量局部搜索明细_对比.csv",
        best_output=MID_DIR / "单变量局部搜索滞后配准_对比.csv",
        lag_prior_table=lag_prior_table,
    )
    physical_lag_table = make_physical_initial_lag_table(
        clean_df,
        process_features,
        train_end_for_lag,
        lag_prior_table,
        max_lag_samples=MAX_LAG_SAMPLES,
    )
    physical_lag_table.to_csv(MID_DIR / "物理初始时滞配准表.csv", index=False, encoding="utf-8-sig")
    lag_table, grouped_lag_candidates, grouped_lag_combo_table = select_grouped_lags_by_monitor_validation(
        clean_df,
        physical_lag_table,
        candidate_output=MID_DIR / "分组时滞候选集合.csv",
        combo_output=MID_DIR / "分组时滞组合搜索.csv",
        best_output=MID_DIR / "分组时滞组合最优配准表.csv",
    )
    lag_table.to_csv(MID_DIR / "最佳滞后配准.csv", index=False, encoding="utf-8-sig")
    grouped_lag_combo_table.to_csv(MID_DIR / "滞后搜索明细.csv", index=False, encoding="utf-8-sig")
    grouped_lag_combo_table.to_csv(MID_DIR / "时滞配准检查.csv", index=False, encoding="utf-8-sig")
    aligned, process_lagged_features, co_features = make_aligned_dataset(clean_df, lag_table)
    aligned = split_by_time(aligned)
    aligned.to_csv(MID_DIR / "滞后配准建模数据.csv", index=False, encoding="utf-8-sig")

    linear_process_model, pred_linear_process, linear_process_info = fit_ridge_model(
        aligned,
        process_lagged_features,
        "线性过程变量模型_对比",
    )
    process_suite_models, process_suite_predictions, process_suite_infos = fit_nonlinear_process_suite(
        aligned,
        process_lagged_features,
    )
    process_model, pred_process, process_info, selected_process_model_name = select_best_nonlinear_model(
        process_suite_models,
        process_suite_predictions,
        process_suite_infos,
        "过程变量模型",
    )
    process_predict_func = make_sklearn_frame_predictor(process_info)
    ar_model, pred_ar, ar_info = fit_ridge_model(aligned, co_features, "历史CO_AR模型")
    monitor_features = process_lagged_features + co_features
    monitor_model, pred_monitor_ridge, monitor_info = fit_ridge_model(
        aligned,
        monitor_features,
        "过程变量+历史CO监测型模型",
    )
    pred_monitor, monitor_blend_info = blend_monitor_prediction(aligned, pred_monitor_ridge)
    monitor_model["融合说明"] = {
        "融合方式": "验证集选择权重，将含过程变量的岭回归预测与 CO_lag1 持续性基线加权平均",
        "岭回归预测权重": monitor_blend_info["ridge_weight"],
        "CO_lag1基线权重": monitor_blend_info["lag1_weight"],
        "融合后验证RMSE": monitor_blend_info["validation_rmse"],
    }
    residual_monitor_model, pred_residual_monitor, residual_monitor_info = fit_residual_ridge_model(
        aligned,
        monitor_features,
        "CO历史浓度__lag1",
        "残差校正监测型模型_对比",
    )
    nonlinear_process_model, pred_nonlinear_process, nonlinear_process_info = fit_random_forest_model(
        aligned,
        process_lagged_features,
        "非线性过程变量模型",
    )
    nonlinear_monitor_model, pred_nonlinear_monitor, nonlinear_monitor_info = fit_random_forest_model(
        aligned,
        monitor_features,
        "非线性监测型模型",
        seed=20260530,
    )
    nonlinear_monitor_suite_models, nonlinear_monitor_suite_predictions, nonlinear_monitor_suite_infos = fit_nonlinear_monitor_suite(
        aligned,
        monitor_features,
    )
    nonlinear_monitor_suite_names = list(nonlinear_monitor_suite_models.keys())

    predictions = aligned[["序列", "时间_s", "数据集", TARGET_COL]].copy()
    predictions["过程变量模型_预测CO浓度"] = pred_process
    predictions["线性过程变量模型_对比_预测CO浓度"] = pred_linear_process
    for model_name, pred in process_suite_predictions.items():
        predictions[f"{model_name}_预测CO浓度"] = pred
    predictions["非线性过程变量模型_预测CO浓度"] = pred_nonlinear_process
    predictions["基线1_CO_lag1持续性模型_预测CO浓度"] = aligned["CO历史浓度__lag1"].to_numpy(dtype=float)
    predictions["基线2_历史CO_AR模型_预测CO浓度"] = pred_ar
    predictions["监测型模型_预测CO浓度"] = pred_monitor
    predictions["残差校正监测型模型_对比_预测CO浓度"] = pred_residual_monitor
    predictions["非线性监测型模型_预测CO浓度"] = pred_nonlinear_monitor
    for model_name, pred in nonlinear_monitor_suite_predictions.items():
        predictions[f"{model_name}_预测CO浓度"] = pred
    predictions["过程变量模型_残差"] = predictions["过程变量模型_预测CO浓度"] - predictions[TARGET_COL]
    predictions["线性过程变量模型_对比_残差"] = predictions["线性过程变量模型_对比_预测CO浓度"] - predictions[TARGET_COL]
    for model_name in process_suite_models:
        predictions[f"{model_name}_残差"] = predictions[f"{model_name}_预测CO浓度"] - predictions[TARGET_COL]
    predictions["非线性过程变量模型_残差"] = predictions["非线性过程变量模型_预测CO浓度"] - predictions[TARGET_COL]
    predictions["基线1_CO_lag1持续性模型_残差"] = predictions["基线1_CO_lag1持续性模型_预测CO浓度"] - predictions[TARGET_COL]
    predictions["基线2_历史CO_AR模型_残差"] = predictions["基线2_历史CO_AR模型_预测CO浓度"] - predictions[TARGET_COL]
    predictions["监测型模型_残差"] = predictions["监测型模型_预测CO浓度"] - predictions[TARGET_COL]
    predictions["残差校正监测型模型_对比_残差"] = predictions["残差校正监测型模型_对比_预测CO浓度"] - predictions[TARGET_COL]
    predictions["非线性监测型模型_残差"] = predictions["非线性监测型模型_预测CO浓度"] - predictions[TARGET_COL]
    for model_name in nonlinear_monitor_suite_names:
        predictions[f"{model_name}_残差"] = predictions[f"{model_name}_预测CO浓度"] - predictions[TARGET_COL]
    predictions.to_csv(RESULT_DATA_DIR / "CO预测结果_全样本.csv", index=False, encoding="utf-8-sig")

    pred_cols = {
        "过程变量模型": "过程变量模型_预测CO浓度",
        "线性过程变量模型_对比": "线性过程变量模型_对比_预测CO浓度",
        "非线性过程变量模型": "非线性过程变量模型_预测CO浓度",
        "基线1_CO_lag1持续性模型": "基线1_CO_lag1持续性模型_预测CO浓度",
        "基线2_历史CO_AR模型": "基线2_历史CO_AR模型_预测CO浓度",
        BASELINE3_MODEL_NAME: "监测型模型_预测CO浓度",
        "残差校正监测型模型_对比": "残差校正监测型模型_对比_预测CO浓度",
        "非线性监测型模型": "非线性监测型模型_预测CO浓度",
    }
    for model_name in process_suite_models:
        pred_cols[model_name] = f"{model_name}_预测CO浓度"
    for model_name in nonlinear_monitor_suite_names:
        pred_cols[model_name] = f"{model_name}_预测CO浓度"
    metrics_table = evaluate_models(predictions, pred_cols)
    process_model_comparison = make_process_model_comparison(
        metrics_table,
        list(process_suite_models.keys()) + ["非线性过程变量模型"],
    )
    baseline_comparison = make_baseline_comparison(metrics_table)
    nonlinear_monitor_comparison = make_nonlinear_monitor_comparison(
        metrics_table,
        ["非线性监测型模型"] + nonlinear_monitor_suite_names,
    )
    lag_summary = summarize_lag(lag_table)
    lag_validation_table = make_lag_validation_experiments(
        clean_df,
        process_features,
        lag_table,
        physical_lag_table,
        local_search_lag_table,
        lag_prior_table,
        train_end_for_lag,
    )

    importance_process = pd.DataFrame(process_model.get("特征重要性", []))
    if importance_process.empty:
        importance_process = pd.DataFrame({"模型特征": process_model["特征字段"], "重要性": np.nan})
    importance_process = importance_process.rename(columns={"模型特征": "特征名", "重要性": "综合影响强度"})
    importance_process["模型"] = "过程变量模型"
    importance_process["标准化线性系数"] = np.nan
    importance_process["标准化平方项系数"] = np.nan
    importance_process["线性影响方向"] = "非线性无固定方向"
    importance_process["原始字段"] = importance_process["特征名"].map(strip_lag_name)
    importance_process = importance_process.merge(
        lag_table[["字段名", "最佳滞后秒数", "训练集相关系数"]],
        left_on="原始字段",
        right_on="字段名",
        how="left",
    )
    importance_process = importance_process.sort_values("综合影响强度", ascending=False)
    importance_process.to_csv(RESULT_DATA_DIR / "变量影响强度_过程变量模型.csv", index=False, encoding="utf-8-sig")

    importance_monitor = pd.DataFrame(monitor_info["importance"]).sort_values("综合影响强度", ascending=False)
    importance_monitor.to_csv(RESULT_DATA_DIR / "变量影响强度_监测型模型.csv", index=False, encoding="utf-8-sig")
    process_perm = make_generic_permutation_importance(
        aligned,
        process_info["feature_cols"],
        process_info["estimator"].predict,
        "过程变量模型",
        lag_table,
        "过程变量模型_permutation_importance.csv",
    )
    nonlinear_process_perm_legacy = make_permutation_importance(
        aligned,
        nonlinear_process_info["forest"],
        lag_table,
        "非线性过程变量模型_permutation_importance.csv",
    )
    nonlinear_monitor_perm = make_permutation_importance(
        aligned,
        nonlinear_monitor_info["forest"],
        lag_table,
        "非线性监测型模型_permutation_importance.csv",
        seed=20260530,
    )
    nonlinear_monitor_suite_perm_tables = []
    for idx, model_name in enumerate(nonlinear_monitor_suite_names):
        info = nonlinear_monitor_suite_infos[model_name]
        perm_table = make_generic_permutation_importance(
            aligned,
            info["feature_cols"],
            info["estimator"].predict,
            model_name,
            lag_table,
            f"{model_name}_permutation_importance.csv",
            seed=20260531 + idx * 997,
        )
        nonlinear_monitor_suite_perm_tables.append(perm_table)
    if nonlinear_monitor_suite_perm_tables:
        nonlinear_monitor_suite_perm_summary = pd.concat(nonlinear_monitor_suite_perm_tables, ignore_index=True)
    else:
        nonlinear_monitor_suite_perm_summary = pd.DataFrame()
    nonlinear_monitor_suite_perm_summary.to_csv(
        RESULT_DATA_DIR / "非线性监测模型_permutation_importance_汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ablation_table = make_group_ablation(
        aligned,
        {
            "过程变量模型": {"model": process_model, "predict_func": process_predict_func},
            "监测型模型": {"model": monitor_model, "predict_func": lambda frame: predict_from_model(frame, monitor_model)},
        },
    )
    sensitivity_process_table = make_sensitivity_analysis(
        aligned,
        process_model,
        lag_table,
        "过程变量敏感性分析_过程变量模型.csv",
        predict_func=process_predict_func,
    )
    sensitivity_monitor_table = make_sensitivity_analysis(
        aligned,
        monitor_model,
        lag_table,
        "过程变量敏感性分析_监测型模型.csv",
    )

    save_models(
        process_model,
        linear_process_model,
        process_suite_models,
        ar_model,
        monitor_model,
        residual_monitor_model,
        nonlinear_process_model,
        nonlinear_monitor_model,
        nonlinear_monitor_suite_models,
        lag_table,
    )

    test_df = predictions[predictions["数据集"] == "测试集"]
    make_line_plot(FIG_DIR / "图2-1_测试集CO实际与预测.svg", test_df, "Test CO: actual vs predicted")
    make_scatter_plot(FIG_DIR / "图2-2_测试集预测散点.svg", test_df, "Predicted vs actual CO")
    make_permutation_importance_plot(
        FIG_DIR / "图2-4_非线性过程变量重要性排序.svg",
        process_perm,
        "Nonlinear process-variable permutation importance",
    )
    make_windbox_direction_heatmap(
        FIG_DIR / "图2-5_风箱负压强度影响方向热力图.svg",
        sensitivity_process_table,
        "风箱负压强度",
        "Windbox pressure-strength effect direction",
    )
    make_windbox_direction_heatmap(
        FIG_DIR / "图2-6_风箱温度影响方向热力图.svg",
        sensitivity_process_table,
        "风箱平均温度",
        "Windbox temperature effect direction",
    )
    make_group_ablation_plot(FIG_DIR / "图2-7_变量组消融贡献.svg", ablation_table, "Variable-group ablation contribution")
    make_sensitivity_plot(FIG_DIR / "图2-8_敏感性分析_CO响应.svg", sensitivity_process_table, "Process-variable sensitivity response")
    for obsolete_name in ["图2-3_变量最佳滞后.svg", "图2-3_风箱位置与最佳时滞.svg", "图2-4_过程模型关键变量影响.svg"]:
        obsolete_plot = FIG_DIR / obsolete_name
        if obsolete_plot.exists():
            obsolete_plot.unlink()

    with open(RESULT_DATA_DIR / "建模运行摘要.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "清洗报告": cleaning_report,
                "最大滞后样本数": MAX_LAG_SAMPLES,
                "最大滞后秒数": MAX_LAG_SECONDS,
                "分段最大滞后设置": [
                    {
                        "变量范围": group_name,
                        "最大滞后样本数": int(
                            lag_prior_table[lag_prior_table["截断窗口优化组"] == group_name]["分段最大滞后样本数"].iloc[0]
                        ),
                        "最大滞后秒数": int(
                            lag_prior_table[lag_prior_table["截断窗口优化组"] == group_name]["分段最大滞后_s"].iloc[0]
                        ),
                        "说明": "由完整监测型模型多验证窗口稳健RMSE寻优得到",
                    }
                    for group_name, _, _ in WIND_BOX_CAP_GROUPS
                ]
                + [
                    {
                        "变量范围": "机速",
                        "最大滞后样本数": SPEED_MAX_LAG_SAMPLES,
                        "最大滞后秒数": SPEED_MAX_LAG_SAMPLES * SAMPLE_INTERVAL_SECONDS,
                        "说明": "机速全线设定量，最大滞后300s",
                    },
                    {
                        "变量范围": "大烟道变量",
                        "最大滞后样本数": DUCT_MAX_LAG_SAMPLES,
                        "最大滞后秒数": DUCT_MAX_LAG_SAMPLES * SAMPLE_INTERVAL_SECONDS,
                        "说明": "大烟道变量靠近检测位置，最大滞后120s",
                    },
                ],
                "时滞截断窗口寻优最优": json_ready(lag_cap_search_table.iloc[0].to_dict()) if not lag_cap_search_table.empty else {},
                "历史CO最大滞后样本数": CO_HISTORY_MAX_LAG_SAMPLES,
                "历史CO最大滞后秒数": CO_HISTORY_MAX_LAG_SAMPLES * SAMPLE_INTERVAL_SECONDS,
                "配准建模样本数": int(len(aligned)),
                "过程变量模型验证RMSE": process_info["validation_rmse"],
                "过程变量模型入选候选": selected_process_model_name,
                "线性过程变量模型对比验证RMSE": linear_process_info["validation_rmse"],
                "过程变量非线性候选验证RMSE": {
                    name: info["validation_rmse"] for name, info in process_suite_infos.items()
                },
                "历史CO_AR模型验证RMSE": ar_info["validation_rmse"],
                "监测型岭回归验证RMSE": monitor_info["validation_rmse"],
                "监测型融合验证RMSE": monitor_blend_info["validation_rmse"],
                "残差校正对比模型验证RMSE": residual_monitor_info["validation_rmse"],
                "非线性过程变量模型验证RMSE": nonlinear_process_info["validation_rmse"],
                "非线性监测型模型验证RMSE": nonlinear_monitor_info["validation_rmse"],
                "非线性监测模型套件验证RMSE": {
                    name: info["validation_rmse"] for name, info in nonlinear_monitor_suite_infos.items()
                },
                "监测型岭回归预测权重": monitor_blend_info["ridge_weight"],
                "监测型CO_lag1基线权重": monitor_blend_info["lag1_weight"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    write_docs(
        metrics_table,
        process_model_comparison,
        baseline_comparison,
        nonlinear_monitor_comparison,
        lag_validation_table,
        lag_table,
        lag_summary,
        importance_process,
        predictions,
        cleaning_report,
        process_model,
        monitor_model,
        ablation_table,
        sensitivity_process_table,
        sensitivity_monitor_table,
        process_perm,
    )

    print("第2问建模完成")
    print("建模口径：", variant_name)
    print(metrics_table.to_string(index=False))
    print("时滞合理性验证")
    print(lag_validation_table.to_string(index=False))
    print("输出目录：", RESULT_DIR)
    return {
        "variant_name": variant_name,
        "result_dir": RESULT_DIR,
        "mid_dir": MID_DIR,
        "result_data_dir": RESULT_DATA_DIR,
        "fig_dir": FIG_DIR,
        "metrics_table": metrics_table,
        "process_model_comparison": process_model_comparison,
        "baseline_comparison": baseline_comparison,
        "nonlinear_monitor_comparison": nonlinear_monitor_comparison,
        "lag_validation_table": lag_validation_table,
        "lag_table": lag_table,
        "predictions": predictions,
        "cleaning_report": cleaning_report,
        "outlier_report": outlier_report,
    }


def metric_row_for_summary(result: dict, model_name: str) -> dict:
    table = result["metrics_table"]
    row = table[(table["数据集"] == "测试集") & (table["模型"] == model_name)].iloc[0]
    return {
        "建模口径": result["variant_name"],
        "模型": model_name,
        "测试样本数": int(row["样本数"]),
        "测试RMSE": float(row["RMSE"]),
        "测试MAPE_%": float(row["MAPE_%"]),
        "测试R2": float(row["R2"]),
        "测试MAE": float(row["MAE"]),
    }


def write_dual_variant_summary(raw_result: dict, corrected_result: dict) -> None:
    summary_rows = []
    for result in [raw_result, corrected_result]:
        for model_name in [
            "过程变量模型",
            "基线1_CO_lag1持续性模型",
            "基线2_历史CO_AR模型",
            BASELINE3_MODEL_NAME,
            "残差校正监测型模型_对比",
            "非线性监测型模型",
            "非线性监测_LightGBM",
            "非线性监测_XGBoost",
            "非线性监测_梯度提升树",
            "非线性监测_随机森林",
        ]:
            if model_name in set(result["metrics_table"]["模型"]):
                summary_rows.append(metric_row_for_summary(result, model_name))
    summary = pd.DataFrame(summary_rows)
    comparison_dir = ROOT / "result" / "第2问"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(comparison_dir / "双口径模型评价对比.csv", index=False, encoding="utf-8-sig")

    raw_outlier = raw_result["outlier_report"]
    corrected_outlier = corrected_result["outlier_report"]
    raw_monitor = metric_row_for_summary(raw_result, BASELINE3_MODEL_NAME)
    corrected_monitor = metric_row_for_summary(corrected_result, BASELINE3_MODEL_NAME)
    delta_rmse = corrected_monitor["测试RMSE"] - raw_monitor["测试RMSE"]
    delta_r2 = corrected_monitor["测试R2"] - raw_monitor["测试R2"]

    summary_md = dataframe_to_markdown(summary, float_digits=4)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    def detail_without_title(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if lines and lines[0].startswith("# "):
            return "\n".join(lines[1:]).lstrip()
        return text

    corrected_detail_md = detail_without_title(corrected_result["result_dir"] / "第2问.md")
    dual_result_md = f"""### 6.1 原始 CO 与异常修正 CO 双口径结果

本次按同一套特征、时滞和模型流程生成两套结果，用于检验 CO 异常值修正对模型评价和影响规律判断的影响：

- **原始数据建模**：保留原始 CO，不删除、不修正样本。
- **异常修正后建模**：不删除样本；将判定为异常的 CO 点置空后，用相邻正常点线性插值修正。

CO 异常修正规则采用全局 IQR 界限和局部 Hampel 型孤立尖峰/跌落联合判断。异常修正口径共修正 {corrected_outlier.get("修正点数", 0)} 个 CO 点，占 {corrected_outlier.get("修正比例_%", 0.0):.3f}%；原始口径修正点数为 {raw_outlier.get("修正点数", 0)}。

双口径测试集对比如下：

{summary_md}

监测型主模型在异常修正后相对原始口径的 RMSE 变化为 {delta_rmse:.4f}，R2 变化为 {delta_r2:.4f}。正值表示指标增大，负值表示指标降低。

两套结果的输出位置如下：

| 口径 | 结果目录 | 中间数据目录 | 图目录 |
|---|---|---|---|
| 原始数据建模 | `result/第2问/{RAW_VARIANT_NAME}` | `data/第2问/{RAW_VARIANT_NAME}/中间数据` | `figure/第2问/{RAW_VARIANT_NAME}/基础图` |
| 异常修正后建模 | `result/第2问/{CORRECTED_VARIANT_NAME}` | `data/第2问/{CORRECTED_VARIANT_NAME}/中间数据` | `figure/第2问/{CORRECTED_VARIANT_NAME}/基础图` |

两种口径共用的风箱位置与最佳时滞图见 `figure/第2问/共用图`。

异常修正明细见 `data/第2问/{CORRECTED_VARIANT_NAME}/中间数据/CO异常修正明细.csv`，修正报告见 `data/第2问/{CORRECTED_VARIANT_NAME}/中间数据/CO异常修正报告.json`。

原始口径完整说明见 `result/第2问/{RAW_VARIANT_NAME}/第2问.md`。以下详细结果采用异常修正后建模作为推荐口径。

### 6.2 推荐口径详细结果
"""
    if "## 6. 结果\n\n" in corrected_detail_md:
        corrected_detail_md = corrected_detail_md.replace("## 6. 结果\n\n", f"## 6. 结果\n\n{dual_result_md}\n", 1)
    else:
        corrected_detail_md = corrected_detail_md + f"\n\n## 原始 CO 与异常修正 CO 双口径结果\n\n{dual_result_md}\n"

    md = f"""# 第2问

{corrected_detail_md}
"""
    (ROOT / "第2问.md").write_text(md, encoding="utf-8")
    (comparison_dir / "第2问.md").write_text(md, encoding="utf-8")
    (comparison_dir / "第2问_双口径对比.md").write_text(md, encoding="utf-8")
    (comparison_dir / "第2问（详细但是没整合版本）.md").write_text(md, encoding="utf-8")
    (comparison_dir / "第2问（整合版本).md").write_text(md, encoding="utf-8")


def mirror_variant_to_legacy_outputs(result: dict) -> None:
    legacy_pairs = [
        (result["mid_dir"], ROOT / "data" / "第2问" / "中间数据"),
        (result["result_data_dir"], ROOT / "data" / "第2问" / "结果数据"),
    ]
    for src, dst in legacy_pairs:
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    duplicate_legacy_fig_dir = ROOT / "figure" / "第2问" / "基础图"
    if duplicate_legacy_fig_dir.exists():
        shutil.rmtree(duplicate_legacy_fig_dir)

    legacy_result_dir = ROOT / "result" / "第2问"
    legacy_result_dir.mkdir(parents=True, exist_ok=True)
    for src_file in result["result_dir"].iterdir():
        if src_file.is_file():
            shutil.copy2(src_file, legacy_result_dir / src_file.name)


def main() -> None:
    ensure_dirs()
    copy_sources()

    raw_data, source_header, header_row = load_source_workbook()
    raw_result = run_modeling_variant(raw_data, RAW_VARIANT_NAME, enable_co_outlier_correction=False)
    corrected_result = run_modeling_variant(raw_data, CORRECTED_VARIANT_NAME, enable_co_outlier_correction=True)
    make_windbox_lag_position_plot(
        SHARED_FIG_DIR / "图2-3_风箱位置与最佳时滞.svg",
        corrected_result["lag_table"],
        "Windbox position vs selected lag",
    )
    mirror_variant_to_legacy_outputs(corrected_result)
    write_dual_variant_summary(raw_result, corrected_result)


if __name__ == "__main__":
    main()
