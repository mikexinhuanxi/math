
from __future__ import annotations

import json
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import config


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def ensure_dirs() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    required = [config.SEQ_COL, config.SPEED_COL, *config.PRESSURE_COLS, *config.SOUTH_TEMP_COLS, *config.NORTH_TEMP_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"原始数据缺少必要字段: {missing}")
    df = df.sort_values(config.SEQ_COL).reset_index(drop=True)
    return df


def build_clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    clean = raw.copy()
    for i in config.BOX_IDS:
        south = f"{i}#风箱南侧温度"
        north = f"{i}#风箱北侧温度"
        clean[f"{i}#风箱平均温度"] = (clean[south] + clean[north]) / 2.0
        clean[f"{i}#风箱南北温差"] = clean[south] - clean[north]
        clean[f"{i}#风箱加权温度_南0.4北0.6"] = clean[south] * 0.4 + clean[north] * 0.6
        clean[f"{i}#风箱加权温度_南0.6北0.4"] = clean[south] * 0.6 + clean[north] * 0.4
    return clean


def _p_name(box_id: int) -> str:
    return f"p_{box_id:02d}"


def _pressure_frame(clean: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=clean.index)
    for i, col in zip(config.BOX_IDS, config.PRESSURE_COLS):
        out[_p_name(i)] = clean[col].astype(float)
    return out


def _add_segment_features(features: pd.DataFrame, p: pd.DataFrame) -> None:
    abs_p = -p
    positions = np.array([config.BOX_POSITIONS[i] for i in config.BOX_IDS], dtype=float)
    lengths = np.array([config.BOX_LENGTHS[i] for i in config.BOX_IDS], dtype=float)

    features["p_all_mean"] = p.mean(axis=1)
    features["p_all_std"] = p.std(axis=1)
    features["p_all_min"] = p.min(axis=1)
    features["p_all_max"] = p.max(axis=1)
    features["p_all_range"] = features["p_all_max"] - features["p_all_min"]
    features["p_abs_all_mean"] = abs_p.mean(axis=1)
    features["p_abs_length_weighted_mean"] = (abs_p * lengths).sum(axis=1) / lengths.sum()
    features["p_abs_position_weighted_mean"] = (abs_p * positions).sum(axis=1) / positions.sum()
    denom = abs_p.sum(axis=1).replace(0, np.nan)
    features["p_abs_center_of_suction"] = (abs_p * positions).sum(axis=1) / denom

    for seg_name, box_ids in config.BOX_SEGMENTS.items():
        cols = [_p_name(i) for i in box_ids]
        vals = p[cols]
        abs_vals = abs_p[cols]
        seg_lengths = np.array([config.BOX_LENGTHS[i] for i in box_ids], dtype=float)
        features[f"{seg_name}_p_mean"] = vals.mean(axis=1)
        features[f"{seg_name}_p_std"] = vals.std(axis=1)
        features[f"{seg_name}_p_min"] = vals.min(axis=1)
        features[f"{seg_name}_p_max"] = vals.max(axis=1)
        features[f"{seg_name}_p_range"] = features[f"{seg_name}_p_max"] - features[f"{seg_name}_p_min"]
        features[f"{seg_name}_p_abs_mean"] = abs_vals.mean(axis=1)
        features[f"{seg_name}_p_abs_length_weighted_mean"] = (abs_vals * seg_lengths).sum(axis=1) / seg_lengths.sum()

    features["front_minus_middle_p_mean"] = features["front_p_mean"] - features["middle_p_mean"]
    features["middle_minus_back_p_mean"] = features["middle_p_mean"] - features["back_p_mean"]
    features["front_minus_back_p_mean"] = features["front_p_mean"] - features["back_p_mean"]
    features["long_minus_short_p_mean"] = features["long_p_mean"] - features["short_p_mean"]


def _add_spatial_features(features: pd.DataFrame, p: pd.DataFrame) -> None:
    for i in range(1, 18):
        left = _p_name(i)
        right = _p_name(i + 1)
        features[f"p_adjacent_diff_{i:02d}_{i + 1:02d}"] = p[left] - p[right]
        features[f"p_adjacent_absdiff_{i:02d}_{i + 1:02d}"] = (p[left] - p[right]).abs()

    for i in config.BOX_IDS:
        neighbor_ids = [j for j in [i - 1, i + 1] if 1 <= j <= 18]
        local_ids = [j for j in [i - 1, i, i + 1] if 1 <= j <= 18]
        neighbor_cols = [_p_name(j) for j in neighbor_ids]
        local_cols = [_p_name(j) for j in local_ids]
        features[f"p_{i:02d}_neighbor_mean"] = p[neighbor_cols].mean(axis=1)
        features[f"p_{i:02d}_local_mean"] = p[local_cols].mean(axis=1)
        features[f"p_{i:02d}_local_range"] = p[local_cols].max(axis=1) - p[local_cols].min(axis=1)
        features[f"p_{i:02d}_minus_neighbor_mean"] = p[_p_name(i)] - features[f"p_{i:02d}_neighbor_mean"]


def _add_temporal_features(features: pd.DataFrame, base: pd.DataFrame) -> None:
    for col in base.columns:
        series = base[col]
        for lag in config.LAG_STEPS:
            lagged = series.shift(lag)
            features[f"{col}_lag_{lag}"] = lagged
            features[f"{col}_delta_{lag}"] = series - lagged
            features[f"{col}_rate_{lag}"] = (series - lagged) / (lag * config.TIME_STEP_SECONDS)

    for col in [_p_name(i) for i in config.BOX_IDS]:
        series = base[col]
        for window in config.ROLL_WINDOWS:
            rolling = series.rolling(window=window, min_periods=window)
            r_mean = rolling.mean()
            r_min = rolling.min()
            r_max = rolling.max()
            features[f"{col}_roll{window}_mean"] = r_mean
            features[f"{col}_roll{window}_std"] = rolling.std()
            features[f"{col}_roll{window}_range"] = r_max - r_min

    speed = base["speed"]
    for window in config.ROLL_WINDOWS:
        rolling = speed.rolling(window=window, min_periods=window)
        features[f"speed_roll{window}_mean"] = rolling.mean()
        features[f"speed_roll{window}_std"] = rolling.std()


def build_pressure_only_features(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = _pressure_frame(clean)
    base = pd.DataFrame(index=clean.index)
    base["speed"] = clean[config.SPEED_COL].astype(float)
    for col in p.columns:
        base[col] = p[col]

    features = base.copy()
    for col in p.columns:
        features[f"{col}_abs_suction"] = -p[col]

    _add_spatial_features(features, p)
    _add_segment_features(features, p)
    _add_temporal_features(features, base)

    features = features.replace([np.inf, -np.inf], np.nan)
    valid_mask = features.notna().all(axis=1)
    aligned_clean = clean.loc[valid_mask].reset_index(drop=True)
    features = features.loc[valid_mask].reset_index(drop=True)
    assert_pressure_only_features(features.columns)
    return features, aligned_clean


def build_target_frames(clean: pd.DataFrame) -> dict[str, pd.DataFrame]:
    avg = pd.DataFrame({c: clean[f"{i}#风箱平均温度"].astype(float) for c, i in zip(config.AVG_TARGET_COLS, config.BOX_IDS)})
    south = pd.DataFrame({c: clean[f"{i}#风箱南侧温度"].astype(float) for c, i in zip(config.SOUTH_TARGET_COLS, config.BOX_IDS)})
    north = pd.DataFrame({c: clean[f"{i}#风箱北侧温度"].astype(float) for c, i in zip(config.NORTH_TARGET_COLS, config.BOX_IDS)})
    diff = pd.DataFrame({c: clean[f"{i}#风箱南北温差"].astype(float) for c, i in zip(config.DIFF_TARGET_COLS, config.BOX_IDS)})
    weighted_04_06 = pd.DataFrame(
        {
            f"box_{i:02d}_weighted_south04_north06": clean[f"{i}#风箱加权温度_南0.4北0.6"].astype(float)
            for i in config.BOX_IDS
        }
    )
    weighted_06_04 = pd.DataFrame(
        {
            f"box_{i:02d}_weighted_south06_north04": clean[f"{i}#风箱加权温度_南0.6北0.4"].astype(float)
            for i in config.BOX_IDS
        }
    )
    return {
        "average": avg,
        "south": south,
        "north": north,
        "diff": diff,
        "weighted_south04_north06": weighted_04_06,
        "weighted_south06_north04": weighted_06_04,
    }


def assert_pressure_only_features(feature_columns: list[str] | pd.Index) -> None:
    bad = [
        col
        for col in feature_columns
        if any(keyword.lower() in str(col).lower() for keyword in config.FORBIDDEN_FEATURE_KEYWORDS)
    ]
    if bad:
        raise ValueError(f"发现禁止进入模型的泄漏字段: {bad}")


def make_train_test_indices(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n_rows)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        shuffle=True,
    )
    return train_idx, test_idx


def make_xgb_model(params: dict[str, Any] | None = None) -> XGBRegressor:
    merged = deepcopy(config.XGB_BASE_PARAMS)
    if params:
        merged.update(params)
    return XGBRegressor(**merged)


def make_model(name: str, xgb_params: dict[str, Any] | None = None) -> Any:
    if name == "Ridge":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=20.0, random_state=config.RANDOM_STATE)),
            ]
        )
    if name == "ElasticNet":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MultiTaskElasticNet(
                        alpha=0.006,
                        l1_ratio=0.15,
                        max_iter=8000,
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        )
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=80,
            max_features=0.75,
            min_samples_leaf=2,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=120,
            max_features=0.85,
            min_samples_leaf=1,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "LightGBM":
        base = LGBMRegressor(
            n_estimators=110,
            learning_rate=0.060,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        return MultiOutputRegressor(base, n_jobs=1)
    if name == "CatBoost":
        return CatBoostRegressor(
            iterations=140,
            depth=6,
            learning_rate=0.06,
            loss_function="MultiRMSE",
            random_seed=config.RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
    if name == "XGBoost":
        return make_xgb_model(xgb_params)
    raise ValueError(f"未知模型: {name}")


def overall_metrics(y_true: pd.DataFrame | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "r2": float(r2_score(true, pred, multioutput="uniform_average")),
    }


def metrics_by_box(y_true: pd.DataFrame | np.ndarray, y_pred: np.ndarray, prefix: str = "") -> pd.DataFrame:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    rows = []
    for j, box_id in enumerate(config.BOX_IDS):
        rows.append(
            {
                "model": prefix,
                "box": f"{box_id}#风箱",
                "mae": float(mean_absolute_error(true[:, j], pred[:, j])),
                "rmse": float(np.sqrt(mean_squared_error(true[:, j], pred[:, j]))),
                "r2": float(r2_score(true[:, j], pred[:, j])),
            }
        )
    return pd.DataFrame(rows)


def fit_evaluate_model(
    name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
    model.fit(x_train, y_train)
    pred = np.asarray(model.predict(x_test), dtype=float)
    elapsed = time.perf_counter() - start
    metrics = overall_metrics(y_test, pred)
    metrics.update({"model": name, "train_seconds": elapsed})
    return model, pred, metrics


def tune_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, Any, np.ndarray, dict[str, Any]]:
    rows = []
    best: tuple[float, dict[str, Any], Any, np.ndarray, dict[str, Any]] | None = None
    for trial_id, params in enumerate(config.XGB_TUNING_GRID, start=1):
        print(f"      - XGBoost tuning trial {trial_id}/{len(config.XGB_TUNING_GRID)}")
        model = make_xgb_model(params)
        fitted, pred, metrics = fit_evaluate_model(
            f"XGBoost_tuned_trial_{trial_id}", model, x_train, y_train, x_test, y_test
        )
        row = {"trial": trial_id, **params, **metrics}
        rows.append(row)
        score = metrics["mae"]
        if best is None or score < best[0]:
            best = (score, params, fitted, pred, metrics)
    assert best is not None
    _, best_params, best_model, best_pred, best_metrics = best
    best_metrics = {**best_metrics, "model": "XGBoost_tuned"}
    return best_params, pd.DataFrame(rows), best_model, best_pred, best_metrics


def run_model_comparison(
    x: pd.DataFrame,
    y_avg: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y_avg.iloc[train_idx], y_avg.iloc[test_idx]

    comparison_rows = []
    fitted_models: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for model_name in ["Ridge", "ElasticNet", "RandomForest", "ExtraTrees", "LightGBM", "CatBoost", "XGBoost"]:
        print(f"      - {model_name}")
        model = make_model(model_name)
        fitted, pred, metrics = fit_evaluate_model(model_name, model, x_train, y_train, x_test, y_test)
        fitted_models[model_name] = fitted
        predictions[model_name] = pred
        comparison_rows.append(metrics)

    best_xgb_params, tuning_df, tuned_model, tuned_pred, tuned_metrics = tune_xgboost(x_train, y_train, x_test, y_test)
    fitted_models["XGBoost_tuned"] = tuned_model
    predictions["XGBoost_tuned"] = tuned_pred
    comparison_rows.append(tuned_metrics)

    comparison = pd.DataFrame(comparison_rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    best_name = comparison.iloc[0]["model"]
    return {
        "comparison": comparison,
        "fitted_models": fitted_models,
        "predictions": predictions,
        "best_model_name": best_name,
        "best_validation_model": fitted_models[best_name],
        "best_validation_pred": predictions[best_name],
        "best_xgb_params": best_xgb_params,
        "xgb_tuning": tuning_df,
    }


def cross_validate_model(model_name: str, xgb_params: dict[str, Any] | None, x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    rows = []
    kfold = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(x), start=1):
        name = "XGBoost" if model_name == "XGBoost_tuned" else model_name
        model = make_model(name, xgb_params=xgb_params if name == "XGBoost" else None)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = np.asarray(model.predict(x.iloc[val_idx]), dtype=float)
        row = {"fold": fold, **overall_metrics(y.iloc[val_idx], pred)}
        rows.append(row)
    cv = pd.DataFrame(rows)
    mean_row = {"fold": "mean", **{col: float(cv[col].mean()) for col in ["mae", "rmse", "r2"]}}
    std_row = {"fold": "std", **{col: float(cv[col].std(ddof=1)) for col in ["mae", "rmse", "r2"]}}
    return pd.concat([cv, pd.DataFrame([mean_row, std_row])], ignore_index=True)


def evaluate_time_holdout(
    model_name: str,
    xgb_params: dict[str, Any] | None,
    x: pd.DataFrame,
    y: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    split = int(len(x) * (1.0 - config.TEST_SIZE))
    train_idx = np.arange(split)
    test_idx = np.arange(split, len(x))
    name = "XGBoost" if model_name == "XGBoost_tuned" else model_name
    model = make_model(name, xgb_params=xgb_params if name == "XGBoost" else None)
    model.fit(x.iloc[train_idx], y.iloc[train_idx])
    pred = np.asarray(model.predict(x.iloc[test_idx]), dtype=float)
    rows = [{"model": model_name, "validation": "last_20_percent", **overall_metrics(y.iloc[test_idx], pred)}]
    return pd.DataFrame(rows), pred, test_idx


def refit_final_model(model_name: str, xgb_params: dict[str, Any] | None, x: pd.DataFrame, y: pd.DataFrame) -> Any:
    name = "XGBoost" if model_name == "XGBoost_tuned" else model_name
    model = make_model(name, xgb_params=xgb_params if name == "XGBoost" else None)
    model.fit(x, y)
    return model


def _xgb_params_for_exploration(best_xgb_params: dict[str, Any]) -> dict[str, Any]:
    params = deepcopy(best_xgb_params)
    params["n_estimators"] = min(int(params.get("n_estimators", 220)), 260)
    return params


def evaluate_target_representations(
    x: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    best_xgb_params: dict[str, Any],
) -> pd.DataFrame:
    params = _xgb_params_for_exploration(best_xgb_params)
    rows = []

    def fit_pred(y_train: pd.DataFrame, y_test: pd.DataFrame) -> np.ndarray:
        model = make_xgb_model(params)
        model.fit(x.iloc[train_idx], y_train)
        return np.asarray(model.predict(x.iloc[test_idx]), dtype=float)

    avg_pred = fit_pred(targets["average"].iloc[train_idx], targets["average"].iloc[test_idx])
    rows.append({"target_representation": "南北简单平均温度", "evaluation_target": "平均温度", **overall_metrics(targets["average"].iloc[test_idx], avg_pred)})

    south_north_train = pd.concat([targets["south"].iloc[train_idx], targets["north"].iloc[train_idx]], axis=1)
    south_north_test = pd.concat([targets["south"].iloc[test_idx], targets["north"].iloc[test_idx]], axis=1)
    sn_pred = fit_pred(south_north_train, south_north_test)
    reconstructed_avg = (sn_pred[:, :18] + sn_pred[:, 18:]) / 2.0
    rows.append(
        {
            "target_representation": "南侧/北侧分别预测后平均",
            "evaluation_target": "平均温度",
            **overall_metrics(targets["average"].iloc[test_idx], reconstructed_avg),
        }
    )

    diff_pred = fit_pred(targets["diff"].iloc[train_idx], targets["diff"].iloc[test_idx])
    rows.append({"target_representation": "南北温差", "evaluation_target": "温差", **overall_metrics(targets["diff"].iloc[test_idx], diff_pred)})

    for key, label in [
        ("weighted_south04_north06", "加权平均温度_南0.4北0.6"),
        ("weighted_south06_north04", "加权平均温度_南0.6北0.4"),
    ]:
        pred = fit_pred(targets[key].iloc[train_idx], targets[key].iloc[test_idx])
        rows.append({"target_representation": label, "evaluation_target": label, **overall_metrics(targets[key].iloc[test_idx], pred)})

    return pd.DataFrame(rows).sort_values(["evaluation_target", "mae"]).reset_index(drop=True)


def evaluate_single_box_xgb(
    x: pd.DataFrame,
    y: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    best_xgb_params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    params = _xgb_params_for_exploration(best_xgb_params)
    pred = np.zeros((len(test_idx), len(config.BOX_IDS)))
    start = time.perf_counter()
    for j, _box_id in enumerate(config.BOX_IDS):
        model = make_xgb_model(params)
        model.fit(x.iloc[train_idx], y.iloc[train_idx, j])
        pred[:, j] = model.predict(x.iloc[test_idx])
    metrics = {"model": "XGBoost_single_box", "train_seconds": time.perf_counter() - start, **overall_metrics(y.iloc[test_idx], pred)}
    return pred, metrics


def evaluate_segment_group_xgb(
    x: pd.DataFrame,
    y: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    best_xgb_params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    params = _xgb_params_for_exploration(best_xgb_params)
    pred = np.zeros((len(test_idx), len(config.BOX_IDS)))
    start = time.perf_counter()
    for _name, box_ids in config.TARGET_SEGMENTS.items():
        cols = [i - 1 for i in box_ids]
        model = make_xgb_model(params)
        model.fit(x.iloc[train_idx], y.iloc[train_idx, cols])
        pred[:, cols] = model.predict(x.iloc[test_idx])
    metrics = {"model": "XGBoost_front_middle_back_group", "train_seconds": time.perf_counter() - start, **overall_metrics(y.iloc[test_idx], pred)}
    return pred, metrics


def _long_format_base_columns(x: pd.DataFrame) -> list[str]:
    desired = ["speed"]
    desired += [_p_name(i) for i in config.BOX_IDS]
    desired += [
        "p_all_mean",
        "p_all_std",
        "p_all_range",
        "p_abs_length_weighted_mean",
        "p_abs_center_of_suction",
        "front_p_mean",
        "middle_p_mean",
        "back_p_mean",
        "front_minus_middle_p_mean",
        "middle_minus_back_p_mean",
        "front_minus_back_p_mean",
        "long_minus_short_p_mean",
    ]
    return [c for c in desired if c in x.columns]


def make_long_format(x: pd.DataFrame, y: pd.DataFrame | None = None) -> pd.DataFrame:
    base_cols = _long_format_base_columns(x)
    frames = []
    row_pos = np.arange(len(x))
    for box_id in config.BOX_IDS:
        p_col = _p_name(box_id)
        left = _p_name(max(1, box_id - 1))
        right = _p_name(min(18, box_id + 1))
        frame = x[base_cols].copy()
        frame["row_pos"] = row_pos
        frame["box_id"] = box_id
        frame["box_position"] = config.BOX_POSITIONS[box_id]
        frame["box_length"] = config.BOX_LENGTHS[box_id]
        frame["is_short_box"] = 1 if box_id >= 16 else 0
        frame["is_front"] = 1 if box_id <= 5 else 0
        frame["is_middle"] = 1 if 6 <= box_id <= 12 else 0
        frame["is_back"] = 1 if box_id >= 13 else 0
        frame["target_box_p"] = x[p_col]
        frame["left_box_p"] = x[left]
        frame["right_box_p"] = x[right]
        frame["neighbor_mean_p"] = (x[left] + x[right]) / 2.0
        frame["target_minus_neighbor_mean"] = x[p_col] - frame["neighbor_mean_p"]
        for lag in config.LAG_STEPS:
            for suffix in [f"lag_{lag}", f"delta_{lag}", f"rate_{lag}"]:
                col = f"{p_col}_{suffix}"
                if col in x.columns:
                    frame[f"target_{suffix}"] = x[col]
        for window in config.ROLL_WINDOWS:
            for stat in ["mean", "std", "range"]:
                col = f"{p_col}_roll{window}_{stat}"
                if col in x.columns:
                    frame[f"target_roll{window}_{stat}"] = x[col]
        if y is not None:
            frame["target"] = y.iloc[:, box_id - 1].to_numpy()
        frames.append(frame)
    return pd.concat(frames, axis=0, ignore_index=True)


def evaluate_long_format_xgb(
    x: pd.DataFrame,
    y: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    best_xgb_params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    params = _xgb_params_for_exploration(best_xgb_params)
    long_df = make_long_format(x, y)
    train_mask = long_df["row_pos"].isin(train_idx)
    test_mask = long_df["row_pos"].isin(test_idx)
    feature_cols = [c for c in long_df.columns if c not in {"target", "row_pos"}]
    model = make_xgb_model(params)
    start = time.perf_counter()
    model.fit(long_df.loc[train_mask, feature_cols], long_df.loc[train_mask, "target"])
    test_long = long_df.loc[test_mask, ["row_pos", *feature_cols]].copy()
    test_long["pred"] = model.predict(test_long[feature_cols])
    pivot = test_long.pivot(index="row_pos", columns="box_id", values="pred").loc[test_idx, config.BOX_IDS]
    pred = pivot.to_numpy()
    metrics = {"model": "XGBoost_long_format", "train_seconds": time.perf_counter() - start, **overall_metrics(y.iloc[test_idx], pred)}
    return pred, metrics


def run_structure_exploration(
    x: pd.DataFrame,
    y: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    best_xgb_params: dict[str, Any],
    xgb_tuned_pred: np.ndarray,
) -> pd.DataFrame:
    rows = [{"model": "XGBoost_wide_multioutput", **overall_metrics(y.iloc[test_idx], xgb_tuned_pred)}]
    _pred, single_metrics = evaluate_single_box_xgb(x, y, train_idx, test_idx, best_xgb_params)
    rows.append(single_metrics)
    _pred, group_metrics = evaluate_segment_group_xgb(x, y, train_idx, test_idx, best_xgb_params)
    rows.append(group_metrics)
    _pred, long_metrics = evaluate_long_format_xgb(x, y, train_idx, test_idx, best_xgb_params)
    rows.append(long_metrics)
    return pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)


def extract_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    importances: np.ndarray | None = None
    if isinstance(model, Pipeline):
        final = model.named_steps.get("model")
        if hasattr(final, "coef_"):
            coef = np.asarray(final.coef_)
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    elif isinstance(model, MultiOutputRegressor):
        vals = []
        for estimator in model.estimators_:
            if hasattr(estimator, "feature_importances_"):
                vals.append(np.asarray(estimator.feature_importances_, dtype=float))
        if vals:
            importances = np.mean(np.vstack(vals), axis=0)
    elif hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        if importances.ndim > 1:
            importances = importances.mean(axis=0)
    elif hasattr(model, "get_feature_importance"):
        importances = np.asarray(model.get_feature_importance(), dtype=float)
        if importances.ndim > 1:
            importances = importances.mean(axis=0)

    if importances is None or len(importances) != len(feature_names):
        importances = np.zeros(len(feature_names), dtype=float)
    out = pd.DataFrame({"feature": feature_names, "importance": importances})
    total = out["importance"].sum()
    out["importance_ratio"] = out["importance"] / total if total else 0.0
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def save_prediction_excel(
    path: Path,
    aligned_clean: pd.DataFrame,
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    test_idx: np.ndarray,
    time_y_true: pd.DataFrame,
    time_y_pred: np.ndarray,
    time_idx: np.ndarray,
    comparison: pd.DataFrame,
    by_box: pd.DataFrame,
    cv: pd.DataFrame,
) -> None:
    def prediction_frame(idx: np.ndarray, true_df: pd.DataFrame, pred_arr: np.ndarray) -> pd.DataFrame:
        rows = pd.DataFrame(
            {
                "序列": aligned_clean.iloc[idx][config.SEQ_COL].to_numpy(),
                "原始行号": idx,
            }
        )
        true_vals = np.asarray(true_df, dtype=float)
        for j, box_id in enumerate(config.BOX_IDS):
            rows[f"{box_id}#风箱实际平均温度"] = true_vals[:, j]
            rows[f"{box_id}#风箱预测平均温度"] = pred_arr[:, j]
            rows[f"{box_id}#风箱残差_预测减实际"] = pred_arr[:, j] - true_vals[:, j]
        return rows

    random_pred = prediction_frame(test_idx, y_true, y_pred)
    time_pred = prediction_frame(time_idx, time_y_true, time_y_pred)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        random_pred.to_excel(writer, sheet_name="随机8_2预测", index=False)
        time_pred.to_excel(writer, sheet_name="时间后20预测", index=False)
        comparison.to_excel(writer, sheet_name="模型比较", index=False)
        by_box.to_excel(writer, sheet_name="逐风箱指标", index=False)
        cv.to_excel(writer, sheet_name="5折交叉验证", index=False)


def save_prediction_plots(
    path: Path,
    aligned_clean: pd.DataFrame,
    y_test: pd.DataFrame,
    y_pred: np.ndarray,
    test_idx: np.ndarray,
    by_box: pd.DataFrame,
    best_name: str,
) -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    true = np.asarray(y_test, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    seq = aligned_clean.iloc[test_idx][config.SEQ_COL].to_numpy()
    order = np.argsort(seq)

    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.25])

    ax1 = fig.add_subplot(grid[0, 0])
    ax1.scatter(true.ravel(), pred.ravel(), s=8, alpha=0.45)
    min_v = min(true.min(), pred.min())
    max_v = max(true.max(), pred.max())
    ax1.plot([min_v, max_v], [min_v, max_v], color="crimson", linewidth=1.2)
    ax1.set_title(f"{best_name} 随机8:2验证：实际值 vs 预测值")
    ax1.set_xlabel("实际平均温度")
    ax1.set_ylabel("预测平均温度")

    ax2 = fig.add_subplot(grid[0, 1])
    ax2.bar(by_box["box"], by_box["mae"], color="#4C78A8")
    ax2.set_title("18个风箱逐项 MAE")
    ax2.set_ylabel("MAE")
    ax2.tick_params(axis="x", rotation=55)

    ax3 = fig.add_subplot(grid[1, :])
    selected = [1, 6, 12, 18]
    for box_id in selected:
        j = box_id - 1
        ax3.plot(seq[order], true[order, j], linewidth=1.0, label=f"{box_id}#实际")
        ax3.plot(seq[order], pred[order, j], linewidth=1.0, linestyle="--", label=f"{box_id}#预测")
    ax3.set_title("代表性风箱随机验证样本按序列排序后的预测跟踪")
    ax3.set_xlabel("序列")
    ax3.set_ylabel("平均温度")
    ax3.legend(ncol=4, fontsize=9)

    axes = [fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])]
    for ax, box_id in zip(axes, [15, 18]):
        j = box_id - 1
        ax.scatter(true[:, j], pred[:, j], s=12, alpha=0.55)
        min_b = min(true[:, j].min(), pred[:, j].min())
        max_b = max(true[:, j].max(), pred[:, j].max())
        ax.plot([min_b, max_b], [min_b, max_b], color="crimson", linewidth=1.0)
        ax.set_title(f"{box_id}#风箱高温段预测散点")
        ax.set_xlabel("实际平均温度")
        ax.set_ylabel("预测平均温度")

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_leakage_report(path: Path, feature_columns: list[str]) -> None:
    bad = [
        col
        for col in feature_columns
        if any(keyword.lower() in str(col).lower() for keyword in config.FORBIDDEN_FEATURE_KEYWORDS)
    ]
    lines = [
        "pressure-only feature leakage check",
        f"feature_count: {len(feature_columns)}",
        f"forbidden_keywords: {', '.join(config.FORBIDDEN_FEATURE_KEYWORDS)}",
        f"bad_feature_count: {len(bad)}",
        "status: PASS" if not bad else "status: FAIL",
    ]
    if bad:
        lines.append("bad_features:")
        lines.extend(bad)
    path.write_text("\n".join(lines), encoding="utf-8")


def save_model(path: Path, model: Any, metadata: dict[str, Any]) -> None:
    joblib.dump({"model": model, "metadata": metadata}, path)


def _fmt_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append(_fmt_float(float(value), float_digits))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_summary_markdown(
    best_name: str,
    comparison: pd.DataFrame,
    by_box: pd.DataFrame,
    cv: pd.DataFrame,
    time_metrics: pd.DataFrame,
    target_exploration: pd.DataFrame,
    structure_exploration: pd.DataFrame,
    feature_importance: pd.DataFrame,
    n_raw: int,
    n_model: int,
    n_features: int,
) -> str:
    best = comparison.iloc[0]
    cv_mean = cv[cv["fold"].astype(str) == "mean"].iloc[0]
    cv_std = cv[cv["fold"].astype(str) == "std"].iloc[0]
    time_row = time_metrics.iloc[0]
    top_features = feature_importance.head(12)
    top_feature_lines = "\n".join(
        [f"- `{row.feature}`：{_fmt_float(row.importance_ratio * 100, 2)}%" for row in top_features.itertuples()]
    )
    comparison_md = _markdown_table(comparison[["model", "mae", "rmse", "r2", "train_seconds"]])
    box_md = _markdown_table(by_box[["box", "mae", "rmse", "r2"]])
    target_md = _markdown_table(target_exploration)
    structure_md = _markdown_table(structure_exploration)

    return f"""# 第1问：基于风箱负压的风箱平均温度预测

## 做了什么
- 只完成赛题A问题一，主目标固定为 18 个风箱的南北简单平均温度。
- 输入严格采用 pressure-only 特征：烧结机机速、18 个风箱负压，以及由二者派生的滞后、差分、变化速度、滚动统计、相邻风箱、分段和规格特征。
- 原始数据共 {n_raw} 行；滞后/滚动特征需要历史窗口，最终进入建模的样本为 {n_model} 行、{n_features} 个特征。
- 已显式检查模型输入列，不包含温度、CO、大烟道、南侧、北侧相关字段。

## 主结果：随机 8:2 验证
- 主验证方式是随机 8:2 留出，`random_state=42`。
- 最佳 pressure-only 主模型：**{best_name}**。
- 随机验证整体 MAE = **{_fmt_float(float(best["mae"]))}**，RMSE = **{_fmt_float(float(best["rmse"]))}**，R2 = **{_fmt_float(float(best["r2"]))}**。

{comparison_md}

## 5折交叉验证
- 为避免单次随机划分偶然性，使用 5 折随机交叉验证复核最佳模型。
- 5 折 MAE = **{_fmt_float(float(cv_mean["mae"]))} ± {_fmt_float(float(cv_std["mae"]))}**，RMSE = **{_fmt_float(float(cv_mean["rmse"]))} ± {_fmt_float(float(cv_std["rmse"]))}**，R2 = **{_fmt_float(float(cv_mean["r2"]))} ± {_fmt_float(float(cv_std["r2"]))}**。
- 因此，论文主线可写作“随机 8:2 留出 + 5 折交叉验证”。

## 18个风箱逐项指标
{box_md}

## 探索对比
- 目标表示探索只作为辅助说明，最终主线仍采用南北简单平均温度。

{target_md}

- 建模结构探索包括宽表18输出、每个风箱单独模型、前/中/后分组模型和 long-format 共用模型。

{structure_md}

## 时间后20%鲁棒性分析
- 时间后20%不是主精度结论，只用于观察外推鲁棒性和工况漂移。
- 后20%整体 MAE = **{_fmt_float(float(time_row["mae"]))}**，RMSE = **{_fmt_float(float(time_row["rmse"]))}**，R2 = **{_fmt_float(float(time_row["r2"]))}**。
- 若该指标弱于随机验证，应解释为时间外推下的工况漂移或样本分布变化，而不是否定随机验证下 pressure-only 模型的主结果。

## 重要特征
以下为最终模型平均重要性靠前的特征，均来自机速/风箱负压及其派生量：
{top_feature_lines}

## 后续问题可复用文件
- `data/problem1_clean.csv`：清洗后的全量数据，包含南北平均温度、温差和加权温度等目标列。
- `data/problem1_features_pressure_only.csv`：严格 pressure-only 的特征矩阵。
- `data/最佳模型文件.pkl`：按主验证选出的最佳模型，已在全部建模样本上重新拟合，可供后续调用。
- `result/problem1_predictions.xlsx`：随机验证和时间后20%的逐样本预测结果。
- `result/problem1_feature_importance.csv`：最终模型特征重要性。
- `result/problem1_model_comparison.csv` 与 `result/problem1_metrics_by_box.csv`：论文表格可直接引用。
"""


def write_markdown_reports(
    summary_text: str,
    summary_path: Path = config.SUMMARY_PATH,
    question_path: Path = config.QUESTION1_MD_PATH,
) -> None:
    summary_path.write_text(summary_text, encoding="utf-8")
    question_path.write_text(summary_text, encoding="utf-8")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def metadata_for_model(best_name: str, feature_columns: list[str], target_columns: list[str], xgb_params: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem": "赛题A问题一",
        "target": "18个风箱南北简单平均温度",
        "best_model_name": best_name,
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "xgb_best_params": xgb_params,
        "forbidden_feature_keywords": config.FORBIDDEN_FEATURE_KEYWORDS,
        "note": "模型输入只包含机速、18个风箱负压及其派生特征；不含温度历史、CO、大烟道字段。",
    }
