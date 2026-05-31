from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config
import problem1_lib as lib


SUMMARY_PATH = config.RESULT_DIR / "problem1_history_pressure_ablation.csv"
BY_BOX_PATH = config.RESULT_DIR / "problem1_history_pressure_ablation_by_box.csv"
PRED_PATH = config.RESULT_DIR / "problem1_history_pressure_ablation_predictions.xlsx"
MD_PATH = config.RESULT_DIR / "problem1_history_pressure_ablation_summary.md"
TEMP_LAGS = [1, 5, 15, 30, 60]
TEMP_DELTA_LAGS = [1, 5, 15, 30]


def load_base() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = lib.load_raw_data(config.RAW_DATA_PATH)
    clean = lib.build_clean_data(raw)
    x_pressure, aligned_clean = lib.build_pressure_only_features(clean)
    y = lib.build_target_frames(aligned_clean)["average"]
    return x_pressure, y, aligned_clean


def compute_psi(x: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> pd.Series:
    vals = {
        col: lib._population_stability_index(x.iloc[train_idx][col].to_numpy(), x.iloc[test_idx][col].to_numpy())
        for col in x.columns
    }
    return pd.Series(vals).sort_values(ascending=False)


def add_relative_pressure_features(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    p_cols = [f"p_{i:02d}" for i in config.BOX_IDS if f"p_{i:02d}" in out.columns]
    p = out[p_cols]
    p_mean = p.mean(axis=1)
    abs_sum = (-p).sum(axis=1).replace(0, np.nan)
    for box_id in config.BOX_IDS:
        col = f"p_{box_id:02d}"
        if col not in out.columns:
            continue
        out[f"{col}_minus_all_mean"] = out[col] - p_mean
        out[f"{col}_abs_suction_share"] = (-out[col]) / abs_sum
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_relative_pressure_drop_drift_features(x: pd.DataFrame) -> pd.DataFrame:
    train_idx, test_idx = lib.make_time_train_test_indices(len(x))
    psi = compute_psi(x, train_idx, test_idx)
    drop_cols = psi[psi >= 1.0].index.tolist()
    return add_relative_pressure_features(x).drop(columns=[c for c in drop_cols if c in x.columns])


def build_history_only_features(y: pd.DataFrame, aligned_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.DataFrame(index=y.index)
    for lag in TEMP_LAGS:
        shifted = y.shift(lag)
        shifted.columns = [f"{col}_lag_{lag}" for col in y.columns]
        features = pd.concat([features, shifted], axis=1)
    for lag in TEMP_DELTA_LAGS:
        delta = y.shift(1) - y.shift(lag)
        delta.columns = [f"{col}_hist_delta_1_{lag}" for col in y.columns]
        features = pd.concat([features, delta], axis=1)
    for seg_name, box_ids in config.TARGET_SEGMENTS.items():
        cols = [f"box_{i:02d}_avg_temp" for i in box_ids]
        for lag in [1, 5, 15, 30]:
            features[f"{seg_name}_temp_mean_lag_{lag}"] = y[cols].mean(axis=1).shift(lag)
            features[f"{seg_name}_temp_range_lag_{lag}"] = (y[cols].max(axis=1) - y[cols].min(axis=1)).shift(lag)
    valid = features.notna().all(axis=1)
    return (
        features.loc[valid].reset_index(drop=True),
        y.loc[valid].reset_index(drop=True),
        aligned_clean.loc[valid].reset_index(drop=True),
    )


def build_history_plus_pressure_features(
    x: pd.DataFrame,
    y: pd.DataFrame,
    aligned_clean: pd.DataFrame,
    lags: list[int] = TEMP_LAGS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = x.copy()
    for lag in lags:
        shifted = y.shift(lag)
        shifted.columns = [f"{col}_lag_{lag}" for col in y.columns]
        features = pd.concat([features, shifted], axis=1)
    for lag in TEMP_DELTA_LAGS:
        delta = y.shift(1) - y.shift(lag)
        delta.columns = [f"{col}_hist_delta_1_{lag}" for col in y.columns]
        features = pd.concat([features, delta], axis=1)
    for seg_name, box_ids in config.TARGET_SEGMENTS.items():
        cols = [f"box_{i:02d}_avg_temp" for i in box_ids]
        for lag in [1, 5, 15, 30]:
            features[f"{seg_name}_temp_mean_lag_{lag}"] = y[cols].mean(axis=1).shift(lag)
            features[f"{seg_name}_temp_range_lag_{lag}"] = (y[cols].max(axis=1) - y[cols].min(axis=1)).shift(lag)
    valid = features.notna().all(axis=1)
    return (
        features.loc[valid].reset_index(drop=True),
        y.loc[valid].reset_index(drop=True),
        aligned_clean.loc[valid].reset_index(drop=True),
    )


def lgbm_multi():
    if lib.LGBMRegressor is None:
        raise ImportError("LightGBM is not available.")
    base = lib.LGBMRegressor(
        n_estimators=90,
        learning_rate=0.06,
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


def metric_row(name: str, y_true: pd.DataFrame, pred: np.ndarray, seconds: float, detail: dict | None = None) -> dict:
    row = {"experiment": name, "train_seconds": seconds, **lib.overall_metrics(y_true, pred)}
    if detail:
        row.update(detail)
    print(f"      {name}: MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, R2={row['r2']:.4f}, {seconds:.1f}s", flush=True)
    return row


def by_box(name: str, y_true: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    return lib.metrics_by_box(y_true, pred, prefix=name).rename(columns={"model": "experiment"})


def direct_predict(x: pd.DataFrame, y: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    model = lgbm_multi()
    model.fit(x.iloc[train_idx], y.iloc[train_idx])
    return np.asarray(model.predict(x.iloc[test_idx]), dtype=float)


def delta_predict(x: pd.DataFrame, y: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    y_prev = y.shift(1)
    delta = y - y_prev
    train_mask = train_idx[train_idx > 0]
    model = lgbm_multi()
    model.fit(x.iloc[train_mask], delta.iloc[train_mask])
    delta_pred = np.asarray(model.predict(x.iloc[test_idx]), dtype=float)
    return y_prev.iloc[test_idx].to_numpy(dtype=float) + delta_pred


def persistence(y: pd.DataFrame, test_idx: np.ndarray, lag: int) -> np.ndarray:
    return y.shift(lag).iloc[test_idx].to_numpy(dtype=float)


def save_prediction(writer: pd.ExcelWriter, name: str, aligned: pd.DataFrame, test_idx: np.ndarray, y_true: pd.DataFrame, pred: np.ndarray) -> None:
    frame = pd.DataFrame({"序列": aligned.iloc[test_idx][config.SEQ_COL].to_numpy(), "原始行号": test_idx})
    true = y_true.to_numpy(dtype=float)
    for j, bid in enumerate(config.BOX_IDS):
        frame[f"{bid}#实际"] = true[:, j]
        frame[f"{bid}#预测"] = pred[:, j]
        frame[f"{bid}#残差_预测减实际"] = pred[:, j] - true[:, j]
    frame.to_excel(writer, sheet_name=name[:31], index=False)


def main() -> None:
    print("[1/4] 构造 history-only 与 history+pressure 特征")
    x_pressure, y0, aligned0 = load_base()
    x_rel = add_relative_pressure_drop_drift_features(x_pressure)
    x_hp, y_hp, aligned_hp = build_history_plus_pressure_features(x_rel, y0, aligned0)
    x_h, y_h, aligned_h = build_history_only_features(y0, aligned0)

    if len(x_h) != len(x_hp) or not y_h.equals(y_hp):
        raise ValueError("history-only 与 history+pressure 没有对齐到同一批样本")
    y = y_h
    aligned = aligned_h
    train_idx, test_idx = lib.make_time_train_test_indices(len(y))
    y_test = y.iloc[test_idx]

    rows: list[dict] = []
    boxes: list[pd.DataFrame] = []
    preds: dict[str, np.ndarray] = {}

    def record(name: str, producer: Callable[[], tuple[np.ndarray, dict | None]]) -> None:
        start = time.perf_counter()
        try:
            pred, detail = producer()
            elapsed = time.perf_counter() - start
            row = metric_row(name, y_test, pred, elapsed, detail)
            rows.append(row)
            boxes.append(by_box(name, y_test, pred))
            preds[name] = pred
        except Exception as exc:
            print(f"      {name}: ERROR {exc}", flush=True)
            rows.append({"experiment": name, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "train_seconds": np.nan, "error": repr(exc)})
        pd.DataFrame(rows).sort_values(["mae", "rmse"], na_position="last").to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
        if boxes:
            pd.concat(boxes, ignore_index=True).to_csv(BY_BOX_PATH, index=False, encoding="utf-8-sig")

    print("[2/4] 运行对照实验")
    record("persistence_lag1", lambda: (persistence(y, test_idx, 1), {"feature_group": "history_only"}))
    record("history_only_direct_lgbm", lambda: (direct_predict(x_h, y, train_idx, test_idx), {"feature_group": "history_only", "feature_count": x_h.shape[1]}))
    record("history_pressure_direct_lgbm", lambda: (direct_predict(x_hp, y, train_idx, test_idx), {"feature_group": "history_plus_pressure", "feature_count": x_hp.shape[1]}))
    record("history_only_delta_lgbm", lambda: (delta_predict(x_h, y, train_idx, test_idx), {"feature_group": "history_only", "feature_count": x_h.shape[1]}))
    record("history_pressure_delta_lgbm", lambda: (delta_predict(x_hp, y, train_idx, test_idx), {"feature_group": "history_plus_pressure", "feature_count": x_hp.shape[1]}))

    print("[3/4] 保存结果")
    result = pd.DataFrame(rows).sort_values(["mae", "rmse"], na_position="last").reset_index(drop=True)
    box_df = pd.concat(boxes, ignore_index=True)
    result.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    box_df.to_csv(BY_BOX_PATH, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(PRED_PATH, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="消融总表", index=False)
        box_df.to_excel(writer, sheet_name="逐风箱指标", index=False)
        for name, pred in preds.items():
            save_prediction(writer, name, aligned, test_idx, y_test, pred)

    print("[4/4] 写 Markdown 总结")
    best = result.iloc[0]
    hp_delta = result[result["experiment"].eq("history_pressure_delta_lgbm")].iloc[0]
    h_delta = result[result["experiment"].eq("history_only_delta_lgbm")].iloc[0]
    md = f"""# 历史温度 vs 历史温度+负压 消融实验

## 结论
- 最优实验：**{best["experiment"]}**
- 后20% MAE = **{best["mae"]:.4f}**，RMSE = **{best["rmse"]:.4f}**，R2 = **{best["r2"]:.4f}**
- 只用历史温度的 ΔT-LightGBM：MAE = **{h_delta["mae"]:.4f}**
- 历史温度 + 负压的 ΔT-LightGBM：MAE = **{hp_delta["mae"]:.4f}**
- 负压特征带来的 MAE 改变量：**{h_delta["mae"] - hp_delta["mae"]:.4f}**

## 实验总表
{lib._markdown_table(result[["experiment", "mae", "rmse", "r2", "train_seconds", "feature_group"]])}

## 说明
该实验均只使用历史温度 lag 和当前/历史负压特征，不使用当前或未来温度标签。ΔT 模型使用上一时刻真实温度还原当前温度，属于在线单步预测口径。
"""
    MD_PATH.write_text(md, encoding="utf-8")
    print(result[["experiment", "mae", "rmse", "r2", "feature_group"]].to_string(index=False))


if __name__ == "__main__":
    main()
