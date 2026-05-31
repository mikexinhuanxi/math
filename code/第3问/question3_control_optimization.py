# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
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

Q1_RESULT_DIR = ROOT / "result" / "第1问"
Q1_DATA_MID = ROOT / "data" / "第1问" / "中间数据"
Q1_DATA_OUT = ROOT / "data" / "第1问" / "结果数据"

Q2_RESULT_DIR = ROOT / "result" / "第2问"
Q2_MODEL_DIR = Q2_RESULT_DIR / "修正异常值数据建模"
Q2_DATA_MID = ROOT / "data" / "第2问" / "中间数据"

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
            shutil.copy2(src, dst)
        return True
    return False


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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
    save_csv(raw, DATA_RAW / "附件1_字段化原始数据.csv")
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
        "第1问报告": Q1_RESULT_DIR / "第1问.md",
        "第1问字段说明": Q1_RESULT_DIR / "风箱温度字段说明.md",
        "第1问模型schema": Q1_DATA_OUT / "q1_model_schema.json",
        "第1问模型特征": Q1_DATA_MID / "q1_model_features.csv",
        "第1问模型目标": Q1_DATA_MID / "q1_model_targets.csv",
        "第2问监测型模型": Q2_MODEL_DIR / "CO预测模型_监测型模型.json",
        "第2问模型接口说明": Q2_RESULT_DIR / "CO预测模型接口说明.md",
        "第2问输入输出字段清单": Q2_RESULT_DIR / "输入输出字段清单.md",
        "第2问数据配准说明": Q2_RESULT_DIR / "数据配准说明.md",
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

    @classmethod
    def fit_from_q1_outputs(cls, deps: dict[str, Path]) -> "Q1TemperatureLinker":
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
        )

    def _linked_feature_vector(self, intensity: np.ndarray, speed: float) -> tuple[np.ndarray, list[str]]:
        raw_pressure = -np.asarray(intensity, dtype=float)
        values = np.zeros(len(self.feature_names), dtype=float)
        unhandled: list[str] = []

        def box_value(box: int) -> float:
            return float(raw_pressure[box - 1])

        def pressure_std(vals: np.ndarray) -> float:
            return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        for idx, name in enumerate(self.feature_names):
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


def temperature_feature_map(model: LinearCOModel) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for name in model.feature_names:
        m = re.match(r"^(\d+)#风箱平均温度__lag\d+$", name)
        if m:
            mapping[int(m.group(1))] = name
    return mapping


def apply_plan_to_frame(frame: pd.DataFrame, intensity: np.ndarray, temperature: np.ndarray, model: LinearCOModel) -> pd.DataFrame:
    planned = frame.copy()
    for box_id, feature in pressure_feature_map(model).items():
        planned[feature] = float(intensity[box_id - 1])
    for box_id, feature in temperature_feature_map(model).items():
        planned[feature] = float(temperature[box_id - 1])
    return planned


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
            temp, _, _, _, clipped_count, unhandled = incremental_temperature_for_plan(
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
            temp, _, _, _, clipped_count, unhandled = incremental_temperature_for_plan(
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
            temp, _, _, _, clipped_count, unhandled = incremental_temperature_for_plan(
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
            if hasattr(control_model, "predict_standard_probability"):
                prob = float(control_model.predict_standard_probability(planned)[0])
            else:
                prob = float(np.clip((EMISSION_STANDARD + 400 - pred) / 800, 0.0, 1.0))
            cache[key] = {"pred_co": pred, "standard_probability": prob, "temperature": temp}
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


def make_historical_compliant_profile(cleaned: pd.DataFrame, latest_actual: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    compliant = cleaned[cleaned[TARGET_COL] <= EMISSION_STANDARD].copy()
    source = compliant if not compliant.empty else cleaned.copy()
    profile = np.array([float(source[f"{i}#风箱负压强度"].median()) for i in BOX_IDS], dtype=float)
    rows = []
    for i in BOX_IDS:
        s = source[f"{i}#风箱负压强度"].astype(float)
        all_s = cleaned[f"{i}#风箱负压强度"].astype(float)
        rows.append(
            {
                "风箱编号": i,
                "统计样本": "CO<=2800历史达标工况" if not compliant.empty else "全历史样本_无达标样本兜底",
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
                "最新实际强度": float(latest_actual[i - 1]),
                "达标中位数较最新变化": float(s.median() - latest_actual[i - 1]),
                "达标中位原始负压": -float(s.median()),
            }
        )
    stats = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "全样本数": int(len(cleaned)),
                "历史达标样本数": int(len(compliant)),
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
    compliant_mask = df[TARGET_COL] <= EMISSION_STANDARD
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
                达标样本数=(TARGET_COL, lambda s: int((s <= EMISSION_STANDARD).sum())),
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
) -> tuple[np.ndarray, float, np.ndarray, float, pd.DataFrame, dict[str, object]]:
    lows, highs = main_allowed_bounds(ranges)
    bounds = list(zip(lows, highs))
    widths = np.maximum(highs - lows, 1e-6)
    center = (lows + highs) / 2
    hist_low = historical_stats["达标强度P25"].to_numpy(dtype=float)
    hist_high = historical_stats["达标强度P75"].to_numpy(dtype=float)
    hist_median = historical_stats["达标强度中位数"].to_numpy(dtype=float)
    starts = [
        np.clip(latest_actual, lows, highs),
        np.clip(historical_profile, lows, highs),
        np.clip(hist_median, lows, highs),
        center,
        ranges["历史强度中位数"].to_numpy(dtype=float).clip(lows, highs),
    ]
    search_rows: list[dict[str, object]] = []

    def score_components(x: np.ndarray) -> dict[str, float]:
        scenario = evaluate_plan(np.asarray(x, dtype=float))
        pred = float(scenario["pred_co"])
        prob = float(scenario["standard_probability"])
        hist_penalty = interval_distance(x, hist_low, hist_high, widths)
        boundary_penalty = boundary_pressure_penalty(x, lows, highs)
        adjust_penalty = normalized_adjustment(x, latest_actual, lows, highs) / len(BOX_IDS)
        score = (pred - EMISSION_STANDARD) / 500.0 - 1.6 * prob + 1.2 * hist_penalty + 0.35 * boundary_penalty + 0.08 * adjust_penalty
        return {
            "低排放综合评分": float(score),
            "模型预测CO_mg_m3": pred,
            "达标倾向概率": prob,
            "历史达标IQR距离": hist_penalty,
            "边界惩罚": boundary_penalty,
            "当前工况标准化调节量": normalized_adjustment(x, latest_actual, lows, highs),
        }

    candidates: list[tuple[float, np.ndarray, dict[str, float]]] = []

    for start_id, start in enumerate(starts, start=1):
        result = minimize(
            lambda x: score_components(np.clip(np.asarray(x, dtype=float), lows, highs))["低排放综合评分"],
            np.asarray(start, dtype=float),
            method="Powell",
            bounds=bounds,
            options={"maxiter": 220, "xtol": 1e-4, "ftol": 1e-5, "disp": False},
        )
        x = np.clip(np.asarray(result.x if result.x is not None else start, dtype=float), lows, highs)
        comp = score_components(x)
        candidates.append((comp["低排放综合评分"], x.copy(), comp))
        search_rows.append(
            {
                "模型": "主模型_非线性低排放评分_Powell",
                "起点": start_id,
                "优化成功": bool(result.success),
                "优化消息": str(result.message),
                "迭代次数": int(getattr(result, "nit", 0)),
                "函数评估数": int(getattr(result, "nfev", 0)),
                **comp,
            }
        )

    de_result = differential_evolution(
        lambda x: score_components(np.asarray(x, dtype=float))["低排放综合评分"],
        bounds=bounds,
        maxiter=18,
        popsize=6,
        tol=0.01,
        polish=False,
        seed=42,
        updating="immediate",
        workers=1,
    )
    de_x = np.clip(np.asarray(de_result.x, dtype=float), lows, highs)
    de_comp = score_components(de_x)
    candidates.append((de_comp["低排放综合评分"], de_x.copy(), de_comp))
    search_rows.append(
        {
            "模型": "主模型_非线性低排放评分_差分进化",
            "起点": 0,
            "优化成功": bool(de_result.success),
            "优化消息": str(de_result.message),
            "迭代次数": int(getattr(de_result, "nit", 0)),
            "函数评估数": int(getattr(de_result, "nfev", 0)),
            **de_comp,
        }
    )

    if not candidates:
        raise RuntimeError("主模型未找到允许范围内的负压组合")

    best_score, final_plan, final_comp = min(candidates, key=lambda item: item[0])
    min_candidate = min(candidates, key=lambda item: item[2]["模型预测CO_mg_m3"])
    min_plan = min_candidate[1]
    min_co = float(min_candidate[2]["模型预测CO_mg_m3"])
    final_co = float(final_comp["模型预测CO_mg_m3"])

    search_df = pd.DataFrame(search_rows)
    save_csv(search_df, DATA_MID / "主模型_全局达标优化搜索记录.csv")
    save_csv(search_df, DATA_MID / "主模型_非线性低排放优化搜索记录.csv")
    diagnostics = {
        "main_range": "P05-P95",
        "model_role": "非线性低排放评分，不单独承担达标判定",
        "target_feasible": bool(final_co <= EMISSION_STANDARD and final_comp["达标倾向概率"] >= 0.5),
        "target_success": bool(final_co <= EMISSION_STANDARD and final_comp["达标倾向概率"] >= 0.5),
        "min_co": float(min_co),
        "final_co": float(final_co),
        "final_adjustment_norm": float(final_comp["当前工况标准化调节量"]),
        "final_standard_probability": float(final_comp["达标倾向概率"]),
        "final_historical_distance": float(final_comp["历史达标IQR距离"]),
        "status": "低排放倾向方案_需实测复核" if final_comp["达标倾向概率"] >= 0.5 else "低排放评分最优方案_达标概率不足",
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
    lows = np.maximum(range_lows, np.asarray(base_intensity, dtype=float) - single_delta)
    highs = np.minimum(range_highs, np.asarray(base_intensity, dtype=float) + single_delta)
    fallback = np.clip(np.asarray(base_intensity, dtype=float), range_lows, range_highs)
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
    latest_actual = latest_intensity_from_cleaned(cleaned)
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
                "最新实际强度": current_latest,
                "第2问当前滞后强度": current_lagged,
                "约束后强度下限": float(lows[i - 1]),
                "约束后强度上限": float(highs[i - 1]),
                "单箱调节上限_Delta_i": float(single_delta[i - 1]),
                "最优目标强度": target,
                "最优目标原始负压": -target,
                "相对最新实际强度变化": delta_latest,
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
    latest_actual = latest_intensity_from_cleaned(cleaned)
    lagged = lagged_intensity_from_row(latest_q2_row, control_model)
    lagged_temp = lagged_temperature_from_row(latest_q2_row, control_model)
    lows, highs = main_allowed_bounds(ranges)
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
                "最新实际强度": current_latest,
                "第2问当前滞后强度": float(lagged[i - 1]),
                "历史达标强度P25": hist["达标强度P25"],
                "历史达标强度中位数": hist["达标强度中位数"],
                "历史达标强度P75": hist["达标强度P75"],
                "最优目标强度": target,
                "最优目标原始负压": -target,
                "相对最新实际强度变化": delta_latest,
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
        decision[["风箱编号", "主模型允许强度下限_P05", "主模型允许强度上限_P95", "历史达标强度中位数", "最新实际强度", "最优目标强度", "最优目标原始负压", "调整方向"]],
        DATA_OUT / "主模型_全局达标负压组合_简表.csv",
    )
    save_csv(
        decision[["风箱编号", "主模型允许强度下限_P05", "主模型允许强度上限_P95", "历史达标强度中位数", "最新实际强度", "最优目标强度", "最优目标原始负压", "调整方向"]],
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
    latest_observed = float(cleaned.loc[cleaned["序列"].idxmax(), TARGET_COL])
    base_eval = evaluate_plan(latest_actual)
    min_eval = evaluate_plan(min_intensity)
    best_eval = evaluate_plan(best_intensity)
    base_direct = float(base_eval["pred_co"])
    lows, highs = main_allowed_bounds(ranges)
    widths = np.maximum(highs - lows, 1e-6)
    hist_low = historical_stats["达标强度P25"].to_numpy(dtype=float)
    hist_high = historical_stats["达标强度P75"].to_numpy(dtype=float)
    rows = [
        {
            "场景": "当前最新工况",
            "模型角色": "基准",
            "实际观测CO_mg_m3": latest_observed,
            "非线性模型预测CO_mg_m3": base_direct,
            "达标倾向概率": float(base_eval["standard_probability"]),
            "相对2800": base_direct - EMISSION_STANDARD,
            "模型是否低于2800_仅参考": base_direct <= EMISSION_STANDARD,
            "较当前模型预测降低量": 0.0,
            "标准化调节量": 0.0,
            "历史达标IQR距离": interval_distance(latest_actual, hist_low, hist_high, widths),
            "说明": "模型口径下的基准比较，不作为达标判定",
        },
        {
            "场景": "主模型_CO最小方案",
            "模型角色": "备用",
            "实际观测CO_mg_m3": np.nan,
            "非线性模型预测CO_mg_m3": min_co,
            "达标倾向概率": float(min_eval["standard_probability"]),
            "相对2800": min_co - EMISSION_STANDARD,
            "模型是否低于2800_仅参考": min_co <= EMISSION_STANDARD,
            "较当前模型预测降低量": base_direct - min_co,
            "标准化调节量": normalized_adjustment(min_intensity, latest_actual, lows, highs),
            "历史达标IQR距离": interval_distance(min_intensity, hist_low, hist_high, widths),
            "说明": "仅看回归预测CO最低，不作为最终推荐",
        },
    ]
    rows.append(
        {
            "场景": "主模型_最终推荐方案",
            "模型角色": "主模型",
            "实际观测CO_mg_m3": np.nan,
            "非线性模型预测CO_mg_m3": best_co,
            "达标倾向概率": float(best_eval["standard_probability"]),
            "相对2800": best_co - EMISSION_STANDARD,
            "模型是否低于2800_仅参考": best_co <= EMISSION_STANDARD,
            "较当前模型预测降低量": base_direct - best_co,
            "标准化调节量": normalized_adjustment(best_intensity, latest_actual, lows, highs),
            "历史达标IQR距离": interval_distance(best_intensity, hist_low, hist_high, widths),
            "说明": diagnostics["status"],
        }
    )
    comp = pd.DataFrame(rows)
    save_csv(comp, DATA_OUT / "主模型_非线性低排放评分对比.csv")
    save_csv(comp, DATA_OUT / "主模型_CO直接预测对比.csv")
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
            "场景": "辅助模型_当前最新工况",
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
    for _, row in eval_source.iterrows():
        speed_feature = f"{SPEED_COL}__lag0"
        speed = float(row[speed_feature]) if speed_feature in aligned.columns else float(row.get(SPEED_COL, 0.0))
        p0 = current_intensity_from_cleaned_sequence(cleaned_by_sequence, row["序列"])
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
                "当前实际总强度": float(np.sum(p0)),
                "最优总强度": float(np.sum(best_p)),
                "总调节量": float(np.sum(np.abs(best_p - p0))),
                "平均温度增量": float(np.mean(temp_delta)),
                "目标函数评估次数": int(diag["evaluated_candidates"]),
                **{f"{i}#当前实际强度": float(p0[i - 1]) for i in BOX_IDS},
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
    ax.bar(x - width / 2, decision["最新实际强度"], width, label="最新实际强度", color="#64748B")
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

- 本问重新训练了 `无历史CO调控模型`，没有使用任何 `CO历史浓度__*` 特征，避免第二问监测型模型中历史 CO 抢占过程变量解释能力。
- 调控模型的训练数据仍沿用第二问异常修正后建模的 `滞后配准建模数据.csv`，因此风箱负压、风箱温度、机速、大烟道变量的时滞口径与第二问一致。
- 主模型改用非线性低排放评分：同时使用非线性 CO 回归、达标倾向概率和历史达标工况距离；不再用 `当前CO + ΔCO` 作为最终判断，也不让单一回归模型独立判定是否达标。
- 温度采用增量叠加口径：先用第一问 `M2_enhanced_online + Ridge_alpha10` 分别预测基准负压和方案负压下的温度，取 `ΔT`，再令 `T_scenario = T_第二问当前配准温度 + ΔT`。
- 第三问负压基准使用当前实际负压强度 `P_i^0`，不是第二问中的滞后负压强度；第二问滞后强度只用于说明配准输入来源。
- 代码分为两个模型：主模型为 `非线性全局低排放优化模型`，在 P05-P95 允许范围内最小化低排放综合评分；辅助模型为 `当前工况平稳调控模型`，保留单箱调节上限、总调节预算和相邻风箱差值约束，用来说明小幅平稳调节的降 CO 幅度。
- 历史 CO<=2800 的样本单独统计风箱负压分布，并通过样本加权、达标倾向分类器和历史达标 IQR 距离进入主模型。
- 历史达标样本同时做连续性、机速、大烟道负压/温度、时间分布和变量组重要性诊断；若达标样本对应整体工况切换，则不能只套用达标样本的风箱负压中位数模板。
- 本问取消使用完整监测型模型作为调控复核指标；非线性模型输出只作为调控评分，不作为现场达标承诺。
- 原始负压为负值，模型输入为 `负压强度=abs(原始负压)`；目标原始负压为 `-最优目标强度`。
- 主模型优化范围为各风箱历史负压强度 P05-P95；辅助平稳调控模型使用 P10-P90 与当前附近小幅约束。

## 模型规模

- 无历史 CO 调控模型特征数：{len(control_model.feature_names)}
- 第一问温度联动模型特征数：{len(q1_linker.feature_names)}
- 第一问温度联动模型训练样本数：{q1_linker.fit_rows}

## 时滞解释

第1问温度模型考虑最多 120s 的压力、机速和过去温度窗口；第2问 CO 模型考虑风箱过程变量最长 900s 的空间时滞。由于两者绝对温度口径并不完全等价，本问只使用第一问给出的温度变化量，不直接用第一问绝对温度替换第二问温度特征。
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
    importance_group_summary: pd.DataFrame,
    control_metrics: pd.DataFrame,
    control_model: LinearCOModel,
    q1_linker: Q1TemperatureLinker,
    main_diagnostics: dict[str, object],
    aux_diagnostics: dict[str, object],
) -> None:
    main_final = main_comp[main_comp["场景"] == "主模型_最终推荐方案"].iloc[0]
    main_min = main_comp[main_comp["场景"] == "主模型_CO最小方案"].iloc[0]
    main_base = main_comp[main_comp["场景"] == "当前最新工况"].iloc[0]
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
            return np.nan
        return float(imp_lookup.loc[group_name, "正向重要性占比_%"])

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
            "最新实际强度",
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
            "最新实际强度",
            "最优目标强度",
            "最优目标原始负压",
            "调整方向",
        ]
    ]
    hist_view = historical_stats[["风箱编号", "达标强度P25", "达标强度中位数", "达标强度P75", "最新实际强度", "达标中位数较最新变化"]]

    files = [
        ("线性无历史CO调控模型", "result/第3问/CO调控模型_无历史CO.json"),
        ("非线性低排放调控模型", "result/第3问/CO调控模型_非线性低排放.json"),
        ("非线性模型评价", "data/第3问/结果数据/非线性低排放调控模型评价指标.csv"),
        ("历史达标工况负压统计", "data/第3问/结果数据/历史达标工况风箱负压统计.csv"),
        ("历史达标连续性诊断", "data/第3问/结果数据/历史达标工况连续性诊断.csv"),
        ("历史达标连续片段明细", "data/第3问/结果数据/历史达标工况连续片段明细.csv"),
        ("历史达标关键变量对比", "data/第3问/结果数据/历史达标工况关键变量对比.csv"),
        ("历史达标时间分布", "data/第3问/结果数据/历史达标工况时间分布.csv"),
        ("非线性模型变量组重要性", "data/第3问/结果数据/非线性低排放模型变量组重要性.csv"),
        ("主模型非线性低排放负压组合", "data/第3问/结果数据/主模型_非线性低排放负压组合.csv"),
        ("主模型非线性低排放评分对比", "data/第3问/结果数据/主模型_非线性低排放评分对比.csv"),
        ("主模型搜索记录", "data/第3问/中间数据/主模型_非线性低排放优化搜索记录.csv"),
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

第3问的目标不是只围绕最新一行工况做小幅试探，而是在各风箱负压允许范围内寻找低排放倾向更强的组合。考虑到单一调控模型精度不足，本文不让模型独立承担“是否达标”的最终判定，而是把 CO 回归预测、达标倾向概率和历史达标工况相似度组合成低排放评分。为此，本问将优化分成两个模型：

- 主模型：全局低排放优化模型。负压范围放宽到各风箱历史强度 P05-P95，只保留允许范围边界；历史达标工况通过训练权重、达标概率模型和历史达标 IQR 距离进入目标函数。
- 辅助模型：当前工况平稳调控模型。保留单箱调节上限、总调节预算、相邻风箱差值等工程平稳约束，用来说明如果只从当前工况附近小幅调整，CO 降低幅度有限。

本问不再用 `当前 CO + ΔCO` 作为最终判断。模型输出只作为优化评分的一部分，方案是否真实达标需结合现场复核：

```text
Score(P) = CO_reg(P) - 达标概率奖励 + 历史达标工况距离惩罚 + 边界惩罚
```

## 2. 历史达标工况

原始历史样本共有 `{int(hist_row["全样本数"])}` 条，其中 CO 不高于 2800 的达标样本 `{int(hist_row["历史达标样本数"])}` 条，占 `{hist_row["历史达标比例_%"]:.2f}%`。这些样本不再只是参考：训练时达标样本权重提高，另训练达标倾向分类器；优化时还加入“偏离历史达标工况 P25-P75 区间”的惩罚。各风箱达标工况负压分布如下：

{markdown_table(hist_view, float_digits=3)}

## 3. 达标样本稳定性与整体工况诊断

仅统计达标样本的风箱负压中位数是不够的，因为这 71 条样本可能不是负压单变量作用，而是机速、主抽风、大烟道温度和后段热状态共同变化后的结果。为此，本文增加连续性、关键变量、时间分布和变量组重要性四类检查。

从连续性看，达标样本被拆成 `{int(compliant_diag_row["连续达标片段数"])}` 个连续片段，其中单点片段 `{int(compliant_diag_row["单点片段数"])}` 个；片段样本数中位数为 `{compliant_diag_row["片段样本数中位数"]:.1f}`，最长连续达标只有 `{int(compliant_diag_row["最长连续样本数"])}` 条，即 `{int(compliant_diag_row["最长连续持续时间_s"])}` s，没有出现持续不少于 60 s 的稳定达标段。因此这些样本更像短时达标片段或阶段性工况切换信号，不能直接等同于可长期保持的稳定达标窗口。

达标样本连续片段如下：

{markdown_table(run_view, float_digits=3)}

达标与未达标关键工况变量对比如下。机速中位数同为 `{metric_of(SPEED_COL, "达标中位数"):.3f}`，但达标样本机速 P25 为 `{metric_of(SPEED_COL, "达标P25"):.3f}`，低于未达标样本 P25 `{metric_of(SPEED_COL, "未达标P25"):.3f}`，说明部分达标片段伴随偏低机速。注意负压强度越小表示原始负压越接近 0；本批达标样本的大烟道负压强度和风箱平均负压强度整体更低，而 1#、2#大烟道温度中位数分别高 `{diff_of("1#大烟道温度"):.3f}` ℃、`{diff_of("2#大烟道温度"):.3f}` ℃，说明达标样本不是“主抽风更强 + 大烟道温度更低”的单一模式。

{markdown_table(compare_view, float_digits=3)}

达标样本也有明显时间集中性。第 9-10 个时间段合计贡献 `{late_time_count}` 条达标样本，占全部达标样本约 `{late_time_share:.2f}%`；第 5 个时间段还有 `{fifth_time_count}` 条，占约 `{fifth_time_share:.2f}%`。这说明达标样本并非均匀散布在全历史中，而是集中在少数阶段。

{markdown_table(time_view, float_digits=3)}

从模型关联看，非线性调控模型的正向置换重要性中，风箱负压组占 `{pressure_share:.2f}%`，热状态组（大烟道温度 + 风箱温度）合计约 `{thermal_share:.2f}%`，其中大烟道温度组单独占 `{flue_temp_share:.2f}%`。这只能说明相关性，不能当作因果证明；但足以说明现有数据不支持“CO 降低主要由风箱负压单变量造成”的解释，至少需要把大烟道温度、主抽风状态和机速作为联合工况条件。也因此，主模型即使靠近历史达标负压 IQR，仍只得到 `{main_final["非线性模型预测CO_mg_m3"]:.3f}` mg/m^3、达标倾向概率 `{main_final["达标倾向概率"]:.3f}`，不能承诺达标。

{markdown_table(importance_view, float_digits=3)}

## 4. 调控模型

调控模型使用第二问监测型模型中的过程变量集合，但剔除所有 `CO历史浓度__*` 特征后重新训练，而不是在已训练监测型模型上事后置零。上一版线性岭回归只能给出单调边界解，本版主模型改用非线性树模型：

```text
CO_reg = HistGradientBoostingRegressor(过程变量, 温度联动变量)
Pr(CO<=2800) = HistGradientBoostingClassifier(过程变量, 温度联动变量)
```

本次主模型为 `{model_type}`，特征数为 `{len(control_model.feature_names)}`。测试集 RMSE 为 `{test_metric["RMSE"]:.3f}` mg/m^3，R2 为 `{test_metric["R2"]:.3f}`；真实达标样本在测试集中占 `{test_metric["真实CO低于2800比例_%"]:.2f}%`。这说明模型仍不足以直接判定排放是否达标，因此本文只把它作为低排放调控评分模型。{model_weight_note}

温度联动仍采用第一问 `M2_enhanced_online + Ridge_alpha10` 的增量口径：

```text
ΔT = T_q1(方案负压) - T_q1(当前实际负压P0)
T_scenario = T_q2_current + ΔT
```

## 5. 主模型：非线性全局低排放优化

主模型的决策变量为 18 个风箱负压强度 `P=(P1,...,P18)`。可行域先放宽为各风箱历史强度 P05-P95，不再一开始加入当前附近调节、单箱调节上限、总调节预算和相邻风箱差值约束。目标函数为：

```text
min Score(P)
Score(P)= (CO_reg(P)-2800)/500
          - 1.6*Pr(CO<=2800|P)
          + 1.2*distance(P, 历史达标IQR)
          + 0.35*boundary_penalty(P)
          + 0.08*adjustment(P,P0)
```

本次主模型状态为 `{main_diagnostics["status"]}`。当前最新工况下非线性模型预测 CO 为 `{main_base["非线性模型预测CO_mg_m3"]:.3f}` mg/m^3，达标倾向概率为 `{main_base["达标倾向概率"]:.3f}`；最终推荐方案预测 CO 为 `{main_final["非线性模型预测CO_mg_m3"]:.3f}` mg/m^3，达标倾向概率为 `{main_final["达标倾向概率"]:.3f}`，历史达标 IQR 距离为 `{main_final["历史达标IQR距离"]:.3f}`。这些数值只表示模型评分下的低排放倾向，不作为现场达标承诺。

主模型对比如下：

{markdown_table(main_comp[["场景", "模型角色", "实际观测CO_mg_m3", "非线性模型预测CO_mg_m3", "达标倾向概率", "相对2800", "模型是否低于2800_仅参考", "较当前模型预测降低量", "标准化调节量", "历史达标IQR距离", "说明"]], float_digits=3)}

主模型推荐负压组合如下：

{markdown_table(main_view, float_digits=3)}

## 6. 辅助模型：当前工况平稳调控

辅助模型保留工程平稳约束：

```text
|P_i - P_i^0| <= Delta_i
|P_i - P_{{i-1}}| <= d_i
sum_i |P_i - P_i^0| <= B
```

其中 `Delta_i` 按历史 P10-P90 宽度的 35% 估计，并限制在 `[0.10, 1.20]`；`d_i` 取相邻风箱历史强度差 P90；`B=0.45*sum(Delta_i)`。这些约束工程上合理，但会显著压缩可行域，因此只作为辅助模型使用。

辅助模型下，当前工况小幅平稳优化后的直接预测 CO 为 `{aux_after["调控模型直接预测CO_mg_m3"]:.3f}` mg/m^3，模型预测降低 `{aux_after["模型预测降低量"]:.3f}` mg/m^3。这说明如果只允许当前附近小幅平稳调整，降 CO 空间有限。

{markdown_table(aux_comp[["场景", "口径", "实际观测CO_mg_m3", "调控模型直接预测CO_mg_m3", "模型增量_mg_m3", "相对2800", "是否达标", "模型预测降低量"]], float_digits=3)}

辅助模型的平稳调控组合如下：

{markdown_table(aux_view, float_digits=3)}

测试集逐样本平稳调控评估只用于说明辅助模型方向稳定性。测试集基准预测 CO 均值为 `{test_summary["基准预测CO均值"]:.3f}` mg/m^3，平稳优化后预测 CO 均值为 `{test_summary["优化后预测CO均值"]:.3f}` mg/m^3，平均模型预测降低 `{test_summary["平均模型预测降低量"]:.3f}` mg/m^3。

## 7. 基础可视化

| 图编号 | 文件路径 | 图的含义 |
|---|---|---|
| 图3-1 | `figure/第3问/基础图/图3-1_各风箱负压强度调节范围.svg` | P10-P90 与 P05-P95 允许范围参考 |
| 图3-2 | `figure/第3问/基础图/图3-2_调控模型最优负压组合.svg` | 最新实际强度与主模型目标强度对比 |
| 图3-3 | `figure/第3问/基础图/图3-3_最优方案温度联动变化.svg` | 主模型方案温度增量叠加 |
| 图3-4 | `figure/第3问/基础图/图3-4_主模型直接预测CO对比.svg` | 当前、CO 最小方案和最终推荐方案的非线性模型预测 CO 对比 |
| 图3-5 | `figure/第3问/基础图/图3-5_辅助模型测试集直接预测分布.svg` | 辅助模型测试集直接预测 CO 分布 |

## 8. 本问生成的文件清单

{file_lines}

## 9. 补充说明

- 主模型不使用当前 CO 锚定，不使用 `当前CO + ΔCO` 判断是否达标；非线性模型预测 CO、达标概率和历史达标距离只共同构成低排放评分。
- 历史达标样本只有 `{int(hist_row["历史达标样本数"])}` 条且连续性不足，最长连续达标 `{int(compliant_diag_row["最长连续持续时间_s"])}` s；达标样本还集中在少数时间段，不能把达标样本风箱负压中位数当成充分调控模板。
- 当前附近调节、单箱调节上限、总调节预算和相邻风箱差值约束只放在辅助模型中。
- 第一问温度只提供增量 `ΔT`，避免第一问绝对温度与第二问配准温度时间口径不一致造成基准漂移。
- 主模型范围为 P05-P95，状态为 `{main_diagnostics["status"]}`；主模型唯一方案评估次数为 `{main_diagnostics.get("unique_plan_evaluations", 0)}`。

## 更新记录

- {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}，第3问子Agent：按“主模型全局达标优化 + 辅助模型当前工况平稳调控”重构，不再使用当前 CO 加模型增量作为最终达标判断。
"""
    (RESULT_DIR / "第3问.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    raw = read_original_excel()
    cleaned = clean_raw_data(raw)
    ranges = make_pressure_ranges(cleaned)
    constraints = make_adjustment_constraints(ranges, cleaned)
    cleaned_by_sequence = cleaned.set_index("序列", drop=False)
    deps = load_dependencies()

    aligned = pd.read_csv(deps["第2问滞后配准建模数据"], encoding="utf-8-sig")
    latest_q2 = aligned[aligned["序列"] == aligned["序列"].max()].copy()
    if latest_q2.empty:
        raise ValueError("第2问滞后配准建模数据中未找到最新样本")

    monitor_model = LinearCOModel.from_json(deps["第2问监测型模型"])
    linear_control_model, linear_control_metrics, _ = fit_no_co_control_model(aligned, monitor_model)
    control_model, control_metrics, control_importance = fit_nonlinear_control_model(aligned, monitor_model)
    q1_linker = Q1TemperatureLinker.fit_from_q1_outputs(deps)

    latest_clean = cleaned.loc[cleaned["序列"].idxmax()]
    latest_speed = float(latest_clean[SPEED_COL])
    latest_actual = latest_intensity_from_cleaned(cleaned)
    historical_stats, historical_summary, historical_profile = make_historical_compliant_profile(cleaned, latest_actual)
    compliant_condition_diagnostics = make_compliant_condition_diagnostics(cleaned)
    importance_group_summary = make_importance_group_summary(control_importance)

    evaluate_main_plan, main_pred_diag = scenario_evaluator_factory(latest_q2, control_model, q1_linker, latest_speed, latest_actual)
    global_best_intensity, global_best_co, global_min_intensity, global_min_co, global_search, main_diagnostics = optimize_global_standard_model(
        ranges,
        evaluate_main_plan,
        latest_actual,
        historical_profile,
        historical_stats,
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
    eval_df = evaluate_plan_over_samples(aligned, cleaned_by_sequence, ranges, constraints, q1_linker, control_model, median_start)
    summary = summarize_eval(eval_df)

    params = {
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "优化目标": "主模型在P05-P95允许范围内最小化非线性低排放评分；辅助模型保留当前工况附近平稳约束",
        "调控模型构造": "使用第二问滞后配准后的过程变量重新训练，剔除全部CO历史特征；主模型采用非线性回归+达标倾向分类器",
        "温度联动": "第一问只提供ΔT=T_q1(方案)-T_q1(基准)，再叠加到第二问当前配准温度，不做绝对温度替换",
        "情景估计": "不用当前CO+ΔCO作为最终判断；非线性模型预测CO、达标概率和历史达标距离共同构成低排放评分，不能单独替代实测达标判定",
        "压力口径": "原始负压为负值；模型输入使用负压强度abs(原始负压)",
        "主模型优化范围": "每个风箱历史负压强度P05-P95",
        "辅助模型优化范围": "当前工况附近小幅平稳约束叠加P10-P90",
        "第1问重构模型": "M2_enhanced_online + Ridge_alpha10",
        "第1问训练样本数": q1_linker.fit_rows,
        "第2问监测型模型": str(deps["第2问监测型模型"]),
        "无历史CO调控模型特征数": len(control_model.feature_names),
        "无历史CO调控模型验证RMSE": control_model.validation_rmse,
        "主模型状态": main_diagnostics["status"],
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
            "无历史CO调控模型测试RMSE": float(control_metrics.loc[control_metrics["数据集"] == "测试集", "RMSE"].iloc[0]),
            "历史达标样本数": int(historical_summary.iloc[0]["历史达标样本数"]),
            "主模型状态": main_diagnostics["status"],
            "主模型最终非线性预测CO": float(global_best_co),
            "主模型最终模型值是否低于2800_仅参考": bool(global_best_co <= EMISSION_STANDARD),
            "主模型距2800_仅参考": float(global_best_co - EMISSION_STANDARD),
            "主模型最终达标倾向概率": float(main_diagnostics.get("final_standard_probability", np.nan)),
            "主模型最终历史达标IQR距离": float(main_diagnostics.get("final_historical_distance", np.nan)),
            "主模型CO最小非线性预测CO": float(global_min_co),
            "辅助模型最优模型增量CO": float(aux_best_delta),
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
        importance_group_summary,
        control_metrics,
        control_model,
        q1_linker,
        main_diagnostics,
        aux_diagnostics,
    )


if __name__ == "__main__":
    main()
