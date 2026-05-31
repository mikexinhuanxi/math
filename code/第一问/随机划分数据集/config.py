from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "附件1.原始数据.xlsx"
DATA_DIR = ROOT_DIR / "data"
RESULT_DIR = ROOT_DIR / "result"

CLEAN_DATA_PATH = DATA_DIR / "problem1_clean.csv"
FEATURES_PATH = DATA_DIR / "problem1_features_pressure_only.csv"
FEATURE_INDEX_PATH = DATA_DIR / "problem1_feature_index.csv"
MODEL_PATH = DATA_DIR / "最佳模型文件.pkl"

MODEL_COMPARISON_PATH = RESULT_DIR / "problem1_model_comparison.csv"
METRICS_BY_BOX_PATH = RESULT_DIR / "problem1_metrics_by_box.csv"
PREDICTIONS_PATH = RESULT_DIR / "problem1_predictions.xlsx"
FEATURE_IMPORTANCE_PATH = RESULT_DIR / "problem1_feature_importance.csv"
PLOTS_PATH = RESULT_DIR / "problem1_prediction_plots.png"
SUMMARY_PATH = RESULT_DIR / "problem1_summary.md"
QUESTION1_MD_PATH = ROOT_DIR / "第1问.md"
XGB_TUNING_PATH = RESULT_DIR / "problem1_xgboost_tuning.csv"
CV_METRICS_PATH = RESULT_DIR / "problem1_cv_metrics.csv"
TARGET_EXPLORATION_PATH = RESULT_DIR / "problem1_target_exploration.csv"
STRUCTURE_EXPLORATION_PATH = RESULT_DIR / "problem1_structure_exploration.csv"
LEAKAGE_REPORT_PATH = RESULT_DIR / "problem1_pressure_only_check.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
TIME_STEP_SECONDS = 2

SEQ_COL = "序列"
SPEED_COL = "烧结机机速L1设定"
BOX_IDS = list(range(1, 19))
PRESSURE_COLS = [f"{i}#风箱负压" for i in BOX_IDS]
SOUTH_TEMP_COLS = [f"{i}#风箱南侧温度" for i in BOX_IDS]
NORTH_TEMP_COLS = [f"{i}#风箱北侧温度" for i in BOX_IDS]
AVG_TEMP_COLS = [f"{i}#风箱平均温度" for i in BOX_IDS]
DIFF_TEMP_COLS = [f"{i}#风箱南北温差" for i in BOX_IDS]

AVG_TARGET_COLS = [f"box_{i:02d}_avg_temp" for i in BOX_IDS]
SOUTH_TARGET_COLS = [f"box_{i:02d}_south_temp" for i in BOX_IDS]
NORTH_TARGET_COLS = [f"box_{i:02d}_north_temp" for i in BOX_IDS]
DIFF_TARGET_COLS = [f"box_{i:02d}_south_minus_north" for i in BOX_IDS]

LAG_STEPS = [1, 5, 10, 30]
ROLL_WINDOWS = [5, 15, 30]
FORBIDDEN_FEATURE_KEYWORDS = ["温度", "CO", "大烟道", "南侧", "北侧"]

BOX_LENGTHS = {i: (4.0 if i <= 15 else 2.0) for i in BOX_IDS}
BOX_POSITIONS = {i: float(i) for i in BOX_IDS}
BOX_SEGMENTS = {
    "front": list(range(1, 6)),
    "middle": list(range(6, 13)),
    "back": list(range(13, 19)),
    "long": list(range(1, 16)),
    "short": list(range(16, 19)),
}
TARGET_SEGMENTS = {
    "front": list(range(1, 6)),
    "middle": list(range(6, 13)),
    "back": list(range(13, 19)),
}

XGB_BASE_PARAMS = {
    "n_estimators": 220,
    "max_depth": 4,
    "learning_rate": 0.045,
    "subsample": 0.9,
    "colsample_bytree": 0.85,
    "min_child_weight": 2.0,
    "reg_alpha": 0.0,
    "reg_lambda": 2.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_TUNING_GRID = [
    {
        "n_estimators": 220,
        "max_depth": 3,
        "learning_rate": 0.055,
        "subsample": 0.90,
        "colsample_bytree": 0.80,
        "min_child_weight": 2.0,
        "reg_lambda": 2.0,
    },
    {
        "n_estimators": 280,
        "max_depth": 4,
        "learning_rate": 0.040,
        "subsample": 0.90,
        "colsample_bytree": 0.85,
        "min_child_weight": 2.0,
        "reg_lambda": 2.0,
    },
    {
        "n_estimators": 320,
        "max_depth": 4,
        "learning_rate": 0.035,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "min_child_weight": 1.0,
        "reg_lambda": 1.5,
    },
    {
        "n_estimators": 260,
        "max_depth": 5,
        "learning_rate": 0.035,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "min_child_weight": 3.0,
        "reg_lambda": 3.0,
    },
    {
        "n_estimators": 360,
        "max_depth": 3,
        "learning_rate": 0.030,
        "subsample": 0.95,
        "colsample_bytree": 0.90,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
    },
]
