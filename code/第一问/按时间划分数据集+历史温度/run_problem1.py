from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config
import problem1_lib as lib


def main() -> None:
    lib.ensure_dirs()
    print("[1/11] 读取并清洗原始数据")
    raw = lib.load_raw_data(config.RAW_DATA_PATH)
    clean = lib.build_clean_data(raw)
    lib.save_csv(clean, config.CLEAN_DATA_PATH)

    print("[2/11] 构造 pressure-only 特征并执行泄漏检查")
    features, aligned_clean = lib.build_pressure_only_features(clean)
    targets = lib.build_target_frames(aligned_clean)
    lib.save_csv(features, config.FEATURES_PATH)
    lib.save_csv(aligned_clean[[config.SEQ_COL]], config.FEATURE_INDEX_PATH)
    lib.write_leakage_report(config.LEAKAGE_REPORT_PATH, list(features.columns))

    print(f"      建模样本: {len(features)}, 特征数: {features.shape[1]}")
    train_idx, test_idx = lib.make_train_test_indices(len(features))
    time_train_idx, time_test_idx = lib.make_time_train_test_indices(len(features))
    y_avg = targets["average"]

    candidate_models = [name for name in ["RandomForest", "ExtraTrees", "LightGBM"] if name in lib.available_model_names()]
    available_models = "/".join(candidate_models)
    print(f"[3/11] 随机8:2辅助比较可用模型: {available_models}")
    comparison_result = lib.run_model_comparison(
        features, y_avg, train_idx, test_idx, model_names=candidate_models, include_xgb_tuning=False
    )
    comparison = comparison_result["comparison"]
    random_best_name = comparison_result["best_model_name"]
    random_best_pred = comparison_result["best_validation_pred"]
    random_best_xgb_params = comparison_result["best_xgb_params"]
    xgb_tuning = comparison_result["xgb_tuning"]
    lib.save_csv(comparison, config.MODEL_COMPARISON_PATH)
    lib.save_csv(xgb_tuning, config.XGB_TUNING_PATH)
    print(f"      随机8:2最佳模型: {random_best_name}")

    print("[4/11] 按时间顺序训练前80%并测试后20%")
    time_comparison_result = lib.run_model_comparison(
        features, y_avg, time_train_idx, time_test_idx, model_names=candidate_models, include_xgb_tuning=False
    )
    time_comparison = time_comparison_result["comparison"]
    time_best_name = time_comparison_result["best_model_name"]
    time_best_pred = time_comparison_result["best_validation_pred"]
    time_best_xgb_params = time_comparison_result["best_xgb_params"]
    lib.save_csv(time_comparison, config.TIME_MODEL_COMPARISON_PATH)
    time_metrics = pd.DataFrame(
        [
            {
                "model": time_best_name,
                "validation": "last_20_percent",
                **lib.overall_metrics(y_avg.iloc[time_test_idx], time_best_pred),
            }
        ]
    )
    time_metrics_path = config.RESULT_DIR / "problem1_time_holdout_metrics.csv"
    lib.save_csv(time_metrics, time_metrics_path)
    print(f"      时间顺序最佳基础模型: {time_best_name}")

    print("[5/11] 诊断后20%是否存在工况/关系漂移")
    drift_diagnostics = lib.diagnose_time_drift(
        features,
        y_avg,
        time_train_idx,
        time_test_idx,
        time_best_pred,
        random_metrics=comparison.iloc[0].to_dict(),
    )
    lib.save_csv(drift_diagnostics, config.DRIFT_DIAGNOSTICS_PATH)

    print("[6/11] 进行漂移校正并重新评估后20%")
    drift_result = lib.run_drift_correction(time_best_name, time_best_xgb_params, features, y_avg)
    drift_correction = drift_result["correction"]
    corrected_pred = drift_result["best_pred"]
    corrected_idx = drift_result["test_idx"]
    best_correction_method = drift_result["best_method"]
    lib.save_csv(drift_correction, config.DRIFT_CORRECTION_PATH)

    print("[7/11] 计算校正后18个风箱逐项指标")
    y_test = y_avg.iloc[corrected_idx]
    by_box = lib.metrics_by_box(y_test, corrected_pred, prefix=f"{time_best_name}+{best_correction_method}")
    lib.save_csv(by_box, config.METRICS_BY_BOX_PATH)

    print("[8/11] 对时间顺序最佳基础模型做5折随机交叉验证")
    cv = lib.cross_validate_model(time_best_name, time_best_xgb_params, features, y_avg)
    lib.save_csv(cv, config.CV_METRICS_PATH)

    print("[9/11] 记录当前任务相关的目标和结构结果")
    target_exploration = pd.DataFrame(
        [
            {
                "target_representation": "南北简单平均温度",
                "evaluation_target": "平均温度",
                **lib.overall_metrics(y_avg.iloc[corrected_idx], corrected_pred),
            }
        ]
    )
    structure_exploration = pd.DataFrame(
        [
            {
                "model": f"{time_best_name}+{best_correction_method}",
                "train_seconds": float("nan"),
                **lib.overall_metrics(y_avg.iloc[corrected_idx], corrected_pred),
            }
        ]
    )
    lib.save_csv(target_exploration, config.TARGET_EXPLORATION_PATH)
    lib.save_csv(structure_exploration, config.STRUCTURE_EXPLORATION_PATH)

    print("[10/11] 按最佳漂移校正策略重训最终模型、保存模型和特征重要性")
    final_model = lib.refit_final_drift_corrected_model(best_correction_method, time_best_name, time_best_xgb_params, features, y_avg)
    metadata = lib.metadata_for_model(
        time_best_name,
        list(features.columns),
        list(y_avg.columns),
        time_best_xgb_params,
        extra={
            "validation_strategy": "time_ordered_last_20_percent",
            "drift_correction_method": best_correction_method,
        },
    )
    metadata["random_8_2_metrics"] = comparison.iloc[0].to_dict()
    metadata["time_ordered_metrics"] = time_comparison.iloc[0].to_dict()
    metadata["drift_correction_metrics"] = drift_correction.iloc[0].to_dict()
    metadata["cv_metrics"] = cv.to_dict(orient="records")
    metadata["time_holdout_metrics"] = time_metrics.iloc[0].to_dict()
    lib.save_model(config.MODEL_PATH, final_model, metadata)
    feature_importance = lib.extract_feature_importance(final_model, list(features.columns))
    lib.save_csv(feature_importance, config.FEATURE_IMPORTANCE_PATH)

    print("[11/11] 导出预测Excel、图片和Markdown总结")
    lib.save_prediction_excel(
        config.PREDICTIONS_PATH,
        aligned_clean,
        y_avg.iloc[test_idx],
        random_best_pred,
        test_idx,
        y_avg.iloc[time_test_idx],
        time_best_pred,
        time_test_idx,
        comparison,
        by_box,
        cv,
        corrected_y_true=y_avg.iloc[corrected_idx],
        corrected_y_pred=corrected_pred,
        corrected_idx=corrected_idx,
        time_comparison=time_comparison,
        drift_correction=drift_correction,
    )
    lib.save_prediction_plots(
        config.PLOTS_PATH,
        aligned_clean,
        y_avg.iloc[corrected_idx],
        corrected_pred,
        corrected_idx,
        by_box,
        f"{time_best_name}+校正",
        validation_label="时间后20%校正验证",
    )
    summary = lib.make_summary_markdown(
        best_name=random_best_name,
        comparison=comparison,
        time_comparison=time_comparison,
        by_box=by_box,
        cv=cv,
        time_metrics=time_metrics,
        drift_diagnostics=drift_diagnostics,
        drift_correction=drift_correction,
        target_exploration=target_exploration,
        structure_exploration=structure_exploration,
        feature_importance=feature_importance,
        n_raw=len(raw),
        n_model=len(features),
        n_features=features.shape[1],
    )
    lib.write_markdown_reports(summary)

    print("\n完成。主结果如下：")
    print("时间顺序最佳基础模型：")
    print(time_comparison.head(1).to_string(index=False))
    print("\n漂移校正最佳策略：")
    print(drift_correction.head(1).to_string(index=False))
    print("\n5折交叉验证：")
    print(cv.tail(2).to_string(index=False))
    print("\n输出目录：")
    print(json.dumps({"data": str(config.DATA_DIR), "result": str(config.RESULT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
