
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
    print("[1/9] 读取并清洗原始数据")
    raw = lib.load_raw_data(config.RAW_DATA_PATH)
    clean = lib.build_clean_data(raw)
    lib.save_csv(clean, config.CLEAN_DATA_PATH)

    print("[2/9] 构造 pressure-only 特征并执行泄漏检查")
    features, aligned_clean = lib.build_pressure_only_features(clean)
    targets = lib.build_target_frames(aligned_clean)
    lib.save_csv(features, config.FEATURES_PATH)
    lib.save_csv(aligned_clean[[config.SEQ_COL]], config.FEATURE_INDEX_PATH)
    lib.write_leakage_report(config.LEAKAGE_REPORT_PATH, list(features.columns))

    print(f"      建模样本: {len(features)}, 特征数: {features.shape[1]}")
    train_idx, test_idx = lib.make_train_test_indices(len(features))
    y_avg = targets["average"]

    print("[3/9] 训练并比较 Ridge/ElasticNet/RandomForest/ExtraTrees/LightGBM/CatBoost/XGBoost")
    comparison_result = lib.run_model_comparison(features, y_avg, train_idx, test_idx)
    comparison = comparison_result["comparison"]
    best_name = comparison_result["best_model_name"]
    best_pred = comparison_result["best_validation_pred"]
    best_xgb_params = comparison_result["best_xgb_params"]
    xgb_tuning = comparison_result["xgb_tuning"]
    lib.save_csv(comparison, config.MODEL_COMPARISON_PATH)
    lib.save_csv(xgb_tuning, config.XGB_TUNING_PATH)
    print(f"      随机8:2最佳模型: {best_name}")

    print("[4/9] 计算18个风箱逐项指标")
    y_test = y_avg.iloc[test_idx]
    by_box = lib.metrics_by_box(y_test, best_pred, prefix=best_name)
    lib.save_csv(by_box, config.METRICS_BY_BOX_PATH)

    print("[5/9] 对最佳模型做5折随机交叉验证")
    cv = lib.cross_validate_model(best_name, best_xgb_params, features, y_avg)
    lib.save_csv(cv, config.CV_METRICS_PATH)

    print("[6/9] 做时间后20%鲁棒性分析")
    time_metrics, time_pred, time_idx = lib.evaluate_time_holdout(best_name, best_xgb_params, features, y_avg)
    time_metrics_path = config.RESULT_DIR / "problem1_time_holdout_metrics.csv"
    lib.save_csv(time_metrics, time_metrics_path)

    print("[7/9] 做目标表示和建模结构探索")
    target_exploration = lib.evaluate_target_representations(features, targets, train_idx, test_idx, best_xgb_params)
    structure_exploration = lib.run_structure_exploration(
        features,
        y_avg,
        train_idx,
        test_idx,
        best_xgb_params,
        comparison_result["predictions"]["XGBoost_tuned"],
    )
    lib.save_csv(target_exploration, config.TARGET_EXPLORATION_PATH)
    lib.save_csv(structure_exploration, config.STRUCTURE_EXPLORATION_PATH)

    print("[8/9] 拟合全量最终模型、保存模型和特征重要性")
    final_model = lib.refit_final_model(best_name, best_xgb_params, features, y_avg)
    metadata = lib.metadata_for_model(best_name, list(features.columns), list(y_avg.columns), best_xgb_params)
    metadata["random_8_2_metrics"] = comparison.iloc[0].to_dict()
    metadata["cv_metrics"] = cv.to_dict(orient="records")
    metadata["time_holdout_metrics"] = time_metrics.iloc[0].to_dict()
    lib.save_model(config.MODEL_PATH, final_model, metadata)
    feature_importance = lib.extract_feature_importance(final_model, list(features.columns))
    lib.save_csv(feature_importance, config.FEATURE_IMPORTANCE_PATH)

    print("[9/9] 导出预测Excel、图片和Markdown总结")
    lib.save_prediction_excel(
        config.PREDICTIONS_PATH,
        aligned_clean,
        y_test,
        best_pred,
        test_idx,
        y_avg.iloc[time_idx],
        time_pred,
        time_idx,
        comparison,
        by_box,
        cv,
    )
    lib.save_prediction_plots(
        config.PLOTS_PATH,
        aligned_clean,
        y_test,
        best_pred,
        test_idx,
        by_box,
        best_name,
    )
    summary = lib.make_summary_markdown(
        best_name=best_name,
        comparison=comparison,
        by_box=by_box,
        cv=cv,
        time_metrics=time_metrics,
        target_exploration=target_exploration,
        structure_exploration=structure_exploration,
        feature_importance=feature_importance,
        n_raw=len(raw),
        n_model=len(features),
        n_features=features.shape[1],
    )
    lib.write_markdown_reports(summary)

    print("\n完成。主结果如下：")
    print(comparison.head(1).to_string(index=False))
    print("\n5折交叉验证：")
    print(cv.tail(2).to_string(index=False))
    print("\n输出目录：")
    print(json.dumps({"data": str(config.DATA_DIR), "result": str(config.RESULT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
