# 问题一可视化图表索引

## 数据口径
- 题目问题一：探索各风箱负压与对应风箱温度间的关系，并构建基于风箱负压的温度预测模型。
- 图表读取现有导出文件，不重新训练模型。
- 当前综合 MAE 最低的导出结果：`history_pressure_delta_lgbm`，MAE = 3.0670。

## 图表文件
- `00_problem1_dashboard.png`
- `01_process_overview.png`
- `02_spatial_profiles.png`
- `03_model_comparison.png`
- `04_ablation_by_box_mae_heatmap.png`
- `05_key_models_by_box_mae.png`
- `06_best_prediction_tracking.png`
- `07_best_prediction_diagnostics.png`
- `08_feature_importance.png`

## 使用建议
- 论文正文优先使用 `00_problem1_dashboard.png`、`03_model_comparison.png`、`04_ablation_by_box_mae_heatmap.png`、`06_best_prediction_tracking.png`。
- 附录或结果分析可补充 `01_process_overview.png`、`02_spatial_profiles.png`、`07_best_prediction_diagnostics.png`、`08_feature_importance.png`。
