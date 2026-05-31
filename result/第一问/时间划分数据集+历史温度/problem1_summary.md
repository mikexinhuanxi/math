# 第1问保留结果汇总

## Pressure-only
| experiment | MAE | RMSE | R2 | 说明 |
| --- | ---: | ---: | ---: | --- |
| time_ordered_random_forest_base | 12.3781 | 23.8890 | -0.1056 | 前80%训练、后20%测试的基础压力特征模型 |
| hot_override_rel_lgbm_l1 | 11.7905 | 23.0480 | -0.0500 | 相对负压 + 删除 PSI>=1 强漂移特征 + 后段目标轻度加权 RF + 12#-14# LGBM-L1 覆盖 |

## 历史温度消融
| experiment | 特征 | MAE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: |
| history_pressure_delta_lgbm | 历史温度 + 负压，预测 ΔT | 3.0670 | 6.6206 | 0.9097 |
| history_only_delta_lgbm | 只有历史温度，预测 ΔT | 3.2302 | 6.7708 | 0.9054 |
| history_pressure_direct_lgbm | 历史温度 + 负压，直接预测 T | 3.8394 | 9.8417 | 0.7856 |
| history_only_direct_lgbm | 只有历史温度，直接预测 T | 4.0322 | 10.0656 | 0.7762 |
| persistence_lag1 | 只用上一时刻温度 | 4.7261 | 10.1055 | 0.8686 |

## 结论
严格 pressure-only 模型在后20%漂移工况下存在明显上限；历史温度反馈是主要增益来源。加入负压后，ΔT-LightGBM 的 MAE 从 3.2302 降到 3.0670，说明负压仍提供额外工况信息。
