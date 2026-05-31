# 历史温度 vs 历史温度+负压 消融实验

## 结论
- 最优实验：**history_pressure_delta_lgbm**
- 后20% MAE = **3.0670**，RMSE = **6.6206**，R2 = **0.9097**
- 只用历史温度的 ΔT-LightGBM：MAE = **3.2302**
- 历史温度 + 负压的 ΔT-LightGBM：MAE = **3.0670**
- 负压特征带来的 MAE 改变量：**0.1632**

## 实验总表
| experiment | mae | rmse | r2 | train_seconds | feature_group |
| --- | --- | --- | --- | --- | --- |
| history_pressure_delta_lgbm | 3.0670 | 6.6206 | 0.9097 | 32.4607 | history_plus_pressure |
| history_only_delta_lgbm | 3.2302 | 6.7708 | 0.9054 | 6.6748 | history_only |
| history_pressure_direct_lgbm | 3.8394 | 9.8417 | 0.7856 | 32.9230 | history_plus_pressure |
| history_only_direct_lgbm | 4.0322 | 10.0656 | 0.7762 | 10.0739 | history_only |
| persistence_lag1 | 4.7261 | 10.1055 | 0.8686 | 0.0004 | history_only |

## 说明
该实验均只使用历史温度 lag 和当前/历史负压特征，不使用当前或未来温度标签。ΔT 模型使用上一时刻真实温度还原当前温度，属于在线单步预测口径。
