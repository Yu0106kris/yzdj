# CrowdLoc Self-Iterating Solver

面向“基于非线性优化的众包离线高精度位置聚合”的**双层自迭代**求解框架。

## 两层迭代

1. **内层 IRLS**：鲁棒 M-估计的迭代重加权最小二乘。每轮用残差重新加权，逐步压低离群好心人的影响。
2. **外层 Meta-Iter**：在验证集上搜索鲁棒核 / 初始化 / 场景判别阈值的组合，按 CEP90 + 移动场景增益选出最优配置。

## 核心假设与赛题对齐

- 优化变量：丢失设备位置 x ∈ R²
- 观测： I_i、BLE 测距 f(r_i)、先验权重 ω_i
- 支持静止 / 移动两种场景，移动场景通过运动一致性门控切换求解策略
- 验证指标：CEP90；移动场景相对静止基线的检测精度提升

## 运行

```bash
pip install -r requirements.txt
python run_meta_iter.py --data-dir data/ --config config.example.yaml
