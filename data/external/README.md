# Benchmark Voting Datasets

本目录包含用于验证模式检测方法的基准投票数据集。

## 数据集来源

| 数据集 | 类型 | 说明 |
|--------|------|------|
| `synthetic_consensus.csv` | 模拟 | 已知公识强度的合成数据 |
| `synthetic_minority.csv` | 模拟 | 已知少数派影响的合成数据 |
| `synthetic_oppression.csv` | 模拟 | 已知压迫水平的合成数据 |
| `synthetic_random.csv` | 模拟 | 随机基线数据 |

## 数据集格式

每个 CSV 文件包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `round` | int | 投票轮次 |
| `agent_id` | int | 代理ID |
| `vote` | string | 候选人ID (或 'abstain') |
| `abstention` | bool | 是否弃权 |
| `original_belief` | float | 代理原始信念值 |
| `influenced_belief` | float | 受影响后的信念值 |

## 使用示例

```python
import pandas as pd

# 加载数据集
df = pd.read_csv('data/external/synthetic_consensus.csv')

# 分析弃权率
abstention_rate = (df['abstention'] == True).mean()
print(f"弃权率: {abstention_rate:.2%}")

# 按轮次聚合
round_stats = df.groupby('round').agg({
    'abstention': 'mean',
    'agent_id': 'count'
}).rename(columns={'abstention': 'abstention_rate', 'agent_id': 'voters'})

print(round_stats.head())
```

## 生成新的基准数据集

使用数据生成器脚本：

```bash
python src/data_generator.py --type consensus --num-agents 100 --num-rounds 50 --output data/external/
```

## 验证数据集用途

- **共识检测验证**：使用 `synthetic_consensus.csv` 验证检测准确率
- **少数派影响验证**：使用 `synthetic_minority.csv` 验证影响阈值
- **压迫检测验证**：使用 `synthetic_oppression.csv` 验证压迫识别
