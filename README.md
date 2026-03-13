# Silence Pattern Decoder in Voting

沉默/弃权投票模式解码器 - 社会科学实验计算分析工具

[![Tests](https://github.com/your-org/silence-decoder/actions/workflows/tests.yml/badge.svg)](https://github.com/your-org/silence-decoder/actions/workflows/tests.yml)
[![Lint](https://github.com/your-org/silence-decoder/actions/workflows/lint.yml/badge.svg)](https://github.com/your-org/silence-decoder/actions/workflows/lint.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

## 简介

这是一个**社会科学研究工具**，使用计算方法分析投票行为，特别关注**沉默模式（abstention patterns）**来检测：

- **隐性共识**（Implicit Consensus）：群体中未明确表达的共同意见
- **少数派影响**（Hidden Minority Influence）：少数群体对多数决策的隐性影响
- **群体压力动态**（Group Pressure Dynamics）：群体施加压力导致的沉默行为

本项目通过**蒙特卡洛模拟**和**图神经网络建模**，从弃权数据中反推群体动力学特征。

## 核心功能

- **多规则投票系统**：支持Approval、Plurality、加权投票等多种投票规则
- **社会影响图谱**：建立个体间影响关系的图神经网络模型
- **模式检测器**：自动检测共识、少数派压力、压迫性动态等模式
- **大规模模拟引擎**：支持数百至数千代理的并行模拟
- **可视化分析**：提供时间序列、分布、网络图等多种可视化工具

## 安装

```bash
cd silence-decoder

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 依赖

- numpy >= 1.24.0
- networkx >= 3.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0

## 使用方法

### 命令行工具

```bash
# 运行单次模拟
python -m silence_decoder.cli run-sim \
    --num-agents 100 \
    --num-candidates 3 \
    --num-rounds 50 \
    --influence-density 0.3 \
    --output results/

# 运行完整实验（共识、少数派、压迫）
python -m silence_decoder.cli run-experiment \
    --experiment-type all \
    --num-iterations 100 \
    --output results/

# 分析已保存的数据
python -m silence_decoder.cli analyze \
    --input results/voting_data.csv \
    --output reports/
```

### Python API

```python
from silence_decoder.src.agent import Agent
from silence_decoder.src.influence import InfluenceGraph
from silence_decoder.src.voting import VotingSystem
from silence_decoder.src.pattern_detector import PatternDetector
from silence_decoder.src.simulation import SimulationEngine

# 创建代理
agent = Agent(
    agent_id=1,
    belief=[0.3, 0.7],
    influence_tolerance=0.5,
    opinion_strength=0.8
)

# 构建影响图
graph = InfluenceGraph()
graph.add_edge(1, 2, weight=0.7)
graph.add_edge(2, 3, weight=0.5)

# 运行投票
voting_system = VotingSystem()
result = voting_system.run_vote(
    agents=[agent1, agent2, agent3],
    candidates=['A', 'B', 'C'],
    rule='approval'
)

# 检测模式
detector = PatternDetector()
consensus = detector.detect_consensus(voting_data)
minority_pressure = detector.detect_minority_pressure(voting_data)
```

## 项目结构

```
silence-decoder/
├── src/                    # 核心代码
│   ├── agent.py           # 代理数据模型
│   ├── influence.py       # 影响力图谱
│   ├── voting.py          # 投票系统
│   ├── simulation.py      # 模拟引擎
│   ├── pattern_detector.py # 模式检测器
│   ├── analysis.py        # 分析工具
│   ├── visualizer.py      # 可视化
│   ├── data_generator.py  # 数据生成器
│   └── cache.py           # 缓存层
├── experiments/            # 实验脚本
│   ├── consensus_test.py
│   ├── minority_test.py
│   ├── oppression_test.py
│   └── config.json
├── notebooks/              # Jupyter 笔记本
│   ├── 01-introduction.ipynb
│   ├── 02-consensus-detection.ipynb
│   └── 03-influence-analysis.ipynb
├── benchmarks/             # 性能测试
│   └── performance.py
├── tests/                  # 单元测试
│   ├── test_agent.py
│   ├── test_influence.py
│   ├── test_voting.py
│   └── test_pattern_detector.py
├── data/                   # 数据文件
├── models/                 # 训练好的模型
├── docs/                   # 文档
├── results/                # 实验结果
└── reports/                # 生成报告
```

## 实验设计

### 共识检测实验

验证在存在隐性共识的场景下，系统能否准确检测并量化共识强度。

- **场景**：100个代理，3个候选人，已知共识强度
- **指标**：检测准确率、共识强度估计误差

### 少数派影响实验

研究少数群体改变多数投票行为的能力。

- **场景**：80个多数派，20个少数派（不同信念）
- **控制变量**：少数派凝聚力、信心、一致性
- **输出**：少数派影响力阈值

### 压迫性动态实验

模拟压迫性群体环境导致的压制投票行为。

- **场景**：高影响力的主导代理
- **控制变量**：压迫水平（高/中/低）
- **输出**：压迫因子与观察到的弃权率相关性

## 深入理解：弃权模式分析

弃权（沉默）不仅是"不投票"，更是**信息信号**：

| 弃权类型 | 原因 | 隐含信息 |
|---------|------|---------|
| 信息不足弃权 | 缺乏足够信息做决定 | 群体信息传播不足 |
| 压迫弃权 | 担心表达真实观点 | 群体压迫强度 |
| 共识弃权 | 相信他人已做正确决定 | 强隐性共识 |
| 抗议弃权 | 对候选人不满 | 潜在不满情绪 |

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可

MIT License - See LICENSE file for details.

---

## 支持作者

如果您觉得这个项目对您有帮助，欢迎打赏支持！

![Buy Me a Coffee](buymeacoffee.png)

**Buy me a coffee (crypto)**

| 币种 | 地址 |
|------|------|
| BTC | `bc1qc0f5tv577z7yt59tw8sqaq3tey98xehy32frzd` |
| ETH / USDT | `0x3b7b6c47491e4778157f0756102f134d05070704` |
| SOL | `6Xuk373zc6x6XWcAAuqvbWW92zabJdCmN3CSwpsVM6sd` |
