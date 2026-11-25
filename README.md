# Poisoning the Normalizer: Statistics Poisoning Attacks and Robust Aggregation for StatAvg-based Federated Intrusion Detection

面向 **Computers, Materials & Continua (CMC)** 专栏 *AI-Driven Network Security and Trustworthy Artificial Intelligence* 的完整实验代码与研究文档。

## 目录结构

```text
StatAvg_Poisoning_CMC/
├── README.md                          # 本文件（快速上手）
├── docs/
│   └── 完整研究方案与实验文档.md        # 完整选题原理、数学公式、实验设计与论文写作底稿
├── statavg_poison/                    # 核心实验代码（忠实复现 IEEE TNSM 2025 StatAvg）
│   ├── poison.py                      # 5 种统计量投毒攻击 + 样本数谎报 (n_inflate)
│   ├── robust_stats.py                # 5 种统计量聚合防御 + MAD 离群客户端检测
│   ├── dataset.py                     # TON_IoT Linux Memory / 通用 IDS / 合成数据加载与 Dirichlet 非独立同分布划分
│   ├── model.py                       # 官方同款 3 隐藏层 MLP (128-128-128, Adam lr=0.002)
│   └── simulator.py                   # 联邦训练引擎（含 FedAvg、StatAvg 与模型层 Multi-Krum 基线）
├── dataset/                           # 存放真实 TON_IoT dataset.csv 或合成验证集
├── results_smoke/                     # 冒烟测试输出（CSV + 3 张高清图）
├── results_main/                      # 完整论文主实验矩阵输出（CSV + 3 张高清图）
├── smoke_test.py                      # 离线核心攻防逻辑单元测试
├── make_synthetic_dataset.py          # 生成 TON_IoT 同结构合成数据集（无真数据时用于冒烟）
├── run_experiments.py                 # 一键运行 smoke / main 实验矩阵
└── collect_and_plot.py                # 汇总 results_summary.csv 并绘制 3 张论文核心图
```

## 快速开始

### 1. 运行离线逻辑自测
```bash
python3 smoke_test.py
```

### 2. 放置真实数据集（推荐用于论文终稿）
- **TON_IoT Linux Memory（主数据集，与 StatAvg 原文一致）**：
  1. 前往 UNSW 官方页面：https://research.unsw.edu.au/projects/toniot-datasets
  2. 进入 `TON_IoT datasets / Train_Test_datasets / Train_Test_Linux_dataset` 下载 `Train_test_linux_memory.csv`
  3. 重命名为 `dataset.csv` 放入 `dataset/dataset.csv`
- **第二数据集（可选，用于泛化验证）**：可将 TON_IoT Network CSV 或 CICIoT2023 抽样子集放入 `dataset/dataset_network.csv`，通过 `--dataset` 参数指定。

### 3. 运行实验与自动出图
```bash
# 快速冒烟测试（约 1 分钟）
python3 run_experiments.py --dataset dataset/dataset.csv --mode smoke --out-dir results_smoke
python3 collect_and_plot.py --results-dir results_smoke

# 完整论文主实验矩阵（3 随机种子，含全部攻防、Multi-Krum 对比、消融与干净开销）
python3 run_experiments.py --dataset dataset/dataset.csv --mode main --seeds 42 43 44 --out-dir results_main
python3 collect_and_plot.py --results-dir results_main
```

详细原理、实验设计与论文写作底稿请阅读 [docs/完整研究方案与实验文档.md](./docs/完整研究方案与实验文档.md)。
