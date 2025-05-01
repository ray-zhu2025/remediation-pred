# 污染场地智能决策模型

## 项目概述
本项目是一个基于机器学习的污染场地智能决策系统，用于预测和评估污染场地的修复方案。系统包含土壤和地下水两个子模型，能够根据场地特征和历史数据，提供科学的修复建议。

## 功能特点
- 支持土壤和地下水污染预测
- 多版本模型管理
- 自动模型训练和评估
- 详细的日志记录
- 模型性能指标分析
- 可解释性分析（SHAP值）

## 项目结构

```
.
├── src/                    # 源代码目录
│   ├── analysis/          # 数据分析模块
│   ├── config/            # 配置文件
│   ├── models/            # 模型定义
│   ├── process/           # 数据处理模块
│   ├── utils/             # 工具函数
│   ├── main.py            # 主程序入口
│   └── test_sampling.py   # 采样测试脚本
├── data/                   # 数据目录
│   ├── training/          # 训练数据
│   ├── prediction/        # 预测数据
│   └── parameters/        # 模型参数
├── docs/                   # 文档目录
├── models/                 # 模型存储目录
├── output/                 # 输出目录
│   ├── analysis/          # 分析结果
│   ├── groundwater/       # 地下水模型输出
│   ├── logs/              # 日志文件
│   ├── metrics/           # 评估指标
│   ├── soil/              # 土壤模型输出
│   └── docs/              # 输出文档
├── requirements.txt        # 依赖包列表
├── run.sh                  # 运行脚本
├── run_all_versions.py     # 多版本运行脚本
└── README_zh.md           # 中文说明文档
```

## 环境要求
- Python 3.8+
- 依赖包见 requirements.txt

## 安装步骤
1. 克隆仓库
```bash
git clone https://github.com/ray-zhu2025/remediation-classification.git
cd remediation-classification
```

2. 创建虚拟环境
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法
1. 运行单个版本
```bash
python src/main.py --version v1.0
```

2. 运行所有版本
```bash
python run_all_versions.py
```

3. 使用运行脚本
```bash
./run.sh
```

## 输出说明
- 模型保存在 `src/models/` 目录
- 日志保存在 `output/logs/` 目录
- 评估指标保存在 `output/metrics/` 目录

## 许可证
MIT License 