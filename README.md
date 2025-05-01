[English](README.md) | [中文](README_zh.md)

# Remediation Classification Model

## Project Overview
This project is a machine learning-based intelligent decision system for contaminated site remediation. It includes two sub-models for soil and groundwater, providing scientific remediation recommendations based on site characteristics and historical data.

## Features
- Soil and groundwater contamination prediction
- Multi-version model management
- Automatic model training and evaluation
- Detailed logging
- Model performance metrics analysis
- Explainability analysis (SHAP values)

## Project Structure

```
.
├── src/                    # Source code directory
│   ├── analysis/          # Data analysis module
│   ├── config/            # Configuration files
│   ├── models/            # Model definitions
│   ├── process/           # Data processing module
│   ├── utils/             # Utility functions
│   ├── main.py            # Main program entry
│   └── test_sampling.py   # Sampling test script
├── data/                   # Data directory
│   ├── training/          # Training data
│   ├── prediction/        # Prediction data
│   └── parameters/        # Model parameters
├── docs/                   # Documentation directory
├── models/                 # Model storage directory
├── output/                 # Output directory
│   ├── analysis/          # Analysis results
│   ├── groundwater/       # Groundwater model output
│   ├── logs/              # Log files
│   ├── metrics/           # Evaluation metrics
│   ├── soil/              # Soil model output
│   └── docs/              # Output documentation
├── requirements.txt        # Dependencies
├── run.sh                  # Run script
├── run_all_versions.py     # Multi-version run script
└── README_zh.md           # Chinese documentation
```

## Environment Requirements
- Python 3.8+
- Dependencies listed in requirements.txt

## Installation
1. Clone repository
```bash
git clone https://github.com/ray-zhu2025/remediation-classification.git
cd remediation-classification
```

2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
1. Run single version
```bash
python src/main.py --version v1.0
```

2. Run all versions
```bash
python run_all_versions.py
```

3. Use run script
```bash
./run.sh
```

## Output
- Models saved in `src/models/`
- Logs saved in `output/logs/`
- Evaluation metrics saved in `output/metrics/`

## Documentation
- [中文文档 (Chinese Documentation)](README_zh.md)

## License
GNU Lesser General Public License v2.1 (LGPL-2.1) 