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
- uv (Python package manager)
- Dependencies listed in requirements.txt

## Installation
1. Clone repository
```bash
git clone https://github.com/ray-zhu2025/remediation-classification.git
cd remediation-classification
```

2. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create virtual environment and install dependencies
```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

uv pip install -r requirements.txt
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

### Script Call Hierarchy
The scripts are organized in a hierarchical structure:

```
run_all_versions.py
    └── run.sh
        └── src/main.py
            ├── Data Processing
            ├── Model Training
            └── Results Saving
```

- `run.sh`: Top-level entry script that handles environment setup and calls main.py
- `run_all_versions.py`: Batch processing script that runs multiple versions (1.0.0 to 1.1.6)
- `src/main.py`: Core program that handles data processing, model training, and result saving

Each version's results are saved independently with complete logging and evaluation metrics.

## Output
- Models saved in `src/models/`