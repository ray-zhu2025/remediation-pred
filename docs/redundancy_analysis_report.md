# 代码冗余分析报告

## 汇总统计
- 总重复问题数: 121
- 受影响文件数: 18
- 高严重程度: 0
- 中严重程度: 16
- 低严重程度: 105

## Duplicate Imports
发现 12 个重复问题

### 1. duplicate_import
**涉及文件:**
- ./run_all_versions.py
- ./src/main.py
- ./src/analysis/data_analyzer.py
- ./src/config/version_config.py
- ./src/utils/logging_config.py
- ./src/models/domain_model.py
- ./src/models/base_model.py
**重复次数:** 7

### 2. duplicate_import
**涉及文件:**
- ./src/main.py
- ./src/utils/logging_config.py
- ./src/models/base_model.py
- ./src/models/tabpfn_model.py
- ./src/process/data_processor.py
**重复次数:** 5

### 3. duplicate_import
**涉及文件:**
- ./src/main.py
- ./src/test_sampling.py
**重复次数:** 2

### 4. duplicate_import
**涉及文件:**
- ./src/main.py
- ./src/models/domain_model.py
- ./src/models/domain_tabpfn_model.py
**重复次数:** 3

### 5. duplicate_import
**涉及文件:**
- ./src/test_sampling.py
- ./src/process/data_processor.py
**重复次数:** 2

... 还有 7 个重复问题

## Duplicate Strings
发现 105 个重复问题

### 1. duplicate_string
**涉及文件:**
- ./run_all_versions.py
- ./run_all_versions.py
- ./src/utils/convert_xlsx_to_csv.py
**重复次数:** 3

### 2. duplicate_string
**涉及文件:**
- ./run_all_versions.py
- ./src/main.py
- ./src/analysis/data_analyzer.py
- ./src/config/version_config.py
- ./src/config/version_config.py
- ./src/config/version_config.py
- ./src/config/version_config.py
- ./src/config/version_config.py
- ./src/process/data_processor.py
- ./src/process/data_processor.py
**重复次数:** 10

### 3. duplicate_string
**涉及文件:**
- ./run_all_versions.py
- ./src/test_sampling.py
- ./src/analysis/data_analyzer.py
- ./src/utils/redundancy_analyzer.py
- ./src/utils/convert_xlsx_to_csv.py
**重复次数:** 5

### 4. duplicate_string
**涉及文件:**
- ./src/main.py
- ./src/main.py
**重复次数:** 2

### 5. duplicate_string
**涉及文件:**
- ./src/main.py
- ./src/test_sampling.py
- ./src/test_sampling.py
**重复次数:** 3

... 还有 100 个重复问题

## Duplicate Patterns
发现 4 个重复问题

### 1. 时间戳生成模式
**涉及文件:**
- {'file': './run_all_versions.py', 'matches': 1}
- {'file': './src/main.py', 'matches': 1}
- {'file': './src/analysis/data_analyzer.py', 'matches': 1}
- {'file': './src/config/version_config.py', 'matches': 3}
- {'file': './src/utils/logging_config.py', 'matches': 1}
- {'file': './src/models/base_model.py', 'matches': 3}

### 2. 目录创建模式
**涉及文件:**
- {'file': './src/main.py', 'matches': 4}
- {'file': './src/analysis/model_comparison.py', 'matches': 2}
- {'file': './src/analysis/data_analyzer.py', 'matches': 3}
- {'file': './src/utils/logging_config.py', 'matches': 1}
- {'file': './src/models/domain_model.py', 'matches': 1}
- {'file': './src/models/base_model.py', 'matches': 3}
- {'file': './src/models/tabpfn_model.py', 'matches': 1}

### 3. 日志器创建模式
**涉及文件:**
- {'file': './src/test_sampling.py', 'matches': 1}
- {'file': './src/analysis/model_comparison.py', 'matches': 2}
- {'file': './src/analysis/data_analyzer.py', 'matches': 1}
- {'file': './src/utils/redundancy_analyzer.py', 'matches': 1}
- {'file': './src/utils/logging_config.py', 'matches': 1}
- {'file': './src/utils/logging_utils.py', 'matches': 2}
- {'file': './src/models/base_model.py', 'matches': 2}
- {'file': './src/models/autogluon_tabpfn_model.py', 'matches': 1}
- {'file': './src/process/data_processor.py', 'matches': 1}

### 4. sklearn指标导入模式
**涉及文件:**
- {'file': './src/models/base_model.py', 'matches': 2}
- {'file': './src/models/tabpfn_model.py', 'matches': 1}

