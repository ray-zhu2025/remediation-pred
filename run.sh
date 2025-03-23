#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 切换到项目根目录
cd "$SCRIPT_DIR"

# 设置输出目录
output_dir="output"
soil_output_dir="$output_dir/soil"
groundwater_output_dir="$output_dir/groundwater"

# 创建输出目录
mkdir -p "$soil_output_dir/evaluation"
mkdir -p "$soil_output_dir/prediction"
mkdir -p "$soil_output_dir/explanation"
mkdir -p "$groundwater_output_dir/evaluation"
mkdir -p "$groundwater_output_dir/prediction"
mkdir -p "$groundwater_output_dir/explanation"

echo "[INFO] 开始运行污染场地智能决策模型系统..."

# 检查Python版本
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python版本: $python_version"

# 检查必要的Python包
echo "[INFO] 检查必要的Python包..."
python3 -c "import shap" 2>/dev/null || {
    echo "[INFO] 安装shap包..."
    pip3 install shap
}

# 设置项目根目录
PROJECT_ROOT="$SCRIPT_DIR"

# 设置Python路径
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# 运行主程序
echo "[INFO] 运行主程序..."
python3 src/main.py

echo "[INFO] 所有模型运行完成！"
echo "[INFO] 结果文件保存在 output/ 目录下"
echo "[INFO] 模型可解释性分析结果保存在 output/*/explanation/ 目录下" 