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

# 训练和评估土壤模型
echo "训练土壤模型..."
python3 -c "
from src.models.soil_model import SoilModel
soil_model = SoilModel('src/config/soil/parameters.json')
soil_model.train('data/training/soil_training.csv')
soil_model.evaluate('data/training/soil_training.csv', '$soil_output_dir')
# soil_model.explain_model('$soil_output_dir/explanation')
"

# 训练和评估地下水模型
echo "训练地下水模型..."
python3 -c "
from src.models.groundwater_model import GroundwaterModel
groundwater_model = GroundwaterModel('src/config/groundwater/parameters.json')
groundwater_model.train('data/training/groundwater_training.csv')
groundwater_model.evaluate('data/training/groundwater_training.csv', '$groundwater_output_dir')
# groundwater_model.explain_model('$groundwater_output_dir/explanation')
"

echo "[INFO] 所有模型运行完成！"
echo "[INFO] 结果文件保存在 output/ 目录下"
echo "[INFO] 模型可解释性分析结果保存在 output/*/explanation/ 目录下"

# 设置项目根目录
PROJECT_ROOT="/Users/evanzhu/Desktop/GitHub/污染场地智能决策模型"

# 设置Python路径
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# 运行模型比较程序
python $PROJECT_ROOT/src/models/model_comparison.py 