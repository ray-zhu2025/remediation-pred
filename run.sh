#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 切换到项目根目录
cd "$SCRIPT_DIR"

# 检查并创建输出目录
for dir in "output/soil" "output/groundwater"; do
    if [ ! -d "$dir" ]; then
        echo "[INFO] 创建输出目录: $dir"
        mkdir -p "$dir"
    fi
done

echo "[INFO] 开始运行污染场地智能决策模型系统..."

# 检查Python版本
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "[INFO] 检测到Python版本: $PYTHON_VERSION"

# 运行土壤模型
echo "[INFO] 开始运行土壤修复决策模型..."
python3 src/models/soil_model.py

# 运行地下水模型
echo "[INFO] 开始运行地下水修复决策模型..."
python3 src/models/groundwater_model.py

echo "[INFO] 所有模型运行完成！"
echo "[INFO] 结果文件保存在 output/ 目录下" 