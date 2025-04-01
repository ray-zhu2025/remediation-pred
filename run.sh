#!/bin/bash

# 设置错误时立即退出
set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${GREEN}[INFO] $1${NC}"
}

print_warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 切换到项目根目录
cd "$SCRIPT_DIR"

# 设置默认版本
DEFAULT_VERSION="1.0.0"

# 获取版本参数，如果没有提供则使用默认版本
VERSION=${1:-$DEFAULT_VERSION}

# 检查Python版本
echo "[INFO] Python版本: $(python3 --version | cut -d' ' -f2)"

# 检查并安装依赖
echo "[INFO] 检查并安装依赖..."
pip install -r requirements.txt

# 检查数据目录
if [ ! -d "data" ]; then
    print_error "data目录不存在，请确保数据文件已放置正确"
    exit 1
fi

# 检查数据文件
if [ ! -f "data/training/soil_training.csv" ] || [ ! -f "data/training/groundwater_training.csv" ]; then
    print_error "训练数据文件不存在，请确保数据文件已放置正确"
    exit 1
fi

# 创建必要的目录
echo "[INFO] 创建必要的目录..."
mkdir -p models/soil/v${VERSION}
mkdir -p models/groundwater/v${VERSION}
mkdir -p output/metrics
mkdir -p output/soil/v${VERSION}/explanation
mkdir -p output/groundwater/v${VERSION}/explanation
mkdir -p logs

# 设置项目根目录
PROJECT_ROOT="$SCRIPT_DIR"

# 设置Python路径
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# 运行主程序
echo "[INFO] 开始运行污染场地智能决策模型系统..."
python3 src/main.py --version ${VERSION}

# 检查运行结果
if [ $? -eq 0 ]; then
    echo "[INFO] 模型训练和评估完成！"
    echo "[INFO] 结果文件保存在 output/ 目录下"
    echo "[INFO] 模型文件保存在 models/ 目录下"
    echo "[INFO] 日志文件保存在 logs/ 目录下"
else
    echo "[ERROR] 程序运行失败，请检查日志文件"
    exit 1
fi 