#!/bin/bash

# 设置错误处理
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

# 设置Python路径
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# 切换到项目根目录
cd "$SCRIPT_DIR"

# 设置默认版本
DEFAULT_VERSION="1.0.0"

# 获取版本参数，如果没有提供则使用默认版本
VERSION=${1:-$DEFAULT_VERSION}

# 检查Python版本
echo -e "${YELLOW}[INFO] 检查Python版本...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}[ERROR] 未找到Python${NC}"
    exit 1
fi
echo -e "${GREEN}[INFO] Python版本: $(python --version | cut -d' ' -f2)${NC}"

# 检查uv是否安装
if ! command -v uv &> /dev/null; then
    echo -e "${RED}[ERROR] 未找到uv，请先安装uv${NC}"
    echo -e "${YELLOW}[INFO] 可以使用以下命令安装uv：${NC}"
    echo -e "${YELLOW}[INFO] curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi

# 检查并安装依赖
echo -e "${YELLOW}[INFO] 检查并安装依赖...${NC}"
uv pip install tabpfn>=2.0.9 imbalanced-learn scikit-learn pandas numpy

# 检查数据目录
if [ ! -d "data" ]; then
    echo -e "${RED}[ERROR] 未找到数据目录${NC}"
    exit 1
fi

# 检查数据文件
required_files=(
    "data/training/soil_training.csv"
    "data/training/groundwater_training.csv"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}[ERROR] 未找到数据文件: $file${NC}"
        exit 1
    fi
done

# 创建必要的目录
directories=(
    "models"
    "models/soil"
    "models/groundwater"
    "output"
    "output/logs"
    "output/metrics"
    "output/plots"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        echo -e "${YELLOW}[INFO] 创建目录: $dir${NC}"
        mkdir -p "$dir"
    fi
done

# 运行模型
echo -e "${YELLOW}[INFO] 开始运行模型...${NC}"

# 获取命令行参数
MODEL_TYPE="autogluon"  # 默认使用AutoGluon

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}[ERROR] 未知参数: $1${NC}"
            exit 1
            ;;
    esac
done

# 验证模型类型
if [[ "$MODEL_TYPE" != "autogluon" && "$MODEL_TYPE" != "tabpfn" ]]; then
    echo -e "${RED}[ERROR] 不支持的模型类型: $MODEL_TYPE${NC}"
    echo -e "${YELLOW}支持的模型类型: autogluon, tabpfn${NC}"
    exit 1
fi

echo -e "${GREEN}[INFO] 使用模型类型: $MODEL_TYPE${NC}"
echo -e "${GREEN}[INFO] 使用版本: $VERSION${NC}"

# 运行主程序
python src/main.py --version "$VERSION" --model_type "$MODEL_TYPE"

echo -e "${GREEN}[INFO] 模型运行完成${NC}"

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