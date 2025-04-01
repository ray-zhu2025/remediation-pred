import pandas as pd
from collections import Counter
from src.process.data_processor import DataProcessor
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sampling():
    # 初始化数据处理器
    processor = DataProcessor(
        data_dir='data/training',
        use_oversampling=True,
        sampling_method='smote'
    )
    
    # 测试土壤数据采样
    logger.info("\n=== 测试土壤数据采样 ===")
    soil_data = processor.load_soil_data()
    logger.info(f"采样前训练集类别分布: {Counter(soil_data['y_train'])}")
    logger.info(f"采样前训练集形状: {soil_data['X_train'].shape}")
    
    # 测试地下水数据采样
    logger.info("\n=== 测试地下水数据采样 ===")
    groundwater_data = processor.load_groundwater_data()
    logger.info(f"采样前训练集类别分布: {Counter(groundwater_data['y_train'])}")
    logger.info(f"采样前训练集形状: {groundwater_data['X_train'].shape}")

if __name__ == "__main__":
    test_sampling() 