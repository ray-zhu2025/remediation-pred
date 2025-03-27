"""
主程序入口
用于训练和评估模型，对比不同版本效果
"""

import os
import logging
from datetime import datetime
from typing import Dict, List
from models.soil_model import SoilModel
from models.groundwater_model import GroundwaterModel
from process.data_processor import DataProcessor
from config.version_config import VersionConfig
from utils.exceptions import *
import pandas as pd
import json
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import traceback

def setup_logging():
    """设置日志配置"""
    log_dir = "output/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"all_v{VersionConfig.CURRENT_VERSION}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== 开始运行 all 模型 v{VersionConfig.CURRENT_VERSION} ===")
    logger.info(f"版本更新记录: {VersionConfig.VERSION_HISTORY[VersionConfig.CURRENT_VERSION]['changes']}")
    logger.info("")
    return logger

def train_and_evaluate_model(model, data: Dict, model_name: str, model_type: str) -> Dict:
    """
    训练和评估模型
    
    Args:
        model: 模型实例
        data: 训练和测试数据
        model_name: 模型名称
        model_type: 模型类型
        
    Returns:
        评估指标字典
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 训练模型
        logger.info(f"开始训练{model_name}...")
        model.train(data['X_train'], data['y_train'])
        logger.info(f"{model_name} 训练完成")
        
        # 评估模型
        logger.info(f"{model_name} 测试集评估结果:")
        metrics = model.evaluate(data['X_test'], data['y_test'])
        
        # 执行交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(data['X_train']), 1):
            logger.info(f"执行第 {fold} 折交叉验证...")
            X_fold_train = data['X_train'][train_idx]
            y_fold_train = data['y_train'][train_idx]
            X_fold_val = data['X_train'][val_idx]
            y_fold_val = data['y_train'][val_idx]
            
            # 训练模型
            model.train(X_fold_train, y_fold_train)
            # 预测并计算准确率
            y_pred = model.predict(X_fold_val)
            accuracy = np.mean(y_pred == y_fold_val)
            cv_scores.append(accuracy)
            logger.info(f"第 {fold} 折准确率: {accuracy:.4f}")
        
        logger.info(f"{model_name} 交叉验证结果:")
        logger.info(f"CV准确率: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        
        # 添加交叉验证结果到评估指标
        metrics['cv_accuracy_mean'] = np.mean(cv_scores)
        metrics['cv_accuracy_std'] = np.std(cv_scores)
        
        logger.info(f"{model_name} 评估完成")
        return metrics
    except Exception as e:
        raise TrainingError(f"模型训练或评估失败: {str(e)}\n{traceback.format_exc()}")

def save_metrics(metrics, model_type):
    """保存评估指标"""
    metrics_dir = "output/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(metrics_dir, f"{model_type}_v{VersionConfig.CURRENT_VERSION}_{timestamp}.json")
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logging.info(f"评估指标已保存到: {metrics_file}")

def main():
    """主函数"""
    logger = setup_logging()
    
    # 创建数据处理器
    data_processor = DataProcessor(
        data_dir='data/training',
        test_size=0.2,
        random_state=42,
        use_oversampling=True,
        sampling_method='adasyn'  # 使用ADASYN作为采样策略
    )
    
    # 训练和评估土壤模型
    logger.info("\n训练和评估土壤模型...")
    soil_model = SoilModel()
    
    # 加载数据
    soil_data = data_processor.load_soil_data()
    
    # 训练模型
    logger.info(f"开始训练SoilModel (v{VersionConfig.CURRENT_VERSION})...")
    soil_model.train(
        soil_data['X_train'],
        soil_data['y_train']
    )
    
    # 评估模型
    soil_metrics = soil_model.evaluate(
        soil_data['X_test'],
        soil_data['y_test']
    )
    
    # 保存评估指标
    save_metrics(soil_metrics, 'soil')
    
    # 训练和评估地下水模型
    logger.info("\n训练和评估地下水模型...")
    groundwater_model = GroundwaterModel()
    
    # 加载数据
    groundwater_data = data_processor.load_groundwater_data()
    
    # 训练模型
    logger.info(f"开始训练GroundwaterModel (v{VersionConfig.CURRENT_VERSION})...")
    groundwater_model.train(
        groundwater_data['X_train'],
        groundwater_data['y_train']
    )
    
    # 评估模型
    groundwater_metrics = groundwater_model.evaluate(
        groundwater_data['X_test'],
        groundwater_data['y_test']
    )
    
    # 保存评估指标
    save_metrics(groundwater_metrics, 'groundwater')
    
    logger.info("\n=== 所有模型运行完成 ===")
    logger.info("结果文件保存在 output/ 目录下")
    logger.info("模型可解释性分析结果保存在 output/*/explanation/ 目录下")

if __name__ == "__main__":
    main() 