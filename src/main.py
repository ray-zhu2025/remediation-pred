"""
主程序入口
用于训练和评估模型，对比SMOTE效果
"""

import os
import logging
from datetime import datetime
from typing import Dict, List
from src.models.medium_models.soil_model import SoilModel
from src.models.medium_models.groundwater_model import GroundwaterModel
from src.process.data_processor import DataProcessor
from src.models.model_explainer import ModelExplainer
import pandas as pd
import json
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

def setup_logging():
    """设置日志"""
    # 创建logs目录
    os.makedirs('output/logs', exist_ok=True)
    
    # 设置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(f'output/logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def train_and_evaluate_model(model, data: Dict, model_name: str) -> Dict:
    """
    训练和评估模型
    
    Args:
        model: 模型实例
        data: 训练和测试数据
        model_name: 模型名称
        
    Returns:
        评估指标字典
    """
    logger = logging.getLogger(__name__)
    
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
        X_fold_train = data['X_train'][train_idx]
        y_fold_train = data['y_train'][train_idx]
        X_fold_val = data['X_train'][val_idx]
        y_fold_val = data['y_train'][val_idx]
        
        # 训练模型
        model.train(X_fold_train, y_fold_train)
        # 预测并计算准确率
        y_pred = model.predict(X_fold_val, output_dir='output/cv_predictions')
        accuracy = np.mean(y_pred == y_fold_val)
        cv_scores.append(accuracy)
    
    logger.info(f"{model_name} 交叉验证结果:")
    logger.info(f"CV准确率: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    
    # 添加交叉验证结果到评估指标
    for model_type in metrics.keys():
        metrics[model_type]['cv_accuracy_mean'] = np.mean(cv_scores)
        metrics[model_type]['cv_accuracy_std'] = np.std(cv_scores)
    
    logger.info(f"{model_name} 评估完成")
    return metrics

def save_metrics(metrics: Dict, version: str):
    """保存评估指标"""
    os.makedirs('output/metrics', exist_ok=True)
    with open(f'output/metrics/metrics_{version}.json', 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 不使用SMOTE的数据处理器
    logger.info("\n=== 训练不使用SMOTE的模型 ===")
    data_processor_no_smote = DataProcessor(
        use_oversampling=False
    )
    
    # 使用SMOTE的数据处理器
    logger.info("\n=== 训练使用SMOTE的模型 ===")
    data_processor_smote = DataProcessor(
        use_oversampling=True,
        sampling_method='smote',
        sampling_strategy='auto'
    )
    
    # 加载两个版本的数据
    logger.info("加载数据...")
    soil_data_no_smote = data_processor_no_smote.load_soil_data()
    groundwater_data_no_smote = data_processor_no_smote.load_groundwater_data()
    
    soil_data_smote = data_processor_smote.load_soil_data()
    groundwater_data_smote = data_processor_smote.load_groundwater_data()
    
    # 训练和评估不使用SMOTE的模型
    metrics_no_smote = {}
    
    logger.info("\n训练和评估不使用SMOTE的土壤模型...")
    soil_model_no_smote = SoilModel(enable_explanation=False)
    metrics_no_smote['soil'] = train_and_evaluate_model(
        soil_model_no_smote, soil_data_no_smote, "SoilModel (No SMOTE)"
    )
    
    logger.info("\n训练和评估不使用SMOTE的地下水模型...")
    groundwater_model_no_smote = GroundwaterModel(enable_explanation=False)
    metrics_no_smote['groundwater'] = train_and_evaluate_model(
        groundwater_model_no_smote, groundwater_data_no_smote, "GroundwaterModel (No SMOTE)"
    )
    
    # 保存不使用SMOTE的评估结果
    save_metrics(metrics_no_smote, 'no_smote')
    
    # 训练和评估使用SMOTE的模型
    metrics_smote = {}
    
    logger.info("\n训练和评估使用SMOTE的土壤模型...")
    soil_model_smote = SoilModel(enable_explanation=False)
    metrics_smote['soil'] = train_and_evaluate_model(
        soil_model_smote, soil_data_smote, "SoilModel (SMOTE)"
    )
    
    logger.info("\n训练和评估使用SMOTE的地下水模型...")
    groundwater_model_smote = GroundwaterModel(enable_explanation=False)
    metrics_smote['groundwater'] = train_and_evaluate_model(
        groundwater_model_smote, groundwater_data_smote, "GroundwaterModel (SMOTE)"
    )
    
    # 保存使用SMOTE的评估结果
    save_metrics(metrics_smote, 'smote')
    
    # 比较结果
    logger.info("\n=== SMOTE效果对比 ===")
    
    for model_type in ['soil', 'groundwater']:
        logger.info(f"\n{model_type.capitalize()}模型对比:")
        for model_name in metrics_smote[model_type].keys():
            for metric in metrics_smote[model_type][model_name].keys():
                no_smote_value = metrics_no_smote[model_type][model_name][metric]
                smote_value = metrics_smote[model_type][model_name][metric]
                diff = smote_value - no_smote_value
                logger.info(f"{model_name} - {metric}: {diff:+.4f} ({no_smote_value:.4f} -> {smote_value:.4f})")

if __name__ == '__main__':
    main() 