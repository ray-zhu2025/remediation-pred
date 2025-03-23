"""
土壤修复技术决策模型
使用多种机器学习模型预测土壤修复技术
"""

import json
import os
from pathlib import Path
import sys
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.append(project_root)

# 本地应用导入
from src.process.data_processor import DataProcessor
from src.utils.logging import setup_logging
from src.models.model_explainer import ModelExplainer
from src.models.base_models.model_factory import ModelFactory

class SoilModel:
    """土壤污染修复决策模型"""
    
    def __init__(self, 
                 config_path: str = "src/config/soil/parameters.json", 
                 use_hyperopt: bool = False, 
                 search_method: str = 'grid',
                 model_types: List[str] = None):
        """
        初始化土壤模型
        
        Args:
            config_path: 配置文件路径
            use_hyperopt: 是否使用超参数优化
            search_method: 超参数搜索方法，'grid' 或 'random'
            model_types: 要使用的基础模型类型列表，可选值：['decision_tree', 'random_forest', 'naive_bayes']
                       如果为None，则使用所有模型
        """
        self.data_processor = DataProcessor()
        self.config = self._load_config(config_path)
        self.use_hyperopt = use_hyperopt
        self.search_method = search_method
        self.model_types = model_types or ['decision_tree', 'random_forest', 'naive_bayes']
        self.models = self._initialize_models()
        self.label_encoders = {}
        self.logger = setup_logging()
        self.feature_names = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _initialize_models(self) -> List:
        """初始化基础模型"""
        models = ModelFactory.create_models(use_hyperopt=self.use_hyperopt)
        # 如果指定了模型类型，只返回指定的模型
        if self.model_types:
            return [models[model_type] for model_type in self.model_types]
        return list(models.values())
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据"""
        df_processed = df.copy()
        
        # 识别分类列
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # 对每个分类列进行编码
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df_processed[column] = self.label_encoders[column].fit_transform(df_processed[column])
            else:
                df_processed[column] = self.label_encoders[column].transform(df_processed[column])
        
        # 保存特征名称
        self.feature_names = df_processed.columns.tolist()
        
        return df_processed
    
    def train(self, train_data_path: str) -> None:
        """训练模型"""
        # 加载和处理训练数据
        train_df = self.data_processor.load_data(train_data_path)
        train_df = self._preprocess_data(train_df)
        X, y = self.data_processor.prepare_training_data(train_df)
        
        # 划分训练集、验证集和测试集
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
        
        # 保存数据集
        self.train_data = (X_train, y_train)
        self.val_data = (X_val, y_val)
        self.test_data = (X_test, y_test)
        
        # 训练所有模型
        for model in self.models:
            model.fit(X_train, y_train)
        
        self.logger.info(f"{self.__class__.__name__} 训练完成")
    
    def predict(self, pred_data_path: str, output_dir: str) -> None:
        """进行预测"""
        # 加载和处理预测数据
        pred_df = self.data_processor.load_data(pred_data_path)
        pred_df = self._preprocess_data(pred_df)
        
        # 确保预测数据包含相同的特征
        if self.feature_names is None:
            raise ValueError("未指定特征名")
            
        # 选择相同的特征
        pred_df = pred_df[self.feature_names]
        
        # 准备预测数据
        X_pred, I_pred, D_pred = self.data_processor.prepare_prediction_data(
            pred_df, self.data_processor.load_data(self.config['train_data_path'])
        )
        
        # 使用所有模型进行预测
        predictions = []
        for model in self.models:
            pred = model.predict(X_pred)
            predictions.append(pred)
        
        # 计算修复成本和周期
        results = []
        for model_pred in predictions:
            model_results = []
            for i, pred in enumerate(model_pred):
                costs, time = self.data_processor.calculate_costs_and_time(
                    np.array([pred]), D_pred[i:i+1],
                    self.config['prices'],
                    self.config['periods']
                )
                
                # 整理结果
                result = pd.DataFrame(np.column_stack((
                    I_pred[i:i+1], np.array([pred]), X_pred[i:i+1, 2], D_pred[i:i+1], costs, time
                )))
                model_results.append(result)
            
            # 合并该模型的所有预测结果
            if model_results:
                combined_result = pd.concat(model_results, ignore_index=True)
                results.append(combined_result)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存预测结果
        output_paths = [
            os.path.join(output_dir, f'prediction_{type(model).__name__}.csv')
            for model in self.models
        ]
        self.data_processor.save_results(results, output_paths)
    
    def evaluate(self, test_data_path: str, output_dir: str) -> None:
        """评估模型性能"""
        try:
            # 加载和处理测试数据
            test_df = self.data_processor.load_data(test_data_path)
            test_df = self._preprocess_data(test_df)
            X_test, y_test = self.data_processor.prepare_training_data(test_df)
            
            # 创建评估输出目录
            eval_output_dir = os.path.join(output_dir, 'evaluation')
            os.makedirs(eval_output_dir, exist_ok=True)
            
            # 评估所有模型
            all_results = []
            
            for model in self.models:
                # 预测
                y_pred = model.predict(X_test)
                
                # 计算评估指标
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                total_samples = len(y_test)
                
                # 记录评估结果
                self.logger.info(f"{type(model).__name__} 测试集评估结果:")
                self.logger.info(f"总样本量: {total_samples}")
                self.logger.info(f"准确率: {accuracy:.4f}")
                self.logger.info(f"精确率: {precision:.4f}")
                self.logger.info(f"召回率: {recall:.4f}")
                self.logger.info(f"F1分数: {f1:.4f}")
                
                # 保存整体结果
                all_results.append({
                    '模型': type(model).__name__,
                    '总样本量': total_samples,
                    '准确率': accuracy,
                    '精确率': precision,
                    '召回率': recall,
                    'F1分数': f1
                })
            
            # 保存评估结果
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(os.path.join(eval_output_dir, 'evaluation_results.csv'), index=False)
            
        except Exception as e:
            self.logger.error(f"评估过程中发生错误: {str(e)}")
            raise 