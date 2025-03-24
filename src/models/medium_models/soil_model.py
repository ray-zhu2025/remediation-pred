"""
土壤修复技术决策模型
使用多种机器学习模型预测土壤修复技术
"""

import json
import os
from pathlib import Path
import sys
import logging
import time
from typing import Dict, List, Optional, Union
from tqdm import tqdm

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
    """土壤模型类"""
    
    def __init__(self, 
                 config_path: str = 'src/config/soil/parameters.json',
                 use_hyperopt: bool = False,
                 search_method: str = 'bayesian',
                 model_types: Optional[List[str]] = None,
                 enable_explanation: bool = False):
        """
        初始化土壤模型
        
        Args:
            config_path: 配置文件路径
            use_hyperopt: 是否使用超参数优化
            search_method: 搜索方法
            model_types: 要使用的模型类型列表
            enable_explanation: 是否启用模型可解释性分析
        """
        self.config_path = config_path
        self.use_hyperopt = use_hyperopt
        self.search_method = search_method
        self.model_types = model_types
        self.enable_explanation = enable_explanation
        self.logger = logging.getLogger(__name__)
        self.label_encoders = {}
        self.feature_names = None
        
        # 加载配置
        self._load_config()
        
        # 初始化数据处理器
        self.data_processor = DataProcessor(
            use_oversampling=True,
            sampling_method='smote',
            sampling_strategy='auto'
        )
        
        # 初始化模型
        self.models = self._initialize_models()
        
    def _load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            
    def _initialize_models(self) -> Dict:
        """
        初始化模型
        
        Returns:
            模型字典
        """
        models = ModelFactory.create_models(use_hyperopt=self.use_hyperopt)
        if self.model_types:
            return {k: v for k, v in models.items() if k in self.model_types}
        return models
        
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
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """训练模型"""
        start_time = time.time()
        self.logger.info("开始训练模型...")
        
        # 使用tqdm创建进度条
        for model_type, model in tqdm(self.models.items(), desc="训练模型"):
            model.fit(X_train, y_train)
            
        train_time = time.time() - start_time
        self.logger.info(f"模型训练完成，耗时: {train_time:.2f}秒")
    
    def predict(self, X: np.ndarray, output_dir: str = None) -> np.ndarray:
        """
        使用模型进行预测
        
        Args:
            X: 输入特征
            output_dir: 输出目录
            
        Returns:
            预测结果
        """
        logger = logging.getLogger(__name__)
        logger.info("开始预测...")
        
        # 使用随机森林模型进行预测
        y_pred = self.models['random_forest'].predict(X)
        
        # 如果指定了输出目录，保存预测结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            pred_df = pd.DataFrame({
                'predicted': y_pred
            })
            pred_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
        
        logger.info("预测完成")
        return y_pred
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, output_dir: str = 'output/evaluation') -> Dict[str, float]:
        """评估模型性能"""
        start_time = time.time()
        self.logger.info("开始评估模型...")
        metrics = {}
        
        # 使用tqdm创建进度条
        for model_type, model in tqdm(self.models.items(), desc="评估模型"):
            # 预测和计算指标
            y_pred = model.predict(X_test)
            metrics[model_type] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            
            # 如果启用了模型可解释性分析
            if self.enable_explanation:
                # 创建模型输出目录
                model_output_dir = os.path.join(output_dir, model_type, 'explanation')
                os.makedirs(model_output_dir, exist_ok=True)
                
                # 获取特征名称
                feature_names = self.feature_names or [f'feature_{i}' for i in range(X_test.shape[1])]
                
                # 生成模型解释
                explainer = ModelExplainer(model, feature_names, model_output_dir)
                with tqdm(total=3, desc=f"生成{model_type}模型解释") as pbar:
                    explainer.analyze_feature_importance(X_test)
                    pbar.update(1)
                    explainer.analyze_feature_effects(X_test)
                    pbar.update(1)
                    explainer.analyze_interactions(X_test)
                    pbar.update(1)
        
        eval_time = time.time() - start_time
        self.logger.info(f"模型评估完成，耗时: {eval_time:.2f}秒")
        
        # 输出汇总结果
        self.logger.info("\n模型评估结果汇总:")
        for model_type, model_metrics in metrics.items():
            self.logger.info(f"\n{model_type}:")
            for metric, value in model_metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")
        
        return metrics 