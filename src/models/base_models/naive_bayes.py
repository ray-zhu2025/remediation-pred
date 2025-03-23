"""
朴素贝叶斯基础模型类
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
from .base_model import BaseModel

class NaiveBayesBaseModel(BaseModel):
    """朴素贝叶斯基础模型类"""
    
    def __init__(self, use_hyperopt: bool = True):
        """
        初始化朴素贝叶斯模型
        
        Args:
            use_hyperopt: 是否使用超参数优化
        """
        super().__init__()
        self.use_hyperopt = use_hyperopt
        self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签向量
        """
        if self.use_hyperopt:
            # 定义参数网格
            param_grid = {
                'gaussiannb__var_smoothing': np.logspace(-10, -8, 30)
            }
            
            # 创建管道
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('gaussiannb', GaussianNB())
            ])
            
            # 创建网格搜索对象
            search = GridSearchCV(
                pipeline,
                param_grid,
                cv=3,  # 使用3折交叉验证
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            # 执行网格搜索
            search.fit(X, y)
            
            # 使用最佳参数创建模型
            self.model = search.best_estimator_
            self.logger.info(f"最佳参数: {search.best_params_}")
        else:
            # 使用默认参数创建模型
            self.model = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('gaussiannb', GaussianNB())
            ])
            self.model.fit(X, y)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本标签
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的标签
        """
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的概率
        """
        return self.model.predict_proba(X)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典
        """
        # 朴素贝叶斯模型没有直接的特征重要性度量
        # 这里使用类条件概率的方差作为近似
        if hasattr(self.model, 'named_steps'):
            nb_model = self.model.named_steps['gaussiannb']
        else:
            nb_model = self.model
            
        # 计算每个特征的类条件概率方差
        feature_importance = np.var(nb_model.theta_, axis=0)
        return dict(enumerate(feature_importance))
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X: 特征矩阵
            y: 真实标签
            
        Returns:
            包含各项评估指标的字典
        """
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y, y_pred_proba, multi_class='ovr')
        }
        
        self.logger.info(f"模型评估结果: {metrics}")
        if self.use_hyperopt:
            self.logger.info(f"最佳参数: {self.model.named_steps['gaussiannb'].get_params()}")
            
        return metrics 