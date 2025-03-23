"""
随机森林基础模型类
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
from .base_model import BaseModel

class RandomForestBaseModel(BaseModel):
    """随机森林基础模型类"""
    
    def __init__(self, use_hyperopt: bool = True):
        """
        初始化随机森林模型
        
        Args:
            use_hyperopt: 是否使用超参数优化
        """
        super().__init__()
        self.use_hyperopt = use_hyperopt
        self.model = None
        self.best_params = None
        self.logger = logging.getLogger(__name__)
        
    def _get_param_grid(self) -> Dict[str, Any]:
        """获取参数网格"""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
        
    def _get_param_distributions(self) -> Dict[str, Any]:
        """获取参数分布"""
        return {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 10),
            'min_samples_split': randint(2, 5),
            'min_samples_leaf': randint(1, 3),
            'criterion': ['gini', 'entropy']
        }
        
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
                'randomforest__n_estimators': [100, 200, 300],
                'randomforest__max_depth': [3, 5, 7, 10],
                'randomforest__min_samples_split': [2, 5, 10],
                'randomforest__min_samples_leaf': [1, 2, 4]
            }
            
            # 创建管道
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('randomforest', RandomForestClassifier(random_state=42))
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
            self.best_params = search.best_params_
            self.logger.info(f"最佳参数: {search.best_params_}")
        else:
            # 使用默认参数创建模型
            self.model = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('randomforest', RandomForestClassifier(random_state=42))
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
        if hasattr(self.model, 'named_steps'):
            rf_model = self.model.named_steps['randomforest']
        else:
            rf_model = self.model
            
        return dict(enumerate(rf_model.feature_importances_))
        
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
            self.logger.info(f"最佳参数: {self.best_params}")
            
        return metrics 