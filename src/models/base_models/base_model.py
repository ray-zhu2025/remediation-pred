"""
基础模型抽象类
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

class BaseModel(ABC):
    """基础模型抽象类"""
    
    def __init__(self):
        """初始化基础模型"""
        self.logger = logging.getLogger(__name__)
        self.model = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签向量
        """
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本标签
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的标签
        """
        return self.model.predict(X)
        
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测样本概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的概率
        """
        return self.model.predict_proba(X)
        
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典
        """
        pass
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X: 特征矩阵
            y: 标签向量
            
        Returns:
            包含各项评估指标的字典
        """
        # 获取预测结果
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y, y_pred_proba, multi_class='ovr')
        }
        
        self.logger.info(f"模型评估结果: {metrics}")
        
        return metrics 