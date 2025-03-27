import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config.version_config import VersionConfig

class BaseModel:
    """AutoGluon模型基类"""
    
    def __init__(
        self,
        model_type: str,  # 模型类型（soil/groundwater）
        time_limit: int = 3600,  # 训练时间限制（秒）
        presets: str = 'medium_quality',  # 预设质量级别
        eval_metric: str = 'accuracy',  # 评估指标
        n_jobs: int = -1,  # 并行任务数
        enable_explanation: bool = True  # 是否启用模型解释
    ):
        self.model_type = model_type
        self.time_limit = time_limit
        self.presets = presets
        self.eval_metric = eval_metric
        self.n_jobs = n_jobs
        self.enable_explanation = enable_explanation
        self.model = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """训练模型"""
        self.logger.info("开始训练模型...")
        
        # 将numpy数组转换为pandas DataFrame
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        train_data = pd.DataFrame(X, columns=feature_names)
        train_data['target'] = y
        
        # 获取模型保存路径
        model_path = VersionConfig.get_model_save_path(self.model_type)
        
        # 创建模型
        self.model = TabularPredictor(
            label='target',
            eval_metric=self.eval_metric,
            path=model_path
        )
        
        # 训练模型
        self.model.fit(
            train_data=train_data,
            time_limit=self.time_limit,
            presets=self.presets,
            verbosity=2
        )
        
        self.logger.info("模型训练完成")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        # 确保列名与训练数据相同
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        return self.model.predict(X_df).values
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        # 确保列名与训练数据相同
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        return self.model.predict_proba(X_df).values
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型性能"""
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        # 确保列名与训练数据相同
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        predictions = self.predict(X)
        leaderboard = self.model.leaderboard()
        
        metrics = {
            'accuracy': np.mean(predictions == y),
            'best_model': leaderboard.iloc[0]['model'],
            'best_score': leaderboard.iloc[0]['score_val']
        }
        
        self.logger.info(f"模型评估结果: {metrics}")
        return metrics
    
    def get_model_details(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        leaderboard = self.model.leaderboard()
        feature_importance = self.model.feature_importance()
        
        return {
            'leaderboard': leaderboard,
            'feature_importance': feature_importance,
            'model_info': self.model.info()
        }
    
    def plot_feature_importance(self, save_path: Optional[str] = None) -> None:
        """绘制特征重要性图"""
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        importance = self.model.feature_importance()
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance)
        plt.title('特征重要性')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def save_model(self, path: str) -> None:
        """保存模型"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        self.model.save(path)
        self.logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str) -> None:
        """加载模型"""
        self.model = TabularPredictor.load(path)
        self.logger.info(f"模型已从 {path} 加载") 