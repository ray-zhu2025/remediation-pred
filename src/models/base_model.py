import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from src.config.version_config import VersionConfig

class BaseModel:
    """基础模型类"""
    
    def __init__(
        self,
        version: str = "1.0.0",
        time_limit: int = 3600,
        presets: str = 'medium_quality',
        eval_metric: str = 'accuracy',
        n_jobs: str = 'auto',
        enable_explanation: bool = True,
        model_type: str = 'base'
    ):
        """
        初始化基础模型
        
        Args:
            version: 模型版本号
            time_limit: 训练时间限制(秒)
            presets: 预设配置
            eval_metric: 评估指标
            n_jobs: 并行任务数
            enable_explanation: 是否启用模型解释
            model_type: 模型类型
        """
        self.version = version
        self.time_limit = time_limit
        self.presets = presets
        self.eval_metric = eval_metric
        self.n_jobs = n_jobs
        self.enable_explanation = enable_explanation
        self.model_type = model_type
        self.predictor = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
        """
        self.logger.info("开始训练模型...")
        
        # 获取系统信息
        self._log_system_info()
        
        # 创建保存路径
        save_path = self._get_save_path()
        self.logger.info(f"模型将保存到: {save_path}")
        
        # 转换数据格式
        train_data = pd.DataFrame(X_train)
        train_data['label'] = y_train
        
        # 初始化预测器
        self.predictor = TabularPredictor(
            label='label',
            path=save_path,
            eval_metric=self.eval_metric,
            problem_type='multiclass'
        )
        
        # 训练模型
        self.predictor.fit(
            train_data,
            time_limit=self.time_limit,
            presets=self.presets,
            num_cpus='auto'
        )
        
        self.logger.info("模型训练完成")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果
        """
        if self.predictor is None:
            raise ValueError("模型未训练")
            
        # 转换数据格式
        test_data = pd.DataFrame(X)
        
        # 预测
        predictions = self.predictor.predict(test_data)
        return predictions.values
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            评估指标
        """
        self.logger.info("开始评估模型...")
        
        # 预测
        y_pred = self.predict(X_test)
        
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # 保存评估指标
        metrics_path = self._get_metrics_path()
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        self.logger.info(f"准确率: {metrics['accuracy']:.4f}")
        self.logger.info(f"评估指标已保存到: {metrics_path}")
        
        return metrics
        
    def get_model_details(self) -> Dict[str, Any]:
        """
        获取模型详细信息
        
        Returns:
            模型信息字典
        """
        if self.predictor is None:
            raise ValueError("模型未训练")
            
        # 获取模型信息
        leaderboard = self.predictor.leaderboard()
        model_info = {
            'version': self.version,
            'type': self.model_type,
            'best_model': leaderboard.iloc[0]['model'],
            'validation_score': leaderboard.iloc[0]['score_val'],
            'fit_time': leaderboard.iloc[0]['fit_time'],
            'pred_time': leaderboard.iloc[0]['pred_time_val']
        }
        
        return model_info
        
    def plot_feature_importance(self, save_path: str):
        """
        绘制特征重要性图
        
        Args:
            save_path: 保存路径
        """
        if self.predictor is None:
            raise ValueError("模型未训练")
            
        # 获取特征重要性
        feature_importance = self.predictor.feature_importance()
        
        # 绘制图形
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title(f'{self.model_type.capitalize()} Model Feature Importance')
        plt.tight_layout()
        
        # 保存图形
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"特征重要性图已保存到: {save_path}")
        
    def _log_system_info(self):
        """记录系统信息"""
        import psutil
        
        # 获取内存信息
        memory = psutil.virtual_memory()
        self.logger.info(f"系统内存: 总计={memory.total/1024/1024/1024:.1f}GB, "
                        f"可用={memory.available/1024/1024/1024:.1f}GB, "
                        f"使用率={memory.percent}%")
                        
        # 获取磁盘信息
        disk = psutil.disk_usage('/')
        self.logger.info(f"磁盘空间: 总计={disk.total/1024/1024/1024:.1f}GB, "
                        f"可用={disk.free/1024/1024/1024:.1f}GB, "
                        f"使用率={disk.percent}%")
                        
    def _get_save_path(self) -> str:
        """获取模型保存路径"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(
            "models",
            self.model_type,
            f"v{self.version}",
            timestamp
        )
        
    def _get_metrics_path(self) -> str:
        """获取评估指标保存路径"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(
            "output",
            "metrics",
            f"{self.model_type}_v{self.version}_{timestamp}.json"
        ) 