"""TabPFN模型基类"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from tabpfn import TabPFNClassifier
from .base_model import BaseModel
from src.config.version_config import VersionConfig
import os
import json
import shap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TabPFNModel(BaseModel):
    """TabPFN模型类"""
    
    def __init__(
        self,
        version: str = "1.0.0",
        time_limit: int = 3600,
        eval_metric: str = 'accuracy',
        n_jobs: str = 'auto',
        enable_explanation: bool = True,
        model_type: str = 'tabpfn'
    ):
        """
        初始化TabPFN模型
        
        Args:
            version: 模型版本号
            time_limit: 训练时间限制(秒)
            eval_metric: 评估指标
            n_jobs: 并行任务数
            enable_explanation: 是否启用模型解释
            model_type: 模型类型
        """
        super().__init__(
            version=version,
            time_limit=time_limit,
            eval_metric=eval_metric,
            n_jobs=n_jobs,
            enable_explanation=enable_explanation,
            model_type=model_type
        )
        
        # 从配置中获取TabPFN参数
        tabpfn_params = VersionConfig.MODEL_PARAMS.get('TabPFN', {})
        self.device = tabpfn_params.get('device', 'cpu')
        self.N_ensemble_configurations = tabpfn_params.get('N_ensemble_configurations', 32)
        self.batch_size_inference = tabpfn_params.get('batch_size_inference', 1024)
        self.base_path = tabpfn_params.get('base_path', None)
        self.c = tabpfn_params.get('c', 1.0)
        self.seed = tabpfn_params.get('seed', 42)
        self.max_num_features = tabpfn_params.get('max_num_features', 1000)
        self.eval_positions = tabpfn_params.get('eval_positions', None)
        self.verbose = tabpfn_params.get('verbose', True)
        
        self.predictor = None
        self.feature_names = None
        
    def _validate_data(self, X, y=None):
        """验证数据是否满足TabPFN的要求"""
        if X.shape[0] > 10000:
            raise ValueError("TabPFN只支持少于10,000行的数据")
            
        if X.shape[1] > 100:
            raise ValueError("TabPFN只支持最多100个特征")
            
        if y is not None:
            n_classes = len(np.unique(y))
            if n_classes > 10:
                raise ValueError("TabPFN只支持最多10个类别的分类任务")
        
    def train(self, X_train, y_train, X_test, y_test):
        """
        训练TabPFN模型
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征
            y_test: 测试集标签
        """
        self.logger.info("开始训练TabPFN模型...")
        
        # 验证数据
        self._validate_data(X_train, y_train)
        
        # 记录系统资源信息
        self._log_system_info()
        
        # 保存特征名称
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
            
        # 初始化模型
        self.predictor = TabPFNClassifier(
            device=self.device,
            N_ensemble_configurations=self.N_ensemble_configurations,
            batch_size_inference=self.batch_size_inference,
            base_path=self.base_path,
            c=self.c,
            seed=self.seed,
            max_num_features=self.max_num_features,
            eval_positions=self.eval_positions,
            verbose=self.verbose
        )
        
        # 训练模型
        start_time = pd.Timestamp.now()
        self.predictor.fit(X_train, y_train)
        training_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        # 记录训练信息
        self.logger.info(f"模型训练完成，耗时: {training_time:.2f}秒")
        self.logger.info(f"模型配置: device={self.device}, N_ensemble_configurations={self.N_ensemble_configurations}")
        
        # 评估模型
        train_metrics = self.evaluate(X_train, y_train)
        test_metrics = self.evaluate(X_test, y_test)
        
        self.logger.info("\n训练集评估结果:")
        self.logger.info(f"准确率: {train_metrics['accuracy']:.4f}")
        self.logger.info(f"分类报告:\n{train_metrics['classification_report']}")
        
        self.logger.info("\n测试集评估结果:")
        self.logger.info(f"准确率: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"分类报告:\n{test_metrics['classification_report']}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.predictor is None:
            raise ValueError("模型未训练")
            
        # 验证数据
        self._validate_data(X)
            
        # 转换数据格式
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.predictor.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self.predictor is None:
            raise ValueError("模型未训练")
            
        # 验证数据
        self._validate_data(X)
            
        # 转换数据格式
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.predictor.predict_proba(X)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """评估模型性能"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'classification_report': classification_report(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'probabilities': y_proba
        }
        
        return metrics
        
    def compute_feature_importance(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """计算特征重要性"""
        if not self.enable_explanation:
            self.logger.warning("模型解释功能未启用")
            return {}
            
        self.logger.info("开始计算特征重要性...")
        
        # 使用SHAP计算特征重要性
        explainer = shap.KernelExplainer(self.predict_proba, X)
        shap_values = explainer.shap_values(X)
        
        # 计算每个特征的重要性
        feature_importance = {}
        for i, feature in enumerate(self.feature_names):
            importance = np.abs(shap_values[i]).mean()
            feature_importance[feature] = float(importance)
            
        return feature_importance
        
    def save(self, path: str):
        """保存模型（TabPFN模型不需要保存，因为它是预训练模型）"""
        self.logger.info("TabPFN是预训练模型，不需要保存模型文件")
        
        # 保存模型配置和特征信息
        config = {
            'version': self.version,
            'model_type': self.model_type,
            'device': self.device,
            'N_ensemble_configurations': self.N_ensemble_configurations,
            'batch_size_inference': self.batch_size_inference,
            'base_path': self.base_path,
            'c': self.c,
            'seed': self.seed,
            'max_num_features': self.max_num_features,
            'eval_positions': self.eval_positions,
            'verbose': self.verbose,
            'feature_names': self.feature_names
        }
        
        os.makedirs(path, exist_ok=True)
        config_path = os.path.join(path, 'model_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"模型配置已保存到: {config_path}")