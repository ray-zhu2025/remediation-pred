"""AutoGluon TabPFN模型实现"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from autogluon.core.models import AbstractModel
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from tabpfn import TabPFNClassifier
import time
import psutil
import os

logger = logging.getLogger(__name__)

class TabPFNModel(AbstractModel):
    """
    AutoGluon中的TabPFN模型实现
    """
    
    def __init__(self, **kwargs):
        """初始化TabPFN模型"""
        super().__init__(**kwargs)
        self._device = kwargs.get('device', 'cpu')
        self._n_ensemble_configurations = kwargs.get('N_ensemble_configurations', 32)
        self._batch_size_inference = kwargs.get('batch_size_inference', 1024)
        self._base_path = kwargs.get('base_path', None)
        self._c = kwargs.get('c', 1.0)
        self._seed = kwargs.get('seed', 42)
        self._max_num_features = kwargs.get('max_num_features', 100)
        self._eval_positions = kwargs.get('eval_positions', None)
        self._verbose = kwargs.get('verbose', False)
        
        self.model = None
        
    def _get_model_type(self):
        """获取模型类型"""
        return 'TabPFN'
        
    def _validate_fit_memory_usage(self, X, **kwargs):
        """验证内存使用"""
        max_memory_usage_ratio = self.params.get('max_memory_usage_ratio', 0.9)
        n_rows = X.shape[0]
        n_cols = X.shape[1]
        
        # 估算内存使用
        approx_mem_usage_bytes = n_rows * n_cols * 8  # 假设每个浮点数占8字节
        available_mem = psutil.virtual_memory().available
        
        if approx_mem_usage_bytes > available_mem * max_memory_usage_ratio:
            raise MemoryError(
                f'预估内存使用({approx_mem_usage_bytes / 1e9:.2f}GB) 超过可用内存'
                f'({available_mem * max_memory_usage_ratio / 1e9:.2f}GB)'
            )
            
    def _validate_data(self, X, y=None):
        """验证数据是否满足TabPFN要求"""
        if X.shape[0] > 10000:
            raise ValueError("TabPFN只支持少于10,000行的数据")
            
        if X.shape[1] > 100:
            raise ValueError("TabPFN只支持最多100个特征")
            
        if y is not None:
            n_classes = len(np.unique(y))
            if n_classes > 10:
                raise ValueError("TabPFN只支持最多10个类别的分类任务")
                
    def _fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            X_val: pd.DataFrame = None,
            y_val: pd.Series = None,
            time_limit: int = None,
            **kwargs):
        """
        训练TabPFN模型
        
        Args:
            X: 训练数据特征
            y: 训练数据标签
            X_val: 验证数据特征
            y_val: 验证数据标签
            time_limit: 时间限制(秒)
        """
        start_time = time.time()
        
        # 验证数据
        self._validate_data(X, y)
        self._validate_fit_memory_usage(X)
        
        # 检查问题类型
        if self.problem_type not in [BINARY, MULTICLASS]:
            raise ValueError(f"TabPFN不支持{self.problem_type}问题类型")
            
        # 初始化模型
        self.model = TabPFNClassifier(
            device=self._device,
            N_ensemble_configurations=self._n_ensemble_configurations,
            batch_size_inference=self._batch_size_inference,
            base_path=self._base_path,
            c=self._c,
            seed=self._seed,
            max_num_features=self._max_num_features,
            eval_positions=self._eval_positions,
            verbose=self._verbose
        )
        
        # 训练模型
        self.model.fit(X.values, y.values)
        
        # 检查时间限制
        if time_limit is not None:
            time_elapsed = time.time() - start_time
            if time_elapsed > time_limit:
                raise TimeLimitExceeded(time_limit=time_limit, time_elapsed=time_elapsed)
                
        return self
        
    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """预测概率"""
        self._validate_data(X)
        
        if self.model is None:
            raise ValueError("模型未训练")
            
        return self.model.predict_proba(X.values)
        
    def _predict_proba_oof(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Out-of-fold预测概率"""
        return self._predict_proba(X, **kwargs)
        
    def _get_default_auxiliary_params(self) -> dict:
        """获取默认辅助参数"""
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_stacker=True,  # 是否可以作为stacker模型
            valid_base=True,  # 是否可以作为base模型
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
        
    @classmethod
    def _get_default_params(cls) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            'device': 'cpu',
            'N_ensemble_configurations': 32,
            'batch_size_inference': 1024,
            'base_path': None,
            'c': 1.0,
            'seed': 42,
            'max_num_features': 100,
            'eval_positions': None,
            'verbose': False,
            'max_memory_usage_ratio': 0.9,
        }
        
    def _get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        params = {
            'device': self._device,
            'N_ensemble_configurations': self._n_ensemble_configurations,
            'batch_size_inference': self._batch_size_inference,
            'base_path': self._base_path,
            'c': self._c,
            'seed': self._seed,
            'max_num_features': self._max_num_features,
            'eval_positions': self._eval_positions,
            'verbose': self._verbose,
        }
        return params
        
    def _more_tags(self) -> Dict[str, Any]:
        """获取更多标签"""
        return {
            'valid_stacker': True,  # 是否可以作为stacker模型
            'valid_base': True,  # 是否可以作为base模型
            'can_refit_full': False,  # 是否支持完全重新训练
        }
        
    @staticmethod
    def can_fit() -> bool:
        """是否可以训练"""
        return True
        
    @staticmethod
    def supported_problem_types() -> list:
        """支持的问题类型"""
        return [BINARY, MULTICLASS]
        
    def get_memory_size(self) -> float:
        """获取模型内存大小（MB）"""
        return 0  # TabPFN是预训练模型，不需要额外内存
        
    def reduce_memory_size(self, remove_fit: bool = False, requires_save: bool = True, **kwargs):
        """减少内存使用"""
        pass  # TabPFN是预训练模型，不需要减少内存
        
    def delete_from_disk(self, silent=True):
        """从磁盘删除模型"""
        pass  # TabPFN是预训练模型，不需要删除 