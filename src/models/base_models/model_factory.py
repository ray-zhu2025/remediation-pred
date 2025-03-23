"""
模型工厂类
用于创建不同类型的模型实例
"""

from typing import Dict, Type, List
from .base_model import BaseModel
from .decision_tree import DecisionTreeBaseModel
from .random_forest import RandomForestBaseModel
from .naive_bayes import NaiveBayesBaseModel

class ModelFactory:
    """模型工厂类"""
    
    # 模型类型映射
    _model_classes: Dict[str, Type[BaseModel]] = {
        'decision_tree': DecisionTreeBaseModel,
        'random_forest': RandomForestBaseModel,
        'naive_bayes': NaiveBayesBaseModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, use_hyperopt: bool = True) -> BaseModel:
        """
        创建单个模型实例
        
        Args:
            model_type: 模型类型
            use_hyperopt: 是否使用超参数优化
            
        Returns:
            模型实例
            
        Raises:
            ValueError: 如果模型类型不存在
        """
        if model_type not in cls._model_classes:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        model_class = cls._model_classes[model_type]
        return model_class(use_hyperopt=use_hyperopt)
    
    @classmethod
    def create_models(cls, use_hyperopt: bool = True) -> Dict[str, BaseModel]:
        """
        创建所有支持的模型实例
        
        Args:
            use_hyperopt: 是否使用超参数优化
            
        Returns:
            模型实例字典，键为模型类型，值为模型实例
        """
        models = {}
        for model_type in cls._model_classes:
            models[model_type] = cls.create_model(model_type, use_hyperopt)
        return models 