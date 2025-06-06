import logging
import pandas as pd
from typing import Dict, Any, List
from .base_model import BaseModel

class GroundwaterModel(BaseModel):
    """地下水模型类"""
    
    def __init__(
        self,
        version: str = "1.0.0",
        time_limit: int = 3600,
        presets: str = 'medium_quality',
        eval_metric: str = 'accuracy',
        n_jobs: str = 'auto',
        enable_explanation: bool = True,
        model_type: str = 'groundwater'
    ):
        super().__init__(
            version=version,
            time_limit=time_limit,
            presets=presets,
            eval_metric=eval_metric,
            n_jobs=n_jobs,
            enable_explanation=enable_explanation,
            model_type=model_type
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_type = "groundwater"
    
    def explain_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """解释模型预测"""
        if not self.enable_explanation:
            self.logger.warning("模型解释功能未启用")
            return {}
            
        self.logger.info("开始生成模型解释...")
        
        # 获取模型详细信息
        model_details = self.get_model_details()
        
        # 绘制特征重要性图
        self.plot_feature_importance('output/plots/groundwater_feature_importance.png')
        
        # 记录模型详细信息
        self.logger.info(f"模型详细信息: {model_details}")
        
        return model_details 

    def train(self, X_train, y_train, X_test, y_test):
        """训练地下水模型"""
        super().train(X_train, y_train, X_test, y_test) 