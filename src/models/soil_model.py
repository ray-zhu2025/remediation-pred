import logging
import pandas as pd
from typing import Dict, Any, List
from .base_model import BaseModel

class SoilModel(BaseModel):
    """土壤模型类"""
    
    def __init__(
        self,
        time_limit: int = 3600,
        presets: str = 'medium_quality',
        eval_metric: str = 'accuracy',
        n_jobs: int = -1,
        enable_explanation: bool = True,
        model_type: str = 'soil'
    ):
        super().__init__(
            time_limit=time_limit,
            presets=presets,
            eval_metric=eval_metric,
            n_jobs=n_jobs,
            enable_explanation=enable_explanation,
            model_type=model_type
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def explain_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """解释模型预测"""
        if not self.enable_explanation:
            self.logger.warning("模型解释功能未启用")
            return {}
            
        self.logger.info("开始生成模型解释...")
        
        # 获取模型详细信息
        model_details = self.get_model_details()
        
        # 绘制特征重要性图
        self.plot_feature_importance('output/plots/soil_feature_importance.png')
        
        # 记录模型详细信息
        self.logger.info(f"模型详细信息: {model_details}")
        
        return model_details 