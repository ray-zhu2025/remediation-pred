import logging
import pandas as pd
import os
import json
from typing import Dict, Any
from datetime import datetime
from .base_model import BaseModel
from src.utils.logging_utils import setup_logging

class DomainModel(BaseModel):
    """通用土壤/地下水模型类"""
    def __init__(
        self,
        domain_type: str = 'soil',
        version: str = "1.0.0",
        time_limit: int = 3600,
        presets: str = 'medium_quality',
        eval_metric: str = 'accuracy',
        n_jobs: str = 'auto',
        enable_explanation: bool = True
    ):
        super().__init__(
            version=version,
            time_limit=time_limit,
            presets=presets,
            eval_metric=eval_metric,
            n_jobs=n_jobs,
            enable_explanation=enable_explanation,
            model_type=domain_type
        )
        self.domain_type = domain_type
        self.logger = setup_logging(self.__class__.__name__)

    def explain_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """解释模型预测"""
        if not self.enable_explanation:
            self.logger.warning("模型解释功能未启用")
            return {}
        self.logger.info("开始生成模型解释...")
        model_details = self.get_model_info(detailed=True)
        # 绘制特征重要性图
        plot_path = f'output/plots/{self.domain_type}_feature_importance.png'
        self.plot_feature_importance(plot_path)
        self.logger.info(f"模型详细信息: {model_details}")
        return model_details

    def save(self, path: str):
        """保存模型配置信息"""
        os.makedirs(path, exist_ok=True)
        
        # 保存模型配置
        config = {
            'version': self.version,
            'domain_type': self.domain_type,
            'model_type': self.model_type,
            'time_limit': self.time_limit,
            'presets': self.presets,
            'eval_metric': self.eval_metric,
            'n_jobs': self.n_jobs,
            'enable_explanation': self.enable_explanation,
            'save_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        config_path = os.path.join(path, 'model_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"模型配置已保存到: {config_path}") 