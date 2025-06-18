import logging
import pandas as pd
from typing import Dict, Any
from .tabpfn_model import TabPFNModel
from src.utils.logging_utils import setup_logging

class DomainTabPFNModel(TabPFNModel):
    """通用土壤/地下水TabPFN模型类"""
    def __init__(
        self,
        domain_type: str = 'soil',
        version: str = "1.0.0",
        time_limit: int = 3600,
        eval_metric: str = 'accuracy',
        n_jobs: str = 'auto',
        enable_explanation: bool = True
    ):
        super().__init__(
            version=version,
            time_limit=time_limit,
            eval_metric=eval_metric,
            n_jobs=n_jobs,
            enable_explanation=enable_explanation,
            model_type=f'{domain_type}_tabpfn'
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
        self.logger.info(f"模型详细信息: {model_details}")
        return model_details 