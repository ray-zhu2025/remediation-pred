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
        self.model_path = self._get_path("models", "")
        
        # 配置日志记录器
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """配置日志记录器"""
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            # 使用主程序的日志处理器
            self.logger.handlers = logging.getLogger('__main__').handlers
        
    def train(self, X_train, y_train, X_test, y_test):
        """训练模型"""
        self.logger.info("开始训练模型...")
        
        # 记录系统资源信息
        self._log_system_info()
        
        # 设置模型保存路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join('models', self.model_type, f'v{self.version}', timestamp)
        os.makedirs(save_path, exist_ok=True)
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
        
        # 记录每个模型的性能指标
        self.logger.info("记录各个模型的性能指标...")
        leaderboard = self.predictor.leaderboard()
        self.logger.info("\n模型性能指标:")
        self.logger.info("-" * 80)
        self.logger.info(f"{'模型名称':<30} {'验证集得分':<10} {'训练时间(秒)':<15}")
        self.logger.info("-" * 80)
        
        for _, row in leaderboard.iterrows():
            model_name = row['model']
            score_val = row['score_val']
            fit_time = row['fit_time']
            self.logger.info(f"{model_name:<30} {score_val:<10.4f} {fit_time:<15.2f}")
        
        self.logger.info("-" * 80)
        
        # 记录模型集成权重
        self.logger.info("\n模型集成权重:")
        self.logger.info("-" * 80)
        best_model = self.predictor.model_best
        self.logger.info(f"最佳模型: {best_model}")
        self.logger.info("-" * 80)
        
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
        
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """评估模型性能"""
        # 确保输入数据格式正确
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        if not isinstance(y_test, pd.Series):
            y_test = pd.Series(y_test)
            
        # 预测
        y_pred = self.predictor.predict(X_test)
        
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'model_info': {
                'version': self.version,
                'type': self.model_type,
                'best_model': self.predictor.model_best,
                'all_models': self._get_model_performance()
            }
        }
        
        return metrics
        
    def get_model_info(self, detailed: bool = False) -> Dict:
        """
        获取模型信息
        
        Args:
            detailed: 是否获取详细信息
            
        Returns:
            模型信息字典
        """
        if self.predictor is None:
            raise ValueError("模型未训练")
            
        leaderboard = self.predictor.leaderboard()
        base_info = {
            'version': self.version,
            'type': self.model_type,
            'best_model': leaderboard.iloc[0]['model'],
            'validation_score': leaderboard.iloc[0]['score_val'],
            'fit_time': leaderboard.iloc[0]['fit_time']
        }
        
        if detailed:
            base_info['all_models'] = leaderboard[[
                'model', 'score_val', 'fit_time', 'pred_time_val'
            ]].to_dict('records')
            
            # 记录详细信息
            self.logger.info(f"\n{self.model_type.capitalize()}模型性能汇总报告:")
            self.logger.info("=" * 50)
            self.logger.info(f"模型总数: {len(leaderboard)}")
            self.logger.info(f"最佳模型: {base_info['best_model']}")
            self.logger.info(f"最佳验证集得分: {base_info['validation_score']:.4f}")
            self.logger.info(f"平均训练时间: {leaderboard['fit_time'].mean():.2f}秒")
            self.logger.info(f"平均预测时间: {leaderboard['pred_time_val'].mean():.2f}秒")
            self.logger.info("-" * 50)
            
            for model in base_info['all_models']:
                self.logger.info(f"模型: {model['model']}")
                self.logger.info(f"  验证集得分: {model['score_val']:.4f}")
                self.logger.info(f"  训练时间: {model['fit_time']:.2f}秒")
                self.logger.info(f"  预测时间: {model['pred_time_val']:.2f}秒")
                self.logger.info("-" * 50)
        
        return base_info
        
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
                        
    def _get_path(self, base_dir: str, suffix: str) -> str:
        """生成带时间戳的路径"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(
            base_dir,
            self.model_type,
            f"v{self.version}",
            f"{timestamp}{suffix}"
        )
        
    def _save_metrics(self, metrics):
        """保存评估指标"""
        # 创建保存目录
        save_dir = f'output/metrics/{self.model_type}'
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'{save_dir}/model_metrics_v{VersionConfig.get_version()}_{timestamp}.json'
        
        # 保存指标
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        self.logger.info(f"评估指标已保存到: {save_path}")
        
    def _get_model_performance(self) -> List[Dict[str, Any]]:
        """获取所有模型的性能指标"""
        if self.predictor is None:
            raise ValueError("模型未训练")
            
        leaderboard = self.predictor.leaderboard()
        model_info = []
        for _, row in leaderboard.iterrows():
            model_info.append({
                'model_name': row['model'],
                'validation_score': row['score_val'],
                'fit_time': row['fit_time']
            })
        return model_info 