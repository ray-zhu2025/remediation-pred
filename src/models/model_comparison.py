"""
模型评估报告脚本
用于对比不同介质下的不同模型效果
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from datetime import datetime
from src.models.base_models.model_factory import ModelFactory
from src.data.data_loader import DataLoader
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelComparison:
    """模型对比分析类"""
    
    def __init__(self, output_dir: str = 'output/model_comparison'):
        """
        初始化模型对比分析
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)
        
    def train_and_evaluate(self, model_type: str, X_train: np.ndarray, X_test: np.ndarray, 
                          y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        训练并评估模型
        
        Args:
            model_type: 模型类型
            X_train: 训练集特征
            X_test: 测试集特征
            y_train: 训练集标签
            y_test: 测试集标签
            
        Returns:
            评估指标字典
        """
        # 创建并训练模型
        model = ModelFactory.create_model(model_type, use_hyperopt=True)
        model.fit(X_train, y_train)
        
        # 获取预测结果
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # 计算多分类ROC AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except ValueError:
            self.logger.warning("无法计算ROC AUC分数，可能是由于某些类别没有样本")
            metrics['roc_auc'] = np.nan
        
        self.logger.info(f"{model_type} 模型评估结果: {metrics}")
        
        return metrics
        
    def compare_models(self, data_loader: DataLoader):
        """
        对比不同介质下的不同模型效果
        
        Args:
            data_loader: 数据加载器
        """
        # 获取数据
        soil_data = data_loader.load_soil_data()
        groundwater_data = data_loader.load_groundwater_data()
        
        # 模型类型列表
        model_types = ['decision_tree', 'random_forest', 'naive_bayes']
        
        # 存储结果
        results = []
        
        # 评估土壤模型
        for model_type in model_types:
            metrics = self.train_and_evaluate(
                model_type,
                soil_data['X_train'], soil_data['X_test'],
                soil_data['y_train'], soil_data['y_test']
            )
            results.append({
                'medium': 'soil',
                'model': model_type,
                **metrics
            })
            
        # 评估地下水模型
        for model_type in model_types:
            metrics = self.train_and_evaluate(
                model_type,
                groundwater_data['X_train'], groundwater_data['X_test'],
                groundwater_data['y_train'], groundwater_data['y_test']
            )
            results.append({
                'medium': 'groundwater',
                'model': model_type,
                **metrics
            })
            
        # 生成报告
        self._generate_report(results)
        
    def _generate_report(self, results: List[Dict]):
        """
        生成评估报告
        
        Args:
            results: 评估结果列表
        """
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df.to_csv(os.path.join(self.output_dir, f'model_comparison_{timestamp}.csv'), index=False)
        
        # 生成可视化图表
        self._plot_comparison(df)
        
        # 生成文本报告
        self._generate_text_report(df)
        
    def _plot_comparison(self, df: pd.DataFrame):
        """
        生成对比图表
        
        Args:
            df: 结果DataFrame
        """
        # 设置样式
        sns.set_style("whitegrid")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型性能对比', fontsize=16)
        
        # 准确率对比
        sns.barplot(data=df, x='model', y='accuracy', hue='medium', ax=axes[0,0])
        axes[0,0].set_title('准确率对比')
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45)
        
        # F1分数对比
        sns.barplot(data=df, x='model', y='f1', hue='medium', ax=axes[0,1])
        axes[0,1].set_title('F1分数对比')
        axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
        
        # ROC-AUC对比
        sns.barplot(data=df, x='model', y='roc_auc', hue='medium', ax=axes[1,0])
        axes[1,0].set_title('ROC-AUC对比')
        axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
        
        # 精确率对比
        sns.barplot(data=df, x='model', y='precision', hue='medium', ax=axes[1,1])
        axes[1,1].set_title('精确率对比')
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(self.output_dir, f'model_comparison_{timestamp}.png'))
        plt.close()
        
    def _generate_text_report(self, df: pd.DataFrame):
        """
        生成文本报告
        
        Args:
            df: 结果DataFrame
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.output_dir, f'model_comparison_report_{timestamp}.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('模型评估报告\n')
            f.write('=' * 50 + '\n\n')
            
            # 土壤模型评估
            f.write('土壤模型评估\n')
            f.write('-' * 30 + '\n')
            soil_df = df[df['medium'] == 'soil']
            for _, row in soil_df.iterrows():
                f.write(f"\n{row['model']}模型:\n")
                f.write(f"准确率: {row['accuracy']:.4f}\n")
                f.write(f"精确率: {row['precision']:.4f}\n")
                f.write(f"召回率: {row['recall']:.4f}\n")
                f.write(f"F1分数: {row['f1']:.4f}\n")
                f.write(f"ROC-AUC: {row['roc_auc']:.4f}\n")
                
            # 地下水模型评估
            f.write('\n地下水模型评估\n')
            f.write('-' * 30 + '\n')
            groundwater_df = df[df['medium'] == 'groundwater']
            for _, row in groundwater_df.iterrows():
                f.write(f"\n{row['model']}模型:\n")
                f.write(f"准确率: {row['accuracy']:.4f}\n")
                f.write(f"精确率: {row['precision']:.4f}\n")
                f.write(f"召回率: {row['recall']:.4f}\n")
                f.write(f"F1分数: {row['f1']:.4f}\n")
                f.write(f"ROC-AUC: {row['roc_auc']:.4f}\n")
                
            # 最佳模型分析
            f.write('\n最佳模型分析\n')
            f.write('-' * 30 + '\n')
            
            # 土壤最佳模型
            best_soil = soil_df.loc[soil_df['f1'].idxmax()]
            f.write(f"\n土壤最佳模型: {best_soil['model']}\n")
            f.write(f"F1分数: {best_soil['f1']:.4f}\n")
            
            # 地下水最佳模型
            best_groundwater = groundwater_df.loc[groundwater_df['f1'].idxmax()]
            f.write(f"\n地下水最佳模型: {best_groundwater['model']}\n")
            f.write(f"F1分数: {best_groundwater['f1']:.4f}\n")
            
            # 总体分析
            f.write('\n总体分析\n')
            f.write('-' * 30 + '\n')
            f.write(f"土壤模型平均F1分数: {soil_df['f1'].mean():.4f}\n")
            f.write(f"地下水模型平均F1分数: {groundwater_df['f1'].mean():.4f}\n")
            
if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建数据加载器
    data_loader = DataLoader()
    
    # 创建模型对比分析器
    comparison = ModelComparison()
    
    # 运行对比分析
    comparison.compare_models(data_loader) 