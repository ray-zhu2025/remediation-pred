"""
模型比较分析脚本
用于读取和比较不同模型的评估结果
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import logging

class ModelComparison:
    """模型比较分析类"""
    
    def __init__(self, eval_dir: str = 'output/evaluation'):
        """
        初始化模型比较分析器
        
        Args:
            eval_dir: 评估结果目录
        """
        self.eval_dir = eval_dir
        self.logger = logging.getLogger(__name__)
        
    def load_evaluation_results(self) -> pd.DataFrame:
        """
        加载评估结果
        
        Returns:
            包含所有模型评估结果的DataFrame
        """
        try:
            results_path = os.path.join(self.eval_dir, 'evaluation_results.csv')
            if not os.path.exists(results_path):
                raise FileNotFoundError(f"评估结果文件不存在: {results_path}")
                
            results = pd.read_csv(results_path)
            self.logger.info(f"成功加载评估结果，共{len(results)}个模型")
            return results
        except Exception as e:
            self.logger.error(f"加载评估结果时发生错误: {str(e)}")
            raise
            
    def compare_models(self, results: pd.DataFrame) -> Dict:
        """
        比较不同模型的性能
        
        Args:
            results: 评估结果DataFrame
            
        Returns:
            包含比较结果的字典
        """
        comparison = {}
        
        # 计算每个指标的最佳模型
        metrics = ['准确率', '精确率', '召回率', 'F1分数', 'ROC-AUC分数']
        for metric in metrics:
            best_model = results.loc[results[metric].idxmax()]
            comparison[f'最佳{metric}'] = {
                '模型': best_model['模型'],
                '得分': best_model[metric]
            }
            
        # 计算综合得分（所有指标的平均值）
        results['综合得分'] = results[metrics].mean(axis=1)
        best_overall = results.loc[results['综合得分'].idxmax()]
        comparison['最佳综合性能'] = {
            '模型': best_overall['模型'],
            '得分': best_overall['综合得分']
        }
        
        return comparison
        
    def plot_comparison(self, results: pd.DataFrame, output_dir: str):
        """
        绘制模型比较图表
        
        Args:
            results: 评估结果DataFrame
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 绘制性能指标对比条形图
        plt.figure(figsize=(12, 6))
        metrics = ['准确率', '精确率', '召回率', 'F1分数', 'ROC-AUC分数']
        results_melted = pd.melt(results, 
                                id_vars=['模型'],
                                value_vars=metrics,
                                var_name='指标',
                                value_name='得分')
        
        sns.barplot(data=results_melted, x='模型', y='得分', hue='指标')
        plt.title('模型性能指标对比')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_metrics_comparison.png'))
        plt.close()
        
        # 2. 绘制综合得分雷达图
        plt.figure(figsize=(10, 10))
        models = results['模型'].tolist()
        metrics_values = results[metrics].values
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        
        ax = plt.subplot(111, projection='polar')
        for i, model in enumerate(models):
            values = metrics_values[i]
            values = np.concatenate((values, [values[0]]))  # 闭合雷达图
            angles_plot = np.concatenate((angles, [angles[0]]))  # 闭合雷达图
            ax.plot(angles_plot, values, 'o-', linewidth=2, label=model)
            ax.fill(angles_plot, values, alpha=0.25)
        
        ax.set_xticks(angles)
        ax.set_xticklabels(metrics)
        plt.title('模型性能雷达图')
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_radar_chart.png'))
        plt.close()
        
    def generate_report(self, comparison_results: Dict, output_path: str):
        """
        生成比较报告
        
        Args:
            comparison_results: 比较结果字典
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 模型比较报告\n\n")
            
            # 写入各项指标的最佳模型
            f.write("## 各指标最佳模型\n\n")
            metrics = ['准确率', '精确率', '召回率', 'F1分数', 'ROC-AUC分数']
            for metric in metrics:
                result = comparison_results[f'最佳{metric}']
                f.write(f"### {metric}\n")
                f.write(f"- 最佳模型: {result['模型']}\n")
                f.write(f"- 得分: {result['得分']:.4f}\n\n")
            
            # 写入综合性能最佳模型
            f.write("## 综合性能最佳模型\n\n")
            best_overall = comparison_results['最佳综合性能']
            f.write(f"- 模型: {best_overall['模型']}\n")
            f.write(f"- 综合得分: {best_overall['得分']:.4f}\n")
            
def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # 创建模型比较分析器
        comparator = ModelComparison()
        
        # 加载评估结果
        results = comparator.load_evaluation_results()
        
        # 比较模型性能
        comparison_results = comparator.compare_models(results)
        
        # 创建输出目录
        output_dir = 'output/model_comparison'
        os.makedirs(output_dir, exist_ok=True)
        
        # 绘制比较图表
        comparator.plot_comparison(results, output_dir)
        
        # 生成比较报告
        report_path = os.path.join(output_dir, 'comparison_report.md')
        comparator.generate_report(comparison_results, report_path)
        
        logger.info("模型比较分析完成")
        
    except Exception as e:
        logger.error(f"模型比较分析过程中发生错误: {str(e)}")
        raise

if __name__ == '__main__':
    main() 