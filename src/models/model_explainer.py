"""
模型可解释性分析模块
使用SHAP值分析特征重要性
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
from pathlib import Path
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB

class ModelExplainer:
    """模型可解释性分析类"""
    
    def __init__(self, model, feature_names: List[str], output_dir: str):
        """
        初始化模型解释器
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称列表
            output_dir: 输出目录
        """
        self.model = model
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化SHAP解释器
        if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # 对于其他模型类型使用KernelExplainer
            # 创建一个简单的背景数据集，特征数量与模型期望的一致
            n_features = len(feature_names)
            if isinstance(model, ComplementNB):
                n_features = model.n_features_in_
            background_data = np.zeros((1, n_features))
            self.explainer = shap.KernelExplainer(
                self.model.predict if not hasattr(self.model, 'predict_proba') 
                else self.model.predict_proba,
                background_data
            )
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'SimSun']
        plt.rcParams['axes.unicode_minus'] = False
    
    def analyze_feature_importance(self, X):
        """分析特征重要性"""
        # 计算SHAP值
        shap_values = self.explainer.shap_values(X)
        
        # 处理多分类问题
        if isinstance(shap_values, list):
            # 对每个类别的SHAP值取绝对值并求平均
            feature_importance = np.mean([np.abs(shap_values[i]) for i in range(len(shap_values))], axis=0)
            # 如果feature_importance是二维的，取每个特征的平均值
            if len(feature_importance.shape) > 1:
                feature_importance = np.mean(feature_importance, axis=0)
        else:
            # 二分类问题
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            # 如果feature_importance是二维的，取每个特征的平均值
            if len(feature_importance.shape) > 1:
                feature_importance = np.mean(feature_importance, axis=0)
        
        # 确保特征重要性的长度与特征名称匹配
        if len(feature_importance) > len(self.feature_names):
            feature_importance = feature_importance[:len(self.feature_names)]
        elif len(feature_importance) < len(self.feature_names):
            self.feature_names = self.feature_names[:len(feature_importance)]
            
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        })
        
        # 按重要性排序
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # 保存特征重要性
        importance_df.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
        
        # 绘制特征重要性条形图
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importance)), importance_df['importance'])
        plt.xticks(range(len(feature_importance)), importance_df['feature'], rotation=45)
        plt.title('特征重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
        plt.close()
        
        # 绘制SHAP摘要图
        plt.figure(figsize=(10, 6))
        if isinstance(shap_values, list):
            # 多分类问题使用第一个类别的SHAP值
            shap.summary_plot(shap_values[0], X, feature_names=self.feature_names, show=False)
        else:
            # 二分类问题
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'shap_summary.png'))
        plt.close()
        
        return importance_df
    
    def analyze_feature_effects(self, X):
        """分析特征效应"""
        # 计算SHAP值
        if isinstance(self.model, (DecisionTreeClassifier, RandomForestClassifier)):
            explainer = shap.TreeExplainer(self.model)
        else:
            # 对于其他模型类型，使用KernelExplainer
            explainer = shap.KernelExplainer(self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict, X)
        
        shap_values = explainer.shap_values(X)
        
        # 如果是多分类问题，计算每个类别的平均SHAP值
        if isinstance(shap_values, list):
            shap_values = np.mean(np.abs(shap_values), axis=0)
        
        # 生成SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names[:X.shape[1]], show=False)
        plt.title('特征重要性分析')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'shap_summary_plot.png'))
        plt.close()
    
    def analyze_interactions(self, X: np.ndarray) -> None:
        """
        分析特征间的交互作用
        
        Args:
            X: 特征数据
        """
        # 只对支持 TreeExplainer 的模型进行交互分析
        if not isinstance(self.model, (DecisionTreeClassifier, RandomForestClassifier)):
            return
            
        # 计算SHAP值
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        # 如果是多分类问题，取所有类别的平均SHAP值
        if isinstance(shap_values, list):
            shap_values = np.mean(shap_values, axis=0)
        
        # 计算特征交互矩阵
        interaction_matrix = np.zeros((len(self.feature_names), len(self.feature_names)))
        
        for i in range(len(self.feature_names)):
            for j in range(i + 1, len(self.feature_names)):
                interaction = np.abs(shap_values[:, i] * shap_values[:, j]).mean()
                interaction_matrix[i, j] = interaction
                interaction_matrix[j, i] = interaction
        
        # 绘制交互热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(interaction_matrix, 
                   xticklabels=self.feature_names,
                   yticklabels=self.feature_names,
                   cmap='YlOrRd')
        plt.title('特征交互热力图')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_interactions.png')
        plt.close() 