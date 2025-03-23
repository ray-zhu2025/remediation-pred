import pandas as pd
import json
import os
import numpy as np
from sklearn.metrics import hamming_loss, jaccard_score, zero_one_loss

def load_evaluation_results(media_type='soil'):
    # 加载模型比较结果
    models_comparison = pd.read_csv(f'output/{media_type}/evaluation/models_comparison.csv')
    
    # 加载各个模型的详细评估结果
    models = ['RandomForestClassifier', 'ComplementNB', 'DecisionTreeClassifier']
    results = {}
    
    for model in models:
        with open(f'output/{media_type}/evaluation/evaluation_results_{model}.json', 'r') as f:
            results[model] = json.load(f)
    
    return models_comparison, results

def calculate_multiclass_metrics(y_true, y_pred, labels):
    """计算多分类评估指标"""
    metrics = {
        'hamming_loss': hamming_loss(y_true, y_pred),
        'jaccard_score': jaccard_score(y_true, y_pred, average='weighted'),
        'zero_one_loss': zero_one_loss(y_true, y_pred)
    }
    return metrics

def analyze_model_performance(models_comparison, results):
    print("\n=== 模型性能对比 ===")
    print(models_comparison.to_string(index=False))
    
    for model, result in results.items():
        print(f"\n=== {model} 详细评估结果 ===")
        print(f"总体准确率: {result['classification_report']['accuracy']:.4f}")
        
        print("\n各类修复技术预测效果:")
        for tech, metrics in result['classification_report'].items():
            if tech != 'accuracy' and tech != 'macro avg':
                print(f"\n技术 {tech}:")
                print(f"  精确率: {metrics['precision']:.4f}")
                print(f"  召回率: {metrics['recall']:.4f}")
                print(f"  F1分数: {metrics['f1-score']:.4f}")
                print(f"  样本量: {metrics['support']}")

def main():
    # 分析土壤模型
    print("\n=== 土壤模型评估结果 ===")
    soil_comparison, soil_results = load_evaluation_results('soil')
    analyze_model_performance(soil_comparison, soil_results)
    
    # 分析地下水模型
    print("\n=== 地下水模型评估结果 ===")
    groundwater_comparison, groundwater_results = load_evaluation_results('groundwater')
    analyze_model_performance(groundwater_comparison, groundwater_results)

if __name__ == "__main__":
    main() 