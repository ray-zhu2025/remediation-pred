"""
数据分析模块
分析训练数据集中修复技术的分布并生成可视化图表
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_tech_labels(config_path: str) -> dict:
    """
    加载修复技术标签映射
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 技术编号到名称的映射字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return {i+1: tech for i, tech in enumerate(config['technologies'])}

def analyze_remediation_tech(data_file: str, output_dir: str, dataset_type: str) -> None:
    """
    分析修复技术分布并生成直方图
    
    Args:
        data_file: 训练数据文件路径
        output_dir: 输出目录路径
        dataset_type: 数据集类型 ('soil' 或 'groundwater')
    """
    # 创建特定数据集的输出目录
    dataset_output_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    try:
        # 读取CSV文件
        print(f"[INFO] 读取{dataset_type}数据文件: {data_file}")
        df = pd.read_csv(data_file, encoding='utf-8')
        
        # 检查是否存在"修复技术"列
        if "修复技术" not in df.columns:
            print("[ERROR] 数据中不存在'修复技术'列！")
            return
            
        # 加载技术标签映射
        config_path = os.path.join(os.path.dirname(os.path.dirname(output_dir)), 
                                 'src', 'config', dataset_type, 'parameters.json')
        tech_labels = load_tech_labels(config_path)
        
        # 确保修复技术列的值是整数
        df["修复技术"] = df["修复技术"].astype(int)
        
        # 统计数值出现次数和计算百分比
        value_counts = df["修复技术"].value_counts().sort_index()
        total = value_counts.sum()
        percentages = (value_counts / total * 100).round(2)
        
        # 创建有序的技术名称映射
        ordered_tech_names = pd.Series({k: tech_labels[k] for k in sorted(tech_labels.keys())})
        
        # 创建最终的统计数据，保持顺序一致
        final_counts = pd.Series(0, index=ordered_tech_names)
        final_percentages = pd.Series(0.0, index=ordered_tech_names)
        
        # 填充实际的统计数据
        for tech_id, count in value_counts.items():
            tech_name = tech_labels[tech_id]
            final_counts[tech_name] = count
            final_percentages[tech_name] = percentages[tech_id]
        
        # 创建图形
        plt.figure(figsize=(15, 8))  # 加大图形尺寸以适应长标签
        
        # 设置Seaborn样式
        sns.set_theme(style="white")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 使用单一颜色 - 选择深蓝色
        color = '#1f77b4'
        
        # 绘制直方图
        ax = sns.barplot(x=final_counts.index, y=final_counts.values, color=color)
        
        # 设置标题和标签
        title = '土壤修复技术分布' if dataset_type == 'soil' else '地下水修复技术分布'
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('修复技术', fontsize=12)
        plt.ylabel('数量', fontsize=12)
        
        # 旋转x轴标签以防止重叠
        plt.xticks(rotation=45, ha='right')
        
        # 添加数值和百分比标签
        for i, (v, p) in enumerate(zip(final_counts.values, final_percentages.values)):
            if v > 0:  # 只为非零值添加标签
                # 获取柱子的高度
                height = ax.patches[i].get_height()
                # 在柱子顶部添加标签
                ax.text(i, height, f'{int(v)}\n({p:.2f}%)', 
                       ha='center', va='bottom',
                       fontsize=10)
        
        # 添加网格线
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # 移除顶部和右侧边框
        sns.despine()
        
        # 调整布局和边距
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(dataset_output_dir, '修复技术_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n[INFO] 已生成{dataset_type}修复技术分布图: {output_path}")
        
        # 打印数值统计
        print(f"\n{dataset_type}修复技术分布统计:")
        stats_df = pd.DataFrame({
            '数量': final_counts,
            '百分比(%)': final_percentages
        })
        print(stats_df)
        print(f"\n总计: {total}")
        print(f"唯一值数量: {len(value_counts)}")
        
        # 保存统计结果到CSV
        stats_path = os.path.join(dataset_output_dir, '修复技术_statistics.csv')
        stats_df.to_csv(stats_path, encoding='utf-8-sig')
        print(f"\n统计结果已保存至: {stats_path}")
        
    except Exception as e:
        print(f"[ERROR] 生成{dataset_type}修复技术分布图失败: {str(e)}")
        raise e

def main():
    """主函数"""
    # 获取项目根目录
    project_root = str(Path(__file__).parent.parent)
    
    # 设置基础目录
    data_dir = os.path.join(project_root, "data", "training")
    output_dir = os.path.join(project_root, "output", "analysis")
    
    # 分析土壤数据
    soil_data_file = os.path.join(data_dir, "soil_training.csv")
    if os.path.exists(soil_data_file):
        analyze_remediation_tech(soil_data_file, output_dir, "soil")
    else:
        print(f"[WARNING] 土壤训练数据文件不存在: {soil_data_file}")
    
    # 分析地下水数据
    groundwater_data_file = os.path.join(data_dir, "groundwater_training.csv")
    if os.path.exists(groundwater_data_file):
        analyze_remediation_tech(groundwater_data_file, output_dir, "groundwater")
    else:
        print(f"[WARNING] 地下水训练数据文件不存在: {groundwater_data_file}")
    
    print("[INFO] 分析完成！")

if __name__ == "__main__":
    main() 