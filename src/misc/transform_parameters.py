"""
参数CSV文件转换工具
将data/parameters下的CSV文件转换为与预测CSV相同的格式
"""

import os
import pandas as pd
from pathlib import Path

def transform_parameters(input_dir: str, output_dir: str) -> None:
    """
    转换参数CSV文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print("[WARNING] 未找到CSV文件！")
        return
    
    print(f"[INFO] 找到 {len(csv_files)} 个CSV文件")
    
    # 定义标准列名
    standard_columns = [
        "场地名称", "场地分区", "所属行业", "场地现状", "土壤特征污染物",
        "污染物超标倍数\\污染程度", "污染区域面积", "污染深度",
        "污染物中是否含持久性有机污染物", "年降水量", "包气带土壤渗透性",
        "污染物挥发性", "污染物迁移性", "土壤PH值", "土壤含水率",
        "土壤颗粒密度", "土壤有机质含量", "敏感目标类型",
        "地块及周边500米内人口数量", "土地利用规划",
        "污染区域离最近敏感目标的距离", "可操作性", "适用土壤渗透性",
        "污染物去除效率", "修复时间", "成熟性", "费用", "二次污染",
        "公众可接受程度", "修复面积", "修复土方量"
    ]
    
    for csv_file in csv_files:
        try:
            # 获取相对路径，用于创建对应的输出目录结构
            rel_path = os.path.relpath(csv_file, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            
            # 创建输出文件的目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 读取CSV文件
            print(f"[INFO] 正在处理: {rel_path}")
            df = pd.read_csv(csv_file, encoding='utf-8')
            
            # 验证数据
            if df.empty:
                print(f"[WARNING] 文件 {rel_path} 为空，跳过转换")
                continue
            
            # 显示原始数据信息
            print(f"[INFO] 原始数据形状: {df.shape}")
            print(f"[INFO] 原始列名: {', '.join(df.columns)}")
            
            # 创建新的DataFrame，包含所有标准列
            new_df = pd.DataFrame(columns=standard_columns)
            
            # 复制现有数据到新DataFrame
            for col in df.columns:
                if col in standard_columns:
                    new_df[col] = df[col]
            
            # 填充缺失值为空字符串
            new_df = new_df.fillna('')
            
            # 保存转换后的文件
            new_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"[INFO] 成功转换: {rel_path} -> {os.path.relpath(output_path, output_dir)}")
            print(f"[INFO] 转换后的数据形状: {new_df.shape}")
            
        except Exception as e:
            print(f"[ERROR] 转换失败 {rel_path}: {str(e)}")
            if os.path.exists(output_path):
                os.remove(output_path)

def main():
    """主函数"""
    # 获取项目根目录
    project_root = str(Path(__file__).parent.parent.parent)
    
    # 设置输入和输出目录
    input_dir = os.path.join(project_root, "data", "parameters")
    output_dir = os.path.join(project_root, "data", "parameters_transformed")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        return
    
    # 执行转换
    transform_parameters(input_dir, output_dir)
    print("[INFO] 转换完成！")

if __name__ == "__main__":
    main() 