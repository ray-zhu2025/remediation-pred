"""
Excel文件转CSV工具
将04模型相关参数文件夹下的xlsx文件转换为csv文件
"""

import os
import pandas as pd
from pathlib import Path

def find_excel_files(directory: str) -> list:
    """
    递归搜索目录下的所有xlsx文件
    
    Args:
        directory: 要搜索的目录
        
    Returns:
        list: 包含所有xlsx文件路径的列表
    """
    excel_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.xlsx'):
                excel_files.append(os.path.join(root, file))
    return excel_files

def convert_excel_to_csv(input_dir: str, output_dir: str) -> None:
    """
    将Excel文件转换为CSV文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 递归获取所有xlsx文件
    excel_files = find_excel_files(input_dir)
    
    if not excel_files:
        print("[WARNING] 未找到xlsx文件！")
        return
    
    print(f"[INFO] 找到 {len(excel_files)} 个xlsx文件")
    
    for excel_file in excel_files:
        try:
            # 获取相对路径，用于创建对应的输出目录结构
            rel_path = os.path.relpath(excel_file, input_dir)
            output_path = os.path.join(output_dir, rel_path.replace('.xlsx', '.csv'))
            
            # 创建输出文件的目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 读取Excel文件
            print(f"[INFO] 正在处理: {rel_path}")
            df = pd.read_excel(excel_file)
            
            # 验证数据
            if df.empty:
                print(f"[WARNING] 文件 {rel_path} 为空，跳过转换")
                continue
                
            if df.isnull().all().all():
                print(f"[WARNING] 文件 {rel_path} 所有数据为空，跳过转换")
                continue
            
            # 显示数据基本信息
            print(f"[INFO] 数据形状: {df.shape}")
            print(f"[INFO] 列名: {', '.join(df.columns)}")
            
            # 保存为CSV文件，使用UTF-8编码
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"[INFO] 成功转换: {rel_path} -> {os.path.relpath(output_path, output_dir)}")
            
            # 验证转换后的文件
            if os.path.getsize(output_path) == 0:
                print(f"[ERROR] 转换后的文件 {output_path} 为空")
                os.remove(output_path)
            else:
                # 验证CSV文件是否可以正确读取
                test_df = pd.read_csv(output_path, encoding='utf-8')
                if test_df.empty:
                    print(f"[ERROR] 转换后的文件 {output_path} 数据为空")
                    os.remove(output_path)
                else:
                    print(f"[INFO] 转换后的数据形状: {test_df.shape}")
            
        except Exception as e:
            print(f"[ERROR] 转换失败 {rel_path}: {str(e)}")
            if os.path.exists(output_path):
                os.remove(output_path)

def main():
    """主函数"""
    # 获取项目根目录
    project_root = str(Path(__file__).parent.parent.parent)
    
    # 设置输入和输出目录
    input_dir = os.path.join(project_root, "04模型相关参数")
    output_dir = os.path.join(project_root, "data", "parameters")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        return
    
    # 执行转换
    convert_excel_to_csv(input_dir, output_dir)
    print("[INFO] 转换完成！")

if __name__ == "__main__":
    main() 