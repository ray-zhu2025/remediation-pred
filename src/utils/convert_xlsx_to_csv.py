import pandas as pd
import os

def convert_xlsx_to_csv():
    # 定义输入输出路径
    input_dir = 'data/prediction'
    output_dir = 'data/prediction'
    
    # 定义文件映射关系
    file_mapping = {
        '示范场地-土壤-矢量值-新分区.xlsx': 'soil_prediction_new.csv',
        '示范场地-地下水-矢量值-新分区.xlsx': 'groundwater_prediction_new.csv'
    }
    
    # 读取现有CSV文件以获取列名
    soil_columns = pd.read_csv(os.path.join(input_dir, 'soil_prediction.csv')).columns
    groundwater_columns = pd.read_csv(os.path.join(input_dir, 'groundwater_prediction.csv')).columns
    
    # 处理每个文件
    for input_file, output_file in file_mapping.items():
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, output_file)
        
        if not os.path.exists(input_path):
            print(f"文件不存在: {input_path}")
            continue
            
        # 读取Excel文件
        df = pd.read_excel(input_path)
        
        # 根据文件类型选择对应的列名
        if '土壤' in input_file:
            target_columns = soil_columns
        else:
            target_columns = groundwater_columns
            
        # 重命名列以匹配目标格式
        # 这里需要根据实际的Excel文件列名进行映射
        # 示例映射（需要根据实际Excel文件调整）
        column_mapping = {
            '场地名称': '场地名称',
            '场地分区': '场地分区',
            '所属行业': '所属行业',
            '场地现状': '场地现状',
            '特征污染物': '土壤特征污染物' if '土壤' in input_file else '地下水特征污染物',
            '污染物超标倍数': '污染物超标倍数\\污染程度',
            '污染区域面积': '污染区域面积',
            '污染深度': '污染深度',
            '污染物中是否含持久性有机污染物': '污染物中是否含持久性有机污染物',
            '年降水量': '年降水量',
            '包气带土壤渗透性': '包气带土壤渗透性',
            '污染物挥发性': '污染物挥发性',
            '污染物迁移性': '污染物迁移性',
            '土壤PH值': '土壤PH值',
            '土壤含水率': '土壤含水率',
            '土壤颗粒密度': '土壤颗粒密度',
            '土壤有机质含量': '土壤有机质含量',
            '敏感目标类型': '敏感目标类型',
            '地块及周边500米内人口数量': '地块及周边500米内人口数量',
            '土地利用规划': '土地利用规划',
            '污染区域离最近敏感目标的距离': '污染区域离最近敏感目标的距离',
            '可操作性': '可操作性',
            '适用土壤渗透性': '适用土壤渗透性',
            '污染物去除效率': '污染物去除效率',
            '修复时间': '修复时间',
            '成熟性': '成熟性',
            '费用': '费用',
            '二次污染': '二次污染',
            '公众可接受程度': '公众可接受程度',
            '修复面积': '修复面积'
        }
        
        # 重命名列
        df = df.rename(columns=column_mapping)
        
        # 确保所有必需的列都存在
        missing_columns = set(target_columns) - set(df.columns)
        if missing_columns:
            print(f"警告: 以下列在Excel文件中缺失: {missing_columns}")
            for col in missing_columns:
                df[col] = None  # 添加缺失的列，填充为None
        
        # 选择并重排列以匹配目标格式
        df = df[target_columns]
        
        # 保存为CSV文件
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"已成功转换: {input_file} -> {output_file}")

if __name__ == '__main__':
    convert_xlsx_to_csv() 