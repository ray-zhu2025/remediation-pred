import os
import logging
from models.medium_models.soil_model import SoilModel
from models.medium_models.groundwater_model import GroundwaterModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 设置输出目录
    soil_output_dir = 'output/soil'
    groundwater_output_dir = 'output/groundwater'
    
    # 训练和评估土壤模型
    logger.info("训练和评估土壤模型...")
    soil_model = SoilModel('src/config/soil/parameters.json')
    soil_model.train('data/training/soil_training.csv')
    soil_model.evaluate('data/training/soil_training.csv', soil_output_dir)
    
    # 训练和评估地下水模型
    logger.info("训练和评估地下水模型...")
    groundwater_model = GroundwaterModel('src/config/groundwater/parameters.json')
    groundwater_model.train('data/training/groundwater_training.csv')
    groundwater_model.evaluate('data/training/groundwater_training.csv', groundwater_output_dir)
    
    logger.info("所有模型训练和评估完成！")

if __name__ == '__main__':
    main() 