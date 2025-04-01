"""
主程序入口
"""
import os
import json
import logging
import argparse
from datetime import datetime
from src.config.version_config import VersionConfig
from src.process.data_processor import DataProcessor
from src.models.soil_model import SoilModel
from src.models.groundwater_model import GroundwaterModel

def setup_logging():
    """设置日志"""
    if not os.path.exists('output/logs'):
        os.makedirs('output/logs')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    version = VersionConfig.get_version()
    log_file = f'output/logs/model_v{version}_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行模型')
    parser.add_argument('--version', type=str, required=True, help='模型版本')
    args = parser.parse_args()
    
    # 先设置版本
    VersionConfig.set_version(args.version)
    # 再设置日志
    logger = setup_logging()
    logger.info(f"开始运行版本 {args.version}")
    
    try:
        # 加载配置
        config = VersionConfig.get_version_config()
        
        # 处理数据
        processor = DataProcessor()
        soil_data = processor.load_soil_data()
        groundwater_data = processor.load_groundwater_data()
        
        # 训练土壤模型
        soil_model = SoilModel(model_type=config['model_type'], version=args.version)
        soil_model.train(
            soil_data['X_train'], 
            soil_data['y_train'],
            soil_data['X_test'],
            soil_data['y_test']
        )
        soil_metrics = soil_model.evaluate(soil_data['X_test'], soil_data['y_test'])
        
        # 训练地下水模型
        groundwater_model = GroundwaterModel(model_type=config['model_type'], version=args.version)
        groundwater_model.train(
            groundwater_data['X_train'],
            groundwater_data['y_train'],
            groundwater_data['X_test'],
            groundwater_data['y_test']
        )
        groundwater_metrics = groundwater_model.evaluate(groundwater_data['X_test'], groundwater_data['y_test'])
        
        # 输出结果
        logger.info("\n模型评估结果:")
        logger.info(f"土壤模型准确率: {soil_metrics['accuracy']:.4f}")
        logger.info(f"地下水模型准确率: {groundwater_metrics['accuracy']:.4f}")
        
        # 保存模型
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        soil_save_path = os.path.join('models', 'soil', f'v{args.version}', timestamp)
        groundwater_save_path = os.path.join('models', 'groundwater', f'v{args.version}', timestamp)
        os.makedirs(os.path.dirname(soil_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(groundwater_save_path), exist_ok=True)
        soil_model.predictor.save(soil_save_path)
        groundwater_model.predictor.save(groundwater_save_path)
        logger.info("模型已保存到 models 目录")
        
        # 保存评估指标
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = f'output/metrics/model_metrics_v{args.version}_{timestamp}.json'
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                'soil': soil_metrics,
                'groundwater': groundwater_metrics
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"评估指标已保存到: {metrics_file}")
        
    except Exception as e:
        logger.error(f"运行出错: {str(e)}")
        raise

if __name__ == '__main__':
    main() 