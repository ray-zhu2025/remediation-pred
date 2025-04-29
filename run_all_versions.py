import subprocess
import json
import os
from datetime import datetime

def run_version(version):
    print(f"Running version {version}...")
    subprocess.run(["./run.sh", version], check=True)
    
def collect_metrics():
    metrics = {}
    for version in os.listdir("output/metrics"):
        if version.startswith("v"):
            with open(f"output/metrics/{version}/metrics.json", "r") as f:
                metrics[version] = json.load(f)
    return metrics

def main():
    versions = [
        "1.0.0", "1.1.0", "1.1.1", "1.1.2", 
        "1.1.3", "1.1.4", "1.1.5", "1.1.6"
    ]
    
    for version in versions:
        run_version(version)
    
    metrics = collect_metrics()
    
    # 保存汇总结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"output/metrics_summary_{timestamp}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nMetrics Summary:")
    for version, data in metrics.items():
        print(f"\nVersion {version}:")
        print(f"  Accuracy: {data['accuracy']:.4f}")
        print(f"  Precision: {data['precision']:.4f}")
        print(f"  Recall: {data['recall']:.4f}")
        print(f"  F1 Score: {data['f1_score']:.4f}")

if __name__ == "__main__":
    main() 