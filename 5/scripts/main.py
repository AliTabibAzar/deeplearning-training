import yaml
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import prepare_and_analyze_dataset
from scripts.train import run_training
from scripts.evaluate import evaluate_models
from utils.reporting import generate_final_report

def main():
    with open('config/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        
    print("preparing dataset (70/15/15) & analyzing distribution...")
    prepare_and_analyze_dataset(cfg)
    
    print("fine-tuning YOLOv8 (s, m, l)...")
    run_training(cfg)
    
    print("evaluating on Test Set & extracting plots...")
    evaluate_models(cfg)
    
    print("generating Final PDF Report...")
    generate_final_report(cfg)
    
    print("\n\npipeline finished. results saved")

if __name__ == "__main__":
    main()