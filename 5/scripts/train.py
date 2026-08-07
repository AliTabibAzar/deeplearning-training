import os
import wandb
import yaml
from ultralytics import YOLO

def run_training(cfg):
    wandb.init(project=cfg['wandb']['project_name'])
    
    for size in cfg['training']['models']:
        print(f"--- Training YOLOv8-{size.upper()} ---")
        model = YOLO(f"yolov8{size}.pt")
        
        # adjust batch
        current_batch = cfg['training']['batch']
        if size == 'l':
            current_batch = 8 
            print("-> Reduced batch size to 8 for Large model to prevent OOM.")
            
        model.train(
            data=os.path.join(cfg['dataset']['out_dir'], 'data.yaml'),
            epochs=cfg['training']['epochs'],
            imgsz=cfg['training']['imgsz'],
            batch=current_batch,
            optimizer=cfg['training']['optimizer'],
            lr0=cfg['training']['lr0'],
            weight_decay=cfg['training']['weight_decay'],
            seed=cfg['training']['seed'],
            project=cfg['training']['project'],
            name=f"yolov8_{size}",
            exist_ok=True,
            save=True,
            plots=True
        )
        
    wandb.finish()
    print("All models trained. Check W&B for detailed logs.")