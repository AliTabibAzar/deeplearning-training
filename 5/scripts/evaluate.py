import os
import shutil
from models.model import get_model

def evaluate_models(cfg):
    out_dir = cfg['dataset']['out_dir']
    
    size = 's'
    print(f"--- Evaluating YOLOv8-{size.upper()} on Test Set ---")
    
    model = get_model(size)
    
    metrics = model.val(
        data=os.path.join(out_dir, 'data.yaml'),
        split='test',
        imgsz=cfg['training']['imgsz'],
        save_json=True,
        project='./runs/eval',
        name=f"eval_{size}",
        plots=True 
    )
    
    print(f"test mAP50: {metrics.box.map50:.4f}")
    
    eval_dir = f"./runs/eval/eval_{size}"
    train_dir = f"./runs/detect/yolov8_{size}"
    
    src_cm = os.path.join(eval_dir, "confusion_matrix.png")
    dst_cm = "./utils/confusion_matrix.png"
    
    if os.path.exists(src_cm):
        shutil.copy(src_cm, dst_cm)
        print("confusion matrix copied successfully.")
    else:
        print("warning: Confusion matrix not found in eval dir.")

    # 2. کپی کردن نمونه پیش‌بینی (با جستجوی هوشمند)
    pred_found = False
    
    for directory in [eval_dir, train_dir]:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.endswith(".jpg") and "pred" in filename.lower():
                    shutil.copy(os.path.join(directory, filename), dst_pred)
                    print(f"prediction sample copied from: {directory}/{filename}")
                    pred_found = True
                    break
        if pred_found:
            break
            
    if not pred_found:
        print("صarning: No prediction sample found. Check runs/ directories manually.")

    print("complete. Report can now be generated.")
    return metrics