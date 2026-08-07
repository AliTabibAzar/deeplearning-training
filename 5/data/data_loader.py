import os
import shutil
import yaml
import json
from collections import defaultdict

def prepare_and_analyze_dataset(cfg):
    raw_imgs = cfg['dataset']['raw_images']
    raw_lbls = cfg['dataset']['raw_labels']
    out_dir = cfg['dataset']['out_dir']
    
    # support multiple image formats
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
    files = sorted([f for f in os.listdir(raw_imgs) if f.endswith(valid_exts)])
    
    if len(files) < 100:
        print(f"Warning: Found only {len(files)} images. Expected 100.")
    
    # adjust split
    n = len(files)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val
    
    splits = {
        'train': files[:n_train],
        'validation': files[n_train:n_train + n_val],
        'test': files[n_train + n_val:]
    }
    
    print(f"Split: {n_train} train, {n_val} val, {n_test} test")
    
    for split in splits:
        os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)
        
    class_counts = defaultdict(int)
    
    for split, file_list in splits.items():
        for f in file_list:
            # Copy image
            shutil.copy(os.path.join(raw_imgs, f), os.path.join(out_dir, 'images', split, f))
            
            # Generate label filename
            base_name = os.path.splitext(f)[0]
            lbl_f = base_name + '.txt'
            lbl_path = os.path.join(raw_lbls, lbl_f)
            dst_lbl = os.path.join(out_dir, 'labels', split, lbl_f)
            
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, dst_lbl)
                with open(lbl_path, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            class_counts[cls_id] += 1
            else:
                print(f"Warning: Label file not found for {f}. Creating empty label.")
                open(dst_lbl, 'a').close()

    # Generate data.yaml
    data_yaml = {
        'path': os.path.abspath(out_dir),
        'train': 'images/train',
        'val': 'images/validation',
        'test': 'images/test',
        'names': cfg['dataset']['classes']
    }
    with open(os.path.join(out_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
        
    # save class distribution
    dist_report = {}
    for cls_id, count in class_counts.items():
        # get class name from config 
        class_name = cfg['dataset']['classes'].get(cls_id) or cfg['dataset']['classes'].get(str(cls_id))
        if class_name:
            dist_report[class_name] = count
    
    with open('class_distribution.json', 'w') as f:
        json.dump(dist_report, f, indent=4, ensure_ascii=False)
        
    print(f"Dataset prepared: {len(files)} images")
    print(f"Class distribution: {dist_report}")