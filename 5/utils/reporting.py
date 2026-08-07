import os
import json
import yaml
from fpdf import FPDF

def generate_final_report(cfg):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Deep Learning Assignment 5: Vehicle Detection", ln=True, align='C')
    pdf.ln(5)
    
    # 1. Workflow and Augmentation 
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Methodology & Augmentation", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, "- Dataset: 100 images, split into 70 Train, 15 Val, 15 Test.\n"
                         "- Augmentation: Mosaic, flip, scale, and color jitter applied ONLY to training set.\n"
                         "- Models: YOLOv8s, YOLOv8m, YOLOv8l (Fine-tuned from COCO pretrained weights).")
    pdf.ln(5)
    
    # 2. Class Distribution
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Class Distribution", ln=True)
    pdf.set_font("Arial", size=11)
    try:
        with open('class_distribution.json', 'r') as f:
            dist = json.load(f)
        for cls, count in dist.items():
            pdf.cell(0, 8, f"- {cls}: {count} instances", ln=True)
        pdf.ln(3)
        pdf.multi_cell(0, 8, "Analysis: The dataset is imbalanced (e.g., 'Car' dominates). This may cause lower recall for minority classes like 'Bicycle' or 'Motorcycle'.")
    except:
        pdf.cell(0, 8, "Class distribution data not found.", ln=True)
    pdf.ln(5)
    
    # 3. Confusion Matrix
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. Confusion Matrix (Test Set)", ln=True)
    cm_path = "./utils/confusion_matrix.png"
    if os.path.exists(cm_path):
        pdf.image(cm_path, x=10, y=pdf.get_y(), w=190)
        pdf.ln(90) # Space for the image
    else:
        pdf.cell(0, 8, "Confusion matrix not generated. Run evaluation first.", ln=True)
        
    # 4. Prediction Examples
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "4. Prediction Examples", ln=True)
    pred_path = "./utils/sample_prediction.jpg"
    if os.path.exists(pred_path):
        pdf.image(pred_path, x=10, y=pdf.get_y(), w=190)
        pdf.ln(90)
    else:
        pdf.cell(0, 8, "Prediction sample not found.", ln=True)

    pdf.output("Final_Report.pdf")
    print("Final Report generated: Final_Report.pdf")