from ultralytics import YOLO

def get_model(size='n'):
    # size 'n' (nano), 's' (small), 'm' (medium)
    return YOLO(f"yolov8{size}.pt")