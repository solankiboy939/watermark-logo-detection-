import os
from ultralytics import YOLO

def main():
    # Optional: Prevent memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Path to the data.yaml file
    data_yaml_path = r"C:\Users\shiva\OneDrive\Desktop\solanki\WatermarkDataset\SplitData\data.yaml"

    # Create a YOLO model object for YOLOv5 Nano (yolov5n.pt)
    model = YOLO('yolov5n.pt')  # Pretrained YOLOv5 Nano weights

    # Train the model
    model.train(
        data=data_yaml_path,  # Path to the data.yaml file
        epochs=300,            # Number of training epochs
        batch=16,              # Batch size for training, adjust based on GPU memory
        imgsz=640,             # Image size for training
        optimizer='Adam',      # Optimizer
        lr0=0.01,              # Initial learning rate
        patience=25,           # Early stopping patience
        device='cuda'          # Optional: force use of GPU
    )

    # Save the trained model
    model.export(format='torchscript')  # Export the model in TorchScript format
0
if __name__ == '__main__':
    main()
