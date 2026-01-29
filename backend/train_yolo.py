#!/usr/bin/env python3
"""
Train YOLOv11 on your custom annotated worksheet dataset
"""

from ultralytics import YOLO
import os

def train_yolo():
    """
    Train YOLOv11 model on question detection dataset
    """
    
    # Check if dataset exists
    if not os.path.exists('train/images') or not os.path.exists('train/labels'):
        print("ERROR: Missing train/images or train/labels folders!")
        print("Please ensure your annotated data is in:")
        print("  - train/images/  (your images)")
        print("  - train/labels/  (your YOLO format labels)")
        return
    
    if not os.path.exists('dataset.yaml'):
        print("ERROR: Missing dataset.yaml!")
        print("Please create dataset.yaml with your dataset configuration")
        return
    
    # Count images and labels
    num_images = len([f for f in os.listdir('train/images') if f.endswith(('.jpg', '.png', '.jpeg'))])
    num_labels = len([f for f in os.listdir('train/labels') if f.endswith('.txt')])
    
    print(f"Found {num_images} images and {num_labels} label files")
    
    if num_images == 0:
        print("ERROR: No images found in train/images/")
        return
    
    if num_labels == 0:
        print("ERROR: No labels found in train/labels/")
        return
    
    if num_images != num_labels:
        print(f"WARNING: Mismatch between images ({num_images}) and labels ({num_labels})")
        print("This might be okay if some images have no annotations")
    
    print("\n" + "="*70)
    print("Starting YOLOv11 Training")
    print("="*70)
    print(f"Images: {num_images}")
    print(f"Labels: {num_labels}")
    print(f"Model: YOLOv11n (nano - fastest)")
    print("="*70 + "\n")
    
    # Load pre-trained YOLOv11 nano model
    model = YOLO('yolo11n.pt')  # Will auto-download if not present
    
    # Train the model
    results = model.train(
        data='dataset.yaml',           # Path to dataset config
        epochs=100,                     # Number of training epochs
        imgsz=640,                      # Image size
        batch=8,                        # Batch size (adjust based on GPU memory)
        patience=20,                    # Early stopping patience
        save=True,                      # Save checkpoints
        project='runs/train',           # Where to save results
        name='question_detector',       # Experiment name
        exist_ok=True,                  # Overwrite existing
        pretrained=True,                # Use pretrained weights
        optimizer='AdamW',              # Optimizer
        lr0=0.001,                      # Initial learning rate
        weight_decay=0.0005,            # Weight decay
        
        # Augmentation (helps with small datasets)
        hsv_h=0.015,                    # Hue augmentation
        hsv_s=0.7,                      # Saturation augmentation
        hsv_v=0.4,                      # Value augmentation
        degrees=10.0,                   # Rotation augmentation
        translate=0.1,                  # Translation augmentation
        scale=0.5,                      # Scale augmentation
        flipud=0.0,                     # Vertical flip probability
        fliplr=0.5,                     # Horizontal flip probability
        mosaic=1.0,                     # Mosaic augmentation
        
        # Performance
        cache=False,                    # Cache images (use if you have RAM)
        device='cpu',                   # Use 'cuda' if GPU available
        workers=4,                      # Number of worker threads
        verbose=True,                   # Verbose output
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best model saved to: runs/train/question_detector/weights/best.pt")
    print(f"Last model saved to: runs/train/question_detector/weights/last.pt")
    print("\nTo use the trained model:")
    print("  1. Copy best.pt to your backend folder")
    print("  2. Run: python split_pdf.py input.pdf output/ --model best.pt")
    print("="*70 + "\n")
    
    # Validate the model
    print("Running validation...")
    metrics = model.val()
    
    print(f"\nValidation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    train_yolo()