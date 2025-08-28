#!/usr/bin/env python3
"""
Google Colab Setup Script for YOLOv5 + EfficientNet Research
"""

import os
import subprocess

def run_command(cmd):
    """Run shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True

def setup_colab_environment():
    """Setup complete environment for Colab"""
    print("🚀 Setting up YOLOv5 + EfficientNet research environment...")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    run_command("pip install timm ultralytics")
    
    # Clone YOLOv5
    print("📥 Cloning YOLOv5...")
    if not os.path.exists('yolov5'):
        run_command("git clone https://github.com/ultralytics/yolov5.git")
    
    os.chdir('yolov5')
    run_command("pip install -r requirements.txt")
    
    # Copy custom files
    print("📋 Installing custom EfficientNet integration...")
    run_command("cp ../models/efficientnet_adapter.py models/")
    run_command("cp ../configs/yolov5s_effnet_multiscale.yaml models/")
    run_command("cp ../modified_files/yolo.py models/")
    
    # Test integration
    print("🧪 Testing EfficientNet integration...")
    run_command("python ../scripts/test_backbone_final.py")
    
    print("✅ Setup complete! Ready to train models.")
    print("\nNext steps:")
    print("1. Train baseline: python train.py --data coco128.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --epochs 10 --name baseline")
    print("2. Train EfficientNet: python train.py --data coco128.yaml --cfg models/yolov5s_effnet_multiscale.yaml --epochs 10 --name efficientnet")

if __name__ == "__main__":
    setup_colab_environment()
