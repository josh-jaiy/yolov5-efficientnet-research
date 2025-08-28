#!/usr/bin/env python3
"""
Final test of EfficientNet backbone integration
"""

import os
import sys
import torch

# Change to yolov5 directory if not already there
if not os.getcwd().endswith('yolov5'):
    os.chdir('yolov5')
sys.path.append('.')

def test_backbone():
    """Test the EfficientNet backbone"""
    print("=== Testing EfficientNet Backbone ===")
    
    try:
        from models.efficientnet_adapter import EfficientNetB7Adapter
        
        # Test adapter
        adapter = EfficientNetB7Adapter(pretrained=False)
        dummy_input = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            output = adapter(dummy_input)
            if isinstance(output, list):
                print(f"✅ EfficientNet adapter working: {[o.shape for o in output]}")
            else:
                print(f"✅ EfficientNet adapter working: {output.shape}")
        
        # Test YAML loading
        import yaml
        with open('models/yolov5s_effnet_multiscale.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ YAML config loaded")
        print(f"   Backbone layers: {len(config['backbone'])}")
        print(f"   Head layers: {len(config['head'])}")
        print(f"   First backbone layer: {config['backbone'][0]}")
        
        # Test model creation with detailed error handling
        from models.yolo import Model
        
        print("\n=== Attempting Model Creation ===")
        try:
            model = Model('models/yolov5s_effnet_multiscale.yaml', ch=3, nc=80)
            print("✅ Model created successfully!")
            
            # Test forward pass
            with torch.no_grad():
                output = model(dummy_input)
                print(f"✅ Forward pass successful: {[o.shape for o in output]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Backbone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("Testing EfficientNet-B7 backbone integration...")
    
    success = test_backbone()
    
    if success:
        print("\n🎉 EfficientNet backbone is working!")
        print("\nNext step: Train the model")
        print("python train.py --data coco128.yaml --cfg models/yolov5s_effnet_multiscale.yaml --epochs 10 --name efficientnet_experiment")
    else:
        print("\n❌ Backbone integration needs more work")
        print("The EfficientNet adapter works, but the full model integration has issues")

if __name__ == "__main__":
    main()
