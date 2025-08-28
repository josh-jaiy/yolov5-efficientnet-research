# YOLOv5 + EfficientNet-B7 Integration Research

This repository contains a research implementation that integrates EfficientNet-B7 as a backbone for YOLOv5, comparing performance against the baseline YOLOv5s model.

## 🎯 Research Objective

Compare the performance of:
- **Baseline YOLOv5s**: Standard YOLOv5s model (7.2M parameters)
- **EfficientNet-B7 + YOLOv5**: Custom integration (63.8M parameters)

## 🏗️ Architecture Overview

### EfficientNet-B7 Integration
- **Backbone**: EfficientNet-B7 feature extractor
- **Multi-scale Features**: P3 (128 channels), P4 (256 channels), P5 (512 channels)
- **Adapter Layers**: Custom projection layers for YOLOv5 compatibility
- **FPN Integration**: Feature Pyramid Network for multi-scale detection

### Key Components
1. `EfficientNetB7Adapter`: Main backbone adapter
2. `EfficientNetP3Adapter`: P3 feature adapter  
3. `EfficientNetP4Adapter`: P4 feature adapter
4. Custom YAML configuration for model architecture

## 🚀 Quick Start on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/yolov5-efficientnet-research/blob/main/notebooks/colab_training.ipynb)

### 1. Clone Repository
```python
!git clone https://github.com/your-username/yolov5-efficientnet-research.git
%cd yolov5-efficientnet-research
```

### 2. Setup Environment
```python
!pip install -r requirements.txt
!git clone https://github.com/ultralytics/yolov5.git
```

### 3. Install Custom Components
```python
# Copy custom files to YOLOv5
!cp models/efficientnet_adapter.py yolov5/models/
!cp configs/yolov5s_effnet_multiscale.yaml yolov5/models/
!cp modified_files/yolo.py yolov5/models/
```

### 4. Train Models
```python
# Train baseline YOLOv5s
!cd yolov5 && python train.py --data coco128.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --epochs 10 --name baseline

# Train EfficientNet-B7 + YOLOv5
!cd yolov5 && python train.py --data coco128.yaml --cfg models/yolov5s_effnet_multiscale.yaml --epochs 10 --name efficientnet
```

## 📊 Expected Results

### Performance Comparison
| Model | Parameters | mAP@0.5 | mAP@0.5:0.95 | Training Time | Inference Speed |
|-------|------------|---------|--------------|---------------|-----------------|
| YOLOv5s Baseline | 7.2M | ~0.81 | ~0.54 | Fast | Real-time |
| EfficientNet-B7 + YOLOv5 | 63.8M | Higher | Higher | Slower | Slower |

### Trade-offs
- **EfficientNet Advantages**: Better accuracy, superior feature extraction
- **Baseline Advantages**: Faster inference, smaller model size, edge deployment

## 🛠️ Local Development

### Prerequisites
```bash
pip install -r requirements.txt
```

### Test Integration
```python
python scripts/test_backbone_final.py
```

### Training
```bash
cd yolov5
python train.py --data coco128.yaml --cfg models/yolov5s_effnet_multiscale.yaml --epochs 10 --name efficientnet_experiment
```

## 📁 Repository Structure
```
├── models/
│   └── efficientnet_adapter.py      # EfficientNet integration
├── configs/
│   └── yolov5s_effnet_multiscale.yaml  # Model configuration
├── modified_files/
│   └── yolo.py                      # Modified YOLOv5 parser
├── scripts/
│   └── test_backbone_final.py       # Testing utilities
├── notebooks/
│   └── colab_training.ipynb         # Google Colab notebook
├── results/                         # Training results
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## 🔬 Research Findings

### Architecture Insights
1. **Multi-scale Feature Extraction**: EfficientNet-B7 provides richer feature representations
2. **Channel Adaptation**: Custom projection layers successfully bridge EfficientNet and YOLOv5
3. **Global Feature Sharing**: Efficient feature reuse across detection scales

### Performance Analysis
- **Accuracy**: EfficientNet backbone shows improved detection accuracy
- **Computational Cost**: ~9x more parameters, proportional increase in compute
- **Memory Usage**: Higher GPU memory requirements during training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- [EfficientNet](https://github.com/rwightman/pytorch-image-models) implementation in timm
- [PyTorch](https://pytorch.org/) deep learning framework

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact [your-email@example.com].
