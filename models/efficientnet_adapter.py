import torch
import torch.nn as nn
import timm

# Global storage for EfficientNet features
_efficientnet_features = {}

class EfficientNetB7Backbone(nn.Module):
    """EfficientNet-B7 feature extractor"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.efficientnet = timm.create_model('efficientnet_b7', pretrained=pretrained, features_only=True)

        # Test to see what features we get
        with torch.no_grad():
            dummy = torch.randn(1, 3, 640, 640)
            test_features = self.efficientnet(dummy)
            print(f"EfficientNet-B7 feature shapes: {[f.shape for f in test_features]}")

        # Store indices for P3, P4, P5
        self.p3_idx, self.p4_idx, self.p5_idx = 2, 3, 4

    def forward(self, x):
        features = self.efficientnet(x)
        # Store features globally for other adapters to access
        global _efficientnet_features
        _efficientnet_features['p3'] = features[self.p3_idx]  # 80x80
        _efficientnet_features['p4'] = features[self.p4_idx]  # 40x40
        _efficientnet_features['p5'] = features[self.p5_idx]  # 20x20
        return features[self.p5_idx]  # Return P5 as main output

class EfficientNetP3Adapter(nn.Module):
    """Adapter for P3 features"""
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(80, 128, 1, 1, 0)  # 80 -> 128 channels
        self.out_channels = 128

    def forward(self, x):
        global _efficientnet_features
        if 'p3' in _efficientnet_features:
            return self.proj(_efficientnet_features['p3'])
        else:
            # Fallback - shouldn't happen in normal flow
            return torch.zeros(x.shape[0], 128, 80, 80, device=x.device)

class EfficientNetP4Adapter(nn.Module):
    """Adapter for P4 features"""
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(224, 256, 1, 1, 0)  # 224 -> 256 channels
        self.out_channels = 256

    def forward(self, x):
        global _efficientnet_features
        if 'p4' in _efficientnet_features:
            return self.proj(_efficientnet_features['p4'])
        else:
            # Fallback - shouldn't happen in normal flow
            return torch.zeros(x.shape[0], 256, 40, 40, device=x.device)

class EfficientNetB7Adapter(nn.Module):
    """Main EfficientNet-B7 adapter that returns P5"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = EfficientNetB7Backbone(pretrained)
        self.proj = nn.Conv2d(640, 512, 1, 1, 0)  # 640 -> 512 channels
        self.out_channels = 512

    def forward(self, x):
        p5_raw = self.backbone(x)  # This also stores P3, P4 globally
        return self.proj(p5_raw)
