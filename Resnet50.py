import torch
import torch.nn as nn
import timm

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout_rate=0.3, model_name='resnet50'):
        """
        Args:
            num_classes: Số lượng class cần phân loại
            pretrained: Sử dụng pretrained weights hay không
            dropout_rate: Tỷ lệ dropout
            model_name: Tên model ResNet ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        """
        super(ResNetClassifier, self).__init__()
        # Load backbone ResNet
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Lấy số features từ layer cuối
        in_features = self.backbone.fc.in_features
        
        # Thay thế classifier head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    def freeze_backbone(self):
        """Freezes all layers except the final classifier head."""
        for name, param in self.backbone.named_parameters():
            # Check if 'fc' is in the parameter name (fc là tên của classifier trong ResNet)
            if 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("✓ Backbone frozen (Classifier head is TRAINABLE)")
    
    def unfreeze_backbone(self):
        """Unfreezes everything."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen (Full model is TRAINABLE)")
    
    def forward(self, x):
        return self.backbone(x)
    
    def extract_features(self, x):
        """Extract deep features before classifier"""
        x = self.backbone.forward_features(x)
        x = self.backbone.global_pool(x)
        return x