import torch
import torch.nn as nn
import timm

class EfficientNetB1Classifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout_rate=0.3):
        super(EfficientNetB1Classifier, self).__init__()
        # Load backbone
        self.backbone = timm.create_model('efficientnet_b1', pretrained=pretrained)
        in_features = self.backbone.classifier.in_features

        # Replace the head
        # We keep the name 'classifier' so timm's internal logic works
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def freeze_backbone(self):
        """Freezes all layers except the final classifier head."""
        for name, param in self.backbone.named_parameters():
            # Check if 'classifier' is in the parameter name
            if 'classifier' in name:
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