import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F

# ===================== GRAD-CAM CORE =====================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # Hook for the forward pass to get activations
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # Hook for the backward pass to get gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        score = output[0, class_idx]
        score.backward()

        # Weight the channels by the corresponding gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU and normalization
        heatmap = torch.clamp(heatmap, min=0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
            
        return heatmap.cpu().numpy(), class_idx

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    


    def upsample_heatmap(self, heatmap, input_tensor):
        """
        heatmap: (H_cam, W_cam) tensor
        input_tensor: (1, C, H, W)
        """
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # (1,1,H_cam,W_cam)
        heatmap = F.interpolate(
            heatmap,
            size=input_tensor.shape[2:],  # (H, W)
            mode="bilinear",
            align_corners=False
        )
        return heatmap.squeeze()  # (H, W)
