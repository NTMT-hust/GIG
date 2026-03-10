import torch
from torch import nn
import torch.optim as optim
import numpy as np


class TemperatureScaler(nn.Module):
    """
    Temperature Scaling for probability calibration.
    Reference: Guo et al. (ICML 2017) — "On Calibration of Modern Neural Networks"

    Usage:
        scaler = TemperatureScaler().to(device)
        scaler.fit_loader(model, val_loader, device)   # fit T on validation set
        calibrated_probs = scaler.calibrate_probs(logits)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))  # khởi tạo T = 1 (không đổi xác suất)

    # ------------------------------------------------------------------
    # Forward: chia logits cho nhiệt độ T
    # ------------------------------------------------------------------
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Trả về logits đã được scale (chưa qua softmax).
        Args:
            logits: Tensor [N, C] — raw logits từ model
        Returns:
            scaled_logits: Tensor [N, C]
        """
        return logits / self.temperature.clamp(min=1e-6)

    # ------------------------------------------------------------------
    # fit: tối ưu T từ logits & labels sẵn có (Tensor)
    # ------------------------------------------------------------------
    def fit(self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            max_iter: int = 100,
            lr: float = 0.01) -> "TemperatureScaler":
        """
        Tối ưu hoá nhiệt độ T bằng NLL loss trên tập validation.
        Args:
            logits : Tensor [N, C] — raw logits (không qua softmax)
            labels : Tensor [N]    — ground-truth class indices
            max_iter: số bước LBFGS tối đa
            lr      : learning rate cho LBFGS
        Returns:
            self (cho phép chain: scaler.fit(...))
        """
        self.to(logits.device)
        nll = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        print(f'  [TemperatureScaler] Learned temperature T = {self.temperature.item():.4f}')
        return self

    # ------------------------------------------------------------------
    # fit_loader: thu thập logits từ DataLoader rồi gọi fit()
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit_loader(self,
                   model: nn.Module,
                   loader,
                   device: torch.device,
                   max_iter: int = 100,
                   lr: float = 0.01) -> "TemperatureScaler":
        """
        Convenience method: chạy model trên toàn bộ val_loader để thu logits,
        sau đó fit nhiệt độ T.
        Args:
            model  : model đã được load best checkpoint, ở eval mode
            loader : DataLoader của tập validation
            device : thiết bị tính toán
            max_iter, lr: truyền xuống fit()
        Returns:
            self
        """
        model.eval()
        model.to(device)
        self.to(device)

        all_logits = []
        all_labels = []

        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)           # [B, C]
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)  # [N, C]
        all_labels = torch.cat(all_labels, dim=0)  # [N]

        # Chuyển về device để tối ưu
        all_logits = all_logits.to(device)
        all_labels = all_labels.to(device)

        # Ghi lại NLL trước khi calibrate
        nll = nn.CrossEntropyLoss()
        nll_before = nll(all_logits, all_labels).item()

        # Fit T (bật gradient tạm thời trong fit())
        with torch.enable_grad():
            self.fit(all_logits, all_labels, max_iter=max_iter, lr=lr)

        # Ghi lại NLL sau khi calibrate
        nll_after = nll(self.forward(all_logits), all_labels).item()
        print(f'  [TemperatureScaler] NLL before: {nll_before:.4f}  →  after: {nll_after:.4f}')

        return self

    # ------------------------------------------------------------------
    # calibrate_probs: trả về xác suất đã calibrate (qua softmax)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Trả về xác suất đã calibrate bằng temperature scaling.
        Args:
            logits: Tensor [N, C]
        Returns:
            probs : Tensor [N, C]  (softmax sau khi scale)
        """
        scaled = self.forward(logits)
        return torch.softmax(scaled, dim=1)

    # ------------------------------------------------------------------
    # get_temperature: tiện lợi để log giá trị T
    # ------------------------------------------------------------------
    @property
    def T(self) -> float:
        """Giá trị nhiệt độ hiện tại (scalar float)."""
        return self.temperature.item()
