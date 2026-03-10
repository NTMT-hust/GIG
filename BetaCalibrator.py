import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


class BetaCalibrator(nn.Module):
    """
    Beta Calibration per class + Per-class Threshold Tuning.

    References:
        Kull et al. (NIPS 2017) — "Beta calibration: a well-founded and
        easily implemented improvement on logistic calibration for binary classifiers"

    How it works (OvR per class):
        For each class c:
            p_c  = uncalibrated probability for class c
            features = [log(p_c), log(1 - p_c)]          # two-feature beta transform
            fit Logistic Regression → P(y=c | features)   # re-calibrated prob

        After fitting, renormalise across all classes so probs sum to 1.

    Threshold Tuning (per class):
        After calibration, sweep threshold t ∈ [0.05, 0.95] for each class and
        pick the t that maximises per-class F1. Final prediction:
            pred = argmax(prob[c] - threshold[c])          # "adjusted argmax"
        If all values are negative → fallback to argmax(prob).

    API mirrors TemperatureScaler for drop-in replacement:
        calibrator = BetaCalibrator(num_classes=C)
        calibrator.fit_loader(model, val_loader, device)   # fit calibrators
        calibrator.tune_thresholds(val_probs, val_labels)  # tune thresholds on val
        probs = calibrator.calibrate_probs(logits)         # tensor in → tensor out
        preds = calibrator.predict_with_thresholds(probs)  # numpy array out
    """

    def __init__(self, num_classes: int = None):
        super().__init__()
        self.num_classes  = num_classes
        self.calibrators  = []          # list[LogisticRegression], one per class
        self.thresholds   = None        # np.ndarray [C], per-class decision thresholds
        self._fitted      = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _beta_features(probs_1d: np.ndarray) -> np.ndarray:
        """
        Compute two-column beta calibration features from a 1-D probability array.
        Returns: [N, 2] array of [log(p), log(1-p)].
        """
        p = np.clip(probs_1d, 1e-7, 1 - 1e-7)
        return np.column_stack([np.log(p), np.log(1.0 - p)])

    # ------------------------------------------------------------------
    # fit: train one LogisticRegression per class (numpy-level)
    # ------------------------------------------------------------------
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "BetaCalibrator":
        """
        Fit beta calibration on pre-computed softmax probabilities.

        Args:
            probs  : np.ndarray [N, C]  — softmax probabilities (NOT raw logits)
            labels : np.ndarray [N]     — integer ground-truth class indices
        Returns:
            self
        """
        n_classes = probs.shape[1]
        self.num_classes = n_classes
        self.calibrators = []

        for c in range(n_classes):
            X = self._beta_features(probs[:, c])
            y = (labels == c).astype(int)

            # LogisticRegression as the beta calibration "sigmoid layer"
            clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                     class_weight='balanced')
            clf.fit(X, y)
            self.calibrators.append(clf)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # calibrate_probs_numpy: apply calibration to numpy probs
    # ------------------------------------------------------------------
    def calibrate_probs_numpy(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply fitted beta calibrators and renormalise.

        Args:
            probs : np.ndarray [N, C]  — raw softmax probabilities
        Returns:
            calibrated : np.ndarray [N, C]  — calibrated, renormalised probs
        """
        assert self._fitted, "Call fit() or fit_loader() before calibrating."
        n, n_classes = probs.shape
        calibrated = np.zeros((n, n_classes), dtype=np.float64)

        for c in range(n_classes):
            X = self._beta_features(probs[:, c])
            # predict_proba returns [P(y=0), P(y=1)]; we want P(y=1)
            calibrated[:, c] = self.calibrators[c].predict_proba(X)[:, 1]

        # Renormalise rows to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated /= np.maximum(row_sums, 1e-9)
        return calibrated.astype(np.float32)

    # ------------------------------------------------------------------
    # fit_loader: convenience — collect probs from DataLoader then fit
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit_loader(self,
                   model: nn.Module,
                   loader,
                   device: torch.device,
                   **kwargs) -> "BetaCalibrator":
        """
        Collect softmax probabilities from val_loader, then fit beta calibrators.

        Args:
            model  : nn.Module in eval mode (best checkpoint already loaded)
            loader : DataLoader of the validation set
            device : compute device
        Returns:
            self
        """
        model.eval()
        model.to(device)

        all_probs  = []
        all_labels = []

        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs  = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

        all_probs  = np.concatenate(all_probs,  axis=0)   # [N, C]
        all_labels = np.array(all_labels)                  # [N]

        # NLL before calibration (as baseline)
        log_p_before = np.log(np.clip(all_probs[np.arange(len(all_labels)), all_labels], 1e-9, 1))
        nll_before   = -log_p_before.mean()

        self.fit(all_probs, all_labels)

        calib_probs   = self.calibrate_probs_numpy(all_probs)
        log_p_after   = np.log(np.clip(calib_probs[np.arange(len(all_labels)), all_labels], 1e-9, 1))
        nll_after     = -log_p_after.mean()

        print(f'  [BetaCalibrator] Fitted {self.num_classes} per-class calibrators.')
        print(f'  [BetaCalibrator] NLL before: {nll_before:.4f}  →  after: {nll_after:.4f}')
        return self

    # ------------------------------------------------------------------
    # calibrate_probs: Tensor in / Tensor out  (matches TemperatureScaler API)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Accept raw logits (Tensor), return calibrated probabilities (Tensor).
        Drop-in replacement for TemperatureScaler.calibrate_probs().

        Args:
            logits : Tensor [N, C]  — raw model output (before softmax)
        Returns:
            Tensor [N, C]  — calibrated probabilities
        """
        probs_np    = torch.softmax(logits, dim=1).cpu().numpy()
        calib_np    = self.calibrate_probs_numpy(probs_np)
        return torch.tensor(calib_np, dtype=torch.float32, device=logits.device)

    # ------------------------------------------------------------------
    # tune_thresholds: sweep thresholds per class on validation probs
    # ------------------------------------------------------------------
    def tune_thresholds(self,
                        probs : np.ndarray,
                        labels: np.ndarray,
                        metric: str = 'f1') -> np.ndarray:
        """
        For each class, find the decision threshold that maximises per-class F1.
        Stores result in self.thresholds [C].

        Args:
            probs  : np.ndarray [N, C]  — *calibrated* probabilities
            labels : np.ndarray [N]     — integer ground-truth labels
            metric : 'f1' (only option currently)
        Returns:
            thresholds : np.ndarray [C]
        """
        n_classes = probs.shape[1]
        thresholds = np.full(n_classes, 0.5)
        candidate_thresholds = np.arange(0.05, 0.95, 0.01)

        print(f'\n  [BetaCalibrator] Tuning per-class thresholds ...')
        for c in range(n_classes):
            y_bin    = (labels == c).astype(int)
            best_t   = 0.5
            best_f1  = 0.0

            for t in candidate_thresholds:
                preds = (probs[:, c] >= t).astype(int)
                score = f1_score(y_bin, preds, zero_division=0)
                if score > best_f1:
                    best_f1, best_t = score, float(t)

            thresholds[c] = best_t
            print(f'    Class {c:>2d}: threshold = {best_t:.2f}   F1 = {best_f1:.4f}')

        self.thresholds = thresholds
        print(f'  [BetaCalibrator] Thresholds: {np.round(thresholds, 3).tolist()}')
        return thresholds

    # ------------------------------------------------------------------
    # predict_with_thresholds: adjusted argmax using per-class thresholds
    # ------------------------------------------------------------------
    def predict_with_thresholds(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply per-class thresholds and return predicted class indices.

        Strategy — "adjusted argmax":
            score[c] = prob[c] - threshold[c]
            pred     = argmax(score)

        This is equivalent to standard argmax when all thresholds = 0.5 and
        probs are renormalised.

        Args:
            probs : np.ndarray [N, C] or torch.Tensor [N, C]
        Returns:
            preds : np.ndarray [N]
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()

        if self.thresholds is None:
            return np.argmax(probs, axis=1)

        adjusted = probs - self.thresholds[np.newaxis, :]   # broadcast [N, C]
        return np.argmax(adjusted, axis=1)

    # ------------------------------------------------------------------
    # Compatibility property (TemperatureScaler had .T)
    # ------------------------------------------------------------------
    @property
    def T(self):
        """Compatibility shim — BetaCalibrator has no single temperature."""
        return None

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        fitted_str = f"fitted={self._fitted}"
        thresh_str = (f"thresholds={np.round(self.thresholds, 3).tolist()}"
                      if self.thresholds is not None else "thresholds=None")
        return f"BetaCalibrator(num_classes={self.num_classes}, {fitted_str}, {thresh_str})"
