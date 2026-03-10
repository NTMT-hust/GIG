import torch
import random
import numpy as np

def aware_augmentation(image, label, class_pools, dataset,
                       noise_std=0.03,
                       mixup_alpha=0.2,
                       smote_alpha_range=(0.1, 0.4),
                       cutmix_ratio_range = 0.5,
                       p_noise=0.2,
                       p_mix=0.1,
                       p_smote=0.1,
                       p_cutmix=0.1):
    """
    image: Tensor (C,H,W)
    label: int (local label)
    class_pools: dict hoặc list chứa LOCAL indices theo class
    dataset: dataset hiện tại (train dataset của fold)
    """

    device = image.device
    image = image.clone()
    C, H, W = image.shape

    prob = random.random()
    # =====================================================
    # 1️⃣ Gaussian Noise (chỉ pixel != 0)
    # =====================================================
    if prob <= 0.2:
        non_zero_mask = image != 0
        noise = torch.randn_like(image) * noise_std
        image = image + noise * non_zero_mask.float()

    # =====================================================
    # Nếu class chỉ có 1 mẫu thì không augment tiếp
    # =====================================================
    pool = class_pools[label]
    if len(pool) <= 1:
        return image, label

    # Chọn 1 index khác idx hiện tại
    rand_idx = random.choice(pool)
    other_img, _ = dataset[rand_idx]
    other_img = other_img.to(device)

    # =====================================================
    # 2️⃣ CutMix (same class)
    # =====================================================
    if 0.2 < prob and prob <= 0.3:
        cutmix_ratio = random.uniform(0.2, cutmix_ratio_range)
        cut_w = int(W * cutmix_ratio)
        cut_h = int(H * cutmix_ratio)

        cx = random.randint(0, W - cut_w)
        cy = random.randint(0, H - cut_h)

        image[:, cy:cy+cut_h, cx:cx+cut_w] = \
            other_img[:, cy:cy+cut_h, cx:cx+cut_w]

    # =====================================================
    # 3️⃣ MixUp (same class)
    # =====================================================
    if 0.3 < prob and prob <= 0.4:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        image = lam * image + (1 - lam) * other_img

    # =====================================================
    # 4️⃣ SMOTE-style interpolation (same class)
    # =====================================================
    if 0.4 < prob <= 0.5:
        lam = random.uniform(*smote_alpha_range)
        image = image + lam * (other_img - image)

    return image, label