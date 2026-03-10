# ===================== IMPORT =====================
import os
import numpy as np
import pandas as pd
import umap
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from collections import deque
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial import KDTree
from shapely.geometry import MultiPoint

# ===================== SNOWFALL =====================
def bfs_find_empty(x, y, occ, A, B):
    visited = set()
    q = deque([(x, y)])
    while q:
        cx, cy = q.popleft()
        if (cx, cy) not in occ:
            return cx, cy
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = cx+dx, cy+dy
            if 1 <= nx <= A and 1 <= ny <= B and (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))
    return x, y

def snowfall_fast(xp, yp, maxA=120, maxB=120):
    xp_new = xp.copy()
    yp_new = yp.copy()
    occupied = set()
    for i in range(len(xp)):
        x = min(max(1, xp_new[i]), maxA)
        y = min(max(1, yp_new[i]), maxB)
        if (x, y) in occupied:
            x, y = bfs_find_empty(x, y, occupied, maxA, maxB)
        xp_new[i] = x
        yp_new[i] = y
        occupied.add((x, y))
    return xp_new, yp_new, max(xp_new), max(yp_new)

# ===================== CONV PIXEL =====================
def ConvPixel(FVec, xp, yp, A, B, Base=0):
    """Convert feature vector to pixel image with black background"""
    M = np.zeros((A, B), dtype=float)
    
    # Đánh dấu pixel nào có gene
    mask = np.zeros((A, B), dtype=bool)
    
    for j in range(len(FVec)):
        M[xp[j]-1, yp[j]-1] = FVec[j]
        mask[xp[j]-1, yp[j]-1] = True

    # Handle duplicate coordinates
    coords = np.vstack((xp, yp)).T
    _, inv = np.unique(coords, axis=0, return_inverse=True)
    for pid in np.unique(inv):
        idx = np.where(inv == pid)[0]
        if len(idx) > 1:
            M[xp[idx[0]]-1, yp[idx[0]]-1] = np.mean(FVec[idx])

    # ✅ Normalization CHỈ cho pixel có gene
    if mask.any():
        gene_values = M[mask]
        
        # Normalize gene values
        gene_values = gene_values - gene_values.min()
        if gene_values.max() > 0:
            gene_values = gene_values / gene_values.max()
        
        # Log transform
        gene_values = np.log1p(gene_values)
        gene_values = gene_values - gene_values.min()
        if gene_values.max() > 0:
            gene_values = gene_values / gene_values.max()
        
        # Gán lại vào ma trận, pixel không có gene vẫn = 0
        M[mask] = gene_values

    return M

# ===================== BOUNDING RECT =====================
def min_bounding_rect(x, y):
    hull = MultiPoint(np.column_stack([x, y])).convex_hull
    rect = hull.minimum_rotated_rectangle
    coords = np.array(rect.exterior.coords)
    return coords[:,0], coords[:,1]

# ===================== CART2PIXEL =====================
def Cart2Pixel(Q, maxA=120, maxB=120, random_state=42):
    """Map features to pixel coordinates using UMAP"""
    print(f"  UMAP embedding for {Q.shape[0]} features...")
    Y = umap.UMAP(
        n_components=2,
        n_neighbors=min(30, Q.shape[0]-1),
        min_dist=0.3,
        metric="cosine",
        init="spectral",
        random_state=random_state
    ).fit_transform(Q)

    x, y = Y[:,0], Y[:,1]
    xrect, yrect = min_bounding_rect(x, y)

    theta = np.arctan2(yrect[1]-yrect[0], xrect[1]-xrect[0])
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])

    z = R @ np.vstack([x, y])
    zx, zy = z[0], z[1]

    xp = np.round(1 + maxA * (zx - zx.min()) / (zx.max() - zx.min())).astype(int)
    yp = np.round(1 - maxB * (zy - zy.max()) / (zy.max() - zy.min())).astype(int)

    xp, yp, A, B = snowfall_fast(xp, yp, maxA, maxB)
    return xp, yp, A, B

# ===================== RESIZE IMAGE =====================
def resize_to_target(img, target_size=(120, 120)):
    """Resize image to target size using scipy zoom"""
    zoom_factors = (target_size[0] / img.shape[0], 
                    target_size[1] / img.shape[1])
    return zoom(img, zoom_factors, order=1)

# ===================== LOAD DATA =====================
BASE = "/kaggle/input/brca-aligned"

print("Loading data...")
mRNA   = pd.read_csv(f"{BASE}/BRCA_mRNA_aligned.csv", index_col=0)
Methy  = pd.read_csv(f"{BASE}/BRCA_Methy_aligned.csv", index_col=0)
CNV    = pd.read_csv(f"{BASE}/BRCA_CNV_aligned.csv", index_col=0)
labels = pd.read_csv(f"{BASE}/BRCA_label_num.csv").iloc[:,0].values

# ===================== ALIGN SAMPLES =====================
common_samples = mRNA.columns.intersection(Methy.columns).intersection(CNV.columns)
print(f"Common samples: {len(common_samples)}")

mRNA  = mRNA[common_samples]
Methy = Methy[common_samples]
CNV   = CNV[common_samples]

# ✅ FIX: Kiểm tra và align labels
if len(labels) != len(common_samples):
    print(f"⚠️ Warning: Labels ({len(labels)}) != Samples ({len(common_samples)})")
    labels = labels[:len(common_samples)]  # Cắt bớt hoặc xử lý khác
    print(f"  → Truncated labels to {len(labels)}")

Qm  = mRNA.values.astype(np.float32)
Qme = Methy.values.astype(np.float32)
Qc  = CNV.values.astype(np.float32)

print(f"mRNA genes: {Qm.shape[0]}, Methy genes: {Qme.shape[0]}, CNV genes: {Qc.shape[0]}")

# ===================== PIXEL MAPPING =====================
TARGET_SIZE = 224

print("\n[1/3] Processing mRNA...")
xp_m, yp_m, A_m, B_m = Cart2Pixel(Qm, maxA=TARGET_SIZE, maxB=TARGET_SIZE)
print(f"  → Image size: {A_m} x {B_m}")

print("\n[2/3] Processing Methylation...")
xp_me, yp_me, A_me, B_me = Cart2Pixel(Qme, maxA=TARGET_SIZE, maxB=TARGET_SIZE)
print(f"  → Image size: {A_me} x {B_me}")

print("\n[3/3] Processing CNV...")
xp_c, yp_c, A_c, B_c = Cart2Pixel(Qc, maxA=TARGET_SIZE, maxB=TARGET_SIZE)
print(f"  → Image size: {A_c} x {B_c}")

# ===================== PREPARE OUTPUT =====================
OUT = "/kaggle/working/OutputData"
idx = np.arange(Qm.shape[1])

# Tạo thư mục cho từng class
for c in np.unique(labels):
    os.makedirs(f"{OUT}/dataset/{c}", exist_ok=True)
    print(f"  Created folder: dataset/{c}")

# ===================== SAVE IMAGES =====================
def save_split(split, indices, y):
    """Generate and save multi-omics images"""
    print(f"\nGenerating {split} set...")
    for i, (idx, lbl) in enumerate(zip(indices, y)):
        img_m  = ConvPixel(Qm[:,idx],  xp_m,  yp_m,  A_m,  B_m)
        img_me = ConvPixel(Qme[:,idx], xp_me, yp_me, A_me, B_me)
        img_c  = ConvPixel(Qc[:,idx],  xp_c,  yp_c,  A_c,  B_c)
        
        img_m  = resize_to_target(img_m,  (TARGET_SIZE, TARGET_SIZE))
        img_me = resize_to_target(img_me, (TARGET_SIZE, TARGET_SIZE))
        img_c  = resize_to_target(img_c,  (TARGET_SIZE, TARGET_SIZE))
        
        img = np.stack([img_m, img_me, img_c], axis=-1)
        
        imageio.imwrite(
            f"{OUT}/{split}/{lbl}/sample_{idx}.png",
            (img * 255).astype(np.uint8)
        )
        
        if (i + 1) % 50 == 0:
            print(f"  {split}: {i+1}/{len(indices)}")

save_split("dataset", idx, labels)

# ===================== VISUALIZATION =====================
# ✅ FIX: Chỉ tạo 1 ảnh visualization mẫu
print("\nGenerating sample visualization...")
sample_idx = idx[0]  # Lấy sample đầu tiên

img_m  = resize_to_target(ConvPixel(Qm[:,sample_idx],  xp_m,  yp_m,  A_m,  B_m),  (TARGET_SIZE, TARGET_SIZE))
img_me = resize_to_target(ConvPixel(Qme[:,sample_idx], xp_me, yp_me, A_me, B_me), (TARGET_SIZE, TARGET_SIZE))
img_c  = resize_to_target(ConvPixel(Qc[:,sample_idx],  xp_c,  yp_c,  A_c,  B_c),  (TARGET_SIZE, TARGET_SIZE))

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(img_m, cmap='Reds')
axes[0].set_title('mRNA (Red)')
axes[0].axis('off')

axes[1].imshow(img_me, cmap='Greens')
axes[1].set_title('Methylation (Green)')
axes[1].axis('off')

axes[2].imshow(img_c, cmap='Blues')
axes[2].set_title('CNV (Blue)')
axes[2].axis('off')

img_rgb = np.stack([img_m, img_me, img_c], axis=-1)
axes[3].imshow(img_rgb)
axes[3].set_title('Combined RGB')
axes[3].axis('off')

plt.tight_layout()
plt.savefig(f"{OUT}/sample_visualization_label{labels[sample_idx]}.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ DONE: Images saved to '{OUT}/dataset/'")
print(f"📊 Visualization saved: {OUT}/sample_visualization_label{labels[sample_idx]}.png")