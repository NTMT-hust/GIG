import os
import glob
import pandas as pd

# =============================
# CONFIG
# =============================
BASE_DIR = "C:\\Users\\Admin\\GIG_CSV"   # đổi thành path của bạn
K = 400
OUTPUT_DIR = "selected_genes_400"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# 1. ĐỌC TOÀN BỘ SAMPLE 5 FOLD
# =============================
all_dfs = []

fold_paths = sorted(glob.glob(os.path.join(BASE_DIR, "Fold_*")))

for fold_path in fold_paths:
    
    sample_files = glob.glob(os.path.join(fold_path, "*.csv"))
    
    for file in sample_files:
        df = pd.read_csv(file)
        all_dfs.append(df)

# Gộp tất cả
full_df = pd.concat(all_dfs, ignore_index=True)

print("Tổng số dòng:", len(full_df))

# =============================
# 2. CHECK CỘT
# =============================
required_cols = ["gene_name", "predicted_label", "omics_type", "attribute_score"]

for col in required_cols:
    if col not in full_df.columns:
        raise ValueError(f"Thiếu cột: {col}")

# =============================
# 3. MEAN THEO predicted_label + omics_type + gene_name
# =============================
mean_df = (
    full_df
    .groupby(["predicted_label", "omics_type", "gene_name"])["attribute_score"]
    .median()
    .reset_index()
)

# =============================
# 4. XỬ LÝ THEO TỪNG predicted_label
# =============================
predicted_labeles = sorted(mean_df["predicted_label"].unique())

for cls in predicted_labeles:
    
    print(f"\nĐang xử lý predicted_label {cls}")
    
    predicted_label_df = mean_df[mean_df["predicted_label"] == cls]
    
    positive_list = []
    negative_list = []
    
    # =============================
    # XỬ LÝ THEO TỪNG omics_type
    # =============================
    for omics_type_type in predicted_label_df["omics_type"].unique():
        
        omics_type_df = predicted_label_df[predicted_label_df["omics_type"] == omics_type_type]
        
        # Top K positive
        top_pos = (
            omics_type_df[omics_type_df["attribute_score"] > 0]
            .sort_values("attribute_score", ascending=False)
            .head(K)
        )
        
        # Top K negative
        top_neg = (
            omics_type_df[omics_type_df["attribute_score"] < 0]
            .sort_values("attribute_score", ascending=True)
            .head(K)
        )
        
        positive_list.append(top_pos)
        negative_list.append(top_neg)
    
    # Gộp tất cả omics_type của predicted_label này
    final_positive = pd.concat(positive_list)
    final_negative = pd.concat(negative_list)
    
    # =============================
    # LƯU FILE
    # =============================
    pos_path = os.path.join(
        OUTPUT_DIR,
        f"predicted_label_{cls}_top_{K}_positive.csv"
    )
    
    neg_path = os.path.join(
        OUTPUT_DIR,
        f"predicted_label_{cls}_top_{K}_negative.csv"
    )
    
    final_positive.to_csv(pos_path, index=False)
    final_negative.to_csv(neg_path, index=False)
    
    print(f"✓ Saved predicted_label {cls}")

print("\n🎉 Hoàn thành. Đã tạo 10 file.")