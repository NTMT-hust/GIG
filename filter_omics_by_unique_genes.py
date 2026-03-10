import pandas as pd
import os

# ══════════════════════════════════════════════════════════════════
# ⚙️  CẤU HÌNH — chỉnh sửa các đường dẫn tại đây
# ══════════════════════════════════════════════════════════════════
OMICS_FILES = {
    "CNV"        : "C:\\Users\\Admin\\Documents\\BioInformatics\\MLOmics\\Main_Dataset\\Classification_datasets\\GS-BRCA\\Aligned\\BRCA_CNV_aligned.csv",        # <-- đổi tên file
    "mRNA"       : "C:\\Users\\Admin\\Documents\\BioInformatics\\MLOmics\\Main_Dataset\\Classification_datasets\\GS-BRCA\\Aligned\\BRCA_mRNA_aligned.csv",        # <-- đổi tên file
    "Methylation": "C:\\Users\\Admin\\Documents\\BioInformatics\\MLOmics\\Main_Dataset\\Classification_datasets\\GS-BRCA\\Aligned\\BRCA_Methy_aligned.csv",       # <-- đổi tên file (hiện chỉ có file này)
}

UNIQUE_GENES_FILE = "C:/Users/Admin/Top/unique_genes_omics.csv"   # <-- đường dẫn file unique_genes_omics
OUTPUT_DIR        = "C:/Users/Admin/Top/"                        # <-- thư mục lưu file output
# ══════════════════════════════════════════════════════════════════

# ── 1. Đọc danh sách gene unique ────────────────────────────────
print("📂 Đọc file unique_genes_omics...")
unique_df = pd.read_csv(UNIQUE_GENES_FILE)
print(f"   Tổng cặp (gene, omics_type): {len(unique_df):,}")
print(f"   Tổng gene unique: {unique_df['unique_genes'].nunique():,}")

# ── 2. Xử lý từng file omics ────────────────────────────────────
print("\n" + "=" * 60)
for omics_name, filepath in OMICS_FILES.items():

    if not os.path.exists(filepath):
        print(f"\n⚠️  [{omics_name}] Không tìm thấy file: {filepath} — BỎ QUA")
        continue

    print(f"\n🧬 [{omics_name}] Đọc file: {filepath}")
    df = pd.read_csv(filepath, index_col=0)

    print(f"   Kích thước gốc : {df.shape[0]:,} gene × {df.shape[1]:,} mẫu")

    # Lấy danh sách gene tương ứng omics_type
    genes_for_omics = set(
        unique_df.loc[unique_df["omics_type"] == omics_name, "unique_genes"]
    )
    print(f"   Gene unique trong omics [{omics_name}]: {len(genes_for_omics):,}")

    # Lọc hàng
    mask        = df.index.isin(genes_for_omics)
    df_filtered = df[mask]

    n_found   = df_filtered.shape[0]
    n_missing = len(genes_for_omics) - n_found
    print(f"   ✅ Giữ lại : {n_found:,} gene  |  ❌ Không khớp: {n_missing:,} gene")

    if n_missing > 0:
        missing_genes = genes_for_omics - set(df.index)
        print(f"   Gene trong unique list nhưng không có trong file ({min(n_missing,10)} ví dụ đầu):")
        print("   " + ", ".join(sorted(missing_genes)[:10]))

    # Lưu file
    base_name   = os.path.splitext(os.path.basename(filepath))[0]
    out_path    = os.path.join(OUTPUT_DIR, f"{base_name}_filtered.csv")
    df_filtered.to_csv(out_path)
    print(f"   💾 Đã lưu  : {out_path}")

print("\n" + "=" * 60)
print("✅ Hoàn tất!")
