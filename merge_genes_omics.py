import pandas as pd
import glob
import os

# ── 1. ĐỌC TẤT CẢ FILE ──────────────────────────────────────────────────────
FOLDER = "C:\\Users\\Admin\\selected_genes_400"
OUTPUT_DIR = "Top"          # <-- đổi thành đường dẫn folder chứa các file CSV nếu cần
pattern = os.path.join(FOLDER, "predicted_label_*_top_400_*.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(glob.glob(pattern))
print(f"✅ Tìm thấy {len(files)} file:")
for f in files:
    print(f"   {os.path.basename(f)}")

dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs, ignore_index=True)
print(f"\n📦 Tổng số dòng gốc (toàn bộ file): {len(combined):,}")

# ── 2. TẠO FILE 2 CỘT: unique_genes + omics_type ────────────────────────────
unique_df = (
    combined[["gene_name", "omics_type"]]
    .drop_duplicates()
    .rename(columns={"gene_name": "unique_genes"})
    .sort_values(["omics_type", "unique_genes"])
    .reset_index(drop=True)
)

output_path = os.path.join(OUTPUT_DIR, "unique_genes_omics.csv")
unique_df.to_csv(output_path, index=False)
print(f"\n💾 Đã lưu file: {output_path}")

# ── 3. THỐNG KÊ ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("📊 THỐNG KÊ TỔNG QUAN")
print("=" * 55)

total_pairs   = len(unique_df)
total_genes   = unique_df["unique_genes"].nunique()
total_omics   = unique_df["omics_type"].nunique()

print(f"  Tổng cặp (gene, omics_type) unique : {total_pairs:,}")
print(f"  Tổng gene unique                   : {total_genes:,}")
print(f"  Số loại omics_type                 : {total_omics}")

# ── 3a. Phân bổ theo omics_type ──
print("\n" + "-" * 55)
print("🧬 Phân bổ theo omics_type")
print("-" * 55)
omics_count = unique_df["omics_type"].value_counts()
for omics, cnt in omics_count.items():
    pct = cnt / total_pairs * 100
    print(f"  {omics:<15} : {cnt:>5,}  ({pct:.1f}%)")

# ── 3b. Gene xuất hiện ở nhiều omics ──
print("\n" + "-" * 55)
print("🔁 Gene xuất hiện ở nhiều omics_type")
print("-" * 55)
gene_omics_count = unique_df.groupby("unique_genes")["omics_type"].count()

for n in range(1, total_omics + 1):
    cnt = (gene_omics_count == n).sum()
    label = "omics" if n > 1 else "omics "
    print(f"  {n} {label}  : {cnt:>5,} gene")

genes_all = gene_omics_count[gene_omics_count == total_omics].index.tolist()
print(f"\n  ⭐ Gene có mặt ở cả {total_omics} omics ({len(genes_all)} gene):")
print("    " + ", ".join(genes_all))

# ── 3c. Unique gene per predicted_label x omics_type ──
print("\n" + "-" * 55)
print("📌 Unique gene theo predicted_label × omics_type")
print("-" * 55)
pivot = (
    combined.groupby(["predicted_label", "omics_type"])["gene_name"]
    .nunique()
    .unstack()
)
print(pivot.to_string())

# ── 3d. Unique gene per label (tổng hợp) ──
print("\n" + "-" * 55)
print("📌 Unique gene per predicted_label (gộp tất cả omics)")
print("-" * 55)
label_unique = combined.groupby("predicted_label")["gene_name"].nunique()
for label, cnt in label_unique.items():
    print(f"  Label {label} : {cnt:,} gene")

# ── 3e. Top 100 gene xuất hiện nhiều nhất ──
print("\n" + "-" * 55)
print("🏆 Top 100 gene xuất hiện nhiều nhất")
print("-" * 55)
top100 = combined["gene_name"].value_counts().head(100).reset_index()
top100.columns = ["gene_name", "count"]
for i, row in top100.iterrows():
    print(f"  {i+1:>3}. {row['gene_name']:<20} : {row['count']} lần")

# ── 4. GENE CÓ TRONG PAM50 ───────────────────────────────────────────────────
PAM50 = {
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA",
    "CCNB1", "CCNE1", "CDC20", "CDC6", "CDH3", "CENPF",
    "CEP55", "CXXC5", "DCN", "EGFR", "ERBB2", "ESR1",
    "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7",
    "KIF2C", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2",
    "MELK", "MIA", "MKI67", "MLPH", "MMP11", "MYBL2",
    "MYC", "NAT1", "NDC80", "NUF2", "ORC6", "PGR",
    "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6",
    "TMEM45B", "TYMS", "UBE2C", "UBE2T"
}

print("\n" + "=" * 55)
print("🎯 GENE CÓ TRONG DANH SÁCH PAM50 (51 gene)")
print("=" * 55)

all_genes_in_data = set(combined["gene_name"].unique())
pam50_found = PAM50 & all_genes_in_data
pam50_missing = PAM50 - all_genes_in_data

print(f"\n  ✅ Tìm thấy trong data : {len(pam50_found)}/{len(PAM50)} gene")
print(f"  ❌ Không có trong data : {len(pam50_missing)}/{len(PAM50)} gene")

# Chi tiết từng gene PAM50 tìm thấy
print("\n" + "-" * 55)
print("  Chi tiết gene PAM50 có trong data:")
print(f"  {'Gene':<15} {'Omics types':<30} {'Số lần xuất hiện':<10} {'Labels'}")
print("  " + "-" * 70)

for gene in sorted(pam50_found):
    sub = combined[combined["gene_name"] == gene]
    omics   = ", ".join(sorted(sub["omics_type"].unique()))
    count   = len(sub)
    labels  = ", ".join(str(l) for l in sorted(sub["predicted_label"].unique()))
    print(f"  {gene:<15} {omics:<30} {count:<18} {labels}")

# Gene PAM50 không tìm thấy
if pam50_missing:
    print("\n" + "-" * 55)
    print("  Gene PAM50 KHÔNG có trong data:")
    print("  " + ", ".join(sorted(pam50_missing)))

# PAM50 trong top 100
pam50_in_top100 = set(top100["gene_name"]) & PAM50
print("\n" + "-" * 55)
print(f"  ⭐ Gene PAM50 nằm trong Top 100 ({len(pam50_in_top100)} gene):")
top100_pam50 = top100[top100["gene_name"].isin(PAM50)]
for _, row in top100_pam50.iterrows():
    print(f"     {row['gene_name']:<15} : {row['count']} lần")

# PAM50 per label
print("\n" + "-" * 55)
print("  Gene PAM50 theo predicted_label:")
pam50_label = (
    combined[combined["gene_name"].isin(PAM50)]
    .groupby("predicted_label")["gene_name"]
    .nunique()
)
for label, cnt in pam50_label.items():
    genes_in_label = sorted(
        combined[(combined["gene_name"].isin(PAM50)) & (combined["predicted_label"] == label)]["gene_name"].unique()
    )
    print(f"  Label {label} : {cnt} gene  →  {', '.join(genes_in_label)}")

print("\n✅ Hoàn tất!")
