import pandas as pd
import gseapy as gp

# ==============================
# 1. Load gene list
# ==============================
file_path = "C:\\Users\\Admin\\Top\\unique_genes_omics.csv"

df = pd.read_csv(file_path)

# lấy danh sách gene unique
genes = df["unique_genes"].dropna().unique().tolist()

print("Total genes:", len(genes))


# ==============================
# 2. Run enrichment (MSigDB Hallmark)
# ==============================
enr = gp.enrichr(
    gene_list=genes,
    gene_sets="MSigDB_Hallmark_2020",
    organism="Human",
    outdir=None
)

results = enr.results


# ==============================
# 3. Đếm overlap gene
# ==============================

results["overlap_gene_count"] = results["Overlap"].apply(
    lambda x: int(x.split("/")[0])
)

results["total_pathway_genes"] = results["Overlap"].apply(
    lambda x: int(x.split("/")[1])
)


# ==============================
# 4. Lấy gene overlap
# ==============================

results["gene_list"] = results["Genes"]


# ==============================
# 5. Chọn các cột quan trọng
# ==============================

final = results[
    [
        "Term",
        "overlap_gene_count",
        "total_pathway_genes",
        "Adjusted P-value",
        "gene_list"
    ]
].sort_values("overlap_gene_count", ascending=False)


# ==============================
# 6. Lưu kết quả
# ==============================

final.to_csv("hallmark_pathway_overlap.csv", index=False)

print("Saved to hallmark_pathway_overlap.csv")


# ==============================
# 7. In top pathways
# ==============================

print("\nTop pathways by overlap:")
print(final.head(20))