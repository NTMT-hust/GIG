import pandas as pd

pam50_genes = {
    "ACTR3B","ANLN","BAG1","BCL2","BIRC5","BLVRA","CCNB1","CCNE1",
    "CDC20","CDC6","NUF2","CDH3","CENPF","CEP55","CXXC5","EGFR",
    "ERBB2","ESR1","EXO1","FGFR4","FOXA1","FOXC1","GPR160","GRB7",
    "KIF2C","NDC80","KRT14","KRT17","KRT5","MAPT","MDM2","MELK",
    "MIA","MKI67","MLPH","MMP11","MYBL2","MYC","NAT1","ORC6",
    "PGR","PHGDH","PTTG1","RRM2","SFRP1","SLC39A6","TMEM45B",
    "TYMS","UBE2C","UBE2T"
}

# Load your file
df = pd.read_csv("G:\My Drive\\bioinfor_training\\28729127\MLOmics\Main_Dataset\Classification_datasets\GS-BRCA\Top\BRCA_mRNA_top.csv")   # change filename if needed

# Unique genes in sample
sample_genes = set(df["value"].str.upper())

# Intersection
pam50_found = sample_genes.intersection(pam50_genes)

# Percentage
percentage = len(pam50_found) / len(pam50_genes) * 100

print(f"PAM50 genes found: {len(pam50_found)}/50")
print(f"Coverage: {percentage:.2f}%")
print("Matched genes:", sorted(pam50_found))
