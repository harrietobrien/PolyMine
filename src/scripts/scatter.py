import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

master = pd.read_csv("data/processed/master_with_energies.csv")
assign = pd.read_csv("data/processed/polymorph_clusters_assignments.csv")

df = master.merge(assign[["cod_id", "cluster_id"]], on="cod_id", how="inner")
f = "PrNiO3"
df_f = df[df["formula"] == f].copy()

struct_feats = ["volume","density","a","b","c","alpha","beta","gamma","n_sites","n_species"]
X = df_f[struct_feats].values
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(5,4))
sc = plt.scatter(X_pca[:,0], X_pca[:,1],
                 c=df_f["cluster_id"], cmap="tab10", s=60, edgecolors="k")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"{f} polymorph clusters (structural PCA)")
plt.colorbar(sc, label="cluster id")
plt.tight_layout()
plt.show()
