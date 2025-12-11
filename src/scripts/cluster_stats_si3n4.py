import pandas as pd
from pathlib import Path

MASTER = Path("data/processed/master_with_energies_1000.csv")
ASSIGN = Path("data/processed/polymorph_clusters_assignments_1000.csv")

master = pd.read_csv(MASTER)
assign = pd.read_csv(ASSIGN)

df = master.merge(assign[["cod_id", "cluster_id"]], on="cod_id", how="left")

# Filter to Si3N4 only
si = df[df["formula"] == "Si3N4"].copy()
si = si.dropna(subset=["cluster_id"])

print("[INFO] Si3N4 rows with clusters:", len(si))
print("[INFO] clusters:", sorted(si["cluster_id"].unique()))

# Group by cluster and compute stats
lines = []
for cid, sub in si.groupby("cluster_id"):
    n = len(sub)
    v_mean = sub["volume"].mean()
    v_std = sub["volume"].std(ddof=0)
    rho_mean = sub["density"].mean()
    rho_std = sub["density"].std(ddof=0)

    line = (
        f"{int(cid)} & {n} & "
        f"{v_mean:6.2f} & {v_std:5.2f} & "
        f"{rho_mean:5.3f} & {rho_std:5.3f} \\\\"
    )
    lines.append(line)

print("\nLaTeX rows:")
for line in lines:
    print(line)
