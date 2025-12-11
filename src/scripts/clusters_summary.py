import pandas as pd

summary = pd.read_csv("data/processed/polymorph_clusters_summary.csv")

# How many clusters per formula?
cluster_counts = summary.groupby("formula")["cluster_id"].nunique().reset_index(name="n_clusters")
print(cluster_counts.sort_values("n_clusters", ascending=False))

# Focus only on formulas that actually split into >1 cluster
multi = cluster_counts[cluster_counts["n_clusters"] > 1]["formula"]
summary_multi = summary[summary["formula"].isin(multi)]

# Look at energy gaps: which formulas have interesting ΔE?
cols = ["formula", "cluster_id", "n_members",
        "mean_e_form", "delta_mean_e_form_to_min",
        "mean_e_hull", "delta_mean_e_hull_to_min"]
print(summary_multi[cols].sort_values(["formula", "delta_mean_e_form_to_min"]))
