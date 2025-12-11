import matplotlib.pyplot as plt

elements = ["Ca", "Mg", "Si", "O"]
fractions = [0.125, 0.125, 0.25, 0.5]  # e.g. Ca1 Mg1 Si2 O6 → 1/8, 1/8, 2/8, 6/8

fig, ax = plt.subplots(figsize=(4.0, 3.0))  # small enough for one column

ax.bar(elements, fractions)
ax.set_ylim(0, 0.6)

ax.set_ylabel("Atomic fraction")
ax.set_title(r"Compositional fingerprint for $\mathrm{CaMg(SiO_3)_2}$")

for x, y in zip(elements, fractions):
    ax.text(x, y + 0.01, f"{y:.3f}", ha="center", va="bottom", fontsize=8)

fig.tight_layout()
fig.savefig("figures/composition_fingerprint_CaMgSiO32.png", dpi=300)
plt.close(fig)
