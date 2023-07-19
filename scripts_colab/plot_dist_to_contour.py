import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = "../Cell_PyF2F_comparison/F9/output/results/"

fig, ax = plt.subplots(figsize=(25,15))
sns.despine()
plt.grid(False)
sns.set_style("white")
sel = np.loadtxt(path + "sel_distances_to_contour.csv")
non_sel = np.loadtxt(path + "non-sel_distances_to_contour.csv")
sns.histplot(non_sel, kde=True, color="tab:grey", ax=ax, fill=True, stat="density", label="non-sel", binwidth=2)
sns.histplot(sel, kde=True, color="mediumturquoise", ax=ax, fill=True, label="sel", stat="density", binwidth=2)
ax.set(xlim=(0, 100))
plt.savefig(path + "dist_contour.png", dpi=120)