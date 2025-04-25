# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

# %%
norm_list1 = stats.norm.rvs(loc=10, scale=1, size=5)
delta = stats.norm.rvs(loc=1.5, scale=1, size=5)
norm_list2 = norm_list1 + delta

print(f"diff of means: {np.mean(norm_list2) - np.mean(norm_list1)}")

ttest, pvalue = stats.ttest_rel(norm_list1, norm_list2)
print(f"rel pvalue = {pvalue}")
ttest, pvalue = stats.ttest_ind(norm_list1, norm_list2)
print(f"ind pvalue = {pvalue}")
wilcoxon, pvalue = stats.wilcoxon(norm_list1, norm_list2)
print(f"wilcoxon = {wilcoxon}, pvalue = {pvalue}")

fig, ax = plt.subplots(figsize=(10, 2))
sns.heatmap(
    np.array([norm_list1, norm_list2]),
    cmap="PuBuGn",
    annot=True,
    fmt=".2f",
    ax=ax,
    yticklabels=["Control", "Treatment"],
    cbar=False,
)
ax.set_xlabel("Index")
ax.set_ylabel("Group")
plt.show()
