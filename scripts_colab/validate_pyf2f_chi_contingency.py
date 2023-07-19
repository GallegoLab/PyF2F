#!/usr/bin/python3.7
# coding=utf-8
"""
Validating the performance of PyF2F-Ruler by comparing
the obtained distance estimations published by Picco et al (Cell 2017)
with the distances estimations obtained running PyF2F-Ruler with the
same dataset.

We run a Chi-square test using the published set as the expected categoty (E)
and the set obtained with PyF2F-Ruler as the observed category (O).

H0: Both groups are dependent, and therefore, are significantly related (similar).
H1: Both groups are independent, and therefore, are not significantly related (different).
"""
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load Data from Exocyst Subunits C/N termini
path_c = "../Cell_PyF2F_comparison/C_terminal.csv"
path_n = "../Cell_PyF2F_comparison/N_terminal.csv"
path_sec2 = "../Cell_PyF2F_comparison/Sec2.csv"

c_data_observed = pd.read_csv(path_c, usecols=["mu_py", "serr_mu_py"])
n_data_observed = pd.read_csv(path_n, usecols=["mu_py", "serr_mu_py"])
c_data_expected = pd.read_csv(path_c, usecols=["mu_cell", "serr_mu_cell"])
n_data_expected = pd.read_csv(path_n, usecols=["mu_cell", "serr_mu_cell"])
sec2_data_observed = pd.read_csv(path_sec2, usecols=["mu_py", "serr_mu_py"])
sec2_data_expected = pd.read_csv(path_sec2, usecols=["mu_cell", "serr_mu_cell"])

# Compare all the exocyst assembly measurements (C & N ter) and 
# the inter-assembly measurements (Sec2). 
# Observed values (PyF2F-Ruler distances) and
# Expected values (published dataset, Picco et al., Cell, 2017)
observed_values_c = c_data_observed.mu_py.tolist()
expected_values_c = c_data_expected.mu_cell.tolist() 
observed_values_n = n_data_observed.mu_py.tolist()
expected_values_n = n_data_expected.mu_cell.tolist()
observed_values_sec2 = sec2_data_observed.mu_py.tolist()
expected_values_sec2 = sec2_data_expected.mu_cell.tolist()
observed_values_all = np.concatenate((observed_values_c, observed_values_n), axis=0)
expected_values_all = np.concatenate((expected_values_c, expected_values_n), axis=0)

###################
# Linear Fit TEST
###################
result = stats.linregress(observed_values_c + observed_values_n + observed_values_sec2, 
	expected_values_c  + expected_values_n + expected_values_sec2)

print(f"Linear Regression Differences:\n")
print(f"\tSlope: {result.slope:.6f}")
print(f"\tIntercept: {result.intercept:.6f}")
print(f"\tR-squared: {result.rvalue**2:.6f}")
print(f"\tp-value: {result.pvalue:.6f}")
print(f"\tSTDerr: {result.stderr:.6f}")

# Plot Linear Fit of the differences
sns.set(font_scale=3)
sns.set_style("white")
fig, ax = plt.subplots(figsize=(40, 15))
ax.plot(observed_values_c, expected_values_c, 'x', markerfacecolor='tab:blue', label='C-ter-GFP-tagged', markersize=35)
ax.plot(expected_values_sec2, observed_values_sec2, '*', markerfacecolor='tab:blue', markeredgecolor='tab:blue', label='Sec2-GFP-tagged', markersize=35)
ax.plot(observed_values_n, expected_values_n, '^', markerfacecolor='tab:blue', markeredgecolor='tab:blue', label='N-ter-GFP-tagged', markersize=35)
ax.plot(expected_values_c + expected_values_n + expected_values_sec2, 
	result.intercept + result.slope*np.array(expected_values_c + expected_values_n + expected_values_sec2), 'r',
        label=f'Fit\nSlope: {result.slope:.6f}\nR-squared: {result.rvalue**2:.6f}\np-value: {result.pvalue**2:.6f}\n')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax.set(xlabel="Expected Distances")
ax.set(ylabel="Observed Distances")
sns.despine()
plt.grid(False)
ax.plot([x for x in range(0, 100)], 
	[result.intercept + y * result.slope  for y in range(0, 100)], 'r--', alpha=0.5)
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
plt.tight_layout()
plt.savefig("../Cell_PyF2F_comparison/linear_fit.png", dpi=100)
plt.clf()

##############
# Chi2 TEST
##############

# Run two separate tests, one for the C-terminal group 
# and one for the N-terminal group
alpha = 0.05   # confidence interval
#-------#
# C-TER #
#-------#
print("\n#-------#\n# C-TER #\n#-------#\n")
data_c = np.array([observed_values_c, expected_values_c])

# Do Chi2 test for the independence
stat_c, p_value_c, dof_c, expected_c = stats.chi2_contingency(data_c)
print(f"critical value for {dof_c} dof is {stat_c}\n")
print("Test p value is " + str(p_value_c))
if p_value_c < alpha:
	print("The two C-TER datasets are different (Reject H0)")
else:
	print("The two C-TER datasets are not different (Cannot reject H0)")
	print(f"\n\tThe probability of independence between the two datasets is {1 - p_value_c}")

#-------#
# N-TER # 
#-------#
print("\n#-------#\n# N-TER #\n#-------#\n")
data_n = np.array([observed_values_n, expected_values_n])

# Do Chi2 test for the independence
stat_n, p_value_n, dof_n, expected_n = stats.chi2_contingency(data_n)
print(f"critical value for {dof_n} dof is {stat_n}\n")
print("p value is " + str(p_value_n))
if p_value_n < alpha:
	print("The two N-TER datasets are different (Reject H0)")
else:
	print("The two N-TER datasets are not different (Cannot reject H0)")
	print(f"\n\tThe probability of independence between the two datasets is {1 - p_value_n}")

#-------#
# SEC-2 # 
#-------#
print("\n#-------#\n# Sec2 #\n#-------#\n")
data_sec2 = np.array([observed_values_sec2, expected_values_sec2])

# Do Chi2 test for the independence
stat_sec2, p_value_sec2, dof_sec2, expected_sec2 = stats.chi2_contingency(data_sec2)
print(f"critical value for {dof_sec2} dof is {stat_sec2}\n")
print("p value is " + str(p_value_sec2))
if p_value_sec2 < alpha:
	print("The two SEC-2 datasets are different (Reject H0)")
else:
	print("The two SEC-2 datasets are not different (Cannot reject H0)")
	print(f"\n\tThe probability of independence between the two datasets is {1 - p_value_sec2}")


print("\nDone!\n")
sys.exit(0)

# END
