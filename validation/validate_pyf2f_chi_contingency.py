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
import scipy.stats as stats

# Load Data from Exocyst Subunits C/N termini
path_c = "../Cell_PyF2F_comparison/C_terminal.csv"
path_n = "../Cell_PyF2F_comparison/N_terminal.csv"
c_data_observed = pd.read_csv(path_c, usecols=["mu_py", "serr_mu_py"])
n_data_observed = pd.read_csv(path_n, usecols=["mu_py", "serr_mu_py"])
c_data_expected = pd.read_csv(path_c, usecols=["mu_cell", "serr_mu_cell"])
n_data_expected = pd.read_csv(path_n, usecols=["mu_cell", "serr_mu_cell"])

# Compare all the values (C & N ter)
# Observed values (PyF2F-Ruler distances) and
# Expected values (published dataset, Picco et al., Cell, 2017)
observed_values_c = c_data_observed.mu_py.tolist()
expected_values_c = c_data_expected.mu_cell.tolist() 
observed_values_n = n_data_observed.mu_py.tolist()
expected_values_n = n_data_expected.mu_cell.tolist()

# Run two separate tests, one for the C-terminal group 
# and one for the N-terminal group

#-------#
# C-TER #
#-------#
print("\n#-------#\n# C-TER #\n#-------#\n")
data_c = np.array([observed_values_c, expected_values_c])

# Do Chi2 test for the independence
stat_c, p_value_c, dof_c, expected_c = stats.chi2_contingency(data_c)
alpha = 0.05   # confidence interval
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
alpha = 0.05   # confidence interval
print(f"critical value for {dof_n} dof is {stat_n}\n")
print("p value is " + str(p_value_n))
if p_value_n < alpha:
	print("The two N-TER datasets are different (Reject H0)")
else:
	print("The two N-TER datasets are not different (Cannot reject H0)")
	print(f"\n\tThe probability of independence between the two datasets is {1 - p_value_n}")


print("\nDone!\n")
sys.exit(0)

# END
