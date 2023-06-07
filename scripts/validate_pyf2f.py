#!/usr/bin/python3.7
# coding=utf-8
"""
Validating the performance of PyF2F-Ruler by comparing
the obtained distance estimations published by Picco et al (Cell 2017)
with the distances estimations obtained running PyF2F-Ruler with the
same dataset.

We run a Chi-square test using the published set as the expected categoty (E)
and the set obtained with PyF2F-Ruler as the observed category (O).
"""
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

# Load Data for the C-ter
path_c = "../Cell_PyF2F_comparison/C_terminal.csv"
path_n = "../Cell_PyF2F_comparison/N_terminal.csv"

c_data = pd.read_csv(path_c, usecols=["mu_cell", "serr_mu_cell", "mu_py", "serr_mu_py"])
n_data = pd.read_csv(path_n, usecols=["mu_cell", "serr_mu_cell", "mu_py", "serr_mu_py"])

# Compare all the values (C & N ter)
# Observed values (PyF2F-Ruler distances) and Expected values (published dataset)
observed_values = np.concatenate((c_data.mu_py.to_numpy(), n_data.mu_py.to_numpy()), axis=0)
expected_values = np.concatenate((c_data.mu_cell.to_numpy(), n_data.mu_cell.to_numpy()), axis=0)
test = np.array([observed_values, expected_values]).T

# Chis-square test: sum((O - E)^2 / E)
chi_square_stat, p_value = stats.chisquare(test)

# Adjust p-value to avoid multiple comparisons bias
reject_list, corrected_p_vals = multipletests(p_value, method='bonferroni')[:2]

# Print Chi-square test statistic and p value
print(f'\n\nChi-square test statistic is {str(chi_square_stat)}\n')
print("\noriginal p-value\tcorrected p-value\treject?")
for p_val, corr_p_val, reject in zip(p_value, corrected_p_vals, reject_list):
    print(f'\t{str(p_value)}\t{str(corrected_p_vals)}\n')
print("\nDone!\n\n")
sys.exit(0)
# END
