# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 11:01:41 2019

This script takes the topic distribution data and changes in financial asset
prices, and then uses the top 'N' coefficients from the EN and bootstrap exercise
(i.e. top N on bootrstrap conditional on them having a non 0 coeff in EN) to 
forecast via the financial variables.

It does it for the skew on q and a, and the standard deviation on ir.

It saves the actual change in the variables, and the fitted values in the appropriate
output folder. Either of which, depending on interpretations, could be used
as a narrative "shock".

@author: Tim
"""

# Initial imports
import numpy as np
import pandas as pd
from sklearn import preprocessing
import os

# -----------------------------------------------------------------------------
# Primitives
# -----------------------------------------------------------------------------

# Set number of coefficients you wish to use
    # Note that for IR, the total number selected is 11 by EN so this is a good choice
N = 11

directory = 'C:/Users/tmund/Documents/Submissions/HigherMomentsRep/'
os.chdir(directory)

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------

# financial data (purged)
m12df = pd.read_pickle('Output/2 - PurgedFinancial/m12df_chg_purged.pkl')

# coefficients
stdev_ircoeffs = np.load('Output/4 - VariableSelection/1 - Permutation/Together/k_30/QuerySeparate/stdev_ircoeffs.npy')
skew_qacoeffs = np.load('Output/4 - VariableSelection/1 - Permutation/Together/k_30/QuerySeparate/skew_qacoeffs.npy')

# topic distributions
irdf_agg_dt_diff_purged = pd.read_pickle('Output/3 - PurgedLDA/Together/k_30/QuerySeparate/irdf_agg_dt_diff_purged.pkl')
irdf_agg_dt_purged = pd.read_pickle('Output/3 - PurgedLDA/Together/k_30/QuerySeparate/irdf_agg_dt_purged.pkl')

qadf_agg_dt_diff_purged = pd.read_pickle('Output/3 - PurgedLDA/Together/k_30/QuerySeparate/qadf_agg_dt_diff_purged.pkl')
qadf_agg_dt_purged = pd.read_pickle('Output/3 - PurgedLDA/Together/k_30/QuerySeparate/qadf_agg_dt_purged.pkl')

# bootstrap percentages
stdevirpercs = np.load('Output/4 - VariableSelection/2 - Bootstrap/Together/k_30/QuerySeparate/Stdevresid_irpercs.npy')
skewqapercs = np.load('Output/4 - VariableSelection/2 - Bootstrap/Together/k_30/QuerySeparate/Skewresid_qapercs.npy')

# -----------------------------------------------------------------------------
# Preprocess data
# -----------------------------------------------------------------------------


irdf_agg_dt_diff_purged = irdf_agg_dt_diff_purged.add_suffix('_chg')
qadf_agg_dt_diff_purged = qadf_agg_dt_diff_purged.add_suffix('_chg')

irdf = pd.concat([irdf_agg_dt_diff_purged, irdf_agg_dt_purged], axis=1)
qadf = pd.concat([qadf_agg_dt_diff_purged, qadf_agg_dt_purged], axis=1)

irdfcomb = pd.concat([m12df, irdf], axis=1)
irdfcomb.dropna(axis=0, how='any', inplace=True)

qadfcomb = pd.concat([m12df, qadf], axis=1)
qadfcomb.dropna(axis=0, how='any', inplace=True)

m12df_nan_ir = irdfcomb.loc[:, m12df.columns].copy()
irdf_nan = irdfcomb.loc[:, irdf.columns].copy()

m12df_nan_qa = qadfcomb.loc[:, m12df.columns].copy()
qadf_nan = qadfcomb.loc[:, qadf.columns].copy()

irdf_scaled = preprocessing.scale(irdf_nan)
qadf_scaled = preprocessing.scale(qadf_nan)

# -----------------------------------------------------------------------------
# Ordering the data based on %
# -----------------------------------------------------------------------------

skew_array = np.array([skewqapercs, skew_qacoeffs])
ir_array = np.array([stdevirpercs, stdev_ircoeffs])

ir_ind = np.argpartition(ir_array[0, :], -N)[-N:]
skew_ind = np.argpartition(skew_array[0, :], -N)[-N:]


# -----------------------------------------------------------------------------
# Get fitted values
# -----------------------------------------------------------------------------

fitted_qa = np.matmul(qadf_scaled[:, skew_ind], skew_qacoeffs[skew_ind]/1000)
fitted_ir = np.matmul(irdf_scaled[:, ir_ind], stdev_ircoeffs[ir_ind]/1000)


# -----------------------------------------------------------------------------
# Create dataframes and save them
# -----------------------------------------------------------------------------

SkewShock = pd.DataFrame(data = m12df_nan_qa.Skewresid)
SkewShock['Skewresid_fitted_qa'] = fitted_qa
SkewShock.to_pickle('Output/7 - Shocks/SkewShock.pkl')

StDevShock = pd.DataFrame(data = m12df_nan_ir.StDevresid)
StDevShock['StDevresid_fitted_ir'] = fitted_ir
StDevShock.to_pickle('Output/7 - Shocks/StDevShock.pkl')






