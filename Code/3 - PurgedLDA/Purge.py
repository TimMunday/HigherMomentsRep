# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 21:07:15 2018

This script differences the topic data first.

Then, this script runs OLS regressions on the financial and text data, and removes
the variance that can be ascribed to the numerical information in the IRs
by running OLS regressions and saving the residuals.

It then pickles and saves those residuals in the relevant Output folder.

@author: Tim
"""

# Initial imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os

# -----------------------------------------------------------------------------
# Primitives
# -----------------------------------------------------------------------------

# Set sub sample if you want to do robustness checks for example
startdate = '1998/01/01'
enddate = '2015/06/01' # super thursday date

#Set working directory where results should be saved
directory = 'C:/Users/tmund/Documents/Submissions/HigherMomentsRep/'
os.chdir(directory)

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------

m12df_chg_irdates = pd.read_pickle('Data/Pickled/m12df_chg_irdates.pkl') # financial option data

irdf_agg_dt = pd.read_pickle('Output/1 - LDA/Together/k_30/QuerySeparate/ir_dist_dt.pkl') # LDA topic distribution data
qadf_agg_dt = pd.read_pickle('Output/1 - LDA/Together/k_30/QuerySeparate/qa_dist_dt.pkl')
statdf_agg_dt = pd.read_pickle('Output/1 - LDA/Together/k_30/QuerySeparate/stat_dist_dt.pkl')

irdf_agg_dt_diff = irdf_agg_dt.diff(periods=1).copy()
qadf_agg_dt_diff = qadf_agg_dt.diff(periods=1).copy()
statdf_agg_dt_diff = statdf_agg_dt.diff(periods=1).copy()

#alltext_agg_dt = pd.read_pickle('C:/Users/Tim/Documents/Nuffield/MphilThesis/Output/LDA/Together/k_30/alltext_agg_dt.pkl')
#alltext_agg_dt_diff = alltext_agg_dt.diff(periods=1).copy()

numerical = pd.read_pickle('Data/Pickled/NumericalData.pkl') # get numerical data

    
# Restrict first two to subsample
    
m12df_chg_irdates = m12df_chg_irdates.loc[((m12df_chg_irdates.index>pd.to_datetime(startdate, format='%Y/%m/%d'))&
                                           (m12df_chg_irdates.index<pd.to_datetime(enddate, format='%Y/%m/%d'))), :].copy()
numerical = numerical.loc[((numerical.index>pd.to_datetime(startdate, format='%Y/%m/%d'))&
                          (numerical.index<pd.to_datetime(enddate, format='%Y/%m/%d'))), :].copy()

# Set up list of text data frames you are purging
#textdflist = ['alltext_agg_dt', 'alltext_agg_dt_diff']
textdflist = ['irdf_agg_dt', 'qadf_agg_dt', 'statdf_agg_dt', 'irdf_agg_dt_diff', 'qadf_agg_dt_diff', 'statdf_agg_dt_diff']
textdfdict={} # initialise empty dictionary

# populate dictionary
for name in textdflist:
    textdfdict[name] = eval(name)    

# Restrict text data to subsample
for name in textdflist:
    textdfdict[name] = textdfdict[name].loc[((textdfdict[name].index>pd.to_datetime(startdate, format='%Y/%m/%d'))&
                                             (textdfdict[name].index<pd.to_datetime(enddate, format='%Y/%m/%d'))), :].copy()

# -----------------------------------------------------------------------------
# Regressions
# -----------------------------------------------------------------------------

# Initialise Dataframes to hold purged data
m12df_chg_irdates_purged = pd.DataFrame()

textdfpurgeddict = {}
for name in textdflist:
    textdfpurgeddict[name] = pd.DataFrame()

# Set up OLS dataframe
olsdf = pd.concat([m12df_chg_irdates, numerical], axis=1)

# Set up the numerical variables you actually want to test against
    # If you use all of them you get overfitting which leads to bad residual measures
    # Use one year ahead measures as that matches the financial data maturity

numericallist = ['cpimode1y', 'cpivar1y', 'cpiskew1y', 'gdpmode1y', 'gdpvar1y',
                 'gdpskew1y', 'cpimode1y_diff', 'cpivar1y_diff', 'cpiskew1y_diff',
                 'gdpmode1y_diff', 'gdpvar1y_diff', 'gdpskew1y_diff']

# Financial data
X = olsdf.loc[:, numericallist].copy() # all numerical data
X = sm.add_constant(X, prepend=False) # add in a constant column    
    
for column in m12df_chg_irdates.columns:
    y = m12df_chg_irdates.loc[:, column].copy()
    mod = sm.OLS(y, X, missing = 'drop')
    results = mod.fit()
    y_hat = results.fittedvalues
    y_hat = pd.DataFrame(y_hat)
    y_hat.columns = ['y_hat'] # rename column
    output_df = pd.concat([y_hat, y], axis=1)
    m12df_chg_irdates_purged[str(column)+'resid'] = output_df[column] - output_df['y_hat']

# Text data    
for name in textdflist:
    olsdf = pd.concat([olsdf, textdfdict[name]], axis=1) # add the text data to OLS dataframe
    X = olsdf.loc[:, numericallist].copy() # all numerical data
    X = sm.add_constant(X, prepend=False) # add in a constant column 
    for column in textdfdict[name].columns:
        y = olsdf.loc[:, column].copy()
        mod = sm.OLS(y, X, missing = 'drop')
        results = mod.fit()
        y_hat = results.fittedvalues
        y_hat = pd.DataFrame(y_hat)
        y_hat.columns = ['y_hat'] # rename column
        output_df = pd.concat([y_hat, y], axis=1)
        textdfpurgeddict[name][str(column)+'resid'] = output_df[column] - output_df['y_hat']
    olsdf.drop(labels=textdfdict[name].columns, inplace=True, axis=1) 


# -----------------------------------------------------------------------------
# Saving the data
# -----------------------------------------------------------------------------

# note if you aren't doing robustness, then these need to be saved in Ouput/ 2- PurgedFinancial, and Output/3 - PurgedLDA 

# Save the financial data
m12df_chg_irdates_purged.to_pickle('Output/5 - Robustness/Purged/PreSupThu/Data/m12df_chg_purged.pkl')

# Save the text data
for name in textdflist:
    textdfpurgeddict[name].to_pickle('Output/5 - Robustness/Purged/PreSupThu/Data/' + name + str('_purged.pkl'))
