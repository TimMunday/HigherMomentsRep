# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 21:07:15 2018

This script formats the financial data into changes in variables on the days
we want, and saves that data as a pickle file.


@author: Tim
"""

# Some initial imports
import numpy as np
import pandas as pd
import os

# Set working directory
directory = 'C:/Users/tmund/Documents/Submissions/HigherMomentsRep/'
os.chdir(directory)

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------

m12df = pd.read_pickle('Data/Pickled/m12df.pkl') # financial option data
alltext_agg_dt = pd.read_pickle('Output/LDA/Together/k_30/alltext_agg_dt.pkl') # LDA topic distribution data
numerical = pd.read_pickle('Data/Pickled/NumericalData.pkl') # get numerical data

# -----------------------------------------------------------------------------
# Manipulate and save financial data
# -----------------------------------------------------------------------------

m12df_1d = m12df.loc[(m12df.index - pd.Timedelta(1, unit='d')), :].copy() #form dataframe of one day before
m12df_1d.set_index(m12df.index, inplace=True) # change the index so we can subtract
m12df_chg_all = m12df - m12df_1d # create dataframe of daily changes
m12df_chg_irdates = m12df_chg_all.loc[alltext_agg_dt.index, :].copy() # create dataframe of the changes on the ir dates

# Save that dataframe
m12df_chg_irdates.to_pickle('Data/Pickled/m12df_chg_irdates.pkl')
