# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 16:58:35 2018

This script takes the numerical data in the IRs and pickles it in a dataframe

@author: Tim
"""

# Initial imports
import numpy as np
import pandas as pd
import os

# Set working directory
directory = 'C:/Users/tmund/Documents/Submissions/HigherMomentsRep/'
os.chdir(directory)

# Import data
ir_numerical_data = pd.read_csv('Data/BoEDataForTim/inflation_report/IR_numerical_data.csv')

# Set index as a pandas date time object
ir_numerical_data['DateTime']=pd.to_datetime(ir_numerical_data.LookUpDate, format = '%Y_%m_%d')
ir_numerical_data.set_index('DateTime', inplace=True)
ir_numerical_data.drop(columns = ['LookUpDate'], inplace=True)

# Pickle it
ir_numerical_data.to_pickle('Data/Pickled/NumericalData.pkl')
