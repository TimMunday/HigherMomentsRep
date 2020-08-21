# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:38:35 2018

This script takes the excel files for the Q and A
and saves them as a pickle file.

@author: Tim
"""



# Some initial imports
import pandas as pd
import numpy as np
import os

# Set working directory
directory = 'C:/Users/tmund/Documents/Submissions/HigherMomentsRep/'
os.chdir(directory)

# Import list of dates
ir_dates = pd.read_csv('Data/BoEDataForTim/inflation_report/ir_dates_csv.csv',
                      header=None, 
                      usecols=[0],
                      dtype=str)
ir_dates.columns = ['NumDate']
ir_dates = ir_dates.dropna(axis=0, how='all')
ir_dates['LookUpYear'] = ir_dates.NumDate.str[0:4]
ir_dates['LookUpDate'] = ir_dates.NumDate.str[0:4] + str('_') + ir_dates.NumDate.str[4:6] + str('_') + ir_dates.NumDate.str[6:8]
ir_dates.drop(ir_dates.index[46:], axis=0, inplace=True) # only keep those we have data for

# Initialise dataframe
data = pd.DataFrame({'Speaker':[],
                     'Paragraph':[],
                     'BoE':[],
                     'Date':[],})

# Import data from excel files
for i in ir_dates.index: #leave out the 90s years where we don't have data
    data_add =  pd.read_excel('Data/BoEDataForTim/press_conferences/'
                    + str(ir_dates.LookUpYear[i])
                    + str('/')
                    + str(ir_dates.LookUpDate[i])
                    + str('_QA_excel.xlsx'),
                             header = 0)
    data_add['Date'] = ir_dates.LookUpDate[i]
    data = data.append(data_add, sort=True)

data['Type']='QandA'
data['Section'] = np.nan
data.drop(columns=['QA'], inplace=True)
data.dropna(subset=['Paragraph'], inplace=True) # remove rows with NAs in the paragraph bit

data['DateTime'] = pd.to_datetime(data.Date, format='%Y_%m_%d') # create pandas datetime column
data.set_index('DateTime', inplace=True) # set the new datetime column as the index
data.rename(columns = {'Date':'LookUpDate'}, inplace=True) # rename Date column to not get confused

data.to_pickle('Data/Pickled/QandA.pkl')