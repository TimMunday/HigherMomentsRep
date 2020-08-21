# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:25:51 2018

This script pickles and cleans the IR text data and saves it in an
appropriate place.

@author: Tim
"""
# Some initial imports
import pandas as pd
import numpy as np
import os

# Set working directory
directory = 'C:/Users/tmund/Documents/Submissions/HigherMomentsRep/'
os.chdir(directory)

# Import IR text data
datairfinal = pd.read_table("Data/BoEDataForTim/inflation_report/ir_data_final.txt",
                     encoding="utf-8")
datairfinal['Date'] = np.nan

# Import list of dates
ir_dates = pd.read_csv('Data/BoEDataForTim/inflation_report/ir_dates_csv.csv',
                      header=None, 
                      usecols=[0],
                      dtype=str)
ir_dates.columns = ['NumDate']
ir_dates = ir_dates.dropna(axis=0, how='all')
ir_dates['ir_date'] = ir_dates.NumDate.str[0:6]
ir_dates['LookUpDate'] = ir_dates.NumDate.str[0:4] + str('_') + ir_dates.NumDate.str[4:6] + str('_') + ir_dates.NumDate.str[6:8]
ir_dates['ir_date'] = pd.to_numeric(ir_dates['ir_date']) # make sure we are using same type of data

# Loop through and put in my dates
for i in range(len(ir_dates['ir_date'])):
    NumberDate = ir_dates.loc[i, 'ir_date']
    MyDate = ir_dates.loc[i, 'LookUpDate']
    datairfinal.loc[datairfinal['ir_date']==NumberDate, 'Date'] = MyDate
    
# Clean up
datairfinal.drop('ir_date', axis=1, inplace=True)
datairfinal.rename(columns = {'paragraph':'Paragraph'}, inplace=True)


datairfinal['Type']='IR'   
datairfinal['BoE'] = 1.
datairfinal['Speaker'] = 'IR'
datairfinal.dropna(subset=['Paragraph'], inplace=True) # remove rows with NAs in the paragraph bit

datairfinal['DateTime'] = pd.to_datetime(datairfinal.Date, format='%Y_%m_%d') # create pandas datetime column
datairfinal.set_index('DateTime', inplace=True) # set the new datetime column as the index
datairfinal.rename(columns = {'Date':'LookUpDate'}, inplace=True) # rename Date column to not get confused

# Initialise dataframe
data_to_merge = pd.DataFrame({'Speaker':[],
                     'Type':[],
                     'Date':[],
                     'Paragraph':[],
                     'Section':[],
                     'Sub section':[],
                     'Sub sub section':[],
                     'Sub sub sub section':[]})


# Import data from excel files
for i in range(13): #leave out the 90s years where other data suffices
    data_add =  pd.read_excel('Data/Text/MyIRs/IR_'
                    + str(ir_dates.LookUpDate[i])
                    + str('_excel.xlsx'),
                             header = 0)
    data_to_merge = data_to_merge.append(data_add, sort=True)

# Rename some columns
data_to_merge.rename(columns = {'Date':'LookUpDate',
                                'Section':'section',
                               'Sub section':'sub_section',
                               'Sub sub section':'sub_sub_section',
                               'Sub sub sub section':'sub_sub_sub_section'}, inplace=True)

    
# Make the same as the dataframe you want to add it to the end of..
data_to_merge['BoE']=1.0
data_to_merge['DateTime'] = pd.to_datetime(data_to_merge.LookUpDate, format='%Y_%m_%d') # create pandas datetime column
data_to_merge.set_index('DateTime', inplace=True) # set the new datetime column as the index

# append the datasets
data = datairfinal.append(data_to_merge, sort=True)
data.rename(columns = {'section':'Section',
                       'sub_section':'Sub_section',
                       'sub_sub_section':'Sub_sub_section',
                       'sub_sub_sub_section':'Sub_sub_sub_section'}, inplace=True)

data.to_pickle('Data/Pickled/IR.pkl')