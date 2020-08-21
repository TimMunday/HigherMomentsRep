#%% 
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 21:33:08 2019

Take the raw reuters survey data, extract and wrangle it, make chart for it

@author: Tim
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
directory = 'C:/Users/tmund/Documents/Submissions/HigherMomentsRep/'
os.chdir(directory)

#----------------------------------------------#
#----- Wrangle the Reuters Data ---------------#
#----------------------------------------------#

# Create list of dates that will be imported
importlist = ['q'+ str(i) + str (j) for i in range(1, 5) for j in range(1998, 2020)]


# Import and wrangle data

Reutersdf = pd.DataFrame()

for stringdate in importlist:
    df = pd.read_excel((str('Data/Surveys/Reuters/')
                       +str(stringdate)
                       +str('.xlsx')),
                       header=8).T
    df = df.iloc[1:, 0:7]
    df.columns = ['Median', 'Mean', 'Mode', 'Min','Max', 'StDev', 'N'] # put in column names
    df['ForecastDate'] = stringdate # put in forecast date
    rows = [r for r in df.index if 'Unnamed' not in r] # get non unnamed rows
    df = df.loc[rows, :].copy() # remove unnamed rows
    df.reset_index(inplace=True)
    df.rename(columns = {'index':'SurveyDate'}, inplace=True) # rename
    df.ForecastDate = pd.to_datetime([str('28') + str(int(x[1])*3) + str(x[2:]) for x in df.ForecastDate], format='%d%m%Y') # set forecast date as pd datetime, reuters survey forecasts the end of the q
    df.loc[:, 'SurveyDate'] = pd.to_datetime(df.loc[:, 'SurveyDate']) # convert survey date to pd datetime
    df['Horizon'] = df.ForecastDate - df.SurveyDate # create variable of how far ahead you are forecasting
    df['Range'] = df.Max - df.Min # add range column
    Reutersdf = pd.concat([Reutersdf, df], axis=0, ignore_index=True) # concatenate frames


    
#Reutersdf = Reutersdf.loc[0:, ['SurveyDate', 'Horizon', 'StDev', 'Range']] # get columns you want
Reutersdf.Horizon = Reutersdf['Horizon'].dt.days # get rid of time delta and get an integer instead
Reutersdf['1YearAbsErr'] = np.abs(Reutersdf.Horizon - 365) # get error from one year
Reutersdf.set_index(['SurveyDate', 'Horizon'], inplace=True) # create multi-index
Reutersdf.sort_index(inplace=True) # sort multi-index

Reutersdf = Reutersdf.groupby(level=0).apply(lambda x: x.loc[x['1YearAbsErr'].idxmin(axis=0), :]) # get one closest to one year ahead forecast for each survey date


# Clean reuters survey -  remove forecasts that are more than a few days off a year ahead
Reutersdf = Reutersdf.loc[Reutersdf.loc[:, '1YearAbsErr']<90, :]


# %% Plots

from matplotlib import pyplot as plt
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Dark2')

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(Reutersdf.index, Reutersdf.Mean, label='Mean')
plt.plot(Reutersdf.index, Reutersdf.Mean + 2*Reutersdf.StDev, 'k', linestyle='--', label='+/- 2 s.d.')
plt.plot(Reutersdf.index, Reutersdf.Mean - 2*Reutersdf.StDev, 'k', linestyle='--')
plt.xlabel('Survey Date')
plt.ylabel('Bank Rate 1-year-ahead forecast')
plt.legend()
plt.savefig('DataVis/MS/Figs/Survey.pdf')
plt.show()

# %%
