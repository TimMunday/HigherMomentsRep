# -*- coding: utf-8 -*-
"""
This script takes the excel spreadsheet of option implied pdfs from the BoE,
does some preliminary renaming of columns, and then saves each maturity set
as a pickle file.

Created on Wed Oct 31 10:13:20 2018

@author: Tim
"""

# Initial imports
import numpy as np
import pandas as pd
from collections import namedtuple
import os

# Set working directory
directory = 'C:/Users/tmund/Documents/Submissions/HigherMomentsRep/'
os.chdir(directory)

# Import the data
Q1df = pd.read_excel("Data/BoEPDFs/shortsterling_pdfs.xlsx",
                     sheet_name ="1st quarterly contract",
                     header=4)

Q2df = pd.read_excel("Data/BoEPDFs/shortsterling_pdfs.xlsx",
                     sheet_name ="2nd quarterly contract",
                     header=4)

m3df = pd.read_excel("Data/BoEPDFs/shortsterling_pdfs.xlsx",
                     sheet_name ="3 month constant maturity",
                     header=4)

m6df = pd.read_excel("Data/BoEPDFs/shortsterling_pdfs.xlsx",
                     sheet_name ="6 month constant maturity",
                     header=4)

m12df = pd.read_excel("Data/BoEPDFs/shortsterling_pdfs.xlsx",
                     sheet_name ="12 month constant maturity",
                     header=4)

m3df = m3df.drop(['Unnamed: 22'], axis=1)
m6df = m6df.drop(['Unnamed: 22'], axis=1)
m12df = m12df.drop(['Unnamed: 22'], axis=1)

m3df = m3df.rename(index=str, columns={'0.25.1':'ImpVol'})
m6df = m6df.rename(index=str, columns={0.5:'ImpVol'})
m12df = m12df.rename(index=str, columns={1.0:'ImpVol'})

# Cleaning, renaming and exporting
dflist = [Q1df, Q2df, m3df, m6df, m12df]
dflistname = ['Q1df', 'Q2df', 'm3df', 'm6df', 'm12df'] # probably a nicer way of doing this
for i in range(len(dflist)):
    dflist[i] = dflist[i].rename(index=str, columns={"Description": "DateTime",
                                "Standard Deviation": "StDev",
                                "Mean.1": "LogMean",
                                "Standard Deviation.1": "LogStDev",
                                "Skew.1": "LogSkew",
                                "Kurtosis.1":"LogKurtosis",
                                0.05:"cp5",
                                0.15:"cp15",
                                0.25:"cp25",
                                0.35:"cp35",
                                0.45:"cp45",
                                0.55:"cp55",
                                0.65:"cp65",
                                0.75:"cp75",
                                0.85:"cp85",
                                0.95:"cp95"})

    dflist[i] = dflist[i].drop(['Unnamed: 6', 'Unnamed: 11'], axis=1)
    dflist[i].set_index('DateTime', inplace=True) # set the new datetime column as the index
    dflist[i].to_pickle('Data/Pickled/'+str(dflistname[i])+".pkl")
    