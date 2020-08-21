# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 14:00:32 2018

This script does elastic net regressions to see which variables explain
the different changes in moments on communication days. For a given sub-sample
of the original full sample.

It first purges the data using the new sub sample, instead of the old purged
data.

It looks at the first, second and third.

It saves the coefficient estimates (as a numpy array), the number of non zero
coefficients, and the number of non zero coefficients that are found in 500 
random permutations of the dependent variable, all in the relevant output folder.

@author: Tim
"""

# Initial imports

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn import preprocessing
import os

# -----------------------------------------------------------------------------
# Primitives
# -----------------------------------------------------------------------------

# Set topic number
k = 30 # Number of topics
if k==30:
    topicnumber = 'k_30'
else:
    topicnumber='k_20'
    

# Put in list of text dataframes you want to examine, which should be purged already
textdflist = ['irdf_agg_dt_diff_purged',
              'irdf_agg_dt_purged',
              'qadf_agg_dt_diff_purged',
              'qadf_agg_dt_purged',
              'statdf_agg_dt_diff_purged',
              'statdf_agg_dt_purged']

textdfdict = {'irdf_agg_dt_diff_purged' :[],
              'irdf_agg_dt_purged':[],
              'qadf_agg_dt_diff_purged':[],
              'qadf_agg_dt_purged':[],
              'statdf_agg_dt_diff_purged':[],
              'statdf_agg_dt_purged':[]}

# Set the working directory, where you want the results saved
directory = 'Output/5 - Robustness/Purged/PreSupThu/'
os.chdir(directory)

# Number of samples to take
sampleno=500

# Set sub-sample
startdate = '1998/01/01' 
enddate = '2015/06/01' # super thurs

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------

# Import financial data
m12df = pd.read_pickle('Data/m12df_chg_purged.pkl') # financial data, changes, and purged

# Import text data
for filename in textdflist:
    textdfdict[filename] = pd.read_pickle((str('Data/')
                                        + str(filename)
                                        + str('.pkl')))

# Add suffix for the change in topic variables so we can distinguish them in queries
textdfdict['irdf_agg_dt_diff_purged'] = textdfdict['irdf_agg_dt_diff_purged'].add_suffix('_chg')
textdfdict['qadf_agg_dt_diff_purged'] = textdfdict['qadf_agg_dt_diff_purged'].add_suffix('_chg')
textdfdict['statdf_agg_dt_diff_purged'] = textdfdict['statdf_agg_dt_diff_purged'].add_suffix('_chg')

# -----------------------------------------------------------------------------
# Manipulate data
# -----------------------------------------------------------------------------

# We have 3 separate types of communcation (if we are doing separate analysis here)
# IR, Q and A, Statement.

# We will want to run them each separately on the moment data, and then afterwards
# we will want to put all of the topics (3 x 30 x 2) on the data, to see if only 
# topics from one of the mediums are chosen by the ElasticNet

# Combine different mediums into 60 topic (30 topics plus differences) dfs
irdf = pd.concat([textdfdict['irdf_agg_dt_diff_purged'], textdfdict['irdf_agg_dt_purged']], axis=1)
qadf = pd.concat([textdfdict['qadf_agg_dt_diff_purged'], textdfdict['qadf_agg_dt_purged']], axis=1)
statdf = pd.concat([textdfdict['statdf_agg_dt_diff_purged'], textdfdict['statdf_agg_dt_purged']], axis=1)

entiretextdf = pd.concat([irdf, qadf, statdf], axis=1)

# Put each of these into a dataframe with the financial data, and remove any NaN rows
irdfcomb = pd.concat([m12df, irdf], axis=1)
irdfcomb.dropna(axis=0, how='any', inplace=True)
qadfcomb = pd.concat([m12df, qadf], axis=1)
qadfcomb.dropna(axis=0, how='any', inplace=True)
statdfcomb = pd.concat([m12df, statdf], axis=1)
statdfcomb.dropna(axis=0, how='any', inplace=True)
entiretextdfcomb = pd.concat([m12df, entiretextdf], axis=1)
entiretextdfcomb.dropna(axis=0, how='any', inplace=True)

# Separate back out into separate dataframes

m12df_nan_ir = irdfcomb.loc[:, m12df.columns].copy()
irdf_nan = irdfcomb.loc[:, irdf.columns].copy()
m12df_nan_qa = qadfcomb.loc[:, m12df.columns].copy()
qadf_nan = qadfcomb.loc[:, qadf.columns].copy()
m12df_nan_stat = statdfcomb.loc[:, m12df.columns].copy()
statdf_nan = statdfcomb.loc[:, statdf.columns].copy()
m12df_nan_entire = entiretextdfcomb.loc[:, m12df.columns].copy()
entiretextdf_nan = entiretextdfcomb.drop(m12df.columns, axis=1)

# Reduce dataframes to sub-sample
m12df_nan_ir = m12df_nan_ir.loc[((m12df_nan_ir.index>pd.to_datetime(startdate, format='%Y/%m/%d'))&
                                 (m12df_nan_ir.index<pd.to_datetime(enddate, format='%Y/%m/%d'))), :]
irdf_nan = irdf_nan.loc[((irdf_nan.index>pd.to_datetime(startdate, format='%Y/%m/%d'))&
                                 (irdf_nan.index<pd.to_datetime(enddate, format='%Y/%m/%d'))), :]
m12df_nan_qa = m12df_nan_qa.loc[((m12df_nan_qa.index>pd.to_datetime(startdate, format='%Y/%m/%d'))&
                                 (m12df_nan_qa.index<pd.to_datetime(enddate, format='%Y/%m/%d'))), :]
qadf_nan = qadf_nan.loc[((qadf_nan.index>pd.to_datetime(startdate, format='%Y/%m/%d'))&
                                 (qadf_nan.index<pd.to_datetime(enddate, format='%Y/%m/%d'))), :]
m12df_nan_stat = m12df_nan_stat.loc[((m12df_nan_stat.index>pd.to_datetime(startdate, format='%Y/%m/%d'))&
                                 (m12df_nan_stat.index<pd.to_datetime(enddate, format='%Y/%m/%d'))), :]
statdf_nan = statdf_nan.loc[((statdf_nan.index>pd.to_datetime(startdate, format='%Y/%m/%d'))&
                                 (statdf_nan.index<pd.to_datetime(enddate, format='%Y/%m/%d'))), :]
m12df_nan_entire = m12df_nan_entire.loc[((m12df_nan_entire.index>pd.to_datetime(startdate, format='%Y/%m/%d'))&
                                 (m12df_nan_entire.index<pd.to_datetime(enddate, format='%Y/%m/%d'))), :]
entiretextdf_nan = entiretextdf_nan.loc[((entiretextdf_nan.index>pd.to_datetime(startdate, format='%Y/%m/%d'))&
                                 (entiretextdf_nan.index<pd.to_datetime(enddate, format='%Y/%m/%d'))), :]
# -----------------------------------------------------------------------------
# Preprocess the data
# -----------------------------------------------------------------------------

# This puts our X values as mean 0 std.dev 1, which we need to run penalized regs

irdf_scaled = preprocessing.scale(irdf_nan)
qadf_scaled = preprocessing.scale(qadf_nan)
statdf_scaled = preprocessing.scale(statdf_nan)
entiretextdf_scaled = preprocessing.scale(entiretextdf_nan)

# -----------------------------------------------------------------------------
# Run separate Elastic Nets
# -----------------------------------------------------------------------------
# For each medium, we want to run an elastic net, save the number of non zero
# coefficients, and then do a bunch of permutations in which we save the number
# of non zero coefficients

# We also want to do this for three moments (1, 2, 3)
momentlist = ['Meanresid', 'StDevresid', 'Skewresid']

# And for three mediums
mediumlist = ['ir', 'qa', 'stat']

textforENdict = {'ir' : irdf_scaled,
                 'qa' : qadf_scaled,
                 'stat' : statdf_scaled}

m12forENdict = {'ir' : m12df_nan_ir,
                'qa' : m12df_nan_qa,
                'stat' : m12df_nan_stat}

coeffnoarray = np.zeros(len(momentlist)*len(mediumlist))

coeffpermdict = {'mean_ir':np.zeros(sampleno),
                 'mean_qa':np.zeros(sampleno),
                 'mean_stat':np.zeros(sampleno),
                 'stdev_ir':np.zeros(sampleno),
                 'stdev_qa':np.zeros(sampleno),
                 'stdev_stat':np.zeros(sampleno),
                 'skew_ir':np.zeros(sampleno),
                 'skew_qa':np.zeros(sampleno),
                 'skew_stat':np.zeros(sampleno)}

coeffpermlist = ['mean_ir',
                 'mean_qa',
                 'mean_stat',
                 'stdev_ir',
                 'stdev_qa',
                 'stdev_stat',
                 'skew_ir',
                 'skew_qa',
                 'skew_stat']

coeffmatdict = {'mean_ir':[],
                 'mean_qa':[],
                 'mean_stat':[],
                 'stdev_ir':[],
                 'stdev_qa':[],
                 'stdev_stat':[],
                 'skew_ir':[],
                 'skew_qa':[],
                 'skew_stat':[]}

# Run an ElasticNet and save the number of non-zero coffecients
i=0
for moment in momentlist:
    for medium in mediumlist:
        coeffname = coeffpermlist[i]
        y = m12forENdict[medium].loc[:, moment].values.copy()*1000 # doing it in basis points means faster convergence
        X = textforENdict[medium].copy()
        
        # Do elastic net incorporating cross validation choice of penalty
        regr = ElasticNetCV(copy_X = True,
                    cv = len(y), # leave one out cross validation
                    fit_intercept = False,
                    alphas = None,
                    l1_ratio=0.99,
                    random_state=0,
                    tol=0.01,
                    n_alphas=500,
                    max_iter = 10000)
        
        regr.fit(X, y)
        coeffnoarray[i] = np.count_nonzero(regr.coef_) # save number of non-zero coefficients
        coeffmatdict[coeffname] = regr.coef_
        
        # Now we can do the permutations
        for k in range(sampleno):
            y = m12forENdict[medium].loc[:, moment].values.copy()*1000
            y = np.random.permutation(y) # randomly shuffle y
            X = textforENdict[medium].copy()
            
            regr = ElasticNetCV(copy_X = True,
                        cv=len(y),
                        fit_intercept = False,
                        alphas = None,
                        l1_ratio=0.99,
                        random_state=0,
                        n_alphas=500,
                        tol=0.01, # make it able to converge by raising the tolerance a bit
                        max_iter = 10000)
            regr.fit(X, y)
            coeffpermdict[coeffname][k] = np.count_nonzero(regr.coef_)
        
        i += 1
        
# -----------------------------------------------------------------------------
# Run entire Elastic Net
# -----------------------------------------------------------------------------        

# We have 180 topic time series (60 for each medium, 30 topics and 30 changes in topics)
# We want to see if only topic series from one medium are chosen by the EN
# So lets shove them all in one big Elastic Net
        
entirecoeffpermdict = {'mean_entire':np.zeros(sampleno),
                       'stdev_entire':np.zeros(sampleno),
                       'skew_entire':np.zeros(sampleno)}

entirecoeffpermlist = ['mean_entire',
                       'stdev_entire',
                       'skew_entire']

entirecoeffmatdict = {'mean_entire':[],
                      'stdev_entire':[],
                      'skew_entire':[]}

entirecoeffnoarray = np.zeros(len(entirecoeffpermlist))

i=0
for moment in momentlist:
    coeffname = entirecoeffpermlist[i]
    y = m12df_nan_entire.loc[:, moment].values.copy()*1000
    X = entiretextdf_scaled.copy()
    
    # Do elastic net incorporating cross validation choice of penalty
    regr = ElasticNetCV(copy_X = True,
                cv = len(y), # leave one out cross validation
                fit_intercept = False,
                alphas = None,
                l1_ratio=0.99,
                random_state=0,
                tol=0.01,
                n_alphas=500,
                max_iter = 10000)
    
    regr.fit(X, y)
    entirecoeffnoarray[i] = np.count_nonzero(regr.coef_)
    entirecoeffmatdict[coeffname] = regr.coef_
    
    # Now we can do the permutations
    for k in range(sampleno):
        y = m12df_nan_entire.loc[:, moment].values.copy()*1000
        y = np.random.permutation(y) # randomly shuffle y
        X = entiretextdf_scaled.copy()
         
        regr = ElasticNetCV(copy_X = True,
                   cv=len(y),
                   fit_intercept = False,
                   alphas = None,
                   l1_ratio=0.99,
                   random_state=0,
                   n_alphas = 500,
                   tol=0.01, # make it able to converge by raising the tolerance a bit
                   max_iter = 10000)
        regr.fit(X, y)
        entirecoeffpermdict[coeffname][k] = np.count_nonzero(regr.coef_)
        
    i += 1

# -----------------------------------------------------------------------------
# Saving the data
# -----------------------------------------------------------------------------
        
# Save the number of non-zero coefficients

np.save('nonzerocoeffs.npy',
        coeffnoarray)
        
np.save('Entire/nonzerocoeffs.npy',
        entirecoeffnoarray)
        
for j in range(len(coeffpermlist)):
    name = coeffpermlist[j]
    np.save((str(name) + str('coeffs.npy')), coeffmatdict[name])

        
for j in range(len(entirecoeffpermlist)):
    name = entirecoeffpermlist[j]
    np.save((str('Entire/')
                + str(name)
                + str('coeffs.npy')), entirecoeffmatdict[name])

# Save the permutation distributions

for j in range(len(coeffpermlist)):
    name = coeffpermlist[j]
    np.save((str(name) + str('.npy')), coeffpermdict[name])

for j in range(len(entirecoeffpermlist)):
    name = entirecoeffpermlist[j]
    np.save((str('Entire/')
                + str(name)
                + str('.npy')), entirecoeffpermdict[name])