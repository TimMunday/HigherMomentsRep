# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 14:49:16 2019

Does a bootstrap procedure to determine which topics are most important. Saves
results in relevant output folder.

Some of this is (first part, cleaning and preprocessing)
similar to the EN scripts.


@author: Tim
"""

# Initial imports

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn import preprocessing
import os
import statsmodels.api as sm
directory = 'C:/Users/tmund/Documents/Submissions/HigherMomentsRep/'
os.chdir(directory)

# -----------------------------------------------------------------------------
# Primitives
# -----------------------------------------------------------------------------

# Set topic number
k = 30 # Number of topics
if k==30:
    topicnumber = 'k_30'
else:
    topicnumber = 'k_20'

# Put in list of text dataframes you want to examine, which should be purged already

textdfdict = {'irdf_agg_dt_diff_purged' :[],
              'irdf_agg_dt_purged':[],
              'qadf_agg_dt_diff_purged':[],
              'qadf_agg_dt_purged':[],
              'statdf_agg_dt_diff_purged':[],
              'statdf_agg_dt_purged':[]}
# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------

# Import financial data
m12df = pd.read_pickle('Output/2 - PurgedFinancial/m12df_chg_purged.pkl') # financial data, changes, and purged

# Import text data
for filename in textdfdict.keys():
    textdfdict[filename] = pd.read_pickle((str('Output/3 - PurgedLDA/Together/')
                                        + str(topicnumber)
                                        + str('/QuerySeparate/')
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

# -----------------------------------------------------------------------------
# Preprocess the data
# -----------------------------------------------------------------------------

# This puts our X values as mean 0 std.dev 1, which we need to run penalized regs

irdf_scaled = preprocessing.scale(irdf_nan)
qadf_scaled = preprocessing.scale(qadf_nan)
statdf_scaled = preprocessing.scale(statdf_nan)
entiretextdf_scaled = preprocessing.scale(entiretextdf_nan)
# Construct own lag residuals
lag_no = 4

olsdf_list = [irdf_scaled, qadf_scaled, statdf_scaled, entiretextdf_scaled]
irdf_resid = pd.DataFrame(index = np.arange((len(irdf_scaled)-lag_no)), columns = irdf_nan.columns)
qadf_resid = pd.DataFrame(index = np.arange((len(qadf_scaled)-lag_no)), columns = qadf_nan.columns)
statdf_resid = pd.DataFrame(index = np.arange((len(statdf_scaled)-lag_no)), columns = statdf_nan.columns)
entiretextdf_resid = pd.DataFrame(index = np.arange((len(entiretextdf_scaled)-lag_no)), columns = entiretextdf_nan.columns)
resid_list = [irdf_resid, qadf_resid, statdf_resid, entiretextdf_resid]

for j, dataframe in enumerate(olsdf_list):
    for i in np.arange(np.shape(dataframe)[1]): # for topic
        y = dataframe[lag_no:, i]
        x_1 = np.roll(dataframe[:, i], 1).reshape(-1, 1)
        x_2 = np.roll(dataframe[:, i], 2).reshape(-1, 1)
        x_3 = np.roll(dataframe[:, i], 3).reshape(-1, 1)
        x_4 = np.roll(dataframe[:, i], 4).reshape(-1, 1)
        lag_list = [x_1, x_2, x_3, x_4]
        X = np.concatenate(lag_list[:lag_no], axis=1)[lag_no:, :] 
        model = sm.OLS(y, X)
        res = model.fit()
        y_hat = res.predict()
        resid = y-y_hat
        resid_df = resid_list[j]
        resid_df.iloc[:, i] = resid

for df in resid_list[:3]:
    df.iloc[:, :30] = df.iloc[:, 30:].diff(1).values
    df.drop(labels=0, axis=0, inplace=True)

slices = np.concatenate((np.arange(0, 30), np.arange(60, 90), np.arange(120, 150)))
slices_1 = np.concatenate((np.arange(30, 60), np.arange(90, 120), np.arange(150, 180)))
entiretextdf_resid.iloc[:, slices] = entiretextdf_resid.iloc[:, slices_1].diff(1).values
entiretextdf_resid.drop(labels=0, axis=0, inplace=True)

# -----------------------------------------------------------------------------
# Run Elastic Nets
# -----------------------------------------------------------------------------

# We want to do this for just the first moment now
momentlist = ['Meanresid']

# And for three mediums
mediumlist = ['ir', 'qa', 'stat']

textforENdict = {'ir' : preprocessing.scale(irdf_resid),
                 'qa' : preprocessing.scale(qadf_resid),
                 'stat' : preprocessing.scale(statdf_resid)}

m12forENdict = {'ir' : m12df_nan_ir.iloc[lag_no+1:, :],
                'qa' : m12df_nan_qa.iloc[lag_no+1:, :],
                'stat' : m12df_nan_stat.iloc[lag_no+1:, :]}

sampleno=500

coeffbootdict = {'Meanresid_ir':np.zeros((sampleno, 60)), # for each row we record the regression coefficients
                 'Meanresid_qa':np.zeros((sampleno, 60)),
                 'Meanresid_stat':np.zeros((sampleno, 60))}

percbootdict = {'Meanresid_ir':np.zeros(60),
                 'Meanresid_qa':np.zeros(60),
                 'Meanresid_stat':np.zeros(60)}

# Run an ElasticNets

for moment in momentlist:
    for medium in mediumlist:
        
        y = preprocessing.scale(m12forENdict[medium].loc[:, moment].values.copy())*1000 # doing it in basis points means faster convergence
        X = textforENdict[medium].copy()
        for i in range(sampleno):
            sample_index = np.random.choice(range(0, len(y)), len(y), replace=True) # create a random sample of indices
            y_sample = y[sample_index]
            X_sample = X[sample_index, :]
        
            # Do elastic net incorporating cross validation choice of penalty
            regr = ElasticNetCV(copy_X = True,
                cv = len(y_sample),
                fit_intercept = True,
                alphas = None,
                l1_ratio=0.99,
                random_state=0,
                tol=0.01,
                n_jobs = -1,
                n_alphas=500,
                max_iter = 10000)
        
            regr.fit(X_sample, y_sample)
            coeffbootdict[(str(moment) + str('_') + str(medium))][i, :] = regr.coef_ # save the coefficient estimates as an array
            print(i)
        
        percbootdict[(str(moment) + str('_') + str(medium))] = ((coeffbootdict[(str(moment) + str('_') + str(medium))] != 0).sum(0))/sampleno # save percentage time something came up


# -----------------------------------------------------------------------------
# Run entire Elastic Net
# -----------------------------------------------------------------------------        

# We have 180 topic time series (60 for each medium, 30 topics and 30 changes in topics)
# We want to see if only topic series from one medium are chosen by the EN
# So lets shove them all in one big Elastic Net
        
entirecoeffpermdict = {'mean_entire':np.zeros(sampleno)}

entirecoeffpermlist = ['mean_entire']

entirecoeffbootdict = {'mean_entire':np.zeros((sampleno, 180))}

entirecoeffpercdict = {'mean_entire':np.zeros(180)}

i=0
for moment in momentlist:
    coeffname = entirecoeffpermlist[i]
    y = preprocessing.scale(m12df_nan_entire.loc[:, moment].values.copy())*1000
    X = entiretextdf_scaled.copy()
    
    for k in range(sampleno):
        sample_index = np.random.choice(range(0, len(y)), len(y), replace=True) # create a random sample of indices
        y_sample = y[sample_index]
        X_sample = X[sample_index, :]
        
        # Do elastic net incorporating cross validation choice of penalty
        regr = ElasticNetCV(copy_X = True,
                cv = len(y_sample),
                fit_intercept = True,
                alphas = None,
                l1_ratio=0.99,
                random_state=0,
                tol=0.01,
                n_jobs=-1,
                n_alphas=500,
                max_iter = 10000)
    
        regr.fit(X_sample, y_sample)
        entirecoeffbootdict[coeffname][k, :] = regr.coef_
        print(k)
    
    entirecoeffpercdict[coeffname] = ((entirecoeffbootdict[coeffname] != 0) .sum(0))/sampleno 
    i += 1
    
    
# -----------------------------------------------------------------------------
# Save data
# -----------------------------------------------------------------------------
    
for key in entirecoeffpercdict.keys():
    np.save((str('Output/4 - VariableSelection/2 - Bootstrap/Together/')
        + str(topicnumber)
        + str('/QuerySeparate/Mean/Entire/')
        + str(key)
        + str('percs.npy')),
        entirecoeffpercdict[key])
            
for key in entirecoeffbootdict.keys():
     np.save((str('Output/4 - VariableSelection/2 - Bootstrap/Together/')
        + str(topicnumber)
        + str('/QuerySeparate/Mean/Entire/')
        + str(key)
        + str('coeffs.npy')),
        entirecoeffpercdict[key])

            
for key in percbootdict.keys():
    np.save((str('Output/4 - VariableSelection/2 - Bootstrap/Together/')
        + str(topicnumber)
        + str('/QuerySeparate/Mean/')
        + str(key)
        + str('percs.npy')),
        percbootdict[key])
            
for key in coeffbootdict.keys():
    np.save((str('Output/4 - VariableSelection/2 - Bootstrap/Together/')
        + str(topicnumber)
        + str('/QuerySeparate/Mean/')
        + str(key)
        + str('coeffs.npy')),
        coeffbootdict[key])