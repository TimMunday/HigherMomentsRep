#%% Initial imports
from re import I
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor, DynamicFactorResults
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ, DynamicFactorMQResults
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
from matplotlib import pyplot as plt
from rpy2.robjects import r
from rpy2.robjects.numpy2ri import numpy2ri
import pickle
import statsmodels.api as sm

#%% Import data

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

# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------

# Import financial data
m12df = pd.read_pickle('D:/DellXPS13_05_2020/Tim/Documents/Nuffield/MphilThesis/Output/2 - PurgedFinancial/m12df_chg_purged.pkl') # financial data, changes, and purged

# Import text data
for filename in textdflist:
    textdfdict[filename] = pd.read_pickle((str('D:/DellXPS13_05_2020/Tim/Documents/Nuffield/MphilThesis/Output/3 - PurgedLDA/Together/k_30')
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

# This puts our X values as mean 0 std.dev 1
irdf_scaled = preprocessing.scale(irdf_nan)
qadf_scaled = preprocessing.scale(qadf_nan)
statdf_scaled = preprocessing.scale(statdf_nan)
entiretextdf_scaled = preprocessing.scale(entiretextdf_nan)
ir_fin_sd = preprocessing.scale(m12df_nan_ir.loc[:, 'StDevresid'].values.reshape(-1, 1))
qa_fin_sd = preprocessing.scale(m12df_nan_qa.loc[:, 'StDevresid'].values.reshape(-1, 1))
stat_fin_sd = preprocessing.scale(m12df_nan_stat.loc[:, 'StDevresid'].values.reshape(-1, 1))
entire_fin_sd = preprocessing.scale(m12df_nan_entire.loc[:, 'StDevresid'].values.reshape(-1, 1))
ir_fin_skew = preprocessing.scale(m12df_nan_ir.loc[:, 'Skewresid'].values.reshape(-1, 1))
qa_fin_skew = preprocessing.scale(m12df_nan_qa.loc[:, 'Skewresid'].values.reshape(-1, 1))
stat_fin_skew = preprocessing.scale(m12df_nan_stat.loc[:, 'Skewresid'].values.reshape(-1, 1))
entire_fin_skew = preprocessing.scale(m12df_nan_entire.loc[:, 'Skewresid'].values.reshape(-1, 1))

#%% Export data for use in r

# data_for_r = np.concatenate((irdf_scaled, ir_fin), axis=1)
# data_for_r = np.array(data_for_r, dtype='float64')
# data_out = numpy2ri(data_for_r)
# r.assign("data_forR", data_out)
# r("save(data_forR, file='C:/Users/tmund/Documents/HigherMoments/Code/Python/dataR.gzip', compress=TRUE)")

#%% Estimate DFM 1

text_est_list = [irdf_scaled, qadf_scaled, statdf_scaled]# irdf_scaled, qadf_scaled, statdf_scaled] #entiretextdf_scaled
fin_est_list = [ir_fin_skew, qa_fin_skew, stat_fin_skew]#, ir_fin_sd, qa_fin_sd, stat_fin_sd ir_fin_skew, qa_fin_skew, stat_fin_skew] #entire_fin_sd, entire_fin_skew
results_list = []
factor_no = 1
for i in np.arange(len(text_est_list)):  
    model = DynamicFactorMQ(endog=pd.DataFrame(np.concatenate((text_est_list[i], fin_est_list[i]), axis=1)),
                      factors=factor_no,#factor_multiplicities=factor_no,
                      factor_orders=5,
                      idiosyncratic_ar1=False) 
    results = model.fit(method='em',
                        cov_type='robust',
                        disp=False)
    results.mle_settings['optimizer'] = 'l8'
    print(i)
    results.save(str('C:/Users/tmund/Documents/HigherMoments/Code/Python/model_' + str(i) + '_.pickle'))

