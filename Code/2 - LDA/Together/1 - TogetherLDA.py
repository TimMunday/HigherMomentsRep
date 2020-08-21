# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 17:36:27 2018

This script performs LDA analysis on all three different types of text data
we have (IR, Q and A, statements) as one unit of text

It extracting topics from the entire corpus and then saves the relevant files with the output
data from the LDA analysis in the Output folder

@author: Tim
"""

# Some initial imports
import pandas as pd
import numpy as np
import topicmodels
import time
import os

# Set working directory
directory = 'C:/Users/tmund/Documents/Submissions/HigherMomentsRep/'
os.chdir(directory)

start = time.time()
# -----------------------------------------------------------------------------
# Import data
# -----------------------------------------------------------------------------
irdf = pd.read_pickle('Data/Pickled/IR.pkl') # IR data
qadf = pd.read_pickle('Data/Pickled/QandA.pkl') # Q and A data
statdf = pd.read_pickle('Data/Pickled/Statement.pkl') # Introductory statement data

# Form df for Q and A that excludes journalist speaking parts
qadfboe = qadf[qadf['BoE']==1].copy()

# Create DataFrame that can hold all of the data
    # To avoid double counting, we only include the BoE part of the QandA
alltextdf = irdf.copy()
alltextdf = alltextdf.append(qadfboe, sort=True)
alltextdf = alltextdf.append(statdf, sort=True)

# -----------------------------------------------------------------------------
# Put into topic models class and pre-process
# -----------------------------------------------------------------------------

docsobj = topicmodels.RawDocs(alltextdf.Paragraph, "long") # Specifying stop words as "long"
docsobj.token_clean(1) # remove all tokens equal to 1 character long
docsobj.stopword_remove('tokens') # remove stopwords
docsobj.stem() # get the stems
docsobj.stopword_remove("stems") # remove  any stopword stems
bowobj = topicmodels.BOW(docsobj.stems) # create a bag of words object

# Here we could remove some stems via TF-IDF method if we wanted to

# -----------------------------------------------------------------------------
# Create LDA objects and get sampling
# -----------------------------------------------------------------------------

# Gibbs sampling
k = 30 # set number of topics, basically an arbitrary number
# We allow the other hyperparameters of Dirichlet to be their default values
burnin = 50 # number of initial iterations in gibbs to discard
thinning = 50 # number of iterations to wait until you next sample, given that adjacent samples are not independent
sampleno = 100 # number of samples to take. Total no of samples is burnin + thinning*sampleno
samplekeep = 20 # number of samples to keep

ldaobj = topicmodels.LDA.LDAGibbs(docsobj.stems, k)
ldaobj.sample(burnin, thinning, sampleno)
ldaobj.samples_keep(samplekeep) # keep the last few samples
ldaobj.topic_content(20, output_file=(str('Output/1 - LDA/Together/k_30/alltext_topic_desc.csv'))) # produces topic_description.csv, shows first 20 words in each topic ranked by probability
    
dt = ldaobj.dt_avg(print_output = False, output_file=(str('Output/1 - LDA/Together/k_30/alltext_dt.csv'))) # averages the document topic distributions and writes to dt.csv
# Note that dt is a numpy array, so lets put it into a pandas format so we can pickle it
alltextdist = pd.DataFrame(dt, index=alltextdf.index,
                          columns = ['Topic'+ str(n) for n in iter(range(ldaobj.K))]) # put it in a dataframe
#alltextdist.to_pickle((str('Output/1 - LDA/Together/k_30/alltext_dt.pkl'))) # pickle it
    
tt = ldaobj.tt_avg(print_output = False, output_file=(str('Output/1 - LDA/Together/k_30/alltext_tt.csv'))) # averages the topic distributions over all the stems and writes to tt.csv, each row is a unique stem
# Note that tt is a numpy array, so lets put it into a pandas format so we can pickle it
tdist = pd.DataFrame(tt,
                     columns = ['Topic'+ str(n) for n in iter(range(ldaobj.K))]) # put it in a dataframe
#tdist.to_pickle((str('Output/1 - LDA/Together/k_30/alltext_tt.pkl'))) # pickle it
    
dictionary = ldaobj.dict_print(output_file=(str('Output/1 - LDA/Together/k_30/alltext_dictionary.csv'))) # saves dictionary
                                                                    
# Now add in the topic distributions to the original speech dataframes, for each paragraph, put the distribution of topics
for m in iter(range(ldaobj.K)):
    alltextdf['Topic' + str(m)] = dt[:,m]
alltextdf.to_pickle((str('Output/1 - LDA/Together/k_30/alltext_text.pkl'))) # pickle it

# -----------------------------------------------------------------------------        
# Querying for document level analysis
# -----------------------------------------------------------------------------


# Use this part if  you want whole document queries
# =============================================================================
# alltextdf['Cleaned'] = [' '.join(s) for s in docsobj.stems] # add cleaned data to text dataframes from docsobj
# aggalltextdf = alltextdf.groupby('DateTime')['Cleaned'].apply(lambda x: ' '.join(x)) # aggregate up
# aggdocs = topicmodels.RawDocs(aggalltextdf) # create new RawDocs object that contains entire document stems in aggdocs.tokens
# aggquery = topicmodels.LDA.QueryGibbs(aggdocs.tokens, ldaobj.token_key, ldaobj.tt) # initialize query object with ldaobj attributes
# 
# aggquery.query(20) # do the query by iterating 20 times
# 
# # Get topic distributions    
# aggdt = aggquery.dt_avg(print_output=False,
#                         output_file = (str('C:/Users/Tim/Documents/Nuffield/MphilThesis/Output/1 - LDA/Together/k_30/alltext_aggdt.csv')))                                                               
# aggdist = pd.DataFrame(aggdt,
#                        index=aggalltextdf.index,
#                        columns = ['Topic'+ str(n) for n in iter(range(aggquery.K))]) #put it in a dataframe
# aggdist.to_pickle((str('C:/Users/Tim/Documents/Nuffield/MphilThesis/Output/1 - LDA/Together/k_30/alltext_agg_dt.pkl')))
#     
# # Save any data you haven't already
# # Save text data
# alltextdf.to_pickle((str('C:/Users/Tim/Documents/Nuffield/MphilThesis/Output/1 - LDA/Together/k_30/alltextdf_text.pkl')))
# 
# # Save aggregated text data
# aggalltextdf.to_pickle((str('C:/Users/Tim/Documents/Nuffield/MphilThesis/Output/1 - LDA/Together/k_30/alltextdf_agg_text.pkl')))
# 
# =============================================================================

# Use this part if you want to query each type of document separately

# IR
docsobj_ir = topicmodels.RawDocs(irdf.Paragraph, "long") # Specifying stop words as "long"
docsobj_ir.token_clean(1) # remove all tokens equal to 1 character long
docsobj_ir.stopword_remove('tokens') # remove stopwords
docsobj_ir.stem() # get the stems
docsobj_ir.stopword_remove("stems") # remove  any stopword stems
irdf['Cleaned'] = [' '.join(s) for s in docsobj_ir.stems]
agg_irdf = irdf.groupby('DateTime')['Cleaned'].apply(lambda x: ' '.join(x)) # aggregate up
aggdocs_ir = topicmodels.RawDocs(agg_irdf) # create new RawDocs object that contains entire document stems in aggdocs.tokens
aggquery_ir = topicmodels.LDA.QueryGibbs(aggdocs_ir.tokens, ldaobj.token_key, ldaobj.tt) # initialize query object with ldaobj attributes
aggquery_ir.query(20)
aggdt_ir = aggquery_ir.dt_avg(print_output=False,
                        output_file = (str('Output/1 - LDA/Together/k_30/QuerySeparate/irdt.csv')))                                                               
aggdist_ir = pd.DataFrame(aggdt_ir,
                          index=agg_irdf.index,
                        columns = ['Topic'+ str(n) for n in iter(range(aggquery_ir.K))]) #put it in a dataframe
aggdist_ir.to_pickle((str('Output/1 - LDA/Together/k_30/QuerySeparate/ir_dist_dt.pkl')))

# QA
docsobj_qa = topicmodels.RawDocs(qadfboe.Paragraph, "long") # Specifying stop words as "long"
docsobj_qa.token_clean(1) # remove all tokens equal to 1 character long
docsobj_qa.stopword_remove('tokens') # remove stopwords
docsobj_qa.stem() # get the stems
docsobj_qa.stopword_remove("stems") # remove  any stopword stems
qadfboe['Cleaned'] = [' '.join(s) for s in docsobj_qa.stems]
agg_qadf = qadfboe.groupby('DateTime')['Cleaned'].apply(lambda x: ' '.join(x)) # aggregate up
aggdocs_qa = topicmodels.RawDocs(agg_qadf) # create new RawDocs object that contains entire document stems in aggdocs.tokens
aggquery_qa = topicmodels.LDA.QueryGibbs(aggdocs_qa.tokens, ldaobj.token_key, ldaobj.tt) # initialize query object with ldaobj attributes
aggquery_qa.query(20)
aggdt_qa = aggquery_qa.dt_avg(print_output=False,
                              output_file = (str('Output/1 - DA/Together/k_30/QuerySeparate/qadt.csv')))                                                               
aggdist_qa = pd.DataFrame(aggdt_qa,
                          index=agg_qadf.index,
                          columns = ['Topic'+ str(n) for n in iter(range(aggquery_qa.K))]) #put it in a dataframe
aggdist_qa.to_pickle((str('Output/1 - LDA/Together/k_30/QuerySeparate/qa_dist_dt.pkl')))

# Stat
docsobj_stat = topicmodels.RawDocs(statdf.Paragraph, "long") # Specifying stop words as "long"
docsobj_stat.token_clean(1) # remove all tokens equal to 1 character long
docsobj_stat.stopword_remove('tokens') # remove stopwords
docsobj_stat.stem() # get the stems
docsobj_stat.stopword_remove("stems") # remove  any stopword stems
statdf['Cleaned'] = [' '.join(s) for s in docsobj_stat.stems]
agg_statdf = statdf.groupby('DateTime')['Cleaned'].apply(lambda x: ' '.join(x)) # aggregate up
aggdocs_stat = topicmodels.RawDocs(agg_statdf) # create new RawDocs object that contains entire document stems in aggdocs.tokens
aggquery_stat = topicmodels.LDA.QueryGibbs(aggdocs_stat.tokens, ldaobj.token_key, ldaobj.tt) # initialize query object with ldaobj attributes
aggquery_stat.query(20)
aggdt_stat = aggquery_stat.dt_avg(print_output=False,
                                  output_file = (str('Output/1 - LDA/Together/k_30/QuerySeparate/statdt.csv')))                                                               
aggdist_stat = pd.DataFrame(aggdt_stat,
                          index=agg_statdf.index,
                        columns = ['Topic'+ str(n) for n in iter(range(aggquery_stat.K))]) #put it in a dataframe
aggdist_stat.to_pickle((str('Output/1 - LDA/Together/k_30/QuerySeparate/stat_dist_dt.pkl')))


end = time.time()
print(end - start)