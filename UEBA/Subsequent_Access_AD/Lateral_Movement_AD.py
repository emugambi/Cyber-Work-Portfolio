#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 17:07:30 2018
@author: emugambi
"""
import pandas as pd
import numpy as np
historical_data = ''     # file path of historical data to import
test_data = ''           # file path of test data to import
trn_period = 30    #days

def get_TClient_Data():                          # test data - new observations
    df = pd.read_csv(test_data,header=0)
    df.columns = ['eventdate','source','machine','user','srcIp','freq']
    df = df[['user','srcIp','machine','freq']].drop_duplicates()
    df.columns = ['user','s_computer','d_computer','hits']
    return df

def get_HClient_Data():                          # history data
    df = pd.read_csv(historical_data,header=0)
    df.columns = ['eventdate','source','machine','user','srcIp','freq']
    df = df[['user','srcIp','machine','freq']].drop_duplicates()
    df.columns = ['user','s_computer','d_computer','hits']
    return df

def compute_discrete_probability(outcomes,t_length):# compute probability of machine-accesses
    outcomes = outcomes[['user','s_computer','hits']]
    outcomes['hits'] = np.ceil(outcomes['hits']/t_length)
    prob = outcomes.groupby(['hits']).size().reset_index()
    prob.columns = ['hits','freq']
    prob['llh'] = prob['freq']/sum(prob['freq'])
    return prob


# compute entropy
def shannon(p):
    """shannon entropy for discrete distributions
    Parameters
    ----------
    p : array-like, dtype=float, shape=n
        counts of hits seen in a network
    """
    eps = 0.00000000001   # v small number to get rid of zeros
    p = (np.asarray(p, dtype=float)/np.sum(p)) + eps
    return np.sum(np.where(p != 0, p * np.log2(p), 0))

def max_entropy(grouped):
    """
    group by user|s_computer - no. of uniques
    """
    eps = 0.0000000001
    grouped = grouped.groupby(['user','s_computer'])
    grp = grouped.agg({'d_computer' : lambda x: len(np.unique(x)) * (1.0/len(np.unique(x)) * np.log2(1.0/len(np.unique(x))))})
    grp = grp.reset_index()
    grp.columns = ['user','s_computer','max_entropy']
    grp['max_entropy'] = grp['max_entropy'] + eps
    return grp

def shannon_entropy(grouped):
    """
    group by user|s_computer - no. of uniques
    """
    eps = 0.0000000001
    grp = grouped.groupby(['user','s_computer','d_computer'])
    grp = grp.agg({'hits' : np.sum})
    grp = grp.reset_index()
    grp2 = grouped.groupby(['user','s_computer'])
    grp2 = grp2.agg({'hits' : np.sum})
    grp2 = grp2.reset_index()
    grp3 = pd.merge(grp,grp2, how = "inner", on = ['user','s_computer'])
    grp3['ll'] = (grp3['hits_x']+eps)/(grp3['hits_y']+eps)
    grp4 = grp3.groupby(['user','s_computer'])
    grp4 = grp4.agg({'ll' : lambda x : shannon(x)})
    grp4 = grp4.reset_index()
    return grp4


def run_shannon_daily(df):
    out = shannon_entropy(df)
    out2 = max_entropy(df)
    out = pd.merge(out, out2, how = 'inner', on = ['user','s_computer'])
    out['pct_shannon'] = out['ll']/out['max_entropy']
    #out['ll']
    #out['max_entropy']
    return out

"""
=============================================================
run LM deviation routine - Model 1
--compute [user,src_IP]--> relative entropy calculation
--compute [user,src_IP]--> connection count probability
--combine both outcomes using fisher method
=============================================================
""" 
def run_LM_AD():        
    eps = 0.000000001                             # run the whole routine
    t_period = trn_period
    xdf = get_HClient_Data()
    prob = compute_discrete_probability(xdf, t_period)
    ydf = get_TClient_Data()
    tcases = ydf[['user','s_computer','hits']]
    shannon_df = run_shannon_daily(ydf)
    out_x = pd.merge(tcases, shannon_df, how = 'inner', on = ['user','s_computer'] )
    out_y = pd.merge(out_x, prob, how = 'inner', on = 'hits')
    out_y['llh'] = out_y['llh'] + eps
    out_y['fisher'] = ((-2.0 * (np.log10(out_y['llh']) + np.log10(out_y['pct_shannon']))))
    return out_y

def agg_Users(outcomes):
    agg_Users = outcomes.groupby(['user'])['fisher'].max().reset_index()
    agg_Users.columns = ['user','fisher_max']
    return agg_Users

def main():
    fisher_scores = run_LM_AD()
    user_fisher_score = agg_Users(fisher_scores)
    return user_fisher_score
    
