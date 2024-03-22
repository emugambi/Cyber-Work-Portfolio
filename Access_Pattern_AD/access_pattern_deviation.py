#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 09:34:02 2021
@author: ernestmugambi
"""
import pandas as pd
import numpy as np

historical_data = ''                                 # file path to the training data
test_data = ''                                        # file path to the test data 
 

def get_TClient_Data():                          # test data - new observations
    df = pd.read_csv(test_data,header=0)
    df.columns = ['eventdate','source','machine','user','srcIp','freq']
    df = df[['machine','user','srcIp']].drop_duplicates()
    return df

def get_HClient_Data():                          # history data
    df = pd.read_csv(historical_data,header=0)
    df.columns = ['eventdate','source','machine','user','srcIp','freq']
    df = df[['machine','user','srcIp']].drop_duplicates()
    return df

def jaccard_1(X,Y):   # original jaccard
    if (len(X) == 0) or (len(Y) == 0):
        return 0
    set_X, set_Y = set(X), set(Y)
    return len(set_Y.intersection(set_X))/len(set_Y.union(set_X))

def jaccard_2(X,Y):   # improved jaccard
    if (len(X) == 0) or (len(Y) == 0):
        return 0
    set_X, set_Y = set(X), set(Y)
    return len(set_Y.difference(set_X))/len(set_Y)

"""
sql like routine for fast computation of Jaccard index to avoid slow FOR-loops
inputs - training and test set: entity[user,src_IP] --> machine_IP
"""
def compute_fast_jaccard(Train,Test):            # run fast jaccard routine
    Train['count'],Test['count'] = 1.0,1.0
    grp_Test = Test.groupby(['srcIp','user'])['count'].sum().reset_index()
    grp_Test.columns = ['srcIp','user','machine_total']
    common_entities = Train.merge(Test, how = 'right', on = ['srcIp','user','machine'])
    grp_cmn_entities = common_entities.groupby(['srcIp','user'])['count_x'].sum().reset_index()
    grp_cmn_entities.columns = ['srcIp','user','common_machines']
    all_test = pd.merge(grp_Test, grp_cmn_entities, how = 'inner', on = ['srcIp','user'])
    all_test['jaccard'] = all_test['common_machines']/all_test['machine_total'] 
    return all_test

"""
compute entity-machine connection probability
"""
def compute_discrete_probability(outcomes):# compute probability of machine-accesses
    outcomes = outcomes[['machine_total']]
    prob = outcomes.groupby(['machine_total']).size().reset_index()
    prob.columns = ['machine_total','machine_freq']
    prob['llh'] = prob['machine_freq']/len(prob['machine_freq'])
    return prob

# toy data set for testing
def test_cases():                               # test cases for jaccard routine
    df1 = pd.DataFrame({'srcIp': ['S1', 'S2'], 'user': ['U1', 'U2'],'machine':['M1','M2']})
    print(df1)
    df2 = pd.DataFrame({'srcIp': ['S1', 'S2','S2'], 'user': ['U1', 'U2','U2'],'machine':['M1','M2','M3']})
    print(df2)
    outcome = compute_fast_jaccard(df1,df2)
    print(outcome)
    df2 = pd.DataFrame({'srcIp': ['S1', 'S2','S2','S2','S2','S2'], 'user': ['U1', 'U2','U2','U2','U2','U2'],'machine':['M1','M2','M3','M4','M8','M7']})
    outcome = compute_fast_jaccard(df1,df2)
    print(outcome)
    
"""
=============================================================
run AP deviation routine - Model 1
--compute [user,src_IP]--> machine Jaccard Index
--compute [user,src_IP]--> machine probability
--combine both outcomes using fisher method
=============================================================
""" 
def run_APatterns_AD():                                     # run the whole routine
    eps = 0.000000001
    bias =  -2.0 * (np.log10(100.0) + np.log10(1.0))
    df_History = get_HClient_Data()
    df_Test_cases = get_TClient_Data()
    jaccard_outcomes = compute_fast_jaccard(df_History,df_Test_cases)
    probs = compute_discrete_probability(jaccard_outcomes)
    outcomes_tbl = pd.merge(jaccard_outcomes, probs, how = 'inner', on = 'machine_total')
    outcomes_tbl['jaccard'] = outcomes_tbl['jaccard'] + eps
    outcomes_tbl['p-value'] = outcomes_tbl['llh']*outcomes_tbl['jaccard']
    outcomes_tbl['fisher'] = ((-2.0 * (np.log10(outcomes_tbl['llh']) + np.log10(outcomes_tbl['jaccard']))) - bias)
    return outcomes_tbl

"""
aggregate scores at user level
"""
def agg_Users(outcomes):
    agg_Users = outcomes.groupby(['user'])['fisher','machine_total','common_machines'].sum().reset_index()
    agg_Users.columns = ['user','fisher_sum','machine_total','common_machines']
    return agg_Users

def main():
    scores = run_APatterns_AD()
    aggregated_scores = agg_Users(scores)
    print("list of abnormal access behavior:\n\n", aggregated_scores.sort_values('fisher_sum',ascending=False).head(10))
    return aggregated_scores
    


    
    
