#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Dec 1 10:26:56 2022
@author: ernestmugambi
"""
import pandas as pd
import numpy as np
from scipy.stats import mode

# set data path
#test_data = '/users/ernestmugambi/Documents/UEBA_Algos/Failed_Logins_AD/ulta_1day.csv'
#test_data = '/users/ernestmugambi/Documents/UEBA_Algos/Failed_Logins_AD/hr_2day.csv'
#test_data = '/users/ernestmugambi/Documents/UEBA_Algos/Failed_Logins_AD/sm_2day.csv'
#test_data = '/users/ernestmugambi/Documents/UEBA_Algos/Failed_Logins_AD/ulta_2day.csv'
#test_data = '/users/ernestmugambi/Documents/UEBA_Algos/Failed_Logins_AD/hrblock_srcip_1day.csv'
test_data = '/users/ernestmugambi/Documents/UEBA_Algos/Failed_Logins_AD/hrblock_machine_1day.csv'


# get test data
def get_TClient_Data():
    df = pd.read_csv(test_data,header=0)
    #df.columns = ['user','Total_Failed','Total_Login','Fail_Rate']
    df = df[['user','Total_Failed','Total_Login','Fail_Rate']]
    # correction to get rid of "infinity"
    df['Total_Login'][df['Total_Login']==0]=1.0
    df['Fail_Rate'] = df['Total_Failed']/df['Total_Login']
    return df

# assuming the distribution is non-unimodal - use this outlier detection routine
def chebyshev_outlier_non_unimodal(p,dat):  
    """
    chebyshev routine for identifying outliers in non-unimodal data
    parameter input - p is required which is the % of outliers expected in data stream
    https://kyrcha.info/2019/11/26/data-outlier-detection-using-the-chebyshev-theorem-paper-review-and-online-adaptation
    """
    k = 1/np.sqrt(p)
    ODV_u = np.mean(dat) + k * np.std(dat)
    outlier_vector = np.zeros(len(dat))
    sum_t = 0
    for i in range(len(outlier_vector)):
        if (dat.values[i] > ODV_u):
            outlier_vector[i] = 1.0
            sum_t = sum_t + 1.0
    print ("outlier count is:",(sum_t+0.0))
    #plot(outlier_vector,'b-*')
    print ("outlier upper limit:", ODV_u)
    return ODV_u

# assuming the distribution is unimodal - use this outlier detection routine
def chebyshev_outlier_unimodal(p,dat):  
    """
    chebyshev routine for identifying outliers in unimodal data
    parameter input - p is required which is the % of outliers expected in data stream
    https://kyrcha.info/2019/11/26/data-outlier-detection-using-the-chebyshev-theorem-paper-review-and-online-adaptation
    """
    #what if dat is empty ?
    k = 2/(3.0*np.sqrt(p))
    B = np.sqrt(np.std(dat)**2.0 + (mode(dat)[0][0] - np.mean(dat))**2.0)
    print("unimodal")
    #print(dat)
    print(mode(dat)[0][0])
    ODV_u = mode(dat)[0][0] + k * B
    outlier_vector = np.zeros(len(dat))
    sum_t = 0
    for i in range(len(outlier_vector)):
        if (dat.values[i] > ODV_u):
            outlier_vector[i] = 1.0
            sum_t = sum_t + 1.0
    print ("outlier count is:",(sum_t+0.0))
    #plot(outlier_vector,'b-*')
    print ("outlier upper limit:", ODV_u)
    return ODV_u

"""
=================================================
outlier detection routines 
=================================================
"""
# sub routine of Chebyshev outlier detection
def run_Chebyshev(dst_type, pval_1, df, select_column):
    p_1 = pval_1
    if dst_type == 'unimodal':
    #unimodal
        outlier_uLimit = chebyshev_outlier_unimodal(p_1, df[select_column])
        outliers = df[df['Fail_Rate'] > outlier_uLimit]      
    else:
    #non_unimodal 
        outlier_uLimit = chebyshev_outlier_non_unimodal(p_1, df[select_column])
        outliers = df[df['Fail_Rate'] > outlier_uLimit]
    print("outlier upper limit", outlier_uLimit)
    print("outlier list\n", outliers)
    return outliers

#  run Chebyshev outlier detection routine
def run_Chebyshev_full(dst_type, pval_1, pval_2, df, select_column):
    p_1, p_2 = pval_1, pval_2        # initialize p-values

    if dst_type == 'unimodal':
    #unimodal
        outlier_uLimit = chebyshev_outlier_unimodal(p_1, df[select_column])
        df2 = df[df[select_column] < outlier_uLimit]
        outlier_uLimit = chebyshev_outlier_unimodal(p_2, df2[select_column])
    else:
    #non_unimodal
        outlier_uLimit = chebyshev_outlier_non_unimodal(p_1, df[select_column])
        df2 = df[df[select_column] < outlier_uLimit]
        outlier_uLimit = chebyshev_outlier_non_unimodal(p_2, df2[select_column])
    return outlier_uLimit

def get_threshold(dst_shape, p_val_init, p_val_second, df, column_1, column_2):    
    fr_threshold = run_Chebyshev_full(dst_shape, p_val_init, p_val_second, df, column_1)
    #median_threshold = run_Chebyshev_full(dst_shape, p_val_init, p_val_second, df, column_2)
    median_threshold = 10
    #print ('best threshold for sample data:', best_threshold)
    return fr_threshold, median_threshold

# calculate final anomalousness scores
def compute_outcome(df,fr_threshold,mn_threshold):
    outliers = df[(df['Fail_Rate'] >= fr_threshold) & (df['Fail_Rate'] >= mn_threshold)]
    outliers = outliers.reset_index(drop=True)
    normal = df[~((df['Fail_Rate'] >= fr_threshold) & (df['Fail_Rate'] >= mn_threshold))]
    normal = normal.reset_index(drop=True)
    max_outliers, max_normal = np.max(outliers['Fail_Rate']), np.max(normal['Fail_Rate'])
    outliers['prediction'] = outliers['Fail_Rate'].apply(lambda x: 75.0 + ((x/max_outliers) * 25.0))
    normal['prediction'] = normal['Fail_Rate'].apply(lambda x: 0.0 + ((x/max_normal) * 75.0))
    all_dat = pd.concat([normal,outliers])
    return all_dat
    
"""
===============================================================================
run FL routine -  Model Logic
--find [machine,user] failure rates that are extreme by identifying outliers
--find [machine,user] entities that have high no. of failed logins (at least 10)
===============================================================================
"""

# run outlier detection routine
def run_Flogins_AD(shape,p_value_1,p_value_2,field_1,field_2):
    test = get_TClient_Data()
    threshold_f1, threshold_mn = get_threshold(shape,p_value_1,p_value_2,test,field_1, field_2)        # run outlier detection
    predictions = compute_outcome(test,threshold_f1, threshold_mn)
    return predictions

def main():
    # set parameters
    shape_1, shape_2, column_name_1, column_name_2 = 'bimodal', 'unimodal', 'Fail_Rate', 'Total_Failed'
    p_value_1 = 0.1                                 # p-value of first outlier detection (very extreme values identified)
    p_value_2 = 0.05                                # p-value of final outlier detection (true outlier boundary identified)
    scores = run_Flogins_AD(shape_2,p_value_1,p_value_2, column_name_1, column_name_2)
    return scores
