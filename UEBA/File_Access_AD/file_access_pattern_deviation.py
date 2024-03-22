#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:03:17 2021
@author: ernestmugambi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
   


# input files
f7 = '/Users/ernestmugambi/Downloads/cloud_gsuite_cynet_07_18_08_17.csv'    # cynet train
f8 = '/Users/ernestmugambi/Downloads/cloud_gsuite_cynet_08_17_08_19.csv'    # cynet test

######### get data set

def get_Trn_Data():
    trnDat = pd.read_csv('/Users/ernestmugambi/Downloads/cloud_gsuite_cynet_07_18_08_17.csv',header = 0)
    print('run get_Trn_Data..1')
    df = trnDat[['actor_email','ipAddress','event_name']]
    print('run get_Trn_Data..2', df)
    #df.drop(df.tail(2).index, inplace=True)              # header is last record (just a quick fix for now)
    #df['proba'] = df['count'].astype(int)/np.sum(df['count'].astype(int))
    #print('run get_Trn_Data..4', df)
    return df

def get_Tst_Data():
    tstDat = pd.read_csv('/Users/ernestmugambi/Downloads/cloud_gsuite_cynet_08_17_08_19.csv')
    print('run get_Tst_Data')
    df = tstDat[['actor_email','ipAddress','event_name']]
    return df

 # create Training probabilities
def create_trn_tbls(trnDat):
    alldat = trnDat
    #alldat['count'] = alldat['count'].astype(int)
    email_trn = alldat.groupby(['actor_email']).size().reset_index()
    email_trn.columns = ['actor_email','count']
    #acct_trn = acct_trn.sort_values(by = ['count'], ascending = False)
    print('actor probs',email_trn)
    email_trn['proba'] = email_trn['count']/np.sum(email_trn['count'])
    email_ip_trn = alldat.groupby(['actor_email','ipAddress']).size().reset_index()
    email_ip_trn.columns = ['actor_email','ipAddress','count']
    email_ip_trn['proba'] = email_ip_trn['count']/np.sum(email_ip_trn['count'])
    ip_event_trn = alldat.groupby(['ipAddress','event_name']).size().reset_index()
    ip_event_trn.columns = ['ipAddress','event_name','count']
    ip_event_trn['proba'] = ip_event_trn['count']/np.sum(ip_event_trn['count'])
    return email_trn, email_ip_trn, ip_event_trn


# Prediction probabilities
def join_Trn_Tst(email_trn, email_ip_trn, ip_event_trn, tstDat):
    alldat = tstDat
    # accounts probability
    email_tst = alldat['actor_email'].drop_duplicates().reset_index()
    email_pred = pd.merge(email_tst, email_trn, how = "left", on = ['actor_email'])
    email_pred = email_pred[['actor_email','proba']]
    email_pred.columns = ['actor_email','proba_email']                        
    # accounts - secId probability
    email_ip_tst = alldat[['actor_email','ipAddress']].drop_duplicates().reset_index()
    email_ip_pred = pd.merge(email_ip_tst, email_ip_trn, how = "left", on = ['actor_email','ipAddress'])
    email_ip_pred = email_ip_pred[['actor_email','ipAddress','proba']]
    email_ip_pred.columns = ['actor_email','ipAddress','proba_email_ip']
    # accounts - srcIp probability
    ip_event_tst = alldat[['ipAddress','event_name']].drop_duplicates().reset_index()
    ip_event_pred = pd.merge(ip_event_tst, ip_event_trn, how = "left", on = ['ipAddress','event_name'])
    ip_event_pred = ip_event_pred[['ipAddress','event_name','proba']]
    ip_event_pred.columns = ['ipAddress','event_name','proba_ip_event']
    # accounts - machineIp probability
    # all probabilities stitched together
    tbl1 = pd.merge(alldat, email_pred, how = 'left', on = 'actor_email')
    tbl2 = pd.merge(tbl1, email_ip_pred, how = 'left', on = ['actor_email','ipAddress'])
    tbl3 = pd.merge(tbl2, ip_event_pred, how = 'left', on = ['ipAddress','event_name'])
    eps = 0.0000000000001
    tbl3 = tbl3.fillna(eps)
    # using fisher probability fusion
    tbl3['f_proba'] = (-2.0 * (np.log10(tbl3['proba_email']) + np.log10(tbl3['proba_email_ip']) +np.log10(tbl3['proba_ip_event'])))
    #tbl4['f_proba'] = tbl4['f_proba'] + 4.0
    return tbl3
    
# Run training and prediction
def main(f_trn,f_tst):
    create_trn_tbls(f_trn)
    join_Trn_Tst(f_tst)
    print("done..")






    
    



