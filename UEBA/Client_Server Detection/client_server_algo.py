#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:29:01 2021
@author: ernestmugambi
"""
import pandas as pd
import numpy as np
#from functools import reduce
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from math import exp
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AgglomerativeClustering


def get_System_Trn_Data():
    """
    get training data
    """
    #df = pd.read_csv('/users/ernestmugambi/Downloads/sm_system_04_14_15.csv',header=0)
    df = pd.read_csv('/users/ernestmugambi/Downloads/omf_system_04_14_15_3.csv',header=0)
    #df = pd.read_csv('/users/ernestmugambi/Downloads/nike_system_5_4_5_1.csv',header=0)
    df.columns = ['eventdate','ip', 'bytesIn', 'bytesOut', 'frequency','connIn','connOut','accessIn']
    del df['eventdate']
    df = df.dropna()
    return df

def get_System_Tst_Data():
    """
    get test data
    """
    #df = pd.read_csv('/users/ernestmugambi/Downloads/sm_system_04_15_16-2.csv',header=0)
    df = pd.read_csv('/users/ernestmugambi/Downloads/omf_system_04_15_16_3.csv',header=0)
    #df = pd.read_csv('/users/ernestmugambi/Downloads/nike_system_5_5_6_1.csv',header=0)
    df.columns = ['eventdate','ip', 'bytesIn', 'bytesOut', 'frequency','connIn','connOut','accessIn']
    del df['eventdate']
    df = df.dropna()
    return df

# data aggregation and min-max normalization
def agg_System_data(df):                                              # OMF    
    df['impact'] = df['connIn'] + df['connOut'] + df['accessIn']
    df['pc_ratio'] = (df['bytesIn'] - df['bytesOut'])/(df['bytesIn'] + df['bytesOut'])
    df['conn_ratio'] = (df['connIn'] + 0.1) / (df['connOut'] + 0.1)
    df['mm_bytesIn'] = (df['bytesIn'] - np.min(df['bytesIn']))/(np.max(df['bytesIn']) - np.min(df['bytesIn']))
    df['mm_bytesOut'] = (df['bytesOut'] - np.min(df['bytesOut']))/(np.max(df['bytesOut']) - np.min(df['bytesOut']))
    df['mm_connIn'] = (df['connIn'] - np.min(df['connIn']))/(np.max(df['connIn']) - np.min(df['connIn']))
    df['mm_connOut'] = (df['connOut'] - np.min(df['connOut']))/(np.max(df['connOut']) - np.min(df['connOut']))
    df['mm_accessIn'] = (df['accessIn'] - np.min(df['accessIn']))/(np.max(df['accessIn']) - np.min(df['accessIn']))
    df['mm_impact'] = (df['impact'] - np.min(df['impact']))/(np.max(df['impact']) - np.min(df['impact']))
    df['mm_frequency'] = (df['frequency'] - np.min(df['frequency']))/(np.max(df['frequency']) - np.min(df['frequency']))
    df['mm_connRatio'] = (df['conn_ratio'] - np.min(df['conn_ratio']))/(np.max(df['conn_ratio']) - np.min(df['conn_ratio']))
    df['pc_ratio'] = df['pc_ratio'].fillna(0.0)
    return df

def euclidian(p,q):
    """
    Compute the euclidian distance between two vectors.
    Input
    -----
    p, q : array-like, dtype=float, shape=n
        counts of hits seen in a network
    """
    p = (np.asarray(p, dtype=np.float))
    q = (np.asarray(q, dtype=np.float))
    return np.sqrt(np.sum((p - q)**2.0))

def compute_sf(kmeans,X):
    """
    Computes the sf metric for a given clusters
    see this paper : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.93.3033&rep=rep1&type=pdf
    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn
    X     :  multidimension np array of data points
    Returns:
    -----------------------------------------
    sf value
    """
    centers = [kmeans.cluster_centers_]  # kmeans clustering centers
    labels  = kmeans.labels_     # number of clusters
    m = kmeans.n_clusters        # size of the clusters
    n = np.bincount(labels)      # size of data set
    N, d = X.shape               # get shape of data set
    main_centroid = np.mean(X)   # centroid for whole data set
    wcd = (1.0 / (N * m)) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])
    bcd = 0
    for i in range(m):
        bcd = bcd + abs((euclidian(centers[0][i],main_centroid) * n[i]))
    bcd = bcd / (m*N)
    sf = 1.0 - 1.0/(exp(exp(bcd-wcd)))
    return sf

# run kmeans clustering on a dataset
def kmeans_cls_opt(df):
    df_s = df[['mm_bytesIn','mm_bytesOut','mm_connIn','mm_connOut','mm_accessIn','mm_impact','mm_frequency','pc_ratio','mm_connRatio']] 
    X = np.array(df_s)
    X = StandardScaler().fit_transform(X) 
    sf = []
    for i in range(2,20):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
        sf.append(compute_sf(kmeans,X))
    print('sf Index',sf)
    plt.plot(sf,'b-*')
    return sf

# run kmeans clustering on training and prediction
def kmeans_run(df,df2,n_clusters):
    df_s = df[['mm_bytesIn','mm_bytesOut','mm_connIn','mm_connOut','mm_accessIn','mm_impact','mm_frequency','pc_ratio','mm_connRatio']] 
    X = np.array(df_s)
    X = StandardScaler().fit_transform(X) 
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    print('centres:',kmeans.cluster_centers_)
    df['labels'] = kmeans.labels_
    #df.to_csv('/users/ernestmugambi/client_server_modeling_V2/sm_test1_1.csv')
    df.to_csv('/users/ernestmugambi/client_server_modeling_V2/omf_test1_1.csv')
    #df.to_csv('/users/ernestmugambi/client_server_modeling_V2/nike_test1_1.csv')
    df_t = df2[['mm_bytesIn','mm_bytesOut','mm_connIn','mm_connOut','mm_accessIn','mm_impact','mm_frequency','pc_ratio','mm_connRatio']]
    Y = np.array(df_t)
    Y = StandardScaler().fit_transform(Y) 
    prediction = kmeans.predict(Y)
    df2['prediction'] = prediction
    #df2.to_csv('/users/ernestmugambi/client_server_modeling_V2/sm_test2_1.csv')
    df2.to_csv('/users/ernestmugambi/client_server_modeling_V2/omf_test2_2.csv')
    #df2.to_csv('/users/ernestmugambi/client_server_modeling_V2/nike_test2_2.csv')
    return 0

def dbscan_fit(df,df2,nclusters):
    df_s = df[['mm_bytesIn','mm_bytesOut','mm_connIn','mm_connOut','mm_accessIn','mm_impact','mm_frequency','pc_ratio','mm_connRatio']] 
    #df_s = df_s.head(50000)
    X = np.array(df_s)
    X = StandardScaler().fit_transform(X) 
    model = AgglomerativeClustering(n_clusters=nclusters)
    clustering = model.fit(X)
    df['labels'] = clustering.labels_
    df.to_csv('/users/ernestmugambi/client_server_modeling_V2/omf_aggl_2.csv')
    df_t = df2[['mm_bytesIn','mm_bytesOut','mm_connIn','mm_connOut','mm_accessIn','mm_impact','mm_frequency','pc_ratio','mm_connRatio']]
    print("Done..1")
    Y = np.array(df_t)
    Y = StandardScaler().fit_transform(Y) 
    prediction = clustering.fit_predict(Y)
    df2['prediction'] = prediction
    #df2.to_csv('/users/ernestmugambi/client_server_modeling_V2/sm_test2_1.csv')
    df2.to_csv('/users/ernestmugambi/client_server_modeling_V2/omf_aggl_3.csv')
    print("Done..2")
    
def main():
    df1 = get_System_Trn_Data()
    df2 = get_System_Tst_Data()
    df11 = agg_System_data(df1)
    df22 = agg_System_data(df2)
    #dbscan_fit(df11,df22,nclusters=3)
    kmeans_run(df11,df22,10)
    return 0
    
# join system table to firewall table based on fw ingress
def join_system_firewall():
    system_df = pd.read_csv('/users/ernestmugambi/client_server_modeling_V2/nike_test1_1.csv',header = 0)
    #system_df = pd.read_csv('/users/ernestmugambi/client_server_modeling_V2/sm_test1.csv',header = 0)
    #system_df = pd.read_csv('/users/ernestmugambi/client_server_modeling_V2/omf_test2_2.csv',header = 0)
    #system_df = system_df[['ip','prediction']]
    system_df = system_df[['ip','labels']].drop_duplicates().reset_index()
    #firewall_df = pd.read_csv('/users/ernestmugambi/Downloads/sm_firewall_cs_04_14_15.csv',header=0)
    #firewall_df = pd.read_csv('/users/ernestmugambi/Downloads/omf_firewall_04_15_16-2.csv',header=0)
    firewall_df = pd.read_csv('/users/ernestmugambi/Downloads/nike_firewall_04_14_15.csv',header=0)
    firewall_df = firewall_df[['dstIp']].drop_duplicates().reset_index()
    df = pd.merge(system_df, firewall_df, how = 'inner', left_on = 'ip', right_on = 'dstIp')
    #grp = df.groupby(['prediction']).size()
    grp = df.groupby(['labels']).size()
    return grp

# tracking ip clusters from day 1 to day 2 
def clusters_join():
    day_1 = pd.read_csv('/users/ernestmugambi/client_server_modeling_V2/sm_test1.csv',header = 0)
    #day_1 = pd.read_csv('/users/ernestmugambi/client_server_modeling_V2/omf_test1_1.csv',header = 0)
    #day_1 = pd.read_csv('/users/ernestmugambi/client_server_modeling_V2/nike_test1_1.csv',header = 0)
    #day_1 = pd.read_csv('/users/ernestmugambi/client_server_modeling_V2/omf_aggl_2.csv',header = 0)
    day_1 = day_1[['ip','labels']]
    day_2 = pd.read_csv('/users/ernestmugambi/client_server_modeling_V2/sm_test2.csv',header = 0)
    #day_2 = pd.read_csv('/users/ernestmugambi/client_server_modeling_V2/omf_test2_2.csv',header = 0)
    #day_2 = pd.read_csv('/users/ernestmugambi/client_server_modeling_V2/nike_test2_2.csv',header = 0)
    #day_2 = pd.read_csv('/users/ernestmugambi/client_server_modeling_V2/omf_aggl_3.csv',header = 0)
    day_2 = day_2[['ip','prediction']]
    joint_ip = pd.merge(day_1, day_2, how = 'inner', on = 'ip')
    cm = confusion_matrix(joint_ip['labels'], joint_ip['prediction'])
    cm_df = pd.DataFrame(cm)
    print(cm_df)
    res = cm_df
    #res.to_csv('/users/ernestmugambi/client_server_modeling_V2/nike_cluster_transition_matrix.csv')
    joint_ip.to_csv('/users/ernestmugambi/client_server_modeling_V2/sm_cls_track.csv')
    
    # joint_ip = pd.merge(day_1, day_2, how = 'inner', on = 'ip') 
    # joint_stats = joint_ip.groupby(['labels','prediction']).size().reset_index()
    # joint_stats.columns = ['before','after','count']
    # joint_stats.to_csv('/users/ernestmugambi/client_server_modeling_V2/cluster_transition.csv')
    return 0    

# distribution of port packet ingress based on cluster
def combine_systems_firewall():
    df_sys = pd.read_csv('/users/ernestmugambi/client_server_modeling_V2/sm_test1.csv',header=0)  # get secops.systems
    df_sys = df_sys[['ip','labels']]
    dff = pd.read_csv('/users/ernestmugambi/Downloads/sm_firewall_04_14_15_validation.csv',header=0)
    dff = dff[['dstIp','dstPort','packets']]
    dff2 = pd.merge(df_sys, dff, how = 'inner', left_on = 'ip', right_on = 'dstIp')
    common_ports = [22.0,23.0,25.0,53.0,80.0,88.0,122.0,123.0,135.0,137.0,161.0,389.0,	443.0,445.0]
    df_ports = pd.DataFrame(common_ports, columns = ['common_ports'])
    df = pd.merge(dff2, df_ports, how = 'inner', left_on = 'dstPort', right_on = 'common_ports')
    grp = df.groupby(['labels','dstPort'])['packets'].sum().reset_index()
    pivot_dst = pd.pivot_table(grp, values='packets', index=['labels'],columns=['dstPort'], aggfunc=np.sum).reset_index()
    return 0

# statistics of port packet ingress based on cluster
def combine_systems_firewall2():
    df_sys = pd.read_csv('/users/ernestmugambi/client_server_modeling_V2/sm_test1.csv',header=0)  # get secops.systems
    df_sys = df_sys[['ip','labels']]
    dff = pd.read_csv('/users/ernestmugambi/Downloads/sm_firewall_04_14_15_validation.csv',header=0)
    dff = dff[['dstIp','dstPort']]
    dff_grp = dff.groupby(['dstIp'])['dstPort'].nunique().reset_index()
    dff2 = pd.merge(df_sys, dff_grp, how = 'inner', left_on = 'ip', right_on = 'dstIp')
    grp = dff2.groupby(['labels'])['dstPort'].agg([min,max,np.mean,np.size])
    pivot_dst = pd.pivot_table(grp, values='packets', index=['labels'],columns=['dstPort'], aggfunc=np.sum).reset_index()
    return 0

    
# how to validate clustering --  assumptions
"""
- servers are called periodically clients randomly (time series)
- servers have lots of incoming connections hence high pagerank/in-centrality
- servers have lot of ports/services/protocols compared to clients
- servers PCR --> 1.0 clients PCR --> -1.0
- servers are on longer and could serve many users/clients at the same time compared to clients

"""


