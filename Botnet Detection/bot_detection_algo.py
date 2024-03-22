#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:52:30 2019
@author: emugambi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
#from functools import reduce 
import random
#from scipy import stats

# bring in the data
def get_Data(day):
    df = pd.read_csv('/Users/emugambi/botnet_traffic/Data/lanl_nflow_a%s' %day, sep = ",", header = None, 
                           names=('time','duration','s_computer','s_port','d_computer','d_port',
                                  'protocol','packets','bytes'))
    df = df[['time','s_computer','d_computer','packets']]
    return df

def get_Data_2(day):
    df = pd.read_csv('/Users/emugambi/botnet_traffic/Data/lanl_nflow_a%s' %day, sep = ",", header = None, 
                           names=('time','duration','s_computer','s_port','d_computer','d_port',
                                  'protocol','packets','bytes'))
    #df = df[['time','s_computer','d_computer','packets']]
    return df

def shannon(p):
    """
    shannon entropy for discrete distributions
    Parameters
    ----------
    p : array-like, dtype=float, shape=n
        counts of hits seen in a network
    """
    eps = 0.00000000001   # v small number to get rid of zeros
    p = (np.asarray(p, dtype=np.float)/np.sum(p)) + eps
    return -1.0*(np.sum(np.where(p != 0, p * np.log2(p), 0)))

def count_diffs(d_hist,d_new):
    """
    counting rare users and their appearances in hosts from time to time
    """
    hist_set, new_set = set(d_hist), set(d_new)
    new_items = new_set.difference(hist_set)
    #all_items = set(d_hist + d_new)
    return (len(new_items)+0.0)/(len(new_set)+0.0001)

def grp_quantity(dat,value):
    out = dat.groupby([value]).size().reset_index()
    return out[0]

# Computes g*
def g_star(Pxx):
    m = len(Pxx)
    max_ = max(Pxx)
    var = (1.0/(2.0*m))*sum(Pxx)
    return max_/var

# Computes z_alpha
def z_alpha(alpha, m):
    c = 1.0-alpha
    z = -2.0*np.log(1.0-c**(1.0/m))
    return z

# computes max_pxx, uses z_alpha function
def max_pxx(df_agg):
    alpha = 0.001
    f, pxx = signal.periodogram(df_agg)
    m, max_ = len(pxx), np.max(pxx)
    var = (1.0/(2.0*m))*sum(pxx)
    z = z_alpha(alpha, m)
    return max_, len(f), z, max_/var

def pxx(df_agg):
    f, pxx = signal.periodogram(df_agg)
    return f, pxx
    

# get traffic from a specific edge
def generate_hourly_multiedge_behavior(dat, this_src, this_dst):
    """
    # generates data segmented based on hosts of interest
    """
    #host = get_Data()                                                      # get your data in
    dat_src = dat[dat['s_computer']==this_src]   
    dat_src = dat_src[dat_src['d_computer']==this_dst]
    dat_dst = dat[dat['s_computer']==this_dst]   
    dat_dst = dat_dst[dat_dst['d_computer']==this_src]                       # select host
    out = pd.concat([dat_src,dat_dst])                         
    return out

# compare effect of intervals on pxx
def get_pxx(dat,src_computer,dst_computer,method):
    intervals = ['1s','2s','3s','4s','5s','6s','7s','8s','9s','10s','11s','12s','13s','14s','15s']
    edge_pxx = {}
    #s,d = 'C17693','C5074'
    s,d = src_computer,dst_computer
    df = dat
    edge_dat = generate_hourly_multiedge_behavior(df, s, d)
    #edge_dat['packets'] = ((edge_dat['packets'])-np.min(edge_dat['packets']))/(np.max(edge_dat['packets'])-np.min(edge_dat['packets']))
    edge_dat['new_time'] = pd.to_datetime(edge_dat['time'], unit='s')
    edge_dat.index = edge_dat['new_time']
    del edge_dat['time']
    del edge_dat['new_time']
    for interval in intervals:
        if method == 'packets':
            new_data = list(edge_dat.resample(interval).sum()['packets'])
            new_data_2 = [1 if x > 0 else 0 for x in new_data]     # tweak to reduce reliance on quantity of packets
        elif method == 'bytes':
            new_data = list(edge_dat.resample(interval).sum()['bytes'])
        h1, h2, h3, h4 = max_pxx(new_data_2)  # h1 = max periodogram, h2 = sample size , h3 = z-statistic , h4 = g* statistic 
        print (interval)
        edge_pxx[interval] = [h1,h2,h3,h4]
    out_df = pd.DataFrame.from_dict(edge_pxx)
    out_df = out_df.T
    out_df.columns = ['max_pxx','samples','z_const','g_x']
    out_df = out_df[out_df['g_x'] >= out_df['z_const']]
    if len(out_df) > 0:
        out_df = out_df.sort_values(by = 'max_pxx', ascending = False)
        out_df = out_df.iloc[0]
    #out_df.to_csv('/Users/emugambi/botnet_traffic/edge_17693_5074_pckts_v2$.csv')
    #print(out_df)
    return out_df

# run individual edges thru periodogram computation
def compute_pxx(data,lst,qty):
    all_pxx = {}
    for i in range(len(lst)):
        all_pxx[lst['s_computer'].iloc[i],lst['d_computer'].iloc[i]] = get_pxx(data,lst['s_computer'].iloc[i],lst['d_computer'].iloc[i],qty)
        print('edge no:',i)
    return all_pxx
        
# identify unidirectional edges     
def unidirectionality_test(data):
    d_x = data[['s_computer','d_computer']].drop_duplicates()
    d_y = d_x.copy()          # duplicate
    d_z = pd.merge(d_x, d_y, how = 'outer', left_on=['s_computer','d_computer'], right_on=['d_computer','s_computer']) # outer join
    d_z = d_z.fillna(0)
    d_z_0 = d_z[d_z['s_computer_y'] == 0]
    d_z_0 = d_z_0[['s_computer_x','d_computer_x']]
    d_z_0.columns = ['s_computer','d_computer']
    return d_z_0

# identify edges with repeated "similar byte" sequences
def byte_behavior(which_dat):
    df = get_Data_2(which_dat)
    dat = df[['s_computer','d_computer','packets','bytes']]
    dat['pckts_per_byte'] = np.floor(dat['bytes']/dat['packets'])
    out = dat.groupby(['s_computer','d_computer','pckts_per_byte']).size().reset_index()
    out.columns = ['s_computer','d_computer','pckts_per_byte','frequency']
    out = out.sort_values('frequency',ascending = False)
    out = out[out['frequency']>1000.0*np.median(out['frequency'])]
    # out.to_csv('/Users/emugambi/botnet_traffic/day_%s_byte_frq.csv' %which_dat)
    out = out[['s_computer','d_computer']].drop_duplicates()
    return out
    
# run botnet algo for each days traffic 
def run_detection_methods(which_day):
    day = which_day
    dat = get_Data(day)
    bb = byte_behavior(which_day)
    uni_data = unidirectionality_test(dat)
    uni_data_final = pd.merge(uni_data, bb, how = 'inner', on = ['s_computer','d_computer'])
    high_pxx_segment = compute_pxx(dat,uni_data_final,'packets')
    pxx_result = pd.DataFrame.from_dict(high_pxx_segment)
    pxx_result = pxx_result.T
    #pxx_result.to_csv('/Users/emugambi/botnet_traffic/results/day_%s_pxx_byte_output_001.csv' %day)
    return pxx_result

def run_cluster_methods(which_day,src,dst):
    #intervals = ['1s','2s','3s','4s','5s','6s','7s','8s','9s','10s','11s','12s','13s','14s','15s']
    interval = '1s'
    #edge_pxx = {}
    day = which_day
    dat = get_Data(day)
    edge_dat = generate_hourly_multiedge_behavior(dat, src, dst)
    edge_dat['new_time'] = pd.to_datetime(edge_dat['time'], unit='s')
    edge_dat.index = edge_dat['new_time']
    del edge_dat['time']
    del edge_dat['new_time']
    new_data = list(edge_dat.resample(interval).sum()['packets'])
    new_data_2 = [1 if x > 0 else 0 for x in new_data]     # tweak to reduce reliance on quantity of packets
    frq, pxx_ = pxx(new_data_2)  # h1 = max periodogram, h2 = sample size , h3 = z-statistic , h4 = g* statistic 
    print (interval)
    #edge_pxx[interval] = [frq, pxx_]
    plt.semilogy(frq, pxx_)
    plt.show()
    return frq, pxx_
    
# plot distribution of edges with high pxx for validation
def edge_traffic_dist(what_day,src,dst):
    dat = get_Data(what_day)
    edge_dat = generate_hourly_multiedge_behavior(dat, src, dst)
    edge_diff = edge_dat['time'].diff()
    edge_diff = edge_diff.reset_index()
    #edge_diff.columns = ['t_diff']
    out = edge_diff.groupby(['time']).size().reset_index()
    out.columns = ['time','counts']
    out = out.sort_values('counts',ascending=False)
    print(edge_diff['time'].describe())
    print(out)
    out.plot.bar(x='time',y='counts',rot=0)
    #out.to_csv('/Users/emugambi/botnet_traffic/results/charts/day_%s_C1015_C15487_time_dist.csv' %what_day)
    return out

def get_port_traffic(dat,lst,kind):
    port_rps = []
    for i in range(len(lst)):
        out = dat[dat['s_computer']==lst['source'][i]]
        out = out[out['d_computer']==lst['destination'][i]]
        if kind == 'source_port':
            port_rps.append(compute_sport_repeatability(out))
        else:
            port_rps.append(compute_dport_repeatability(out))
        print(i)
    lst[kind] = port_rps
    return lst

# source port repeatability
def compute_sport_repeatability(dat):
    time_diffs = [0]
    times = np.unique(dat['time'])
    for i in range(len(times)-1):
        x = np.unique(dat[dat['time']== times[i]]['s_port'])
        y = np.unique(dat[dat['time']== times[i+1]]['s_port'])
        time_diffs.append(count_diffs(x,y))
    return np.median(time_diffs)

# source port repeatability
def compute_dport_repeatability(dat):
    time_diffs = [0]
    times = np.unique(dat['time'])
    for i in range(len(times)-1):
        x = np.unique(dat[dat['time']== times[i]]['d_port'])
        y = np.unique(dat[dat['time']== times[i+1]]['d_port'])
        time_diffs.append(count_diffs(x,y))
    return np.median(time_diffs)
    #return time_diffs

def run_Scans(day):
    dat = get_Data_2(day)
    dat['time'] = np.floor(dat['time']/3600.0)
    df = pd.read_csv('/Users/emugambi/botnet_traffic/results/day_%s_pxx_byte_output_001.csv' %day)
    #df = df.head(10)
    out = get_port_traffic(dat,df,'dst_port')
    out.to_csv('/Users/emugambi/botnet_traffic/scan_behavior/day_%s_dport_scans.csv' %day)
    
def get_daily_trf(day,src,dst):
    day_x = get_Data_2(day)
    day_x = day_x[day_x['s_computer'] == src]
    day_x = day_x[day_x['d_computer'] == dst]
    return day_x

def compare_daily_scans(day_one,day_two,src,dst):
    day_1 = get_daily_trf(day_one,src,dst)    
    day_1_ports = pd.unique(day_1['d_port'])
    day_2 = get_daily_trf(day_two,src,dst)    
    day_2_ports = pd.unique(day_2['d_port'])
    return count_diffs(day_1_ports,day_2_ports)

def dport_dist(day):
    dat = get_Data_2(day)
    #dat = dat[['s_computer','d_computer','s_port','d_port']]
    dat_2 = dat[['s_computer','d_computer','d_port']]
    dat_2 = dat_2.drop_duplicates().reset_index()
    dport = dat_2.groupby(['d_port'])['s_computer','d_computer'].size().reset_index()
    dport.columns = ['d_port','count']
    dport = dport[dport['count']==1]
    #dat_3 = dat[['s_computer','d_computer','d_port']].drop_duplicates()
    out = pd.merge(dat_2, dport, how = 'inner', on = ['d_port'])
    out = out.groupby(['s_computer'])['d_port'].size().reset_index()
    out.columns = ['s_computer','uniq_dports']
    #out = out.groupby(['count']).size().reset_index()
    #out.to_csv('/Users/emugambi/botnet_traffic/scan_behavior/day_%s_rare_dports_2.csv' %day)
    print(out)
    return out

def dport_novelty(day, host_o, host_i):
    data = get_Data_2(day)
    target = data[data['s_computer']==host_o]
    target = target[target['d_computer']==host_i]
    population = data[data['s_computer']!=host_o]
    population = population[population['d_computer']!=host_i]
    print("population_size:",len(np.unique(population['d_port'])))
    print("target_size:",len(np.unique(target['d_port'])))
    return count_diffs(np.unique(population['d_port']),np.unique(target['d_port']))

def run_dport_novelty(s_host,d_host):
    res = []
    days = ['b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    for d in days:
        res.append(dport_novelty(d, s_host, d_host))
        print("done:",d)
    return res
        
x = run_dport_novelty('C1340','C787')
y = run_dport_novelty('C3871','C23147')
z = run_dport_novelty('C1015','C11114')
a = run_dport_novelty('C6177','C15348')
b = run_dport_novelty('C5474','C1970')
    
def run_rand_sample():
    out = {}
    d = get_Data_2('b')
    d = d[['s_computer','d_computer']]
    d = d.drop_duplicates().reset_index()
    samples = random.sample(range(len(d)),10)
    x = d.iloc[samples]
    #for i in range(len(x)):
    for i in range(len(x)):
        out[x['s_computer'].iloc[i], x['d_computer'].iloc[i]] = run_dport_novelty(x['s_computer'].iloc[i], x['d_computer'].iloc[i])
        out_df = pd.DataFrame.from_dict(out)
        out_df = out_df.T
        out_df.to_csv('/Users/emugambi/botnet_traffic/scan_behavior/random_edges_dports_2_%s.csv' %i)
        print("done:", i)
    return 0
        
    
    
    
    

    
    
    
