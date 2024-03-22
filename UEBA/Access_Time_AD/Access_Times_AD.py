#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 06:20:04 2017
@author: emugambi
comparing the performance of different statistical measures in detecting time of access behavior change
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from numpy import linspace,hstack
import scipy as scp
import pandas as pd
from scipy.stats import wasserstein_distance

# negative or positive ?
def sign(x):
    return abs(x) == x


def shannon(p):
    """shannon entropy for discrete distributions
    Parameters
    ----------
    p : array-like, dtype=float, shape=n
        counts of hits seen in a network
    """
    eps = 0.00000000001   # v small number to get rid of zeros
    p = (np.asarray(p, dtype=np.float)/np.sum(p)) + eps
    return np.sum(np.where(p != 0, p * np.log2(p), 0))

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
        counts of hits seen in a network
    """
    eps = 0.00000000001   # v small number to get rid of zeros
    p = (np.asarray(p, dtype=np.float)/np.sum(p)) + eps
    q = (np.asarray(q, dtype=np.float)/np.sum(q)) + eps
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def wt_kl(p, q, wt):
    """weighted Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
        counts of hits seen in a network
    """
    eps = 0.00000000001   # v small number to get rid of zeros
    p = (np.asarray(p, dtype=np.float)/np.sum(p)) + eps
    q = (np.asarray(q, dtype=np.float)/np.sum(q)) + eps
    wt = np.asarray(wt, dtype=np.float)
    return np.sum(np.where(p != 0, wt * p * np.log(p / q), 0))

def kl_scipy(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    using scipy matrix computation routine
    Parameters
    ----------
    p, q : matrix of events
        counts of hits seen in a network
    """
    eps = 0.00000000001   # v small number to get rid of zeros
    p,q = p + eps, q + eps
    return scp.stats.entropy(p.transpose(), q.transpose(), base=None)

def jsdiv(p, q):
    """Compute the Jensen-Shannon divergence between two probability distributions.
    Input
    -----
    p, q : array-like, dtype=float, shape=n
        counts of hits seen in a network
    """
    eps = 0.00000000001   # v small number to get rid of zeros
    p = (np.asarray(p, dtype=np.float)/np.sum(p)) + eps
    q = (np.asarray(q, dtype=np.float)/np.sum(q)) + eps
    m = 0.5 * (p + q)
    return 0.5 * (kl(p, m) + kl(q, m))

def wt_jsdiv(p, q, wt):
    """Compute the weighted Jensen-Shannon divergence between two probability distributions.
    Input
    -----
    p, q : array-like, dtype=float, shape=n
        counts of hits seen in a network
    """
    eps = 0.00000000001   # v small number to get rid of zeros
    q = (np.asarray(q, dtype=np.float)) + eps
    wt = (np.asarray(wt, dtype=np.float)) + eps
    wt_p = wt * p
    p = (np.asarray(wt_p, dtype=np.float)/np.sum(wt_p)) + eps
    q = (np.asarray(q, dtype=np.float)/np.sum(q)) + eps
    #wt = (np.asarray(wt, dtype=np.float)/np.sum(wt)) + eps
    m = 0.5 * (p + q)
    return 0.5 * (kl(p, m) + kl(q, m))

def jeffery(p, q):
    """Compute the Jefferey divergence between two probability distributions.
    Input
    -----
    p, q : array-like, dtype=float, shape=n
        counts of hits seen in a network
    """
    eps = 0.00000000001   # v small number to get rid of zeros
    p = (np.asarray(p, dtype=np.float)/np.sum(p)) + eps
    q = (np.asarray(q, dtype=np.float)/np.sum(q)) + eps
    m = 0.5 * (p + q)
    return np.sum(p * np.log(p / m) + q * np.log(q / m))

def sibson(p, q):
    """Compute the Sibson divergence between two probability distributions.
    Input
    -----
    p, q : array-like, dtype=float, shape=n
        counts of hits seen in a network
    """
    eps = 0.00000000001   # v small number to get rid of zeros
    p = (np.asarray(p, dtype=np.float)/np.sum(p)) + eps
    q = (np.asarray(q, dtype=np.float)/np.sum(q)) + eps
    return 0.5 * (kl(p,0.5*(p+q)) + kl(q,0.5*(p+q)))

def hellinger(p, q):
    """Compute the Hellinger divergence between two probability distributions.
    Input
    -----
    p, q : array-like, dtype=float, shape=n
        counts of hits seen in a network
    """
    eps = 0.00000000001   # v small number to get rid of zeros
    p = (np.asarray(p, dtype=np.float)/np.sum(p)) + eps
    q = (np.asarray(q, dtype=np.float)/np.sum(q)) + eps
    return np.sqrt(np.sum((np.sqrt(p)-np.sqrt(q))**2.0))

def euclidian(p,q):
    """Compute the euclidian distance between two vectors.
    Input
    -----
    p, q : array-like, dtype=float, shape=n
        counts of hits seen in a network
    """
    p = (np.asarray(p, dtype=np.float))
    q = (np.asarray(q, dtype=np.float))
    return np.sqrt(np.sum((p - q)**2.0))
    
def manhattan(p,q):
    """Compute the manhattan alias city-block between two vectors.
    Input
    -----
    p, q : array-like, dtype=float, shape=n
        counts of hits seen in a network
    """
    p = (np.asarray(p, dtype=np.float))
    q = (np.asarray(q, dtype=np.float))
    return np.sum(abs(p - q))

def stouffer(z_scores):
    """Useful for aggregating z-scores - also akin to Fisher's method
    check out:https://en.wikipedia.org/wiki/Fisher%27s_method#Relation_to_Stouffer.27s_Z-score_method
    """
    return np.sum(np.array(z_scores))/np.sqrt(len(z_scores))

def corr(X,Y):
    """ correlation coefficient between two data streams """
    x = np.array(X)
    y = np.array(Y)
    cor = np.corrcoef(x,y)[1][0]
    if sign(cor):
        return 1.0 -  cor
    else:
        return cor + 1.0

def contigency(X,Y):
    """Compute the chi-square contgency probability between two vectors.
    Input
    -----
    X, Y : array-like, dtype=float, shape=n
        counts of logins per hour in a 24 hour period
    output : probability of new logins compared with history
    """
    out = np.matrix((X,Y))+0.0
    expected = np.matrix((X,Y))+0.0
    rows = len(out)
    cols = np.size(out)/rows
    sum_cols = out.sum(0)
    sum_rows = out.sum(1)
    for i in range(rows):
            expected[i,:] = (sum_rows[i]*sum_cols)/np.sum(sum_rows)
    out_n = np.nansum((np.square(out - expected))/expected)
    df = (rows - 1.0)*(cols - 1.0)
    #chi = chisqprob(out_n, df)
    chi = scp.stats.distributions.chi2.sf(out_n,df)
    return (1.0 - chi)

def wasserstein(X,Y):
    """Compute wasserstein distance AKA "earth-movers distance"
    Input
    -----
    X, Y : array-like, dtype=float, shape=n
        counts of logins per hour in a 24 hour period
    output : "how much need to be moved to turn one dist into another"
    """
    return wasserstein_distance(X,Y)
            
def wt_mean(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)
        
def compare_distances():
    """
    hist - contains 24 hour profile vector of login counts
    curr_* - new 24 hour vector of login counts 
    """
    out = []
    hist = [0,0,0,0,0,0,0,0,2,5,6,5,2,7,3,4,2,1,2,1,0,0,0,0]
    curr_1 = [0,0,0,0,0,0,0,1,2,5,6,5,2,7,3,4,2,1,2,1,0,0,0,0]
    curr_2 = [0,0,0,0,0,0,0,2,2,5,6,5,2,7,3,4,2,1,2,1,0,0,0,0]
    curr_3 = [0,0,0,0,0,0,1,2,2,5,6,5,2,7,3,4,2,1,2,1,0,0,0,0]
    curr_4 = [0,0,0,0,0,1,2,2,2,5,6,5,2,7,3,4,2,1,2,1,2,0,0,0]
    curr_5 = [0,0,0,0,2,1,2,2,2,5,6,5,2,7,3,4,2,1,2,1,2,1,0,0]
    curr_6 = [0,0,0,2,2,1,2,2,2,5,6,5,2,7,3,4,2,1,2,1,2,1,0,0]
    curr_7 = [0,0,0,2,2,1,2,2,2,5,6,5,2,7,3,4,2,1,2,1,2,2,1,0]
    curr_8 = [0,0,0,2,2,1,2,2,2,5,6,5,2,7,3,4,2,1,2,1,2,2,1,1]
    curr_9 = [0,0,1,2,2,1,2,2,2,5,6,5,2,7,3,4,2,1,2,1,2,2,1,1]
    curr_10 = [1,0,1,2,2,1,2,2,2,5,6,5,2,7,3,4,2,1,2,1,2,2,1,1]
    curr_11 = [0,1,1,2,2,1,2,2,2,5,6,5,2,7,3,4,2,1,2,1,2,2,1,1]
    curr_12 = [1,1,1,2,2,1,2,2,2,5,6,5,2,7,3,4,2,1,2,1,2,2,1,1]
    curr_13 = [10,1,1,2,2,1,2,2,2,5,6,5,2,7,3,4,2,1,2,1,2,2,1,1]
    curr_14 = [1,1,1,2,2,1,2,2,2,5,6,5,2,7,3,4,2,1,2,1,2,2,1,10]
    curr_15 = [1,1,1,2,2,1,2,2,2,5,6,2,2,7,3,4,2,1,2,1,2,2,1,10]
    curr_16 = [1,1,1,2,2,1,2,2,2,5,1,2,2,7,3,4,2,1,2,1,2,2,1,10]
    curr_17 = [1,1,1,2,2,1,2,2,2,5,1,2,2,7,3,4,2,1,2,1,2,2,1,1]
    curr_18 = [1,1,1,2,2,1,2,2,2,0,1,2,2,7,3,4,2,1,2,1,2,2,1,1]
    curr_19 = [1,1,1,2,2,1,2,2,2,0,1,2,2,0,3,4,2,1,2,1,2,2,1,1]
    curr_20 = [1,1,1,2,2,1,2,2,2,0,1,2,2,0,0,0,2,1,2,1,2,2,1,1]
    curr_21 = [1,1,1,2,2,1,2,2,2,0,1,0,0,0,0,0,2,1,2,1,2,2,1,1]
    curr_22 = [1,1,1,2,2,1,2,2,0,0,0,0,0,0,0,0,0,1,2,1,2,2,1,1]
    curr_23 = [1,1,1,2,2,1,2,2,0,0,0,0,10,0,0,0,0,0,2,1,2,2,1,1]
    curr_24 = [1,1,1,2,2,1,0,0,0,0,0,0,1,0,0,0,2,1,2,1,2,2,1,1]
    curr_25 = [1,1,1,2,2,0,0,0,0,0,0,0,10,0,0,0,0,0,2,1,2,2,1,1]
    curr_26 = [1,1,1,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,2,2,1,1]
    curr_27 = [1,1,21,20,12,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,2,2,1,1]
    curr_28 = [11,10,21,20,12,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,2,2,1,1]
    curr_29 = [11,10,21,20,12,0,0,0,0,0,0,0,0,0,0,0,0,0,2,11,12,2,11,11]
    curr_30 = [11,10,21,20,12,0,0,0,0,0,0,0,0,0,0,0,0,0,20,10,10,20,10,10]
    curr_31 = [0,0,0,0,0,0,0,0,0,0,0,0,10,0,0,0,2,1,2,1,2,2,1,1]
    curr_32 = [0,0,0,0,0,0,0,0,0,0,0,0,10,0,0,0,20,10,20,10,20,20,10,10]
    curr_33 = [1,1,1,2,2,1,2,2,2,0,0,0,10,0,0,0,0,0,0,0,0,0,0,0]
    curr_34 = [1,1,1,2,2,1,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    curr_35 = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    curr_36 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    curr_37 = [1000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    curr_38 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1000]
    curr_39 = [0,0,0,0,0,0,0,0,0,0,0,1000,0,0,0,0,0,0,0,0,0,0,0,0]
    curr_40 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    curr_41 = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    curr_42 = [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    curr_43 = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    curr_44 = [0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0]
    curr_45 = [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    curr_46 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]
    curr_47 = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
    curr_48 = [1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2]
    curr_49 = [1,1,1,2,2,1,2,2,2,0,0,0,100,0,0,0,2,1,2,1,2,2,1,1]
    curr_50 = [1,1,1,2,2,1,2,2,2,0,0,0,1000,0,0,0,2,1,2,1,2,2,1,1]
    curr_51 = [1,1,1,2,2,1,2,2,2,0,0,0,1,0,0,0,2,1,2,1,2,2,1,1]
    curr_52 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]  
    
    currs = [curr_1,curr_2,curr_3,curr_4,curr_5,curr_6,curr_7,curr_8,curr_9,curr_10,curr_11,curr_12,curr_13,
             curr_14,curr_15,curr_16,curr_17,curr_18,curr_19,curr_20,curr_21,curr_22,curr_23,curr_24,curr_25,curr_26,curr_27,curr_28,curr_29,curr_30,curr_31,
             curr_32,curr_33,curr_34,curr_35,curr_36,curr_37,curr_38,curr_39,curr_40,curr_41,curr_42,curr_43,curr_44,curr_45,curr_46,curr_47,curr_48,
             curr_49,curr_50,curr_51,curr_52]
    
    for curr in currs:
        out.append([abs(np.mean(curr)-np.mean(hist)),euclidian(hist,curr),manhattan(hist,curr),kl(hist,curr),jsdiv(hist,curr),
                    jeffery(hist,curr),corr(hist,curr),hellinger(hist,curr),contigency(hist,curr),wasserstein(hist, curr)])
    
    out_df = pd.DataFrame(out,index = range(0,len(out)))
    out_df.columns = ['mean_diff','Euclidian','Manhattan','kullback','Jensen','Jeffery','Correlation','Hellinger','contingency','wasserstein']
    out_df.to_csv('/Users/ernestmugambi/Documents/results/distance_patterns.csv')
    currs_df = pd.DataFrame(currs,index = range(0,len(currs)))
    currs_df.columns = ['H-0','H-1','H-2','H-3','H-4','H-5','H-6','H-7','H-8','H-9','H-10','H-11','H-12','H-13','H-14','H-15','H-16','H-17',
                'H-18','H-19','H-20','H-21','H-22','H-23']
    res = pd.concat([currs_df,out_df],axis=1,join='inner')
    res.to_csv('/Users/ernestmugambi/Documents/results/compare_measures.csv')        # file path to results table
    print("Done....")
    #return 0
    
    plt.close('all')
    f, ax = plt.subplots(2, sharex = True)
    ax[0].bar(range(24),currs_df.loc[0],color = 'red')
    ax[0].set_title('Original Profile')
    ax[1].bar(range(24),currs_df.loc[23],color = 'blue')
    ax[1].set_title('New Profile')
    plt.show()
    
    #plt.plot(range(0,len(out_df)),out_df['Sibson'],'b-d')
    plt.plot(range(0,len(out_df)),out_df['contingency'],'r-*',label='Contingency')
    plt.plot(range(0,len(out_df)),out_df['Correlation'],'k-*',label='Correlation')
    plt.plot(range(0,len(out_df)),out_df['Jensen'],'m-*',label='Jensen')
    plt.plot(range(0,len(out_df)),out_df['Jeffery'],'c-*',label='Jeffery')
    plt.plot(range(0,len(out_df)),out_df['Hellinger'],'g-*',label='Hellinger')
    plt.legend(loc="upper left")
    plt.savefig('/Users/ernestmugambi/Documents/results/compare_measures4.png')       # file path to plots
    plt.close('all')
    
    
        







    
    