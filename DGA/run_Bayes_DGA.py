#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:48:02 2024
@author: ernestmugambi
"""
import pandas as pd
import matplotlib.pyplot as plt

alexa = '/users/ernestmugambi/Documents/UEBA_Algos/domain_scoring/data_files/top-1m.csv'      #-- get good domains e.g Alexa 1 Million Urls
bad = '/users/ernestmugambi/Documents/UEBA_Algos/domain_scoring/data_files/bad_url_3.csv'     #-- bad urls obtained from bluecoat website
phish_tank = '/users/ernestmugambi/Documents/UEBA_Algos/domain_scoring/data_files/phish_tank_2.csv'  # confirmed phishing urls from phishtank

"""
=================================================
data sources
=================================================
"""
def get_good_data():
    alexa_df = pd.read_csv(alexa, header=None)
    alexa_df.columns = ['row_num','url']
    return alexa_df

def get_test_data():
    bad_df = pd.read_csv(bad, header=0)
    bad_df = bad_df[['new_url']]
    bad_df.columns = ['url']
    return bad_df

#-- extract features (good features)
trn_data = get_good_data()    
good_features = domain_feature_generator.generate_features(trn_data)
#-- get test domains of interest (test features)
tst_data = get_test_data()    
test_features = domain_feature_generator.generate_features(tst_data)
#-- running the good and test features thru the Bayesian detection algorithm
feature_scoring = bayesian_detection_engine.main_inference(good_features, test_features)
output = pd.concat([tst_data, feature_scoring], axis = 1)

"""
=================================================
plot results
=================================================
"""
# plotting results
fig, ax = plt.subplots()
sub_output = output[['url','t_llh']]  # extract only needed cols
random_rows = sub_output.sample(n = 30) # extract 30 random samples
top = sub_output[sub_output['t_llh']==max(sub_output['t_llh'])] # top
bottom = sub_output[sub_output['t_llh']==min(sub_output['t_llh'])] # bottom
samples = pd.concat([bottom,random_rows,top])
samples = samples.sort_values('t_llh')
y_pos = range(len(samples))
ax.barh(y_pos, samples['t_llh'])
ax.set_yticks(y_pos, labels = samples['url'], fontsize=6) 
ax.invert_yaxis()
ax.set_xlabel('probability of url being "good" ')
plt.show()