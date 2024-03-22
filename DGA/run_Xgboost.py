#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Mar 12 12:48:02 2024
@author: ernestmugambi
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

alexa = '/users/ernestmugambi/Documents/UEBA_Algos/domain_scoring/data_files/top-1m.csv'      #-- get good domains e.g Alexa 1 Million Urls
bad = '/users/ernestmugambi/Documents/UEBA_Algos/domain_scoring/data_files/bad_url_3.csv'     #-- bad urls obtained from bluecoat website
phish_tank = '/users/ernestmugambi/Documents/UEBA_Algos/domain_scoring/data_files/phish_tank_2.csv'   # confirmed phishing urls from phishtank

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

def get_test_data_2():
    bad_df = pd.read_csv(phish_tank, header=0)
    bad_df = bad_df[['url']]
    return bad_df

#-- extract features (good features)
trn_data = get_good_data()    
good_features = domain_feature_generator.generate_features(trn_data)
good_features['class'] = 0
good_features_sub = good_features.sample(n = 100000)
#-- get test domains of interest (test features)
tst_data = get_test_data()    
test_features = domain_feature_generator.generate_features(tst_data)
test_features['class'] = 1
#-- combined data set
trainDat = pd.concat([good_features_sub,test_features])
#-- get new unseen test data
tst_data_2 = get_test_data_2()    
test_features_2 = domain_feature_generator.generate_features(tst_data_2)
test_features_2['class'] = 1
good_features_sub2 = good_features.sample(n = 100000)
good_features_sub2['class'] = 0
testDat = pd.concat([test_features_2, good_features_sub2])
#-- running the good and bad features thru the XGBoost algorithm
y = trainDat[['class']]
X = trainDat.drop(['class'], axis = 1)
y2 = testDat[['class']]
X2 = testDat.drop(['class'], axis = 1)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
print("accuracy", clf.score(X2, y2))
prediction = clf.predict(X2)

"""
=================================================
compute classification performance
=================================================
"""
print("confusion matrix:",confusion_matrix(y2, prediction))
tn, fp, fn, tp = confusion_matrix(y2, prediction).ravel()
precision, recall = tp/(tp+fp), tp/(tp+fn)
print("precision:", precision)
print("recall:", recall)
print("F1:", (2*precision*recall)/(precision+recall))


