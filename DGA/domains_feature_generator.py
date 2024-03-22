#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 08:31:59 2020
@author: ernestmugambi
"""
#import os
import pandas as pd
import numpy as np
import math

class domain_feature_generator:
    
    """
    =================================================
    string / character manipulation
    =================================================
    """
    def remove_http(f):
        f = f.dropna()
        new_url = []
        x = ''
        for i in range(len(f)):
            x = f['url'][i]
            cut = x.find('://') + 3
            new_url.append(x[cut:])
        f['new_url'] = new_url
        return f
    
    def remove_sub_domain(f):
        f = f.dropna()
        new_url = []
        x = ''
        for i in range(len(f)):
            x = f['url'][i]
            cut = x.find('/')
            new_url.append(x[:cut])
        f['new_url'] = new_url
        return f
    """
    =================================================
    feature functions
    =================================================
    """
    def entropy(string):
        "Calculates the Shannon entropy of a string"
        # get probability of chars in string
        prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]
        # calculate the entropy
        entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])
        return entropy
    
    # count no. of vowels
    def vowels(text):
        num_vowels=0
        for char in text:
            if char in "aeiouAEIOU":
               num_vowels = num_vowels+1
        return num_vowels
    
    #count no. of digits
    def numbers(text):
        num_numbers=0
        for char in text:
            if char in "1234567890":
               num_numbers = num_numbers+1
        return num_numbers
    
    # count no. of dots
    def dots(text):
        num_numbers=0
        for char in text:
            if char in ".":
               num_numbers = num_numbers+1
        return num_numbers
    
    # count no. of forward-slashes
    def slashes(text):
        num_numbers=0
        for char in text:
            if char in "/":
               num_numbers = num_numbers+1
        return num_numbers
    
    # count number of symbols found in string
    def count_symbols(s):
        t = 0
        symbols = ['!','@','#','$','%','^','&','*','(',')','_','-','+','=','{','}','[',']','\\','//','?','<','>','|','~','`','.',',','/']
        for i in symbols:
            if i in s:
                t += 1
        return t
    
    def split(word): 
        return [char for char in word]  
    
    # what is in a string ?
    def categorize_pattern(in_put):
        if in_put.isdigit():                        # numbers 0-9
            return 0
        if in_put.isalpha():                        # letters only
            return 1
        else:
            return 2                                # symbols and combination of symbols/letters/numbers
        
    # count pattern types
    def count_types(d):
        strings = d.split()
        str_types = []
        for i in strings:
            str_types.append(domain_feature_generator.categorize_pattern(i))
        return (str_types.count(0),str_types.count(1),str_types.count(2),str_types.count(3))
            
    # form a transition matrix from chars
    def count_transitions(d):
        char_list = domain_feature_generator.split(d)
        res = np.zeros((3, 3))
        for i in range(len(char_list)-1):
            j = i + 1
            res[domain_feature_generator.categorize_pattern(char_list[i]),domain_feature_generator.categorize_pattern(char_list[j])] = res[domain_feature_generator.categorize_pattern(char_list[i]),domain_feature_generator.categorize_pattern(char_list[j])] + 1
        res.resize((1,9))
        return res[0]
    
    # calls all the original feature functions
    def get_original_features(data_in):
        features = pd.DataFrame(columns=['shannon','vowels','numbers','dots'])
        s,v,n,d,l,sym,slsh = [],[],[],[],[],[],[]
        for i in range(len(data_in)):
            s.append(np.floor(domain_feature_generator.entropy(data_in['url'][i])))
            #print(i)
            v.append(domain_feature_generator.vowels(data_in['url'][i]))
            n.append(domain_feature_generator.numbers(data_in['url'][i]))
            d.append(domain_feature_generator.dots(data_in['url'][i]))
            l.append(len(data_in['url'][i]))
            sym.append(domain_feature_generator.count_symbols(data_in['url'][i]))
            slsh.append(domain_feature_generator.slashes(data_in['url'][i]))
            #trn.append(count_transitions(data_in['url'][i]))
        features['shannon'] = s
        features['vowels'] = v
        features['numbers'] = n
        features['dots'] = d
        features['length'] = l
        features['symbols'] = sym
        features['f_slash'] = slsh
        return features
    
    # calls the sequence based features
    def get_sequences(data_in):
        trn = []
        for i in range(len(data_in)):
            trn.append(domain_feature_generator.count_transitions(data_in['url'][i]))
        df_trn_matrix = pd.DataFrame(data=trn, columns = ['d_d','d_a','d_s','a_d','a_a','a_s','s_d','s_a','s_s'])
        return df_trn_matrix    
    """
    =================================================
    MAIN : combines all the features together and saves output to a file
    =================================================
    """
    def generate_features(data):
        #data = get_data_4()        # choose correct data file
        out_a = domain_feature_generator.get_original_features(data)
        out_b = domain_feature_generator.get_sequences(data)
        out = pd.concat([out_a,out_b],axis=1)
        #out.to_csv(feature_output)
        #print("Done....")
        return out

