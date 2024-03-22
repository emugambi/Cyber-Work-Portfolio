import pandas as pd
import numpy as np


class bayesian_detection_engine:
    
    """
    =================================================
    simple probabilistic Inference
    =================================================
    """
    def compute_var_trn_probs(dat,var):
        grp = dat.groupby([var]).size().reset_index()
        grp.columns = ['feature_trn','counts']
        grp['prb'] = grp['counts']/np.sum(grp['counts'])
        del grp['counts']
        return grp
    
    def compute_var_tst_probs(dat,var):
        grp = dat.groupby([var]).size().reset_index()
        grp.columns = ['feature_tst','counts']
        grp['prb'] = grp['counts']/np.sum(grp['counts'])
        del grp['counts']
        return grp
    
    def infer_probs(dat_trn, dat_tst, var_num):
        #which training set ?
        #dat_trn = get_trn_cases()
        trn_p = bayesian_detection_engine.compute_var_trn_probs(dat_trn,var_num)
        #dat_tst = get_tst_cases()      # abnormal test set
        tst_p = dat_tst[var_num]  # problem here
        tst_p = tst_p.reset_index()
        tst_p.columns = ['index','feature_tst']
        return trn_p,tst_p
    
    def combine_train_test(trn,tst):
        out = pd.merge(tst,trn,how='left',left_on='feature_tst',right_on='feature_trn')
        return out['prb']
    
    def run_inference(trn_dat, tst_dat):
        res = pd.DataFrame()
        features = ['shannon', 'vowels', 'numbers', 'dots', 'length',
           'symbols', 'f_slash', 'd_d', 'd_a', 'd_s', 'a_d', 'a_a', 'a_s', 's_d','s_a', 's_s']
        for feature in features:
            x,y = bayesian_detection_engine.infer_probs(trn_dat, tst_dat, feature)
            res[feature] = bayesian_detection_engine.combine_train_test(x,y)
        res = res.fillna(0.00000001)
        return res
    
    def main_inference(trn_dat, tst_dat):
        output = bayesian_detection_engine.run_inference(trn_dat, tst_dat)
        cols = ['shannon', 'vowels', 'numbers', 'dots', 'length','symbols', 'f_slash', 'd_d', 'd_a', 'd_s', 'a_d', 'a_a', 'a_s', 's_d','s_a', 's_s']
        weights = np.array([0.05,0.05,0.05,0.04,0.08,0.14,0.06,0.06,0.05,0.04,0.04,0.08,0.09,0.04,0.09,0.04])
        weights = weights/np.sum(weights)
        output = output[cols]
        out = output.dot(weights)
        output['llh'] = out
        output['t_llh'] = (output['llh']/np.max(output['llh']))*100.0
        #output_urls = pd.read_csv('/users/ernestmugambi/domain_modeling/omf_urls.csv',header=0)
        #output['url'] = omf_urls['url']
        #output.to_csv(predicted_set)
        print("Done....")
        return output























