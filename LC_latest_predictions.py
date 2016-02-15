# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 15:13:14 2016

@author: James McFarland
"""
import sys
import os
import re
import numpy as np
import pandas as pd
import dill
import requests
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict, namedtuple

base_dir = os.path.dirname(os.path.realpath(__file__))
# base_dir = '/Users/james/Data_Incubator/loan-picker'
sys.path.append(base_dir)

data_dir = os.path.join(base_dir,'static/data/')

import LC_helpers as LCH
import LC_loading as LCL
import LC_modeling as LCM


#%% Load in model and transformer
def load_pickled_models():
    model_data = {}
    with open(os.path.join(base_dir,'static/data/LC_model.pkl'),'rb') as in_strm:
        model_data['RF_est'] = dill.load(in_strm)
    
    with open(os.path.join(base_dir,'static/data/LC_def_model.pkl'),'rb') as in_strm:
        model_data['RF_defClass'] = dill.load(in_strm)
        
    with open(os.path.join(base_dir,'static/data/trans_tuple.pkl'),'rb') as in_strm:
        variables = dill.load(in_strm)
        var_names = ['dict_vect','col_dict','tran_dict','predictors']
        for idx in xrange(len(var_names)):
            model_data[var_names[idx]] = variables[idx]
        
    return model_data
 

#%% pull in latest loan data
def make_record(loan, record_map):
    record = {}
    for rec_name, rec_fun in record_map:
        record[rec_name] = rec_fun(loan)
    return record 


def get_latest_records(auth_key, zip3_loc_data, use_grades = ['A','B','C','D','E','F']):
    header = {'Authorization' : auth_key, 
              'Content-Type': 'application/json'}
    apiVersion = 'v1'
    loanListURL = 'https://api.lendingclub.com/api/investor/' + apiVersion + \
            '/loans/listing'
    payload = {'showAll' : 'true'}
    resp = requests.get(loanListURL, headers=header, params=payload)
    loans = resp.json()['loans']
        
    '''Make a list of tuples specifying the name of each data column to pull, and 
    a function to use to grab that piece of data from the raw loan data'''
    record_map = (('acc_now_delinq', lambda x: x['accNowDelinq']),
                ('annual_inc', lambda x: x['annualInc']),
                ('collections_12_mths_ex_med', lambda x: x['collections12MthsExMed']),
                ('cr_line_dur', lambda x: LCL.get_cr_line_dur(x['earliestCrLine'])),
                ('delinq_2yrs', lambda x: x['delinq2Yrs']),
                ('desc_length', lambda x: LCL.get_desc_length(x['desc'])),
                ('dti', lambda x: x['dti']),
                ('emp_length', lambda x: LCL.get_emp_length(x['empLength'])), #convert to years from months
                ('id', lambda x: x['id']),
    #            ('funded_amnt', lambda x: x['loanAmount']), #use amount requested rather than funded amnt!
                ('loan_amnt', lambda x: x['loanAmount']), #use amount requested rather than funded amnt!
                ('inq_last_6mths', lambda x: x['inqLast6Mths']), 
                ('int_rate', lambda x: x['intRate']), 
                ('mths_since_last_delinq', lambda x: LCL.get_mnths_since(x['mthsSinceLastDelinq'])), 
                ('mths_since_last_major_derog', lambda x: LCL.get_mnths_since(x['mthsSinceLastMajorDerog'])), 
                ('mths_since_last_record', lambda x: LCL.get_mnths_since(x['mthsSinceLastRecord'])), 
                ('num_add_desc', lambda x: LCL.get_num_descs(x['desc'])),
                ('open_acc', lambda x: x['openAcc']),
                ('pub_rec', lambda x: x['pubRec']),
                ('revol_bal', lambda x: x['revolBal']),
                ('revol_util', lambda x: x['revolUtil']),
                ('term', lambda x: x['term']),
                ('total_acc', lambda x: x['totalAcc']),
                ('tot_cur_bal', lambda x: x['totCurBal']),
                ('addr_state', lambda x: x['addrState']),
                ('home_ownership', lambda x: x['homeOwnership']),
                ('grade', lambda x: x['grade']),
                ('purpose', lambda x: x['purpose']),
                ('latitude', lambda x: LCL.get_zip_loc(x['addrZip'], zip3_loc_data, 'latitude')),
                ('longitude', lambda x: LCL.get_zip_loc(x['addrZip'], zip3_loc_data, 'longitude')))
    
    records = [make_record(loan, record_map) for loan in loans]
    records = [record for record in records if record['grade'] in use_grades]
    return records
    
#%% Build X mat for new loan data
def transform_records(records, model_data):
    X_rec = model_data['dict_vect'].transform(records) #use the preloaded dict vectorizer
    
    #apply any additional transfomers to ordinal data as needed.
    feature_names = model_data['dict_vect'].get_feature_names()
  #    use_cols = [pred.col_name for pred in model_data['predictors']]
    for idx, feature_name in enumerate(feature_names):
        short_name = re.findall('[^=]*',feature_name)[0] #get the part before the equals sign, if there is onee
        if short_name in model_data['tran_dict']:
            tran = model_data['tran_dict'][short_name]
            X_rec[:,idx] = tran.transform(X_rec[:,idx].reshape(-1,1)).squeeze()
            
    return X_rec
    
#%% Generate model-predicted returns
def get_model_predictions(X, model_data, loan_grades, loan_ids):
    pred_returns = model_data['RF_est'].predict(X)
    pred_dp = model_data['RF_defClass'].predict_proba(X)
    
    predictions = pd.DataFrame({'returns': pred_returns*100, 'dp':pred_dp[:,1],
                                'grades': loan_grades, 'ids':loan_ids})
    predictions.sort_values(by='returns', ascending=False, inplace=True)
    return predictions

#%%
def make_dp_ret_figure(predictions, pick_K, allowed_loans):
            
    grade_order = sorted(predictions.grades.unique())
    pal = sns.cubehelix_palette(n_colors=len(grade_order))
    sns.lmplot(x='dp',y='returns',data=predictions,hue='grades',
               hue_order = grade_order, fit_reg=False, 
               palette=pal, legend=False, size=4.0, aspect=1.2)
        
    pick_K = min([pick_K, len(allowed_loans)])
    if pick_K != 0:
        plt.plot(allowed_loans.iloc[:pick_K]['dp'],allowed_loans.iloc[:pick_K]['returns'],'r.',
                ms=10, label='picked')
        plt.title('Selected Loans', fontsize=18)
    plt.legend(loc='lower left', fontsize=12)
    plt.xlabel('Predicted default probability',fontsize=14)
    plt.ylabel('Predicted annual returns (%)',fontsize=14)
    plt.margins(.01, .01)   
    return plt.gcf()
    
#%% Load and unpack validation lookup-table data
def get_validation_data():
    with open(os.path.join(base_dir,'static/data/LC_test_res.pkl'),'rb') as in_strm:
        val_set = dill.load(in_strm)
    test_data = [np.array(tup) for tup in zip(*val_set)]
    test_pred, test_ROI, test_net_returns, test_weights = test_data
    
    sim_lookup = pd.DataFrame({'pred':test_pred, 'ROI':test_ROI, 
                               'net_ret': test_net_returns, 'weights': test_weights})
    sim_lookup.sort_values(by='pred', ascending=False, inplace=True)
    return sim_lookup
  
#%%
def make_return_dist_fig(sim_lookup, predictions, pick_K=100, n_bins=200, n_boots=5000):

    sim_net = sim_lookup['net_ret'].values
    sim_weights = sim_lookup['weights'].values

    bin_locs = np.linspace(0, 100, n_bins)[::-1]
    bins = np.percentile(sim_lookup['pred'].values, bin_locs)
    
    sim_samps_per_bin = len(sim_lookup)/float(n_bins)
    pred_bins = np.digitize(predictions['returns'] / 100., bins) #find bins of first max_K points in prediction
    
    sim_returns = np.zeros(n_boots)
    boot_samps = sim_samps_per_bin*pred_bins[:pick_K] + np.random.randint(0, sim_samps_per_bin, size=(n_boots, pick_K))
    boot_samps = boot_samps.astype(int)
    sim_returns = np.sum(sim_net[boot_samps], axis=1) / np.sum(sim_weights[boot_samps], axis=1)                
    sim_returns = LCM.annualize_returns(sim_returns)
    
    fig,ax=plt.subplots(figsize=(5.0,4.0))
    sns.distplot(sim_returns,bins=100, hist=False, rug=False,
                 ax=ax, kde_kws={'color':'k','lw':3})
    plt.xlabel('Annual returns (%)',fontsize=14)
    plt.ylabel('Probability',fontsize=14)
    plt.title('Estimated portfolio returns', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.margins(.01, .01)   
    plt.tight_layout()
    return fig
    
    
def get_LC_loans(auth_key, model_data, zip3_loc_data, use_grades):
    records = get_latest_records(auth_key, zip3_loc_data, use_grades)       
    loan_grades = [rec['grade'] for rec in records]
    loan_ids = [rec['id'] for rec in records]
    
    X = transform_records(records, model_data)
    predictions = get_model_predictions(X, model_data, loan_grades, loan_ids)
 
    return predictions
