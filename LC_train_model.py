# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:58:57 2016

@author: james
"""

import os
import sys
import re

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from collections import defaultdict, namedtuple
import datetime

#base_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = '/Users/james/Data_Incubator/loan-picker'
    
sys.path.append(base_dir)
import LC_helpers as LCH
import LC_loading as LCL
import LC_models as LCM

import dill

#set paths
data_dir = os.path.join(base_dir,'static/data/')

#%%
#load data 
data_name = 'all_loans_proc'
LD = pd.read_csv(data_dir + data_name, parse_dates=['issue_d',])

#load long/lat data for each zip-code
zip3_data = LCL.load_location_data(data_dir,group_by='zip3')        
LD = pd.merge(LD,zip3_data,how='inner', left_on='zip3', right_index=True)

#%%
predictor = namedtuple('predictor', ['col_name', 'full_name', 'norm_type'])

transformer_map = {'minMax':MinMaxScaler(),
                   'maxAbs':MaxAbsScaler(),
                   'standScal':StandardScaler(),
                   'log_minmax': LCM.log_minmax(),
                   'robScal':RobustScaler()
                   }

predictors = [
            predictor('acc_now_delinq','num delinq accounts','maxAbs'),
            predictor('annual_inc','annual income','log_minmax'),
            predictor('collections_12_mths_ex_med','num recent collections','maxAbs'),
            predictor('cr_line_dur', 'duration cred line','standScal'),
            predictor('delinq_2yrs', 'num recent delinq','maxAbs'),
            predictor('desc_length', 'loan desc length','maxAbs'),
            predictor('dti', 'debt-income ratio','standScal'),
            predictor('emp_length', 'employment length','maxAbs'),
            predictor('funded_amnt','loan amount','maxAbs'),
            predictor('inq_last_6mths', 'num recent inqs','maxAbs'),
            predictor('int_rate', 'interest rate','maxAbs'),
#            predictor('issue_day', 'issue date','maxAbs'),
            predictor('mths_since_last_delinq','mths since delinq','maxAbs'),
            predictor('mths_since_last_major_derog','mths since derog','maxAbs'),
            predictor('mths_since_last_record','mths since rec','maxAbs'),
            predictor('num_add_desc','num descripts added','maxAbs'),
            predictor('open_acc','num open accounts','robScal'),
            predictor('pub_rec', 'num pub rec','robScal'),
            predictor('revol_bal','revolv cred bal','robScal'),
            predictor('revol_util','revolv cred util','robScal'),
            predictor('term','loan duration','maxAbs'),
            predictor('total_acc','num accounts','robScal'),
            predictor('tot_cur_bal','total balance','log_minmax'),
            predictor('addr_state', 'borrower state','cat'),
            predictor('home_ownership', 'home ownership','cat'),
            predictor('grade','loan grade','cat'),
            predictor('purpose','loan purpose','cat'),
            predictor('latitude','latitude','minMax'),
            predictor('longitude','longitude','minMax')
            ]

#%%
response_var = 'ROI'
y = LD[response_var].values  # response variable
Npymnts = LD['exp_num_pymnts'].values # expected number of payments
weight_y = (LD['mnthly_ROI'] * LD['exp_num_pymnts']).values #duration-weighted returns
    
LD.fillna(0, inplace=True)

print('Transforming data')
use_cols = [pred.col_name for pred in predictors]
col_titles = {pred.col_name:pred.full_name for pred in predictors}
recs = LD[use_cols].to_dict(orient='records') #convert to list of dicts
dict_vect = DictVectorizer(sparse=False)
X = dict_vect.fit_transform(recs)                  
print('Done')
dict_vect.predictors = predictors

#%%
feature_names = dict_vect.get_feature_names()
col_dict = defaultdict(list)
for idx, feature_name in enumerate(feature_names):
    short_name = re.findall('[^=]*',feature_name)[0] #get the part before the equals sign, if there is onee
    col_dict[short_name].append(idx)
    pidx = use_cols.index(short_name)
    if predictors[pidx].norm_type in transformer_map:
        tran = transformer_map[predictors[pidx].norm_type]
        X[:,idx] = tran.fit_transform(X[:,idx].reshape(-1,1)).squeeze()

dict_vect.col_dict = col_dict  
      
#%% COMPILE LIST OF MODELS TO COMPARE

max_depth=14 #16
min_samples_leaf=50
min_samples_split=100
n_trees=100 #100
RF_est = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, 
                               min_samples_leaf=min_samples_leaf, 
                               min_samples_split=min_samples_split,n_jobs=4)

#%%   
RF_est.fit(X,y)

#%%
with open(os.path.join(base_dir,'static/data/LC_model.pkl'),'wb') as out_strm:
    dill.dump(RF_est, out_strm)
    
with open(os.path.join(base_dir,'static/data/dict_vect.pkl'),'wb') as out_strm:
    dill.dump(dict_vect, out_strm)