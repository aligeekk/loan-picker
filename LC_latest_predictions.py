# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 15:13:14 2016

@author: james
"""
import sys
import os
import datetime
import re

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict, namedtuple

#base_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = '/Users/james/Data_Incubator/loan-picker'
sys.path.append(base_dir)

data_dir = os.path.join(base_dir,'static/data/')

import LC_helpers as LCH
import LC_loading as LCL
import LC_models as LCM

import dill
import requests
predictor = namedtuple('predictor', ['col_name', 'full_name', 'norm_type'])

#%%
with open(os.path.join(base_dir,'static/data/LC_model.pkl'),'rb') as in_strm:
    RF_est = dill.load(in_strm)
    
with open(os.path.join(base_dir,'static/data/trans_tuple.pkl'),'rb') as in_strm:
    dict_vect, col_dict, tran_dict, predictors = dill.load(in_strm)

zip3_data = LCL.load_location_data(data_dir,group_by='zip3')        
 
#%%
header = {'Authorization' : 'CL8mtxpJKxUjSpgjunpqV0nE1Xo=', 
          'Content-Type': 'application/json'}
apiVersion = 'v1'
loanListURL = 'https://api.lendingclub.com/api/investor/' + apiVersion + \
        '/loans/listing'
payload = {'showAll' : 'true'}
resp = requests.get(loanListURL, headers=header, params=payload)
loans = resp.json()['loans']

#%%
def get_cr_line_dur(date_string):
    '''Get number of days since credit line was opened'''
    cur_date = datetime.date.today()
    date_string = re.findall(r'[^T]+',date_string)[0]
    cr_start = datetime.datetime.strptime(date_string, '%Y-%m-%d').date()    
    return (cur_date - cr_start).days
    
def get_desc_length(desc):
    '''Return length of description if any'''
    if desc:
        only_desc = desc.replace(r'Borrower added on [\S]{8} >','')  
        return len(only_desc)
    else: 
        return 0

def get_num_descs(desc):
    '''Return number of descriptions added'''
    if desc:
        return len(re.findall(r'Borrower added on',desc))
    else: 
        return 0
        
def get_mnths_since(mths_since):
    '''Handle features that are months since some event that might not have happened'''
    if mths_since:
        return mths_since
    else:
        return 200. #this is just an arbitrary value to fill in for Nans, better than 0.
        
def get_emp_length(emp_length):
    '''Handle possible missing employment length data'''
    if emp_length:
        return emp_length/12. # in years
    else:
        return 0
        
def get_zip_loc(addrZip,loc_type):
    '''Get lat or long for zip3'''
    zip3 = int(addrZip[:3])
    if zip3 in zip3_data:
        return zip3_data.ix[zip3][loc_type]
    else:
        nearest_zip = min(zip3_data.index.values, key=lambda x:abs(x-zip3))
        return zip3_data.ix[nearest_zip][loc_type]
        
#%% 
record_map = (('acc_now_delinq', lambda x: x['accNowDelinq']),
            ('annual_inc', lambda x: x['annualInc']),
            ('collections_12_mths_ex_med', lambda x: x['collections12MthsExMed']),
            ('cr_line_dur', lambda x: get_cr_line_dur(x['earliestCrLine'])),
            ('delinq_2yrs', lambda x: x['delinq2Yrs']),
            ('desc_length', lambda x: get_desc_length(x['desc'])),
            ('dti', lambda x: x['dti']),
            ('emp_length', lambda x: get_emp_length(x['empLength'])), #convert to years from months
            ('funded_amnt', lambda x: x['loanAmount']), #use amount requested rather than funded amnt!
            ('inq_last_6mths', lambda x: x['inqLast6Mths']), 
            ('int_rate', lambda x: x['intRate']), 
            ('mths_since_last_delinq', lambda x: get_mnths_since(x['mthsSinceLastDelinq'])), 
            ('mths_since_last_major_derog', lambda x: get_mnths_since(x['mthsSinceLastMajorDerog'])), 
            ('mths_since_last_record', lambda x: get_mnths_since(x['mthsSinceLastRecord'])), 
            ('num_add_desc', lambda x: get_num_descs(x['desc'])),
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
            ('latitude', lambda x: get_zip_loc(x['addrZip'], 'latitude')),
            ('longitude', lambda x: get_zip_loc(x['addrZip'], 'longitude')))

def make_record(loan, record_map):
    record = {}
    for rec_name, rec_fun in record_map:
        record[rec_name] = rec_fun(loan)
    return record
    
records = [make_record(loan, record_map) for loan in loans]
#%%
X_rec = dict_vect.transform(records)

use_cols = [pred.col_name for pred in predictors]
feature_names = dict_vect.get_feature_names()
for idx, feature_name in enumerate(feature_names):
    short_name = re.findall('[^=]*',feature_name)[0] #get the part before the equals sign, if there is onee
    pidx = use_cols.index(short_name)
    if short_name in tran_dict:
        tran = tran_dict[short_name]
        X_rec[:,idx] = tran.transform(X_rec[:,idx].reshape(-1,1)).squeeze()

#%%
pred_returns = RF_est.predict(X_rec)