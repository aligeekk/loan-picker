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

#base_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = '/Users/james/Data_Incubator/loan-picker'
sys.path.append(base_dir)

data_dir = os.path.join(base_dir,'static/data/')

import LC_helpers as LCH
import LC_loading as LCL
import LC_modeling as LCM

predictor = namedtuple('predictor', ['col_name', 'full_name', 'norm_type'])

#%% Load in model and transformer
with open(os.path.join(base_dir,'static/data/LC_model.pkl'),'rb') as in_strm:
    RF_est = dill.load(in_strm)

with open(os.path.join(base_dir,'static/data/LC_def_model.pkl'),'rb') as in_strm:
    RF_defClass = dill.load(in_strm)
    
with open(os.path.join(base_dir,'static/data/trans_tuple.pkl'),'rb') as in_strm:
    dict_vect, col_dict, tran_dict, predictors = dill.load(in_strm)

#get lat/long coordinates for each 3-digit zip
zip3_data = LCL.load_location_data(data_dir,group_by='zip3')        
 

#%% pull in latest loan data
header = {'Authorization' : 'CL8mtxpJKxUjSpgjunpqV0nE1Xo=', 
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
            ('latitude', lambda x: LCL.get_zip_loc(x['addrZip'], zip3_data, 'latitude')),
            ('longitude', lambda x: LCL.get_zip_loc(x['addrZip'], zip3_data, 'longitude')))

def make_record(loan, record_map):
    record = {}
    for rec_name, rec_fun in record_map:
        record[rec_name] = rec_fun(loan)
    return record 
records = [make_record(loan, record_map) for loan in loans]

use_grades = ['A','B','C','D','E','F']
records = [record for record in records if record['grade'] in use_grades]
grades = [rec['grade'] for rec in records]
loan_ids = [rec['id'] for rec in records]
#%% Build X mat for new loan data
X_rec = dict_vect.transform(records) #use the preloaded dict vectorizer

#apply any additional transfomers to ordinal data as needed.
use_cols = [pred.col_name for pred in predictors]
feature_names = dict_vect.get_feature_names()
for idx, feature_name in enumerate(feature_names):
    short_name = re.findall('[^=]*',feature_name)[0] #get the part before the equals sign, if there is onee
    pidx = use_cols.index(short_name)
    if short_name in tran_dict:
        tran = tran_dict[short_name]
        X_rec[:,idx] = tran.transform(X_rec[:,idx].reshape(-1,1)).squeeze()

#%% Generate model-predicted returns
pred_returns = RF_est.predict(X_rec)
pred_dp = RF_defClass.predict_proba(X_rec)

predictions = pd.DataFrame({'returns': pred_returns*100, 'dp':pred_dp[:,1],
                            'grades': grades, 'ids':loan_ids})
predictions.sort_values(by='returns', ascending=False, inplace=True)

#%%
pick_K = 100

grade_order = sorted(predictions.grades.unique())
pal = sns.cubehelix_palette(n_colors=len(grade_order))
sns.lmplot(x='dp',y='returns',data=predictions,hue='grades',
           hue_order = grade_order, fit_reg=False, 
           palette=pal, legend=False)
plt.plot(predictions.iloc[:pick_K]['dp'],predictions.iloc[:pick_K]['returns'],'r.',
        ms=10, label='picked')
plt.legend(loc='lower left')
plt.xlabel('Default probability',fontsize=14)
plt.ylabel('Predicted annual returns (%)',fontsize=14)
plt.margins(.01, .01)
#%% Load and unpack validation lookup-table data
with open(os.path.join(base_dir,'static/data/LC_test_res.pkl'),'rb') as in_strm:
    val_set = dill.load(in_strm)
test_data = [np.array(tup) for tup in zip(*val_set)]
test_pred, test_ROI, test_net_returns, test_weights = test_data

#%%
n_bins = 100
n_boots = 500 #5000
prctiles = np.linspace(1,99,25)
K = 100

bin_locs = np.linspace(0, 100, n_bins)
bins = np.percentile(test_pred, bin_locs)

sim_pred_inds = np.digitize(test_pred,bins)
boot_inds = np.digitize(pred_returns, bins) #find bins of first max_K points in prediction

est_return = np.zeros((n_bins,len(prctiles)))
boot_vec = np.zeros(n_boots)
for bin in xrange(1,n_bins):
    cur_resp_set = np.nonzero(sim_pred_inds == bin)[0]
    if len(cur_resp_set) > 0:
        for boot in xrange(n_boots):    
            boot_samp = np.random.choice(cur_resp_set, size=K, replace=True)
            boot_vec[boot] = np.sum(test_net_returns[boot_samp])/np.sum(test_weights[boot_samp])
#            boot_vec[boot] = np.mean(test_ROI[boot_samp])
        est_return[bin,:] = np.percentile(boot_vec, prctiles)
        
est_return = LCM.annualize_returns(est_return)
#est_return = est_return * 100.

bins = 100*bins

#%%
def smooth(x,beta):
     """ kaiser window smoothing """
     window_len=11
     # extending the data at beginning and at the end
     # to apply the window at the borders
     s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
     w = np.kaiser(window_len,beta)
     y = np.convolve(w/w.sum(),s,mode='valid')
     return y[5:len(y)-5]

beta = 20
sm_returns = est_return.copy()
for p in xrange(len(prctiles)):
    sm_returns[:,p] = smooth(est_return[:,p], beta)    
    
#%%
#plt.close('all')
alphas = np.linspace(0,1.0,len(prctiles)//2)
fig,ax = plt.subplots()
for idx, grade in enumerate(grade_order):
    plt.plot(np.nan, np.nan, color=pal[idx], label=grade)
plt.legend(loc='best')

for q in xrange(len(prctiles)//2):
    cur_low = sm_returns[:,q]
    cur_high = sm_returns[:,q+1]
    ax.fill_between(bins, cur_low,cur_high,
                     alpha=alphas[q], facecolor='k')
  
    cur_low = sm_returns[:,-q]
    cur_high = sm_returns[:,-(q+1)]
    ax.fill_between(bins, cur_low,cur_high,
                     alpha=alphas[q], facecolor='k')
med_idx = np.argwhere(prctiles == 50)[0]
ax.plot(bins,sm_returns[:,med_idx],'w')
#plt.plot(poss_choose_K, q_bands.T,'r',lw=1)
ax.set_ylim(0,20)
#ax.set_xlim(-10,15)

ax2 = ax.twinx()
#sns.distplot(pred_returns*100,rug=True, hist=False, kde=False, ax=ax2,
#             rug_kws={'color':'k','lw':1.0,'alpha':0.5})
for idx, grade in enumerate(grade_order):
    sns.distplot(predictions[predictions['grades'] == grade].returns, 
                 rug=True,hist=False,kde=True,ax=ax2, 
                 rug_kws={'color':pal[idx],'lw':1.0,'alpha':0.5},
                 kde_kws={'color':pal[idx]})
#sns.distplot(test_pred*100,rug=False, hist=False,kde=True, ax=ax2,
#             kde_kws={'color':'k'})
yl2 = ax2.get_ylim()
ax2.set_ylim(0,yl2[1]*1.5)
ax2.set_yticks([])
ax2.set_xlim(-8,15)

#%%
#from matplotlib.ticker import ScalarFormatter
#plt.close('all')
##ax = sns.tsplot(data=est_return, time=poss_choose_K, err_style='unit_traces', 
##                color="b")
##plt.figure(figsize=(6.0,4.0))
#fig,ax = plt.subplots(figsize=(5.0,4.0))
#plt.plot(poss_choose_K,est_return.T,'b',lw=1.0,alpha=0.05)
#plt.plot(np.nan, np.nan,'r',lw=8,label='90% CI',alpha=0.35)
#plt.plot(np.nan, np.nan,'r',lw=4,label='50% CI',alpha=0.5)
#plt.plot(np.nan, np.nan,'y',lw=3,label='median',alpha=1.0)
#plt.legend(loc='best',fontsize=14)
#
#quantiles = [5, 95]
#q_bands = np.percentile(est_return,quantiles,axis=0)
#plt.fill_between(poss_choose_K, q_bands[0,:],q_bands[1,:],
#    alpha=0.35, facecolor='r', zorder=10000)
##plt.plot(poss_choose_K, q_bands.T,'r',lw=1)
##
#quantiles = [25, 75]
#q_bands = np.percentile(est_return,quantiles,axis=0)
#plt.fill_between(poss_choose_K, q_bands[0,:],q_bands[1,:],
#    alpha=0.5, facecolor='r', zorder=10001)
##plt.plot(poss_choose_K, q_bands.T,'r',lw=2)
#
#quantiles = [50]
#q_bands = np.percentile(est_return,quantiles,axis=0)
#plt.plot(poss_choose_K, q_bands.T,'y',lw=2,zorder=10002)
#
#plt.xscale('log')
#plt.xlim(min(poss_choose_K),max(poss_choose_K))
#plt.xlabel('Number of loans selected',fontsize=14)
#plt.ylabel('Estimated annual returns (%)',fontsize=14)
#plt.ylim(-15,25)pl
#plt.axhline(y=0,color='k',ls='dashed')
#ax.set_xticks(np.concatenate((np.arange(10,100,10),np.arange(100,1100,100)),axis=0))
#ax.get_xaxis().set_major_formatter(ScalarFormatter())
#ax.set_xticklabels(["10",'','','','','','','','',"100",
#                    '','','','','','','','',"1000"],fontsize=12)
#plt.tight_layout()
##plt.savefig(fig_dir + 'newloan_predict.png', dpi=500, format='png')
