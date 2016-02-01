# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:17:38 2016

@author: james
"""


import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.cross_validation import ShuffleSplit, KFold
from sklearn.pipeline import Pipeline

from collections import defaultdict, namedtuple
import datetime

#base_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = '/Users/james/Data_Incubator/LC_app'
    
sys.path.append(base_dir)
import LC_helpers as LCH
import LC_loading as LCL
import LC_models as LCM

#set paths
data_dir = os.path.join(base_dir,'static/data/')
fig_dir = os.path.join(base_dir,'static/images/')
movie_dir = os.path.join(base_dir,'static/movies/')

#%%
#load data 
data_name = 'all_loans_proc'
LD = pd.read_csv(data_dir + data_name, parse_dates=['issue_d',])

#load long/lat data for each zip-code
zip3_data = LCL.load_location_data(data_dir,group_by='zip3')        
LD = pd.merge(LD,zip3_data,how='inner', left_on='zip3', right_index=True)

#%%
predictor = namedtuple('predictor', ['col_name', 'full_name', 'norm_type'])

ordinal_preds = [
            predictor('acc_now_delinq','num delinq accounts','maxAbs'),
            predictor('annual_inc','annual income','maxAbs'),
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
            predictor('open_acc','num open accounts','maxAbs'),
            predictor('pub_rec', 'num pub rec','maxAbs'),
            predictor('revol_bal','revolv cred bal','maxAbs'),
            predictor('revol_util','revolv cred util','maxAbs'),
            predictor('term','loan duration','maxAbs'),
            predictor('total_acc','num accounts','maxAbs'),
            predictor('tot_cur_bal','total balance','maxAbs'),
                ]

categorical_preds = [
            predictor('addr_state', 'borrower state','cat'),
            predictor('home_ownership', 'home ownership','cat'),
            predictor('grade','loan grade','cat'),
            predictor('purpose','loan purpose','cat'),
            predictor('verification_status','verification status','cat'),
                    ]                     

grouped_preds = [
                ('borrower location',
                [predictor('latitude','latitude','minMax'),
                predictor('longitude','longitude','minMax')])
                ]

transformer_map = {'minMax':MinMaxScaler(),
                   'maxAbs':MaxAbsScaler(),
                   'standScal':StandardScaler()}

                   
#%%
response_var = 'ROI'
y = LD[response_var].values  # response variable
Npymnts = LD['exp_num_pymnts'].values # expected number of payments
weight_y = (LD['mnthly_ROI'] * LD['exp_num_pymnts']).values #duration-weighted returns
    
LD.fillna(0, inplace=True)

col_dict = {}
col_types = {}
col_titles = {}
X = np.zeros((len(y), len(ordinal_preds)))            
for idx, pred in enumerate(ordinal_preds):
    data = LD[pred.col_name].values.reshape(-1,1)
    cur_pred = transformer_map[pred.norm_type].fit_transform(data)
    X[:,idx] = cur_pred.squeeze() 
    col_dict[pred.col_name] = [idx]
    col_types[pred.col_name] = 'ord'
    col_titles[pred.col_name] = pred.full_name
    

#add in grouped predictor variables
for group_name, pred_list in grouped_preds:
    col_dict[group_name] = np.arange(len(pred_list)) + X.shape[1]
    col_types[group_name] = 'ord'
    col_titles[group_name] = group_name
    for pred in pred_list:
        data = LD[pred.col_name].values.reshape(-1,1)
        cur_pred = transformer_map[pred.norm_type].fit_transform(data)       
        X = np.concatenate((X, cur_pred), axis=1)

#add in categorical variables
for cat_pred in categorical_preds: # add categorical predictors
    dummyX = pd.get_dummies(LD[cat_pred.col_name])
    col_dict[cat_pred.col_name] = np.arange(dummyX.shape[1]) + X.shape[1]
    col_types[cat_pred.col_name] = 'cat'
    col_titles[cat_pred.col_name] = cat_pred.full_name
    X = np.concatenate((X, dummyX.values), axis=1)

all_preds = categorical_preds + ordinal_preds
all_pred_cnames = [pred.col_name for pred in all_preds]
for gname, gpred in grouped_preds:
    all_pred_cnames.append(gname)


#%% COMPILE LIST OF MODELS TO COMPARE
max_depth=14 #16
min_samples_leaf=50
min_samples_split=100
n_trees=100 #100
#%%
act_test = (LD.issue_d == LD.issue_d.max()).values
act_train = (LD.issue_d < LD.issue_d.max()).values
RF_est = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, 
                               min_samples_leaf=min_samples_leaf, 
                               min_samples_split=min_samples_split,n_jobs=4,
                               max_features='auto')

RF_est.fit(X[act_train],y[act_train])
pred_ret = RF_est.predict(X[act_test])
pred_ret = sorted(pred_ret, reverse = True)

#%%
n_iter = 1
test_size = 0.5
SS = ShuffleSplit(len(LD), n_iter = n_iter, test_size = test_size, 
                  random_state=0)
train, test = list(SS)[0]
RF_sim = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, 
                               min_samples_leaf=min_samples_leaf, 
                               min_samples_split=min_samples_split,n_jobs=4,
                               max_features='auto')

RF_sim.fit(X[train],y[train])
sim_pred = RF_est.predict(X[test])
#%%
n_bins = 1000
bins = np.percentile(sim_pred, np.linspace(0,100,n_bins))
sim_pred_inds = np.digitize(sim_pred,bins)

n_boots = 5000 #5000
poss_choose_K = np.logspace(np.log10(5),np.log10(1000),50).astype(int)
poss_choose_K = np.unique(poss_choose_K)
max_K = max(poss_choose_K)
boot_inds = np.digitize(pred_ret[:max_K], bins)

est_return = np.zeros((n_boots,len(poss_choose_K)))

for b_idx in xrange(n_boots):
    sampled_y = np.zeros(max_K)
    sampled_N = np.zeros(max_K)
    for ii in xrange(max_K):
        cur_resp_set = np.nonzero(sim_pred_inds == boot_inds[ii])[0]
        rand_choice = cur_resp_set[np.random.randint(0,len(cur_resp_set))]
        sampled_y[ii] = weight_y[test[rand_choice]]
        sampled_N[ii] = Npymnts[test[rand_choice]]
    
    for k_idx, choose_K in enumerate(poss_choose_K):
        est_return[b_idx,k_idx] = np.mean(sampled_y[:choose_K])/np.mean(sampled_N[:choose_K])
    
est_return = LCM.annualize_returns(est_return)
   
#%%
from matplotlib.ticker import ScalarFormatter
plt.close('all')
#ax = sns.tsplot(data=est_return, time=poss_choose_K, err_style='unit_traces', 
#                color="b")
#plt.figure(figsize=(6.0,4.0))
fig,ax = plt.subplots(figsize=(5.0,4.0))
plt.plot(poss_choose_K,est_return.T,'b',lw=1.0,alpha=0.05)
plt.plot(np.nan, np.nan,'r',lw=8,label='90% CI',alpha=0.35)
plt.plot(np.nan, np.nan,'r',lw=4,label='50% CI',alpha=0.5)
plt.plot(np.nan, np.nan,'y',lw=3,label='median',alpha=1.0)
plt.legend(loc='best',fontsize=14)

quantiles = [5, 95]
q_bands = np.percentile(est_return,quantiles,axis=0)
plt.fill_between(poss_choose_K, q_bands[0,:],q_bands[1,:],
    alpha=0.35, facecolor='r', zorder=10000)
#plt.plot(poss_choose_K, q_bands.T,'r',lw=1)
#
quantiles = [25, 75]
q_bands = np.percentile(est_return,quantiles,axis=0)
plt.fill_between(poss_choose_K, q_bands[0,:],q_bands[1,:],
    alpha=0.5, facecolor='r', zorder=10001)
#plt.plot(poss_choose_K, q_bands.T,'r',lw=2)

quantiles = [50]
q_bands = np.percentile(est_return,quantiles,axis=0)
plt.plot(poss_choose_K, q_bands.T,'y',lw=2,zorder=10002)

plt.xscale('log')
plt.xlim(min(poss_choose_K),max(poss_choose_K))
plt.xlabel('Number of loans selected',fontsize=14)
plt.ylabel('Estimated ROI (%)',fontsize=14)
plt.ylim(-15,25)
plt.axhline(y=0,color='k',ls='dashed')
ax.set_xticks(np.concatenate((np.arange(10,100,10),np.arange(100,1100,100)),axis=0))
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.set_xticklabels(["10",'','','','','','','','',"100",
                    '','','','','','','','',"1000"],fontsize=12)
plt.tight_layout()
plt.savefig(fig_dir + 'newloan_predict.png', dpi=500, format='png')
