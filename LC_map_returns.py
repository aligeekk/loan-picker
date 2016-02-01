# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:14:14 2016

@author: james
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import pandas as pd
from LC_helpers import *
from sklearn.cross_validation import KFold

#%%
data_dir = 'static/data/'
fig_dir = 'static/images/'

data_name = 'all_loans_proc'

LD = pd.read_csv(data_dir + data_name)
LD['issue_d'] = pd.to_datetime(LD['issue_d'],format = '%Y-%m-%d',unit='D')

#%%
def make_group_cumdist_fig(LD,group_by,pcol):
    
    name_legend_map = {'counts': 'Number of loans (thousands)',
				 'ROI': 'Average ROI (%)',
				  'int_rate': 'interest rate (%)',
				  'default_prob': 'default probability',
				  'dti': 'Debt-to-income ratio',
				  'emp_length': 'employment length (months)',
                        'annual_inc': 'annual income ($)'}

    min_group_loans = 100 #only use states with at least this many loans
    good_groups = LD.groupby(group_by).filter(lambda x: x[pcol].count() >= min_group_loans)
    n_groups = len(good_groups[group_by].unique())
    group_stats = good_groups.groupby(group_by)[pcol].agg(['mean','sem'])
    group_stats.sort_values(by='mean',inplace=True)
    ov_avg = good_groups[pcol].mean()
    
    boot_samps = 100
    shuff_avgs = np.zeros((boot_samps,n_groups))
    shuff_data = good_groups.copy()
    for cnt in xrange(boot_samps):
        shuff_data[pcol] = np.random.permutation(shuff_data[pcol].values)
        shuff_avgs[cnt,:] = shuff_data.groupby(group_by)[pcol].mean().sort_values()
    
    yax = np.arange(n_groups)
    rmean = np.mean(shuff_avgs,axis=0)
    rsem = np.std(shuff_avgs,axis=0)

    #plot avg and SEM of within-state returns
    fig, ax1 = plt.subplots(1,1,figsize=(6.0,5.0))
    if group_by == 'zip3':
        ax1.errorbar(group_stats['mean'].values,yax)
    else:
        ax1.errorbar(group_stats['mean'],yax,xerr=group_stats['sem'])
        plt.fill_betweenx(yax, rmean-rsem, rmean+rsem,facecolor='r',alpha=0.5,linewidth=0)
        plt.yticks(yax, group_stats.index,fontsize=6)
    
    ax1.plot(rmean,yax,'r')
    plt.legend(['Measured-avgs','Shuffled-avgs'],loc='best')
    if group_by == 'zip3':    
        plt.ylabel('Zip codes',fontsize=16)
    else:    
        plt.ylabel('States',fontsize=16)
#    if pcol == 'weighted_ROI':
#        plt.xlabel('Annual return (%)',fontsize=16)
#    elif pcol == 'int_rate':
#        plt.xlabel('Interest rate (%)',fontsize=16)
    plt.xlabel(name_legend_map[pcol],fontsize=16)   
    plt.xlim(np.min(group_stats['mean']),np.max(group_stats['mean']))
    plt.ylim(0,n_groups)
    ax1.axvline(ov_avg,color='k',ls='dashed')
    plt.tight_layout()

    return fig

def make_CV_returns_fig(LD,group_by,pcol):
    
    n_samples = len(LD)   
    n_folds = 10 #number of resampling iterations
    #test_size = 0.2 #proportion of samples to set aside for validation
    #shuffler = ShuffleSplit(n_samples,n_iter=n_iter,test_size=test_size) 
    rng_seed = 1
    shuffler = KFold(n_samples,n_folds,shuffle=True, random_state = rng_seed)    
    
    n_groups = len(LD[group_by].unique())
    
    #set of pick best K samples
    if group_by == 'zip3':
        min_pick_K = 25
        pick_K_space = 5
    else:
        min_pick_K = 5
        pick_K_space = 1
    pick_K_axis = np.arange(min_pick_K,n_groups,pick_K_space)
    
    #run n_iter resampling iterations
    good_avgs = np.nan*np.zeros((n_folds,len(pick_K_axis))) #initialization
    for cnt,cur_shuff in enumerate(shuffler):
        train_set, test_set = cur_shuff
        train_avgs = LD.iloc[train_set].groupby(group_by).mean() #compute within-state avgs on the training sample
        train_avgs = train_avgs.sort_values(by=pcol,ascending=False) #sort states by avgs 
        
        #now test performance on the xval set using the top K states, varying K over a range
        xval_data = LD.iloc[test_set]
        for idx,snums in enumerate(pick_K_axis):
            good_groups = train_avgs.index[:snums]
            good_avgs[cnt,idx] = xval_data[pcol][xval_data[group_by].isin(good_groups)].mean()
        good_avgs[cnt,:] = 100*(good_avgs[cnt,:] - xval_data[pcol].mean())/xval_data[pcol].mean()
    

        
    #make plot showing avg cross-validated returns based on state-selectivity
    fig, ax1 = plt.subplots(1,1,figsize=(6.0,5.0))    
    if group_by == 'zip3':    
        plt.xlabel('Number of zip codes picked',fontsize=16)
        plt.errorbar(pick_K_axis,np.mean(good_avgs,axis=0),np.std(good_avgs,axis=0),lw=2,alpha = 0.5)
    elif group_by == 'addr_state':
        plt.xlabel('Number of states picked',fontsize=16)
        plt.errorbar(pick_K_axis,np.mean(good_avgs,axis=0),np.std(good_avgs,axis=0),lw=2,alpha = 0.8)
    if pcol == 'weighted_ROI':
        plt.ylabel('Percent Improvement',fontsize=16)
    elif pcol == 'int_rate':
        plt.ylabel('Percent Difference in interest rates',fontsize=16)
    plt.hlines(y=0,xmin=0,xmax=n_groups,color='black',linestyle='--')
    plt.xlim((0,n_groups))
    plt.tight_layout()
    
    return fig
    
#%%
grouping_vars = ['addr_state','zip3']
response_vars = ['ROI',
                 'int_rate',
                 'annual_inc',
                 'default_prob',
                 'dti',
                 'emp_length']

for group_var in grouping_vars:
    for resp_var in response_vars:
        fig = make_group_cumdist_fig(LD, group_var, resp_var)
        fname = '{}_{}_cdist.png'.format(group_var,resp_var)
        fig.savefig(fig_dir + fname, dpi=500, format='png')
        plt.close('all')
            
#%%
#to_print = [('addr_state','ROI','state_ROI_CVret'),
#            ('zip3','ROI','zip_ROI_CVret')]    
#    
#for print_det in to_print:
#    fig = make_CV_returns_fig(LD,print_det[0],print_det[1])
#    fig.savefig(fig_dir + print_det[2] + '.png',dpi=500,format='png')
#    plt.close('all')
#    
