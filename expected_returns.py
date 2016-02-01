# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 10:52:06 2016

@author: James McFarland
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import NelsonAalenFitter

base_dir = '/Users/james/Data_Incubator/LC_app'
#base_dir = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(base_dir)
import LC_helpers as LCH
import LC_loading as LCL

#%%
data_dir = '/Users/james/Data_Incubator/LC_analysis/Data/'
fig_dir = '/Users/james/Data_Incubator/LC_app/static/images/'
temp_data_dir = '/Users/james/Data_Incubator/LC_app/static/data/'

"""
possible files:  {'LoanStats3a.csv': '2007-2011',
                  'LoanStats3b.csv': '2012-2013',
                  'LoanStats3c.csv': '2013-2014',
                  'LoanStats3d.csv': '2015'}
"""

load_files = ['LoanStats3a.csv','LoanStats3b.csv','LoanStats3c.csv','LoanStats3d.csv'] 

#list of loan statuses to keep
keep_status = ['Current','Fully Paid','Late (16-30 days)','Late (31-120 days)',
               'Default','Charged Off','In Grace Period'] 

keep_terms = [36, 60] #list of loan terms to keep (36,60)
keep_grades = ['A','B','C','D','E','F'] #list of loan grades to keep [A-G]
load_files = [data_dir + file for file in load_files]

LD = LCL.load_lending_data(load_files,keep_status,keep_terms,keep_grades)

print('loaded {0} loans'.format(len(LD)))

print_figs = False

#%% Compute hazard functions for each loan grade and term 
term_bandwidths = [4., 8.] #list of NAF smoothing bandwidth (for each term)
naf = NelsonAalenFitter(nelson_aalen_smoothing=False) #init NAF model

all_hazards = {} #initialize dict to store hazard functions
for idx,term in enumerate(keep_terms): #compute all hazard functions for each term
    
    cur_data = LD[LD.term==term]
    lifetimes = cur_data['num_pymnts'].copy() #lifetime is number of payments received
    lifetimes.ix[cur_data.loan_status == 'Fully Paid'] = term #if the loan is fully paid set the lifetime to the full term
    is_observed = cur_data.loan_status.isin(['Charged Off']) #observed loans are just the ones that have been charged off, rest are censored   
    
    all_hazards[term] = np.zeros((len(keep_grades),term+1)) #initialize matrix of hazard functions

    for gidx,grade in enumerate(keep_grades): #fit model for each grade
        grade_data = cur_data.grade == grade
        naf.fit(lifetimes[grade_data],event_observed=is_observed[grade_data],label=grade,timeline=np.arange(term+1))
        all_hazards[term][gidx,:] = naf.smoothed_hazard_(term_bandwidths[idx]).squeeze()
        
#%% plot hazard functions
if print_figs:
    sns.set_context("talk")
    pal = sns.cubehelix_palette(len(keep_grades)) #set color pallette 
    fig = plt.figure(figsize = (10.0,4.0))
    fig, axarr = plt.subplots(1,2, sharey='row',figsize=(10.0,4.0))
    
    for idx,term in enumerate(keep_terms):
    #    fig = plt.figure(figsize=(6.0,4.0))
        time_ax = np.arange(term+1) 
        plt.sca(axarr[idx])
    #    ax = plt.subplot(1,2,idx+1,sharey=True)     
        #plot hfuns for all grades
        for gidx,grade in enumerate(keep_grades):
            plt.step(time_ax,all_hazards[term][gidx,:],label=grade,color=pal[gidx])    
        plt.xlim((0,term))
        plt.xlabel('Loan duration (months)')
        plt.ylabel('Default probability')
        plt.legend()
        plt.tight_layout()
        plt.title('{} month loans'.format(term))
    #    plt.savefig(fig_dir + '{}month_hazards.png'.format(term), transparent = True, dpi=500, format='png')
    #    plt.close()
    axarr[1].set_ylabel('')
    plt.tight_layout()
    
    plt.savefig(fig_dir + 'hazard_funs.png', dpi=500, format='png')
    plt.close()

    sns.set_context("talk")
    pal = sns.cubehelix_palette(len(keep_grades)) #set color pallette 
    
    for idx,term in enumerate(keep_terms):
        fig = plt.figure(figsize=(5.0,4.0))
        time_ax = np.arange(term+1) 
        for gidx,grade in enumerate(keep_grades):
            plt.step(time_ax,all_hazards[term][gidx,:],label=grade,color=pal[gidx])    
        plt.xlim((0,term))
        plt.xlabel('Month',fontsize=14)
        plt.ylabel('Default probability',fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir + '{}month_hazards.png'.format(term), dpi=500, format='png')
        plt.close()
    
#%%
terms = LD.term.unique() #set of unique loan terms
for term in terms: #for each possible loan term
    
    #get relevant set of loans
    cur_loans = LD.term == term 
    cur_LD = LD[cur_loans]
    
    (NAR, weighted_NAR, MAR) = LCH.get_NARs(cur_LD, term)
    LD.ix[cur_loans,'NAR'] = NAR #measured performance of each loan
    LD.ix[cur_loans,'weighted_NAR'] = weighted_NAR #performance weighted by time contribution to imaginary portfolio
    LD.ix[cur_loans,'MAR'] = MAR #principal weighted avg monthly returns

    (exp_NAR, weighted_exp_NAR, exp_MAR, tot_default_prob, exp_num_pymnts) = \
            LCH.get_expected_NARs(cur_LD, term, all_hazards[term])
    LD.ix[cur_loans,'exp_NAR'] = exp_NAR
    LD.ix[cur_loans,'weighted_exp_NAR'] = weighted_exp_NAR
    LD.ix[cur_loans,'exp_MAR'] = exp_MAR # expected monghtly avg returns
    LD.ix[cur_loans,'default_prob'] = tot_default_prob
    LD.ix[cur_loans,'exp_num_pymnts'] = exp_num_pymnts
    
#for observed loans use the NAR and for censored used the expected NAR
LD['ROI'] = np.zeros((len(LD),))
LD['weighted_ROI'] = np.zeros((len(LD),))

LD.ix[LD.is_observed,'exp_num_pymnts'] = LD.ix[LD.is_observed,'num_pymnts']

LD.ix[LD.is_observed,'ROI'] = LD.ix[LD.is_observed,'NAR'] * 100
LD.ix[~LD.is_observed,'ROI'] = LD.ix[~LD.is_observed,'exp_NAR'] * 100

LD.ix[LD.is_observed,'weighted_ROI'] = LD.ix[LD.is_observed,'weighted_NAR'] * 100
LD.ix[~LD.is_observed,'weighted_ROI'] = LD.ix[~LD.is_observed,'weighted_exp_NAR'] * 100

LD.ix[LD.is_observed,'mnthly_ROI'] = LD.ix[LD.is_observed,'MAR']
LD.ix[~LD.is_observed,'mnthly_ROI'] = LD.ix[~LD.is_observed,'exp_MAR']

#%%
if print_figs:
    grades = np.sort(LD.grade.unique()) #set of unique loan grades
    status_order = ['Fully Paid','Current','In Grace Period','Late (16-30 days)','Late (31-120 days)','Default','Charged Off']
    pal = sns.cubehelix_palette(len(status_order)) #set color pallette 
    
    sns.set_context("talk")

    fig = plt.figure(figsize = (8.0,6.0))
    
    #plot distribution of loan-based ROIs
    ax = sns.boxplot(x="grade", y="ROI", hue='loan_status',data=LD,
                     order=grades, hue_order = status_order,fliersize=0,
                     palette = pal)
    plt.legend(loc='upper left',fontsize=12)
    plt.ylabel('Expected returns (%)',fontsize=16)
    plt.xlabel('Grade',fontsize=16)
    plt.axhline(y=0.,color='k',ls='dashed')
    plt.ylim(-100,80)
    plt.tight_layout()
    
    plt.savefig(fig_dir + 'returns_box.png', dpi=500, format='png')
    plt.close()


    import matplotlib.lines as mlines
    not_early_pay = ~((LD.loan_status == 'Fully Paid') & (LD.num_pymnts < LD.term/10.))
    loan_status = ['Fully Paid', 'Current',  'In Grace Period',
            'Late (16-30 days)','Late (31-120 days)', 'Default','Charged Off']
    pal = sns.cubehelix_palette(n_colors=len(loan_status))
    scatter_kws={'alpha':0.5}
    sns.lmplot(x='ROI',y='default_prob',data=LD.ix[not_early_pay],hue='loan_status',
               hue_order = loan_status, palette=pal,fit_reg=False,

               scatter_kws=scatter_kws, size=5, aspect=1.4, 
               legend=False)
    items = []
    for idx, ls in enumerate(loan_status):
        items.append(mlines.Line2D([], [], color=pal[idx], marker='o',
                          markersize=10, label=ls,lw=0))
    plt.legend(items,loan_status,loc='center right',fontsize=12)
    plt.ylim(0,1)
    plt.xlim(-100,80)
    plt.xlabel('Expected returns (%)',fontsize=16)
    plt.ylabel('Probability of default',fontsize=16)
    plt.savefig(fig_dir + 'ROI_def_prob.png', dpi=500, format='png')
    plt.close()

#%%
stat_counts_by_grade = LD.groupby(['grade','loan_status']).agg({'grade':'count'}) #cuisine-conditional violations counts
tot_by_grade = LD.groupby('grade').agg({'grade':'count'}) #total violations by cuisine
grade_cond_stat_dist = stat_counts_by_grade.div(tot_by_grade,level='grade') #cuisine-conditional violation dist
grade_cond_stat_dist.rename(columns={'grade':'num'},inplace=True)
grade_cond_stat_dist = grade_cond_stat_dist.reset_index()

tot_by_status = LD.groupby('loan_status').agg({'loan_status':'count'}) #total violations by cuisine
tot_by_status = tot_by_status/1000 #convert counts to thousands
tot_by_status.rename(columns={'loan_status':'num'},inplace=True)
tot_by_status = tot_by_status.reset_index()

#%%
if print_figs:
    status_order = ['Fully Paid','In Grace Period','Late (16-30 days)','Late (31-120 days)','Default','Charged Off']
    pal = sns.cubehelix_palette(len(status_order)) #set color pallette 
    
    sns.set_context("talk")
    g = sns.factorplot(x="grade", y="num", hue="loan_status", size=5,aspect = 1.2,
                      data=grade_cond_stat_dist, hue_order = status_order, kind="bar",
                      palette=pal,legend_out = False)
    plt.ylabel('Relative frequency')
    plt.xlabel('Grade')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plt.savefig(fig_dir + 'grade_cond_dists.png', dpi=500, format='png')
    plt.close()
    
    #%%
    status_order = ['Fully Paid','Current','In Grace Period','Late (16-30 days)','Late (31-120 days)','Default','Charged Off']
    pal = sns.cubehelix_palette(len(status_order)) #set color pallette 
    
    sns.set_context("talk")
    g = sns.factorplot(x="loan_status", y="num", size=5.0,aspect = 1.0,
                      data=tot_by_status, order = status_order, kind="bar",
                      palette=pal,legend_out = False)
    
    plt.xticks(rotation=70)
    plt.xlabel('Loan Status',fontsize=16)
    plt.ylabel('Number of loans (thousands)',fontsize=16)
    plt.legend(loc='upper right',fontsize=14)
    plt.tight_layout()
    
    plt.savefig(fig_dir + 'marg_status_dist.png', dpi=500, format='png')
    plt.close()

#%%
if print_figs:
    LD['year'] = LD['issue_d'].dt.year
    last_2015_loan = LD.ix[LD.year==2015,'issue_d'].dt.month.max()
    obs_frac_2015 = last_2015_loan/12.0
    
    stat_counts_by_year = LD.groupby(['year','loan_status']).agg({'year':'count'}) #cuisine-conditional violations counts
    #stat_counts_by_year = stat_counts_by_year/1e3
    stat_counts_by_year.ix[2015] = stat_counts_by_year.ix[2015].values/obs_frac_2015
    stat_counts_by_year = stat_counts_by_year.rename(columns={'year':'num'})
    stat_counts_by_year = stat_counts_by_year.reset_index()
    
    status_order = ['Fully Paid','Current','In Grace Period','Late (16-30 days)','Late (31-120 days)','Default','Charged Off']
    pal = sns.cubehelix_palette(len(status_order)) #set color pallette 
    
    sns.set_context("talk")
    g = sns.factorplot(x="year", y="num", hue="loan_status", size=5,aspect = 1.2,
                      data=stat_counts_by_year, hue_order = status_order, kind="bar",
                      palette=pal,legend_out = False)
    plt.ylabel('Number of loans')
    plt.xlabel('Issue Year')
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.savefig(fig_dir + 'year_cond_dist.png', dpi=500, format='png')
    plt.close()

#%%
#status_order = ['Fully Paid','Current','In Grace Period','Late (16-30 days)','Late (31-120 days)','Default','Charged Off']
#pal = sns.cubehelix_palette(len(status_order)) #set color pallette 
#
#scatter_kws = {'alpha':0.05, 's':8,'linewidth': None}
## sns.lmplot(x='int_rate',y='weighted_exp_NAR',hue='loan_status',data=LD[~LD.is_observed],
##               hue_order = status_order, scatter_kws = scatter_kws)
#sns.lmplot(x='int_rate',y='weighted_ROI',hue='loan_status',data=LD,
#               hue_order = status_order,scatter_kws = scatter_kws, scatter=True,fit_reg = False)

#%%
if print_figs:
    status_order = [['Current'],['In Grace Period'],['Late (16-30 days)','Late (31-120 days)'],['Default']]
    status_titles = ['Current','Grace Period','Late','Defaulted']
    plt.close('all')
    fig, axarr = plt.subplots(2,2,figsize=(8,8),sharex="all",sharey="all")
    axarr = axarr.flat
    max_dp = 10000
    for idx, status_list in enumerate(status_order):
        subset = LD.ix[LD.loan_status.isin(status_list)]
        if len(subset) > max_dp:
            subset = subset.sample(max_dp)
        #plt.subplot(2,2,idx+1)
        sns.kdeplot(subset.int_rate,subset.weighted_ROI,shade=True, cmap='Reds', 
                    shade_lowest = False, ax=axarr[idx],bw=(1.5,1.5))
        axarr[idx].set_title(status_titles[idx])
        axarr[idx].set_xlabel('Interest rate (%)')
        axarr[idx].set_ylabel('Expected ROI (%)')
        axarr[idx].set_xlim(0,30)
        axarr[idx].set_ylim(-50,30)
        axarr[idx].axhline(y=0.,color='k',ls='dashed')
    axarr[1].set_ylabel('')
    axarr[3].set_ylabel('')
    axarr[0].set_xlabel('')
    axarr[1].set_xlabel('')
    plt.tight_layout()
    plt.savefig(fig_dir + 'int_ROI_joints.png', dpi=500, format='png')
    plt.close()
  
#%% total dollar issuance by years
if print_figs:
    last_2015_loan = LD.ix[LD.year==2015,'issue_d'].dt.month.max()
    obs_frac_2015 = last_2015_loan/12.0
    
    tot_by_year = LD.groupby('year').agg({'funded_amnt':'sum'}) #cuisine-conditional violations counts
    tot_by_year = tot_by_year/int(1e9)
    tot_by_year.ix[2015] = tot_by_year.ix[2015].values/obs_frac_2015
    
    tot_by_year.reset_index(inplace=True)
    fig,ax=plt.subplots(figsize=(5.0,4.0))
    sns.barplot("year", y="funded_amnt", data=tot_by_year,
                     palette="Blues", ax=ax)
    
    ax.set_ylabel('Loan issuance (billions USD)',fontsize=16)
    ax.set_xlabel('Year',fontsize=16)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=70)
    plt.tight_layout()
    plt.savefig(fig_dir + 'tot_loan_issuance.png', dpi=500, format='png')
    plt.close()

#%% PLOT EXAMPLE PAYMENT PROB PLOT
if print_figs:
    outcome_map = { #expected principal recovery given status, view this is prob of all princ being charged off
        'Current': 0.,
        'In Grace Period': 28./100,
        'Late (16-30 days)': 58./100,
        'Late (31-120 days)': 74./100}
    
    outcome_ord = ['Current','In Grace Period','Late (16-30 days)','Late (31-120 days)']
    late_mnths = [0,0,1,2]
    
    plt.close('all')
    pal = sns.dark_palette('seagreen',n_colors = len(outcome_ord), reverse=True)
        
    age = 6
    term = 36
    grad_idx = 5
    
    print_stages = [[outcome_ord[0]], outcome_ord[:2], outcome_ord]
    for print_idx in xrange(len(print_stages)):   
        fig,ax=plt.subplots(figsize=(5.0,4.0))
        cur_outcomes = print_stages[print_idx]
        cond_pays = {}
        for idx, out in enumerate(cur_outcomes):
            cur_haz = all_hazards[term][grad_idx,:].copy()
            cur_age = age - late_mnths[idx]
            cur_haz[:cur_age] = 0.0
            pay_prob = 1 - np.cumsum(cur_haz)
            cond_pays[out] = pay_prob.copy()
            cond_pays[out][cur_age:] = pay_prob[cur_age:] * (1-outcome_map[out])
            plt.plot(cond_pays[out],'o-',color=pal[idx], label=out, alpha=0.5, ms=6)
        
        ax.axvspan(0, age, alpha=0.25, color='black')
        plt.ylim(0,1)
        plt.xlim(0,term)
        plt.ylabel('Probability payment received',fontsize=14)
        plt.xlabel('Month',fontsize=14)
    #    plt.legend(loc='best',fontsize=14)
        plt.tight_layout()
        plt.savefig(fig_dir + 'payprob_examp_{}.png'.format(print_idx), dpi=500, format='png')
        plt.close()

#%%
save_columns = ['ROI',
 'acc_now_delinq',
 'addr_state',
 'annual_inc',
 'annual_inc_joint',
 'collections_12_mths_ex_med',
 'cr_line_dur',
 'delinq_2yrs',
 'desc_length',
 'default_prob',
 'dti',
 'dti_joint',
 'emp_length',
 'exp_num_pymnts',
 'funded_amnt',
 'grade',
 'home_ownership',
 'inq_last_6mths',
 'initial_list_status',
 'int_rate',
 'is_observed',
 'issue_d',
 'issue_day',
 'last_credit_pull_dur',
 'loan_status',
 'mnthly_ROI',
 'mths_since_last_delinq',
 'mths_since_last_major_derog',
 'mths_since_last_record',
 'num_add_desc',
 'num_pymnts',
 'open_acc',
 'pub_rec',
 'purpose',
 'revol_bal',
 'revol_util',
 'sub_grade',
 'term',
 'total_acc',
 'tot_coll_amt',
 'tot_cur_bal',
 'verification_status',
 'weighted_ROI',
 'zip3']
 
csv_name = 'all_loans_proc'
save_loans = LD[save_columns]
save_loans.to_csv(temp_data_dir + csv_name) 