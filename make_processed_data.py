# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 10:52:06 2016

@author: James McFarland
"""
import numpy as np
import pandas as pd
from lifelines import NelsonAalenFitter

base_dir = '/Users/james/Data_Incubator/loan-picker'
#base_dir = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(base_dir)
import LC_helpers as LCH
import LC_loading as LCL

#%%
data_dir = '/Users/james/Data_Incubator/LC_analysis/Data/'
fig_dir = '/Users/james/Data_Incubator/loan-picker/static/images/'
temp_data_dir = '/Users/james/Data_Incubator/loan-picker/static/data/'

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

#%%
#load long/lat data for each zip-code
zip3_data = LCL.load_location_data(data_dir,group_by='zip3') 
LD['zip3'] = LD['zip3'].astype(int)       
LD = pd.merge(LD, zip3_data, how='inner', left_on='zip3', right_index=True)

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
        
#%%
terms = LD.term.unique() #set of unique loan terms
for term in terms: #for each possible loan term  
    #get relevant set of loans
    cur_loans = LD.term == term 
    cur_LD = LD[cur_loans]
    
    (NAR, net_returns, p_csum) = LCH.get_NARs(cur_LD, term)
    LD.ix[cur_loans,'ROI'] = NAR #measured performance of each loan
    LD.ix[cur_loans,'net_returns'] = net_returns #principal weighted avg monthly returns
    LD.ix[cur_loans,'prnc_weight'] = p_csum #principal weighted avg monthly returns
    LD.ix[cur_loans,'default_prob'] = LD.ix[cur_loans,'is_observed'].astype(float) #principal weighted avg monthly returns

    (exp_NAR, tot_default_prob, exp_num_pymnts, exp_net_returns, exp_csum) = \
            LCH.get_expected_NARs(cur_LD, term, all_hazards[term])
    LD.ix[cur_loans,'ROI'] = exp_NAR
    LD.ix[cur_loans,'default_prob'] = tot_default_prob
    LD.ix[cur_loans,'exp_num_pymnts'] = exp_num_pymnts
    LD.ix[cur_loans,'net_returns'] = exp_net_returns
    LD.ix[cur_loans,'prnc_weight'] = exp_csum   
    
    LD.ix[cur_loans, 'best_NAR'] = LCH.get_best_returns(cur_LD, term)

LD.ix[LD.is_observed,'exp_num_pymnts'] = LD.ix[LD.is_observed,'num_pymnts']

#%%
save_columns = ['ROI',
 'acc_now_delinq',
 'addr_state',
 'annual_inc',
 'annual_inc_joint',
 'best_NAR',
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
 'latitude',
 'longitude',
 'loan_amnt',
 'loan_status',
 'mths_since_last_delinq',
 'mths_since_last_major_derog',
 'mths_since_last_record',
 'net_returns',
 'num_add_desc',
 'num_pymnts',
 'open_acc',
 'prnc_weight',
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
 'zip3']
 
csv_name = 'all_loans_proc'
save_loans = LD[save_columns]
save_loans.to_csv(temp_data_dir + csv_name) 

#%% make compressed dataset for pushing to Heroku web app
save_columns = ['ROI',
 'addr_state',
 'annual_inc',
 'default_prob',
 'dti',
 'emp_length',
 'funded_amnt',
 'grade',
 'home_ownership',
 'int_rate',
 'issue_d',
 'funded_amnt',
 'loan_status',
 'purpose',
 'term',
 'zip3']
 
csv_name = 'all_loans_proc_short'
save_loans = LD[save_columns]
save_loans.to_csv(temp_data_dir + csv_name) 