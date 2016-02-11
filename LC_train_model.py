# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:58:57 2016

@author: James McFarland
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict, namedtuple
import datetime
import dill
import matplotlib.pylab as plt
import matplotlib.lines as mlines

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

#base_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = '/Users/james/Data_Incubator/loan-picker'
    
sys.path.append(base_dir)
import LC_helpers as LCH
import LC_loading as LCL
import LC_modeling as LCM

plot_calibration = True

#set paths
data_dir = os.path.join(base_dir,'static/data/')
fig_dir = os.path.join(base_dir,'static/images/')
plot_figures = True
#%%
#load data 
data_name = 'all_loans_proc'
LD = pd.read_csv(data_dir + data_name, parse_dates=['issue_d',])

#%%
'''Store info for each predictor as a named tuple containing the col-name within
the pandas dataframe, the full_name (human readable), and the type of normalization
to apply to that feature.'''
predictor = namedtuple('predictor', ['col_name', 'full_name', 'norm_type'])

transformer_map = {'minMax':MinMaxScaler(),
                   'maxAbs':MaxAbsScaler(),
                   'standScal':StandardScaler(),
                   'log_minmax': LCM.log_minmax(),
                   'robScal':RobustScaler()}

predictors = [
            predictor('acc_now_delinq','num delinq accounts','maxAbs'),
            predictor('annual_inc','annual income','log_minmax'),
            predictor('collections_12_mths_ex_med','num recent collections','maxAbs'),
            predictor('cr_line_dur', 'duration cred line','standScal'),
            predictor('delinq_2yrs', 'num recent delinq','maxAbs'),
            predictor('desc_length', 'loan desc length','maxAbs'),
            predictor('dti', 'debt-income ratio','standScal'),
            predictor('emp_length', 'employment length','maxAbs'),
#            predictor('funded_amnt','loan amount','maxAbs'),
            predictor('loan_amnt','loan amount','maxAbs'),
            predictor('inq_last_6mths', 'num recent inqs','maxAbs'),
            predictor('int_rate', 'interest rate','maxAbs'),
#            predictor('issue_day', 'issue date','maxAbs'),
            predictor('mths_since_last_delinq','mths since delinq','maxAbs'),
            predictor('mths_since_last_major_derog','mths since derog','maxAbs'),
            predictor('mths_since_last_record','mths since rec','maxAbs'),
#            predictor('num_add_desc','num descripts added','maxAbs'),
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

#%% Build X and y data
response_var = 'ROI'
y = LD[response_var].values  # response variable
dp = LD['default_prob'].values  # response variable
net_returns = LD['net_returns'].values
prnc_weights = LD['prnc_weight'].values
   
LD.fillna(0, inplace=True)
is_observed = LD.is_observed.values

print('Transforming data')
use_cols = [pred.col_name for pred in predictors]
col_titles = {pred.col_name:pred.full_name for pred in predictors}
recs = LD[use_cols].to_dict(orient='records') #convert to list of dicts
dict_vect = DictVectorizer(sparse=False)
X = dict_vect.fit_transform(recs)                  
print('Done')

#%%
feature_names = dict_vect.get_feature_names()
col_dict = defaultdict(list)
tran_dict = {}
for idx, feature_name in enumerate(feature_names):
    short_name = re.findall('[^=]*',feature_name)[0] #get the part before the equals sign, if there is one
    col_dict[short_name].append(idx)
    pidx = use_cols.index(short_name)
    if predictors[pidx].norm_type in transformer_map:
        tran_dict[short_name] = clone(transformer_map[predictors[pidx].norm_type])
        X[:,idx] = tran_dict[use_cols[pidx]].fit_transform(X[:,idx].reshape(-1,1)).squeeze()

transformer_tuple = (dict_vect, col_dict, tran_dict, predictors)

#%% Save data for transformers
with open(os.path.join(base_dir,'static/data/trans_tuple.pkl'),'wb') as out_strm:
    dill.dump(transformer_tuple, out_strm)
   
   
#%%
def make_class_train_data(X, dp, is_observed):
    '''Helper function that formats feature matrix and set of default probabilities
    so that they can be fed into sklearn's classifier as sample_weights'''
    X_NO = X[~is_observed,:]
    dp_NO = dp[~is_observed]
    X_NO = np.vstack([X_NO, X_NO])
    class_NO = np.hstack([np.ones(len(dp_NO)), np.zeros(len(dp_NO))])
    sample_NO = np.hstack([dp_NO, 1 - dp_NO])
    
    clf_X = np.vstack([X[is_observed,:], X_NO])
    clf_Y = np.hstack([dp[is_observed], class_NO])
    clf_SW = np.hstack([np.ones(np.sum(is_observed)), sample_NO])
    return clf_X, clf_Y, clf_SW

def get_calibration_curve(obs, pred, n_bins):
    '''Make a classifier calibration curve'''
    bins = np.percentile(pred, np.linspace(0,100, n_bins + 1))
    binids = np.digitize(pred, bins) - 1
    unique_binids = np.unique(binids)
    bin_pred = np.zeros(len(unique_binids)-1)
    bin_true = np.zeros(len(unique_binids)-1)
    for idx,binid in enumerate(unique_binids[:-1]):
        curset = binids == binid
        bin_pred[idx] = np.mean(pred[curset])
        bin_true[idx] = np.mean(obs[curset])
    
    return bin_true, bin_pred

#%% DEFINE MODELS
#CLASSIFIER
max_depth=16 #16
min_samples_leaf=50
min_samples_split=100
n_trees=100 #100
RF_defClass = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, 
                               min_samples_leaf=min_samples_leaf, 
                               min_samples_split=min_samples_split,n_jobs=4, 
                               max_features='auto')

#REGRESSOR
max_depth=16
min_samples_leaf=50
min_samples_split=100
n_trees=100 #100
RF_est = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, 
                               min_samples_leaf=min_samples_leaf, 
                               min_samples_split=min_samples_split,n_jobs=4, 
                               max_features='auto')
#%%
if plot_calibration:
    test_frac = 0.2
    test_inds = np.random.choice(len(X), size=int(test_frac*len(X)), replace=False)
    train_inds = np.setdiff1d(np.arange(len(X)), test_inds)
    
    clf_X, clf_Y, clf_SW = make_class_train_data(X[train_inds,:], dp[train_inds], 
                                                 is_observed[train_inds])
    
    
    RF_defClass.fit(clf_X, clf_Y, sample_weight=clf_SW)
    prob_test = RF_defClass.predict_proba(X[test_inds,:])[:,1]
    prob_train = RF_defClass.predict_proba(X[train_inds,:])[:,1]
    
    n_bins = 30
    uncal_true, uncal_pred = get_calibration_curve(dp[test_inds], prob_test, n_bins)
    
    fig,ax = plt.subplots(figsize=(5.0,4.0))    
    ax.plot(uncal_pred, uncal_true, "s-")
    #ax.plot(cal_pred, cal_true, "rs-", label='uncalibrated')
    ax.plot([0, 1], [0, 1], 'k--')
    plt.xlim(0,0.3)
    plt.ylim(0,0.3)
    plt.xlabel('Predicted probability',fontsize=14)
    plt.ylabel('Observed probability',fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.savefig(fig_dir + 'class_calib.png', dpi=500, format='png')
    plt.close()

#%% Fit random forest classifier for predicting default probabilities
'''To train the classifier using default probs (rather than just class labels)
as target variables, make a copy of all the data whose class is not observed,
one for each class label, with the default probs counting as sample_weights'''

clf_X, clf_Y, clf_SW = make_class_train_data(X, dp, is_observed)
RF_defClass.fit(clf_X, clf_Y, sample_weight=clf_SW)

#%% Fit Random Forest model
RF_est.fit(X,y)

#%% Plot relationship between predicted returns and default prob
if plot_figures:
    pred_ret = RF_est.predict(X)
    pred_dp = RF_defClass.predict_proba(X)
    df = pd.DataFrame({'ret':pred_ret * 100, 'dp':pred_dp[:,1], 'grade': LD['grade']})
    
    grade_order = sorted(LD['grade'].unique())
    
    pal = sns.cubehelix_palette(n_colors=len(grade_order))
    fgrid = sns.lmplot(x='ret', y='dp', data=df, hue='grade', fit_reg=False,
               hue_order = grade_order, legend=False, palette=pal,
               size=4.0, aspect=1.2,
               scatter_kws={'s':4, 'alpha':0.25})
    ax = fgrid.axes[0][0]
    leg_hands = []
    leg_labels = []
    for idx, grade in enumerate(grade_order):
        leg_hands.append(mlines.Line2D([],[],marker='.',linewidth=0,
                                       markersize=12,color=pal[idx]))
        leg_labels.append(grade)
    ax.legend(leg_hands, leg_labels, loc='lower left',fontsize=12)
    
    plt.xlabel('Predicted annual returns (%)',fontsize=14)
    plt.ylabel('Predicted default probability',fontsize=14)
    plt.xlim(-10,20)
    plt.ylim(0,0.35)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig(fig_dir + 'ret_dp_compare.png', dpi=500, format='png')
    plt.close()

#%% Save Random Forest models
with open(os.path.join(base_dir,'static/data/LC_model.pkl'),'wb') as out_strm:
    dill.dump(RF_est, out_strm)
    
with open(os.path.join(base_dir,'static/data/LC_def_model.pkl'),'wb') as out_strm:
    dill.dump(RF_defClass, out_strm)
    
#%%
'''Now get a reference set of cross-validated tuples of (pred, weight_obs, dur)
Need to refit a separate model using a train-test split. Then use the validation set
to generate a lookup table of predicted and measured returns, so we can estimate
likely realized portfolio returns for a given set of model predicted returns'''
#split_date = datetime.datetime(2015,01,01)
#train = (LD['issue_d'] < split_date).values
#test = (LD['issue_d'] >= split_date).values
test_size = 0.33
SS = ShuffleSplit(len(LD), n_iter=1, test_size=test_size, random_state=0)
train, test = list(SS)[0]

RF_xval = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, 
                               min_samples_leaf=min_samples_leaf, 
                               min_samples_split=min_samples_split,n_jobs=4,
                               max_features='auto')

RF_xval.fit(X[train],y[train])
test_pred = RF_xval.predict(X[test])
test_net_returns = net_returns[test]
test_ROI = y[test]
test_weights = prnc_weights[test]

val_set = zip(test_pred, test_ROI, test_net_returns, test_weights)
val_set = sorted(val_set, reverse=True)

#%% Save the set of validation data
with open(os.path.join(base_dir,'static/data/LC_test_res.pkl'),'wb') as out_strm:
    dill.dump(val_set, out_strm)

