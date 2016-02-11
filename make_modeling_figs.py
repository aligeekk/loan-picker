"""
@author: James McFarland
"""

import os
import sys
import re
# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn import tree  # decision tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cross_validation import ShuffleSplit, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, LinearSVR
from sklearn.grid_search import GridSearchCV

from collections import defaultdict, namedtuple
import datetime

plot_figures = True
run_CV = False
base_dir = os.path.dirname(os.path.realpath(__file__))
#base_dir = '/Users/james/Data_Incubator/loan-picker'
    
sys.path.append(base_dir)
import LC_helpers as LCH
import LC_loading as LCL
import LC_modeling as LCM

#set paths
data_dir = os.path.join(base_dir,'static/data/')
fig_dir = os.path.join(base_dir,'static/images/')
#fig_dir = os.path.join(base_dir,'tmp/')

#%%
#load data 
data_name = 'all_loans_proc'
LD = pd.read_csv(data_dir + data_name, parse_dates=['issue_d',])

#%% Set up list of predictors and their properties
'''Store info for each predictor as a named tuple containing the col-name within
the pandas dataframe, the full_name (human readable), and the type of normalization
to apply to that feature.'''
predictor = namedtuple('predictor', ['col_name', 'full_name', 'norm_type'])

#dict to create transformers for each specified type
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
            predictor('verification_status','verification status','cat'),
            predictor('latitude','latitude','minMax'),
            predictor('longitude','longitude','minMax')
            ]

#%% Build X and y data
response_var = 'ROI'
y = LD[response_var].values  # response variable
net_returns = LD['net_returns'].values
prnc_weights = LD['prnc_weight'].values
LD.fillna(0, inplace=True)

print('Transforming data')
use_cols = [pred.col_name for pred in predictors]
col_titles = {pred.col_name:pred.full_name for pred in predictors}
recs = LD[use_cols].to_dict(orient='records') #convert to list of dicts
dict_vect = DictVectorizer(sparse=False)
X = dict_vect.fit_transform(recs)                  
print('Done')

feature_names = dict_vect.get_feature_names()
col_dict = defaultdict(list)
tran_dict = {}
for idx, feature_name in enumerate(feature_names):
    short_name = re.findall('[^=]*',feature_name)[0] #get the part before the equals sign, if there is onee
    col_dict[short_name].append(idx)
    pidx = use_cols.index(short_name)
    if predictors[pidx].norm_type in transformer_map:
        tran_dict[use_cols[pidx]] = transformer_map[predictors[pidx].norm_type]
        X[:,idx] = tran_dict[use_cols[pidx]].fit_transform(X[:,idx].reshape(-1,1)).squeeze()

dict_vect.tran_dict = tran_dict

#%% COMPILE LIST OF MODELS TO COMPARE  
Lin_est = Ridge()

svr_est = LinearSVR(epsilon=0)

max_depth=16 
min_samples_leaf=50
min_samples_split=100
n_trees=100 #100
RF_est = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, 
                               min_samples_leaf=min_samples_leaf, 
                               min_samples_split=min_samples_split,n_jobs=-1)

GBR_est = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_trees, 
                                min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split, 
                                max_depth=2)

#%% Run CV grid search if desired.
if run_CV:
    print('Performing grid search on linear model')
    params = {'alpha':[0.1,1.0,10.0,100.,1000.]}
    Lin_model = GridSearchCV(Lin_est, params)
    Lin_model.fit(X,y)
    print('Best {}'.format(Lin_model.best_params_))


    print('Performing grid search on SVR')
    params = {'C':[0.1,1.0,10.]}
    SVR_model = GridSearchCV(svr_est, params)
    SVR_model.fit(X,y)
    print('Best {}'.format(SVR_model.best_params_))


    print('Performing grid search on RF')
    n_features = X.shape[1]
    params = {'max_features':['auto','sqrt','log2']}
    RF_model = GridSearchCV(RF_est, params)
    RF_model.fit(X,y)
    print('Best {}'.format(RF_model.best_params_))


    print('Performing grid search on GBR')
    n_features = X.shape[1]
    params = {'max_features':['auto','sqrt','log2'],
              'max_depth':[2, 3]}
    GBR_model = GridSearchCV(GBR_est, params)
    GBR_model.fit(X,y)
    print('Best {}'.format(GBR_model.best_params_))
else:
    Lin_model = Lin_est.set_params(alpha=100.0)
    SVR_model = svr_est.set_params(C=1.0)
    RF_model = RF_est.set_params(max_features='auto')
    GBR_model = GBR_est.set_params(max_features='auto',
                                    max_depth=3)


#%% Specify set of models to test
model_set = [('Null',LCM.rand_pick_mod()),
            ('Lin', Lin_model),
            ('Lin_SVR',SVR_model),
            ('GBR',GBR_model),
            ('RF', RF_model)]
# model_set = [('Null',LCM.rand_pick_mod()),
#             ('Lin', Lin_model),
#              ('RF', RF_model)]

leg_titles = {'Null':'Random\nPicking',
              'Lin':'Linear\nModel',
              'Lin_SVR':'Linear SVM',
              'GBR':'Gradient\nBoosting',
              'RF':'Random\nForest'}

#%% Compute returns for all models using K-fold cross-val
n_folds = 10
kf = KFold(len(X), n_folds=n_folds, shuffle=True, random_state=0)

pick_K_list = [10, 100, 1000] #list of portfolio sizes to test
grade_pick_K = 100 #portfolio size for computing separate grade-based portfolio returns
n_feature_shuffs = 3 #number of times to shuffle features (at each fold) for computing feature-importances
grade_group = 'grade' #either grade or sub-grade

unique_grades = sorted(LD[grade_group].unique())
test_R2 = defaultdict(list)
returns = defaultdict(list)
marg_returns = []
grade_returns = defaultdict(list)
RF_feature_imp = defaultdict(list)
grade_makeup = {name:np.zeros((len(kf),len(unique_grades))) for name, _ in model_set}
cnt = 0
np.random.seed(0)
for train, test in kf:
 print('CV split {} of {}'.format(cnt, n_folds))
 marg_returns.append(np.sum(net_returns[test]) / np.sum(prnc_weights[test]))
 for name, model in model_set:
     model.fit(X[train,:], y[train])
     test_R2[name].append(model.score(X[test,:],y[test]))
     test_pred = model.predict(X[test,:])

     if name == 'RF': #test feature importance for RF model
         for col_name in use_cols:
             Xshuff = X[test,:].copy()
             col_idx = col_dict[col_name]
             obj = slice(col_idx[0],col_idx[-1]+1)
             shuff_R2 = np.zeros(n_feature_shuffs)
             for n in np.arange(n_feature_shuffs):                
                 np.random.shuffle(Xshuff[:,obj])
                 shuff_R2[n] = model.score(Xshuff, y[test])    
             RF_feature_imp[col_name].append(test_R2[name][-1] - np.mean(shuff_R2))
  
     returns[name].append(LCM.pick_K_returns( 
                             test_pred, net_returns[test], prnc_weights[test],
                            pick_K_list, n_boots=100, sub_marg=False))
                            
     grade_returns[name].append(LCM.pick_K_returns_by_grade(
                             test_pred, net_returns[test], prnc_weights[test],
                            LD.iloc[test][grade_group], grade_pick_K))
                            
     grade_makeup[name][cnt,:] = LCM.get_choice_grade_makeup(test_pred, LD.iloc[test][grade_group], 
                                                             unique_grades, grade_pick_K)     
 cnt += 1

# Annualize portfolio returns, and convert them into numpy arrays as needed
rel_returns = {}
marg_returns = LCM.annualize_returns(np.array(marg_returns))
for name, model in model_set:
    returns[name] = LCM.annualize_returns(np.array(returns[name]))
    rel_returns[name] = returns[name] - marg_returns[:,np.newaxis,np.newaxis]  
    returns[name] = returns[name].reshape(-1, len(pick_K_list))
    rel_returns[name] = rel_returns[name].reshape(-1, len(pick_K_list))
    
    grade_returns[name] = LCM.annualize_returns(np.array(grade_returns[name]))
    grade_returns[name] = grade_returns[name].reshape(-1, len(unique_grades))


#%% PLOT RELATIVE FEATURE IMPORTANCES FOR FULL RF MODEL
feature_imp_df = pd.DataFrame(RF_feature_imp).T
feature_imp_df = feature_imp_df / feature_imp_df.apply(max)
feature_imp_df['avg'] = feature_imp_df.mean(axis=1)
feature_imp_df['sem'] = feature_imp_df.sem(axis=1)
feature_imp_df.sort_values(by='avg',inplace=True)
pal = sns.color_palette("muted")
#colors = [pal[0] if val != 'location' else pal[2] for val in feature_imp_df.index.values]
colors = [pal[0] for val in feature_imp_df.index.values]
bar_width = 0.75
cutoff = 1e-4
feature_imp_df = feature_imp_df.ix[feature_imp_df.avg >= cutoff,:]

fig,ax = plt.subplots(1,1)
bars = plt.bar(np.arange(len(feature_imp_df)),feature_imp_df['avg'], width=bar_width,
     yerr=feature_imp_df['sem'], color=colors)

use_titles = [col_titles[col_name] for col_name in feature_imp_df.index.values]
ax.set_xticks(np.arange(len(feature_imp_df)) + bar_width/2)
ax.set_xticklabels(use_titles, rotation=90, fontsize=12)

plt.ylabel('Relative importance', fontsize=16)
plt.ylim(cutoff,1)
plt.xlim(1,len(use_titles))
plt.yscale('log')
ax.xaxis.grid(None)
for tick in ax.get_xticklabels():
 if tick.get_text() == 'borrower location':
     tick.set_color('red')
plt.tight_layout()

if plot_figures:
    plt.savefig(fig_dir + 'fullRF_feature_imp.png', dpi=500, format='png')
    plt.close()


#%% Plot comparison of model returns
model_names = [name for name, _ in model_set if name != 'Null']
group_titles = [leg_titles[name] for name in model_names]

new_dict = {key: arr[:,0] for key,arr in zip(returns.keys(), 
                         returns.values())}
df1 = pd.melt(pd.DataFrame(new_dict))                           
df1['Loans selected'] = pick_K_list[0]                           
new_dict = {key: arr[:,1] for key,arr in zip(returns.keys(), 
                         returns.values())}
df2 = pd.melt(pd.DataFrame(new_dict))                           
df2['Loans selected'] = pick_K_list[1]                           
new_dict = {key: arr[:,2] for key,arr in zip(returns.keys(), 
                         returns.values())}
df3 = pd.melt(pd.DataFrame(new_dict))                           
df3['Loans selected'] = pick_K_list[2]                           
df = pd.concat([df1,df2,df3],axis=0)

factor_width=0.75

pal = sns.color_palette("muted")
plt.figure(figsize=(8.0,6.0))
ax = sns.violinplot(x='variable',y='value',data=df, hue='Loans selected',
                 split=False, width=factor_width,
                 order = model_names, linewidth=0.5)
plt.ylabel('Annual returns (%)', fontsize=16)
plt.xlabel('Selection method', fontsize=16)
plt.ylim(0,30)
#ax.axhline(y=0,color='k',ls='dashed')
xlim = plt.xlim()
xcent = np.mean(xlim)
ax.axhline(y=np.mean(marg_returns),color=pal[4],lw=3,ls='dashed')
ax.annotate('Average returns', xy=(xcent, np.mean(marg_returns)), xycoords='data',
                xytext=(-10, -50), textcoords='offset points', 
                arrowprops=dict(facecolor='black', shrink=0.05, width=3),
                horizontalalignment='right', verticalalignment='bottom',
                size=14)

ax.set_xticklabels(group_titles, rotation=90, fontsize=13)
ax.legend(loc='lower right')
plt.tight_layout()

n_groups = len(pick_K_list)
pal = sns.color_palette("muted")
medians = df.groupby(['variable','Loans selected'])['value'].median()
medians = medians.ix[model_names]
for idx, pick_N in enumerate(pick_K_list):
    plt.plot(np.arange(len(group_titles)) + idx*factor_width/n_groups - np.ceil(n_groups/2)*factor_width/n_groups,
             medians.ix[:,pick_N],ls='dashed',lw=1,color=pal[idx], alpha=0.5,
             zorder = -1)

if plot_figures:
    plt.savefig(fig_dir + 'full_mod_compare_ROI.png', dpi=500, format='png')
    plt.close()


#%% Plot proportion of grades picked by each model
use_models = ['Null','Lin','RF']
model_names = [name for name,_ in model_set if name in use_models]
n_mods = len(model_names)

all_dfs = []
for name,_ in model_set:
    if name in use_models:
        all_dfs.append(pd.DataFrame({'values':np.mean(grade_makeup[name],axis=0)/100.,'Model':name},
                        index=unique_grades))

makeup_df = pd.concat(all_dfs,axis=0).reset_index()
makeup_df=makeup_df.rename(columns={'index':'grade'})
fig,ax=plt.subplots(figsize=(5.0,4.0))
pal = sns.cubehelix_palette(n_colors=len(unique_grades))
sns.barplot(x='Model',y='values',data=makeup_df,hue='grade',palette=pal,
            hue_order=unique_grades,ax=ax)

ax.legend_.remove()
big_grades = ['A','B','C','D','E','F']
leg_hands = []
if grade_group == 'sub_grade':
    n_subgrades=5
    big_grade_idx = np.arange(len(big_grades))*n_subgrades + n_subgrades//2
else:
    big_grade_idx = np.arange(len(big_grades))
    
for idx in big_grade_idx:
    leg_hands.append(mlines.Line2D([],[],linewidth=4, color=pal[idx]))
ax.legend(leg_hands,big_grades,loc='upper left')
    
plt.ylabel('Proportion of picked loans',fontsize=16)
plt.xlabel('Selection Method',fontsize=16)
plt.tight_layout()
if plot_figures:
    plt.savefig(fig_dir + 'picked_grades_props_{}.png'.format(grade_group), 
                dpi=500, format='png')
    plt.close()


#%% plot avg returns by grade for different models
grades = np.sort(LD[grade_group].unique())
best_returns = LD.groupby(grade_group)['best_NAR'].mean() * 100
best_returns.sort_index(inplace=True)

use_models = ['Null','Lin','RF']
model_names = [name for name,_ in model_set if name in use_models]
n_mods = len(model_names)
#mod_names = ['Random','Linear','Linear SVR','Gradient Boosting','Random Forest']
pal = sns.color_palette("muted", n_colors=len(model_names))

jitt_x = 0.3
alpha = 0.75
#err_norm = np.sqrt(n_folds)
err_norm = 1.0
fig = plt.figure(figsize=(5.0,4.0))
ax = plt.subplot(1,1,1)
for idx, mod_name in enumerate(model_names):
    plt.errorbar(np.arange(len(grades)) + idx*jitt_x/n_mods - jitt_x/(2.),
                 np.mean(grade_returns[mod_name],axis=0),
                 np.std(grade_returns[mod_name],axis=0)/err_norm, 
                color=pal[idx], label=model_names[idx], lw=2, fmt='-', ms=10, alpha=alpha)
                 
plt.plot(np.arange(len(grades)), best_returns,'ko--',lw=2, 
             label='Best')
             
plt.xlim(-0.4, len(grades)+0.4)
plt.ylim(0,30)
grades = np.concatenate(([''],grades))
ax.set_xticks(np.arange(len(grades))-1)
if grade_group == 'sub_grade':
    ax.set_xticklabels(grades, fontsize=12,rotation='vertical')
else:
    ax.set_xticklabels(grades, fontsize=12)
plt.xlabel('Loan Grade',fontsize=16)
plt.ylabel('Annualized ROI (%)',fontsize=16)
plt.legend(loc='best', fontsize=14)
plt.tight_layout()

if plot_figures:
    plt.savefig(fig_dir + 'grade_returns_ROI_{}.png'.format(grade_group), 
                dpi=500, format='png')
    plt.close()


#%% validate generalization across time by training on new data and testing on older and older data
train_day = 2800 #train on data after this day
test_dates = LD.ix[LD.issue_day < train_day,'issue_day'] #set of test loans 

'''Generate a grid of time-points for validation. Set it up to have approximately equi-populated
bins (so the spacing becomes wider further into the past.'''
n_samps = 50 #number of sample time points to test
test_pts = np.percentile(test_dates,np.linspace(100./n_samps,100-100./n_samps,n_samps))
test_pts = test_pts[::-1]

'''Set up Random Forest model to use'''
#dont use time for this analysis
use_cols = [col for col in use_cols if col != 'issue_day']
max_depth=14
n_trees=100
min_samples_leaf=100
RF_mod = RandomForestRegressor(n_estimators=n_trees, max_depth=10, max_features='auto',
                               min_samples_leaf=min_samples_leaf, n_jobs=-1)
                        
train_set = (LD['issue_day'] < train_day).values #bool mask for test set membership
RF_mod.fit(X[train_set], y[train_set]) #fit model

#compute returns at each validation time point
stream_R2 = np.zeros(len(test_pts),)*np.nan
stream_returns = np.zeros((len(test_pts),100))*np.nan
pick_K_list = [100]
n_loans = len(y)
for idx, test_pt in enumerate(test_pts):
    print('stream test {} of {}'.format(idx,len(test_pts)))
    if idx < len(test_pts) - 1:
        test_set = ((LD['issue_day'] <= test_pt) & \
                    (LD['issue_day'] > test_pts[idx+1])).values
    else:
        test_set = (LD['issue_day'] <= test_pt)
        
    n_train, n_test = (np.sum(train_set), np.sum(test_set))
    if n_test > 0:    
        stream_R2[idx] = RF_mod.score(X[test_set,:],y[test_set])
        test_pred = RF_mod.predict(X[test_set,:])
        stream_returns[idx,:] = (LCM.pick_K_returns( 
                                test_pred, net_returns[test_set], 
                                prnc_weights[test_set],pick_K_list,
                                n_boots=100, sub_marg=False)).squeeze()
                 
stream_returns = LCM.annualize_returns((stream_returns))

#%% Plot returns for past-validation
marg_returns = LD.groupby('issue_d')['net_returns'].sum() / LD.groupby('issue_d')['prnc_weight'].sum()
marg_returns = LCM.annualize_returns(marg_returns)

actual_dates = [LD['issue_d'].min() + datetime.timedelta(days=test_pt) for test_pt in test_pts]

actual_dates = np.array(actual_dates)
stream_avg_returns = np.mean(stream_returns,axis=1)
use_pts = ~np.isnan(stream_avg_returns)

pal = sns.color_palette("muted")

fig,ax=plt.subplots()
ax.plot(actual_dates[use_pts],stream_avg_returns[use_pts],'o-',
        color=pal[0],label='model-based selection')
marg_returns.plot(label='marginal returns',color=pal[2])
ax.set_ylim(0,25)
ax2 = ax.twinx()
date_counts = LD.groupby('issue_d')['issue_d'].count()  
ax2.bar(date_counts.index.values,date_counts/1e3,width=10,fc='k',alpha=0.25) 
   
ax.set_ylabel('Annualized returns (%)', color='k',fontsize=14)
ax2.set_ylabel('Loan volume', color='gray',fontsize=14)
ax.set_xlabel('Issue date',fontsize=14)
ax.set_xlim(datetime.date(2009,01,01),LD['issue_d'].max())
ax.legend(loc='upper left',fontsize=14)
ax2.set_yticks([])

ymin, ymax= ax.get_ylim()
xmin, xmax = ax.get_xlim()
xmin = datetime.timedelta(days=train_day) + LD['issue_d'].min()
xmax = LD['issue_d'].max()

from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
xmin = mdates.date2num(xmin)
xmax = mdates.date2num(xmax)
ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, alpha=0.25,
                       facecolor=pal[1]))
ax.annotate('Training data', xy=((xmin+xmax)/2-50., ymax-1.5), xytext=(xmin-700, ymax-2),
            arrowprops=dict(facecolor='black', shrink=0.05))

if plot_figures:
    fig.savefig(fig_dir + 'time_validation_ROI.png', dpi=500, format='png')
    plt.close()
