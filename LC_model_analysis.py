import os
import sys

# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn import tree  # decision tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cross_validation import ShuffleSplit, KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, LinearSVR
from sklearn.grid_search import GridSearchCV

from collections import defaultdict, namedtuple
import datetime

plot_figures = True
run_CV = False
base_dir = os.path.dirname(os.path.realpath(__file__))
#base_dir = '/Users/james/Data_Incubator/LC_app'
    
sys.path.append(base_dir)
import LC_helpers as LCH
import LC_loading as LCL
import LC_models as LCM

#set paths
data_dir = os.path.join(base_dir,'static/data/')
#fig_dir = os.path.join(base_dir,'static/images/')
fig_dir = os.path.join(base_dir,'tmp/')
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
                   'standScal':StandardScaler(),
                    'log_minmax': LCM.log_minmax(),
                    'robScal':RobustScaler()}

                   
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
    
Lin_est = Ridge()

svr_est = LinearSVR(epsilon=0)

max_depth=14 #16
min_samples_leaf=50
min_samples_split=100
n_trees=100 #100
RF_est = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, 
                               min_samples_leaf=min_samples_leaf, 
                               min_samples_split=min_samples_split,n_jobs=4)

GBR_est = GradientBoostingRegressor(learning_rate=0.1, n_estimators=n_trees, 
                                min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split, 
                                max_depth=2)

#%%   
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


#%%
model_set = [('Null',LCM.rand_pick_mod()),
             ('Lin', Lin_model),
             ('Lin_SVR',SVR_model),
             ('GBR',GBR_model),
             ('RF', RF_model)]
#model_set = [('Lin', Lin_model),
#             ('RF', RF_model)]

#%%
n_folds = 10
kf = KFold(len(X), n_folds=n_folds, shuffle=True, random_state=0)

pick_K_list = [10, 100, 1000]
grade_pick_K = 100
n_feature_shuffs = 3

grade_group = 'grade' #either grade or sub-grade
unique_grades = sorted(LD[grade_group].unique())
train_R2 = defaultdict(list)
test_R2 = defaultdict(list)
returns = defaultdict(list)
marg_returns = []
grade_returns = defaultdict(list)
RF_feature_imp = defaultdict(list)
grade_makeup = {name:np.zeros((len(kf),len(unique_grades))) for name,_ in model_set}
cnt = 0
np.random.seed(0)
for train, test in kf:
 print('CV split {} of {}'.format(cnt, n_folds))
 marg_returns.append(np.mean(weight_y[test]) / np.mean(Npymnts[test]))
 for name, model in model_set:
     model.fit(X[train,:], y[train])
     train_R2[name].append(model.score(X[train,:],y[train]))
     test_R2[name].append(model.score(X[test,:],y[test]))
     train_pred = model.predict(X[train,:])
     test_pred = model.predict(X[test,:])

     if name == 'RF': #test feature importance for RF model
         for col_name in all_pred_cnames:
             Xshuff = X[test,:].copy()
             col_idx = col_dict[col_name]
             obj = slice(col_idx[0],col_idx[-1]+1)
             shuff_R2 = np.zeros(n_feature_shuffs)
             for n in np.arange(n_feature_shuffs):                
                 np.random.shuffle(Xshuff[:,obj])
                 shuff_R2[n] = model.score(Xshuff, y[test])    
             RF_feature_imp[col_name].append(test_R2[name][-1] - np.mean(shuff_R2))
  
     returns[name].append(LCM.pick_K_returns_BTSTRP(weight_y[test], 
                             test_pred, pick_K_list,Npymnts[test],n_boots=100, sub_marg=False))
     grade_returns[name].append(LCM.pick_K_returns_by_grade(weight_y[test], 
                             test_pred,LD.iloc[test][grade_group],grade_pick_K,
                             Npymnts[test]))
     grade_makeup[name][cnt,:] = LCM.get_choice_grade_makeup(test_pred, LD.iloc[test][grade_group], 
                                                            grade_group, unique_grades, grade_pick_K)     
 cnt += 1

rel_returns = {}
marg_returns = LCM.annualize_returns(np.array(marg_returns))
for name, model in model_set:
    returns[name] = LCM.annualize_returns(np.array(returns[name]))
    rel_returns[name] = returns[name] - marg_returns[:,np.newaxis,np.newaxis]  
    grade_returns[name] = LCM.annualize_returns(np.array(grade_returns[name]))
    returns[name] = np.reshape(returns[name], (-1,len(pick_K_list)))
    rel_returns[name] = np.reshape(rel_returns[name], (-1,len(pick_K_list)))


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

#plt.xticks(np.arange(len(feature_imp_df))+bar_width, use_titles, rotation='vertical')

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

#%%
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
                 order = ['Lin','Lin_SVR','GBR', 'RF'],
                linewidth=0.5)
plt.ylabel('Annual returns (%)',fontsize=16)
plt.xlabel('Selection method',fontsize=16)
plt.ylim(0,30)
#ax.axhline(y=0,color='k',ls='dashed')
xlim = plt.xlim()
xcent = np.mean(xlim)
ax.axhline(y=np.mean(marg_returns),color=pal[4],lw=3,ls='dashed')
ax.annotate('Random Picking', xy=(xcent, np.mean(marg_returns)), xycoords='data',
                xytext=(-10, -50), textcoords='offset points', 
                arrowprops=dict(facecolor='black', shrink=0.05, width=3),
                horizontalalignment='right', verticalalignment='bottom',
                size=14)
group_titles = ['Random\nPicking',
                'Linear\nModel',
                'Linear SVM',
                'Gradient\nBoosting',
                'Random\nForest']
ax.set_xticklabels(group_titles, rotation=90, fontsize=13)
ax.legend(loc='lower right')
plt.tight_layout()

n_groups = len(pick_K_list)
pal = sns.color_palette("muted")
medians = df.groupby(['variable','Loans selected'])['value'].median()
medians = medians.ix[['Lin','Lin_SVR','GBR','RF']]
for idx, pick_N in enumerate(pick_K_list):
    plt.plot(np.arange(len(group_titles)) + idx*factor_width/n_groups - np.ceil(n_groups/2)*factor_width/n_groups,
             medians.ix[:,pick_N],ls='dashed',lw=1,color=pal[idx], alpha=0.5,
             zorder = -1)

if plot_figures:
    plt.savefig(fig_dir + 'full_mod_compare.png', dpi=500, format='png')
    plt.close()

#%%
all_dfs = []
for name,_ in model_set:
    all_dfs.append(pd.DataFrame({'values':np.mean(grade_makeup[name],axis=0)/100.,'Model':name},
                                index=unique_grades))

makeup_df = pd.concat(all_dfs,axis=0).reset_index()
makeup_df=makeup_df.rename(columns={'index':'grade'})
fig,ax=plt.subplots(figsize=(5.0,4.0))
pal = sns.cubehelix_palette(n_colors=len(unique_grades))
sns.barplot(x='Model',y='values',data=makeup_df,hue='grade',palette=pal,
            hue_order=unique_grades,ax=ax)
plt.ylabel('Proportion of picked loans',fontsize=16)
plt.xlabel('Selection Method',fontsize=16)
plt.tight_layout()
if plot_figures:
    plt.savefig(fig_dir + 'picked_grades_props.png', dpi=500, format='png')
    plt.close()

#%% plot avg returns by grade for different models
grades = np.sort(LD[grade_group].unique())
avg_Gint_rates = LD.groupby(grade_group)['int_rate'].mean()
avg_Gint_rates.sort_index(inplace=True)
best_returns = 100 * ((avg_Gint_rates/100/12 + 1) ** 12 - 1)

pal = sns.color_palette("muted", n_colors=4)

jitt_x = 0.6
alpha = 0.75
#err_norm = np.sqrt(n_folds)
err_norm = 1.0
mod_list = ['Null','Lin','Lin_SVR','GBR','RF']
n_mods = len(mod_list)
mod_names = ['Random','Linear','Linear SVR','Gradient Boosting','Random Forest']
fig = plt.figure(figsize=(6.0,5.0))
ax = plt.subplot(1,1,1)
for idx, mod_name in enumerate(mod_list):
    plt.errorbar(np.arange(len(grades)) + idx*jitt_x/n_mods - jitt_x/(2.),
                 np.mean(grade_returns[mod_name],axis=0),
                 np.std(grade_returns[mod_name],axis=0)/err_norm, 
                color=pal[idx], label=mod_names[idx], lw=2, fmt='o-', ms=10, alpha=alpha)
                 
plt.plot(np.arange(len(grades)), best_returns,'k',lw=2, 
             ls='dashed', label='Interest rate')
             
plt.xlim(-0.2, 5.2)
plt.ylim(0,25)
grades = np.concatenate(([''],grades))
ax.set_xticklabels(grades, fontsize=14)
plt.xlabel('Loan Grade',fontsize=16)
plt.ylabel('Annualized ROI (%)',fontsize=16)
plt.legend(loc='best', fontsize=14)
plt.tight_layout()

if plot_figures:
    plt.savefig(fig_dir + 'grade_returns.png', dpi=500, format='png')
    plt.close()

#%% validate streaming models
train_day = 2800
test_dates = LD.ix[LD.issue_day < train_day,'issue_day']
n_samps = 50
test_pts = np.percentile(test_dates,np.linspace(100./n_samps,100-100./n_samps,n_samps))
test_pts = test_pts[::-1]

max_depth=14
n_trees=100
min_samples_leaf=100

#dont use time for this analysis
use_cols = [col for col in all_pred_cnames if col != 'issue_day']
RF_est = RandomForestRegressor(n_estimators=n_trees, max_depth=10, 
                               min_samples_leaf=min_samples_leaf, n_jobs=4)
RF_mod = Pipeline([('col_sel',LCM.col_selector(use_cols,col_dict)),
                        ('est',RF_est)])
                        
train_set = (LD['issue_day'] < train_day).values
RF_mod.fit(X[train_set], y[train_set])

stream_R2 = np.zeros(len(test_pts),)*np.nan
cont_Nshuff=5
cont_R2 = np.zeros((len(test_pts),cont_Nshuff))
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
        stream_returns[idx,:] = (LCM.pick_K_returns_BTSTRP(weight_y[test_set], 
                                test_pred, pick_K_list,Npymnts[test_set],
                                n_boots=100, sub_marg=False)).squeeze()
                 
stream_returns = LCM.annualize_returns((stream_returns))

#%%
LD['weight_y'] = (LD['mnthly_ROI'] * LD['exp_num_pymnts'])
marg_returns = LD.groupby('issue_d')['weight_y'].mean()
marg_returns = marg_returns / LD.groupby('issue_d')['exp_num_pymnts'].mean()
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
ax2 = ax.twinx()
date_counts = LD.groupby('issue_d')['issue_d'].count()  
ax2.bar(date_counts.index.values,date_counts/1e3,width=10,fc='k',alpha=0.25) 
   
ax.set_ylabel('Annualized returns (%)', color='k',fontsize=14)
ax2.set_ylabel('Loan volume (thousands)', color='gray',fontsize=14)
ax.set_xlabel('Issue date',fontsize=14)
ax.set_xlim(datetime.date(2009,01,01),LD['issue_d'].max())
ax.legend(loc='upper left',fontsize=14)

ymin, ymax= ax.get_ylim()
xmin, xmax = ax.get_xlim()
xmin = datetime.timedelta(days=train_day) + LD['issue_d'].min()
xmax = LD['issue_d'].max()
#ax.axvspan(datetime.timedelta(days=train_day) + LD['issue_d'].min(),
#           xmax, alpha=0.25, color='red')

from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
xmin = mdates.date2num(xmin)
xmax = mdates.date2num(xmax)
ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, alpha=0.25,
                       facecolor=pal[1]))
ax.annotate('Training data', xy=((xmin+xmax)/2-50., ymax-2), xytext=(xmin-700, ymax-3),
            arrowprops=dict(facecolor='black', shrink=0.05))

if plot_figures:
    fig.savefig(fig_dir + 'time_validation.png', dpi=500, format='png')
    plt.close()
