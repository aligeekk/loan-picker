
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn import tree  # decision tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import ShuffleSplit, KFold
from sklearn.pipeline import Pipeline, FeatureUnion

from collections import defaultdict

import pickle
from scipy.spatial import KDTree  # for finding KNN
import scipy 

base_dir = '/Users/james/Data_Incubator/LC_app'
#base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)
import LC_helpers as LCH
import LC_loading as LCL
import LC_models as LCM

#set paths
data_dir = os.path.join(base_dir,'static/data/')
fig_dir = os.path.join(base_dir,'static/images/')
movie_dir = os.path.join(base_dir,'static/movies/')

#load data 
data_name = 'all_loans_proc'
LD = pd.read_csv(data_dir + data_name, parse_dates=['issue_d',])

#load long/lat data for each zip-code
zip3_data = LCL.load_location_data(group_by='zip3')        
LD = pd.merge(LD,zip3_data,how='inner', left_on='zip3', right_index=True)
y = LD['ROI']  # response variable
npymnts = LD['num_pymnts']
LD['weight_y'] = LD['mnthly_ROI'] * LD['num_pymnts']
weight_y = LD['weight_y']
    
#%%
pal = sns.color_palette("muted")

fig,ax = plt.subplots(1,1, figsize=(5.0,4.0))
sns.distplot(LD['ROI'], bins=400, kde_kws={"color": pal[2], "lw": 3, "label": "KDE"},
             hist_kws={"alpha": 1, "color":pal[0]})
plt.xlim(-100,30)
plt.xlabel('Annualized return (%)',fontsize=16)
plt.ylabel('Relative frequency',fontsize=16)
plt.tight_layout()

#plt.savefig(fig_dir + 'ROI_dist.png', dpi=500, format='png')
#plt.close()

#%% Fit set of basic lookup-table models, and compute train and test R2   
model_set = [('zip_avg', LCM.zip_avg_mod()),
             ('state_avg', LCM.state_avg_mod())]

n_folds = 10
kf = KFold(len(LD), n_folds=n_folds, shuffle=True, random_state=0)
   
train_R2 = defaultdict(list)
test_R2 = defaultdict(list)
cnt = 0
for train, test in kf:
    print('CV split {} of {}'.format(cnt,n_folds))
    for name, model in model_set:
        model.fit(LD.iloc[train], y.iloc[train])
        train_R2[name].append(model.score(LD.iloc[train],y.iloc[train]))
        test_R2[name].append(model.score(LD.iloc[test],y.iloc[test]))
    cnt += 1
    
#%% Fit decision tree and random forest models with variable tree depth
max_depth = 20
min_samples_leaf = 100
depth_spacing = 2
depth_ax = np.arange(depth_spacing,max_depth+depth_spacing,depth_spacing)

DT_train_R2 = np.zeros((n_folds, len(depth_ax)))
DT_test_R2 = np.zeros((n_folds, len(depth_ax)))
cnt = 0
for train, test in kf:
    print('CV split {} of {}'.format(cnt,n_folds))
    for idx, depth in enumerate(depth_ax):
        DT_mod = tree.DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_samples_leaf)
        DT_mod = Pipeline([('col_sel',LCM.col_selector(['longitude','latitude'],col_dict)),
                           ('est',DT_mod)])
        DT_mod.fit(X[train,:], y.iloc[train])
        DT_train_R2[cnt,idx] = DT_mod.score(X[train,:],y.iloc[train])
        DT_test_R2[cnt,idx] = DT_mod.score(X[test,:],y.iloc[test])
    cnt += 1
 
forest_sizes = [10, 100]  # list of forest sizes to test
RF_train_R2 = np.zeros((n_folds, len(depth_ax), len(forest_sizes)))
RF_test_R2 = np.zeros((n_folds, len(depth_ax), len(forest_sizes)))
cnt = 0
for train, test in kf:
    print('CV split {} of {}'.format(cnt,n_folds))
    for idx,depth in enumerate(depth_ax):
        for fidx,forest_size in enumerate(forest_sizes):
            RF_mod = RandomForestRegressor(n_estimators=forest_size, n_jobs=4,
                                           max_depth=depth, min_samples_leaf=min_samples_leaf)
            RF_mod = Pipeline([('col_sel',LCM.col_selector(['longitude','latitude'],col_dict)),
                               ('est',RF_mod)])
            RF_mod.fit(X[train,:], y.iloc[train])
            RF_train_R2[cnt,idx,fidx] = RF_mod.score(X[train,:],y.iloc[train])
            RF_test_R2[cnt,idx,fidx] = RF_mod.score(X[test,:],y.iloc[test])
    cnt += 1

#%% Make plot of in- and out-of-sample predict accuracy across models
pal = sns.cubehelix_palette(len(forest_sizes)+1)
fig = plt.figure(figsize=(10.0,4.0))

ax=plt.subplot(1,2,1) #plot for train error
y_base = np.zeros_like(depth_ax)  # dummy array of zeros

#make dummy plots for legend
plt.plot(np.nan,np.nan,lw=5,color=pal[0],label='Single Tree')
for idx in xrange(len(forest_sizes)):
    plt.plot(np.nan,np.nan,lw=5,color=pal[idx+1],label='{}-Tree Forest'.format(forest_sizes[idx]))
plt.axhline(y=np.mean(train_R2['state_avg']) * 1e3,color='g',label='State Avgs')
plt.axhline(y=np.mean(train_R2['zip_avg']) * 1e3,color='b',label='Zip3 Avgs')
plt.legend(loc='center right',fontsize=12)
#make shaded error regions for zip- and state-avg models
plt.fill_between(depth_ax, 1e3*(y_base + np.mean(train_R2['state_avg']) - np.std(train_R2['state_avg'])/np.sqrt(n_folds)),
                 1e3*(y_base + np.mean(train_R2['state_avg']) + np.std(train_R2['state_avg'])/np.sqrt(n_folds)),
                 color='g',alpha=0.25)
plt.fill_between(depth_ax, 1e3*(y_base + np.mean(train_R2['zip_avg']) - np.std(train_R2['zip_avg'])/np.sqrt(n_folds)),
                 1e3*(y_base + np.mean(train_R2['zip_avg']) + np.std(train_R2['zip_avg'])/np.sqrt(n_folds)),
                 color='b',alpha=0.25)

#plot train R2 for tree models
sns.tsplot(time=depth_ax, data=DT_train_R2 * 1e3, color=pal[0], ax=ax,lw=4)
for idx in xrange(len(forest_sizes)):
    sns.tsplot(time=depth_ax, data=RF_train_R2[:,:,idx] * 1e3, color=pal[idx+1],
               ax=ax, lw=4)

plt.xlim(2,18)
plt.ylabel('In-sample $R^2$ (x $10^{-3}$)',fontsize=16)
plt.xlabel('Tree depth',fontsize=16)


ax=plt.subplot(1,2,2) #plot for test error
plt.axhline(y=np.mean(test_R2['state_avg']) * 1e3,color='g',label='state avgs')
plt.axhline(y=np.mean(test_R2['zip_avg']) * 1e3,color='b',label='zip3 avgs')
plt.fill_between(depth_ax, 1e3*(y_base + np.mean(test_R2['state_avg']) - np.std(test_R2['state_avg'])/np.sqrt(n_folds)),
                 1e3*(y_base + np.mean(test_R2['state_avg']) + np.std(test_R2['state_avg'])/np.sqrt(n_folds)),
                color='g',alpha=0.25)
plt.fill_between(depth_ax, 1e3*(y_base + np.mean(test_R2['zip_avg']) - np.std(test_R2['zip_avg'])/np.sqrt(n_folds)),
                 1e3*(y_base + np.mean(test_R2['zip_avg']) + np.std(test_R2['zip_avg'])/np.sqrt(n_folds)),
                color='b',alpha=0.25)
sns.tsplot(time=depth_ax, data=DT_test_R2 * 1e3, color=pal[0], ax=ax, lw=4)
for idx in xrange(len(forest_sizes)):
    sns.tsplot(time=depth_ax, data=RF_test_R2[:,:,idx] * 1e3, color=pal[idx+1], 
               ax=ax, lw=4)

plt.xlim(2,18)
plt.xlabel('Tree depth',fontsize=16)
plt.ylabel('Out-of-sample $R^2$ (x $10^{-3}$)',fontsize=16)
plt.axhline(y=0,ls='dashed',color='k')
plt.tight_layout()

#plt.savefig(fig_dir + 'R2_vs_treedepth.png', dpi=500, format='png')
#plt.close()

