# -*- coding: utf-8 -*-

import os
import sys
base_dir = '/Users/james/Data_Incubator/LC_app'
#base_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(base_dir)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn as sns

import pandas as pd

import LC_helpers as LCH
import LC_loading as LCL
import LC_models as LCM

data_dir = os.path.join(base_dir,'static/data/')
fig_dir = os.path.join(base_dir,'static/images/')
movie_dir = os.path.join(base_dir,'static/movies/')

data_name = 'all_loans_proc'
LD = pd.read_csv(data_dir + data_name)
#LD['issue_d'] = pd.to_datetime(LD['issue_d'],format = '%Y-%m-%d',unit='D')
LD['weighted_ROI']  = 100*LD['weighted_ROI']  # convert to percent
LD['ROI']  = 100*LD['ROI']  # convert to percent

#%%
marg_kws = {'bins':500}
x_jitter = np.random.randn(len(LD),)*jitter_SD
y_jitter = np.random.randn(len(LD),)*jitter_SD

LD['int_rate'] = LD['int_rate'] + x_jitter
LD['ROI'] = LD['ROI'] + y_jitter
g = sns.jointplot(x='int_rate', y='ROI', data=LD.sample(10000), kind='reg', 
               marginal_kws=marg_kws,ratio=4, space=0.2, size=6)             
g.ax_joint.cla()  
plt.sca(g.ax_joint)
jitter_SD = 1

plt.plot(LD['int_rate'],LD['ROI'],'.',ms=6,alpha=0.05)
fit = np.polyfit(LD['int_rate'],LD['weighted_ROI'],1)
fit_fn = np.poly1d(fit) 
xax = np.linspace(5,25,500)
plt.plot(xax,fit_fn(xax),'k',lw=4)

plt.ylim(-100,30)
plt.xlim(0, 30)
plt.axhline(y=0.,color='k',ls='dashed')
plt.plot([0,30],[0,30],'k',ls='dashed')
plt.xlabel('Interest Rate (%)',fontsize=16)
plt.ylabel('Expected ROI (%)',fontsize=16)

plt.tight_layout()
plt.savefig(fig_dir + 'int_rate_scatter.png', dpi=500, format='png')
plt.close()

