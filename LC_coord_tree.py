# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:44:59 2016

@author: james
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn as sns
from sklearn import tree  # decision tree
from sklearn.ensemble import RandomForestRegressor  
import pandas as pd
from bs4 import BeautifulSoup

import LC_helpers as LCH
import LC_loading as LCL

import pickle
import cairosvg
from scipy.spatial import KDTree  # for finding KNN
import scipy 
import os

base_dir = '/Users/james/Data_Incubator/LC_app'
#base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir,'static/data/')
fig_dir = os.path.join(base_dir,'static/images/')
movie_dir = os.path.join(base_dir,'static/movies/')

data_name = 'all_loans_proc'
LD = pd.read_csv(data_dir + data_name)
#LD['issue_d'] = pd.to_datetime(LD['issue_d'],format = '%Y-%m-%d',unit='D')

fips_data = LCL.load_location_data(group_by='fips')
zip3_data = LCL.load_location_data(group_by='zip3')
fips_to_zip = LCL.make_fips_dict(group_by='zip')
        
#%%
LD = pd.merge(LD,zip3_data,how='inner', left_on='zip3', right_index=True)

#%%
base_map = LCL.load_base_map(fig_dir + 'USA_Counties_text.svg', ax_xml=True)
(county_paths,state_paths) = LCL.get_map_paths(base_map,fips_to_zip)
title_path = base_map.findAll('text')[0]
map_coords = LCH.extract_fips_coords(county_paths)
ktree = KDTree(map_coords.values) #make nearest neighbor tree

#%%
X = LD[['longitude','latitude']]
y = LD['ROI'] #plot average return by area, not portfolio return

#%% make sequence of decision trees and build a movie
max_levels = 16
min_samples_leaf = 50
pred_arr = np.zeros((len(fips_data),max_levels))
for n in xrange(max_levels):
    clf = tree.DecisionTreeRegressor(max_depth=n+1, min_samples_leaf=min_samples_leaf, random_state=0)
    clf.fit(X, y)
    pred_arr[:,n] = clf.predict(fips_data[['longitude','latitude']].values)
    
#%%
image_list_path = os.path.join(movie_dir,'image_list.txt')
F = open(image_list_path,'wb+')
pred_prc = scipy.percentile(pred_arr,[10, 90]) 
prc_range = pred_prc[1]-pred_prc[0]  # overall range of data

# MAKE COLORBAR FIGURE WITH FIXED RANGE
temp = pred_arr[:,-1].copy()
cur_range = scipy.percentile(temp,[10,90])  # range for this map
temp  = (temp - cur_range[0])*prc_range/np.diff(cur_range) + pred_prc[0]  # scale to have same range
fips_data['preds'] = temp
new_data = LCH.fill_missing_fips(ktree,map_coords,fips_data)
cbar_fig = LCH.paint_map(new_data['preds'], base_map, county_paths, fips_to_zip, color = 'cube', get_cbar=True)  
plt.savefig(movie_dir + 'tree_cbar.png', dpi=500, format='png')
plt.close()

for n in xrange(max_levels):
    cur_range = scipy.percentile(pred_arr[:,n],[10,90])  # range for this map
    pred_arr[:,n]  = (pred_arr[:,n] - cur_range[0])*prc_range/np.diff(cur_range)  # scale to have same range
    fips_data['preds'] = pred_arr[:,n]
    new_data = LCH.fill_missing_fips(ktree,map_coords,fips_data)
    LCH.paint_map(new_data['preds'], base_map, county_paths, fips_to_zip, color = 'cube', get_cbar=False)        
    title_path.string = 'Depth {} Tree'.format(n)
    bytestring = base_map.encode()
    out_file = os.path.join(movie_dir,'tree_depth%d.png' % n)
    cairosvg.svg2png(bytestring=bytestring, write_to = out_file)
    F.write(out_file + '\n')

F.close()
#%%
import subprocess
outfile = os.path.join(movie_dir,'tree_movie.mp4')
subprocess.call(['convert -delay 26 @{} -quality 100% -fill white -opaque none -alpha off {}'.format(image_list_path,outfile)],
                 shell=True)
#

