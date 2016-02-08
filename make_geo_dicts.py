# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:44:05 2016

@author: James McFarland
Just builds lookup tables for converting FIPS to zip-codes 
and state to fips and dumps them in pickle files
"""
import os
import sys
import dill
import us

base_dir = os.path.dirname(os.path.realpath(__file__))    
data_dir = os.path.join(base_dir,'static/data/')
sys.path.append(base_dir)

from LC_loading import *

fips_to_zip = make_fips_to_zip_dict(data_dir)
data_loc = data_dir + 'fips_to_zip.p'
dill.dump(fips_to_zip, open(data_loc, "wb" ) )

state_fips_dict = dict((state.abbr,state.fips) for state in us.STATES)
data_loc = data_dir + 'state_fips_dict.p'
dill.dump(state_fips_dict, open(data_loc, "wb" ) )

zip3_loc_data = load_location_data(data_dir,group_by='zip3')
data_loc = data_dir + 'zip3_loc_data.p'
dill.dump(state_fips_dict, open(data_loc, "wb" ) )



