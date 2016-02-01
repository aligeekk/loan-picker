# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:44:05 2016

@author: james
"""

import pickle
from LC_helpers import *
save_loc = 'static/data/'

fips_to_zip = make_fips_dict()
data_loc = save_loc + 'fips_to_zip.p'
pickle.dump(fips_to_zip, open(data_loc, "wb" ) )

state_fips_dict = dict((state.abbr,state.fips) for state in us.STATES)
data_loc = save_loc + 'state_fips_dict.p'
pickle.dump(state_fips_dict, open(data_loc, "wb" ) )


