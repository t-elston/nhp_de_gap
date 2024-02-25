"""
NHP_DEgap_main.py
This is the top of the stack for assessing behavior
@author: Thomas Elston
"""
# %% imports
import NHP_DEgap_behavior as bhv
import importlib

# %% reload module
importlib.reload(bhv)

# %% specify data directories
description_dir = 'C:/Users/Thomas Elston\Documents/PYTHON/NHP_DEgap/bhv_data/description_generalization/'
experience_dir = 'C:/Users/Thomas Elston\Documents/PYTHON/NHP_DEgap/bhv_data/experience_generalization/'

# %% load and aggregate data
d_data = bhv.load_and_aggregate_data(description_dir)
e_data = bhv.load_and_aggregate_data(experience_dir)

#%% get / plot choice performance for first X trials in each session
bhv.plot_first_X_trials(d_data, e_data, 400, 20)


# %%
# now look at switch cost in the blocked behavioral experiment
blockDE_dir = 'C:/Users/Thomas Elston/Documents/PYTHON/NHP_DEgap/bhv_data/DE_blocks/'
block_data = bhv.load_and_aggregate_data(blockDE_dir)

bhv.check_switch_cost(block_data)

# %%
# now look at probability distortion
bhv.check_prob_distortion(block_data)

# %%
