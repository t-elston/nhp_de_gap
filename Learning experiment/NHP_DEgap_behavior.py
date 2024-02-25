"""
Module for assessing behavior of NHPs in a DE gap task
@author: Thomas Elston
"""
# ---------------------------------------------------------------
# imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
import pingouin as pg
import utils as ut
# ---------------------------------------------------------------

def load_and_aggregate_data(datadir):
    '''
    This function loads the .csv files specified in datadir and aggregates them into a single dataframe
    '''

    # get files names with path
    fnames = [os.path.join(datadir, _) for _ in os.listdir(datadir) if _.endswith('.csv')]

    # initialize a dataframe for ALL of the data
    alldata = pd.DataFrame()

    for f in fnames:
        f_data = pd.read_csv(f, header=[0])

        alldata = pd.concat([alldata, f_data])

    alldata = alldata.reset_index(drop=True)
    return alldata


# -------------------------------------------------------------------

def plot_first_X_trials(d_data, e_data, n_trials, win_size):
    '''
    This function assesses choice performance in the first n_trials of each session. 
    Choice performance is assess via a moving average of win_size
    '''

    # get file names
    d_files = np.unique(d_data['fname'])
    e_files = np.unique(e_data['fname'])

    # initialize output
    d_learning = np.empty((len(d_files), n_trials))
    e_learning = np.empty((len(e_files), n_trials))
    d_learning[:] = np.nan
    e_learning[:] = np.nan

    for d_ix, d_file in enumerate(d_files):
        f_data = d_data.loc[(d_data['fname'] == d_file)]

        d_learning[d_ix, :] = ut.moving_average(f_data['picked_best'].values, win_size)[0:n_trials]

    for e_ix, e_file in enumerate(e_files):
        f_data = e_data.loc[(e_data['fname'] == e_file)]

        e_learning[e_ix, :] = ut.moving_average(f_data['picked_best'].values, win_size)[0:n_trials]

    d_mean = np.nanmean(d_learning, axis=0)
    d_sem = np.nanstd(d_learning, axis=0) / len(d_files)

    e_mean = np.nanmean(e_learning, axis=0)
    e_sem = np.nanstd(e_learning, axis=0) / len(e_files)

    t_nums = np.arange(n_trials) + 1
    p_vals = np.zeros(shape=(len(t_nums)), )

    # do unpaired t-tests at each trial
    for t in range(len(t_nums)):
        p_vals[t] = pg.ttest(d_learning[:, t], e_learning[:, t])['p-val'].values[0]

    # find the significant p_vals
    sig_trials = p_vals <= .05

    # now plot
    fig = plt.figure(dpi=150)
    plt.plot(t_nums, d_mean, color='tab:blue', label='Description')
    plt.fill_between(t_nums, d_mean - d_sem, d_mean + d_sem, color='tab:blue', alpha=.4)
    plt.plot(t_nums, e_mean, color='red', label='Experience')
    plt.fill_between(t_nums, e_mean - e_sem, e_mean + e_sem, color='tab:red', alpha=.4)
    plt.scatter(t_nums[sig_trials], np.ones_like(t_nums[sig_trials]) * .1, color='black', marker='*',
                label='p < .05')
    plt.ylim((0, 1))
    plt.legend(loc='best')
    plt.xlabel('Trial in Session')
    plt.ylabel('p(Correct)')
    plt.title('Generalizing to Novel Stimuli')
    plt.show()
    print('done plotting')


# -------------------------------------------------------------------


def check_switch_cost(block_data):
    """
    this function looks at reaction times relative to block changes
    notes on variables in block_data:
    block_type = 1 for description
    block_type = 2 for experience
    """

    # find the switch trials
    switch_trials = np.where((block_data['block_tnum'] == 1) & (block_data['tnum']>1))[0]

    # define trials relative to switch to assess 
    t_rel2_switch = np.arange(12)-1

    # initialize output
    rt_data = pd.DataFrame()
    choice_data = pd.DataFrame()

    alignment_indices = np.empty(shape = (len(switch_trials), len(t_rel2_switch)) )
    alignment_indices[:] = np.nan
    block_types = np.empty(shape = (len(switch_trials),1) )
    
    # loop over switches and get the indices of relevant trials and their types
    for s_ix, switch_trial in enumerate(switch_trials):

        alignment_indices[s_ix,:] = switch_trial + t_rel2_switch
        block_types[s_ix,0] = block_data['block_type'].iloc[switch_trial]

    # now loop over the trials relative to the switch
    for t_ix in range(len(t_rel2_switch)):

        des_trials = alignment_indices[(block_types==1).flatten(), t_ix]
        exp_trials = alignment_indices[(block_types==2).flatten(), t_ix]

        des_trials = np.delete(des_trials, des_trials >= len(block_data))
        exp_trials = np.delete(exp_trials, exp_trials >= len(block_data))

        rt_data.at[t_ix, 't_num'] = t_rel2_switch[t_ix]
        rt_data.at[t_ix, 'd_mean'] = block_data['rt'].iloc[des_trials].mean()
        rt_data.at[t_ix, 'd_sem'] = block_data['rt'].iloc[des_trials].sem()
        rt_data.at[t_ix, 'e_mean'] = block_data['rt'].iloc[exp_trials].mean()
        rt_data.at[t_ix, 'e_sem'] = block_data['rt'].iloc[exp_trials].sem()

        rt_data.at[t_ix, 'p_val'] = pg.ttest(block_data['rt'].iloc[des_trials],
                                             block_data['rt'].iloc[exp_trials])['p-val'].values[0]

        choice_data.at[t_ix, 't_num'] = t_rel2_switch[t_ix]
        choice_data.at[t_ix, 'd_mean'] = block_data['picked_best'].iloc[des_trials].mean()
        choice_data.at[t_ix, 'd_sem'] = block_data['picked_best'].iloc[des_trials].sem()
        choice_data.at[t_ix, 'e_mean'] = block_data['picked_best'].iloc[exp_trials].mean()
        choice_data.at[t_ix, 'e_sem'] = block_data['picked_best'].iloc[exp_trials].sem()

        choice_data.at[t_ix, 'p_val'] = pg.ttest(block_data['picked_best'].iloc[des_trials],
                                block_data['picked_best'].iloc[exp_trials])['p-val'].values[0]


    x_vals = rt_data['t_num']

    fig, ax = plt.subplots(2, 1, dpi=150)
    fig.tight_layout(pad=2)

    ax[0].errorbar(x_vals, choice_data['d_mean'], choice_data['d_sem'], color='tab:blue', marker='s',
                   label='Switch to Description')
    ax[0].errorbar(x_vals, choice_data['e_mean'], choice_data['e_sem'], color='tab:red', marker='s',
                   label='Switch to Experience')
    ax[0].scatter(x_vals[choice_data['p_val'] < .05], np.ones_like(x_vals[choice_data['p_val'] < .05]) * .1,
                  color='black', marker='*', label='p < .05')
    ax[0].legend(loc = 'lower right')
    ax[0].set_ylim((0, 1))
    ax[0].set_xticks(x_vals)
    ax[0].set_ylabel('p(Correct)')

    ax[1].errorbar(x_vals, rt_data['d_mean'], rt_data['d_sem'], color='tab:blue', marker='s')
    ax[1].errorbar(x_vals, rt_data['e_mean'], rt_data['e_sem'], color='tab:red', marker='s')
    ax[1].scatter(x_vals[rt_data['p_val'] < .05], np.ones_like(x_vals[rt_data['p_val'] < .05]) * 250,
                  color='black', marker='*', label='p < .05')
    ax[1].set_ylabel('Saccade RT (ms)')
    ax[1].set_xticks(x_vals)
    ax[1].set_xlabel('Trials from Switch')
    plt.show()

def check_switch_cost2(block_data):
    """
    this function looks at reaction times relative to block changes
    notes on variables in block_data:
    block_type = 1 for description
    block_type = 2 for experience
    """

    # find the switch trials
    switch_trials = (block_data['block_tnum'] == 1) & (block_data['tnum']>1)
    repetition_trials = (block_data['block_tnum'] == 2) & (block_data['tnum']>1)
    d_blocks = block_data['block_type'] == 1
    e_blocks = block_data['block_type'] == 2


    d_switch_mean = block_data['picked_best'].loc[switch_trials & d_blocks].mean()
    d_switch_sem = block_data['picked_best'].loc[switch_trials & d_blocks].sem()

    d_rep_mean = block_data['picked_best'].loc[repetition_trials & d_blocks].mean()
    d_rep_sem = block_data['picked_best'].loc[repetition_trials & d_blocks].sem()


    e_switch_mean = block_data['picked_best'].loc[switch_trials & e_blocks].mean()
    e_switch_sem = block_data['picked_best'].loc[switch_trials & e_blocks].sem()

    e_rep_mean = block_data['picked_best'].loc[repetition_trials & e_blocks].mean()
    e_rep_sem = block_data['picked_best'].loc[repetition_trials & e_blocks].sem()


    description_ttest = pg.ttest(block_data['picked_best'].loc[switch_trials & d_blocks], 
                                 block_data['picked_best'].loc[repetition_trials & d_blocks])
    experience_ttest = pg.ttest(block_data['picked_best'].loc[switch_trials & e_blocks], 
                                block_data['picked_best'].loc[repetition_trials & e_blocks])

    fig = plt.figure(dpi=150, figsize=(3,4))
    plt.errorbar([0,1], [d_switch_mean, d_rep_mean],[d_switch_sem, d_rep_sem],
        color = 'tab:blue', label = 'Description', marker='s')
    plt.errorbar([0,1], [e_switch_mean, e_rep_mean],[e_switch_sem, e_rep_sem],
        color = 'tab:red', label = 'Experience', marker='s')
    plt.xticks([0,1], ['Switch','Repeat'])
    plt.ylim((0,1))
    plt.ylabel('p(Correct)')
    plt.xlabel('Choice Condition')
    plt.legend()


    
    



def check_prob_distortion(block_data):
    """Computes the likelihood of the monkey picking a given probability when it's present in a trial

    Args:
        block_data (dataframe): each trial's features
    """

    probs = np.unique(block_data['left_prob'])

    d_ix = block_data['block_type'] == 1
    e_ix = block_data['block_type'] == 2

    # initialize output
    prob_distortion = pd.DataFrame()

    picked_p=[]
    block_type=[]
    p_level=[]

    # loop over each probability
    for p_ix, p in enumerate(probs):
        prob_in_trial = (block_data['left_prob'] == p) | (block_data['right_prob'] == p)

        prob_distortion.at[p_ix,'prob'] = p
        prob_distortion.at[p_ix,'p_optimal'] = (block_data['best_prob'].loc[prob_in_trial] == p).mean()*100
        prob_distortion.at[p_ix,'picked_d'] = (block_data['chosen_prob'].loc[prob_in_trial & d_ix] == p).mean()*100
        prob_distortion.at[p_ix,'picked_e'] = (block_data['chosen_prob'].loc[prob_in_trial & e_ix] == p).mean()*100
        prob_distortion.at[p_ix, 'd_p_val'] = pg.ttest((block_data['chosen_prob'].loc[prob_in_trial & d_ix] == p),
                                                       (block_data['best_prob'].loc[prob_in_trial] == p)) \
                                                       ['p-val'].values[0]
        prob_distortion.at[p_ix, 'e_p_val'] = pg.ttest((block_data['chosen_prob'].loc[prob_in_trial & e_ix] == p),
                                                       (block_data['best_prob'].loc[prob_in_trial] == p)) \
                                                       ['p-val'].values[0]

        # now build up an array for the regression
        picked_p.extend((block_data['chosen_prob'].loc[prob_in_trial] == p).values)
        block_type.extend((block_data['block_type'].loc[prob_in_trial]).values)
        p_level.extend(np.ones(shape = (np.sum(prob_in_trial), 1) )*p)
                                                      
    reg_df = pd.DataFrame()

    reg_df['picked_p'] = np.array(picked_p).astype(int)
    reg_df['block_type'] = np.array(block_type)
    reg_df['p_level'] = np.array(p_level)

    reg_df['block_type'].loc[reg_df['block_type']==2] = -1

    log_reg = smf.logit("picked_p ~ block_type*p_level", data=reg_df).fit()
    print(log_reg.summary2())

    fig = plt.figure(dpi=150)
    plt.plot(prob_distortion['prob'], prob_distortion['p_optimal'], color='tab:gray',
             linestyle = '--', label = 'Optimal')
    plt.plot([0,100], [0,100], color='tab:green', linestyle = '-', label = 'Unity')
    plt.plot(prob_distortion['prob'], prob_distortion['picked_d'],
             color='tab:blue', marker ='o', label = 'Description')
    plt.plot(prob_distortion['prob'], prob_distortion['picked_e'], 
             color='tab:red', marker ='o', label = 'Experience')
    plt.ylabel('p(Choose P)')
    plt.xlabel('Probability')
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.legend()


 

