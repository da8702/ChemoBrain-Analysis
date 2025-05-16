# %%
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def proc_foraging_df(behav_df, subj_name, condition, session):
    n_trials = 0
    n_epochs = -1
    switched = 0.0
    n_rewards = -1

    trial_number = []
    reward_number = []
    epoch_number = []
    starting_rsize = []
    reward_amount = []
    probe_poke= ["first"]
    left = []

    if ~np.isnan(behav_df.iloc[0]['LeftAmount']) and ~np.isnan(behav_df.iloc[1]['LeftAmount']):
        current_side = 'left'
        rsize = behav_df.iloc[0]['LeftAmount']
    elif ~np.isnan(behav_df.iloc[0]['RightAmount']) and ~np.isnan(behav_df.iloc[1]['RightAmount']):
        current_side = 'right'
        rsize = behav_df.iloc[0]['RightAmount']
    elif ~np.isnan(behav_df.iloc[0]['LeftAmount']) and np.isnan(behav_df.iloc[1]['LeftAmount']):
        current_side = 'left'
        rsize = behav_df.iloc[0]['LeftAmount']
        switched = 1.0
    elif ~np.isnan(behav_df.iloc[0]['RightAmount']) and np.isnan(behav_df.iloc[1]['RightAmount']):
        current_side = 'right'
        rsize = behav_df.iloc[0]['RightAmount']
        switched = 1.0

    for ind, row in behav_df.iterrows():
        left_amt = row['LeftAmount']
        right_amt = row['RightAmount']
        n_epochs += 1
        n_rewards +=1

        if np.isnan(left_amt) and (current_side == 'left'):
            assert (np.isnan(right_amt) == False)
            rsize = right_amt
            current_side = 'right'
            switched = 1.0
            n_trials += 1
            n_epochs = 0

        elif np.isnan(right_amt) and (current_side == 'right'):
            assert (np.isnan(left_amt) == False)
            rsize = left_amt
            current_side = 'left'

            switched = 1.0
            n_trials += 1
            n_epochs = 0
        else:
            switched = 0.0

        if current_side == 'left':
            assert(~np.isnan(left_amt))

            if (len(reward_amount) > 0) & (probe_poke[-1] == 0):
                if (switched == 0) & (left_amt > (reward_amount[-1] + .5)):
                    probe_poke.append(1)
                elif (switched == 0) & (left_amt < (.8*reward_amount[-1] - .5)):
                    probe_poke.append(-2)
                else:
                    probe_poke.append(0)
            else:
                probe_poke.append(0)

            reward_amount.append(left_amt)

        elif current_side == 'right':
            assert(~np.isnan(right_amt))

            if (len(reward_amount) > 0)  & (probe_poke[-1] == 0):
                if (switched == 0) & (right_amt > (reward_amount[-1])+.5):
                    probe_poke.append(1)
                elif (switched == 0) & (right_amt < (.8*reward_amount[-1] - .5)):
                    probe_poke.append(-2)
                else:
                    probe_poke.append(0)
            else:
                probe_poke.append(0)

            reward_amount.append(right_amt)


        left.append(switched)
        reward_number.append(n_rewards)
        starting_rsize.append(rsize)
        trial_number.append(n_trials)
        epoch_number.append(n_epochs)

    assert(len(left) == len(behav_df['LeftAmount']))
    behav_df['reward_number'] = reward_number
    behav_df['name'] = subj_name
    behav_df['condition'] = condition
    behav_df['session_number'] = session
    behav_df['switch'] = left
    behav_df['switch_prob'] = behav_df['switch'].shift(-1)
    behav_df['stay_prob'] = 1 - behav_df['switch_prob']
    behav_df['trial_number'] = trial_number
    behav_df['poke_number'] = epoch_number
    behav_df['trial_type'] = starting_rsize
    behav_df['reward_amount'] = np.around(np.array(reward_amount), decimals=4)
    return behav_df


#%%
name = "W3_IL6_RHRH"
behavfiles = glob.glob("/Volumes/Aelita/foraging data/data/processed/" + name + "/*RecBehav.mat")
# photomfiles = glob.glob("/Users/ameliachristensen/Desktop/processed/" + name + "/*AlignedPhoto.mat")
dfs = []
for i, file in enumerate(behavfiles):
    print(i, file)
    SessionData = loadmat(file,squeeze_me=True )
    n_trials = len(SessionData['LeftAmount'])
    behav_df = pd.DataFrame.from_dict({'LeftAmount': SessionData['LeftAmount'], 'RightAmount': SessionData['RightAmount'],
                    'name': [name]*n_trials, 'session': [i]*n_trials})
    df = proc_foraging_df(behav_df.reset_index(), subj_name=name,
                         condition="IL6", session=i)
    dfs.append(df)
# %%
    

behav_df = pd.concat(dfs)
# %%


sns.pointplot(data = behav_df[behav_df.switch_prob == 1], x = "session", y = "reward_amount", hue = "trial_type")
# %%

behav_df.to_pickle("/Users/ameliachristensen/Desktop/processed/" + name + "_behav_df.pickle")

# %%
