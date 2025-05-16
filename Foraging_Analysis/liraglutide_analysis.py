# %%
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
from scipy.ndimage import gaussian_filter1d

plt.rcParams['font.size'] = '16'

plt.rcParams['savefig.dpi'] = 80
plt.rcParams['figure.dpi'] = 80
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
pd.set_option('display.max_rows', 30)
pd.set_option('display.min_rows', 30)

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

def extract_datetime(file_name):
    date_format = "%d-%b-%Y %H:%M:%S"
    # Extract the date and time part from the file name
    date_time_str = file_name.split('/')[-1].split('_')[0]
    # Parse the date and time string into a datetime object
    return datetime.datetime.strptime(date_time_str, date_format)

#%%
#search for files of specific pattern; * matches zero or more characters 
name = ['PBSNR01-LH', 'PBSNR01_BH', 'PBSNR02-LH', 'PBSNR01_NH', 'PBSNR01-RH','PBSNR02-NH']
behavfiles=[]
photomfiles=[]

for i in name:

    ind_files = glob.glob("/Volumes/Aelita/Glp1r_agonist/processed2/selected_days/" + i + "/*RecBehav.mat")
    photomfiles_ind = glob.glob("/Volumes/Aelita/Glp1r_agonist/processed2/selected_days/" + i + "/*AlignedPhoto.mat")
    behavfiles.append(ind_files)
    photomfiles.append(photomfiles_ind)

#%%
for i in range(len(behavfiles)):

    # Create a list of tuples with (datetime, file_name)
    dated_files = [(extract_datetime(fn), fn) for fn in behavfiles[i]]
    # Sort the list by datetime
    dated_files.sort()
    behavfiles[i] = [fn for dt, fn in dated_files]

    dated_files_photo = [(extract_datetime(fn), fn) for fn in photomfiles[i]]
    dated_files_photo.sort()
    photomfiles[i] = [fn for dt, fn in dated_files_photo]

#%%
# Process each animal's data
df_tot = []
behav_only_all=[]
for j in range(len(behavfiles)):
    dfs = []
    behav_only=[]
    
    for i, (behav_file, photom_file) in enumerate(zip(behavfiles[j], photomfiles[j])):
        # Load behavior data
        SessionData = loadmat(behav_file, squeeze_me=True)
        
        # Load photometry data
        PhotometryData = loadmat(photom_file, squeeze_me=True)
        ra_traces = PhotometryData['aligned_photo']  # reward-aligned dopamine traces
        n_trials, n_frames = ra_traces.shape
        
        # Ensure that the dopamine data matches the behavior data
        assert n_trials == len(SessionData['LeftAmount'])
        
        # Optional: Gaussian filter the dopamine traces to smooth
        ra_traces = gaussian_filter1d(ra_traces, axis=1, sigma=1)
        
        # Create a dataframe with the basic behavioral info
        behav_df = pd.DataFrame.from_dict({
            'LeftAmount': SessionData['LeftAmount'],
            'RightAmount': SessionData['RightAmount'],
            'name': [name[j]] * n_trials,
            'session': [i] * n_trials
        })
        
        # Process the foraging behavior data to add more useful data for plotting
        df = proc_foraging_df(behav_df.reset_index(), subj_name=name[j], condition="Glp1r", session=i)
        
        #keep for analyzing bahavior data only
        behav_only.append(df)
        
        # Extend the dataframe to long form so that the dopamine info can be added
        behav_df_long = {key: np.repeat(df[key], n_frames) for key in df}
        
        # Linearize the traces so they are also n_trials * n_frames
        tr = ra_traces.reshape([-1])
        
        # Add dopamine data to the dataframe
        behav_df_long['dopamine'] = tr
        
        # Add frames column to the dataframe
        #this needs to match the number of seconds in the matlab code! So the below is accurate for a case where you kept two seconds of dopamine data from before the reward and 3 seconds after the reward
        behav_df_long['frames'] = np.tile(np.linspace(-2, 3, n_frames), n_trials)
        
        dfs.append(pd.DataFrame(behav_df_long))
    
    df_tot.append(pd.concat(dfs))
    behav_only_all.append(behav_only)

# # Concatenate all animals' data into one dataframe
# all_dfs = pd.concat(df_tot)

# %%
plt.figure(figsize=(4,4))
# g1=sns.relplot(data=df_tot[2].query("session==6"), x='frames', y='dopamine', hue='session', kind='line', palette='husl')
# g2=sns.relplot(data=df_tot[2].query("session==7"), x='frames', y='dopamine', hue='session', kind='line', palette='husl')
g3=sns.pointplot(data=df_tot[0][df_tot[0].switch_prob==1], x='session', y='reward_amount', hue='trial_type')

#%%
def filter_session(session_data):
    ''' filter 60% of the initial data based on session'''
    threshold = int(0.6 * len(session_data))
    return session_data.head(threshold)

behav_df_only=[] 
lev_df=[]
water_intake=[] 
trial_tot=[]  

for i in range(len(df_tot)):

    ind_df = pd.concat(behav_only_all[i])
    behav_df_only.append(ind_df)
    
    df_all=ind_df.iloc[:,[4,9, 11, 13,14] ].reset_index(drop=True)


    #leaving value calculation
    lev_sel=df_all.query("switch_prob==1").reset_index(drop=True)
    lev_sel=lev_sel.groupby('session').apply(filter_session).reset_index(drop=True) #filter only first 60% sessions
    lev_lm=lev_sel.groupby(['session']).mean().reset_index().drop(columns=['session', 'switch_prob','trial_type', 'trial_number'])
    lev_lm=lev_lm.rename(columns={'reward_amount': name[i]})

    lev_df.append(lev_lm)
    
    #water intake calculation
    water_ind=df_all.groupby('session').apply(filter_session).reset_index(drop=True)
    water_ind=water_ind.groupby("session").sum().reset_index().drop(columns=['session', 'switch_prob','trial_type','trial_number'])
    water_ind=water_ind.rename(columns={'reward_amount': name[i]})
    water_intake.append(water_ind)
    
    
    #tot trial number calculation
    trials=df_all.groupby('session').apply(filter_session).reset_index(drop=True)
    trials=trials.groupby("session").max().reset_index().drop(columns=['session', 'switch_prob','trial_type','reward_amount'])
    trials=trials.rename(columns={'trial_number': name[i]})
    trial_tot.append(trials)
    
#%%
Group=['photometry','photometry', 'photometry', 'behavior', 'behavior','behavior']
Group=pd.Series(Group, name='Group')
water_intake_all=pd.concat(water_intake, axis=1).T.reset_index().rename(columns={'index': 'Animal_ID'})

water_intake_all=pd.concat([Group, water_intake_all], axis=1)
trial_tot_all=pd.concat(trial_tot, axis=1).T.reset_index().rename(columns={'index': 'Animal_ID'})
trial_tot_all=pd.concat([Group, trial_tot_all], axis=1)

lev_df_all=pd.concat(lev_df, axis=1).T.reset_index().rename(columns={'index': 'Animal_ID'})
lev_df_all=pd.concat([Group, lev_df_all], axis=1)
    
water_intake_all_l=pd.melt(water_intake_all, id_vars=['Group', 'Animal_ID'], var_name='session', value_name='water_intake')
trial_tot_all_l=pd.melt(trial_tot_all, id_vars=['Group', 'Animal_ID'], var_name='session', value_name='trials')
lev_df_all_l=pd.melt(lev_df_all, id_vars=['Group', 'Animal_ID'], var_name='session', value_name='leaving_value')

#%%
plt.figure(figsize=(4,4))
g4=sns.lineplot(data=water_intake_all_l, x='session', y='water_intake', units='Animal_ID',hue='Animal_ID',  estimator=None, linewidth=2.5, dashes=False,\
                markers=True, palette='husl', style='Animal_ID')
g4.legend(bbox_to_anchor=(1.1, 1.05))

plt.figure(figsize=(4,4))
g5=sns.lineplot(data=trial_tot_all_l, x='session', y='trials', units='Animal_ID',hue='Animal_ID',  estimator=None, linewidth=2.5, dashes=False,\
                markers=True, palette='husl', style='Animal_ID')
g5.legend(bbox_to_anchor=(1.1, 1.05))

plt.figure(figsize=(4,4))
g6=sns.lineplot(data=lev_df_all_l, x='session', y='leaving_value', units='Animal_ID',hue='Animal_ID',  estimator=None, linewidth=2.5, dashes=False,\
                markers=True, palette='husl', style='Animal_ID')
g6.legend(bbox_to_anchor=(1.1, 1.05))

#%%
plt.figure(figsize=(4,4))
plot_df=df_tot[2].groupby('session').apply(filter_session).reset_index(drop=True)
g7=sns.relplot(data = plot_df.query("session==3 and poke_number==2") , x = "frames", y = "dopamine", hue = "poke_number",  kind = "line")


# %%
# group by DA data by trials (wide format) so that can be plotted easilly as heatmap
session_all=[]
for i in range(len(df_tot)):

    ind_DA=df_tot[i].iloc[:, [4, 11, 15, 16]]
    # Group by session_number and pivot
    session_dfs = {}
    for session in ind_DA['session'].unique():
        session_df = ind_DA[ind_DA['session'] == session]
        pivoted_session_df = session_df.groupby(['trial_number', 'frames'])['dopamine'].mean().unstack('frames')
        session_dfs[session] = pivoted_session_df
    session_all.append(session_dfs)
# %%
plt.figure()
data_plot=session_all[2][3]
g1=sns.heatmap(data_plot, cbar=True, xticklabels=10, yticklabels=10, cmap='bwr')
g1.set_xticklabels([])
x_zero_index = data_plot.columns.get_loc(0)
g1.axvline(x=x_zero_index, color='red', linestyle='--', linewidth=2)
# %%
