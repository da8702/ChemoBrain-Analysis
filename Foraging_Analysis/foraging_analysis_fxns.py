# Import required libraries
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import numpy as np
import datetime
from scipy.ndimage import gaussian_filter1d
import os

# Set plotting parameters
plt.rcParams['font.size'] = '16'
plt.rcParams['savefig.dpi'] = 80
plt.rcParams['figure.dpi'] = 80
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
pd.set_option('display.max_rows', 30)
pd.set_option('display.min_rows', 30)

def proc_foraging_df(behav_df, subj_name, condition, session):
    """Process foraging behavior data to add useful metrics for plotting."""
    n_trials = 0
    n_epochs = -1
    switched = 0.0
    n_rewards = -1

    trial_number = []
    reward_number = []
    epoch_number = []
    starting_rsize = []
    reward_amount = []
    probe_poke = ["first"]
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
        n_rewards += 1

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

            if (len(reward_amount) > 0) & (probe_poke[-1] == 0):
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
    """Extract datetime from filename."""
    date_format = "%d-%b-%Y %H:%M:%S"
    date_time_str = file_name.split('/')[-1].split('_')[0]
    return datetime.datetime.strptime(date_time_str, date_format)

# Define subject names
names = ['DA76-TNT-LH', 'DA77-TRT-RH', 'DA78-TNT-LRH', 'DA79-TDT-NH', 
         'DA80-TDT-LH', 'DA81-TDT-RH', 'DA82-TDT-NH', 'DA83-TDT-LRH',
         'DA84-TNT-NH', 'DA85-TNT-LH', 'DA86-TNT-RH']

# Find all behavior and photometry files
behavfiles = []
photomfiles = []

for i in names:
    ind_files = glob.glob(f"/Volumes/ChemoBrain/ChemoBrain-Analysis/Data/Foraging_Cisplatin/Cis3_processed/{i}/*RecBehav.mat")
    photomfiles_ind = glob.glob(f"/Volumes/ChemoBrain/ChemoBrain-Analysis/Data/Foraging_Cisplatin/Cis3_processed/{i}/*AlignedPhoto.mat")
    behavfiles.append(ind_files)
    photomfiles.append(photomfiles_ind)

# Sort files by datetime
for i in range(len(behavfiles)):
    # Sort behavior files
    dated_files = [(extract_datetime(fn), fn) for fn in behavfiles[i]]
    dated_files.sort()
    behavfiles[i] = [fn for dt, fn in dated_files]

    # Sort photometry files
    dated_files_photo = [(extract_datetime(fn), fn) for fn in photomfiles[i]]
    dated_files_photo.sort()
    photomfiles[i] = [fn for dt, fn in dated_files_photo]

# Process each animal's data
df_tot = []
behav_only_all = []

for j in range(len(behavfiles)):
    dfs = []
    behav_only = []
    
    for i, (behav_file, photom_file) in enumerate(zip(behavfiles[j], photomfiles[j])):
        print(f"\nProcessing {names[j]}, session {i}:")
        print(f"Behavior file: {behav_file}")
        print(f"Photometry file: {photom_file}")
        
        # Load behavior data
        SessionData = loadmat(behav_file, squeeze_me=True)
        print(f"SessionData keys: {SessionData.keys()}")
        
        # Load photometry data
        PhotometryData = loadmat(photom_file, squeeze_me=True)
        print(f"PhotometryData keys: {PhotometryData.keys()}")
        
        ra_traces = PhotometryData['aligned_photo']
        print(f"ra_traces shape: {ra_traces.shape}")
        print(f"ra_traces type: {type(ra_traces)}")
        
        # Ensure ra_traces is 2D
        if len(ra_traces.shape) == 1:
            print("Warning: ra_traces is 1D, reshaping...")
            ra_traces = ra_traces.reshape(1, -1)
        
        n_trials, n_frames = ra_traces.shape
        print(f"n_trials: {n_trials}, n_frames: {n_frames}")
        
        # Handle both array and scalar cases for LeftAmount and RightAmount
        if isinstance(SessionData['LeftAmount'], (int, float)):
            left_amount = np.array([SessionData['LeftAmount']] * n_trials)
            right_amount = np.array([SessionData['RightAmount']] * n_trials)
        else:
            left_amount = SessionData['LeftAmount']
            right_amount = SessionData['RightAmount']
            
        print(f"LeftAmount type: {type(left_amount)}, shape: {left_amount.shape if hasattr(left_amount, 'shape') else 'scalar'}")
        print(f"RightAmount type: {type(right_amount)}, shape: {right_amount.shape if hasattr(right_amount, 'shape') else 'scalar'}")
        
        # Ensure data alignment
        assert n_trials == len(left_amount)
        
        # Smooth photometry data
        ra_traces = gaussian_filter1d(ra_traces, axis=1, sigma=1)
        
        # Create behavior dataframe
        behav_df = pd.DataFrame.from_dict({
            'LeftAmount': left_amount,
            'RightAmount': right_amount,
            'name': [names[j]] * n_trials,
            'session': [i] * n_trials
        })
        
        # Process behavior data
        df = proc_foraging_df(behav_df.reset_index(), subj_name=names[j], 
                             condition="Control", session=i)
        
        # Store behavior-only data
        behav_only.append(df)
        
        # Create long-form dataframe for photometry data
        behav_df_long = {key: np.repeat(df[key], n_frames) for key in df}
        tr = ra_traces.reshape([-1])
        behav_df_long['dopamine'] = tr
        behav_df_long['frames'] = np.tile(np.linspace(-2, 3, n_frames), n_trials)
        
        dfs.append(pd.DataFrame(behav_df_long))
    
    df_tot.append(pd.concat(dfs))
    behav_only_all.append(behav_only)

# Plot example data
plt.figure(figsize=(4,4))
g3 = sns.pointplot(data=df_tot[0][df_tot[0].switch_prob==1], 
                   x='session', y='reward_amount', hue='trial_type')

def filter_session(session_data):
    """Filter 60% of the initial data based on session"""
    threshold = int(0.6 * len(session_data))
    return session_data.head(threshold)

# Calculate leaving value, water intake, and total trials
behav_df_only = []
lev_df = []
water_intake = []
trial_tot = []

for i in range(len(df_tot)):
    ind_df = pd.concat(behav_only_all[i])
    behav_df_only.append(ind_df)
    
    print(f"\nColumns in ind_df: {ind_df.columns.tolist()}")
    print(f"First few rows of ind_df:\n{ind_df.head()}")
    
    # Select specific columns by name instead of index
    df_all = ind_df[['session', 'switch_prob', 'trial_type', 'trial_number', 'reward_amount']].reset_index(drop=True)
    
    print(f"\nColumns in df_all: {df_all.columns.tolist()}")
    print(f"First few rows of df_all:\n{df_all.head()}")
    
    # Leaving value calculation
    lev_sel = df_all.query("switch_prob==1").reset_index(drop=True)
    # First filter the data
    lev_sel = lev_sel.groupby('session').apply(filter_session).reset_index(drop=True)
    # Then calculate means
    lev_lm = lev_sel.groupby('session')['reward_amount'].mean().reset_index()
    lev_lm = lev_lm.rename(columns={'reward_amount': names[i]})
    lev_df.append(lev_lm)
    
    # Water intake calculation
    water_ind = df_all.groupby('session').apply(filter_session).reset_index(drop=True)
    water_ind = water_ind.groupby('session')['reward_amount'].sum().reset_index()
    water_ind = water_ind.rename(columns={'reward_amount': names[i]})
    water_intake.append(water_ind)
    
    # Total trial number calculation
    trials = df_all.groupby('session').apply(filter_session).reset_index(drop=True)
    trials = trials.groupby('session')['trial_number'].max().reset_index()
    trials = trials.rename(columns={'trial_number': names[i]})
    trial_tot.append(trials)

# Create group labels and combine data
Group = ['Cisplatin'] * len(names)
Group = pd.Series(Group, name='Group')

# Combine water intake data
water_intake_all = pd.concat(water_intake, axis=1).T.reset_index().rename(columns={'index': 'Animal_ID'})
water_intake_all = pd.concat([Group, water_intake_all], axis=1)

# Combine trial data
trial_tot_all = pd.concat(trial_tot, axis=1).T.reset_index().rename(columns={'index': 'Animal_ID'})
trial_tot_all = pd.concat([Group, trial_tot_all], axis=1)

# Combine leaving value data
lev_df_all = pd.concat(lev_df, axis=1).T.reset_index().rename(columns={'index': 'Animal_ID'})
lev_df_all = pd.concat([Group, lev_df_all], axis=1)

# Convert to long format for plotting
water_intake_all_l = pd.melt(water_intake_all, id_vars=['Group', 'Animal_ID'], 
                             var_name='session', value_name='water_intake')
trial_tot_all_l = pd.melt(trial_tot_all, id_vars=['Group', 'Animal_ID'], 
                          var_name='session', value_name='trials')
lev_df_all_l = pd.melt(lev_df_all, id_vars=['Group', 'Animal_ID'], 
                       var_name='session', value_name='leaving_value')

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Plot water intake
plt.figure(figsize=(10,6))
g4 = sns.lineplot(data=water_intake_all_l, x='session', y='water_intake', 
                  units='Animal_ID', hue='Animal_ID', estimator=None, 
                  linewidth=2.5, dashes=False, markers=True, 
                  palette='husl', style='Animal_ID')
g4.legend(bbox_to_anchor=(1.1, 1.05))
plt.title('Water Intake Over Sessions')
plt.tight_layout()
plt.savefig('plots/water_intake.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot total trials
plt.figure(figsize=(10,6))
g5 = sns.lineplot(data=trial_tot_all_l, x='session', y='trials', 
                  units='Animal_ID', hue='Animal_ID', estimator=None, 
                  linewidth=2.5, dashes=False, markers=True, 
                  palette='husl', style='Animal_ID')
g5.legend(bbox_to_anchor=(1.1, 1.05))
plt.title('Total Trials Over Sessions')
plt.tight_layout()
plt.savefig('plots/total_trials.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot leaving value
plt.figure(figsize=(10,6))
g6 = sns.lineplot(data=lev_df_all_l, x='session', y='leaving_value', 
                  units='Animal_ID', hue='Animal_ID', estimator=None, 
                  linewidth=2.5, dashes=False, markers=True, 
                  palette='husl', style='Animal_ID')
g6.legend(bbox_to_anchor=(1.1, 1.05))
plt.title('Leaving Value Over Sessions')
plt.tight_layout()
plt.savefig('plots/leaving_value.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot photometry data for a specific session and poke number
plt.figure(figsize=(10,6))
plot_df = df_tot[0].groupby('session').apply(filter_session).reset_index(drop=True)
# Ensure 'session' is a column for query
if 'session' not in plot_df.columns:
    plot_df = plot_df.reset_index()
g7 = sns.relplot(data=plot_df.query("session==7 and poke_number==2"), 
                 x="frames", y="dopamine", hue="poke_number", kind="line")
plt.title('Dopamine Response by Poke Number')
plt.tight_layout()
plt.savefig('plots/dopamine_response.png', dpi=300, bbox_inches='tight')
plt.close()

# Group DA data by trials (wide format) for easy heatmap plotting
session_all = []
for i in range(len(df_tot)):
    # Select columns by name instead of index
    ind_DA = df_tot[i][['session', 'trial_number', 'frames', 'dopamine']]
    session_dfs = {}
    for session in ind_DA['session'].unique():
        session_df = ind_DA[ind_DA['session'] == session]
        pivoted_session_df = session_df.groupby(['trial_number', 'frames'])['dopamine'].mean().unstack('frames')
        session_dfs[session] = pivoted_session_df
    session_all.append(session_dfs)

# Plot heatmap for a specific session
plt.figure()
data_plot = session_all[0][7]  # Adjust indices as needed
g1 = sns.heatmap(data_plot, cbar=True, xticklabels=10, yticklabels=10, cmap='bwr')
g1.set_xticklabels([])
x_zero_index = data_plot.columns.get_loc(0)
g1.axvline(x=x_zero_index, color='red', linestyle='--', linewidth=2)
plt.title('Dopamine Heatmap by Trial')
plt.tight_layout()
plt.savefig('plots/dopamine_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plots have been saved to the 'plots' directory:")
print("1. water_intake.png - Shows water intake over sessions for each animal")
print("2. total_trials.png - Shows total number of trials over sessions for each animal")
print("3. leaving_value.png - Shows leaving value over sessions for each animal")
print("4. dopamine_response.png - Shows dopamine response aligned to rewards")
print("5. dopamine_heatmap.png - Shows dopamine heatmap by trial") 