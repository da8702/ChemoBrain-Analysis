#%%
#sample code to process the behavior data and photometry data from the foraging task
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat
import pandas as pd
import numpy as np

def proc_foraging_df(df, subj_name, condition, session):
    """
    Process foraging behavior data to add useful metrics for plotting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the raw behavior data
    subj_name : str
        Name of the subject
    condition : str
        Experimental condition (e.g., 'Control', 'pre_d', 'DOI', etc.)
    session : int
        Session number
        
    Returns:
    --------
    pandas.DataFrame
        Processed DataFrame with additional metrics
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Add basic metrics
    df['total_reward'] = df['LeftAmount'] + df['RightAmount']
    df['choice'] = np.where(df['LeftAmount'] > df['RightAmount'], 'Left', 'Right')
    df['larger_reward'] = np.maximum(df['LeftAmount'], df['RightAmount'])
    df['smaller_reward'] = np.minimum(df['LeftAmount'], df['RightAmount'])
    df['reward_ratio'] = df['larger_reward'] / df['smaller_reward']
    
    # Add metadata
    df['subject'] = subj_name
    df['condition'] = condition
    df['session_num'] = session
    
    return df

all_dfs = []
names = ['DA76-TNT-LH', 'DA77-TRT-RH', 'DA78-TNT-LRH', 'DA79-TDT-NH', 
         'DA80-TDT-LH', 'DA81-TDT-RH', 'DA82-TDT-NH', 'DA83-TDT-LRH',
         'DA84-TNT-NH', 'DA85-TNT-LH', 'DA86-TNT-RH']

for name in names:
    # Load both files using scipy.io.loadmat
    SessionData = loadmat(f"/Volumes/ChemoBrain/ChemoBrain-Analysis/Data/Foraging_Cisplatin/Cis3_processed/{name}/31-Mar-2025 12:26:04_RecBehav.mat")
    ra_traces = loadmat(f"/Volumes/ChemoBrain/ChemoBrain-Analysis/Data/Foraging_Cisplatin/Cis3_processed/{name}/31-Mar-2025 12:26:04_AlignedPhoto.mat")['aligned_photo']
    n_trials, n_frames = ra_traces.shape
    
    print(f"\nProcessing {name}:")
    print(f"Available keys in SessionData: {SessionData.keys()}")
    
    # Convert 1x192 arrays to 192x1 arrays
    left_amount = SessionData['LeftAmount'].T
    right_amount = SessionData['RightAmount'].T
    
    assert(n_trials == len(left_amount))
    ra_traces = gaussian_filter1d(ra_traces, axis = 1, sigma = 3)

    behav_df = pd.DataFrame.from_dict({
        'LeftAmount': left_amount.flatten(),
        'RightAmount': right_amount.flatten(),
        'name': [name]*n_trials,
        'session': SessionData['TrialTypes'].flatten()  # Using TrialTypes instead of session
    })

    print(name)
    temp_dfs = []
    for session in behav_df.session.unique():
        condition = "Control"
        if (name in ['DA79-TDT-NH', 'DA80-TDT-LH', 'DA81-TDT-RH', 'DA82-TDT-NH', 'DA83-TDT-LRH']):
            group = "A"
            if (session == 7):
                condition = "pre_d"
            if (session == 8):
                condition = "DOI"
            if (session >= 9) and (session < 14):
                condition = "post_d"
        elif (name in ['DA76-TNT-LH', 'DA77-TRT-RH', 'DA78-TNT-LRH', 'DA84-TNT-NH', 'DA85-TNT-LH', 'DA86-TNT-RH']):
            group = "B"
            if (session == 7): #or (session == 6):
                condition = "pre_a"
            if (session == 8):
                condition = "amphetamine"
            if (session >= 9) and (session < 14):
                condition = "post_a"

        bd = behav_df[behav_df.session == session].copy()
        df = proc_foraging_df(bd.reset_index(), subj_name=name,
                         condition=condition, session=session)

        df['group'] = group
        assert(len(bd) == len(df))
        temp_dfs.append(df)

    behav_df = pd.concat(temp_dfs)
    behav_df = {key:np.repeat(behav_df[key], n_frames) for key in behav_df}

    tr = ra_traces.reshape([-1])

    behav_df['session']  = behav_df['session'] + 2
    behav_df['dopamine'] = tr
    #this needs to match the number of seconds in the matlab code!
    behav_df['frames'] = np.tile(np.linspace(-2, 3, n_frames), n_trials)

    all_dfs.append(behav_df)

#%%
#general sample code 
filepath = "sessiondatafilepath"  #filepath of the SessionData object that is saved after matlab processing of photometry and combining of sessions each animal  
SessionData = loadmat(filepath)['SessionData']
ra_traces = SessionData['all'] #reward aligned dopamine traces
n_trials, n_frames = ra_traces.shape
assert(n_trials == len(SessionData['LeftAmount'])) #just making sure that the dopamine data matches the behavior data

#optional, gaussian filter the dopamine traces to smooth
ra_traces = gaussian_filter1d(ra_traces, axis = 1, sigma = 1)

#create a dataframe with the basic behavioral info
behav_df = pd.DataFrame.from_dict({'LeftAmount': SessionData['LeftAmount'], 'RightAmount': SessionData['RightAmount'],
                  'name': [name]*n_trials, 'session': SessionData['session']})

#iterate through all sessions in a single animals behavior data
print(name)
temp_dfs = []
for session in behav_df.session.unique():
    bd = behav_df[behav_df.session == session].copy()
    #process the foraging behavior data to add lots more useful data for plotting
    #I sent this function previously
    df = proc_foraging_df(bd.reset_index(), subj_name=name,
                         condition=condition, session=session)
    temp_dfs.append(df)

#concatenate all sessions into one
behav_df = pd.concat(temp_dfs)

#This extends the dataframe to long form so that the dopamine info can be added. #We can discuss this more if it doesn't make sense, but right now the dataframe #has as many entries as there are trials (e.g. rewards), but the dopamine data has #n_frames (like 60fps * 6 seconds = alot) of data per trial. For ease of plotting #and manipulation we just make a long form dataframe where there is one row for #each frame (time point) in each trial. So the dataframe now has n_trials*n_frames #rows, and we add a new column for n_frame (time) and for the dopamine data 

behav_df = {key:np.repeat(behav_df[key], n_frames) for key in behav_df}

#linearize the traces so they are also n_trials*n_frames
tr = ra_traces.reshape([-1])
#add them to the dataframe
behav_df['dopamine'] = tr

#here is where we make the frames column of the dataframe
#this needs to match the number of seconds in the matlab code! So the below is accurate for a case where you kept two seconds of dopamine data from before the reward and 3 seconds after the reward
behav_df['frames'] = np.tile(np.linspace(-2, 3, n_frames), n_trials)

all_dfs.append(behav_df)