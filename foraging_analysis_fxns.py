import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
import glob
import datetime
import matplotlib.ticker as mticker

# Set plotting parameters
plt.rcParams['font.size'] = '16'
plt.rcParams['savefig.dpi'] = 80
plt.rcParams['figure.dpi'] = 80
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
pd.set_option('display.max_rows', 30)
pd.set_option('display.min_rows', 30)

def extract_datetime(file_name):
    """Extract datetime from filename."""
    date_format = "%d-%b-%Y %H:%M:%S"
    date_time_str = file_name.split('/')[-1].split('_')[0]
    return datetime.datetime.strptime(date_time_str, date_format)

def proc_foraging_df(behav_df, subj_name, condition, session):
    """Process foraging behavior data to add useful metrics for plotting."""
    print(f"[DEBUG] Starting proc_foraging_df for {subj_name} session {session} | behav_df shape: {behav_df.shape}")
    # Handle case where there are fewer than 2 trials (must be first!)
    if len(behav_df) < 2:
        print(f"[DEBUG] Session too short for {subj_name} session {session}: {len(behav_df)} trial(s)")
        behav_df['reward_number'] = [0] * len(behav_df)
        behav_df['name'] = subj_name
        behav_df['condition'] = condition
        behav_df['session_number'] = session
        behav_df['switch'] = [0] * len(behav_df)
        behav_df['switch_prob'] = [0] * len(behav_df)
        behav_df['stay_prob'] = [1] * len(behav_df)
        behav_df['trial_number'] = list(range(len(behav_df)))
        behav_df['poke_number'] = [0] * len(behav_df)
        behav_df['trial_type'] = behav_df['LeftAmount'] if 'LeftAmount' in behav_df else [None] * len(behav_df)
        behav_df['reward_amount'] = behav_df['LeftAmount'] if 'LeftAmount' in behav_df else [None] * len(behav_df)
        print(f"[DEBUG] Returning early for short session: columns = {behav_df.columns.tolist()}")
        return behav_df

    # Robustly determine starting side
    current_side = None
    rsize = None
    if len(behav_df) >= 2:
        if not np.isnan(behav_df.iloc[0]['LeftAmount']) and not np.isnan(behav_df.iloc[1]['LeftAmount']):
            current_side = 'left'
            rsize = behav_df.iloc[0]['LeftAmount']
        elif not np.isnan(behav_df.iloc[0]['RightAmount']) and not np.isnan(behav_df.iloc[1]['RightAmount']):
            current_side = 'right'
            rsize = behav_df.iloc[0]['RightAmount']
        elif not np.isnan(behav_df.iloc[0]['LeftAmount']):
            current_side = 'left'
            rsize = behav_df.iloc[0]['LeftAmount']
        elif not np.isnan(behav_df.iloc[0]['RightAmount']):
            current_side = 'right'
            rsize = behav_df.iloc[0]['RightAmount']
    # If still ambiguous, try just the first trial
    if current_side is None:
        if not np.isnan(behav_df.iloc[0]['LeftAmount']):
            current_side = 'left'
            rsize = behav_df.iloc[0]['LeftAmount']
        elif not np.isnan(behav_df.iloc[0]['RightAmount']):
            current_side = 'right'
            rsize = behav_df.iloc[0]['RightAmount']
    # If still ambiguous, fill with NaNs and return
    if current_side is None:
        print(f"[DEBUG] Could not determine starting side for {subj_name} session {session}. Filling with NaNs.")
        n = len(behav_df)
        behav_df['reward_number'] = [np.nan] * n
        behav_df['name'] = [subj_name] * n
        behav_df['condition'] = [condition] * n
        behav_df['session_number'] = [session] * n
        behav_df['switch'] = [np.nan] * n
        behav_df['switch_prob'] = [np.nan] * n
        behav_df['stay_prob'] = [np.nan] * n
        behav_df['trial_number'] = list(range(n))
        behav_df['poke_number'] = [np.nan] * n
        behav_df['trial_type'] = [np.nan] * n
        behav_df['reward_amount'] = [np.nan] * n
        print(f"[DEBUG] Returning early for ambiguous session: columns = {behav_df.columns.tolist()}")
        return behav_df

    print(f"[DEBUG] Proceeding with full processing for {subj_name} session {session} | starting side: {current_side}, rsize: {rsize}")
    probe_poke = ["first"]
    left = []
    reward_amount = []
    reward_number = []
    starting_rsize = []
    trial_number = []
    epoch_number = []
    n_trials = 0
    n_epochs = 0
    n_rewards = 0

    for i in range(len(behav_df)):
        left_amt = behav_df.iloc[i]['LeftAmount']
        right_amt = behav_df.iloc[i]['RightAmount']
        print(f"[DEBUG] Trial {i}: LeftAmount={left_amt}, RightAmount={right_amt}, current_side={current_side}")
        n_epochs += 1
        n_rewards += 1
        switched = 0.0
        # If current_side is None, append NaN and continue
        if current_side is None:
            print(f"[DEBUG] Trial {i}: current_side is None, appending NaNs to all lists.")
            left.append(np.nan)
            reward_amount.append(np.nan)
            reward_number.append(np.nan)
            starting_rsize.append(np.nan)
            trial_number.append(i)
            epoch_number.append(np.nan)
            continue
        if np.isnan(left_amt) and (current_side == 'left'):
            print(f"[DEBUG] Trial {i}: left_amt is NaN and current_side is left")
            if not np.isnan(right_amt):
                rsize = right_amt
                current_side = 'right'
                switched = 1.0
                n_trials += 1
                n_epochs = 0
            else:
                print(f"[DEBUG] Trial {i}: Both left_amt and right_amt are NaN. Appending NaNs.")
                left.append(np.nan)
                reward_amount.append(np.nan)
                reward_number.append(np.nan)
                starting_rsize.append(np.nan)
                trial_number.append(i)
                epoch_number.append(np.nan)
                continue
        elif np.isnan(right_amt) and (current_side == 'right'):
            print(f"[DEBUG] Trial {i}: right_amt is NaN and current_side is right")
            if not np.isnan(left_amt):
                rsize = left_amt
                current_side = 'left'
                switched = 1.0
                n_trials += 1
                n_epochs = 0
            else:
                print(f"[DEBUG] Trial {i}: Both right_amt and left_amt are NaN. Appending NaNs.")
                left.append(np.nan)
                reward_amount.append(np.nan)
                reward_number.append(np.nan)
                starting_rsize.append(np.nan)
                trial_number.append(i)
                epoch_number.append(np.nan)
                continue
        # Normal processing
        if current_side == 'left':
            print(f"[DEBUG] Trial {i}: current_side is left")
            if np.isnan(left_amt):
                print(f"[DEBUG] Trial {i}: left_amt is NaN when current_side is left. Appending NaNs.")
                left.append(np.nan)
                reward_amount.append(np.nan)
                reward_number.append(np.nan)
                starting_rsize.append(np.nan)
                trial_number.append(i)
                epoch_number.append(np.nan)
                continue
            reward_amount.append(left_amt)
        elif current_side == 'right':
            print(f"[DEBUG] Trial {i}: current_side is right")
            if np.isnan(right_amt):
                print(f"[DEBUG] Trial {i}: right_amt is NaN when current_side is right. Appending NaNs.")
                left.append(np.nan)
                reward_amount.append(np.nan)
                reward_number.append(np.nan)
                starting_rsize.append(np.nan)
                trial_number.append(i)
                epoch_number.append(np.nan)
                continue
            reward_amount.append(right_amt)
        else:
            print(f"[DEBUG] Trial {i}: current_side is invalid. Appending NaNs.")
            left.append(np.nan)
            reward_amount.append(np.nan)
            reward_number.append(np.nan)
            starting_rsize.append(np.nan)
            trial_number.append(i)
            epoch_number.append(np.nan)
            continue
        left.append(switched)
        reward_number.append(n_rewards)
        starting_rsize.append(rsize)
        trial_number.append(n_trials)
        epoch_number.append(n_epochs)

    # Check all output lists for correct length
    n = len(behav_df)
    output_lists = [left, reward_amount, reward_number, starting_rsize, trial_number, epoch_number]
    output_names = ['left', 'reward_amount', 'reward_number', 'starting_rsize', 'trial_number', 'epoch_number']
    for name, lst in zip(output_names, output_lists):
        if len(lst) != n:
            print(f"[DEBUG] Length mismatch for {name}: {len(lst)} vs {n}. Filling with NaNs.")
            lst[:] = [np.nan] * n
    print(f"[DEBUG] Finished loop: left={len(left)}, behav_df['LeftAmount']={len(behav_df['LeftAmount'])}")
    behav_df['reward_number'] = reward_number
    behav_df['name'] = subj_name
    behav_df['condition'] = condition
    behav_df['session_number'] = session
    behav_df['switch'] = left
    behav_df['switch_prob'] = pd.Series(left).shift(-1)
    behav_df['stay_prob'] = 1 - behav_df['switch_prob']
    behav_df['trial_number'] = trial_number
    behav_df['poke_number'] = epoch_number
    behav_df['trial_type'] = starting_rsize
    behav_df['reward_amount'] = np.around(np.array(reward_amount), decimals=4)
    print(f"[DEBUG] Returning processed DataFrame for {subj_name} session {session} | shape: {behav_df.shape}, columns: {behav_df.columns.tolist()}")
    return behav_df

def filter_session(session_data):
    """Filter 60% of the initial data based on session"""
    threshold = int(0.6 * len(session_data))
    return session_data.head(threshold)

def load_animal_data(animal_id, base_path="/Volumes/ChemoBrain/ChemoBrain-Analysis/Data/Foraging_Cisplatin/Cis3_processed"):
    """Load and process data for a single animal."""
    animal_path = os.path.join(base_path, animal_id)
    behav_files = glob.glob(os.path.join(animal_path, "*RecBehav.mat"))
    photom_files = glob.glob(os.path.join(animal_path, "*AlignedPhoto.mat"))

    # Sort files by datetime
    dated_files = [(extract_datetime(fn), fn) for fn in behav_files]
    dated_files.sort()
    behav_files = [fn for dt, fn in dated_files]

    dated_files_photo = [(extract_datetime(fn), fn) for fn in photom_files]
    dated_files_photo.sort()
    photom_files = [fn for dt, fn in dated_files_photo]

    dfs = []
    behav_only = []

    for i, (behav_file, photom_file) in enumerate(zip(behav_files, photom_files)):
        print(f"\nProcessing {animal_id}, session {i}:")
        print(f"Behavior file: {behav_file}")
        print(f"Photometry file: {photom_file}")

        # Extract session date from filename
        session_date = extract_datetime(behav_file)

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
            'name': [animal_id] * n_trials,
            'session': [i] * n_trials,
            'date': [session_date] * n_trials
        })

        # Process behavior data
        df = proc_foraging_df(behav_df.reset_index(), subj_name=animal_id,
                             condition="Control", session=i)
        # Add date column to processed DataFrame (in case proc_foraging_df drops it)
        df['date'] = [session_date] * len(df)

        # Store behavior-only data
        behav_only.append(df)

        # Create long-form dataframe for photometry data
        behav_df_long = {key: np.repeat(df[key], n_frames) for key in df}
        tr = ra_traces.reshape([-1])
        behav_df_long['dopamine'] = tr
        behav_df_long['frames'] = np.tile(np.linspace(-2, 3, n_frames), n_trials)

        dfs.append(pd.DataFrame(behav_df_long))

    return pd.concat(dfs), behav_only

def plot_water_intake(behav_data, animal_ids=None):
    """
    Plot water intake over sessions for one or more animals.
    - behav_data: list of DataFrames (single animal) or dict of {animal_id: list of DataFrames}
    - animal_ids: list of animal IDs to plot (optional, only for dict input)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    plt.figure(figsize=(10, 6))

    # If behav_data is a dict, plot all animals in the dict
    if isinstance(behav_data, dict):
        if animal_ids is None:
            animal_ids = list(behav_data.keys())
        for aid in animal_ids:
            df_all = pd.concat(behav_data[aid], ignore_index=True)
            water_ind = df_all.groupby('session')['reward_amount'].sum().reset_index()
            sns.lineplot(data=water_ind, x='session', y='reward_amount', marker='o', label=aid)
        plt.title('Water Intake Over Sessions')
        plt.xlabel('Session')
        plt.ylabel('Total Water Intake (uL)')
        plt.tight_layout()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    else:
        # Single animal
        df_all = pd.concat(behav_data, ignore_index=True)
        water_ind = df_all.groupby('session')['reward_amount'].sum().reset_index()
        # Do not set label, and do not call plt.legend for single animal
        sns.lineplot(data=water_ind, x='session', y='reward_amount', marker='o')
        if isinstance(animal_ids, str):
            plt.title(f'Water Intake Over Sessions for {animal_ids}')
        else:
            plt.title('Water Intake Over Sessions')
        plt.xlabel('Session')
        plt.ylabel('Total Water Intake (uL)')
        plt.tight_layout()

    plt.show()

def plot_leaving_value(behav_only, animal_id):
    """Plot leaving value over sessions for a single animal."""
    df_all = pd.concat(behav_only)
    df_all = df_all[['session', 'switch_prob', 'trial_type', 'trial_number', 'reward_amount']].reset_index(drop=True)
    lev_sel = df_all.query("switch_prob==1").reset_index(drop=True)
    lev_sel = lev_sel.groupby('session').apply(filter_session).reset_index(drop=True)
    
    # Calculate mean and SEM for each session
    lev_stats = lev_sel.groupby('session').agg({
        'reward_amount': ['mean', 'sem']
    }).reset_index()
    lev_stats.columns = ['session', 'mean', 'sem']
    
    plt.figure(figsize=(10,6))
    # Plot the mean line
    plt.plot(lev_stats['session'], lev_stats['mean'], 'o-', 
             color='#1f77b4', label=animal_id, linewidth=2, markersize=8)
    
    # Add shaded error bars (SEM)
    plt.fill_between(lev_stats['session'], 
                     lev_stats['mean'] - lev_stats['sem'],
                     lev_stats['mean'] + lev_stats['sem'],
                     color='#1f77b4', alpha=0.2)
    
    plt.title(f'Leaving Value Over Sessions - {animal_id}')
    plt.xlabel('Session')
    plt.ylabel('Leaving Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_dopamine_response(df_tot, animal_id, session=7, poke_number=2, date=None, title=None):
    """Plot dopamine response for a specific session and poke number, or by date/date range if provided. Optionally set a custom title."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    plt.figure(figsize=(10,6))
    plot_df = df_tot.groupby('session').apply(filter_session).reset_index(drop=True)
    if 'session' not in plot_df.columns:
        plot_df = plot_df.reset_index()
    # If date is provided, map to session or date range
    if date is not None:
        if isinstance(date, (tuple, list)) and len(date) == 2:
            # Date range
            start_dt = pd.to_datetime(date[0]).normalize()
            end_dt = pd.to_datetime(date[1]).normalize()
            if 'date' in plot_df.columns:
                plot_df['date_norm'] = pd.to_datetime(plot_df['date']).dt.normalize()
                date_mask = (plot_df['date_norm'] >= start_dt) & (plot_df['date_norm'] <= end_dt)
                plot_df = plot_df[date_mask & (plot_df['poke_number'] == poke_number)]
            else:
                raise ValueError("No 'date' column found in input DataFrame.")
        else:
            # Single date
            date_dt = pd.to_datetime(date).normalize()
            if 'date' in plot_df.columns:
                plot_df['date_norm'] = pd.to_datetime(plot_df['date']).dt.normalize()
                session_matches = plot_df[plot_df['date_norm'] == date_dt]['session'].unique()
                if len(session_matches) == 0:
                    raise ValueError(f"No session found for date {date}")
                session = session_matches[0]  # Use the first matching session
            else:
                raise ValueError("No 'date' column found in input DataFrame.")
            plot_df = plot_df.query(f"session=={session} and poke_number=={poke_number}")
    else:
        plot_df = plot_df.query(f"session=={session} and poke_number=={poke_number}")
    # Plot mean ± SEM using seaborn lineplot
    import numpy as np
    if len(plot_df) == 0:
        raise ValueError("No data found for the specified session/date and poke_number.")
    g = sns.lineplot(data=plot_df, x="frames", y="dopamine", errorbar="se", estimator=np.mean, color="#1f77b4")
    if title is None:
        plt.title(f'Dopamine Response by Poke Number - {animal_id}')
    else:
        plt.title(title)
    plt.xlabel('Time from Reward (s)')
    plt.tight_layout()
    plt.show()

def plot_dopamine_heatmap(df_tot, animal_id, session=7):
    """Plot dopamine heatmap for a specific session."""
    ind_DA = df_tot[['session', 'trial_number', 'frames', 'dopamine']]
    session_df = ind_DA[ind_DA['session'] == session]
    pivoted_session_df = session_df.groupby(['trial_number', 'frames'])['dopamine'].mean().unstack('frames')

    plt.figure()
    g = sns.heatmap(pivoted_session_df, cbar=True, xticklabels=10, yticklabels=10, cmap='bwr')
    g.set_xticklabels([])
    x_zero_index = pivoted_session_df.columns.get_loc(0)
    g.axvline(x=x_zero_index, color='red', linestyle='--', linewidth=2)
    plt.title(f'Dopamine Heatmap by Trial - {animal_id}')
    plt.tight_layout()
    plt.show()

def process_cohort(animal_ids, cohort, base_path, global_ns=None):
    """
    Process a cohort of animals, returning a dict mapping animal_id to (df_tot, behav_only).
    If global_ns is provided (e.g., globals()), also creates variables in the caller's namespace:
      - <animal_id>_behav for behav_only
      - <animal_id>_df for df_tot
    """
    data_dict = {}
    for animal_id in animal_ids:
        print(f"\nProcessing animal: {animal_id}")
        df_tot, behav_only = load_animal_data(animal_id, base_path=base_path)
        data_dict[animal_id] = (df_tot, behav_only)
        if global_ns is not None:
            safe_id = animal_id.replace('-', '_')
            global_ns[f'{safe_id}_behav'] = behav_only
            global_ns[f'{safe_id}_df'] = df_tot
    return data_dict

def plot_water_intake_groups(groups, colors=None, title='Water Intake per Session by Group', plot_mode='average', x_date_range=None, x_labels=None, tick_interval=1, date_range_shaded=None, shaded_label=None, baseline=None):
    """
    Plot water intake per session for groups of animals.
    Each group is a dict of {animal_id: list of DataFrames (per session)}.
    colors: dict mapping group_name to hex color code.
    plot_mode: 'average' (default) plots group mean ± SEM, 'individual' plots all animals as individual traces.
    x_date_range: tuple of (start_date, end_date) as strings (MM/DD/YY or similar, must match extract_datetime output)
    x_labels: list of strings for x-axis labels (should match number of sessions after filtering)
    tick_interval: integer, show every Nth tick label
    date_range_shaded: list of (start_date, end_date) tuples to shade on the plot (date-based x-axis only)
    shaded_label: label for the shaded region(s)
    baseline: str or tuple, date or date range to use as baseline for normalization (water intake will be shown as % baseline)
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import matplotlib.ticker as mticker
    plt.figure(figsize=(12, 7))
    
    all_dates = []  # Collect all dates across animals
    
    for group_name, animal_dict in groups.items():
        color = colors[group_name] if colors and group_name in colors else None
        all_animals = []
        first_animal = True
        for animal_id, behav_list in animal_dict.items():
            df_all = pd.concat(behav_list, ignore_index=True)
            # --- BASELINE NORMALIZATION ---
            baseline_value = None
            if baseline is not None:
                if not np.issubdtype(df_all['date'].dtype, np.datetime64):
                    df_all['date'] = pd.to_datetime(df_all['date'])
                # Normalize all dates to midnight for comparison
                df_all['date_norm'] = df_all['date'].dt.normalize()
                if isinstance(baseline, str):
                    baseline_dates = [pd.to_datetime(baseline).normalize()]
                elif isinstance(baseline, (tuple, list)) and len(baseline) == 2:
                    start = pd.to_datetime(baseline[0]).normalize()
                    end = pd.to_datetime(baseline[1]).normalize()
                    baseline_dates = pd.date_range(start, end, freq='D').normalize()
                else:
                    baseline_dates = []
                baseline_mask = df_all['date_norm'].isin(baseline_dates)
                baseline_df = df_all[baseline_mask]
                baseline_value = baseline_df.groupby('date_norm')['reward_amount'].sum().mean() if not baseline_df.empty else None
                if baseline_value is None or baseline_value == 0:
                    print(f"[WARNING] Animal {animal_id} missing or zero baseline value. All values set to NaN.")
            # --- X-AXIS HANDLING ---
            if x_date_range is not None:
                if not np.issubdtype(df_all['date'].dtype, np.datetime64):
                    df_all['date'] = pd.to_datetime(df_all['date'])
                start = pd.to_datetime(x_date_range[0])
                end = pd.to_datetime(x_date_range[1])
                mask = (df_all['date'] >= start) & (df_all['date'] <= end)
                df_all = df_all[mask]
                water_per_session = df_all.groupby(['date'])['reward_amount'].sum().reset_index()
                # Normalize if baseline is set
                if baseline is not None:
                    if baseline_value is not None and baseline_value != 0:
                        water_per_session['reward_amount'] = (water_per_session['reward_amount'] / baseline_value) * 100
                    else:
                        water_per_session['reward_amount'] = np.nan
                all_dates.append(water_per_session['date'])  # Add to all_dates as a Series
                date_to_idx = {date: idx for idx, date in enumerate(water_per_session['date'])}
                water_per_session['x_idx'] = water_per_session['date'].map(date_to_idx)
                all_animals.append(water_per_session.set_index('x_idx')['reward_amount'])
                x_vals = water_per_session['x_idx']
                y_vals = water_per_session['reward_amount']
            else:
                water_per_session = df_all.groupby('session')['reward_amount'].sum().reset_index()
                # Normalize if baseline is set
                if baseline is not None:
                    if baseline_value is not None and baseline_value != 0:
                        water_per_session['reward_amount'] = (water_per_session['reward_amount'] / baseline_value) * 100
                    else:
                        water_per_session['reward_amount'] = np.nan
                all_animals.append(water_per_session.set_index('session')['reward_amount'])
                x_vals = water_per_session['session']
                y_vals = water_per_session['reward_amount']
            if plot_mode == 'individual':
                if first_animal:
                    plt.plot(x_vals, y_vals, color=color, alpha=1, linewidth=2, label=group_name)
                    first_animal = False
                else:
                    plt.plot(x_vals, y_vals, color=color, alpha=1, linewidth=2, label=None)
        if plot_mode == 'average' and all_animals:
            group_df = pd.DataFrame(all_animals)
            mean = group_df.mean(axis=0)
            sem = group_df.sem(axis=0)
            plt.plot(mean.index, mean.values, color=color, label=group_name, linewidth=3)
            plt.fill_between(mean.index, mean - sem, mean + sem, color=color, alpha=0.2)
    
    ax = plt.gca()
    # --- SHADED REGIONS FOR DATE-BASED X-AXIS ---
    if x_date_range is not None and date_range_shaded is not None and all_dates:
        # Normalize all dates to just the date (no time)
        all_dates_normalized = [d.dt.normalize() if hasattr(d, 'dt') else pd.to_datetime(d).normalize() for d in all_dates]
        unique_dates_for_shading = pd.concat(all_dates_normalized).drop_duplicates().sort_values().tolist()
        for i, (shade_start, shade_end) in enumerate(date_range_shaded):
            shade_start_dt = pd.to_datetime(shade_start).normalize()
            shade_end_dt = pd.to_datetime(shade_end).normalize()
            idx_start = None
            idx_end = None
            for idx, dt in enumerate(unique_dates_for_shading):
                if idx_start is None and dt >= shade_start_dt:
                    idx_start = idx
                if dt <= shade_end_dt:
                    idx_end = idx
            if idx_start is not None and idx_end is not None:
                ax.axvspan(idx_start, idx_end, color='gray', alpha=0.2, label=shaded_label if i == 0 and shaded_label else None)
    
    if x_date_range is not None:
        plt.xlabel('Session')
        if x_labels is not None:
            n_ticks = len(x_labels)
            ticks = list(range(0, n_ticks, tick_interval))
            ax.set_xticks(ticks)
            ax.set_xticklabels([x_labels[i] for i in ticks])
    else:
        plt.xlabel('Session')
        ticks = ax.get_xticks()
        if x_labels is not None and len(x_labels) == len(ticks):
            ax.set_xticklabels(x_labels[::tick_interval])
        else:
            ax.set_xticks(ticks[::tick_interval])
    
    plt.ylabel('Total Water Intake (% baseline)' if baseline is not None else 'Total Water Intake (uL)')
    plt.title(title)
    plt.tight_layout()
    x_data = None
    if x_date_range is not None:
        ax.set_xlim(left=-0.5)
    else:
        lines = ax.get_lines()
        if lines:
            x_data = lines[0].get_xdata()
            if len(x_data) > 1:
                min_x = np.min(x_data)
                max_x = np.max(x_data)
                pad = (max_x - min_x) * 0.05
                ax.set_xlim(left=min_x - pad)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

def plot_leaving_value_groups(
    groups,
    colors=None,
    title='Leaving Value per Session by Group',
    plot_mode='average',
    x_date_range=None,
    x_labels=None,
    tick_interval=1,
    date_range_shaded=None,
    shaded_label=None,
    baseline=None
):
    """
    Plot leaving value per session for groups of animals.
    Each group is a dict of {animal_id: list of DataFrames (per session)}.
    colors: dict mapping group_name to hex color code.
    plot_mode: 'average' (default) plots group mean ± SEM, 'individual' plots all animals as individual traces with SEM bands.
    x_date_range: tuple of (start_date, end_date) as strings (MM/DD/YY or similar, must match extract_datetime output)
    x_labels: list of strings for x-axis labels (should match number of sessions after filtering)
    tick_interval: integer, show every Nth tick label
    date_range_shaded: list of (start_date, end_date) tuples to shade on the plot (date-based x-axis only)
    shaded_label: label for the shaded region(s)
    baseline: str or tuple, date or date range to use as baseline for normalization (leaving value will be shown as % baseline)
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import matplotlib.ticker as mticker

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    unique_dates_for_shading = None

    all_x_vals = []
    all_y_vals = []

    for group_name, animal_dict in groups.items():
        group_color = colors[group_name] if colors and group_name in colors else None
        all_animals = []
        all_dates = []
        first_animal = True  # Track first animal for legend
        for animal_id, behav_list in animal_dict.items():
            df_all = pd.concat(behav_list, ignore_index=True)
            if 'date' in df_all.columns:
                if not np.issubdtype(df_all['date'].dtype, np.datetime64):
                    df_all['date'] = pd.to_datetime(df_all['date'])
                df_all['date_norm'] = df_all['date'].dt.normalize()
            else:
                df_all['date_norm'] = pd.NaT
            if x_date_range is not None:
                start = pd.to_datetime(x_date_range[0]).normalize()
                end = pd.to_datetime(x_date_range[1]).normalize()
                mask = (df_all['date_norm'] >= start) & (df_all['date_norm'] <= end)
                df_all = df_all[mask]
            lev_sel = df_all[['session', 'date_norm', 'switch_prob', 'reward_amount']].copy()
            lev_sel = lev_sel.query("switch_prob==1").reset_index(drop=True)
            lev_sel = lev_sel.groupby('session').apply(filter_session).reset_index(drop=True)
            if x_date_range is not None:
                groupby_col = 'date_norm'
            else:
                groupby_col = 'session'
            lev_stats = lev_sel.groupby(groupby_col)['reward_amount'].agg(['mean', 'sem']).reset_index()
            if x_date_range is not None:
                unique_dates = lev_stats[groupby_col].dropna().sort_values().unique()
                date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
                lev_stats['x_idx'] = lev_stats[groupby_col].map(date_to_idx)
                x_vals = lev_stats['x_idx']
                all_dates.append(lev_stats[groupby_col])
            else:
                x_vals = lev_stats[groupby_col]
            y_vals = lev_stats['mean']
            yerr = lev_stats['sem']
            baseline_value = None
            if baseline is not None:
                if groupby_col == 'date_norm':
                    if isinstance(baseline, str):
                        baseline_dates = [pd.to_datetime(baseline).normalize()]
                    elif isinstance(baseline, (tuple, list)) and len(baseline) == 2:
                        start_b = pd.to_datetime(baseline[0]).normalize()
                        end_b = pd.to_datetime(baseline[1]).normalize()
                        baseline_dates = pd.date_range(start_b, end_b, freq='D').normalize()
                    else:
                        baseline_dates = []
                    baseline_mask = lev_stats[groupby_col].isin(baseline_dates)
                    baseline_df = lev_stats[baseline_mask]
                    baseline_value = baseline_df['mean'].mean() if not baseline_df.empty else None
                else:
                    if isinstance(baseline, (int, float, str)):
                        baseline_sessions = [int(baseline)]
                    elif isinstance(baseline, (tuple, list)) and len(baseline) == 2:
                        baseline_sessions = list(range(int(baseline[0]), int(baseline[1])+1))
                    else:
                        baseline_sessions = []
                    baseline_mask = lev_stats[groupby_col].isin(baseline_sessions)
                    baseline_df = lev_stats[baseline_mask]
                    baseline_value = baseline_df['mean'].mean() if not baseline_df.empty else None
                if baseline_value is not None and baseline_value != 0:
                    y_vals = (y_vals / baseline_value) * 100
                    yerr = (yerr / baseline_value) * 100
                else:
                    y_vals = np.nan * y_vals
                    yerr = np.nan * yerr
            all_animals.append(pd.Series(y_vals.values, index=x_vals))
            # Individual mode: plot each animal's trace with SEM band, only first animal gets group label
            if plot_mode == 'individual':
                if first_animal:
                    plt.plot(x_vals, y_vals, alpha=1.0, linewidth=2, label=group_name, color=group_color)
                    plt.fill_between(x_vals, y_vals - yerr, y_vals + yerr, alpha=0.2, color=group_color)
                    first_animal = False
                else:
                    plt.plot(x_vals, y_vals, alpha=1.0, linewidth=2, label=None, color=group_color)
                    plt.fill_between(x_vals, y_vals - yerr, y_vals + yerr, alpha=0.2, color=group_color)
                all_x_vals.extend(x_vals)
                all_y_vals.extend((y_vals - yerr).tolist())
                all_y_vals.extend((y_vals + yerr).tolist())
        # Average mode: plot group mean ± SEM as shaded error band
        if plot_mode == 'average' and all_animals:
            group_df = pd.DataFrame(all_animals)
            group_df = group_df.T
            mean = group_df.mean(axis=1)
            sem = group_df.sem(axis=1)
            x_vals = group_df.index
            plt.plot(x_vals, mean.values, color=group_color, label=group_name, linewidth=3)
            plt.fill_between(x_vals, mean - sem, mean + sem, color=group_color, alpha=0.2)
            all_x_vals.extend(x_vals)
            all_y_vals.extend((mean - sem).tolist())
            all_y_vals.extend((mean + sem).tolist())
    if x_date_range is not None and date_range_shaded is not None and all_dates:
        unique_dates_for_shading = pd.concat(all_dates).drop_duplicates().sort_values().tolist()
        for i, (shade_start, shade_end) in enumerate(date_range_shaded):
            shade_start_dt = pd.to_datetime(shade_start)
            shade_end_dt = pd.to_datetime(shade_end)
            idx_start = None
            idx_end = None
            for idx, dt in enumerate(unique_dates_for_shading):
                if idx_start is None and dt >= shade_start_dt:
                    idx_start = idx
                if dt <= shade_end_dt:
                    idx_end = idx
            if idx_start is not None and idx_end is not None:
                ax.axvspan(idx_start, idx_end, color='gray', alpha=0.2, label=shaded_label if i == 0 and shaded_label else None)
    if x_date_range is not None:
        plt.xlabel('Session')
        if x_labels is not None:
            n_ticks = len(x_labels)
            ticks = list(range(0, n_ticks, tick_interval))
            ax.set_xticks(ticks)
            ax.set_xticklabels([x_labels[i] for i in ticks])
    else:
        plt.xlabel('Session')
        ticks = ax.get_xticks()
        if x_labels is not None and len(x_labels) == len(ticks):
            ax.set_xticklabels(x_labels[::tick_interval])
        else:
            ax.set_xticks(ticks[::tick_interval])
    plt.ylabel('Leaving Value (% baseline)' if baseline is not None else 'Leaving Value')
    plt.title(title)
    if all_x_vals:
        x_vals_arr = pd.Series(all_x_vals)
        if np.issubdtype(x_vals_arr.dtype, np.datetime64):
            pass
        else:
            min_x = np.nanmin(x_vals_arr)
            max_x = np.nanmax(x_vals_arr)
            pad_x = max((max_x - min_x) * 0.05, 0.5)
            ax.set_xlim(left=min_x - pad_x, right=max_x + pad_x)
    if all_y_vals:
        y_vals_arr = np.array(all_y_vals, dtype=float)
        min_y = np.nanmin(y_vals_arr)
        max_y = np.nanmax(y_vals_arr)
        pad_y = max((max_y - min_y) * 0.05, 0.5)
        ax.set_ylim(bottom=min_y - pad_y, top=max_y + pad_y)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

def plot_dopamine_response_groups(groups, colors=None, poke_number=2, title=None):
    """
    Plot mean ± SEM dopamine response for groups of animals, optionally over a date or date range for each animal.
    groups: dict mapping group name to dict of {animal_name: (DataFrame, date/date_range/None) or DataFrame}
    colors: dict mapping group name to color
    poke_number: which poke number to average (default: 2)
    title: optional custom plot title
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    plt.figure(figsize=(10,6))
    group_means = {}
    group_sems = {}
    frames_axis = None
    for group_name, animal_dict in groups.items():
        color = colors[group_name] if colors and group_name in colors else None
        all_traces = []
        for animal_name, animal_val in animal_dict.items():
            # Accept (df, date) or just df
            if isinstance(animal_val, tuple):
                df, date = animal_val
            else:
                df = animal_val
                date = None
            plot_df = df.groupby('session').apply(filter_session).reset_index(drop=True)
            if 'session' not in plot_df.columns:
                plot_df = plot_df.reset_index()
            # Date filtering
            if date is not None:
                if isinstance(date, (tuple, list)) and len(date) == 2:
                    start_dt = pd.to_datetime(date[0]).normalize()
                    end_dt = pd.to_datetime(date[1]).normalize()
                    if 'date' in plot_df.columns:
                        plot_df['date_norm'] = pd.to_datetime(plot_df['date']).dt.normalize()
                        date_mask = (plot_df['date_norm'] >= start_dt) & (plot_df['date_norm'] <= end_dt)
                        plot_df = plot_df[date_mask & (plot_df['poke_number'] == poke_number)]
                    else:
                        raise ValueError("No 'date' column found in input DataFrame.")
                else:
                    date_dt = pd.to_datetime(date).normalize()
                    if 'date' in plot_df.columns:
                        plot_df['date_norm'] = pd.to_datetime(plot_df['date']).dt.normalize()
                        session_matches = plot_df[plot_df['date_norm'] == date_dt]['session'].unique()
                        if len(session_matches) == 0:
                            continue
                        session = session_matches[0]
                    else:
                        raise ValueError("No 'date' column found in input DataFrame.")
                    plot_df = plot_df.query(f"session=={session} and poke_number=={poke_number}")
            else:
                plot_df = plot_df.query(f"poke_number=={poke_number}")
            # For each trial, get the dopamine trace (frames, dopamine)
            if len(plot_df) == 0:
                continue
            # Group by trial (trial_number) and frames, then average across trials
            trial_group = plot_df.groupby(['trial_number', 'frames'])['dopamine'].mean().unstack('frames')
            all_traces.append(trial_group.values)
            if frames_axis is None:
                frames_axis = trial_group.columns.values
        # Stack all animal traces for this group
        if len(all_traces) == 0:
            continue
        all_traces = np.vstack(all_traces)  # shape: (n_trials_total, n_frames)
        mean_trace = np.nanmean(all_traces, axis=0)
        sem_trace = np.nanstd(all_traces, axis=0) / np.sqrt(all_traces.shape[0])
        group_means[group_name] = mean_trace
        group_sems[group_name] = sem_trace
        plt.plot(frames_axis, mean_trace, color=color, label=group_name, linewidth=3)
        plt.fill_between(frames_axis, mean_trace - sem_trace, mean_trace + sem_trace, color=color, alpha=0.2)
    if title is None:
        plt.title('Dopamine Response by Poke Number - Groups')
    else:
        plt.title(title)
    plt.xlabel('Time from Reward (s)')
    plt.ylabel('Dopamine')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_dopamine_peak(df_tot, animal_id, poke_number=2, date=None, x_labels=None, title=None):
    """
    Plot the mean ± SEM of the peak dopamine value for a given poke_number across a date range.
    x-axis: session (or date), y-axis: mean peak dopamine for that session.
    Parameters:
        df_tot: DataFrame with dopamine data
        animal_id: str, for labeling
        poke_number: int, which poke number to analyze (default: 2)
        date: tuple/list of two strings (start, end) for date range, or single date string
        x_labels: list of strings for x-axis labels (optional)
        title: custom plot title (optional)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    # Filter and normalize dates
    plot_df = df_tot.groupby('session').apply(filter_session).reset_index(drop=True)
    if 'session' not in plot_df.columns:
        plot_df = plot_df.reset_index()
    if date is not None:
        if isinstance(date, (tuple, list)) and len(date) == 2:
            start_dt = pd.to_datetime(date[0]).normalize()
            end_dt = pd.to_datetime(date[1]).normalize()
            if 'date' in plot_df.columns:
                plot_df['date_norm'] = pd.to_datetime(plot_df['date']).dt.normalize()
                date_mask = (plot_df['date_norm'] >= start_dt) & (plot_df['date_norm'] <= end_dt)
                plot_df = plot_df[date_mask & (plot_df['poke_number'] == poke_number)]
            else:
                raise ValueError("No 'date' column found in input DataFrame.")
        else:
            date_dt = pd.to_datetime(date).normalize()
            if 'date' in plot_df.columns:
                plot_df['date_norm'] = pd.to_datetime(plot_df['date']).dt.normalize()
                session_matches = plot_df[plot_df['date_norm'] == date_dt]['session'].unique()
                if len(session_matches) == 0:
                    raise ValueError(f"No session found for date {date}")
                session = session_matches[0]
            else:
                raise ValueError("No 'date' column found in input DataFrame.")
            plot_df = plot_df.query(f"session=={session} and poke_number=={poke_number}")
    else:
        plot_df = plot_df.query(f"poke_number=={poke_number}")
    # For each session/date, compute mean and SEM of peak dopamine
    if 'date_norm' in plot_df.columns:
        group_col = 'date_norm'
    elif 'date' in plot_df.columns:
        group_col = 'date'
    else:
        group_col = 'session'
    peak_stats = plot_df.groupby([group_col, 'trial_number']).apply(lambda x: x['dopamine'].max()).reset_index()
    # Now group by session/date and compute mean and SEM
    peak_summary = peak_stats.groupby(group_col)[0].agg(['mean', 'sem']).reset_index()
    # X-axis: session index or date
    x_vals = np.arange(len(peak_summary))
    y_mean = peak_summary['mean']
    y_sem = peak_summary['sem']
    plt.figure(figsize=(10,6))
    plt.errorbar(x_vals, y_mean, yerr=y_sem, fmt='o-', color='#1f77b4', linewidth=2, markersize=8, capsize=4)
    if x_labels is not None and len(x_labels) == len(x_vals):
        plt.xticks(x_vals, x_labels)
        plt.xlabel('Session')
    else:
        # Use date/session as x-tick labels
        plt.xticks(x_vals, [str(val)[:10] for val in peak_summary[group_col]])
        plt.xlabel('Session/Date')
    plt.ylabel('Peak Dopamine')
    if title is None:
        plt.title(f'Peak Dopamine by Session - {animal_id}')
    else:
        plt.title(title)
    plt.tight_layout()
    plt.show() 