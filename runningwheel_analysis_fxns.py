import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import scipy.stats as stats
import re
import inspect

# Set matplotlib parameters for better quality
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Default colors for plotting
default_colors = plt.cm.tab10.colors

# Global variable to store the running wheel data
running_wheel_data = None

def get_base_path():
    """
    Get the base path for data files based on the operating system.
    
    Returns:
    --------
    str
        The base path for data files
    """
    if os.name == 'posix':  # macOS
        return '/Volumes/ChemoBrain/ChemoBrain-Analysis/Data/'
    elif os.name == 'nt':  # Windows
        return 'E:/Data/'  # Change this to match your Windows path
    else:
        raise Exception("Unsupported operating system")

def get_running_wheel_data():
    """
    Get the running wheel data, loading it if necessary.
    
    Returns:
    --------
    pandas.DataFrame
        The loaded running wheel data
    """
    global running_wheel_data
    if running_wheel_data is None:
        base_path = get_base_path()
        # Load the running wheel data
        running_wheel_data = pd.read_csv(os.path.join(base_path, 'RunningWheel_Data.csv'))
    return running_wheel_data

def import_RW_main(animal_ID, date, cohort, experiment='RW_Cisplatin'):
    """
    Import running wheel data for a specific animal on a specific date.
    
    Parameters:
    -----------
    animal_ID : str
        The ID of the animal
    date : str
        The date in MMDDYY format
    cohort : str
        The cohort name
    experiment : str, optional
        The experiment name (default: 'RW_Cisplatin')
    
    Returns:
    --------
    pandas.DataFrame or None
        The imported data or None if the file was not found
    """
    base_path = get_base_path()
    date_reversed = date[4:] + date[0:2] + date[2:4]
    file_path = os.path.join(base_path, experiment, cohort, animal_ID, f'{date_reversed}.CSV')
    
    # Use try-except to handle file not found scenario
    try:
        df = pd.read_csv(file_path)
        
        temp = df.index
        df.set_index(np.arange(len(df)), inplace=True)
        df.insert(0, 'YYYY-MM-DD HH:MM:SS.sss', temp)
        date_header = df.columns[2]
        date_str = date_header.split(': ')[1].split()[0]
        formatted_date = pd.to_datetime(date_str, format='%Y/%m/%d').strftime('%Y-%m-%d')

        df = df[~df.iloc[:, 0].astype(str).str.contains(r'24:', na=False)]  # get rid of any indices containing 24: as it is the next day
        
        timestamps = df.iloc[:, 0]
        second_col_values = df.iloc[:, 1]
        combined_datetimes = []

        for i, time in enumerate(timestamps):
            if len(time.split('.')[-1]) == 2:
                time_part, millis = time.split('.')
                new_millis = millis + str(second_col_values.iloc[i])[-1]
                time = f"{time_part}.{new_millis}"
            combined_datetime = f"{formatted_date} {time}"
            combined_datetimes.append(combined_datetime)

        df.loc[:, 'YYYY-MM-DD HH:MM:SS.sss'] = pd.to_datetime(combined_datetimes, format='%Y-%m-%d %H:%M:%S.%f')
        df.set_index(np.arange(len(df)), inplace=True)
        
        return df
    
    except FileNotFoundError:
        print(f"File not found: {file_path}. Skipping this date.")
        return None  # Return None if the file is not found

def import_RW(animal_ID, start_date, end_date, cohort, experiment='RW_Cisplatin'):
    """
    Import running wheel data for a specific animal or range of animals over a date range.
    
    Parameters:
    -----------
    animal_ID : str, tuple, or list
        Either a single animal ID (e.g., 'DA68'), a tuple of two animal IDs 
        representing a range (e.g., ('DA55', 'DA89')), or a list of specific animal IDs
        (e.g., ['DA5', 'DA6', 'DA15', 'DA16'])
    start_date : str
        The start date in MMDDYY format
    end_date : str
        The end date in MMDDYY format
    cohort : str
        The cohort name
    experiment : str, optional
        The experiment name (default: 'RW_Cisplatin')
    
    Returns:
    --------
    dict
        A dictionary mapping animal IDs to DataFrames
    """
    # Handle single animal ID
    if isinstance(animal_ID, str):
        # Convert dates to datetime objects for date_range
        start_dt = datetime.strptime(start_date, '%m%d%y')
        end_dt = datetime.strptime(end_date, '%m%d%y')
        date_list = pd.date_range(start=start_dt, end=end_dt, freq='D').strftime('%m%d%y').tolist()

        dataframes = []

        for date in date_list:
            df = import_RW_main(animal_ID, date, cohort, experiment)
            
            # Only append the dataframe if it is not None
            if df is not None:
                dataframes.append(df)
        
        if dataframes:  # Check if there are any dataframes to concatenate
            first_df_headers = dataframes[0].columns

            for i in range(len(dataframes)):
                dataframes[i].columns = first_df_headers

            concatenated_df = pd.concat(dataframes, axis=0, ignore_index=True)
            
            # Create variable name
            var_name = f"{animal_ID}_{cohort}"
            print(f"Created {var_name}")
            
            return {var_name: concatenated_df}
        else:
            print("No dataframes were created. Please check the date range and file paths.")
            return {}
    
    # Handle range of animal IDs
    elif isinstance(animal_ID, tuple) and len(animal_ID) == 2:
        start_id, end_id = animal_ID
        
        # Extract the prefix and number parts
        start_match = re.match(r'([A-Za-z]+)(\d+)', start_id)
        end_match = re.match(r'([A-Za-z]+)(\d+)', end_id)
        
        if start_match and end_match:
            prefix = start_match.group(1)
            start_num = int(start_match.group(2))
            end_num = int(end_match.group(2))
            
            # Generate all animal IDs in the range
            animal_ids = [f"{prefix}{num}" for num in range(start_num, end_num + 1)]
            
            result_dict = {}
            
            for animal_id in animal_ids:
                print(f"Importing data for {animal_id}...")
                single_result = import_RW(animal_id, start_date, end_date, cohort, experiment)
                result_dict.update(single_result)
            
            return result_dict
        else:
            print("Invalid animal ID format. Expected format: 'DA68' or ('DA55', 'DA89')")
            return {}
    
    # Handle list of animal IDs
    elif isinstance(animal_ID, list):
        result_dict = {}
        
        for animal_id in animal_ID:
            print(f"Importing data for {animal_id}...")
            single_result = import_RW(animal_id, start_date, end_date, cohort, experiment)
            result_dict.update(single_result)
        
        return result_dict
    
    else:
        print("Invalid animal_ID parameter. Expected a string, a tuple of two strings, or a list of strings.")
        return {}

def RW_plot(data, bin_size='8h', plot_title='Running Wheel Activity per {bin_size}', 
            plot_xlabel='Day', plot_ylabel='Wheel Activity', circadian=False):
    """
    Plot running wheel activity data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The running wheel data to plot
    bin_size : str, optional
        The size of time bins for aggregation (default: '8h')
    plot_title : str, optional
        Title for the plot
    plot_xlabel : str, optional
        Label for x-axis
    plot_ylabel : str, optional
        Label for y-axis
    circadian : bool, optional
        Whether to plot circadian rhythm (default: False)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot
    """
    data_copy = data.copy()
    data_copy.set_index('YYYY-MM-DD HH:MM:SS.sss', inplace=True)
    
    binned_activity = data_copy.resample(bin_size).size()

    plot_title = plot_title.format(bin_size=bin_size)

    plt.figure(figsize=(10, 6))
    plt.plot(binned_activity.index, binned_activity, marker='o', linestyle='-', color='b')
    if circadian == True:
        for date in binned_activity.index.date:
            plt.axvspan(pd.Timestamp(date) + pd.Timedelta(hours=18), 
                    pd.Timestamp(date + pd.Timedelta(days=1)) + pd.Timedelta(hours=6), 
                    color='lightgrey', alpha=0.5)
    plt.title(plot_title)
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    plt.xticks(rotation=45)
    plt.show()

def multi_RW_plot(data_dict, bin_size='1D', plot_title='Running Wheel Activity per {bin_size}',
                 plot_xlabel='Day', plot_ylabel='Wheel Activity', circadian=False, 
                 x_date_range=None, x_labels=None, tick_interval=None, 
                 date_range_shaded=None, baseline_date=None, group_colors=None):
    """
    Plot multiple running wheel activity datasets.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary mapping group names to lists of data labels
    bin_size : str, optional
        The size of time bins for aggregation (default: '1D')
    plot_title : str, optional
        Title for the plot
    plot_xlabel : str, optional
        Label for x-axis
    plot_ylabel : str, optional
        Label for y-axis
    circadian : bool, optional
        Whether to plot circadian rhythm (default: False)
    x_date_range : tuple, optional
        Tuple of (start_date, end_date) for x-axis limits
    x_labels : list, optional
        List of labels for x-axis ticks
    tick_interval : int, optional
        Interval between x-axis ticks
    date_range_shaded : tuple, optional
        Tuple of (start_date, end_date) for shaded region
    baseline_date : str, optional
        Date to use as baseline
    group_colors : list, optional
        List of colors for each group
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot
    """
    plt.figure(figsize=(10, 6))

    # Ensure the data_dict contains DataFrames for each group
    if not isinstance(data_dict, dict):
        raise ValueError("data_dict must be a dictionary with group names as keys and lists of data labels as values")

    # Prepare default labels and colors
    group_names = list(data_dict.keys())
    if group_colors is None:
        group_colors = plt.cm.tab10.colors[:len(group_names)]

    # Iterate over the groups
    for group_idx, (group_name, group_labels) in enumerate(data_dict.items()):
        # Collect the DataFrames for the current group
        group_data = [data[label] for label in group_labels]

        # Set the title
        title = plot_title.format(bin_size=bin_size)

        # Plot individual data for each label
        for data in group_data:
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Expected a Pandas DataFrame but got {type(data)} for group {group_name}")

            # Copy the DataFrame and set the index
            data_copy = data.copy()
            data_copy.set_index('YYYY-MM-DD HH:MM:SS.sss', inplace=True)

            # Bin the data
            binned_activity = data_copy.resample(bin_size).size()

            # Plot the binned activity for the individual label
            plt.plot(
                binned_activity.index, 
                binned_activity, 
                marker='o', 
                linestyle='-', 
                label=f"{group_name} - {group_labels}", 
                color=group_colors[group_idx]
            )

    # Add circadian shading if specified
    if circadian:
        for date in binned_activity.index.date:
            plt.axvspan(
                pd.Timestamp(date) + pd.Timedelta(hours=18),
                pd.Timestamp(date + pd.Timedelta(days=1)) + pd.Timedelta(hours=6),
                color='lightgrey',
                alpha=0.5
            )

    # Add shaded date range if specified
    if date_range_shaded:
        plt.axvspan(
            pd.Timestamp(date_range_shaded[0]),
            pd.Timestamp(date_range_shaded[1]),
            color='lightblue',
            alpha=0.3
        )

    # Format x-axis with date limits and intervals
    if x_date_range:
        plt.xlim(pd.Timestamp(x_date_range[0]), pd.Timestamp(x_date_range[1]))
    if x_labels and x_date_range:
        start_date, end_date = pd.to_datetime(x_date_range[0]), pd.to_datetime(x_date_range[1])
        all_dates = pd.date_range(start=start_date, end=end_date, periods=len(x_labels))
        selected_dates = all_dates[::tick_interval] if tick_interval else all_dates
        plt.gca().set_xticks(selected_dates)
        plt.gca().set_xticklabels(x_labels[::tick_interval] if tick_interval else x_labels, rotation=45)

    plt.title(title)
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    plt.legend()
    plt.show()

def group_RW_plot(groups, bin_size='1D', plot_title='Running Wheel Activity per {bin_size}', 
                 plot_xlabel='Day', plot_ylabel='Wheel Activity', circadian=False,
                 group_colors=None, x_date_range=None, x_labels=None, tick_interval=1,
                 dashed_line_date1=None, dashed_line_label1=None, 
                 dashed_line_date2=None, dashed_line_label2=None,
                 show_sem='shaded', y_limit_max=None, plot_mode='average',
                 date_range_shaded=None, shaded_label=None, baseline=None):
    """
    Plot running wheel activity for groups of animals.
    
    Parameters:
    -----------
    groups : dict
        Dictionary mapping group labels to lists of DataFrames
    bin_size : str, optional
        The size of time bins for aggregation (default: '1D')
    plot_title : str, optional
        Title for the plot
    plot_xlabel : str, optional
        Label for x-axis
    plot_ylabel : str, optional
        Label for y-axis
    circadian : bool, optional
        Whether to plot circadian rhythm (default: False)
    group_colors : dict, optional
        Dictionary mapping group labels to colors
    x_date_range : tuple, optional
        Tuple of (start_date, end_date) for x-axis limits
    x_labels : list, optional
        List of labels for x-axis ticks
    tick_interval : int, optional
        Interval between x-axis ticks (default: 1)
    dashed_line_date1 : list, optional
        List of dates for first set of dashed vertical lines
    dashed_line_label1 : str, optional
        Label for first set of dashed lines
    dashed_line_date2 : list, optional
        List of dates for second set of dashed vertical lines
    dashed_line_label2 : str, optional
        Label for second set of dashed lines
    show_sem : str, optional
        How to display standard error of the mean ('shaded', 'error_bars', or None)
    y_limit_max : float, optional
        Maximum value for y-axis
    plot_mode : str, optional
        Plot mode ('average' or 'individual')
    date_range_shaded : list, optional
        List of tuples for shaded date ranges
    shaded_label : str, optional
        Label for shaded regions
    baseline : str or tuple, optional
        Baseline date or date range for normalization
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot
    """
    plt.figure(figsize=(10, 6))

    # Set default colors or use provided group_colors
    if not group_colors:
        group_colors = {}
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))

    max_value = 0  # Track the maximum value across all groups for y-axis scaling
    combined_binned_activity = None  # Initialize to handle circadian and shaded range

    for group_index, (group_label, group_data) in enumerate(groups.items()):
        color = group_colors.get(group_label, colors[group_index])

        if plot_mode == 'average':
            all_binned_activity = []

            for data in group_data:
                data_copy = data.copy()
                data_copy.set_index('YYYY-MM-DD HH:MM:SS.sss', inplace=True)

                binned_activity = data_copy.resample(bin_size).size()

                # Apply baseline normalization if specified
                if baseline:
                    # Calculate baseline based on single date or date range
                    if isinstance(baseline, str):  # Single date
                        baseline_date = pd.to_datetime(baseline)
                        baseline_value = binned_activity.loc[baseline_date] if baseline_date in binned_activity.index else None
                    elif isinstance(baseline, tuple):  # Range of dates
                        start, end = pd.to_datetime(baseline[0]), pd.to_datetime(baseline[1])
                        baseline_subset = binned_activity.loc[start:end]
                        baseline_value = baseline_subset.mean() if not baseline_subset.empty else None
                    else:
                        baseline_value = None

                    # Normalize to percentage if baseline is valid
                    if baseline_value is not None and baseline_value > 0:
                        binned_activity = (binned_activity / baseline_value) * 100
                        plot_ylabel = 'Wheel Activity (% baseline)'

                all_binned_activity.append(binned_activity)

            combined_binned_activity = pd.concat(all_binned_activity, axis=1)

            # Handle missing data
            combined_binned_activity.fillna(0, inplace=True)

            averaged_activity = combined_binned_activity.mean(axis=1)
            sem_activity = combined_binned_activity.sem(axis=1)

            max_value = max(max_value, (averaged_activity + sem_activity).max())

            # Plot averages and SEMs
            plt.plot(
                averaged_activity.index, 
                averaged_activity, 
                marker='o', 
                linestyle='-', 
                color=color, 
                label=group_label
            )
            if show_sem == 'shaded':
                plt.fill_between(
                    averaged_activity.index, 
                    averaged_activity - sem_activity, 
                    averaged_activity + sem_activity, 
                    color=color, 
                    alpha=0.2
                )
            elif show_sem == 'error_bars':
                plt.errorbar(
                    averaged_activity.index, 
                    averaged_activity, 
                    yerr=sem_activity, 
                    fmt='o', 
                    color=color, 
                    elinewidth=2, 
                    capsize=4
                )
        
        elif plot_mode == 'individual':
            for individual_index, data in enumerate(group_data):
                data_copy = data.copy()
                data_copy.set_index('YYYY-MM-DD HH:MM:SS.sss', inplace=True)

                binned_activity = data_copy.resample(bin_size).size()

                # Apply baseline normalization if specified (same logic as in average mode)
                if baseline:
                    if isinstance(baseline, str):  # Single date
                        baseline_date = pd.to_datetime(baseline)
                        baseline_value = binned_activity.loc[baseline_date] if baseline_date in binned_activity.index else None
                    elif isinstance(baseline, tuple):  # Range of dates
                        start, end = pd.to_datetime(baseline[0]), pd.to_datetime(baseline[1])
                        baseline_subset = binned_activity.loc[start:end]
                        baseline_value = baseline_subset.mean() if not baseline_subset.empty else None
                    else:
                        baseline_value = None

                    # Normalize to percentage if baseline is valid
                    if baseline_value is not None and baseline_value > 0:
                        binned_activity = (binned_activity / baseline_value) * 100
                        plot_ylabel = 'Wheel Activity (% baseline)'

                max_value = max(max_value, binned_activity.max())

                # Plot individual traces with group label
                plt.plot(
                    binned_activity.index, 
                    binned_activity, 
                    marker='o', 
                    linestyle='-', 
                    color=color, 
                    alpha=1,
                    label=group_label if individual_index == 0 else "_nolegend_"
                )

    # Handle dashed vertical lines
    if dashed_line_date1:
        for i, date_str in enumerate(dashed_line_date1):
            date = pd.to_datetime(date_str)
            if i == 0:
                plt.axvline(date, color='black', linestyle='--', label=dashed_line_label1)
            else:
                plt.axvline(date, color='black', linestyle='--')

    if dashed_line_date2:
        for i, date_str in enumerate(dashed_line_date2):
            date = pd.to_datetime(date_str)
            if i == 0:
                plt.axvline(date, color='blue', linestyle='--', label=dashed_line_label2)
            else:
                plt.axvline(date, color='blue', linestyle='--')

    # Apply circadian shading if enabled
    if circadian and combined_binned_activity is not None:
        for date in combined_binned_activity.index.date:
            plt.axvspan(
                pd.Timestamp(date) + pd.Timedelta(hours=18), 
                pd.Timestamp(date + pd.Timedelta(days=1)) + pd.Timedelta(hours=6), 
                color='lightgrey', 
                alpha=0.5
            )

    # Add shaded date ranges with a single label
    if date_range_shaded:
        for i, (start, end) in enumerate(date_range_shaded):
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            if i == 0:
                plt.axvspan(start, end, color='lightgrey', alpha=0.5, label=shaded_label)
            else:
                plt.axvspan(start, end, color='lightgrey', alpha=0.5)

    # Apply x_date_range
    if x_date_range:
        start_date, end_date = pd.to_datetime(x_date_range[0]), pd.to_datetime(x_date_range[1])
        plt.xlim(start_date, end_date)

    # Apply x_labels and tick_interval
    if x_labels and x_date_range:
        start_date, end_date = pd.to_datetime(x_date_range[0]), pd.to_datetime(x_date_range[1])
        all_dates = pd.date_range(start=start_date, end=end_date, periods=len(x_labels))
        selected_dates = all_dates[::tick_interval]
        plt.gca().set_xticks(selected_dates)
        plt.gca().set_xticklabels(x_labels[::tick_interval])

    # Adjust y-axis limits
    if y_limit_max is not None:
        y_max = y_limit_max
    else:
        y_max = max_value

    plt.ylim(0, y_max)

    plt.title(plot_title.format(bin_size=bin_size))
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

def RW_plot_circadian(data, bin_size='1h', plot_title='Average Running Wheel Activity per {bin_size}',
                     plot_xlabel='Hours Since Start of Light Cycle', plot_ylabel='Average Wheel Activity'):
    """
    Plot circadian rhythm of running wheel activity.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The running wheel data to plot
    bin_size : str, optional
        The size of time bins for aggregation (default: '1h')
    plot_title : str, optional
        Title for the plot
    plot_xlabel : str, optional
        Label for x-axis
    plot_ylabel : str, optional
        Label for y-axis
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot
    """
    data_copy = data.copy()
    data_copy.set_index('YYYY-MM-DD HH:MM:SS.sss', inplace=True)

    binned_activity = data_copy.resample(bin_size).size()

    binned_activity_df = binned_activity.to_frame(name='activity')
    binned_activity_df['hour'] = binned_activity_df.index.hour
    binned_activity_df['date'] = binned_activity_df.index.date

    hourly_group = binned_activity_df.groupby('hour')['activity']
    hourly_mean = hourly_group.mean()
    hourly_sem = hourly_group.sem()

    # Adjust the hours so that the light cycle starts at hour 0 (6 AM is hour 0)
    hours_since_light = (hourly_mean.index - 6) % 24
    hourly_mean.index = hours_since_light
    hourly_sem.index = hours_since_light

    hourly_mean = hourly_mean.sort_index()
    hourly_sem = hourly_sem.sort_index()

    hours = list(range(25))
    
    mean_values = hourly_mean.values.tolist() + [hourly_mean.values[0]]
    sem_values = hourly_sem.values.tolist() + [hourly_sem.values[0]]

    plt.figure(figsize=(10, 6))
    plt.errorbar(hours, mean_values, yerr=sem_values, 
                 fmt='o-', capsize=5)

    # Shading for dark periods (6 PM to 6 AM)
    plt.axvspan(12, 24, color='lightgrey', alpha=0.5)

    plt.title(plot_title.format(bin_size=bin_size))
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    plt.xticks(range(0, 25))
    plt.tight_layout()
    plt.legend()
    plt.show()

def multi_RW_plot_circadian(dfs, labels, bin_size='1h', plot_title='Average Running Wheel Activity per {bin_size}', 
                           plot_xlabel='Hours Since Start of Light Cycle', plot_ylabel='Average Wheel Activity',
                           date_ranges=None):
    """
    Plot multiple circadian rhythms of running wheel activity.
    
    Parameters:
    -----------
    dfs : list of pandas.DataFrame
        List of running wheel datasets to plot
    labels : list of str
        Labels for each dataset
    bin_size : str, optional
        The size of time bins for aggregation (default: '1h')
    plot_title : str, optional
        Title for the plot
    plot_xlabel : str, optional
        Label for x-axis
    plot_ylabel : str, optional
        Label for y-axis
    date_ranges : dict, optional
        A dictionary where keys are labels and values are lists of start and end dates 
        in 'MM/DD/YY' format to filter the data
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot
    """
    plt.figure(figsize=(10, 6))

    for data, label in zip(dfs, labels):
        data_copy = data.copy()

        # Set datetime as index if not already
        data_copy.set_index('YYYY-MM-DD HH:MM:SS.sss', inplace=True)
        
        # Apply date filtering if date_ranges is provided
        if date_ranges and label in date_ranges:
            start_date, end_date = date_ranges[label]
            data_copy = data_copy[data_copy.index >= pd.to_datetime(start_date)]
            data_copy = data_copy[data_copy.index <= pd.to_datetime(end_date)]

        # Resample the data based on the bin size
        binned_activity = data_copy.resample(bin_size).size()

        # Create DataFrame for binned activity
        binned_activity_df = binned_activity.to_frame(name='activity')
        binned_activity_df['hour'] = binned_activity_df.index.hour
        binned_activity_df['date'] = binned_activity_df.index.date

        # Group by hour and calculate mean and SEM
        hourly_group = binned_activity_df.groupby('hour')['activity']
        hourly_mean = hourly_group.mean()
        hourly_sem = hourly_group.sem()

        # Adjust hours so that light cycle starts at hour 0 (6 AM is hour 0)
        hours_since_light = (hourly_mean.index - 6) % 24
        hourly_mean.index = hours_since_light
        hourly_sem.index = hours_since_light

        hourly_mean = hourly_mean.sort_index()
        hourly_sem = hourly_sem.sort_index()

        hours = list(range(25))
        
        # Add the first value to the end to close the cycle
        mean_values = hourly_mean.values.tolist() + [hourly_mean.values[0]]
        sem_values = hourly_sem.values.tolist() + [hourly_sem.values[0]]

        # Plot with error bars
        plt.errorbar(hours, mean_values, yerr=sem_values, 
                    fmt='o-', capsize=5, label=label)

    # Shading for dark periods (6 PM to 6 AM)
    plt.axvspan(12, 24, color='lightgrey', alpha=0.5)

    # Set plot title and labels
    plt.title(plot_title.format(bin_size=bin_size))
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    plt.xticks(range(0, 25))
    plt.tight_layout()
    plt.legend()

    # Save and show the plot
    plt.show()

def group_RW_plot_circadian(
    groups,
    group_time_points,
    time_col='YYYY-MM-DD HH:MM:SS.sss',
    baseline_date=None,
    group_colors=None,
    plot_title='Average Running Wheel Activity by Hour of Light Cycle',
    group_time_points_colors=None,
    group_time_points_labels=None,
    y_label='Average Wheel Activity'
):
    # If group_time_points is a single list, apply it to all groups
    if isinstance(group_time_points, list):
        universal_time_points = group_time_points
        group_time_points = {
            g_label: universal_time_points for g_label in groups.keys()
        }

    plt.figure(figsize=(10, 6))

    # If group_colors is missing or incomplete, create fallback color cycle
    if not group_colors:
        color_cycle = plt.cm.viridis(np.linspace(0, 1, len(groups)))
        group_colors = dict(zip(groups.keys(), color_cycle))

    # Iterate over each group
    for i, (group_label, df_list) in enumerate(groups.items()):

        # If no time points for this group, skip
        if group_label not in group_time_points:
            continue

        # For each date in group_time_points[group_label]
        for date_str in group_time_points[group_label]:
            target_date = pd.to_datetime(date_str).date()

            # Set color and label for this date
            color = group_time_points_colors[group_label].get(date_str, group_colors[group_label])
            legend_label = group_time_points_labels[group_label].get(date_str, f"{group_label} ({date_str})")

            # Collect per-hour data from all DataFrames for this date
            hourly_activity_list = []

            for df in df_list:
                data_copy = df.copy()

                # Check if time_col exists
                if time_col not in data_copy.columns:
                    continue

                # Set up time index
                try:
                    data_copy[time_col] = pd.to_datetime(data_copy[time_col], format='%m/%d/%Y %H:%M:%S')
                    data_copy.set_index(time_col, inplace=True)
                except Exception:
                    continue

                # Resample to hourly bins
                binned_activity = data_copy.resample('1h').size()
                binned_activity_df = binned_activity.to_frame(name='activity')
                binned_activity_df['hour'] = binned_activity_df.index.hour
                binned_activity_df['date'] = binned_activity_df.index.date

                # Filter to target_date only
                day_df = binned_activity_df[binned_activity_df['date'] == target_date].copy()

                if day_df.empty:
                    continue

                # Apply baseline subtraction if needed
                if baseline_date:
                    baseline_date_dt = pd.to_datetime(baseline_date).date()
                    baseline_data = binned_activity_df[binned_activity_df['date'] == baseline_date_dt]
                    if not baseline_data.empty:
                        baseline_hourly_mean = baseline_data.groupby('hour')['activity'].mean()
                        day_df = day_df.merge(
                            baseline_hourly_mean.rename('baseline_activity'),
                            left_on='hour',
                            right_index=True,
                            how='left'
                        )
                        day_df['activity'] -= day_df['baseline_activity'].fillna(0)

                # Append aggregated hourly activity (sum) to the list
                hourly_activity_list.append(day_df.groupby('hour')['activity'].sum())

            if not hourly_activity_list:
                continue

            # Combine hourly activity across all DataFrames for this date
            combined_hourly_activity = pd.concat(hourly_activity_list, axis=1)

            # Calculate mean and SEM across all DataFrames for each hour
            hourly_mean = combined_hourly_activity.mean(axis=1)
            hourly_sem = combined_hourly_activity.sem(axis=1)

            # Shift hours so 6 AM = 0, and 6 PM = 12
            hourly_mean.index = (hourly_mean.index - 6) % 24
            hourly_sem.index = (hourly_sem.index - 6) % 24

            # Sort by shifted hour
            hourly_mean = hourly_mean.sort_index()
            hourly_sem = hourly_sem.sort_index()

            # Wrap from 0 to 24 hours
            hours = list(range(25))
            mean_vals = hourly_mean.values.tolist() + [hourly_mean.values[0]]
            sem_vals = hourly_sem.values.tolist() + [hourly_sem.values[0]]

            plt.errorbar(
                hours,
                mean_vals,
                yerr=sem_vals,
                fmt='o-',
                color=color,
                capsize=5,
                label=legend_label
            )

    plt.axvspan(12, 24, color='lightgrey', alpha=0.5)
    plt.title(plot_title)
    plt.xlabel('Hours Since Light Onset (0 = 6 AM)')
    plt.ylabel(y_label)
    plt.xticks(range(0, 25))
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

def boxplot_RW_plot(
    groups,
    group_time_points,
    x_labels,
    time_point_colors,
    point_alpha=0.7,
    y_label=None,
    plot_title="Running Wheel Activity Comparisons by Group and Time Point",
    baseline=None,
    time_point_legend_labels=None,
    bin_size='1D'
):
    """
    Create a boxplot of running wheel activity with individual data points.
    
    Parameters:
    -----------
    groups : dict
        Dictionary mapping group labels to lists of DataFrames
    group_time_points : dict
        Dictionary mapping group labels to lists of dates to plot
    x_labels : list
        List of labels for x-axis
    time_point_colors : dict
        Dictionary mapping time point indices to colors
    point_alpha : float, optional
        Alpha value for individual data points (default: 0.7)
    y_label : str, optional
        Label for y-axis
    plot_title : str, optional
        Title for the plot
    baseline : str or tuple, optional
        Baseline date or date range for normalization
    time_point_legend_labels : dict, optional
        Dictionary mapping time point indices to legend labels
    bin_size : str, optional
        The size of time bins for aggregation (default: '1D')
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot
    """
    time_point_label_color_map = {}
    max_time_points = max(len(times) for times in group_time_points.values())

    for idx in range(max_time_points):
        label = time_point_legend_labels.get(str(idx), f"Round {idx + 1}") if time_point_legend_labels else f"Round {idx + 1}"
        color = time_point_colors.get(str(idx), '#cccccc')  # Default color
        time_point_label_color_map[idx] = (label, color)

    all_data = []
    all_groups = []
    all_time_points = []

    for group, animal_ids in groups.items():
        group_time_points_list = group_time_points.get(group, [])
        for animal in animal_ids:
            for idx, time_point in enumerate(group_time_points_list):
                data_copy = animal.copy()
                data_copy['YYYY-MM-DD HH:MM:SS.sss'] = pd.to_datetime(data_copy['YYYY-MM-DD HH:MM:SS.sss'])
                data_copy.set_index('YYYY-MM-DD HH:MM:SS.sss', inplace=True)

                binned_activity = data_copy.resample(bin_size).size()

                if baseline:
                    if isinstance(baseline, str):
                        baseline_date = pd.to_datetime(baseline)
                        baseline_value = binned_activity.loc[baseline_date] if baseline_date in binned_activity.index else None
                    elif isinstance(baseline, tuple):
                        start, end = pd.to_datetime(baseline[0]), pd.to_datetime(baseline[1])
                        baseline_subset = binned_activity.loc[start:end]
                        baseline_value = baseline_subset.mean() if not baseline_subset.empty else None
                    else:
                        baseline_value = None

                    if baseline_value is not None and baseline_value > 0:
                        binned_activity = (binned_activity / baseline_value) * 100
                        y_label = 'Wheel Activity (% baseline)'

                time_point_date = pd.to_datetime(time_point)
                time_point_data = binned_activity.loc[time_point_date] if time_point_date in binned_activity.index else 0

                all_data.append(time_point_data)
                all_groups.append(group)
                all_time_points.append(idx)

    data_df = pd.DataFrame({
        "Group": pd.Categorical(all_groups, categories=x_labels, ordered=True),
        "Time Point": pd.Categorical(all_time_points, categories=range(max_time_points), ordered=True),
        "Activity": all_data
    })

    palette = {idx: time_point_colors.get(str(idx), '#cccccc') for idx in range(max_time_points)}

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=data_df,
        x="Group",
        y="Activity",
        hue="Time Point",
        palette=palette,
        showcaps=True,
        boxprops={'edgecolor': 'black'},
        medianprops={'color': 'black'},
        whiskerprops={'linewidth': 1.5},
        showfliers=False
    )

    sns.stripplot(
        data=data_df,
        x="Group",
        y="Activity",
        hue="Time Point",
        dodge=True,
        palette=palette,
        alpha=point_alpha,
        jitter=True,
        marker="o",
        linewidth=1,
        edgecolor='black',
        legend=False
    )

    plt.title(plot_title)
    plt.xlabel("")
    plt.ylabel(y_label if y_label else "Wheel Activity")
    plt.gca().set_xticklabels(x_labels)

    handles, labels = ax.get_legend_handles_labels()
    legend_labels = [time_point_label_color_map[idx][0] for idx in range(max_time_points)]
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=time_point_colors.get(str(idx), '#cccccc')) for idx in range(max_time_points)]
    plt.legend(legend_handles, legend_labels, title="Time Point", loc='best')

    plt.tight_layout()
    plt.show()

def group_velocity_plot(
    groups, 
    bin_size='1D', 
    plot_title='Running Wheel Velocity per {bin_size}', 
    plot_xlabel='Day', 
    plot_ylabel='Velocity (rev/min)', 
    circadian=False,
    group_colors=None,
    x_date_range=None,
    x_labels=None,
    tick_interval=1,
    dashed_line_date1=None, 
    dashed_line_label1=None, 
    dashed_line_date2=None, 
    dashed_line_label2=None,
    show_sem=None,  # Options: 'shaded', 'error_bars', or None
    y_limit_max=None,  # Optional hard cap for max y-axis value
    plot_mode='average',  # 'average' or 'individual'
    date_range_shaded=None,
    shaded_label=None,
    baseline=None  # Set baseline date (str) or date range (tuple) for normalization
):
    """
    Single-function version that calculates SEM in both 'average' and 'individual' modes.

    - 'average' mode:
        For each mouse, compute one velocity value per bin. Then compute the group-level
        mean ± SEM across mice for each bin.

    - 'individual' mode:
        For each mouse, compute mean ± SEM of minute-level velocities within each bin,
        then plot one line per mouse (with error bars or shaded SEM if requested).
    
    Parameters:
    -----------
    groups : dict
        Dictionary mapping group labels to lists of DataFrames
    bin_size : str, optional
        The size of time bins for aggregation (default: '1D')
    plot_title : str, optional
        Title for the plot
    plot_xlabel : str, optional
        Label for x-axis
    plot_ylabel : str, optional
        Label for y-axis
    circadian : bool, optional
        Whether to plot circadian rhythm (default: False)
    group_colors : dict, optional
        Dictionary mapping group labels to colors
    x_date_range : tuple, optional
        Tuple of (start_date, end_date) for x-axis limits
    x_labels : list, optional
        List of labels for x-axis ticks
    tick_interval : int, optional
        Interval between x-axis ticks (default: 1)
    dashed_line_date1 : list, optional
        List of dates for first set of dashed vertical lines
    dashed_line_label1 : str, optional
        Label for first set of dashed lines
    dashed_line_date2 : list, optional
        List of dates for second set of dashed vertical lines
    dashed_line_label2 : str, optional
        Label for second set of dashed lines
    show_sem : str, optional
        How to display standard error of the mean ('shaded', 'error_bars', or None)
    y_limit_max : float, optional
        Maximum value for y-axis
    plot_mode : str, optional
        Plot mode ('average' or 'individual')
    date_range_shaded : list, optional
        List of tuples for shaded date ranges
    shaded_label : str, optional
        Label for shaded regions
    baseline : str or tuple, optional
        Baseline date or date range for normalization
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot
    """
    plt.figure(figsize=(10, 6))

    # Set default colors or use provided group_colors
    if not group_colors:
        group_colors = {}
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))

    max_value = 0.0  # Track max value (for y-axis scaling)
    combined_binned_velocity = None  # We'll store group-level data here in 'average' mode

    # --------------------------------------------------------------------------
    # Function to apply baseline normalization (mean, sem) if requested
    # --------------------------------------------------------------------------
    def apply_baseline_normalization(binned_data, baseline, col_mean='velocity', col_sem=None):
        """
        binned_data: A Series or DataFrame that includes columns like 'velocity_mean', 'velocity_sem',
                     or just a single column 'velocity'.
        baseline can be a string (single date) or tuple (start_date, end_date).
        col_mean: name of the column for mean velocity
        col_sem:  name of the column for SEM (if any)
        """
        # Figure out the baseline value
        if isinstance(baseline, str):
            baseline_date = pd.to_datetime(baseline)
            if baseline_date in binned_data.index:
                baseline_value = binned_data.loc[baseline_date, col_mean]
            else:
                baseline_value = None
        elif isinstance(baseline, tuple):
            start, end = pd.to_datetime(baseline[0]), pd.to_datetime(baseline[1])
            subset = binned_data.loc[start:end, col_mean]
            baseline_value = subset.mean() if not subset.empty else None
        else:
            baseline_value = None

        # Apply normalization
        if baseline_value is not None and baseline_value > 0:
            if isinstance(binned_data, pd.DataFrame):
                binned_data[col_mean] = (binned_data[col_mean] / baseline_value) * 100
                if col_sem and (col_sem in binned_data.columns):
                    binned_data[col_sem] = (binned_data[col_sem] / baseline_value) * 100
            elif isinstance(binned_data, pd.Series):
                binned_data = (binned_data / baseline_value) * 100

        return binned_data

    # --------------------------------------------------------------------------
    # MAIN LOOP OVER GROUPS
    # --------------------------------------------------------------------------
    for group_index, (group_label, group_data_list) in enumerate(groups.items()):
        color = group_colors.get(group_label, colors[group_index])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1) "AVERAGE" MODE
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if plot_mode == 'average':
            # We'll collect 1 column per mouse, each Series is 'velocity' for that mouse
            all_binned_velocity = []

            for data in group_data_list:
                data_copy = data.copy()

                # 1a) Ensure DateTimeIndex if needed
                #     If your data has a column "YYYY-MM-DD HH:MM:SS.sss", use it as index
                if 'YYYY-MM-DD HH:MM:SS.sss' in data_copy.columns:
                    data_copy.index = pd.to_datetime(data_copy['YYYY-MM-DD HH:MM:SS.sss'], errors='coerce')
                    data_copy.drop(columns=['YYYY-MM-DD HH:MM:SS.sss'], inplace=True, errors='ignore')

                # 1b) Minute-level ticks
                minute_ticks = data_copy.resample('1min').size().astype(int)
                minute_ticks = minute_ticks.to_frame('ticks')
                minute_ticks['active'] = minute_ticks['ticks'] > 12

                # 1c) Aggregator: single velocity per bin
                #     We do sum of ticks for active minutes / # of active minutes, then /4 => rev
                def agg_single_velocity(x):
                    active_ticks = x.loc[x['active'], 'ticks']
                    if len(active_ticks) > 0:
                        return (active_ticks.sum() / len(active_ticks)) / 4.0
                    else:
                        return 0.0

                binned_velocity = minute_ticks.resample(bin_size).apply(agg_single_velocity).astype(float)
                binned_velocity = binned_velocity.to_frame('velocity')

                # 1d) Apply baseline normalization if requested
                if baseline:
                    binned_velocity = apply_baseline_normalization(
                        binned_velocity, baseline,
                        col_mean='velocity', col_sem=None
                    )
                    # If we changed the scale, let's change the y-label
                    if isinstance(binned_velocity, pd.DataFrame):
                        if (binned_velocity['velocity'] > 100).any():
                            plot_ylabel = 'Velocity (% baseline)'

                # We'll keep just the Series, which has a single "velocity" column
                # We rename to avoid confusion, each mouse will be one column in the final DataFrame
                binned_velocity_series = binned_velocity['velocity'].rename(f"mouse_{len(all_binned_velocity)}")
                all_binned_velocity.append(binned_velocity_series)

            # 1e) Combine across mice -> group-level mean, group-level SEM
            combined_binned_velocity = pd.concat(all_binned_velocity, axis=1).fillna(0).astype(float)
            group_mean = combined_binned_velocity.mean(axis=1)
            group_sem  = combined_binned_velocity.sem(axis=1)

            # Update max_value
            max_value = max(max_value, (group_mean + group_sem).max())

            # 1f) Plot
            plt.plot(
                group_mean.index,
                group_mean,
                marker='o',
                linestyle='-',
                color=color,
                label=group_label
            )
            if show_sem == 'shaded':
                plt.fill_between(
                    group_mean.index,
                    group_mean - group_sem,
                    group_mean + group_sem,
                    color=color,
                    alpha=0.2
                )
            elif show_sem == 'error_bars':
                plt.errorbar(
                    group_mean.index,
                    group_mean,
                    yerr=group_sem,
                    fmt='o',
                    color=color,
                    elinewidth=2,
                    capsize=4
                )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2) "INDIVIDUAL" MODE
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif plot_mode == 'individual':
            # Each DataFrame = one mouse. We'll do minute-level aggregator returning mean+sem per bin
            for individual_index, data in enumerate(group_data_list):
                data_copy = data.copy()

                # 2a) Ensure DateTimeIndex
                if 'YYYY-MM-DD HH:MM:SS.sss' in data_copy.columns:
                    data_copy.index = pd.to_datetime(data_copy['YYYY-MM-DD HH:MM:SS.sss'], errors='coerce')
                    data_copy.drop(columns=['YYYY-MM-DD HH:MM:SS.sss'], inplace=True, errors='ignore')

                # 2b) Minute-level
                minute_ticks = data_copy.resample('1min').size().astype(int)
                minute_ticks = minute_ticks.to_frame('ticks')
                minute_ticks['active'] = minute_ticks['ticks'] > 12

                # 2c) Aggregator returning mean & SEM across active minutes
                def agg_mean_sem(x):
                    active_ticks = x.loc[x['active'], 'ticks']
                    if len(active_ticks) > 0:
                        # Convert from ticks -> rev for each minute, then compute distribution stats
                        minute_vels = active_ticks / 4.0
                        mean_val = minute_vels.mean()
                        sem_val  = minute_vels.std() / np.sqrt(len(minute_vels))
                        return pd.Series({'velocity_mean': mean_val, 'velocity_sem': sem_val})
                    else:
                        return pd.Series({'velocity_mean': 0.0, 'velocity_sem': 0.0})

                binned_stats = minute_ticks.resample(bin_size).apply(agg_mean_sem).fillna(0)
                binned_stats = binned_stats.astype(float)

                # 2d) Baseline normalization (both mean & sem)
                if baseline:
                    binned_stats = apply_baseline_normalization(
                        binned_stats, baseline,
                        col_mean='velocity_mean',
                        col_sem='velocity_sem'
                    )
                    if (binned_stats['velocity_mean'] > 100).any():
                        plot_ylabel = 'Velocity (% baseline)'

                # 2e) Extract
                velocity_mean = binned_stats['velocity_mean']
                velocity_sem  = binned_stats['velocity_sem']

                # Update max_value
                max_value = max(max_value, (velocity_mean + velocity_sem).max())

                # 2f) Plot each mouse's data
                plt.plot(
                    velocity_mean.index,
                    velocity_mean,
                    marker='o',
                    linestyle='-',
                    color=color,
                    alpha=1,
                    label=group_label if individual_index == 0 else "_nolegend_"
                )

                # Show error bars or shaded region
                if show_sem == 'shaded':
                    plt.fill_between(
                        velocity_mean.index,
                        velocity_mean - velocity_sem,
                        velocity_mean + velocity_sem,
                        color=color,
                        alpha=0.2
                    )
                elif show_sem == 'error_bars':
                    plt.errorbar(
                        velocity_mean.index,
                        velocity_mean,
                        yerr=velocity_sem,
                        fmt='o',
                        color=color,
                        elinewidth=2,
                        capsize=4
                    )
        else:
            raise ValueError(f"Invalid plot_mode: {plot_mode}. Use 'average' or 'individual'.")

    # ----------------------------------------------------------------------------
    # Dashed vertical lines (e.g. treatment days)
    # ----------------------------------------------------------------------------
    if dashed_line_date1:
        for i, date_str in enumerate(dashed_line_date1):
            date = pd.to_datetime(date_str)
            if i == 0:
                plt.axvline(date, color='black', linestyle='--', label=dashed_line_label1)
            else:
                plt.axvline(date, color='black', linestyle='--')

    if dashed_line_date2:
        for i, date_str in enumerate(dashed_line_date2):
            date = pd.to_datetime(date_str)
            if i == 0:
                plt.axvline(date, color='blue', linestyle='--', label=dashed_line_label2)
            else:
                plt.axvline(date, color='blue', linestyle='--')

    # ----------------------------------------------------------------------------
    # CIRCADIAN SHADING (optional)
    # ----------------------------------------------------------------------------
    # Only straightforward if we have a single combined index for 'average' mode.
    # If you want circadian shading for 'individual' mode, adapt similarly.
    if circadian and (combined_binned_velocity is not None):
        for date in combined_binned_velocity.index.date:
            plt.axvspan(
                pd.Timestamp(date) + pd.Timedelta(hours=18), 
                pd.Timestamp(date) + pd.Timedelta(days=1, hours=6), 
                color='lightgrey', 
                alpha=0.5
            )

    # ----------------------------------------------------------------------------
    # SHADED DATE RANGES (optional)
    # ----------------------------------------------------------------------------
    if date_range_shaded:
        for i, (start, end) in enumerate(date_range_shaded):
            start_ts = pd.to_datetime(start)
            end_ts   = pd.to_datetime(end)
            if i == 0:
                plt.axvspan(start_ts, end_ts, color='lightgrey', alpha=0.5, label=shaded_label)
            else:
                plt.axvspan(start_ts, end_ts, color='lightgrey', alpha=0.5)

    # ----------------------------------------------------------------------------
    # X-AXIS OPTIONS
    # ----------------------------------------------------------------------------
    if x_date_range:
        start_date, end_date = pd.to_datetime(x_date_range[0]), pd.to_datetime(x_date_range[1])
        plt.xlim(start_date, end_date)

    if x_labels and x_date_range:
        start_date, end_date = pd.to_datetime(x_date_range[0]), pd.to_datetime(x_date_range[1])
        all_dates = pd.date_range(start=start_date, end=end_date, periods=len(x_labels))
        selected_dates = all_dates[::tick_interval]
        plt.gca().set_xticks(selected_dates)
        plt.gca().set_xticklabels(x_labels[::tick_interval])

    # ----------------------------------------------------------------------------
    # Y-AXIS OPTIONS
    # ----------------------------------------------------------------------------
    if y_limit_max is not None:
        plt.ylim(0, y_limit_max)
    else:
        plt.ylim(0, max_value)

    # ----------------------------------------------------------------------------
    # Final labeling
    # ----------------------------------------------------------------------------
    plt.title(plot_title.format(bin_size=bin_size))
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

def boxplot_velocity_plot(
    groups,
    group_time_points,
    x_labels,
    time_point_colors,
    point_alpha=0.7,
    y_label=None,
    plot_title="Running Wheel Velocity Comparisons by Group and Time Point",
    baseline=None,
    time_point_legend_labels=None,
    bin_size='1D'
):
    """
    Create a boxplot of running wheel velocity (rev/min) with individual data points.
    
    Parameters:
    -----------
    groups : dict
        Dictionary mapping group labels to a list of DataFrames (one per mouse).
    group_time_points : dict
        Dictionary mapping group labels to a list of time points (strings or datetimes).
    x_labels : list
        Labels for the groups, in the order that will appear on the x-axis.
    time_point_colors : dict
        Dictionary mapping stringified time point indices to hex color codes.
    point_alpha : float, optional
        Alpha for the stripplot points (default: 0.7).
    y_label : str, optional
        Y-axis label. Defaults to "Velocity" or "Velocity (% baseline)" if baseline is applied.
    plot_title : str, optional
        Title of the plot.
    baseline : None, str, or tuple, optional
        - None: no normalization
        - str: single date for baseline
        - tuple: (start_date, end_date) for baseline
    time_point_legend_labels : dict, optional
        Dict mapping stringified time point indices to legend labels.
    bin_size : str, optional
        Resampling bin size (e.g. '1D', '12H', '1W', etc.) (default: '1D').
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot
    """

    # --------------------------------------------------------------------------
    # Prepare label-color map for time points
    # --------------------------------------------------------------------------
    time_point_label_color_map = {}
    max_time_points = max(len(times) for times in group_time_points.values())

    for idx in range(max_time_points):
        label = (time_point_legend_labels.get(str(idx), f"Round {idx + 1}")
                 if time_point_legend_labels else f"Round {idx + 1}")
        color = time_point_colors.get(str(idx), '#cccccc')  # Default color
        time_point_label_color_map[idx] = (label, color)

    # --------------------------------------------------------------------------
    # Collect all data for boxplot
    # --------------------------------------------------------------------------
    all_data = []
    all_groups = []
    all_time_points_idx = []

    for group, df_list in groups.items():
        # time points for this group
        group_time_points_list = group_time_points.get(group, [])

        for df in df_list:
            # 1) Ensure the DataFrame has a DateTime index
            df_copy = df.copy()
            if 'YYYY-MM-DD HH:MM:SS.sss' in df_copy.columns:
                df_copy.index = pd.to_datetime(df_copy['YYYY-MM-DD HH:MM:SS.sss'], errors='coerce')
                df_copy.drop(columns=['YYYY-MM-DD HH:MM:SS.sss'], inplace=True, errors='ignore')

            # 2) Resample at 1-min intervals to count ticks
            minute_ticks = df_copy.resample('1min').size().astype(int)
            minute_ticks = minute_ticks.to_frame('ticks')

            # 3) Mark active minutes (ticks > 12)
            minute_ticks['active'] = minute_ticks['ticks'] > 12

            # 4) Compute velocity per bin:
            #    sum-of-active-ticks / number-of-active-minutes / 4 => rev/min
            binned_velocity = (
                minute_ticks
                .resample(bin_size)
                .apply(
                    lambda x: (
                        (x.loc[x['active'], 'ticks'].sum() / len(x.loc[x['active'], 'ticks'])) / 4.0
                    ) if len(x.loc[x['active'], 'ticks']) > 0 else 0.0
                )
                .astype(float)
                .to_frame(name='velocity')
            )

            # 5) If baseline is set, find the baseline value and normalize
            if baseline:
                if isinstance(baseline, str):
                    baseline_date = pd.to_datetime(baseline)
                    if baseline_date in binned_velocity.index:
                        baseline_value = binned_velocity.loc[baseline_date, 'velocity']
                    else:
                        baseline_value = None
                elif isinstance(baseline, tuple):
                    start, end = pd.to_datetime(baseline[0]), pd.to_datetime(baseline[1])
                    baseline_subset = binned_velocity.loc[start:end, 'velocity']
                    baseline_value = baseline_subset.mean() if not baseline_subset.empty else None
                else:
                    baseline_value = None

                if baseline_value is not None and baseline_value > 0:
                    binned_velocity['velocity'] = (
                        binned_velocity['velocity'] / baseline_value
                    ) * 100
                    if not y_label:
                        y_label = "Velocity (% baseline)"

            # 6) For each time point in this group, pick the velocity
            for idx, time_point in enumerate(group_time_points_list):
                time_point_date = pd.to_datetime(time_point)
                if time_point_date in binned_velocity.index:
                    val = binned_velocity.loc[time_point_date, 'velocity']
                else:
                    val = 0.0

                all_data.append(val)
                all_groups.append(group)
                all_time_points_idx.append(idx)

    # --------------------------------------------------------------------------
    # Build a DataFrame for plotting
    # --------------------------------------------------------------------------
    data_df = pd.DataFrame({
        "Group": pd.Categorical(all_groups, categories=x_labels, ordered=True),
        "Time Point": pd.Categorical(all_time_points_idx, categories=range(max_time_points), ordered=True),
        "Velocity": all_data
    })

    # Prepare color palette for each time point index
    palette = {idx: time_point_colors.get(str(idx), '#cccccc') for idx in range(max_time_points)}

    # --------------------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Boxplot
    ax = sns.boxplot(
        data=data_df,
        x="Group",
        y="Velocity",
        hue="Time Point",
        palette=palette,
        showcaps=True,
        boxprops={'edgecolor': 'black'},
        medianprops={'color': 'black'},
        whiskerprops={'linewidth': 1.5},
        showfliers=False
    )

    # Stripplot (individual points)
    sns.stripplot(
        data=data_df,
        x="Group",
        y="Velocity",
        hue="Time Point",
        dodge=True,
        palette=palette,
        alpha=point_alpha,
        jitter=True,
        marker="o",
        linewidth=1,
        edgecolor='black',
        legend=False  # We'll handle legend separately
    )

    # --------------------------------------------------------------------------
    # Final styling
    # --------------------------------------------------------------------------
    plt.title(plot_title)
    plt.xlabel("")
    if not y_label:
        y_label = "Velocity"  # Default label if no baseline normalization
    plt.ylabel(y_label)
    plt.gca().set_xticklabels(x_labels)

    # Create a custom legend matching time points to colors
    legend_labels = [time_point_label_color_map[idx][0] for idx in range(max_time_points)]
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=time_point_colors.get(str(idx), '#cccccc')) 
        for idx in range(max_time_points)
    ]
    plt.legend(
        legend_handles, 
        legend_labels, 
        title="Time Point", 
        loc='best'
    )

    plt.tight_layout()
    plt.show()

# Add your running wheel analysis functions here
# For example:
# def plot_circadian_rhythm()
# def calculate_activity_metrics()
# def analyze_sleep_wake_cycles()
# etc. 