from manim import config
config["no_latex_cleanup"] = True      # keep our LaTeX logs around
config["max_files_cached"] = 1_000_000 # disable cache‐cleanup errors

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import scipy.stats as stats
import re
import inspect
from collections import OrderedDict
import traceback
import sys
import math
from manim import *
from manim import (
    Scene, Axes, BarChart, FadeIn, FadeOut,
    Text, VGroup, Square, UP, DOWN, LEFT, UR
)
import numpy as np
import pandas as pd


# Set matplotlib parameters for better quality
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Default colors for plotting
default_colors = plt.cm.tab10.colors

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

def concat_FED(dfs):
    """
    Concatenate multiple FED DataFrames, maintaining cumulative values for specific columns.
    
    Parameters:
    -----------
    dfs : list
        List of DataFrames to concatenate
        
    Returns:
    --------
    pandas.DataFrame
        Concatenated DataFrame with cumulative values
    """
    if len(dfs) < 2:
        raise ValueError("At least two DataFrames are required for concatenation.")

    # Define columns that need cumulative values
    cols_to_add = ['Left_Poke_Count', 'Right_Poke_Count', 'Pellet_Count']
    
    # Start with the first DataFrame
    concatenated_df = dfs[0].copy()

    # Process each subsequent DataFrame
    for i in range(1, len(dfs)):
        df = dfs[i].copy()
        
        # Drop any all-NA columns before concatenation
        df.dropna(axis=1, how='all', inplace=True)

        # Check if concatenated_df is empty before accessing iloc[-1]
        if not concatenated_df.empty:
            last_values = concatenated_df[cols_to_add].iloc[-1]
            
            # Add the last values to each row in the new DataFrame
            for col in cols_to_add:
                if col in df.columns:  # Only modify if column exists
                    df[col] += last_values[col]

        # Concatenate
        concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)

    return concatenated_df

def import_FED(animal_ID, cohort, experiment='Cisplatin'):
    """
    Import FED data for a group of animals.
    
    Parameters:
    -----------
    animal_ID : str or list
        Either a single animal ID (e.g., 'DA1') or a list of animal IDs
        (e.g., ['DA12', 'DA1', 'DA11', 'DA2'])
    cohort : str
        The cohort name (e.g., 'Optm1')
    experiment : str, optional
        The experiment name (default: 'Cisplatin')
    
    Returns:
    --------
    dict
        A dictionary mapping animal IDs to their respective DataFrames
    """
    # Convert single animal ID to list for consistent processing
    if isinstance(animal_ID, str):
        animal_ID = [animal_ID]
    
    # Get the base path
    base_path = get_base_path()
    
    # Construct the experiment directory name with FED_ prefix
    experiment_dir = f"FED_{experiment}"
    
    # Dictionary to store results
    results = {}
    
    # Process each animal
    for animal in animal_ID:
        print(f"\nProcessing animal: {animal}")
        
        # Construct the animal's directory path
        animal_dir = os.path.join(base_path, experiment_dir, cohort, animal)
        
        if not os.path.exists(animal_dir):
            print(f"Directory not found: {animal_dir}")
            continue
        
        # Get all CSV files in the directory
        all_files = [
            f for f in os.listdir(animal_dir)
            if f.endswith('.CSV') and not f.startswith('._')
        ]
        
        if not all_files:
            print(f"No CSV files found for {animal}")
            continue
        
        # Sort files by date and sequence number
        file_info = []
        for file in all_files:
            match = re.search(r"FED\d+_(\d{6})_(\d{2})\.CSV", file)
            if match:
                date = match.group(1)
                seq = int(match.group(2))
                file_info.append((date, seq, file))
        
        file_info.sort(key=lambda x: (x[0], x[1]))
        
        # Expected columns to check for
        expected_cols = {'Left_Poke_Count', 'Right_Poke_Count', 'Pellet_Count'}
        
        # The final column name we want in the concatenated DataFrame
        final_datetime_col = "MM:DD:YYYY hh:mm:ss"
        
        # Possible original column names for date/time
        possible_original_cols = [
            "MM:DD:YYYY hh:mm:ss",   # if it already exists
            "MM:DD:YYYY_hh:mm:ss",  # or with underscore
            "MM/DD/YYYY hh:mm:ss",  # slash-based name
            # Add others as needed...
        ]
        
        dfs = []
        bad_files = []
        
        for _, _, file in file_info:
            file_path = os.path.join(animal_dir, file)
            
            # Read with fallback encodings
            try:
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
            
            # Clean column names (leave colons alone)
            df.columns = (
                df.columns
                  .str.strip()
                  .str.replace(r'\s+', '_', regex=True)
            )
            
            # Skip files that only have 'Unnamed: 0'
            if set(df.columns) == {'Unnamed:_0'}:
                continue
            
            # Check for required columns
            if not expected_cols.issubset(df.columns):
                bad_files.append(file)
                continue
            
            # If we find one of the possible date/time columns, rename it
            found_dt_col = None
            for possible_col in possible_original_cols:
                if possible_col in df.columns:
                    found_dt_col = possible_col
                    df.rename(columns={possible_col: final_datetime_col}, inplace=True)
                    break
            
            # Append DataFrame to list
            dfs.append(df)
        
        if bad_files:
            print(f"Skipped {len(bad_files)} file(s) due to missing columns: {bad_files}")
        
        if dfs:
            # Use concat_FED to ensure cumulative values
            results[f"{animal}_{cohort}"] = concat_FED(dfs)
            print(f"Successfully imported data for {animal}")
        else:
            print(f"No valid data found for {animal}")
    
    return results

# ===== Plotting Functions =====

def pellet_plot(data, time_col='MM:DD:YYYY hh:mm:ss', pellet_col='Pellet_Count', plot_title='Pellets Per 4-Hour Interval', plot_xlabel='Day', plot_ylabel='Pellet Count per 4 hours'):
    """
    Plot pellet counts in 4-hour intervals with shading for lights-out hours.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing FED data
    time_col : str, optional
        Name of the time column (default: 'MM:DD:YYYY hh:mm:ss')
    pellet_col : str, optional
        Name of the pellet count column (default: 'Pellet_Count')
    plot_title : str, optional
        Title for the plot (default: 'Pellets Per 4-Hour Interval')
    plot_xlabel : str, optional
        Label for x-axis (default: 'Day')
    plot_ylabel : str, optional
        Label for y-axis (default: 'Pellet Count per 4 hours')
    """
    # Make a copy of the DataFrame to avoid modifying the original
    data_copy = data.copy()

    # Convert time column to datetime and set as index
    data_copy[time_col] = pd.to_datetime(data_copy[time_col], format='%m/%d/%Y %H:%M:%S')
    data_copy.set_index(time_col, inplace=True)

    # Resample to 4-hour bins and calculate differences
    four_hour_bins = data_copy[pellet_col].resample('4H').last()
    four_hour_bins_diff = four_hour_bins.diff().clip(lower=0)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(four_hour_bins_diff.index, four_hour_bins_diff, marker='o', label='Pellets per 4 Hours')

    plt.title(plot_title)
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    plt.xticks(rotation=45)

    # Set y-ticks
    max_pellets = four_hour_bins_diff.max()
    y_ticks = range(0, int(max_pellets) + 1, max(1, int(max_pellets) // 10))
    plt.yticks(y_ticks)

    # Shading the background for lights-out hours (6 PM to 6 AM)
    for date in four_hour_bins_diff.index.date:
        plt.axvspan(pd.Timestamp(date) + pd.Timedelta(hours=18), 
                    pd.Timestamp(date + pd.Timedelta(days=1)) + pd.Timedelta(hours=6), 
                    color='lightgrey', alpha=0.5)

    plt.tight_layout()
    plt.legend()
    plt.show()

def multi_pellet_plot(dfs, time_col='MM:DD:YYYY hh:mm:ss', pellet_col='Pellet_Count', 
                      labels=None, bin_size='4H', y_scale='linear', y_max=None, 
                      y_ticks_interval=None):
    """
    Plot pellet counts for multiple DataFrames in specified time intervals with shading for lights-out hours.
    
    Parameters:
    -----------
    dfs : list
        List of DataFrames containing FED data
    time_col : str, optional
        Name of the time column (default: 'MM:DD:YYYY hh:mm:ss')
    pellet_col : str, optional
        Name of the pellet count column (default: 'Pellet_Count')
    labels : list, optional
        List of labels for each DataFrame in the plot (default: None)
    bin_size : str, optional
        Time interval for binning data (default: '4H')
    y_scale : str or float, optional
        Y-axis scale type ('linear', 'log') or maximum value (default: 'linear')
    y_max : float, optional
        Maximum value for y-axis (default: None)
    y_ticks_interval : int, optional
        Interval for y-axis ticks (default: None)
    """
    plt.figure(figsize=(10, 6))

    for i, data in enumerate(dfs):
        data_copy = data.copy()

        # Convert time column to datetime and set as index
        data_copy[time_col] = pd.to_datetime(data_copy[time_col], format='%m/%d/%Y %H:%M:%S')
        data_copy.set_index(time_col, inplace=True)

        try:
            # Resample to specified bins and calculate differences
            binned_data = data_copy[pellet_col].resample(bin_size).last()
            binned_data_diff = binned_data.diff().clip(lower=0)

            # Plotting
            plt.plot(
                binned_data_diff.index, binned_data_diff, marker='o',
                label=labels[i] if (labels and i < len(labels)) else f'DataFrame {i+1}'
            )

            # --- Lights-out shading (only if we have binned_data_diff) ---
            for date in binned_data_diff.index.date:
                plt.axvspan(
                    pd.Timestamp(date) + pd.Timedelta(hours=18),
                    pd.Timestamp(date) + pd.Timedelta(days=1, hours=6),
                    color='lightgrey', alpha=0.5
                )

        except Exception as e:
            print(f"Error processing DataFrame {i+1}: {e}")

    plt.title(f'Pellets Per {bin_size} Interval')
    plt.xlabel('Day')
    plt.ylabel(f'Pellets (per {bin_size})')
    plt.xticks(rotation=45)

    # Y-scale logic
    if y_scale == 'log':
        plt.yscale('log')
    elif isinstance(y_scale, (int, float)):
        plt.ylim(bottom=0, top=y_scale)
    elif y_scale != 'linear':
        raise ValueError("Invalid y_scale value. Use 'linear', 'log', or a numeric value.")

    plt.tight_layout()
    plt.legend()
    plt.show()

def multi_poke_plot(dfs, time_col='MM:DD:YYYY hh:mm:ss', poke_col='Left_Poke_Count', labels=None, 
                    bin_size='4H', y_scale='linear', y_max=None, y_ticks_interval=None):
    """
    Plot poke counts for multiple DataFrames in specified time intervals with shading for lights-out hours.
    
    Parameters:
    -----------
    dfs : list
        List of DataFrames containing FED data
    time_col : str, optional
        Name of the time column (default: 'MM:DD:YYYY hh:mm:ss')
    poke_col : str, optional
        Name of the poke count column (default: 'Left_Poke_Count')
    labels : list, optional
        List of labels for each DataFrame in the plot (default: None)
    bin_size : str, optional
        Time interval for binning data (default: '4H')
    y_scale : str or float, optional
        Y-axis scale type ('linear', 'log') or maximum value (default: 'linear')
    y_max : float, optional
        Maximum value for y-axis (default: None)
    y_ticks_interval : int, optional
        Interval for y-axis ticks (default: None)
    """
    plt.figure(figsize=(10, 6))

    for i, data in enumerate(dfs):
        # Create a copy of the DataFrame to avoid modifying the original
        data_copy = data.copy()

        # Convert time column to datetime and set as index
        data_copy[time_col] = pd.to_datetime(data_copy[time_col], format='%m/%d/%Y %H:%M:%S')
        data_copy.set_index(time_col, inplace=True)

        # Resample to specified bins and calculate differences
        try:
            binned_data = data_copy[poke_col].resample(bin_size).last()
            binned_data_diff = binned_data.diff().clip(lower=0)

            # Plotting
            plt.plot(binned_data_diff.index, binned_data_diff, marker='o', 
                     label=labels[i] if labels and i < len(labels) else f'DataFrame {i+1}')
        except Exception as e:
            print(f"Error processing DataFrame {i+1}: {e}")

    plt.title(f'Pokes Per {bin_size} Interval')
    plt.xlabel('Day')
    plt.ylabel(f'Pokes per {bin_size})')
    plt.xticks(rotation=45)

    # Set y-axis scale
    if y_scale == 'log':
        plt.yscale('log')
    elif isinstance(y_scale, (int, float)):
        plt.ylim(bottom=0, top=y_scale)
    elif y_scale != 'linear':
        raise ValueError("Invalid y_scale value. Use 'linear', 'log', or a numeric value.")

    # Shading the background for lights-out hours (6 PM to 6 AM)
    for date in binned_data_diff.index.date:
        plt.axvspan(pd.Timestamp(date) + pd.Timedelta(hours=18), 
                    pd.Timestamp(date + pd.Timedelta(days=1)) + pd.Timedelta(hours=6), 
                    color='lightgrey', alpha=0.5)

    plt.tight_layout()
    plt.legend()
    plt.show()

def multi_pellet_plot_averaged(dfs, time_col='MM:DD:YYYY hh:mm:ss', pellet_col='Pellet_Count', 
                                  bin_size='4H', y_scale='linear', y_max=None, dashed_line_date1=None, 
                                  dashed_line_label1=None, dashed_line_date2=None, dashed_line_label2=None):
    """
    Plot averaged pellet counts for multiple DataFrames with error bars in specified time intervals.
    
    Parameters:
    -----------
    dfs : list
        List of DataFrames containing FED data
    time_col : str, optional
        Name of the time column (default: 'MM:DD:YYYY hh:mm:ss')
    pellet_col : str, optional
        Name of the pellet count column (default: 'Pellet_Count')
    bin_size : str, optional
        Time interval for binning data (default: '4H')
    y_scale : str or float, optional
        Y-axis scale type ('linear', 'log') or maximum value (default: 'linear')
    y_max : float, optional
        Maximum value for y-axis (default: None)
    dashed_line_date1 : str, optional
        Date for first vertical dashed line (default: None)
    dashed_line_label1 : str, optional
        Label for first vertical dashed line (default: None)
    dashed_line_date2 : str, optional
        Date for second vertical dashed line (default: None)
    dashed_line_label2 : str, optional
        Label for second vertical dashed line (default: None)
    """
    plt.figure(figsize=(10, 6))

    # Initialize an empty list to store the binned data from each DataFrame
    all_binned_data = []

    # Process each DataFrame
    for i, data in enumerate(dfs):
        # Create a copy of the DataFrame to avoid modifying the original
        data_copy = data.copy()

        # Convert time column to datetime and set as index
        data_copy[time_col] = pd.to_datetime(data_copy[time_col], format='%m/%d/%Y %H:%M:%S')
        data_copy.set_index(time_col, inplace=True)

        # Resample to specified bins and get the last value of the pellet count for each bin
        binned_data = data_copy[pellet_col].resample(bin_size).last()

        # Calculate the difference to get pellet counts in each interval
        binned_data_diff = binned_data.diff().clip(lower=0)

        # Append the binned data to the list
        all_binned_data.append(binned_data_diff)

    # Combine all binned data into a single DataFrame (columns correspond to each DataFrame)
    combined_binned_data = pd.concat(all_binned_data, axis=1)

    # Calculate the mean across all DataFrames for each time bin
    averaged_data = combined_binned_data.mean(axis=1)

    # Calculate the SEM across all DataFrames for each time bin
    sem_data = combined_binned_data.sem(axis=1)

    # Remove NaN values if they exist in the averaged data or SEM
    averaged_data = averaged_data.dropna()
    sem_data = sem_data.dropna()

    # Plotting the averaged data with SEM error bars
    plt.errorbar(averaged_data.index, averaged_data, yerr=sem_data, fmt='-o', 
                 elinewidth=2, capsize=4)

    # Plot vertical dashed line if a date is provided
    if dashed_line_date1:
        dashed_line_date1 = pd.to_datetime(dashed_line_date1)
        plt.axvline(x=dashed_line_date1, color='black', linestyle='--', label=dashed_line_label1)

    if dashed_line_date2:
        dashed_line_date2 = pd.to_datetime(dashed_line_date2)
        plt.axvline(x=dashed_line_date2, color='blue', linestyle='--', label=dashed_line_label2)

    plt.title(f'Average Pellets Per {bin_size} Interval')
    plt.xlabel('Date')
    plt.ylabel(f'Average Pellets (per {bin_size})')
    plt.xticks(rotation=45)

    # Set y-axis scale
    if y_scale == 'log':
        plt.yscale('log')
    elif isinstance(y_scale, (int, float)):
        plt.ylim(bottom=0, top=y_scale)
    elif y_scale != 'linear':
        raise ValueError("Invalid y_scale value. Use 'linear', 'log', or a numeric value.")

    # Shading the background for lights-out hours (6 PM to 6 AM)
    for date in averaged_data.index.date:
        plt.axvspan(pd.Timestamp(date) + pd.Timedelta(hours=18), 
                    pd.Timestamp(date + pd.Timedelta(days=1)) + pd.Timedelta(hours=6), 
                    color='lightgrey', alpha=0.5)

    plt.tight_layout()
    plt.legend()
    plt.show()

def multi_poke_plot_averaged(dfs, time_col='MM:DD:YYYY hh:mm:ss', poke_col='Left_Poke_Count', 
                             bin_size='4H', y_scale='linear', y_max=None, dashed_line_date1=None, 
                             dashed_line_label1=None, dashed_line_date2=None, dashed_line_label2=None):
    """
    Plot averaged poke counts for multiple DataFrames with error bars in specified time intervals.
    
    Parameters:
    -----------
    dfs : list
        List of DataFrames containing FED data
    time_col : str, optional
        Name of the time column (default: 'MM:DD:YYYY hh:mm:ss')
    poke_col : str, optional
        Name of the poke count column (default: 'Left_Poke_Count')
    bin_size : str, optional
        Time interval for binning data (default: '4H')
    y_scale : str or float, optional
        Y-axis scale type ('linear', 'log') or maximum value (default: 'linear')
    y_max : float, optional
        Maximum value for y-axis (default: None)
    dashed_line_date1 : str, optional
        Date for first vertical dashed line (default: None)
    dashed_line_label1 : str, optional
        Label for first vertical dashed line (default: None)
    dashed_line_date2 : str, optional
        Date for second vertical dashed line (default: None)
    dashed_line_label2 : str, optional
        Label for second vertical dashed line (default: None)
    """
    plt.figure(figsize=(10, 6))

    all_binned_data = []

    for i, data in enumerate(dfs):
        data_copy = data.copy()
        data_copy[time_col] = pd.to_datetime(data_copy[time_col], format='%m/%d/%Y %H:%M:%S')
        data_copy.set_index(time_col, inplace=True)

        binned_data = data_copy[poke_col].resample(bin_size).last()
        binned_data_diff = binned_data.diff().clip(lower=0)

        all_binned_data.append(binned_data_diff)

    combined_binned_data = pd.concat(all_binned_data, axis=1)
    averaged_data = combined_binned_data.mean(axis=1)
    sem_data = combined_binned_data.sem(axis=1)

    averaged_data = averaged_data.dropna()
    sem_data = sem_data.dropna()

    # Plotting the averaged data with SEM error bars
    plt.errorbar(averaged_data.index, averaged_data, yerr=sem_data, fmt='-o', label='Average Pokes', 
                 elinewidth=2, capsize=4)

    if dashed_line_date1:
        dashed_line_date1 = pd.to_datetime(dashed_line_date1)
        plt.axvline(x=dashed_line_date1, color='black', linestyle='--', label=dashed_line_label1)

    if dashed_line_date2:
        dashed_line_date2 = pd.to_datetime(dashed_line_date2)
        plt.axvline(x=dashed_line_date2, color='blue', linestyle='--', label=dashed_line_label2)

    plt.title(f'Average Pokes Per {bin_size} Interval')
    plt.xlabel('Date')
    plt.ylabel(f'Average Pokes per {bin_size}')
    plt.xticks(rotation=45)

    if y_scale == 'log':
        plt.yscale('log')
    elif isinstance(y_scale, (int, float)):
        plt.ylim(bottom=0, top=y_scale)
    elif y_scale != 'linear':
        raise ValueError("Invalid y_scale value. Use 'linear', 'log', or a numeric value.")

    for date in averaged_data.index.date:
        plt.axvspan(pd.Timestamp(date) + pd.Timedelta(hours=18), 
                    pd.Timestamp(date + pd.Timedelta(days=1)) + pd.Timedelta(hours=6), 
                    color='lightgrey', alpha=0.5)

    plt.tight_layout()
    plt.legend()
    plt.show()

def group_pellet_plot(groups, time_col='MM:DD:YYYY hh:mm:ss', pellet_col='Pellet_Count', 
                               group_labels=None, bin_size='4H', y_scale='linear', 
                               y_max=None, y_ticks_interval=None):
    """
    Plot pellet counts for multiple groups of DataFrames with different colors for each group.
    
    Parameters:
    -----------
    groups : list of lists
        List of groups, where each group is a list of DataFrames
    time_col : str, optional
        Name of the time column (default: 'MM:DD:YYYY hh:mm:ss')
    pellet_col : str, optional
        Name of the pellet count column (default: 'Pellet_Count')
    group_labels : list, optional
        List of labels for each group (default: None)
    bin_size : str, optional
        Time interval for binning data (default: '4H')
    y_scale : str or float, optional
        Y-axis scale type ('linear', 'log') or maximum value (default: 'linear')
    y_max : float, optional
        Maximum value for y-axis (default: None)
    y_ticks_interval : int, optional
        Interval for y-axis ticks (default: None)
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))  # Generate a set of colors

    for group_index, group in enumerate(groups):
        # Check for valid group labels
        group_label = group_labels[group_index] if group_labels and group_index < len(group_labels) else f'Group {group_index + 1}'
        
        # Initialize a variable to hold the binned data diff for the legend
        binned_data_diff_group = None

        for i, data in enumerate(group):
            # Create a copy of the DataFrame to avoid modifying the original
            data_copy = data.copy()

            # Convert time column to datetime and set as index
            data_copy[time_col] = pd.to_datetime(data_copy[time_col], format='%m/%d/%Y %H:%M:%S')
            data_copy.set_index(time_col, inplace=True)

            # Resample to specified bins and calculate differences
            try:
                binned_data = data_copy[pellet_col].resample(bin_size).last()
                binned_data_diff = binned_data.diff().clip(lower=0)

                # Plotting
                plt.plot(binned_data_diff.index, binned_data_diff, marker='o', 
                         color=colors[group_index], alpha=0.8)  # Plot without legend label

                # Store the binned data difference for the legend
                if binned_data_diff_group is None:
                    binned_data_diff_group = binned_data_diff
            except Exception as e:
                print(f"Error processing DataFrame {group_index + 1}, DataFrame {i+1}: {e}")

        # Add a single legend entry for the group
        if binned_data_diff_group is not None:
            plt.plot([], [], marker='o', color=colors[group_index], label=group_label)  # Add empty plot for legend

    plt.title(f'Pellets Per {bin_size} Interval')
    plt.xlabel('Day')
    plt.ylabel(f'Pellets per {bin_size}')
    plt.xticks(rotation=45)

    # Set y-axis scale
    if y_scale == 'log':
        plt.yscale('log')
    elif isinstance(y_scale, (int, float)):
        plt.ylim(bottom=0, top=y_scale)
    elif y_scale != 'linear':
        raise ValueError("Invalid y_scale value. Use 'linear', 'log', or a numeric value.")

    # Shading the background for lights-out hours (6 PM to 6 AM)
    for date in pd.date_range(start=data_copy.index.min(), end=data_copy.index.max(), freq='D'):
        plt.axvspan(pd.Timestamp(date) + pd.Timedelta(hours=18), 
                    pd.Timestamp(date + pd.Timedelta(days=1)) + pd.Timedelta(hours=6), 
                    color='lightgrey', alpha=0.5)

    plt.tight_layout()
    plt.legend()
    plt.show()

def group_poke_plot(groups, time_col='MM:DD:YYYY hh:mm:ss', pellet_col='Left_Poke_Count', 
                               group_labels=None, bin_size='4H', y_scale='linear', 
                               y_max=None, y_ticks_interval=None):
    """
    Plot poke counts for multiple groups of DataFrames with different colors for each group.
    
    Parameters:
    -----------
    groups : list of lists
        List of groups, where each group is a list of DataFrames
    time_col : str, optional
        Name of the time column (default: 'MM:DD:YYYY hh:mm:ss')
    pellet_col : str, optional
        Name of the poke count column (default: 'Left_Poke_Count')
    group_labels : list, optional
        List of labels for each group (default: None)
    bin_size : str, optional
        Time interval for binning data (default: '4H')
    y_scale : str or float, optional
        Y-axis scale type ('linear', 'log') or maximum value (default: 'linear')
    y_max : float, optional
        Maximum value for y-axis (default: None)
    y_ticks_interval : int, optional
        Interval for y-axis ticks (default: None)
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))  # Generate a set of colors

    for group_index, group in enumerate(groups):
        # Check for valid group labels
        group_label = group_labels[group_index] if group_labels and group_index < len(group_labels) else f'Group {group_index + 1}'
        
        # Initialize a variable to hold the binned data diff for the legend
        binned_data_diff_group = None

        for i, data in enumerate(group):
            # Create a copy of the DataFrame to avoid modifying the original
            data_copy = data.copy()

            # Convert time column to datetime and set as index
            data_copy[time_col] = pd.to_datetime(data_copy[time_col], format='%m/%d/%Y %H:%M:%S')
            data_copy.set_index(time_col, inplace=True)

            # Resample to specified bins and calculate differences
            try:
                binned_data = data_copy[pellet_col].resample(bin_size).last()
                binned_data_diff = binned_data.diff().clip(lower=0)

                # Plotting
                plt.plot(binned_data_diff.index, binned_data_diff, marker='o', 
                         color=colors[group_index], alpha=0.8)  # Plot without legend label

                # Store the binned data difference for the legend
                if binned_data_diff_group is None:
                    binned_data_diff_group = binned_data_diff
            except Exception as e:
                print(f"Error processing DataFrame {group_index + 1}, DataFrame {i+1}: {e}")

        # Add a single legend entry for the group
        if binned_data_diff_group is not None:
            plt.plot([], [], marker='o', color=colors[group_index], label=group_label)  # Add empty plot for legend

    plt.title(f'Pokes Per {bin_size} Interval')
    plt.xlabel('Day')
    plt.ylabel(f'Pokes per {bin_size}')
    plt.xticks(rotation=45)

    # Set y-axis scale
    if y_scale == 'log':
        plt.yscale('log')
    elif isinstance(y_scale, (int, float)):
        plt.ylim(bottom=0, top=y_scale)
    elif y_scale != 'linear':
        raise ValueError("Invalid y_scale value. Use 'linear', 'log', or a numeric value.")

    # Shading the background for lights-out hours (6 PM to 6 AM)
    for date in pd.date_range(start=data_copy.index.min(), end=data_copy.index.max(), freq='D'):
        plt.axvspan(pd.Timestamp(date) + pd.Timedelta(hours=18), 
                    pd.Timestamp(date + pd.Timedelta(days=1)) + pd.Timedelta(hours=6), 
                    color='lightgrey', alpha=0.5)

    plt.tight_layout()
    plt.legend()
    plt.show()

def group_poke_averaged(
    groups,
    time_col='MM:DD:YYYY hh:mm:ss',
    poke_col='Left_Poke_Count',
    bin_size='1D',
    y_scale='linear',
    y_max=None,
    dashed_line_date1=None,
    dashed_line_label1=None,
    dashed_line_date2=None,
    dashed_line_label2=None,
    dashed_line_date3=None,
    dashed_line_label3=None,
    dashed_line_date4=None,
    dashed_line_label4=None,
    show_sem='shaded',
    plot_mode='average',
    plot_title='Average Pokes per {bin_size}',
    plot_xlabel='Day',
    plot_ylabel='Average Pokes',
    group_colors=None,
    x_date_range=None,
    x_labels=None,
    tick_interval=1,
    baseline=None,
    date_range_shaded=None,
    shaded_label=None
):
    """
    Plot averaged poke counts for multiple groups of DataFrames with error bars.
    
    Parameters:
    -----------
    groups : dict
        Dictionary mapping group labels to lists of DataFrames
    time_col : str, optional
        Name of the time column (default: 'MM:DD:YYYY hh:mm:ss')
    poke_col : str, optional
        Name of the poke count column (default: 'Left_Poke_Count')
    bin_size : str, optional
        Time interval for binning data (default: '1D')
    y_scale : str or float, optional
        Y-axis scale type ('linear', 'log') or maximum value (default: 'linear')
    y_max : float, optional
        Maximum value for y-axis (default: None)
    dashed_line_date1, dashed_line_date2, dashed_line_date3, dashed_line_date4 : str or list, optional
        Date(s) for vertical dashed lines (default: None)
    dashed_line_label1, dashed_line_label2, dashed_line_label3, dashed_line_label4 : str, optional
        Label(s) for vertical dashed lines (default: None)
    show_sem : str, optional
        How to display standard error of the mean ('shaded', 'error_bars', or None) (default: 'shaded')
    plot_mode : str, optional
        Plot mode ('average' or 'individual') (default: 'average')
    plot_title : str, optional
        Title for the plot (default: 'Average Pokes per {bin_size}')
    plot_xlabel : str, optional
        Label for x-axis (default: 'Day')
    plot_ylabel : str, optional
        Label for y-axis (default: 'Average Pokes')
    group_colors : dict, optional
        Dictionary mapping group labels to colors (default: None)
    x_date_range : list, optional
        List of [start_date, end_date] for x-axis limits (default: None)
    x_labels : list, optional
        List of labels for x-axis ticks (default: None)
    tick_interval : int, optional
        Interval for x-axis tick labels (default: 1)
    baseline : str or list, optional
        Baseline date(s) for normalization (default: None)
    date_range_shaded : list, optional
        List of [start_date, end_date] pairs for shaded regions (default: None)
    shaded_label : str, optional
        Label for shaded regions (default: None)
    """
    plt.figure(figsize=(10, 6))
    all_binned_data = {}
    baseline_used = False  # Track if baseline normalization is applied

    # Update y-axis label if plot_mode is 'individual'
    if plot_mode == 'individual':
        plot_ylabel = "Pokes"

    # Process each group
    for group_label, group in groups.items():
        color = (group_colors[group_label] if group_colors and group_label in group_colors
                 else plt.cm.viridis(len(all_binned_data) / len(groups)))
        legend_added = False  # for individual plotting

        # Process each individual's DataFrame in the group
        for data in group:
            try:
                # Copy and convert time column
                data_copy = data.copy()
                data_copy[time_col] = pd.to_datetime(data_copy[time_col], format='%m/%d/%Y %H:%M:%S', errors='coerce')
                data_copy.set_index(time_col, inplace=True)

                # Resample to get the last poke count in each bin and compute the difference
                binned_data = data_copy[poke_col].resample(bin_size).last()
                binned_diff = binned_data.diff().clip(lower=0)
                binned_diff = pd.to_numeric(binned_diff, errors='coerce').dropna()

                # --- Baseline normalization ---
                if baseline:
                    baseline_used = True
                    if isinstance(baseline, (list, tuple)):
                        baseline_dates = pd.to_datetime(list(baseline), format='%m/%d/%y', errors='coerce')
                    else:
                        baseline_dates = pd.to_datetime([baseline], format='%m/%d/%y', errors='coerce')
                    baseline_dates = baseline_dates.dropna()
                    bp_dates = pd.Index(binned_diff.index.date)
                    baseline_values = binned_diff[bp_dates.isin(baseline_dates.date)]
                    if not baseline_values.empty:
                        baseline_value = baseline_values.mean()
                    else:
                        try:
                            loc = binned_diff.index.get_loc(baseline_dates.iloc[0], method='nearest')
                            baseline_value = binned_diff.iloc[loc]
                        except Exception:
                            baseline_value = np.nan
                    if pd.notnull(baseline_value) and baseline_value > 0:
                        binned_diff = (binned_diff / baseline_value) * 100

                # --- Plot individual data or accumulate for averaging ---
                if plot_mode == 'individual':
                    label = group_label if not legend_added else "_nolegend_"
                    plt.plot(binned_diff.index, binned_diff, '-o', color=color, markersize=4, label=label)
                    legend_added = True
                else:
                    if group_label not in all_binned_data:
                        all_binned_data[group_label] = []
                    all_binned_data[group_label].append(binned_diff)

            except Exception as e:
                print(f"Error processing individual in Group '{group_label}': {e}")

        # In average mode, combine individual series, average them, and compute SEM
        if plot_mode == 'average' and group_label in all_binned_data:
            combined_data = pd.concat(all_binned_data[group_label], axis=1)
            combined_data = combined_data.apply(pd.to_numeric, errors='coerce').dropna(how='all')
            averaged_data = combined_data.mean(axis=1)
            sem_data = combined_data.sem(axis=1)
            valid_mask = averaged_data.notna() & sem_data.notna()
            averaged_data = averaged_data[valid_mask]
            sem_data = sem_data[valid_mask]
            if averaged_data.empty or sem_data.empty:
                print(f"Skipping {group_label}: No valid data after cleaning.")
                continue

            plt.plot(averaged_data.index, averaged_data, '-o', color=color, label=group_label)
            if show_sem == 'shaded':
                plt.fill_between(averaged_data.index,
                                 averaged_data - sem_data,
                                 averaged_data + sem_data,
                                 color=color, alpha=0.2)
            elif show_sem == 'error_bars':
                plt.errorbar(averaged_data.index, averaged_data,
                             yerr=sem_data, fmt='o', color=color, elinewidth=2, capsize=4)

    # --- Handle dashed vertical lines ---
    def add_dashed_lines(dates, label, line_color):
        if dates:
            for idx, date_str in enumerate(dates):
                date = pd.to_datetime(date_str, format='%m/%d/%y', errors='coerce')
                if pd.notnull(date):
                    plt.axvline(date, color=line_color, linestyle='--', label=label if idx == 0 else None)

    add_dashed_lines(dashed_line_date1, dashed_line_label1, 'black')
    add_dashed_lines(dashed_line_date2, dashed_line_label2, 'blue')
    add_dashed_lines(dashed_line_date3, dashed_line_label3, 'red')
    add_dashed_lines(dashed_line_date4, dashed_line_label4, 'green')

    # --- Add shaded date ranges (with one legend entry) ---
    if date_range_shaded:
        for i, (start, end) in enumerate(date_range_shaded):
            start = pd.to_datetime(start, format='%m/%d/%y', errors='coerce')
            end = pd.to_datetime(end, format='%m/%d/%y', errors='coerce')
            if pd.notnull(start) and pd.notnull(end):
                plt.axvspan(start, end, color='lightgrey', alpha=0.5, label=shaded_label if i == 0 else None)

    # --- Set x-axis limits and labels ---
    if x_date_range:
        plt.xlim(pd.to_datetime(x_date_range[0], format='%m/%d/%y', errors='coerce'),
                 pd.to_datetime(x_date_range[1], format='%m/%d/%y', errors='coerce'))
    if x_labels and x_date_range:
        start_date = pd.to_datetime(x_date_range[0], format='%m/%d/%y', errors='coerce')
        end_date = pd.to_datetime(x_date_range[1], format='%m/%d/%y', errors='coerce')
        all_dates = pd.date_range(start=start_date, end=end_date, periods=len(x_labels))
        selected_dates = all_dates[::tick_interval]
        plt.xticks(selected_dates, x_labels[::tick_interval])

    # --- Set y-axis scaling ---
    if isinstance(y_scale, (int, float)):
        plt.ylim(0, y_scale)
    elif y_scale == 'log':
        plt.yscale('log')
    elif y_scale == 'linear':
        if y_max is None:
            y_max = plt.gca().get_ylim()[1]
        plt.ylim(0, y_max)

    # --- Final labeling and legend ---
    # Always append (% baseline) if baseline was used
    if baseline_used:
        if plot_ylabel is None:
            if plot_mode == 'individual':
                plot_ylabel = 'Pokes (% baseline)'
            else:
                plot_ylabel = 'Average Pokes (% baseline)'
        elif '(% baseline)' not in plot_ylabel:
            plot_ylabel = plot_ylabel.strip() + ' (% baseline)'
    plt.title(plot_title.format(bin_size=bin_size))
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def group_pellet_averaged(
    groups,
    time_col='MM:DD:YYYY hh:mm:ss',
    pellet_col='Pellet_Count',
    bin_size='1D',
    y_scale='linear',
    y_max=None,
    dashed_line_date1=None,
    dashed_line_label1=None,
    dashed_line_date2=None,
    dashed_line_label2=None,
    dashed_line_date3=None,
    dashed_line_label3=None,
    dashed_line_date4=None,
    dashed_line_label4=None,
    show_sem='shaded',
    plot_mode='average',
    plot_title='Average Pellets per {bin_size}',
    plot_xlabel='Day',
    plot_ylabel='Average Pellets',
    group_colors=None,
    x_date_range=None,
    x_labels=None,
    tick_interval=1,
    baseline=None,
    date_range_shaded=None,
    shaded_label=None
):
    """
    Plot averaged pellet counts for multiple groups of DataFrames with error bars.
    
    Parameters:
    -----------
    groups : dict
        Dictionary mapping group labels to lists of DataFrames
    time_col : str, optional
        Name of the time column (default: 'MM:DD:YYYY hh:mm:ss')
    pellet_col : str, optional
        Name of the pellet count column (default: 'Pellet_Count')
    bin_size : str, optional
        Time interval for binning data (default: '1D')
    y_scale : str or float, optional
        Y-axis scale type ('linear', 'log') or maximum value (default: 'linear')
    y_max : float, optional
        Maximum value for y-axis (default: None)
    dashed_line_date1, dashed_line_date2, dashed_line_date3, dashed_line_date4 : str or list, optional
        Date(s) for vertical dashed lines (default: None)
    dashed_line_label1, dashed_line_label2, dashed_line_label3, dashed_line_label4 : str, optional
        Label(s) for vertical dashed lines (default: None)
    show_sem : str, optional
        How to display standard error of the mean ('shaded', 'error_bars', or None) (default: 'shaded')
    plot_mode : str, optional
        Plot mode ('average' or 'individual') (default: 'average')
    plot_title : str, optional
        Title for the plot (default: 'Average Pellets per {bin_size}')
    plot_xlabel : str, optional
        Label for x-axis (default: 'Day')
    plot_ylabel : str, optional
        Label for y-axis (default: 'Average Pellets')
    group_colors : dict, optional
        Dictionary mapping group labels to colors (default: None)
    x_date_range : list, optional
        List of [start_date, end_date] for x-axis limits (default: None)
    x_labels : list, optional
        List of labels for x-axis ticks (default: None)
    tick_interval : int, optional
        Interval for x-axis tick labels (default: 1)
    baseline : str or list, optional
        Baseline date(s) for normalization (default: None)
    date_range_shaded : list, optional
        List of [start_date, end_date] pairs for shaded regions (default: None)
    shaded_label : str, optional
        Label for shaded regions (default: None)
    """
    plt.figure(figsize=(10, 6))
    all_binned_data = {}
    baseline_used = False  # Track if baseline normalization is applied

    # Update y-axis label based on plot_mode
    if plot_mode == 'individual':
        plot_ylabel = "Pellets Collected"

    for group_label, group in groups.items():
        color = (
            group_colors[group_label] if group_colors and group_label in group_colors
            else plt.cm.viridis(len(all_binned_data) / len(groups))
        )

        legend_added = False  # Ensure one legend entry per group

        for data in group:
            try:
                # Convert time column and set index
                data_copy = data.copy()
                data_copy[time_col] = pd.to_datetime(data_copy[time_col], format='%m/%d/%Y %H:%M:%S', errors='coerce')
                data_copy.set_index(time_col, inplace=True)

                # Bin the data, forward‐fill missing cumulative values, and compute differences (zeros on no‐event days)
                binned_data = (
                    data_copy[pellet_col]
                    .resample(bin_size)
                    .last()
                    .ffill()     # carry forward last known cumulative total
                    .fillna(0)  # initial NA → zero
                )
                binned_diff = binned_data.diff().clip(lower=0)
                # Ensure numeric dtype (retain zeros)
                binned_diff = pd.to_numeric(binned_diff, errors='coerce')

                # --- Baseline normalization ---
                if baseline:
                    baseline_used = True
                    # If baseline is a list/tuple, convert each element; else, wrap in list.
                    if isinstance(baseline, (list, tuple)):
                        baseline_dates = pd.to_datetime(list(baseline), format='%m/%d/%y', errors='coerce')
                    else:
                        baseline_dates = pd.to_datetime([baseline], format='%m/%d/%y', errors='coerce')
                    baseline_dates = baseline_dates.dropna()
                    # Convert the index of binned_diff to a pandas Index of dates
                    bp_dates = pd.Index(binned_diff.index.date)
                    # Use .isin() on the pandas Index to select values on baseline date(s)
                    baseline_values = binned_diff[bp_dates.isin(baseline_dates.date)]
                    if not baseline_values.empty:
                        baseline_value = baseline_values.mean()
                    else:
                        # Fallback: try to get the nearest value to the first baseline date
                        try:
                            loc = binned_diff.index.get_loc(baseline_dates.iloc[0], method='nearest')
                            baseline_value = binned_diff.iloc[loc]
                        except Exception:
                            baseline_value = np.nan
                    if pd.notnull(baseline_value) and baseline_value > 0:
                        binned_diff = (binned_diff / baseline_value) * 100
                        plot_ylabel = 'Pellets (% baseline)'

                # --- Plotting logic (individual vs average) ---
                if plot_mode == 'individual':
                    label = group_label if not legend_added else "_nolegend_"
                    plt.plot(binned_diff.index, binned_diff, '-o', color=color, markersize=4, label=label)
                    legend_added = True  
                else:
                    if group_label not in all_binned_data:
                        all_binned_data[group_label] = []
                    all_binned_data[group_label].append(binned_diff)

            except Exception as e:
                print(f"Error processing individual in Group '{group_label}': {e}")

        if plot_mode == 'average' and group_label in all_binned_data:
            combined_data = pd.concat(all_binned_data[group_label], axis=1)
            combined_data = combined_data.apply(pd.to_numeric, errors='coerce').dropna(how='all')

            averaged_data = combined_data.mean(axis=1)
            sem_data = combined_data.sem(axis=1)

            valid_mask = averaged_data.notna() & sem_data.notna()
            averaged_data = averaged_data[valid_mask]
            sem_data = sem_data[valid_mask]

            if averaged_data.empty or sem_data.empty:
                print(f"Skipping {group_label}: No valid data after cleaning.")
                continue

            plt.plot(averaged_data.index, averaged_data, '-o', color=color, label=group_label)

            if show_sem == 'shaded':
                plt.fill_between(averaged_data.index, 
                                 averaged_data - sem_data, 
                                 averaged_data + sem_data,
                                 color=color, alpha=0.2)
            elif show_sem == 'error_bars':
                plt.errorbar(averaged_data.index, averaged_data,
                             yerr=sem_data, fmt='o', color=color, elinewidth=2, capsize=4)

    # --- Handle dashed vertical lines ---
    def add_dashed_lines(dates, label, line_color):
        if dates:
            for idx, date_str in enumerate(dates):
                date = pd.to_datetime(date_str, format='%m/%d/%y', errors='coerce')
                if pd.notnull(date):
                    plt.axvline(date, color=line_color, linestyle='--', label=label if idx == 0 else None)

    add_dashed_lines(dashed_line_date1, dashed_line_label1, 'black')
    add_dashed_lines(dashed_line_date2, dashed_line_label2, 'blue')
    add_dashed_lines(dashed_line_date3, dashed_line_label3, 'red')
    add_dashed_lines(dashed_line_date4, dashed_line_label4, 'green')

    # --- Add shaded date ranges (with one legend entry) ---
    if date_range_shaded:
        for i, (start, end) in enumerate(date_range_shaded):
            start = pd.to_datetime(start, format='%m/%d/%y', errors='coerce')
            end = pd.to_datetime(end, format='%m/%d/%y', errors='coerce')
            if pd.notnull(start) and pd.notnull(end):
                plt.axvspan(start, end, color='lightgrey', alpha=0.5, label=shaded_label if i==0 else None)

    # --- Set x-axis limits and labels ---
    if x_date_range:
        plt.xlim(pd.to_datetime(x_date_range[0], format='%m/%d/%y', errors='coerce'),
                 pd.to_datetime(x_date_range[1], format='%m/%d/%y', errors='coerce'))
    if x_labels and x_date_range:
        start_date = pd.to_datetime(x_date_range[0], format='%m/%d/%y', errors='coerce')
        end_date = pd.to_datetime(x_date_range[1], format='%m/%d/%y', errors='coerce')
        all_dates = pd.date_range(start=start_date, end=end_date, periods=len(x_labels))
        selected_dates = all_dates[::tick_interval]
        plt.xticks(selected_dates, x_labels[::tick_interval])

    # --- Adjust y-axis ---
    if isinstance(y_scale, (int, float)):
        plt.ylim(0, y_scale)
    elif y_scale == 'log':
        plt.yscale('log')
    elif y_scale == 'linear':
        if y_max is None:
            y_max = plt.gca().get_ylim()[1]
        plt.ylim(0, y_max)

    # --- Final labeling and legend ---
    # Always append (% baseline) if baseline was used
    if baseline_used:
        if plot_ylabel is None:
            if plot_mode == 'individual':
                plot_ylabel = 'Pellets (% baseline)'
            else:
                plot_ylabel = 'Average Pellets (% baseline)'
        elif '(% baseline)' not in plot_ylabel:
            plot_ylabel = plot_ylabel.strip() + ' (% baseline)'
    plt.title(plot_title.format(bin_size=bin_size))
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def group_breakpoint_plot(
    groups,
    percentile=None,
    time_col='MM:DD:YYYY hh:mm:ss',
    fr_col='FR',
    bin_size='1D',
    baseline=None,
    show_sem='shaded',
    group_colors=None,
    x_date_range=None,
    x_labels=None,
    tick_interval=1,
    dashed_line_date1=None,
    dashed_line_label1=None,
    dashed_line_date2=None,
    dashed_line_label2=None,
    date_range_shaded=None,
    shaded_label=None,
    y_scale='linear',
    y_max=None,
    plot_title=None,
    plot_xlabel='Day',
    plot_ylabel=None
):
    """
    Plot per-day breakpoint statistic (max or percentile) for each animal, baseline-normalize, then average across groups.

    Parameters:
    -----------
    groups : dict
        Dictionary mapping group labels to lists of DataFrames
    percentile : int or None, optional
        If int, compute this percentile of breakpoints per bin. If None, use the max breakpoint per bin. (default: None)
    time_col : str, optional
        Name of the time column (default: 'MM:DD:YYYY hh:mm:ss')
    fr_col : str, optional
        Name of the FR column (default: 'FR')
    bin_size : str, optional
        Time interval for binning data (default: '1D')
    baseline : str or list, optional
        Baseline date(s) for normalization (default: None)
    show_sem : str, optional
        How to display standard error of the mean ('shaded', 'error_bars', or None) (default: 'shaded')
    group_colors : dict, optional
        Dictionary mapping group labels to colors (default: None)
    x_date_range : list, optional
        List of [start_date, end_date] for x-axis limits (default: None)
    x_labels : list, optional
        List of labels for x-axis ticks (default: None)
    tick_interval : int, optional
        Interval for x-axis tick labels (default: 1)
    dashed_line_date1, dashed_line_date2 : str or list, optional
        Date(s) for vertical dashed lines (default: None)
    dashed_line_label1, dashed_line_label2 : str, optional
        Label(s) for vertical dashed lines (default: None)
    date_range_shaded : list, optional
        List of [start_date, end_date] pairs for shaded regions (default: None)
    shaded_label : str, optional
        Label for shaded regions (default: None)
    y_scale : str or float, optional
        Y-axis scale type ('linear', 'log') or maximum value (default: 'linear')
    y_max : float, optional
        Maximum value for y-axis (default: None)
    plot_title : str, optional
        Title for the plot (default: None)
    plot_xlabel : str, optional
        Label for x-axis (default: 'Day')
    plot_ylabel : str, optional
        Label for y-axis (default: None)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    group_stats = {}
    baseline_used = False  # Track if baseline normalization is applied

    for group_label, dfs in groups.items():
        color = (
            group_colors.get(group_label)
            if group_colors and group_label in group_colors
            else plt.cm.viridis(len(group_stats) / len(groups))
        )
        individual_series = []

        for df in dfs:
            df_copy = df.copy()
            df_copy[time_col] = pd.to_datetime(
                df_copy[time_col], format='%m/%d/%Y %H:%M:%S', errors='coerce'
            )
            df_copy.sort_values(by=time_col, inplace=True)
            df_copy.set_index(time_col, inplace=True)

            # Extract breakpoints via FR resets
            max_fr = 0
            vals, times = [], []
            fr_vals = df_copy[fr_col].fillna(method='ffill')
            for i in range(len(fr_vals)):
                fr = fr_vals.iat[i]
                t = fr_vals.index[i]
                max_fr = max(max_fr, fr)
                if i + 1 < len(fr_vals) and fr_vals.iat[i+1] == 1:
                    vals.append(max_fr)
                    times.append(t)
                    max_fr = 0
            if not vals:
                continue

            bp_df = pd.DataFrame({'breakpoint': vals}, index=pd.to_datetime(times))

            # Compute daily statistic: max or percentile
            if percentile is None:
                daily_stat = bp_df['breakpoint'].resample(bin_size).max()
            else:
                def calc_pct(x):
                    return np.nan if len(x) == 0 else np.percentile(x, percentile)
                daily_stat = bp_df['breakpoint'].resample(bin_size).apply(calc_pct)

            # Baseline normalization per animal
            if baseline:
                baseline_used = True
                if isinstance(baseline, (list, tuple)):
                    baseline_dates = pd.to_datetime(
                        list(baseline), format='%m/%d/%y', errors='coerce'
                    )
                else:
                    baseline_dates = pd.to_datetime(
                        [baseline], format='%m/%d/%y', errors='coerce'
                    )
                baseline_dates = baseline_dates.dropna()
                bp_dates = pd.Index(daily_stat.index.date)
                base_vals = daily_stat[bp_dates.isin(baseline_dates.date)]
                if not base_vals.empty:
                    base_mean = base_vals.mean()
                    if base_mean > 0:
                        daily_stat = (daily_stat / base_mean) * 100

            individual_series.append(daily_stat)

        if not individual_series:
            continue

        combined = pd.concat(individual_series, axis=1)
        mean_series = combined.mean(axis=1)
        sem_series = combined.sem(axis=1)
        group_stats[group_label] = (mean_series, sem_series, color)

    # Plot group means + SEM
    for label, (mean_s, sem_s, color) in group_stats.items():
        plt.plot(mean_s.index, mean_s, '-o', color=color, label=label)
        if show_sem == 'shaded':
            plt.fill_between(mean_s.index, mean_s - sem_s, mean_s + sem_s, color=color, alpha=0.2)
        elif show_sem == 'error_bars':
            plt.errorbar(mean_s.index, mean_s, yerr=sem_s, fmt='o', color=color, capsize=4)

    # Add dashed lines
    def add_lines(dates, lab, col):
        if dates:
            for i, d in enumerate(dates):
                dt = pd.to_datetime(d, format='%m/%d/%y', errors='coerce')
                if pd.notnull(dt):
                    plt.axvline(dt, color=col, linestyle='--', label=lab if i == 0 else None)

    add_lines(dashed_line_date1, dashed_line_label1, 'black')
    add_lines(dashed_line_date2, dashed_line_label2, 'blue')

    # Shaded date ranges
    if date_range_shaded:
        for i, (start, end) in enumerate(date_range_shaded):
            st = pd.to_datetime(start, format='%m/%d/%y', errors='coerce')
            en = pd.to_datetime(end, format='%m/%d/%y', errors='coerce')
            if pd.notnull(st) and pd.notnull(en):
                plt.axvspan(st, en, color='lightgrey', alpha=0.5, label=shaded_label if i == 0 else None)

    # X-axis formatting
    if x_date_range:
        plt.xlim(
            pd.to_datetime(x_date_range[0], format='%m/%d/%y', errors='coerce'),
            pd.to_datetime(x_date_range[1], format='%m/%d/%y', errors='coerce')
        )
    if x_labels and x_date_range:
        sd = pd.to_datetime(x_date_range[0], format='%m/%d/%y', errors='coerce')
        ed = pd.to_datetime(x_date_range[1], format='%m/%d/%y', errors='coerce')
        dates = pd.date_range(start=sd, end=ed, periods=len(x_labels))
        ticks = dates[::tick_interval]
        labs = x_labels[::tick_interval]
        plt.xticks(ticks, labs)

    # Y-axis scaling
    if isinstance(y_scale, (int, float)):
        plt.ylim(0, y_scale)
    elif y_scale == 'log':
        plt.yscale('log')
    else:
        if y_max is None:
            y_max = plt.gca().get_ylim()[1]
        plt.ylim(0, y_max)

    # Labels and title
    if plot_title is None:
        if percentile is None:
            plot_title = f"Max Breakpoint per {bin_size}"
        else:
            plot_title = f"{percentile}th Percentile Breakpoint per {bin_size}"
    plt.title(plot_title)
    plt.xlabel(plot_xlabel)
    # Always append (% baseline) if baseline was used
    if baseline_used:
    if plot_ylabel is None:
        if percentile is None:
                plot_ylabel = 'Max Breakpoint (% baseline)'
        else:
                plot_ylabel = f'{percentile}th Percentile Breakpoint (% baseline)'
        elif '(% baseline)' not in plot_ylabel:
            plot_ylabel = plot_ylabel.strip() + ' (% baseline)'
    else:
        if plot_ylabel is None:
            if percentile is None:
                plot_ylabel = 'Max Breakpoint'
            else:
                plot_ylabel = f'{percentile}th Percentile Breakpoint'
    plt.ylabel(plot_ylabel)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def pellet_plot_circadian(data, time_col='MM:DD:YYYY hh:mm:ss', pellet_col='Pellet_Count', date_ranges=None):
    """
    Plot pellet counts by hour of the light cycle, showing circadian patterns.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing FED data
    time_col : str, optional
        Name of the time column (default: 'MM:DD:YYYY hh:mm:ss')
    pellet_col : str, optional
        Name of the pellet count column (default: 'Pellet_Count')
    date_ranges : list of tuples, optional
        List of (start_date, end_date) tuples to analyze separately (default: None)
        If None, uses the full date range in the data
    """
    # Copy data and prepare date and time columns
    data_copy = data.copy()
    data_copy[time_col] = pd.to_datetime(data_copy[time_col], format='%m/%d/%Y %H:%M:%S')
    data_copy['Date'] = data_copy[time_col].dt.date
    data_copy['Hour'] = data_copy[time_col].dt.hour
    data_copy['Hour_Shifted'] = (data_copy['Hour'] - 6) % 24
    data_copy[pellet_col] = data_copy[pellet_col].diff().fillna(0)
    
    # Set default date range to full range if none is provided
    if date_ranges is None:
        start_date = data_copy['Date'].min()
        end_date = data_copy['Date'].max()
        date_ranges = [(start_date, end_date)]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(date_ranges)))  # Color palette for multiple ranges
    
    for idx, (start_date, end_date) in enumerate(date_ranges):
        # Ensure date objects for consistency
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()
        
        # Filter data for the current date range
        range_data = data_copy[(data_copy['Date'] >= start_date) & (data_copy['Date'] <= end_date)]
        
        # Group data by Date and Hour_Shifted to sum pellet counts
        hourly_data = range_data.groupby(['Date', 'Hour_Shifted'])[pellet_col].sum().reset_index()
        
        # Ensure every hour is represented, filling missing values with zero
        full_range = pd.DataFrame({'Hour_Shifted': range(24)})
        hourly_data = full_range.merge(hourly_data, on='Hour_Shifted', how='left').fillna(0)
        
        # Calculate mean and SEM for the current date range
        hourly_mean = hourly_data.groupby('Hour_Shifted')[pellet_col].mean()
        hourly_sem = hourly_data.groupby('Hour_Shifted')[pellet_col].sem()
        
        # Adjust values for plotting (wrap the values for circular plot)
        hours = list(range(25))
        mean_values = hourly_mean.tolist() + [hourly_mean.iloc[0]]
        sem_values = hourly_sem.tolist() + [hourly_sem.iloc[0]]
        
        # Plot with error bars for each date range
        plt.errorbar(hours, mean_values, yerr=sem_values, fmt='o-', 
                     color=colors[idx], label=f'{start_date} to {end_date}', capsize=5)
    
    # Add plot title, labels, and legend
    plt.title('Average Pellet Count by Hour of Light Cycle')
    plt.xlabel('Hours Since Light Onset (0 = 6 AM)')
    plt.ylabel('Average Pellet Count')
    plt.xticks(range(0, 25))
    
    # Shade dark period (6 PM - 6 AM)
    plt.axvspan(12, 24, color='lightgrey', alpha=0.5)
    
    plt.tight_layout()
    plt.legend()
    plt.show()

def poke_plot_circadian(data, time_col='MM:DD:YYYY hh:mm:ss', poke_col='Left_Poke_Count', date_ranges=None):
    """
    Plot poke counts by hour of the light cycle, showing circadian patterns.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing FED data
    time_col : str, optional
        Name of the time column (default: 'MM:DD:YYYY hh:mm:ss')
    poke_col : str, optional
        Name of the poke count column (default: 'Left_Poke_Count')
    date_ranges : list of tuples, optional
        List of (start_date, end_date) tuples to analyze separately (default: None)
        If None, uses the full date range in the data
    """
    # Copy data and prepare date and time columns
    data_copy = data.copy()
    data_copy[time_col] = pd.to_datetime(data_copy[time_col], format='%m/%d/%Y %H:%M:%S')
    data_copy['Date'] = data_copy[time_col].dt.date
    data_copy['Hour'] = data_copy[time_col].dt.hour
    data_copy['Hour_Shifted'] = (data_copy['Hour'] - 6) % 24
    data_copy[poke_col] = data_copy[poke_col].diff().fillna(0)
    
    # Set default date range to full range if none is provided
    if date_ranges is None:
        start_date = data_copy['Date'].min()
        end_date = data_copy['Date'].max()
        date_ranges = [(start_date, end_date)]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(date_ranges)))  # Color palette for multiple ranges
    
    for idx, (start_date, end_date) in enumerate(date_ranges):
        # Ensure date objects for consistency
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()
        
        # Filter data for the current date range
        range_data = data_copy[(data_copy['Date'] >= start_date) & (data_copy['Date'] <= end_date)]
        
        # Group data by Date and Hour_Shifted to sum poke counts
        hourly_data = range_data.groupby(['Date', 'Hour_Shifted'])[poke_col].sum().reset_index()
        
        # Ensure every hour is represented, filling missing values with zero
        full_range = pd.DataFrame({'Hour_Shifted': range(24)})
        hourly_data = full_range.merge(hourly_data, on='Hour_Shifted', how='left').fillna(0)
        
        # Calculate mean and SEM for the current date range
        hourly_mean = hourly_data.groupby('Hour_Shifted')[poke_col].mean()
        hourly_sem = hourly_data.groupby('Hour_Shifted')[poke_col].sem()
        
        # Adjust values for plotting (wrap the values for circular plot)
        hours = list(range(25))
        mean_values = hourly_mean.tolist() + [hourly_mean.iloc[0]]
        sem_values = hourly_sem.tolist() + [hourly_sem.iloc[0]]
        
        # Plot with error bars for each date range
        plt.errorbar(hours, mean_values, yerr=sem_values, fmt='o-', 
                     color=colors[idx], label=f'{start_date} to {end_date}', capsize=5)
    
    # Add plot title, labels, and legend
    plt.title('Average Poke Count by Hour of Light Cycle')
    plt.xlabel('Hours Since Light Onset (0 = 6 AM)')
    plt.ylabel('Average Poke Count')
    plt.xticks(range(0, 25))
    
    # Shade dark period (6 PM - 6 AM)
    plt.axvspan(12, 24, color='lightgrey', alpha=0.5)
    
    plt.tight_layout()
    plt.legend()
    plt.show()

def multi_pellet_plot_circadian(dataframes, time_col='MM:DD:YYYY hh:mm:ss', pellet_col='Pellet_Count', date_ranges=None):

    # Define base colors for different dataframes and colormaps for shading
    color_maps = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]  # Vibrant colormaps

    plt.figure(figsize=(10, 6))

    for df_idx, df in enumerate(dataframes):
        data_copy = df.copy()
        data_copy[time_col] = pd.to_datetime(data_copy[time_col], format='%m/%d/%Y %H:%M:%S')
        data_copy['Date'] = data_copy[time_col].dt.date
        data_copy['Hour'] = data_copy[time_col].dt.hour
        data_copy['Hour_Shifted'] = (data_copy['Hour'] - 6) % 24
        data_copy[pellet_col] = data_copy[pellet_col].diff().fillna(0)

        # Set default date range to full range if none is provided
        if date_ranges is None:
            start_date = data_copy['Date'].min()
            end_date = data_copy['Date'].max()
            date_ranges = [(start_date, end_date)]

        for range_idx, (start_date, end_date) in enumerate(date_ranges):
            # Ensure date objects for consistency
            start_date = pd.to_datetime(start_date).date()
            end_date = pd.to_datetime(end_date).date()

            # Filter data for the current date range
            range_data = data_copy[(data_copy['Date'] >= start_date) & (data_copy['Date'] <= end_date)]
    
            # Group data by Date and Hour_Shifted to sum pellet counts
            hourly_data = range_data.groupby(['Date', 'Hour_Shifted'])[pellet_col].sum().reset_index()

            # Ensure every hour is represented, filling missing values with zero
            full_range = pd.DataFrame({'Hour_Shifted': range(24)})
            hourly_data = full_range.merge(hourly_data, on='Hour_Shifted', how='left').fillna(0)

            # Calculate mean and SEM for the current date range
            hourly_mean = hourly_data.groupby('Hour_Shifted')[pellet_col].mean()
            hourly_sem = hourly_data.groupby('Hour_Shifted')[pellet_col].sem()

            # Adjust values for plotting (wrap the values for circular plot)
            hours = list(range(25))
            mean_values = hourly_mean.tolist() + [hourly_mean.iloc[0]]
            sem_values = hourly_sem.tolist() + [hourly_sem.iloc[0]]

            # Select a color from the color map based on range index
            color = color_maps[df_idx](0.7 if len(date_ranges) == 1 else 0.3 + 0.7 * range_idx / (len(date_ranges) - 1))

            # Plot with error bars for each date range in the current dataframe
            plt.errorbar(hours, mean_values, yerr=sem_values, fmt='o-', 
                         color=color, label=f'DF {df_idx + 1} ({start_date} to {end_date})', capsize=5, alpha=1)

    # Add plot title, labels, and legend
    plt.title('Average Pellet Count by Hour of Light Cycle across Multiple Animals')
    plt.xlabel('Hours Since Light Onset (0 = 6 AM)')
    plt.ylabel('Average Pellet Count')
    plt.xticks(range(0, 25))

    # Shade dark period (6 PM - 6 AM)
    plt.axvspan(12, 24, color='lightgrey', alpha=0.5)
    
    plt.tight_layout()
    plt.legend()
    plt.show()

def multi_poke_plot_circadian(dataframes, time_col='MM:DD:YYYY hh:mm:ss', poke_col='Left_Poke_Count', date_ranges=None):

    # Define base colors for different dataframes and colormaps for shading
    color_maps = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]  # Vibrant colormaps

    plt.figure(figsize=(10, 6))

    for df_idx, df in enumerate(dataframes):
        data_copy = df.copy()
        data_copy[time_col] = pd.to_datetime(data_copy[time_col], format='%m/%d/%Y %H:%M:%S')
        data_copy['Date'] = data_copy[time_col].dt.date
        data_copy['Hour'] = data_copy[time_col].dt.hour
        data_copy['Hour_Shifted'] = (data_copy['Hour'] - 6) % 24
        data_copy[poke_col] = data_copy[poke_col].diff().fillna(0)

        # Set default date range to full range if none is provided
        if date_ranges is None:
            start_date = data_copy['Date'].min()
            end_date = data_copy['Date'].max()
            date_ranges = [(start_date, end_date)]

        for range_idx, (start_date, end_date) in enumerate(date_ranges):
            # Ensure date objects for consistency
            start_date = pd.to_datetime(start_date).date()
            end_date = pd.to_datetime(end_date).date()

            # Filter data for the current date range
            range_data = data_copy[(data_copy['Date'] >= start_date) & (data_copy['Date'] <= end_date)]

            # Group data by Date and Hour_Shifted to sum poke counts
            hourly_data = range_data.groupby(['Date', 'Hour_Shifted'])[poke_col].sum().reset_index()

            # Ensure every hour is represented, filling missing values with zero
            full_range = pd.DataFrame({'Hour_Shifted': range(24)})
            hourly_data = full_range.merge(hourly_data, on='Hour_Shifted', how='left').fillna(0)

            # Calculate mean and SEM for the current date range
            hourly_mean = hourly_data.groupby('Hour_Shifted')[poke_col].mean()
            hourly_sem = hourly_data.groupby('Hour_Shifted')[poke_col].sem()

            # Adjust values for plotting (wrap the values for circular plot)
            hours = list(range(25))
            mean_values = hourly_mean.tolist() + [hourly_mean.iloc[0]]
            sem_values = hourly_sem.tolist() + [hourly_sem.iloc[0]]

            # Select a color from the color map based on range index
            color = color_maps[df_idx](0.7 if len(date_ranges) == 1 else 0.3 + 0.7 * range_idx / (len(date_ranges) - 1))

            # Plot with error bars for each date range in the current dataframe
            plt.errorbar(hours, mean_values, yerr=sem_values, fmt='o-', 
                         color=color, label=f'DF {df_idx + 1} ({start_date} to {end_date})', capsize=5, alpha=1)

    # Add plot title, labels, and legend
    plt.title('Average Poke Count by Hour of Light Cycle across Multiple Animals')
    plt.xlabel('Hours Since Light Onset (0 = 6 AM)')
    plt.ylabel('Average Poke Count')
    plt.xticks(range(0, 25))

    # Shade dark period (6 PM - 6 AM)
    plt.axvspan(12, 24, color='lightgrey', alpha=0.5)
    
    plt.tight_layout()
    plt.legend()
    plt.show()

def group_pellet_plot_circadian(
    groups,
    group_time_points,
    time_col='MM:DD:YYYY hh:mm:ss',
    poke_col='Pellet_Count',
    group_colors=None,
    plot_title='Average Pellets by Hour of Light Cycle',
    group_time_points_colors=None,
    group_time_points_labels=None
):

    # If group_time_points is a single list, apply to all groups
    if isinstance(group_time_points, list):
        universal_time_points = group_time_points
        group_time_points = {
            g_label: universal_time_points for g_label in groups.keys()
        }

    # Prepare figure
    plt.figure(figsize=(10, 6))

    # If group_colors is missing or incomplete, create fallback color cycle
    if not group_colors:
        color_cycle = plt.cm.viridis(np.linspace(0, 1, len(groups)))
        group_colors = dict(zip(groups.keys(), color_cycle))

    # Iterate over each group
    for i, (group_label, df_list) in enumerate(groups.items()):
        # If this group has no time points specified, skip
        if group_label not in group_time_points:
            print(f"No time_points found for group '{group_label}'. Skipping.")
            continue

        # For each date string in this group's time points
        for date_str in group_time_points[group_label]:
            target_date = pd.to_datetime(date_str).date()

            # 1) Determine color for this (group, date_str)
            if group_time_points_colors and group_label in group_time_points_colors:
                color = group_time_points_colors[group_label].get(
                    date_str,
                    group_colors.get(group_label, plt.cm.viridis(i / len(groups)))
                )
            else:
                color = group_colors.get(group_label, plt.cm.viridis(i / len(groups)))

            # 2) Determine label for this (group, date_str)
            if group_time_points_labels and group_label in group_time_points_labels:
                legend_label = group_time_points_labels[group_label].get(
                    date_str,
                    f"{group_label} ({date_str})"
                )
            else:
                legend_label = f"{group_label} ({date_str})"

            # Collect hourly sums for each DataFrame (individual) in this group+date
            all_hourly_dfs = []

            for df in df_list:
                # Copy and sort by timestamp so diff() is chronological
                df_copy = df.copy()
                df_copy[time_col] = pd.to_datetime(df_copy[time_col], format='%m/%d/%Y %H:%M:%S')
                df_copy.sort_values(by=time_col, inplace=True)

                # Filter for this specific day
                df_copy['Date'] = df_copy[time_col].dt.date
                df_day = df_copy[df_copy['Date'] == target_date].copy()
                if df_day.empty:
                    continue

                # Compute a diff on poke_col, clip negative
                df_day[poke_col] = df_day[poke_col].diff().fillna(0).clip(lower=0)

                # Determine shifted hour
                df_day['Hour'] = df_day[time_col].dt.hour
                df_day['Hour_Shifted'] = (df_day['Hour'] - 6) % 24

                # Sum the clipped increments for each Hour_Shifted
                hourly_sum = (
                    df_day.groupby('Hour_Shifted')[poke_col]
                    .sum()
                    .reset_index(name='poke_sum')
                )
                if hourly_sum.empty:
                    continue

                # Keep for combining across individuals
                all_hourly_dfs.append(hourly_sum)

            if not all_hourly_dfs:
                print(f"No data found for group '{group_label}' on {date_str}. Skipping this time point.")
                continue

            # Combine all individuals' hourly sums
            combined_data = pd.concat(all_hourly_dfs, axis=0)
            # Now compute mean ± SEM per hour
            group_stats = (
                combined_data
                .groupby('Hour_Shifted')['poke_sum']
                .agg(['mean','sem'])
                .reset_index()
            )

            # Ensure we have hours 0..23
            full_hours = pd.DataFrame({'Hour_Shifted': range(24)})
            group_stats = full_hours.merge(group_stats, on='Hour_Shifted', how='left').fillna(0)
            group_stats.sort_values('Hour_Shifted', inplace=True)

            # Wrap for circular plot (hour 24 = hour 0 again)
            hours = list(range(25))  # 0..24
            mean_vals = list(group_stats['mean']) + [group_stats['mean'].iloc[0]]
            sem_vals  = list(group_stats['sem'])  + [group_stats['sem'].iloc[0]]

            # Plot
            plt.errorbar(
                hours,
                mean_vals,
                yerr=sem_vals,
                fmt='o-',
                color=color,
                capsize=4,
                alpha=1.0,
                label=legend_label
            )

    # Final plot details
    plt.title(plot_title)
    plt.xlabel('Hours Since Light Onset (0 = 6 AM)')
    plt.ylabel('Average Pellet Count')
    plt.xticks(range(0, 25))

    # Shade dark period: from hour 12..24 = 6 PM..6 AM
    plt.axvspan(12, 24, color='lightgrey', alpha=0.5)

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def multi_poke_plot_circadian(dataframes, time_col='MM:DD:YYYY hh:mm:ss', poke_col='Left_Poke_Count', date_ranges=None):

    # Define base colors for different dataframes and colormaps for shading
    color_maps = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]  # Vibrant colormaps

    plt.figure(figsize=(10, 6))

    for df_idx, df in enumerate(dataframes):
        data_copy = df.copy()
        data_copy[time_col] = pd.to_datetime(data_copy[time_col], format='%m/%d/%Y %H:%M:%S')
        data_copy['Date'] = data_copy[time_col].dt.date
        data_copy['Hour'] = data_copy[time_col].dt.hour
        data_copy['Hour_Shifted'] = (data_copy['Hour'] - 6) % 24
        data_copy[poke_col] = data_copy[poke_col].diff().fillna(0)

        # Set default date range to full range if none is provided
        if date_ranges is None:
            start_date = data_copy['Date'].min()
            end_date = data_copy['Date'].max()
            date_ranges = [(start_date, end_date)]

        for range_idx, (start_date, end_date) in enumerate(date_ranges):
            # Ensure date objects for consistency
            start_date = pd.to_datetime(start_date).date()
            end_date = pd.to_datetime(end_date).date()

            # Filter data for the current date range
            range_data = data_copy[(data_copy['Date'] >= start_date) & (data_copy['Date'] <= end_date)]

            # Group data by Date and Hour_Shifted to sum poke counts
            hourly_data = range_data.groupby(['Date', 'Hour_Shifted'])[poke_col].sum().reset_index()

            # Ensure every hour is represented, filling missing values with zero
            full_range = pd.DataFrame({'Hour_Shifted': range(24)})
            hourly_data = full_range.merge(hourly_data, on='Hour_Shifted', how='left').fillna(0)

            # Calculate mean and SEM for the current date range
            hourly_mean = hourly_data.groupby('Hour_Shifted')[poke_col].mean()
            hourly_sem = hourly_data.groupby('Hour_Shifted')[poke_col].sem()

            # Adjust values for plotting (wrap the values for circular plot)
            hours = list(range(25))
            mean_values = hourly_mean.tolist() + [hourly_mean.iloc[0]]
            sem_values = hourly_sem.tolist() + [hourly_sem.iloc[0]]

            # Select a color from the color map based on range index
            color = color_maps[df_idx](0.7 if len(date_ranges) == 1 else 0.3 + 0.7 * range_idx / (len(date_ranges) - 1))

            # Plot with error bars for each date range in the current dataframe
            plt.errorbar(hours, mean_values, yerr=sem_values, fmt='o-', 
                         color=color, label=f'DF {df_idx + 1} ({start_date} to {end_date})', capsize=5, alpha=1)

    # Add plot title, labels, and legend
    plt.title('Average Poke Count by Hour of Light Cycle across Multiple Animals')
    plt.xlabel('Hours Since Light Onset (0 = 6 AM)')
    plt.ylabel('Average Poke Count')
    plt.xticks(range(0, 25))

    # Shade dark period (6 PM - 6 AM)
    plt.axvspan(12, 24, color='lightgrey', alpha=0.5)
    
    plt.tight_layout()
    plt.legend()
    plt.show()

def multi_breakpoint_plot(dfs, time_col='MM:DD:YYYY hh:mm:ss', fr_col='FR', labels=None, 
                          bin_size='4h', y_scale='linear', y_max=None, y_ticks_interval=None):
    
    plt.figure(figsize=(10, 6))

    for i, data in enumerate(dfs):
        # Check if the required columns exist
        if fr_col not in data.columns or time_col not in data.columns:
            print(f"Skipping DataFrame {i+1}: Missing columns.")
            continue
        
        # Create a copy of the DataFrame to avoid modifying the original
        data_copy = data.copy()

        # Convert time column to datetime and set as index
        data_copy[time_col] = pd.to_datetime(data_copy[time_col], format='%m/%d/%Y %H:%M:%S')
        data_copy.set_index(time_col, inplace=True)

        # Extract breakpoints: last FR before reset (FR = 1)
        breakpoints = []
        breakpoint_times = []
        max_fr = 0  # Track max FR before reset

        for j in range(len(data_copy)):
            current_fr = data_copy.iloc[j][fr_col]
            current_time = data_copy.index[j]

            if current_fr > max_fr:
                max_fr = current_fr  # Update max FR in the sequence
            
            # Detect FR reset (back to 1)
            if j < len(data_copy) - 1 and data_copy.iloc[j + 1][fr_col] == 1:
                breakpoints.append(max_fr)
                breakpoint_times.append(current_time)
                max_fr = 0  # Reset tracking

        # Create a DataFrame for the breakpoints and timestamps
        breakpoint_df = pd.DataFrame({'breakpoints': breakpoints}, index=pd.to_datetime(breakpoint_times))

        try:
            # Resample to find max breakpoint within each bin
            binned_breakpoints = breakpoint_df['breakpoints'].resample(bin_size).max()

            # Plotting
            plt.plot(binned_breakpoints.index, binned_breakpoints, marker='o', 
                     label=labels[i] if labels and i < len(labels) else f'DataFrame {i+1}')
        except Exception as e:
            print(f"Error processing DataFrame {i+1}: {e}")

    plt.title(f'Max Breakpoint Per {bin_size} Interval')
    plt.xlabel('Time')
    plt.ylabel(f'Max Breakpoint per {bin_size}')
    plt.xticks(rotation=45)

    # Set y-axis scale
    if y_scale == 'log':
        plt.yscale('log')
    elif isinstance(y_scale, (int, float)):
        plt.ylim(bottom=0, top=y_scale)
    elif y_scale != 'linear':
        raise ValueError("Invalid y_scale value. Use 'linear', 'log', or a numeric value.")

    # Shading the background for lights-out hours (6 PM to 6 AM)
    if not binned_breakpoints.empty:  # Ensure there's data before plotting
        for date in binned_breakpoints.index.date:
            plt.axvspan(pd.Timestamp(date) + pd.Timedelta(hours=18), 
                        pd.Timestamp(date + pd.Timedelta(days=1)) + pd.Timedelta(hours=6), 
                        color='lightgrey', alpha=0.5)

    plt.tight_layout()

class HistogramBreakpointPlot(Scene):
    def __init__(
        self, dataframes, date_ranges, *, hist_colors=None,
        time_col="MM:DD:YYYY hh:mm:ss", event_col="Event",
        bin_size=1, x_max=None, alpha=0.8,
        time_per_hist=1.0, max_active=3,
        y_max=None, **kwargs
    ):
        super().__init__(**kwargs)

        # 1) flatten the date_ranges structure
        if any(isinstance(v, (list, tuple)) for v in date_ranges.values()):
            self.date_to_group = {d: g
                                  for g, ds in date_ranges.items()
                                  for d in (ds if isinstance(ds,(list,tuple)) else [ds])}
            self.date_order  = [d for g in date_ranges
                                for d in (date_ranges[g] if isinstance(date_ranges[g],(list,tuple))
                                          else [date_ranges[g]])]
            self.group_order = list(date_ranges.keys())
        else:
            self.date_to_group = date_ranges.copy()
            self.date_order    = list(date_ranges.keys())
            self.group_order   = []
            for d in self.date_order:
                g = self.date_to_group[d]
                if g not in self.group_order:
                    self.group_order.append(g)

        self.hist_colors  = hist_colors or {}
        self.time_col, self.event_col = time_col, event_col
        # Store bin_size and alpha; x_max and y_max determined after collecting break lengths
        self.bin_size    = bin_size
        self.x_max       = x_max
        self.alpha       = alpha
        self.time_per, self.max_active = time_per_hist, max_active
        self.y_max_param = y_max

        # 2) collect breakpoint lengths for each date
        date_breaks = {d: [] for d in self.date_order}
        for df in dataframes:
            df = df.copy()
            df[self.time_col] = pd.to_datetime(df[self.time_col],
                                               format="%m/%d/%Y %H:%M:%S")
            for d in self.date_order:
                start = pd.to_datetime(d, format="%m/%d/%y")
                end   = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                sub   = df[(df[self.time_col] >= start) & (df[self.time_col] <= end)]

                left = 0
                for _, row in sub.iterrows():
                    if row[self.event_col] == "Left":
                        left += 1
                    elif row[self.event_col] == "Pellet" and left:
                        date_breaks[d].append(left)
                        left = 0

        # 3) determine x_max and y_max from collected break lengths
        all_vals = [v for lst in date_breaks.values() for v in lst]
        if not all_vals:
            raise ValueError("No breakpoints found in data for animation.")
        # fallback x_max to data maximum if not provided
        if self.x_max is None:
            self.x_max = int(max(all_vals))
        else:
            try:
                self.x_max = int(self.x_max)
            except Exception:
                raise ValueError(f"Invalid x_max value: {self.x_max}")
        # determine y-axis max using 5 bins of roughly equal size
        natural_max = max(len(lst) for lst in date_breaks.values())
        self.y_max = self.y_max_param or int(math.ceil(natural_max / 5.0) * 5)
        # store break data
        self.date_breaks = date_breaks

    # ────────────────────────────────────────────────────────────────────
    def construct(self):
        # compute bin_edges upfront
        bin_edges = np.arange(0.5, self.x_max + self.bin_size, self.bin_size)
        self.bin_edges = bin_edges
        # 4) axes
        axes = Axes(
            x_range=(bin_edges[0], bin_edges[-1], self.bin_size),
            y_range=(0, self.y_max, max(1, math.ceil(self.y_max/5))),
            axis_config={"include_tip": False},
            x_axis_config={"include_numbers": False},
            y_axis_config={"include_numbers": False},
        )
        self.add(axes)

        # 5) numeric tick labels (whole numbers only)
        if self.bin_size == 1:
            for x in range(1, self.x_max + 1):
                lbl = Text(str(x), font_size=12)
                lbl.next_to(axes.c2p(x, 0), DOWN, buff=0.1)
                self.add(lbl)
        else:
            starts = np.arange(1, self.x_max + 1, self.bin_size)
            centers = starts + self.bin_size/2 - 0.5
            for i, left in enumerate(starts):
                lbl = Text(f"{left}–{min(left+self.bin_size-1, self.x_max)}", font_size=12)
                x_c = centers[i]
                lbl.next_to(axes.c2p(x_c, 0), DOWN, buff=0.1)
                self.add(lbl)

        for y in range(0, self.y_max + 1, max(1, math.ceil(self.y_max/5))):
            lbl = Text(str(y), font_size=12)
            lbl.next_to(axes.c2p(self.bin_edges[0], y), LEFT, buff=0.1)
            self.add(lbl)

        # 6) title and axis labels
        Text("Breakpoint Frequency Distribution", font_size=24)\
            .next_to(axes, UP, buff=0.35).add_to_back()

        x_lab = Text("Breakpoint Value", font_size=18)
        x_lab.next_to(axes, DOWN, buff=0.45)
        self.add(x_lab)

        y_lab = Text("Frequency", font_size=18).rotate(PI/2)
        y_lab.next_to(axes, LEFT, buff=0.45)
        self.add(y_lab)

        # 7) legend (stacked rows: Saline, Cisplatin, …)
        legend = VGroup()
        for grp in self.group_order:
            if isinstance(self.hist_colors.get(grp), dict) and self.hist_colors[grp]:
                colour_hex = next(iter(self.hist_colors[grp].values()))
            else:
                first_date = next(d for d in self.date_order if self.date_to_group[d] == grp)
                colour_hex = self._pick_color(first_date, 1)[0]

            legend.add(
                VGroup(
                    Square(0.25, fill_color=colour_hex,
                           fill_opacity=self.alpha, stroke_color=colour_hex),
                    Text(grp, font_size=14, weight="BOLD")
                ).arrange(RIGHT, buff=0.12)
            )
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend.next_to(axes, UR, buff=0.05)
        legend.shift(LEFT*0.35 + DOWN*0.25)
        self.add(legend)

        # 8) animated bars
        unit_w = axes.x_axis.unit_size * (self.bin_edges[1]-self.bin_edges[0]) * 0.9
        active = []
        for d in self.date_order:
            counts, _ = np.histogram(self.date_breaks[d], bins=self.bin_edges)
            bars = VGroup()
            for i, c in enumerate(counts):
                if c == 0:
                    continue
                x_c = 0.5 * (self.bin_edges[i] + self.bin_edges[i+1])
                x_px, base_y, _ = axes.c2p(x_c, 0)
                bar_h = axes.y_axis.unit_size * c
                rect  = Rectangle(width=unit_w, height=bar_h)
                colour = self._pick_color(d, 1)[0]
                rect.set_fill(colour, opacity=self.alpha).set_stroke(colour, width=1)
                rect.move_to((x_px, base_y + bar_h / 2, 0))
                bars.add(rect)

            self.play(FadeIn(bars), run_time=self.time_per)
            active.append(bars)
            if len(active) > self.max_active:
                self.play(FadeOut(active.pop(0)), run_time=0.5)

        self.wait(2)

    # helper ---------------------------------------------------------------
    def _pick_color(self, date, n):
        hexcol = (
            self.hist_colors.get(date)
            or self.hist_colors.get(date.replace('/', ''))
            or self.hist_colors.get(self.date_to_group[date], {}).get(date)
            or "#888888"
        )
        return [hexcol] * n


# ─── wrapper callable from a notebook ─────────────────────────────────────
def histogram_breakpoint_plot_manim(
    dataframes, date_ranges, *, hist_colors=None,
    bin_size=1, x_max=None, alpha=0.8,
    time_per_hist=1.0, max_active=3,
    y_max=None, **scene_kwargs
):
    HistogramBreakpointPlot(
        dataframes        = dataframes,
        date_ranges       = date_ranges,
        hist_colors       = hist_colors,
        bin_size          = bin_size,
        x_max             = x_max,
        alpha             = alpha,
        time_per_hist     = time_per_hist,
        max_active        = max_active,
        y_max             = y_max,
        **scene_kwargs
    ).render()

# alias backward compatibility
group_breakpoint_plot = group_breakpoint_plot

# simple histogram of breakpoints by date group
def histogram_breakpoint_plot_simple(
    dataframes,
    date_ranges,
    *,
    hist_colors=None,
    time_col="MM:DD:YYYY hh:mm:ss",
    event_col="Event",
    bin_size=1,
    x_max=None,
    style="histogram",
    alpha=1.0,
    plot_title=None,
    plot_xlabel="Breakpoint Value",
    plot_ylabel=None,
    percentiles=None
):
    """Plot breakpoint distributions for given dates by style (histogram, density, line, CDF)."""
    # 1. normalize date → group mapping
    if any(isinstance(v, (list, tuple)) for v in date_ranges.values()):
        date_to_group = {d: g for g, ds in date_ranges.items() for d in (ds if isinstance(ds,(list,tuple)) else [ds])}
        date_order = [d for g in date_ranges for d in (date_ranges[g] if isinstance(date_ranges[g],(list,tuple)) else [date_ranges[g]])]
    else:
        date_to_group = date_ranges.copy()
        date_order = list(date_ranges.keys())

    # 2. collect breakpoints
    date_breaks = {d: [] for d in date_order}
    for df in dataframes:
        df2 = df.copy()
        df2[time_col] = pd.to_datetime(df2[time_col], format='%m/%d/%Y %H:%M:%S')
        for d in date_order:
            start = pd.to_datetime(d, format='%m/%d/%y')
            end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            sub = df2[(df2[time_col] >= start) & (df2[time_col] <= end)]
            left = 0
            for _, row in sub.iterrows():
                if row.get(event_col) == "Left":
                    left += 1
                elif row.get(event_col) == "Pellet" and left:
                    date_breaks[d].append(left)
                    left = 0

    all_vals = [v for vs in date_breaks.values() for v in vs]
    if not all_vals:
        print("No breakpoints found.")
        return

    max_val = int(x_max) if x_max is not None else int(max(all_vals))
    bin_edges = np.arange(0.5, max_val + bin_size, bin_size)

    # 3. plot setup
    plt.figure(figsize=(10, 6))
    plt.xlim(bin_edges[0], bin_edges[-1])

    if plot_title is None:
        if style == 'histogram':
            plot_title = 'Breakpoint Frequency Distribution'
        elif style == 'cdf':
            plot_title = 'Breakpoint CDF'
        else:
            plot_title = 'Breakpoint Probability Density'
    if plot_ylabel is None:
        if style == 'histogram': plot_ylabel = 'Frequency'
        elif style == 'cdf': plot_ylabel = 'Cumulative Probability'
        else: plot_ylabel = 'Probability Density'

    for i, d in enumerate(date_order):
        bpts = date_breaks[d]
        if not bpts: continue
        colour = None
        if hist_colors:
            c = hist_colors.get(d) or hist_colors.get(d.replace('/',''))
            if c is None: c = hist_colors.get(date_to_group[d],{}).get(d)
            colour = c
        if colour is None:
            colour = plt.cm.viridis(i/len(date_order))
        label = f"{date_to_group[d]} – {d}"
        if style == 'histogram':
            plt.hist(bpts, bins=bin_edges, density=False, alpha=alpha,
                     color=colour, edgecolor=colour, histtype='stepfilled', linewidth=1, label=label)
        elif style == 'density':
            plt.hist(bpts, bins=bin_edges, density=True, alpha=alpha,
                     color=colour, edgecolor=colour, histtype='stepfilled', linewidth=1, label=label)
        elif style == 'line':
            counts, edges = np.histogram(bpts, bins=bin_edges, density=True)
            centers = edges[:-1] + bin_size/2
            plt.plot(centers, counts, '-o', color=colour, alpha=alpha, label=label)
        elif style == 'cdf':
            counts, edges = np.histogram(bpts, bins=bin_edges, density=False)
            cum = np.cumsum(counts)
            cum_prob = cum / cum[-1]
            centers = edges[:-1] + bin_size/2
            plt.step(centers, cum_prob, where='post', color=colour, alpha=alpha, label=label)
        else:
            raise ValueError("Unknown style: use 'histogram','density','line','cdf'.")

    # 4. x-ticks
    if bin_size == 1:
        ticks = np.arange(1, max_val+1)
        plt.xticks(ticks, [str(int(x)) for x in ticks])
    else:
        starts = np.arange(1, max_val+1, bin_size)
        labels = [f"{int(s)}–{int(min(s+bin_size-1,max_val))}" for s in starts]
        centers = starts + bin_size/2 - 0.5
        plt.xticks(centers, labels, rotation=45)

    # 5. optional percentiles lines
    if percentiles is not None:
        grp_data = {g: [] for g in set(date_to_group.values())}
        for d, g in date_to_group.items(): grp_data[g].extend(date_breaks[d])
        if isinstance(percentiles, (list,tuple)):
            cmap = plt.cm.tab10(np.linspace(0,1,len(percentiles)))
            for p,c in zip(percentiles,cmap):
                v = np.percentile(all_vals,p)
                plt.axvline(v,color=c,ls='--',label=f"All {int(p)}th %ile = {v:.1f}")
        else:
            for g,spec in percentiles.items():
                data=grp_data.get(g,[])
                if not data: continue
                if isinstance(spec,(list,tuple)):
                    cmap = plt.cm.tab10(np.linspace(0,1,len(spec)))
                    for p,c in zip(spec,cmap):
                        v = np.percentile(data,p)
                        plt.axvline(v,color=c,ls='--',label=f"{g} {int(p)}th %ile = {v:.1f}")
                else:
                    for p,c in spec.items():
                        v = np.percentile(data,float(p))
                        plt.axvline(v,color=c,ls='--',label=f"{g} {int(p)}th %ile = {v:.1f}")

    # 6. finish
    plt.title(plot_title)
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()