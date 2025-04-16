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

def import_FED(animal_ID, start_date, end_date, cohort, experiment='Cisplatin'):
    """
    Import FED data for a group of animals over a date range.
    
    Parameters:
    -----------
    animal_ID : str or list
        Either a single animal ID (e.g., 'DA1') or a list of animal IDs
        (e.g., ['DA12', 'DA1', 'DA11', 'DA2'])
    start_date : str
        The start date in MMDDYY format (e.g., '083024')
    end_date : str
        The end date in MMDDYY format (e.g., '120224')
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
    
    # Convert dates to datetime objects for date range
    start_dt = datetime.strptime(start_date, '%m%d%y')
    end_dt = datetime.strptime(end_date, '%m%d%y')
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
    
    # Dictionary to store results
    results = {}
    
    # Process each animal
    for animal in animal_ID:
        print(f"\nProcessing animal: {animal}")
        animal_data = []
        
        # Construct the animal's directory path
        animal_dir = os.path.join(base_path, experiment_dir, cohort, animal)
        
        if not os.path.exists(animal_dir):
            print(f"Directory not found: {animal_dir}")
            continue
        
        # Process each date
        for date in date_range:
            date_str = date.strftime('%m%d%y')
            
            # Find all files matching the pattern FEDXXX_MMDDYY_XX.CSV
            matching_files = []
            for file in os.listdir(animal_dir):
                if file.startswith('FED') and file.endswith('.CSV'):
                    # Check if the date part matches
                    if f"_{date_str}_" in file:
                        matching_files.append(os.path.join(animal_dir, file))
            
            if not matching_files:
                print(f"No matching files found for {animal} on {date_str}")
                continue
            
            # Try to read each matching file
            for file_path in matching_files:
                try:
                    print(f"Reading file: {file_path}")
                    # Read CSV file with the first row as header
                    df = pd.read_csv(file_path, header=0, low_memory=False)
                    
                    # Remove any rows with 24: in the timestamp (next day)
                    df = df[~df.iloc[:, 0].astype(str).str.contains(r'24:', na=False)]
                    
                    # Get the timestamp column (first column)
                    timestamps = df.iloc[:, 0].astype(str)
                    
                    # Create a new column for the parsed timestamps
                    df['YYYY-MM-DD HH:MM:SS.sss'] = None
                    
                    # Process each timestamp individually
                    for i, ts in enumerate(timestamps):
                        try:
                            # Split the timestamp by spaces
                            parts = ts.split()
                            
                            # The format is typically "YYYY-MM-DD M/D/YYYY HH:MM:SS"
                            # We want to extract the date and time part
                            if len(parts) >= 3:
                                # Extract the date part (M/D/YYYY)
                                date_part = parts[1]
                                # Extract the time part (HH:MM:SS)
                                time_part = parts[2]
                                
                                # Combine them
                                datetime_str = f"{date_part} {time_part}"
                                
                                # Parse the datetime
                                dt = pd.to_datetime(datetime_str, format='%m/%d/%Y %H:%M:%S')
                                
                                # Format it as desired
                                df.at[i, 'YYYY-MM-DD HH:MM:SS.sss'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                # If we can't parse it, try using pandas' built-in parser
                                dt = pd.to_datetime(ts)
                                df.at[i, 'YYYY-MM-DD HH:MM:SS.sss'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except Exception as e:
                            print(f"Error parsing timestamp '{ts}': {e}")
                            # If we can't parse it, try using pandas' built-in parser
                            try:
                                dt = pd.to_datetime(ts)
                                df.at[i, 'YYYY-MM-DD HH:MM:SS.sss'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                # If all else fails, just use the original timestamp
                                df.at[i, 'YYYY-MM-DD HH:MM:SS.sss'] = ts
                    
                    # Remove rows where we couldn't parse the timestamp
                    df = df.dropna(subset=['YYYY-MM-DD HH:MM:SS.sss'])
                    
                    # Add the animal ID and cohort as columns
                    df['Animal_ID'] = animal
                    df['Cohort'] = cohort
                    
                    # Add to the animal's data
                    animal_data.append(df)
                    
                except Exception as e:
                    print(f"Error reading file {file_path}: {str(e)}")
                    continue
        
        # Combine all data for this animal
        if animal_data:
            # Use animal_cohort as the key, matching the format of import_RW
            results[f"{animal}_{cohort}"] = pd.concat(animal_data, ignore_index=True)
            print(f"Successfully imported data for {animal}")
        else:
            print(f"No data found for {animal}")
    
    return results 