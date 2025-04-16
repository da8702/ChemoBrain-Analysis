import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Union
import h5py
import platform
import glob

def load_mat_file(file_path: str) -> Dict:
    """
    Load a MATLAB .mat file and return its contents as a dictionary.
    
    Parameters:
    -----------
    file_path : str
        Path to the .mat file
        
    Returns:
    --------
    Dict
        Dictionary containing the contents of the .mat file
    """
    try:
        # Try loading with scipy.io first
        return sio.loadmat(file_path)
    except NotImplementedError:
        # If that fails, try loading with h5py
        with h5py.File(file_path, 'r') as f:
            return {key: f[key][()] for key in f.keys()}

def get_session_data(base_path: str) -> pd.DataFrame:
    """
    Recursively find and load all session data from .mat files.
    
    Parameters:
    -----------
    base_path : str
        Base path to start searching for .mat files
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing all session data
    """
    # This is a placeholder - we'll need to customize this based on
    # the actual structure of your .mat files
    pass

def process_session_data(mat_data: Dict) -> pd.DataFrame:
    """
    Process raw MATLAB data into a pandas DataFrame.
    
    Parameters:
    -----------
    mat_data : Dict
        Dictionary containing MATLAB data
        
    Returns:
    --------
    pd.DataFrame
        Processed data in DataFrame format
    """
    # This is a placeholder - we'll need to customize this based on
    # the actual structure of your .mat files
    pass

def plot_session_data(data: pd.DataFrame, 
                     animal_id: str,
                     save_path: str = None) -> None:
    """
    Create plots for session data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Processed session data
    animal_id : str
        ID of the animal being plotted
    save_path : str, optional
        Path to save the plot. If None, plot is displayed.
    """
    # This is a placeholder - we'll need to customize this based on
    # what plots you want to create
    pass

def analyze_foraging_behavior(data: pd.DataFrame) -> Dict:
    """
    Analyze foraging behavior metrics.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Processed session data
        
    Returns:
    --------
    Dict
        Dictionary containing analysis results
    """
    # This is a placeholder - we'll need to customize this based on
    # what analyses you want to perform
    pass

def get_base_path() -> str:
    """
    Get the base path for data based on the operating system.
    
    Returns:
    --------
    str
        Base path for the data directory
    """
    system = platform.system()
    if system == "Darwin":  # macOS
        return "/Volumes/ChemoBrain/ChemoBrain-Analysis/Data/Foraging_Cisplatin"
    elif system == "Windows":
        # Adjust this path for Windows if needed
        return "\\\\path\\to\\windows\\ChemoBrain\\ChemoBrain-Analysis\\Data\\Foraging_Cisplatin"
    else:
        raise OSError(f"Unsupported operating system: {system}")

def import_animal_data(cohort: str, animal_id: Union[str, Tuple[str, str], List[str]], 
                      global_vars: Dict = None) -> Dict[str, Dict]:
    """
    Import data for specified animals and assign to global variables.
    
    Parameters:
    -----------
    cohort : str
        Cohort name (e.g., "Cis3")
    animal_id : Union[str, Tuple[str, str], List[str]]
        Animal ID(s) to import. Can be:
        - Single ID (e.g., "DA76")
        - Range tuple (e.g., ("DA76", "DA86"))
        - List of IDs (e.g., ["DA76", "DA79", "DA83"])
    global_vars : Dict, optional
        Dictionary to store global variables. If None, uses globals()
        
    Returns:
    --------
    Dict[str, Dict]
        Dictionary containing imported data for each animal
    """
    if global_vars is None:
        global_vars = globals()
    
    base_path = get_base_path()
    cohort_path = os.path.join(base_path, cohort)
    
    # Determine which animals to process
    if isinstance(animal_id, str):
        animal_ids = [animal_id]
    elif isinstance(animal_id, tuple) and len(animal_id) == 2:
        start_id, end_id = animal_id
        # Extract numbers from IDs
        start_num = int(''.join(filter(str.isdigit, start_id)))
        end_num = int(''.join(filter(str.isdigit, end_id)))
        animal_ids = [f"DA{num}" for num in range(start_num, end_num + 1)]
    elif isinstance(animal_id, list):
        animal_ids = animal_id
    else:
        raise ValueError("animal_id must be a string, tuple of two strings, or list of strings")
    
    # Dictionary to store all animal data
    all_animal_data = {}
    
    # Process each animal
    for animal_id in animal_ids:
        # Find the animal's directory (it might have different suffixes)
        animal_dirs = glob.glob(os.path.join(cohort_path, f"{animal_id}-*"))
        
        if not animal_dirs:
            print(f"Warning: No directory found for animal {animal_id}")
            continue
            
        animal_dir = animal_dirs[0]  # Take the first matching directory
        session_data_path = os.path.join(animal_dir, "Randelay_photo_singlevalue", "Session Data")
        
        if not os.path.exists(session_data_path):
            print(f"Warning: Session data path not found for animal {animal_id}")
            continue
        
        # Get all .mat files for this animal
        mat_files = glob.glob(os.path.join(session_data_path, "*.mat"))
        
        if not mat_files:
            print(f"Warning: No .mat files found for animal {animal_id}")
            continue
        
        # Dictionary to store this animal's data
        animal_data = {}
        
        # Load each session file
        for mat_file in mat_files:
            try:
                session_data = load_mat_file(mat_file)
                # Extract date from filename
                date_str = os.path.basename(mat_file).split('_')[-2]  # Assuming format includes date
                animal_data[date_str] = session_data
            except Exception as e:
                print(f"Error loading {mat_file}: {str(e)}")
        
        # Store in the global dictionary
        var_name = f"{animal_id}"
        global_vars[var_name] = animal_data
        all_animal_data[animal_id] = animal_data
        
        print(f"Successfully imported data for {animal_id}")
    
    return all_animal_data 