import os
import numpy as np
import datetime
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib
from pandas import Timestamp
import seaborn as sns
from collections import defaultdict

# Set matplotlib parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Default colors for plotting
default_colors = plt.cm.tab10.colors

# Global variable to store the weight data
weight_data = None

def get_weight_data():
    """Get the weight data, loading it if necessary."""
    global weight_data
    if weight_data is None:
        # Detect the platform (macOS or Windows)
        if os.name == 'posix':
            base_path = '/Volumes/ChemoBrain/ChemoBrain-Analysis/Data/'
        elif os.name == 'nt':  # Windows
            base_path = 'E:/Data/'  # Change 'E:' to the actual drive letter of your external drive on Windows
        else:
            raise Exception("Unsupported operating system")
        
        # Load the weight data
        weight_data = pd.read_csv(os.path.join(base_path, 'Mice_Log(Weight).csv'))
    return weight_data

def get_animal_weights(animal_id):
    """
    Get weight measurements for a specific animal.
    
    Parameters:
    -----------
    animal_id : str or int
        The ID of the animal to get weights for
        
    Returns:
    --------
    list of tuples
        List of (date, weight) tuples for the specified animal
    """
    data = get_weight_data()
    
    # Check if the provided animal ID exists in the DataFrame
    if animal_id in data.iloc[:, 0].values:
        # Get the index of the animal ID
        index = np.where(data.iloc[:, 0].values == animal_id)[0][0]

        # Extract weights and dates, and filter out any invalid dates
        weights = data.iloc[index, 1:].values
        dates = data.columns[1:].values
        valid_weights_with_dates = [(d, w) for d, w in zip(dates, weights) if pd.to_datetime(d, errors='coerce') is not pd.NaT]

        return valid_weights_with_dates  # Return only valid date-weight pairs
    else:
        return None

def plot_animal_weights(
    animal_ids, 
    x_date_range=None, 
    x_labels=None, 
    tick_interval=1,  # Controls the frequency of x-ticks
    trace_colors=None,  # Dictionary of colors for each animal ID
    plot_title="Weights over Time for Multiple Animals"  # Default plot title
):
    """
    Plot weight measurements for multiple animals.
    
    Parameters:
    -----------
    animal_ids : list
        List of animal IDs to plot
    x_date_range : tuple, optional
        Tuple of (start_date, end_date) to filter the date range
    x_labels : list, optional
        Custom labels for x-axis
    tick_interval : int, optional
        Controls the frequency of x-ticks
    trace_colors : dict, optional
        Dictionary mapping animal IDs to colors
    plot_title : str, optional
        Title for the plot
    """
    plt.figure(figsize=(12, 6))  # Create the figure once for all plots

    all_dates = []  # To collect all dates for x-axis management
    for animal_id in animal_ids:
        weights_with_dates = get_animal_weights(animal_id)
        if weights_with_dates is not None:
            # Unzip the list of tuples into two separate lists
            dates, weights = zip(*weights_with_dates)
            dates = [pd.to_datetime(date) for date in dates]  # Ensure dates are datetime objects
            all_dates.extend(dates)

            # Filter data by x_date_range
            if x_date_range:
                start_date, end_date = pd.to_datetime(x_date_range[0]), pd.to_datetime(x_date_range[1])
                filtered_indices = [i for i, d in enumerate(dates) if start_date <= d <= end_date]
                dates = [dates[i] for i in filtered_indices]
                weights = [weights[i] for i in filtered_indices]

            # Determine color for the plot
            color = trace_colors.get(animal_id, None) if trace_colors else None

            # Plotting the weights for each animal
            plt.plot(dates, weights, marker='o', linestyle='-', label=animal_id, color=color)
        else:
            print(f"No data found for {animal_id}.")

    # Set x-axis limits
    if x_date_range:
        plt.xlim(pd.to_datetime(x_date_range[0]), pd.to_datetime(x_date_range[1]))

    # Update xticks and labels based on tick_interval
    if x_labels and all_dates:
        tick_indices = range(0, len(all_dates), tick_interval)
        plt.xticks([all_dates[i] for i in tick_indices], [x_labels[i] for i in tick_indices])
    else:
        plt.xticks(rotation=45)  # Rotate dates for better visibility

    # Apply the dynamic title
    plt.title(plot_title)
    plt.xlabel("Day")
    plt.ylabel("Weight (g)")
    plt.legend()  # Add a legend to differentiate between animals
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.show()

def plot_animal_groups(
    grouped_animals, 
    pr_date1=None, pr_label1=None, 
    pr_date2=None, pr_label2=None, 
    pr_date3=None, pr_label3=None, 
    group_colors=None, 
    plot_title="Weights over Time for Animal Groups", 
    x_date_range=None, 
    x_labels=None,  
    tick_interval=1,
    date_range_shaded=None,
    shaded_label=None,
    baseline=None
):
    """
    Plot weight measurements for groups of animals.
    
    Parameters:
    -----------
    grouped_animals : dict
        Dictionary mapping group labels to lists of animal IDs
    pr_date1, pr_date2, pr_date3 : tuple or str, optional
        Dates to plot vertical lines for
    pr_label1, pr_label2, pr_label3 : str, optional
        Labels for the vertical lines
    group_colors : dict, optional
        Dictionary mapping group labels to colors
    plot_title : str, optional
        Title for the plot
    x_date_range : tuple, optional
        Tuple of (start_date, end_date) to filter the date range
    x_labels : list, optional
        Custom labels for x-axis
    tick_interval : int, optional
        Controls the frequency of x-ticks
    date_range_shaded : list of tuples, optional
        List of (start_date, end_date) tuples for shaded regions
    shaded_label : str, optional
        Label for shaded regions
    baseline : str or tuple, optional
        Date or date range to use as baseline for percentage calculation
    """
    plt.figure(figsize=(12, 6))

    group_plotted = set()

    # Parse date range if provided
    if x_date_range:
        x_date_range = pd.to_datetime(x_date_range, format='%m/%d/%y')
        start_date, end_date = x_date_range
    else:
        start_date, end_date = None, None

    for idx, (group_label, animal_ids) in enumerate(grouped_animals.items()):
        color = group_colors.get(group_label, default_colors[idx % len(default_colors)]) if group_colors else default_colors[idx % len(default_colors)]
        
        group_baseline_weights = []
        for animal_id in animal_ids:
            weights_with_dates = get_animal_weights(animal_id)
            if weights_with_dates is not None:
                dates, weights = zip(*weights_with_dates)
                dates = pd.to_datetime(dates)

                if baseline:
                    # Filter dates for baseline calculation
                    baseline_weights = []
                    if isinstance(baseline, str):  # Single date
                        baseline_date = pd.to_datetime(baseline)
                        baseline_weights = [weight for date, weight in zip(dates, weights) if date == baseline_date]
                    elif isinstance(baseline, tuple):  # Range of dates
                        start, end = pd.to_datetime(baseline[0]), pd.to_datetime(baseline[1])
                        baseline_weights = [weight for date, weight in zip(dates, weights) if start <= date <= end]
                    else:
                        baseline_weights = []

                    if baseline_weights:
                        baseline_value = sum(baseline_weights) / len(baseline_weights)  # Average baseline value for each animal
                        weights = [(weight / baseline_value) * 100 for weight in weights]  # Normalize to percentage

                if start_date and end_date:
                    filtered = [(date, weight) for date, weight in zip(dates, weights) if start_date <= date <= end_date]
                    if filtered:
                        dates, weights = zip(*filtered)
                    else:
                        continue

                # Plot the normalized weight data
                plt.plot(dates, weights, marker='o', linestyle='-', color=color)

        if group_label not in group_plotted:
            plt.plot([], [], marker='o', linestyle='-', color=color, label=group_label)
            group_plotted.add(group_label)

    # Handle pr_date1, pr_date2, pr_date3
    def plot_vertical_line(date_range, label, color):
        if isinstance(date_range, tuple):  # Only handle tuples of dates
            for i, date_str in enumerate(date_range):
                date = pd.to_datetime(date_str)
                if i == 0:
                    plt.axvline(date, color=color, linestyle='--', label=label)
                else:
                    plt.axvline(date, color=color, linestyle='--')  # No label for subsequent dates

    if pr_date1 and pr_label1:
        plot_vertical_line(pr_date1, pr_label1, 'black')

    if pr_date2 and pr_label2:
        plot_vertical_line(pr_date2, pr_label2, 'blue')

    if pr_date3 and pr_label3:
        plot_vertical_line(pr_date3, pr_label3, 'orange')

    # Add shaded date ranges with a single label
    if date_range_shaded and shaded_label:
        for i, (start, end) in enumerate(date_range_shaded):
            start = pd.to_datetime(start, format='%m/%d/%y')
            end = pd.to_datetime(end, format='%m/%d/%y')
            if i == 0:
                plt.axvspan(start, end, color='lightgrey', alpha=0.5, label=shaded_label)
            else:
                plt.axvspan(start, end, color='lightgrey', alpha=0.5)

    if x_labels and x_date_range is not None:
        all_dates = pd.date_range(start=start_date, end=end_date, periods=len(x_labels))
        selected_dates = all_dates[::tick_interval]
        plt.gca().set_xticks(selected_dates)
        plt.gca().set_xticklabels(x_labels[::tick_interval])

    plt.title(plot_title)
    plt.xlabel("Day")
    plt.ylabel("Weight (g)" if not baseline else "Weight (% baseline)")
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.tight_layout()
    plt.show()

def plot_animal_groups_averaged(
    animals, 
    x_date_range=None, 
    x_labels=None,
    pr_date1=None, 
    pr_label1=None, 
    pr_date2=None, 
    pr_label2=None, 
    pr_date3=None, 
    pr_label3=None, 
    pr_date4=None, 
    pr_label4=None, 
    tick_interval=1,        # Controls the frequency of x-ticks
    show_sem='shaded',      # Options: 'shaded', 'error_bars', or None
    baseline=None,          # Baseline parameter for normalization
    date_range_shaded=None, # List of (start_date, end_date) tuples to shade
    shaded_label=None       # Label for the shaded region(s)
):
    """
    Plot averaged weight measurements for groups of animals with error bars or shaded SEM.
    
    Parameters:
    -----------
    animals : dict
        Dictionary mapping group labels to lists of animal IDs
    x_date_range : tuple, optional
        Tuple of (start_date, end_date) to filter the date range
    x_labels : list, optional
        Custom labels for x-axis
    pr_date1, pr_date2, pr_date3, pr_date4 : str, optional
        Dates to plot vertical lines for
    pr_label1, pr_label2, pr_label3, pr_label4 : str, optional
        Labels for the vertical lines
    tick_interval : int, optional
        Controls the frequency of x-ticks
    show_sem : str, optional
        How to display standard error: 'shaded', 'error_bars', or None
    baseline : str or tuple, optional
        Date or date range to use as baseline for percentage calculation
    date_range_shaded : list of tuples, optional
        List of (start_date, end_date) tuples for shaded regions
    shaded_label : str, optional
        Label for shaded regions
    """
    plt.figure(figsize=(12, 6))
    group_plotted = set()

    for idx, (group_label, animal_ids) in enumerate(animals.items()):
        color = default_colors[idx % len(default_colors)]

        # Collect normalized weights for the group by date
        all_weights_by_date = defaultdict(list)

        for animal_id in animal_ids:
            weights_with_dates = get_animal_weights(animal_id)  # user-provided function
            if weights_with_dates:
                dates, weights = zip(*weights_with_dates)
                dates = pd.to_datetime(dates)

                # -------------- BASELINE NORMALIZATION --------------
                if baseline:
                    # If baseline is a single string, convert it to one pd.Timestamp
                    if isinstance(baseline, str):
                        baseline_date = pd.to_datetime(baseline, errors='coerce')
                        # Collect weights exactly on that baseline date
                        baseline_weights = [
                            w for d, w in zip(dates, weights) 
                            if (d == baseline_date and pd.notna(w))
                        ]
                    # If baseline is a tuple/list of dates, convert each to pd.Timestamp
                    elif isinstance(baseline, (list, tuple)):
                        baseline_list = [pd.to_datetime(b, errors='coerce') for b in baseline]
                        # Collect weights on ANY of those baseline dates
                        baseline_weights = [
                            w for d, w in zip(dates, weights) 
                            if any(d == bdate for bdate in baseline_list) and pd.notna(w)
                        ]
                    else:
                        baseline_weights = []

                    # If we found any baseline weights, average them and normalize
                    if baseline_weights:
                        baseline_value = sum(baseline_weights) / len(baseline_weights)
                        if baseline_value > 0:
                            weights = [
                                (w / baseline_value) * 100 if pd.notna(w) else np.nan 
                                for w in weights
                            ]
                # ---------------------------------------------------

                # Add these weights to our group-aggregated dictionary
                for d, w in zip(dates, weights):
                    if pd.notna(w):
                        all_weights_by_date[d].append(w)

        # If no valid data for this group, skip it
        if not all_weights_by_date:
            continue

        # Compute group averages and SEMs per date
        avg_dates, avg_weights, sems = [], [], []
        for date, wlist in sorted(all_weights_by_date.items()):
            if wlist:  
                avg_dates.append(date)
                avg_weights.append(np.mean(wlist))
                sems.append(np.std(wlist) / np.sqrt(len(wlist)))

        # Filter data by x_date_range
        if x_date_range:
            start_date, end_date = pd.to_datetime(x_date_range[0]), pd.to_datetime(x_date_range[1])
            filtered_indices = [i for i, d in enumerate(avg_dates) if start_date <= d <= end_date]
            avg_dates   = [avg_dates[i]   for i in filtered_indices]
            avg_weights = [avg_weights[i] for i in filtered_indices]
            sems        = [sems[i]        for i in filtered_indices]

        # Plot group average and SEM
        plt.plot(avg_dates, avg_weights, marker='o', linestyle='-', color=color)
        if show_sem == 'shaded':
            plt.fill_between(
                avg_dates, 
                np.array(avg_weights) - np.array(sems), 
                np.array(avg_weights) + np.array(sems), 
                color=color, alpha=0.2
            )
        elif show_sem == 'error_bars':
            plt.errorbar(avg_dates, avg_weights, yerr=sems, fmt='o', color=color, capsize=5)

        # Add group label to legend (just once)
        if group_label not in group_plotted:
            plt.plot([], [], marker='o', linestyle='-', color=color, label=group_label)
            group_plotted.add(group_label)

    # Plot vertical lines for provided pr_date# with different colors
    pr_dates  = [pr_date1,  pr_date2,  pr_date3,  pr_date4]
    pr_labels = [pr_label1, pr_label2, pr_label3, pr_label4]
    for i, (pdate, plabel) in enumerate(zip(pr_dates, pr_labels)):
        if pdate and plabel:
            vcolor = default_colors[i % len(default_colors)]
            plt.axvline(pd.to_datetime(pdate), color=vcolor, linestyle='--', label=plabel)

    # ----- Add shaded date ranges (with one legend entry) -----
    if date_range_shaded:
        for i, (start, end) in enumerate(date_range_shaded):
            start = pd.to_datetime(start, errors='coerce')
            end   = pd.to_datetime(end,   errors='coerce')
            if pd.notnull(start) and pd.notnull(end):
                if i == 0:
                    plt.axvspan(start, end, color='lightgrey', alpha=0.5, label=shaded_label)
                else:
                    plt.axvspan(start, end, color='lightgrey', alpha=0.5)

    # Set plot title and labels
    plt.title("Average Weights over Time for Animal Groups")
    plt.xlabel("Day")
    plt.ylabel("Average Weight (% baseline)" if baseline else "Average Weight (g)")

    # Set x-axis limits
    if x_date_range:
        plt.xlim(pd.to_datetime(x_date_range[0]), pd.to_datetime(x_date_range[1]))

    # Update xticks and labels based on tick_interval
    if x_labels and len(avg_dates) > 0:
        tick_indices = range(0, len(avg_dates), tick_interval)
        plt.xticks([avg_dates[i] for i in tick_indices], [x_labels[i] for i in tick_indices])
    else:
        plt.xticks()

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.tight_layout()
    plt.show()

def boxplot_animal_groups(
    animals,
    group_time_points,  # Map each group to a list of time points (dates)
    x_labels,
    time_point_colors,
    point_alpha=0.7,
    y_label=None,
    plot_title="Weight Comparisons by Group and Time Point",
    baseline_date=None,
    time_point_legend_labels=None
):
    """
    Create a boxplot comparing weights across groups and time points.
    
    Parameters:
    -----------
    animals : dict
        Dictionary mapping group labels to lists of animal IDs
    group_time_points : dict
        Dictionary mapping each group to a list of time points (dates)
    x_labels : list
        Labels for the x-axis groups
    time_point_colors : dict
        Dictionary mapping time point indices to colors
    point_alpha : float, optional
        Transparency of individual data points
    y_label : str, optional
        Label for the y-axis
    plot_title : str, optional
        Title for the plot
    baseline_date : str, optional
        Date to use as baseline for percentage calculation
    time_point_legend_labels : dict, optional
        Dictionary mapping time point indices to legend labels
    """
    all_data = []
    all_groups = []
    all_time_points = []
    
    baseline_values = {}

    # If baseline_date is provided, normalize the weights to the baseline
    if baseline_date is not None:
        for group, animal_ids in animals.items():
            for animal in animal_ids:
                weights_with_dates = get_animal_weights(animal)
                if weights_with_dates:
                    dates, weights = zip(*weights_with_dates)
                    dates = pd.to_datetime(dates)
                    baseline_weight = [weight for date, weight in zip(dates, weights) if date == pd.to_datetime(baseline_date)]
                    if baseline_weight:
                        baseline_values[animal] = baseline_weight[0]
                    else:
                        baseline_values[animal] = 1  # Fallback value if no baseline is found

    # Create a map that associates each time point index with a label and color, regardless of the actual date
    time_point_label_color_map = {}
    all_time_points_seen = set()
    
    for group, time_points in group_time_points.items():
        for idx, time_point in enumerate(time_points):
            if idx not in all_time_points_seen:
                all_time_points_seen.add(idx)
                label = time_point_legend_labels.get(str(idx), f"Round {idx + 1}") if time_point_legend_labels else f"Round {idx + 1}"
                color = time_point_colors.get(str(idx), '#cccccc')  # Default color
                time_point_label_color_map[idx] = (label, color)

    for group, animal_ids in animals.items():
        # Retrieve the specific time points for each group
        group_time_points_list = group_time_points.get(group, [])
        
        for animal in animal_ids:
            for idx, time_point in enumerate(group_time_points_list):
                weights_with_dates = get_animal_weights(animal)
                if weights_with_dates:
                    dates, weights = zip(*weights_with_dates)
                    dates = pd.to_datetime(dates)
                    weight = [weight for date, weight in zip(dates, weights) if date == pd.to_datetime(time_point)]
                    if weight:
                        weight = weight[0]
                    else:
                        weight = 0  # Default if no weight found for this time point
                    
                    if baseline_date is not None:
                        baseline_weight = baseline_values.get(animal, 1)
                        normalized_weight = (weight / baseline_weight) * 100 if baseline_weight != 0 else 0
                        all_data.append(normalized_weight)
                    else:
                        all_data.append(weight)
                    all_groups.append(group)
                    all_time_points.append(idx)  # Use index as the time point

    data_df = pd.DataFrame({
        "Group": all_groups,
        "Time Point": all_time_points,
        "Weight": all_data
    })
    
    # Use the map to assign the colors based on the index of time points
    indexed_time_point_colors = [time_point_label_color_map[idx][1] for idx in all_time_points]
    
    # Initialize plot
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=data_df,
        x="Group",
        y="Weight",
        hue="Time Point",
        palette=indexed_time_point_colors,
        showcaps=True,
        boxprops={'edgecolor': 'black'},
        medianprops={'color': 'black'},
        whiskerprops={'linewidth': 1.5},
        flierprops=None,
        showfliers=False  # Prevent outliers from being plotted separately
    )
    
    # Overlay individual points with jitter and a border
    sns.stripplot(
        data=data_df,
        x="Group",
        y="Weight",
        hue="Time Point",
        dodge=True,
        palette=indexed_time_point_colors,
        alpha=point_alpha,
        jitter=True,
        marker="o",
        linewidth=1,  # Add border
        edgecolor='black',  # Border color
        legend=False
    )

    ax.set_title(f"{plot_title}", fontsize=14)
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel(f"{y_label}" if y_label else ("Weight (% baseline)" if baseline_date else "Weight (g)"), 
                  fontsize=12)
    ax.set_xticklabels(x_labels)
    
    # Adjust legend based on the mapped time points
    handles, labels = ax.get_legend_handles_labels()
    
    # Update legend with the labels for time points
    legend_labels = [time_point_label_color_map[idx][0] for idx in sorted(set(all_time_points))]
    ax.legend(handles=handles, labels=legend_labels, title="Time Point")
    
    plt.tight_layout()
    plt.show()

def survival_plot(
    x_date_range,
    x_labels,
    tick_interval,
    groups,
    group_colors,
    deaths,
    initial_individuals,
    plot_title="Survival Plot",
    percent=False,
    dashed_line_date1=None,
    dashed_line_label1=None,
    dashed_line_date2=None,
    dashed_line_label2=None,
    dashed_line_date3=None,
    dashed_line_label3=None,
    dashed_line_date4=None,
    dashed_line_label4=None,
    date_range_shaded=None,
    shaded_label=None
):
    """
    Create a survival plot showing the number or percentage of surviving individuals over time.
    
    Parameters:
    -----------
    x_date_range : tuple
        Tuple of (start_date, end_date) in 'mm/dd/yy' format
    x_labels : list
        Labels for the x-axis
    tick_interval : int
        Controls the frequency of x-ticks
    groups : list
        List of group names
    group_colors : dict
        Dictionary mapping group names to colors
    deaths : dict
        Dictionary mapping group names to lists of death dates in 'mm/dd/yy' format
    initial_individuals : dict
        Dictionary mapping group names to initial number of individuals
    plot_title : str, optional
        Title for the plot
    percent : bool, optional
        Whether to show survival as percentage (True) or count (False)
    dashed_line_date1, dashed_line_date2, dashed_line_date3, dashed_line_date4 : str or list, optional
        Dates to plot vertical lines for
    dashed_line_label1, dashed_line_label2, dashed_line_label3, dashed_line_label4 : str, optional
        Labels for the vertical lines
    date_range_shaded : list of tuples, optional
        List of (start_date, end_date) tuples for shaded regions
    shaded_label : str, optional
        Label for shaded regions
    """
    # Convert the x_date_range to datetime objects
    start_date = datetime.datetime.strptime(x_date_range[0], '%m/%d/%y')
    end_date = datetime.datetime.strptime(x_date_range[1], '%m/%d/%y')
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date)

    # Initialize survival data dictionary
    survival_data = {group: [initial_individuals[group]] for group in groups}

    # Calculate survival data step-wise
    for current_date in date_range[1:]:
        for group in groups:
            current_survival = survival_data[group][-1]
            # Check if deaths occurred on this date
            death_count = deaths.get(group, []).count(current_date.strftime('%m/%d/%y'))
            # Append the updated survival count
            survival_data[group].append(current_survival - death_count)

    # Adjust data to percentages if required
    if percent:
        survival_data = {
            group: [(count / initial_individuals[group]) * 100 for count in counts]
            for group, counts in survival_data.items()
        }

    # Determine y-axis max
    y_max = 100 if percent else max(initial_individuals.values())

    # Plotting
    plt.figure(figsize=(12, 6))
    for group in groups:
        plt.step(
            date_range,
            survival_data[group],
            where='post',
            label=group,
            color=group_colors[group]
        )

    # Customize x-axis labels
    plt.xticks(
        ticks=[date_range[i] for i in range(0, len(date_range), tick_interval)],
        labels=x_labels[::tick_interval]
    )

    # Add dashed vertical lines
    for dashed_date, dashed_label, color in zip(
        [dashed_line_date1, dashed_line_date2, dashed_line_date3, dashed_line_date4],
        [dashed_line_label1, dashed_line_label2, dashed_line_label3, dashed_line_label4],
        ['black', 'blue', 'red', 'brown']
    ):
        if dashed_date:
            if isinstance(dashed_date, str):
                dashed_date = [dashed_date]
            for idx, date_str in enumerate(dashed_date):
                d = datetime.datetime.strptime(date_str, '%m/%d/%y')
                plt.axvline(
                    x=d, color=color, linestyle='--',
                    label=(dashed_label if idx == 0 else None)
                )

    # Add shaded date range
    if date_range_shaded:
        for i, (start, end) in enumerate(date_range_shaded):
            start_date = datetime.datetime.strptime(start, '%m/%d/%y')
            end_date = datetime.datetime.strptime(end, '%m/%d/%y')
            if i == 0:
                plt.axvspan(start_date, end_date, color='lightgrey', alpha=0.5, label=shaded_label)
            else:
                plt.axvspan(start_date, end_date, color='lightgrey', alpha=0.5)

    # Plot customization
    plt.title(plot_title)
    plt.xlabel("Days")
    plt.ylabel("Survival (%)" if percent else "Survival Count")
    
    # Set y-axis limits and ticks
    plt.ylim(0, y_max)
    if not percent:  # Only show whole numbers on y-axis in "number mode"
        plt.gca().set_yticks(range(0, math.ceil(y_max) + 1, 1))

    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()


