import os
import glob
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

def import_mgh_covid_data(
    data_dir="../Data/MGH_Olink_COVID_Apr_27_2021/",
    na_values=None,
    return_var_desc=False,
    id_col="subject_id"
):
    """
    Import and preprocess MGH COVID clinical, OLINK NPX, and variable description data.
    Automatically finds and merges all relevant files for downstream analysis.
    NPX data is automatically pivoted to wide format (one row per subject, one column per assay).

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the data files (default: '../Data/MGH_Olink_COVID_Apr_27_2021/').
    na_values : list or str, optional
        Values to treat as missing (default: ['', 'NA', 'N/A']).
    return_var_desc : bool, optional
        Whether to return the variable description DataFrame.
    id_col : str, optional
        Name of the subject/sample ID column to use for merging (default: 'subject_id').

    Returns
    -------
    merged_df : pd.DataFrame
        Merged clinical + OLINK NPX data, index = subject_id.
    var_desc_df : pd.DataFrame, optional
        Variable descriptions, if requested and available.
    """
    if na_values is None:
        na_values = ['', 'NA', 'N/A']

    # --- Find files ---
    clinical_pattern = os.path.join(data_dir, '*Clinical_Info.txt')
    npx_pattern = os.path.join(data_dir, '*OLINK_NPX.txt')
    var_desc_pattern = os.path.join(data_dir, '*Variable_descriptions*.xlsx')

    clinical_files = glob.glob(clinical_pattern)
    npx_files = glob.glob(npx_pattern)
    var_desc_files = glob.glob(var_desc_pattern)

    if not clinical_files:
        raise FileNotFoundError(f"No clinical info file found in {data_dir}")
    if not npx_files:
        raise FileNotFoundError(f"No OLINK NPX file found in {data_dir}")
    clinical_file = clinical_files[0]
    npx_file = npx_files[0]
    var_desc_file = var_desc_files[0] if var_desc_files else None

    # --- Import clinical info ---
    clinical_df = pd.read_csv(
        clinical_file,
        sep=';',
        na_values=na_values,
        dtype=str
    )
    clinical_df.columns = clinical_df.columns.str.strip()
    for col in clinical_df.columns:
        try:
            clinical_df[col] = pd.to_numeric(clinical_df[col])
        except Exception:
            pass
    if 'subject_id' in clinical_df.columns:
        clinical_df['subject_id'] = clinical_df['subject_id'].astype(str).str.strip()
    else:
        raise ValueError("'subject_id' column not found in clinical info file.")

    # --- Import OLINK NPX data (long format) ---
    npx_df = pd.read_csv(
        npx_file,
        sep=';',  # Use semicolon delimiter
        na_values=na_values,
        dtype=str
    )
    npx_df.columns = npx_df.columns.str.strip()
    # Try to find the ID column (case-insensitive match)
    id_candidates = [c for c in npx_df.columns if c.lower() == id_col.lower()]
    if not id_candidates:
        print(f"ID column '{id_col}' not found in OLINK NPX file. Available columns:")
        print(npx_df.columns.tolist())
        raise ValueError(f"'{id_col}' column not found in OLINK NPX file.")
    npx_id_col = id_candidates[0]
    npx_df[npx_id_col] = npx_df[npx_id_col].astype(str).str.strip()
    # Pivot to wide format: subject_id as index, each Assay as a column, values=NPX
    if 'Assay' not in npx_df.columns or 'NPX' not in npx_df.columns:
        raise ValueError("NPX file must contain 'Assay' and 'NPX' columns for pivoting.")
    npx_wide = npx_df.pivot_table(
        index=npx_id_col,
        columns='Assay',
        values='NPX',
        aggfunc='first'  # or np.mean if there are duplicates
    )
    npx_wide.reset_index(inplace=True)
    npx_wide['subject_id'] = npx_wide[npx_id_col].astype(str).str.strip()
    # Convert numeric columns (except subject_id)
    for col in npx_wide.columns:
        if col in [npx_id_col, 'subject_id']:
            continue
        try:
            npx_wide[col] = pd.to_numeric(npx_wide[col])
        except Exception:
            pass

    # --- Merge clinical and NPX data using on='subject_id' ---
    merged_df = pd.merge(clinical_df, npx_wide, on='subject_id', how='inner')
    # Set index to subject_id
    merged_df.set_index('subject_id', inplace=True)
    # Do NOT drop the subject_id column (it's now the index)

    # --- Import variable descriptions ---
    var_desc_df = None
    if var_desc_file is not None:
        try:
            var_desc_df = pd.read_excel(var_desc_file)
            var_desc_df.columns = var_desc_df.columns.str.strip()
        except Exception as e:
            print(f"Warning: Could not read variable description file: {e}")
            var_desc_df = None

    if return_var_desc:
        return merged_df, var_desc_df
    else:
        return merged_df

def mgh_LDA(
    data,
    label_col,
    feature_cols=None,
    n_components=2,
    ax=None,
    legend_loc='best',
    title=None,
    cmap='tab10',
    alpha=0.8,
    s=40,
    vertical_jitter=True
):
    """
    Perform and plot LDA on MGH COVID data, coloring by any group/label column.
    Handles both binary and multiclass LDA.

    Parameters
    ----------
    data : pd.DataFrame
        Merged clinical + OLINK NPX data (from import_mgh_covid_data).
    label_col : str
        Column in data to use for coloring/groups (e.g., 'COVID', 'Acuity_max', etc.).
    feature_cols : list of str, optional
        Columns to use as features (default: all columns not in clinical or label_col).
    n_components : int, optional
        Number of LDA components to compute/plot (default: 2).
    ax : matplotlib axis, optional
        Axis to plot on (default: creates new figure).
    legend_loc : str, optional
        Location of legend (default: 'best').
    title : str, optional
        Plot title (default: auto-generated).
    cmap : str, optional
        Matplotlib colormap for groups (default: 'tab10').
    alpha : float, optional
        Point transparency (default: 0.8).
    s : int, optional
        Point size (default: 40).
    vertical_jitter : bool, optional
        If True and n_components==1, add vertical jitter for 1D LDA plot (default: True).

    Returns
    -------
    lda : LinearDiscriminantAnalysis
        Fitted LDA object.
    X_lda : np.ndarray
        LDA-transformed data.
    """
    # Identify features if not provided
    if feature_cols is None:
        clinical_cols = [
            'COVID', 'Age_cat', 'BMI_cat', 'HEART', 'LUNG', 'KIDNEY', 'DIABETES', 'HTN', 'IMMUNO',
            'Resp_Symp', 'Fever_Sympt', 'GI_Symp', 'D0_draw', 'D3_draw', 'D7_draw', 'DE_draw',
            'Acuity_0', 'Acuity_3', 'Acuity_7', 'Acuity_28', 'Acuity_max',
            'abs_neut_0_cat', 'abs_lymph_0_cat', 'abs_mono_0_cat', 'creat_0_cat', 'crp_0_cat',
            'ddimer_0_cat', 'ldh_0_cat', 'Trop_72h', 'abs_neut_3_cat', 'abs_lymph_3_cat',
            'abs_mono_3_cat', 'creat_3_cat', 'crp_3_cat', 'ddimer_3_cat', 'ldh_3_cat',
            'abs_neut_7_cat', 'abs_lymph_7_cat', 'abs_mono_7_cat', 'creat_7_cat', 'crp_7_cat',
            'ddimer_7_cat', 'ldh_7_cat', 'SampleID', 'subject_id', label_col
        ]
        feature_cols = [col for col in data.columns if col not in clinical_cols]
    # Drop rows with missing label or features
    df = data.dropna(subset=[label_col] + feature_cols)
    X = df[feature_cols].values
    y = df[label_col].values
    # Fit LDA
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X, y)
    # Plot
    is_1d = X_lda.shape[1] == 1
    if ax is None:
        if is_1d and not vertical_jitter:
            fig, ax = plt.subplots(figsize=(8, 2))
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = None
    groups = np.unique(y)
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(groups)))
    if is_1d:
        # 1D LDA: plot LDA 1 vs. vertical jitter or y=0
        for i, group in enumerate(groups):
            mask = y == group
            if vertical_jitter:
                yvals = np.random.normal(i, 0.04, size=np.sum(mask))
            else:
                yvals = np.zeros(np.sum(mask))
            ax.scatter(X_lda[mask, 0], yvals, label=str(group), color=colors[i], alpha=alpha, s=s, edgecolor='k', linewidth=0.5)
        ax.set_xlabel('LDA 1')
        if vertical_jitter:
            ax.set_yticks([])
            ax.set_ylabel('')
        else:
            # Remove y-axis for true 1D plot
            ax.set_yticks([])
            ax.set_ylabel('')
            ax.set_ylim(-0.1, 0.1)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_position('zero')
    else:
        # 2D LDA: plot LDA 1 vs. LDA 2
        for i, group in enumerate(groups):
            mask = y == group
            ax.scatter(X_lda[mask, 0], X_lda[mask, 1], label=str(group), color=colors[i], alpha=alpha, s=s, edgecolor='k', linewidth=0.5)
        ax.set_xlabel('LDA 1')
        ax.set_ylabel('LDA 2')
    if title is None:
        title = f"LDA: {label_col}"
    ax.set_title(title)
    ax.legend(loc=legend_loc)
    if fig is not None:
        plt.tight_layout()
        plt.show()
    return lda, X_lda
