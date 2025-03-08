import os
import random
import numpy as np
import nibabel as nib
from scipy.stats import pearsonr, ttest_1samp
from statsmodels.stats.multitest import multipletests
from nilearn import masking
import matplotlib.pyplot as plt
import seaborn as sns

def compute_subject_isc_map(
    this_subj_data,
    other_subjs_data
):
    """
    For a single subject, compute the correlation between that subject's data
    and the mean of the other subjects' data, voxel by voxel.
    
    Parameters
    ----------
    this_subj_data   : np.ndarray, shape (n_voxels, n_time)
    other_subjs_data : list of np.ndarray or a 3D array. Each is (n_voxels, n_time).
    
    Returns
    -------
    subject_isc_1d : np.ndarray (n_voxels,)
        The correlation for each voxel.
    """
    # Average the "other" subjects
    # If other_subjs_data is a list
    data_3d = np.stack(other_subjs_data, axis=0)  # shape (n_others, n_vox, n_time)
    others_mean = np.mean(data_3d, axis=0)        # shape (n_vox, n_time)

    n_vox, n_time = this_subj_data.shape
    corrs = np.zeros(n_vox, dtype=float)

    for v in range(n_vox):
        std_subj = np.std(this_subj_data[v])
        std_others = np.std(others_mean[v])
        if std_subj < 1e-12 or std_others < 1e-12:
            corrs[v] = 0.0
        else:
            r, _ = pearsonr(this_subj_data[v], others_mean[v])
            corrs[v] = r

    return corrs

def compute_and_save_subject_level_isc(
    narrative_masked, 
    nonnarr_masked, 
    mask_img, 
    subjects,
    out_dir
):
    """
    For each subject s in [0..n_subj-1]:
      1) Remove that subject's data from the list => average the others
      2) Correlate with subject s => produce (n_voxels,) correlations
      3) Unmask => 3D volume, save as sub-XX_ISC_narrative.nii.gz, etc.

    Returns nothing; saves each subject's 3D NIfTI for narrative & non-narr.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_subj = len(narrative_masked)

    for s in range(n_subj):
        subj_id = subjects[s]

        # Output file paths for sub s
        isc_narr_file = os.path.join(out_dir, f"{subj_id}_ISC_narrative.nii.gz")
        isc_non_file  = os.path.join(out_dir, f"{subj_id}_ISC_non_narrative.nii.gz")

        # Check if they already exist
        if os.path.exists(isc_narr_file) and os.path.exists(isc_non_file):
            print(f"Subject {subj_id}: ISC maps already exist, skipping.")
            continue

        # Otherwise, compute
        print(f"Subject {subj_id}: computing subject-level ISC...")

        # Subject s data for narrative
        this_subj_narr = narrative_masked[s]
        # Other subjects
        others_narr = narrative_masked[:s] + narrative_masked[s+1:]

        # Subject s data for non-narr
        this_subj_non_narr = nonnarr_masked[s]
        others_non_narr = nonnarr_masked[:s] + nonnarr_masked[s+1:]

        # 1D arrays
        s_isc_narr_1d = compute_subject_isc_map(this_subj_narr, others_narr)
        s_isc_non_narr_1d  = compute_subject_isc_map(this_subj_non_narr, others_non_narr)

        # Convert 1D => 3D
        s_isc_narr_3d = masking.unmask(s_isc_narr_1d, mask_img)
        s_isc_non_narr_3d  = masking.unmask(s_isc_non_narr_1d, mask_img)

        # Save
        s_isc_narr_3d.to_filename(isc_narr_file)
        s_isc_non_narr_3d.to_filename(isc_non_file)

    print(f"Saved subject-level ISC maps in {out_dir}")

def extract_yeo_roi_means_single_subject(isc_map_path, atlas_img, labels=(1,2,3,4,5,6,7)):
    """
    Load subject's 3D ISC map from `isc_map_path`, compute mean in each Yeo label.
    Returns a dict {label: mean_val}.
    """
    map_data = nib.load(isc_map_path).get_fdata()
    atlas_data = atlas_img.get_fdata()

    # Make sure both are
    map_data = np.squeeze(map_data)  
    # print("isc_data shape:", map_data.shape)      
    atlas_data = np.squeeze(atlas_data)
    # print("atlas_data shape:", atlas_data.shape)

    results = {}
    for lab in labels:
        mask_bool = (atlas_data == lab)
        vals = map_data[mask_bool]
        if len(vals) > 0:
            results[lab] = np.mean(vals)
        else:
            results[lab] = np.nan
    return results

def label_flip_permutation(narr_vals, non_vals, n_permutations=5000):
    """
    Label-flip permutation for within-subject narrative vs. non-narrative at ROI level.

    narr_vals, non_vals : shape (n_subj, n_labels)

    Returns p-values and observed differences.
    """    
    n_subj, n_labels = narr_vals.shape

    # Observed t-stat
    obs_tstats = np.zeros(n_labels)
    for i_lab in range(n_labels):
        diffs = narr_vals[:, i_lab] - non_vals[:, i_lab]
        t_val, _ = ttest_1samp(diffs, 0, nan_policy='omit')
        obs_tstats[i_lab] = t_val

    null_tstats = np.zeros((n_labels, n_permutations))

    for p in range(n_permutations):
        # Flip labels for each subject w/ 50% chance
        perm_narr = np.zeros_like(narr_vals)
        perm_non  = np.zeros_like(non_vals)
        for s in range(n_subj):
            if random.random() < 0.5:
                perm_narr[s] = narr_vals[s]
                perm_non[s]  = non_vals[s]
            else:
                perm_narr[s] = non_vals[s]
                perm_non[s]  = narr_vals[s]

        # compute t-stats
        for i_lab in range(n_labels):
            diffs = perm_narr[:, i_lab] - perm_non[:, i_lab]
            t_val, _ = ttest_1samp(diffs, 0, nan_policy='omit')
            null_tstats[i_lab, p] = t_val

    # p-value => fraction of null as extreme as observed
    pvals = np.zeros(n_labels)
    for i_lab in range(n_labels):
        obs = obs_tstats[i_lab]
        distribution = null_tstats[i_lab, :]
        more_extreme = np.sum(np.abs(distribution) >= np.abs(obs))
        pvals[i_lab] = more_extreme / n_permutations

    return obs_tstats, pvals, null_tstats

def cohen_d_effect_size(diffs):
    """
    Given a 1D array of within-subject differences (e.g. narrative - non-narrative)
    across subjects, compute Cohen's d = mean(diffs)/std(diffs).
    """
    mean_diff = np.mean(diffs)
    std_diff  = np.std(diffs, ddof=1)  # sample std
    if std_diff < 1e-12:
        return 0.0
    return mean_diff / std_diff

def pairwise_network_tests(
    diffs,
    network_labels=None,
    alpha=0.05,
    method='fdr_bh',
    plot_heatmap=True
):
    """
    Perform pairwise comparisons among Yeo networks (i, j) using within-subject
    differences. Then optionally create a heatmap of FDR-corrected p-values.

    Parameters
    ----------
    diffs : np.ndarray, shape (n_subj, n_labels)
        Each row=subject, each col=one network's difference (narr - nonnarr).
    network_labels : list of str or None
        If None, defaults to "1..n" strings. Otherwise used on x/y ticks.
    alpha : float
        Significance level for multiple comparisons correction.
    method : str
        Correction method (e.g. 'fdr_bh', 'bonferroni').
    plot_heatmap : bool
        If True, produce a Seaborn heatmap of the FDR-corrected p-values.

    Returns
    -------
    p_matrix : (n_labels, n_labels) float
        Raw p-values for each pair (i, j).
    t_matrix : (n_labels, n_labels) float
        T-values for each pair (i, j).
    corrected_p_matrix : (n_labels, n_labels) float
        FDR-corrected p-values.
    sig_matrix : (n_labels, n_labels) bool
        True if that pair is significant after correction.
    """
    n_subj, n_labels = diffs.shape
    if network_labels is None:
        network_labels = [str(i+1) for i in range(n_labels)]

    p_matrix = np.zeros((n_labels, n_labels), dtype=float)
    t_matrix = np.zeros((n_labels, n_labels), dtype=float)

    # Pairwise T-tests
    for i_lab in range(n_labels):
        for j_lab in range(i_lab+1, n_labels):
            diff_of_diff = diffs[:, i_lab] - diffs[:, j_lab]
            t_val, p_val = ttest_1samp(diff_of_diff, 0, nan_policy='omit')
            p_matrix[i_lab, j_lab] = p_val
            p_matrix[j_lab, i_lab] = p_val
            t_matrix[i_lab, j_lab] = t_val
            t_matrix[j_lab, i_lab] = t_val

    # Diagonal => 0
    np.fill_diagonal(p_matrix, 0)
    np.fill_diagonal(t_matrix, 0)

    # Flatten upper triangle => multiple comparisons
    triu_idx = np.triu_indices(n_labels, k=1)
    pvals_tri = p_matrix[triu_idx]
    reject_tri, pvals_corr_tri, _, _ = multipletests(pvals_tri, alpha=alpha, method=method)

    corrected_p_matrix = np.zeros_like(p_matrix)
    sig_matrix = np.zeros((n_labels, n_labels), dtype=bool)

    idx_count = 0
    for i_lab, j_lab in zip(triu_idx[0], triu_idx[1]):
        corrected_p_matrix[i_lab, j_lab] = pvals_corr_tri[idx_count]
        corrected_p_matrix[j_lab, i_lab] = pvals_corr_tri[idx_count]
        sig_matrix[i_lab, j_lab] = reject_tri[idx_count]
        sig_matrix[j_lab, i_lab] = reject_tri[idx_count]
        idx_count += 1

    # (Optional) Heatmap of corrected p-values, no numeric annotations
    if plot_heatmap:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corrected_p_matrix,
            xticklabels=network_labels,
            yticklabels=network_labels,
            cmap="viridis",
            vmin=0, vmax=0.05,
            annot=False    # <--- no numeric text in each cell
        )
        plt.title(f"Pairwise FDR-corrected p-values\n(method={method}, alpha={alpha})")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Save to disk
        plt.tight_layout()
        plt.savefig('Results/Plots/Pairwise_Network_Heatmap.png', dpi=150)
        plt.close()

    return p_matrix, t_matrix, corrected_p_matrix, sig_matrix