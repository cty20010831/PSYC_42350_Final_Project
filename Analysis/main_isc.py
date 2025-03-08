import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, image, masking
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from load_and_mask import load_and_mask_fmri
from isc_analysis import (
    compute_and_save_subject_level_isc, 
    extract_yeo_roi_means_single_subject, 
    label_flip_permutation, 
    cohen_d_effect_size, 
    pairwise_network_tests
)

def main():
    # ----------------------------
    # 1) Load & Mask
    # ----------------------------
    data_dir = "Preprocessed_Data"
    out_dir = "Results"
    subjects = sorted([s for s in os.listdir(data_dir) if s.startswith('sub-')])

    # Load narrative vs. non-narrative data in 3mm MNI space
    narrative_masked, non_narrative_masked, final_mask = load_and_mask_fmri(
        data_dir=data_dir,
        narrative_task='dme_run-01',
        non_narrative_task='inscapes',
        file_name='func_pp_nofilt_sm0.mni152.3mm.nii.gz',
        resolution=3
    )
    print(f"Loaded {len(subjects)} subjects.")

    # --------------------------------------------------
    # 2) **Trim Timepoints** so all subjects match
    # --------------------------------------------------
    # Trim each condition separately
    # Find minimum time dimension for narrative
    time_dims_narr = [arr.shape[1] for arr in narrative_masked]
    min_time_narr = min(time_dims_narr)
    # Truncate any larger arrays
    for i, arr in enumerate(narrative_masked):
        if arr.shape[1] > min_time_narr:
            narrative_masked[i] = arr[:, :min_time_narr]

    # Find minimum time dimension for non-narrative
    time_dims_non = [arr.shape[1] for arr in non_narrative_masked]
    min_time_non = min(time_dims_non)
    # Truncate
    for i, arr in enumerate(non_narrative_masked):
        if arr.shape[1] > min_time_non:
            non_narrative_masked[i] = arr[:, :min_time_non]

    # ----------------------------
    # 3) Compute ISC
    # ----------------------------
    isc_sub_dir = os.path.join(out_dir, "ISC_SubjectMaps")
    compute_and_save_subject_level_isc(
        narrative_masked, non_narrative_masked, 
        mask_img=final_mask,
        subjects=subjects,
        out_dir=isc_sub_dir
    )
    
    # ----------------------------
    # 4) Fetch Yeo 17 Atlas, Resample, Extract DMN, etc.
    # ----------------------------
    # Yeo 17 Network Labels
    # Label   Putative Network Mapping
    # -----   ------------------------
    #     1    Visual Network (subdivision A)
    #     2    Visual Network (subdivision B)
    #     3    Visual Network (subdivision C)
    #     4    Somatomotor Network (subdivision A)
    #     5    Somatomotor Network (subdivision B)
    #     6    Dorsal Attention (subdivision A)
    #     7    Dorsal Attention (subdivision B)
    #     8    Salience / Ventral Attention (subdiv. A)
    #     9    Salience / Ventral Attention (subdiv. B)
    #     10    Limbic Network (subdivision A)
    #     11    Limbic Network (subdivision B)
    #     12    Control / Frontoparietal (subdivision A)
    #     13    Control / Frontoparietal (subdivision B)
    #     14    Control / Frontoparietal (subdivision C)
    #     15    Default Mode Network (subdivision A)
    #     16    Default Mode Network (subdivision B)
    #     17    Default Mode Network (subdivision C)

    yeo = datasets.fetch_atlas_yeo_2011()
    yeo_atlas = nib.load(yeo['thick_17'])
    # Use first subject's narrative map as reference for shape/affine
    ref_sub = subjects[0]
    ref_narr_path = os.path.join(isc_sub_dir, f"{ref_sub}_ISC_narrative.nii.gz")
    ref_img = nib.load(ref_narr_path)

    yeo_atlas_3mm = image.resample_to_img(
        source_img=yeo_atlas,
        target_img=ref_img,
        interpolation='nearest'
    )

    # Extract ROI means for each subject
    labels = list(range(1, 18))
    n_subj = len(subjects)
    n_labels = len(labels)

    roi_narr_vals = np.zeros((n_subj, n_labels))
    roi_non_vals  = np.zeros((n_subj, n_labels))

    for s, subj in enumerate(subjects):
        s_narr_path = os.path.join(isc_sub_dir, f"{subj}_ISC_narrative.nii.gz")
        s_non_path  = os.path.join(isc_sub_dir, f"{subj}_ISC_non_narrative.nii.gz")
        narr_dict   = extract_yeo_roi_means_single_subject(s_narr_path, yeo_atlas_3mm, labels)
        non_dict    = extract_yeo_roi_means_single_subject(s_non_path,  yeo_atlas_3mm, labels)

        for i, lab in enumerate(labels):
            roi_narr_vals[s, i] = narr_dict[lab]
            roi_non_vals[s, i]  = non_dict[lab]
    # ----------------------------
    # 5) ISC Related Analyses (narrative versus non-narrative)
    # ----------------------------
    # Permutation test
    obs_tstats, pvals_perm, null_dist = label_flip_permutation(roi_narr_vals, roi_non_vals, n_permutations=2000)

    # Multiple comparisons
    reject, pvals_corrected, _, _ = multipletests(pvals_perm, alpha=0.05, method='fdr_bh')

    print("\n===== Permutation Test Results (Yeo17) =====")
    for i, lab in enumerate(labels):
        print(f"Network {lab}: t={obs_tstats[i]:.3f}, p_perm={pvals_perm[i]:.4g}, "
              f"p_corr={pvals_corrected[i]:.4g}, sig={reject[i]}")

    # Compute effect size (Cohen's d) and do repeated-measures ANOVA
    # A) For each network, we have diffs[s, i] = (roi_narr_vals[s, i] - roi_non_vals[s, i])
    diffs = roi_narr_vals - roi_non_vals  # shape (n_subj, n_labels)

    # A.1) Print effect size per network
    print("\n===== Effect Size of Difference (sorted by Cohen's d) =====")
    effect_sizes = []
    for i, lab in enumerate(labels):
        d_vals = diffs[:, i]  # shape (n_subj,)
        c_d = cohen_d_effect_size(d_vals)
        effect_sizes.append((lab, c_d))

    # Sort by Cohen's d in descending order
    sorted_effects = sorted(effect_sizes, key=lambda x: x[1], reverse=True)
    for (lab, c_d) in sorted_effects:
        print(f"Network {lab}: Cohen's d={c_d:.3f}")

    # B) Repeated-measures ANOVA across networks
    print("\n===== Repeated-Measures ANOVA Across Networks =====")
    #    Is there an overall effect of "Network" on the difference?
    #    We'll transform 'diffs' into a long-format DataFrame for statsmodels' AnovaRM.

    df_list = []
    for s_idx, subj in enumerate(subjects):
        for i, lab in enumerate(labels):
            df_list.append({
                'Subject': subj,
                'Network': f"Net_{lab}",
                'Diff': diffs[s_idx, i]
            })
    df = pd.DataFrame(df_list)
    # Run repeated-measures ANOVA with "Network" as the within-subject factor,
    # and "Diff" as the dependent variable.

    anova = AnovaRM(data=df, depvar='Diff', subject='Subject', within=['Network'])
    anova_res = anova.fit()
    print(anova_res)

    # Post-hoc Pairwise T-Tests Among Networks + Heatmap
    print("\n===== Post-hoc Pairwise T-Tests Among Networks + Heatmap =====")
    network_labels_str = [f"Net_{lab}" for lab in labels]
    p_matrix, t_matrix, corr_p_matrix, sig_matrix = pairwise_network_tests(
        diffs=diffs,
        network_labels=network_labels_str,
        alpha=0.05,
        method='fdr_bh',
        plot_heatmap=True
    )

    print("\nISC Analysis complete!")

if __name__ == "__main__":
    main()
