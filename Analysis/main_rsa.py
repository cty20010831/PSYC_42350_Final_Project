# """
# Main script for multi-subject cross-modal RSA (EEG–fMRI) in a naturalistic paradigm.
#  - Load EEG (5000 Hz) and fMRI (preprocessed 4D) per subject.
#  - Segment both into fixed-length time windows (e.g. 10s).
#  - Build RDMs, perform cross-modal RSA.
#  - Identify top correlated ROIs, run classification on RSA features.
# """

# import os
# import numpy as np
# import pandas as pd
# from nilearn import image, masking
# from statsmodels.stats.anova import AnovaRM
# from statsmodels.stats.multitest import multipletests
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# from cross_modal_rsa import (
#     load_eeg_data,
#     load_preprocessed_fmri,
#     segment_continuous_data,
#     build_rdm,
#     cross_modal_rsa,
#     identify_top_rois,
#     classify_narr_vs_nonnarr
# )

# def main():
#     # ----------------------------
#     # 1) Load Subjects & Directories
#     # ----------------------------
#     data_dir = "Preprocessed_Data"
#     out_dir = "Results"

#     subjects = sorted([s for s in os.listdir(data_dir) if s.startswith('sub-')])
#     print(f"Processing {len(subjects)} subjects.")

#     segment_sec = 10.0   # Window size for segmentation
#     rsa_results = []     # Store RSA results per subject
#     cond_labels = []     # Store condition labels

#     # ----------------------------
#     # 2) Load & Process Each Subject
#     # ----------------------------
#     for subj_id in subjects:
#         print(f"\nProcessing {subj_id}...")

#         # EEG Paths
#         eeg_narr_file = os.path.join(data_dir, f"{subj_id}", "ses-01", "eeg", f"{subj_id}_ses-01_task-dme_run-01_eeg.set")
#         eeg_non_narr_file  = os.path.join(data_dir, f"{subj_id}", "ses-01", "eeg", f"{subj_id}_ses-01_task-inscapes_eeg.set")

#         # fMRI Paths
#         fmri_narr_file = os.path.join(data_dir, f"{subj_id}", "ses-01", "func", f"{subj_id}_ses-01_task-dme_run-01_bold", "func_preproc", "func_pp_nofilt_sm0.mni152.3mm.nii.gz")
#         fmri_non_narr_file  = os.path.join(data_dir, f"{subj_id}", "ses-01", "func", f"{subj_id}_ses-01_task-inscapes_bold", "func_preproc", "func_pp_nofilt_sm0.mni152.3mm.nii.gz")

#         if any([not os.path.exists(f) for f in [eeg_narr_file, eeg_non_narr_file, fmri_narr_file, fmri_non_narr_file]]):
#             print(f"Skipping {subj_id}: missing data.")
#             continue

#         # Load EEG Data
#         eeg_narr_data, eeg_narr_srate = load_eeg_data(eeg_narr_file)
#         eeg_non_narr_data, eeg_non_narr_srate = load_eeg_data(eeg_non_narr_file)

#         # Load fMRI Data
#         fmri_narr_data, fmri_narr_srate = load_preprocessed_fmri(fmri_narr_file, TR=2.1)
#         fmri_non_narr_data, fmri_non_narr_srate = load_preprocessed_fmri(fmri_non_narr_file, TR=2.1)

#         # ----------------------------
#         # 3) Segment EEG & fMRI
#         # ----------------------------
#         eeg_narr_segments = segment_continuous_data(eeg_narr_data, eeg_narr_srate, segment_sec)
#         eeg_non_narr_segments = segment_continuous_data(eeg_non_narr_data, eeg_non_narr_srate, segment_sec)
#         fmri_narr_segments = segment_continuous_data(fmri_narr_data, fmri_narr_srate, segment_sec)
#         fmri_non_narr_segments = segment_continuous_data(fmri_non_narr_data, fmri_non_narr_srate, segment_sec)

#         # Ensure equal number of segments for both EEG & fMRI
#         n_seg = min(eeg_narr_segments.shape[0], fmri_narr_segments.shape[0], eeg_non_narr_segments.shape[0], fmri_non_narr_segments.shape[0])
#         eeg_narr_segments = eeg_narr_segments[:n_seg, :]
#         fmri_narr_segments = fmri_narr_segments[:n_seg, :]
#         eeg_non_narr_segments = eeg_non_narr_segments[:n_seg, :]
#         fmri_non_narr_segments = fmri_non_narr_segments[:n_seg, :]

#         # ----------------------------
#         # 4) Build RDMs & Compute RSA
#         # ----------------------------
#         eeg_narr_rdm = build_rdm(eeg_narr_segments, metric='correlation')
#         fmri_narr_rdm = build_rdm(fmri_narr_segments, metric='correlation')
#         eeg_non_narr_rdm = build_rdm(eeg_non_narr_segments, metric='correlation')
#         fmri_non_narr_rdm = build_rdm(fmri_non_narr_segments, metric='correlation')

#         c_narr, p_narr = cross_modal_rsa(eeg_narr_rdm, fmri_narr_rdm, method='spearman')
#         c_non_narr, p_non_narr = cross_modal_rsa(eeg_non_narr_rdm, fmri_non_narr_rdm, method='spearman')

#         rsa_results.append([subj_id, c_narr, p_narr, c_non_narr, p_non_narr])
#         cond_labels.extend([1] * n_seg + [0] * n_seg)  # 1=narrative, 0=non-narrative

#     # ----------------------------
#     # 5) Identify Top Networks & Classification
#     # ----------------------------
#     rsa_df = pd.DataFrame(rsa_results, columns=['Subject', 'Narrative_RSA', 'Narr_pval', 'NonNarr_RSA', 'NonNarr_pval'])
#     rsa_df.to_csv(os.path.join(out_dir, "rsa_results.csv"), index=False)

#     # Identify Top Networks
#     n_rois = 17
#     cross_modal_vals = np.random.rand(n_rois)  # Placeholder for ROI-level RSA
#     rois = [f"Network_{i+1}" for i in range(n_rois)]
#     top_5 = identify_top_rois(cross_modal_vals, rois, top_k=5)

#     print("\nTop 5 Networks by cross-modal RSA correlation:")
#     for label, val in top_5:
#         print(f"{label}: {val:.3f}")

#     # Classification
#     n_segments_total = len(cond_labels)
#     rsa_features = np.random.randn(n_segments_total, n_rois)  # Simulated RSA features per ROI
#     cond_labels = np.array(cond_labels)

#     acc = classify_narr_vs_nonnarr(rsa_features, cond_labels)
#     print(f"\nDiscrimination analysis accuracy: {acc:.3f}")

#     print("\nCross-modal RSA completed for all subjects.")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import nibabel as nib
import mne
from nilearn import datasets, image, masking
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_1samp, ttest_rel

from cross_modal_rsa import (
    load_eeg_data,
    load_preprocessed_fmri,
    segment_continuous_data,
    build_rdm,
    cross_modal_rsa,
    classify_narr_vs_nonnarr
)

def main():
    # ----------------------------
    # 1) Load Subjects & Directories
    # ----------------------------
    data_dir = "Preprocessed_Data"
    out_dir = "Results"
    os.makedirs(out_dir, exist_ok=True)

    subjects = sorted([s for s in os.listdir(data_dir) if s.startswith('sub-')])
    print(f"Processing {len(subjects)} subjects.")

    segment_sec = 10.0  # Default window size for segmentation
    rsa_narr_list = []  # Store RSA values for narrative (one per subject)
    rsa_non_list = []   # Store RSA values for non-narrative (one per subject)

    # ----------------------------
    # 2) Compute RSA for Each Subject Separately
    # ----------------------------
    for subj_id in subjects:
        print(f"\nProcessing {subj_id}...")

        # EEG Paths
        eeg_narr_file = os.path.join(data_dir, subj_id, "ses-01", "eeg", f"{subj_id}_ses-01_task-dme_run-01_eeg.set")
        eeg_non_narr_file = os.path.join(data_dir, subj_id, "ses-01", "eeg", f"{subj_id}_ses-01_task-inscapes_eeg.set")

        # fMRI Paths
        fmri_narr_file = os.path.join(data_dir, subj_id, "ses-01", "func", f"{subj_id}_ses-01_task-dme_run-01_bold", "func_preproc", "func_pp_nofilt_sm0.mni152.3mm.nii.gz")
        fmri_non_narr_file = os.path.join(data_dir, subj_id, "ses-01", "func", f"{subj_id}_ses-01_task-inscapes_bold", "func_preproc", "func_pp_nofilt_sm0.mni152.3mm.nii.gz")

        if not all(os.path.exists(f) for f in [eeg_narr_file, eeg_non_narr_file, fmri_narr_file, fmri_non_narr_file]):
            print(f"⚠️ Skipping {subj_id}: missing data.")
            continue

        # Load EEG Data
        eeg_narr_data, eeg_narr_srate = load_eeg_data(eeg_narr_file)
        eeg_non_narr_data, eeg_non_narr_srate = load_eeg_data(eeg_non_narr_file)

        # Load fMRI Data
        fmri_narr_data, fmri_narr_srate = load_preprocessed_fmri(fmri_narr_file, TR=2.1)
        fmri_non_narr_data, fmri_non_narr_srate = load_preprocessed_fmri(fmri_non_narr_file, TR=2.1)

        # ----------------------------
        # 3) Segment EEG & fMRI
        # ----------------------------
        eeg_narr_segments = segment_continuous_data(eeg_narr_data, eeg_narr_srate, segment_sec)
        eeg_non_narr_segments = segment_continuous_data(eeg_non_narr_data, eeg_non_narr_srate, segment_sec)
        fmri_narr_segments = segment_continuous_data(fmri_narr_data, fmri_narr_srate, segment_sec)
        fmri_non_narr_segments = segment_continuous_data(fmri_non_narr_data, fmri_non_narr_srate, segment_sec)

        # Ensure equal number of segments
        n_seg = min(eeg_narr_segments.shape[0], fmri_narr_segments.shape[0], eeg_non_narr_segments.shape[0], fmri_non_narr_segments.shape[0])
        eeg_narr_segments = eeg_narr_segments[:n_seg, :]
        fmri_narr_segments = fmri_narr_segments[:n_seg, :]
        eeg_non_narr_segments = eeg_non_narr_segments[:n_seg, :]
        fmri_non_narr_segments = fmri_non_narr_segments[:n_seg, :]

        # ----------------------------
        # 4) Compute RSA Per Subject
        # ----------------------------
        eeg_narr_rdm = build_rdm(eeg_narr_segments, metric='correlation')
        fmri_narr_rdm = build_rdm(fmri_narr_segments, metric='correlation')
        eeg_non_narr_rdm = build_rdm(eeg_non_narr_segments, metric='correlation')
        fmri_non_narr_rdm = build_rdm(fmri_non_narr_segments, metric='correlation')

        c_narr, _ = cross_modal_rsa(eeg_narr_rdm, fmri_narr_rdm, method='spearman')
        c_non_narr, _ = cross_modal_rsa(eeg_non_narr_rdm, fmri_non_narr_rdm, method='spearman')

        rsa_narr_list.append(c_narr)
        rsa_non_list.append(c_non_narr)

    # ----------------------------
    # 5) Compute Group-Level RSA & Statistical Significance
    # ----------------------------
    mean_rsa_narr = np.nanmean(rsa_narr_list)
    mean_rsa_non = np.nanmean(rsa_non_list)

    # Perform a one-sample t-test for statistical significance
    t_stat_narr, p_val_narr = ttest_1samp(rsa_narr_list, 0, nan_policy='omit')
    t_stat_non, p_val_non = ttest_1samp(rsa_non_list, 0, nan_policy='omit')

    print(f"\nGroup-Level EEG-fMRI RSA (Narrative): r={mean_rsa_narr:.3f}, p={p_val_narr:.4g}")
    print(f"Group-Level EEG-fMRI RSA (Non-Narrative): r={mean_rsa_non:.3f}, p={p_val_non:.4g}")
    
    # Suppose rsa_narr_list and rsa_non_list each have N entries,
    # where each entry = RSA for one subject in that condition.

    t_stat, p_val = ttest_rel(rsa_narr_list, rsa_non_list, nan_policy='omit')
    print(f"Paired T-test comparing Narrative vs. Non-Narrative: t={t_stat}, p={p_val}")

    # If RSA is stronger for narrative segments, it means that representations in EEG and fMRI align more strongly during structured, story-driven stimuli.
    # If non-narrative has weak or inconsistent RSA, it suggests less neural coordination across modalities for fragmented or incoherent stimuli.

    # ----------------------------
    # 6) RSA-Based Classification (Using Individual Subject RSA Scores)
    # ----------------------------
    rsa_array = np.array([[c_narr] for c_narr in rsa_narr_list] + [[c_non] for c_non in rsa_non_list])
    cond_labels = np.array([1] * len(rsa_narr_list) + [0] * len(rsa_non_list))  # 1 = Narrative, 0 = Non-Narrative

    acc = classify_narr_vs_nonnarr(rsa_array, cond_labels)
    print(f"\nClassification Accuracy using RSA: {acc:.3f}")

    print("\nCross-modal RSA completed at the group level.")

if __name__ == "__main__":
    main()
