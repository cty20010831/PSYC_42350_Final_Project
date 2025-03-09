# import os
# import numpy as np
# import pandas as pd
# import nibabel as nib
# import mne
# from nilearn import datasets, image, masking
# from statsmodels.stats.multitest import multipletests
# from scipy.stats import ttest_1samp, ttest_rel

# from cross_modal_rsa import (
#     load_eeg_data,
#     load_preprocessed_fmri,
#     segment_continuous_data,
#     build_rdm,
#     plot_rdm,
#     cross_modal_rsa,
#     classify_narr_vs_nonnarr
# )

# def main():
#     # ----------------------------
#     # 1) Load Subjects & Directories
#     # ----------------------------
#     data_dir = "Preprocessed_Data"
#     out_dir = "Results"
#     os.makedirs(out_dir, exist_ok=True)

#     subjects = sorted([s for s in os.listdir(data_dir) if s.startswith('sub-')])
#     print(f"Processing {len(subjects)} subjects.")

#     segment_sec = 10.0  # Default window size for segmentation
#     eeg_narr_rdm_list = []  # Store individual EEG RDMs for narrative condition
#     fmri_narr_rdm_list = [] # Store individual fMRI RDMs for narrative condition
#     eeg_non_narr_rdm_list = []  # Store individual EEG RDMs for non-narrative condition
#     fmri_non_narr_rdm_list = [] # Store individual fMRI RDMs for narrative condition
#     rsa_narr_list = []  # Store RSA values for narrative (one per subject)
#     rsa_non_list = []   # Store RSA values for non-narrative (one per subject)

#     # ----------------------------
#     # 2) Compute RSA for Each Subject Separately
#     # ----------------------------
#     for subj_id in subjects:
#         print(f"\nProcessing {subj_id}...")

#         # EEG Paths
#         eeg_narr_file = os.path.join(data_dir, subj_id, "ses-01", "eeg", f"{subj_id}_ses-01_task-dme_run-01_eeg.set")
#         eeg_non_narr_file = os.path.join(data_dir, subj_id, "ses-01", "eeg", f"{subj_id}_ses-01_task-inscapes_eeg.set")

#         # fMRI Paths
#         fmri_narr_file = os.path.join(data_dir, subj_id, "ses-01", "func", f"{subj_id}_ses-01_task-dme_run-01_bold", "func_preproc", "func_pp_nofilt_sm0.mni152.3mm.nii.gz")
#         fmri_non_narr_file = os.path.join(data_dir, subj_id, "ses-01", "func", f"{subj_id}_ses-01_task-inscapes_bold", "func_preproc", "func_pp_nofilt_sm0.mni152.3mm.nii.gz")

#         if not all(os.path.exists(f) for f in [eeg_narr_file, eeg_non_narr_file, fmri_narr_file, fmri_non_narr_file]):
#             print(f"⚠️ Skipping {subj_id}: missing data.")
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

#         # Ensure equal number of segments
#         n_seg = min(eeg_narr_segments.shape[0], fmri_narr_segments.shape[0], eeg_non_narr_segments.shape[0], fmri_non_narr_segments.shape[0])
#         eeg_narr_segments = eeg_narr_segments[:n_seg, :]
#         fmri_narr_segments = fmri_narr_segments[:n_seg, :]
#         eeg_non_narr_segments = eeg_non_narr_segments[:n_seg, :]
#         fmri_non_narr_segments = fmri_non_narr_segments[:n_seg, :]

#         # ----------------------------
#         # 4) Compute RSA Per Subject
#         # ----------------------------
#         eeg_narr_rdm = build_rdm(eeg_narr_segments, metric='correlation')
#         fmri_narr_rdm = build_rdm(fmri_narr_segments, metric='correlation')
#         eeg_non_narr_rdm = build_rdm(eeg_non_narr_segments, metric='correlation')
#         fmri_non_narr_rdm = build_rdm(fmri_non_narr_segments, metric='correlation')

#         # Append individual RDMs for later averaging (narrative condition)
#         eeg_narr_rdm_list.append(eeg_narr_rdm)
#         fmri_narr_rdm_list.append(fmri_narr_rdm)
#         eeg_non_narr_rdm_list.append(eeg_non_narr_rdm)
#         fmri_non_narr_rdm_list.append(fmri_non_narr_rdm)

#         c_narr, _ = cross_modal_rsa(eeg_narr_rdm, fmri_narr_rdm, method='spearman')
#         c_non_narr, _ = cross_modal_rsa(eeg_non_narr_rdm, fmri_non_narr_rdm, method='spearman')

#         rsa_narr_list.append(c_narr)
#         rsa_non_list.append(c_non_narr)

#     # ----------------------------
#     # 5) Compute and plot average RDMs across subjects for both narrative and 
#     # non-narrative conditions
#     # ----------------------------
#     # Narratives
#     avg_eeg_narr_rdm = np.mean(np.array(eeg_narr_rdm_list), axis=0)
#     avg_fmri_narr_rdm = np.mean(np.array(fmri_narr_rdm_list), axis=0)
#     plot_rdm(avg_eeg_narr_rdm, title="Average EEG RDM - Narrative", save_path=os.path.join(out_dir, "Avg_EEG_RDM_narr.png"))
#     plot_rdm(avg_fmri_narr_rdm, title="Average fMRI RDM - Narrative", save_path=os.path.join(out_dir, "Avg_fMRI_RDM_narr.png"))
    
#     # Non-Narratives
#     avg_eeg_non_narr_rdm = np.mean(np.array(eeg_non_narr_rdm), axis=0)
#     avg_fmri_non_narr_rdm = np.mean(np.array(fmri_non_narr_rdm), axis=0)
#     plot_rdm(avg_eeg_non_narr_rdm, title="Average EEG RDM - Non-Narrative", save_path=os.path.join(out_dir, "Avg_EEG_RDM_non_narr.png"))
#     plot_rdm(avg_fmri_non_narr_rdm, title="Average fMRI RDM - Non-Narrative", save_path=os.path.join(out_dir, "Avg_fMRI_RDM_non_narr.png"))
    
#     # ----------------------------
#     # 6) Compute Group-Level RSA & Statistical Significance
#     # ----------------------------
#     mean_rsa_narr = np.nanmean(rsa_narr_list)
#     mean_rsa_non = np.nanmean(rsa_non_list)

#     # Perform a one-sample t-test for statistical significance
#     t_stat_narr, p_val_narr = ttest_1samp(rsa_narr_list, 0, nan_policy='omit')
#     t_stat_non, p_val_non = ttest_1samp(rsa_non_list, 0, nan_policy='omit')

#     print(f"\nGroup-Level EEG-fMRI RSA (Narrative): r={mean_rsa_narr:.3f}, p={p_val_narr:.4g}")
#     print(f"Group-Level EEG-fMRI RSA (Non-Narrative): r={mean_rsa_non:.3f}, p={p_val_non:.4g}")
    
#     # Suppose rsa_narr_list and rsa_non_list each have N entries,
#     # where each entry = RSA for one subject in that condition.

#     t_stat, p_val = ttest_rel(rsa_narr_list, rsa_non_list, nan_policy='omit')
#     print(f"Paired T-test comparing Narrative vs. Non-Narrative: t={t_stat}, p={p_val}")

#     # If RSA is stronger for narrative segments, it means that representations in EEG and fMRI align more strongly during structured, story-driven stimuli.
#     # If non-narrative has weak or inconsistent RSA, it suggests less neural coordination across modalities for fragmented or incoherent stimuli.

#     # ----------------------------
#     # 7) RSA-Based Classification (Using Individual Subject RSA Scores)
#     # ----------------------------
#     rsa_array = np.array([[c_narr] for c_narr in rsa_narr_list] + [[c_non] for c_non in rsa_non_list])
#     cond_labels = np.array([1] * len(rsa_narr_list) + [0] * len(rsa_non_list))  # 1 = Narrative, 0 = Non-Narrative

#     acc = classify_narr_vs_nonnarr(rsa_array, cond_labels)
#     print(f"\nClassification Accuracy using RSA: {acc:.3f}")

#     print("\nCross-modal RSA completed at the group level.")

# if __name__ == "__main__":
#     main()

import os
import numpy as np
import pandas as pd
import nibabel as nib
import mne
from nilearn import datasets, image, masking
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

from cross_modal_rsa import (
    load_eeg_data,
    load_preprocessed_fmri,
    segment_continuous_data,
    build_rdm,
    plot_rdm,
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
    eeg_narr_rdm_list = []  # Store individual EEG RDMs for narrative condition
    fmri_narr_rdm_list = [] # Store individual fMRI RDMs for narrative condition
    eeg_non_narr_rdm_list = []  # Store individual EEG RDMs for non-narrative condition
    fmri_non_narr_rdm_list = [] # Store individual fMRI RDMs for non-narrative condition
    rsa_narr_list = []  # Store RSA values for narrative (one per subject)
    rsa_non_list = []   # Store RSA values for non-narrative (one per subject)
    seg_counts = []     # To store the number of segments for each subject

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

        # Determine number of segments for current subject
        n_seg = min(eeg_narr_segments.shape[0], fmri_narr_segments.shape[0],
                    eeg_non_narr_segments.shape[0], fmri_non_narr_segments.shape[0])
        seg_counts.append(n_seg)

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

        # Append individual RDMs for later averaging
        eeg_narr_rdm_list.append(eeg_narr_rdm)
        fmri_narr_rdm_list.append(fmri_narr_rdm)
        eeg_non_narr_rdm_list.append(eeg_non_narr_rdm)
        fmri_non_narr_rdm_list.append(fmri_non_narr_rdm)

        # Compute cross-modal RSA for narrative and non-narrative conditions
        c_narr, _ = cross_modal_rsa(eeg_narr_rdm, fmri_narr_rdm, method='spearman')
        c_non_narr, _ = cross_modal_rsa(eeg_non_narr_rdm, fmri_non_narr_rdm, method='spearman')
        rsa_narr_list.append(c_narr)
        rsa_non_list.append(c_non_narr)

    # ----------------------------
    # 5) Adjust RDMs to a Common Shape and Plot Average RDMs
    # ----------------------------
    # Find the global minimum number of segments across subjects
    global_min_seg = min(rdm.shape[0] for rdm in eeg_narr_rdm_list)
    print(f"Global minimum segments across subjects: {global_min_seg}")
    
    # Resize each subject's RDM to (global_min_seg, global_min_seg)
    eeg_narr_rdm_list_resized = [rdm[:global_min_seg, :global_min_seg] for rdm in eeg_narr_rdm_list]
    fmri_narr_rdm_list_resized = [rdm[:global_min_seg, :global_min_seg] for rdm in fmri_narr_rdm_list]
    eeg_non_narr_rdm_list_resized = [rdm[:global_min_seg, :global_min_seg] for rdm in eeg_non_narr_rdm_list]
    fmri_non_narr_rdm_list_resized = [rdm[:global_min_seg, :global_min_seg] for rdm in fmri_non_narr_rdm_list]
    
    # Compute average RDMs across subjects for narrative condition
    avg_eeg_narr_rdm = np.mean(np.array(eeg_narr_rdm_list_resized), axis=0)
    avg_fmri_narr_rdm = np.mean(np.array(fmri_narr_rdm_list_resized), axis=0)
    
    plot_rdm(avg_eeg_narr_rdm, title="Average EEG RDM - Narrative", 
             save_path=os.path.join(out_dir, "Plots", "Avg_EEG_RDM_narr.png"))
    plot_rdm(avg_fmri_narr_rdm, title="Average fMRI RDM - Narrative", 
             save_path=os.path.join(out_dir, "Plots", "Avg_fMRI_RDM_narr.png"))
    
    # Similarly, compute average RDMs for non-narrative condition if desired
    avg_eeg_non_narr_rdm = np.mean(np.array(eeg_non_narr_rdm_list_resized), axis=0)
    avg_fmri_non_narr_rdm = np.mean(np.array(fmri_non_narr_rdm_list_resized), axis=0)
    
    plot_rdm(avg_eeg_non_narr_rdm, title="Average EEG RDM - Non-Narrative", 
             save_path=os.path.join(out_dir, "Plots", "Avg_EEG_RDM_non_narr.png"))
    plot_rdm(avg_fmri_non_narr_rdm, title="Average fMRI RDM - Non-Narrative", 
             save_path=os.path.join(out_dir, "Plots", "Avg_fMRI_RDM_non_narr.png"))
    
    # ----------------------------
    # 6) Compute Group-Level RSA & Statistical Significance
    # ----------------------------
    mean_rsa_narr = np.nanmean(rsa_narr_list)
    mean_rsa_non = np.nanmean(rsa_non_list)
    t_stat_narr, p_val_narr = ttest_1samp(rsa_narr_list, 0, nan_policy='omit')
    t_stat_non, p_val_non = ttest_1samp(rsa_non_list, 0, nan_policy='omit')
    print(f"\nGroup-Level EEG-fMRI RSA (Narrative): r={mean_rsa_narr:.3f}, p={p_val_narr:.4g}")
    print(f"Group-Level EEG-fMRI RSA (Non-Narrative): r={mean_rsa_non:.3f}, p={p_val_non:.4g}")
    t_stat, p_val = ttest_rel(rsa_narr_list, rsa_non_list, nan_policy='omit')
    print(f"Paired T-test comparing Narrative vs. Non-Narrative: t={t_stat}, p={p_val}")
    
    rsa_array = np.array([[c] for c in rsa_narr_list] + [[c] for c in rsa_non_list])
    cond_labels = np.array([1] * len(rsa_narr_list) + [0] * len(rsa_non_list))
    acc = classify_narr_vs_nonnarr(rsa_array, cond_labels)
    print(f"\nClassification Accuracy using RSA: {acc:.3f}")
    
    print("\nCross-modal RSA completed at the group level.")

if __name__ == "__main__":
    main()