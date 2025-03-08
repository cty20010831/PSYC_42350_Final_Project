"""
This script only keeps files related to Inscapes and Despicable Me (English) [Run 1] in session 1. 
"""

import os
import shutil

root_dir = 'Preprocessed_Data'

# List of tasks to keep for fMRI and EEG data
fmri_keep = ['dme_run-01', 'inscapes']
eeg_keep = ['dme_run-01', 'inscapes']

for subj in os.listdir(root_dir):
    subj_path = os.path.join(root_dir, subj)

    if not os.path.isdir(subj_path):
        continue

    for ses in os.listdir(subj_path):
        ses_path = os.path.join(subj_path, ses)
        if ses == '.DS_Store':
            continue

        if ses != 'ses-01':
            # Remove sessions other than session 1
            print(f'Removing {ses_path} directory')
            shutil.rmtree(ses_path)
        else:
            # fMRI data cleanup
            fmri_path = os.path.join(ses_path, 'func')
            if os.path.exists(fmri_path):
                for fname in os.listdir(fmri_path):
                    if fname == '.DS_Store':
                        continue
                    
                    if not any(task in fname for task in fmri_keep):
                        fpath = os.path.join(fmri_path, fname)
                        print(f'Removing directory: {fpath}')
                        shutil.rmtree(fpath)

            # EEG data cleanup
            eeg_path = os.path.join(ses_path, 'eeg')
            if os.path.exists(eeg_path):
                for fname in os.listdir(eeg_path):
                    if fname == '.DS_Store':
                        continue

                    if not any(task in fname for task in eeg_keep):
                        fpath = os.path.join(eeg_path, fname)
                        print(f'Removing EEG file: {fpath}')
                        os.remove(fpath)

print("Cleanup done!")