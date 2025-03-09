import os
import nibabel as nib
from nilearn import datasets, image, masking

def list_subject_files(
    data_dir, 
    task_label, 
    file_name='func_pp_nofilt_sm0.mni152.3mm.nii.gz'
):
    """
    Return a list of file paths (4D NIfTI) for all subjects for a given task.
    """
    subjects = sorted([s for s in os.listdir(data_dir) if s.startswith('sub-')])
    file_paths = []

    for subj in subjects:
        img_path = os.path.join(
            data_dir,
            subj,
            'ses-01',
            'func',
            f'{subj}_ses-01_task-{task_label}_bold',
            'func_preproc',
            file_name
        )
        if os.path.exists(img_path):
            file_paths.append(img_path)

    return file_paths

def load_and_mask_fmri(
    data_dir,
    narrative_task='dme_run-01',
    non_narrative_task='inscapes',
    file_name='func_pp_nofilt_sm0.mni152.3mm.nii.gz',
    resolution=3
):
    """
    Loads two sets of fMRI data (e.g. narrative vs non-narrative) from a given directory,
    resamples an MNI template mask to 3mm, and applies it, returning masked arrays.
    
    Returns:
    --------
    narrative_masked : list of np.ndarray
        A list of shape (n_voxels, n_timepoints) for each subject's narrative data.
    non_narrative_masked : list of np.ndarray
        A list of shape (n_voxels, n_timepoints) for each subject's non-narrative data.
    final_mask : nib.Nifti1Image
        The 3mm MNI mask in the same space/affine as your reference image.
    """

    # 1) Gather file paths
    narrative_files = list_subject_files(data_dir, narrative_task, file_name)
    non_narrative_files = list_subject_files(data_dir, non_narrative_task, file_name)

    # 2) Load nibabel images
    narrative_data = [nib.load(f) for f in narrative_files]
    non_narrative_data = [nib.load(f) for f in non_narrative_files]

    if len(narrative_data) == 0:
        raise ValueError(f"No narrative data found for task={narrative_task} in {data_dir}.")

    # 3) Load MNI template mask (2mm or 1mm by default) at desired resolution=3
    template_mask = datasets.load_mni152_brain_mask(resolution=resolution)

    # 4) Resample the template mask to match the 4D reference image
    reference_img = narrative_data[0]
    final_mask = image.resample_to_img(
        source_img=template_mask,
        target_img=reference_img,
        interpolation='nearest'
    )

    # 5) Mask => shape (n_timepoints, n_voxels), then transpose
    narrative_masked = [
        masking.apply_mask(img, final_mask).T  # => (n_voxels, n_timepoints)
        for img in narrative_data
    ]
    non_narrative_masked = [
        masking.apply_mask(img, final_mask).T
        for img in non_narrative_data
    ]

    return narrative_masked, non_narrative_masked, final_mask