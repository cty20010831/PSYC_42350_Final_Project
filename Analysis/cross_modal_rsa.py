import numpy as np
import nibabel as nib
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

def load_eeg_data(eeg_file):
    """Load preprocessed EEG from EEGLAB .set using MNE."""
    # Load raw EEG data
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
    
    # Fix Annotation Warning (crop annotations to valid time range)
    if raw.annotations:
        raw.set_annotations(raw.annotations.crop(0, raw.times[-1], verbose=False))
    
    # Handle Boundary Events (remove or redefine)
    events, event_id = mne.events_from_annotations(raw)
    if 'boundary' in event_id:
        del event_id['boundary']  # Remove boundary events

    # Select only EEG channels (ignore EOG, ECG, etc.)
    raw.pick("eeg")

     # Get EEG data & sampling rate
    data = raw.get_data().T  # shape (n_time, n_channels)
    sample_rate = raw.info['sfreq']  # Get the sampling rate (e.g., 5000 Hz)

    print(f"EEG data shape: {data.shape}, Sampling Rate: {sample_rate} Hz")
    
    return data, sample_rate

def load_preprocessed_fmri(fmri_file, TR=2.1):
    """Load a 4D preprocessed fMRI NIfTI, flatten to (n_time, n_vox)."""
    img_4d = nib.load(fmri_file)
    data_4d = img_4d.get_fdata()  # shape (x, y, z, t)
    x, y, z, t = data_4d.shape
    data_2d = data_4d.reshape((x*y*z, t)).T  # => (t, n_vox)
    sample_rate_fmri = 1.0 / TR  # ~0.476 Hz
    return data_2d, sample_rate_fmri

def segment_continuous_data(data, sample_rate, segment_length_sec):
    """
    Chop continuous data (time x features) into fixed-length segments (e.g. 10-second windows).

    Parameters
    ----------
    data : np.ndarray
        shape (n_timepoints, n_features). For EEG, (time, channels/freq-bands).
        For fMRI, (time, voxels/ROIs) if you treat each volume as one sample.
    sample_rate : float
        Samples per second (EEG might be 5000 Hz; fMRI might be 1/TR=~0.476).
    segment_length_sec : float
        Duration (in seconds) of each segment.

    Returns
    -------
    segment_patterns : np.ndarray
        shape (n_segments, n_features). Each row = average pattern for that segment.
    """
    n_time, n_features = data.shape
    samples_per_segment = int(np.floor(sample_rate * segment_length_sec))
    if samples_per_segment < 1:
        raise ValueError("Segment length too short for given sample_rate.")

    n_segments = n_time // samples_per_segment
    segs = []
    for seg_idx in range(n_segments):
        start = seg_idx * samples_per_segment
        end = (seg_idx + 1) * samples_per_segment
        seg_data = data[start:end, :]
        avg_pattern = np.mean(seg_data, axis=0)
        segs.append(avg_pattern)

    return np.array(segs)  # shape => (n_segments, n_features)

def build_rdm(segment_data, metric='correlation'):
    """
    Build an RDM from segment-level data.

    Parameters
    ----------
    segment_data : np.ndarray (n_segments, n_features)
    metric : str, e.g. 'correlation', 'euclidean'

    Returns
    -------
    rdm : (n_segments, n_segments) distance matrix
    """
    dist_vec = pdist(segment_data, metric=metric)
    rdm = squareform(dist_vec)
    return rdm

def plot_rdm(rdm, title='RDM', save_path=None):
    """
    Plot the Representational Dissimilarity Matrix (RDM) using Seaborn.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(rdm, cmap='viridis', square=True, cbar=True)
    plt.title(title)
    plt.xlabel('Segment')
    plt.ylabel('Segment')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def cross_modal_rsa(rdm1, rdm2, method='spearman'):
    """
    Cross-modal RSA: correlate upper triangles of two RDMs.

    rdm1, rdm2 : (n_segments, n_segments)
    method: 'spearman' or 'pearson'

    Returns
    -------
    rsa_corr : float
    rsa_pval : float
    """
    assert rdm1.shape == rdm2.shape, "RDM shapes must match!"
    n_seg = rdm1.shape[0]

    iu = np.triu_indices(n_seg, k=1)
    vec1 = rdm1[iu]
    vec2 = rdm2[iu]

    if method == 'spearman':
        rsa_corr, rsa_p = spearmanr(vec1, vec2)
    else:
        rsa_corr, rsa_p = pearsonr(vec1, vec2)

    return rsa_corr, rsa_p


def identify_top_rois(rsa_values, labels, top_k=5):
    """
    Suppose rsa_values is a list/array of RSA correlation values for each ROI or network.
    Return the top_k indices (or labels) with highest correlation.

    Parameters
    ----------
    rsa_values : np.ndarray
        shape (n_rois,)
    labels : list of rois (e.g. Yeo network IDs) of same length
    top_k : int, how many to pick

    Returns
    -------
    top_list : list of (label, rsa_val) sorted descending
    """
    idx_sorted = np.argsort(rsa_values)[::-1]  # descending
    top_idx = idx_sorted[:top_k]
    top_list = [(labels[i], rsa_values[i]) for i in top_idx]
    return top_list


def classify_narr_vs_nonnarr(rsa_features, condition_labels):
    """
    Example classification of narrative vs. non-narrative segments
    given RSA features.

    rsa_features : (n_segments, n_features)
    condition_labels : (n_segments,) -> 0 = non-narr, 1 = narr

    Returns
    -------
    mean_acc : float
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = SVC(kernel='linear')
    accuracies = []

    for train_idx, test_idx in skf.split(rsa_features, condition_labels):
        clf.fit(rsa_features[train_idx], condition_labels[train_idx])
        acc = clf.score(rsa_features[test_idx], condition_labels[test_idx])
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    
    return mean_acc