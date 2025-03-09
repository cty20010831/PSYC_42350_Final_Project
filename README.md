# PSYC 42350 Final Project

This repository contains the code, analysis scripts, and results for a final project in the University of Chicago’s PSYC 42350 course. The project investigates how narrative content modulates neural synchrony using simultaneous EEG-fMRI data from an open-access dataset published by Telesford et al. (2023). Two main analytical approaches—inter-subject correlation (ISC) and representational similarity analysis (RSA)—were applied to examine whether coherent narratives evoke stronger neural alignment than non-narrative stimuli.

---

## Data
I used an open-access dataset from [Telesford et al. (2023)](https://fcon_1000.projects.nitrc.org/indi/retro/nat_view.html), which includes EEG and fMRI recordings of participants viewing both narrative (e.g., *Despicable Me*) and non-narrative (e.g., *Inscapes*) video clips.  
- [**Data Portal**](https://fcon_1000.projects.nitrc.org/indi/retro/nat_view.html)  
- [**Direct Download Links**](https://fcon_1000.projects.nitrc.org/indi/retro/NAT_VIEW/nat_view_links.html)

---

## Research Questions
1. **ISC:** Do coherent narratives evoke stronger inter-subject correlation in fMRI compared to non-narrative stimuli?  
2. **Cross-Modal RSA:** Does cross-modal EEG–fMRI representational similarity differ between narrative and non-narrative conditions, and can these multimodal signals classify whether a segment is narrative vs. non-narrative?

---

## Methods
1. **Dataset Preprocessing:**  
   - 22 healthy adults (ages 23–51) with no history of psychiatric/neurological illness.  
   - EEG recorded at ~5000 Hz, fMRI at 3T with TR=2.1 s.  
   - Preprocessing (motion correction, MNI alignment, etc.) done with the Connectome Computation System (CCS).  

2. **Segmentation & Analyses:**  
   - **Segmentation:** Both EEG and fMRI time series were segmented into 10-second windows, creating averaged neural patterns per segment.  
   - **ISC:** Only fMRI data were used, correlating each subject’s voxel-wise time course with the mean of all other subjects. ISC maps were aggregated within the 17 Yeo networks.  
   - **RSA:** Constructed representational dissimilarity matrices (RDMs) for EEG and fMRI (using 1 – Pearson correlation) and computed cross-modal alignment.  
   - **Classification:** A linear SVM was used to classify narrative vs. non-narrative segments based on RSA features.

---

## Results
1. **ISC (fMRI):**  
   - Narratives induced significantly higher inter-subject correlation across 16 of the 17 Yeo networks, particularly in high-level cortical networks such as the Control/Frontoparietal and Default Mode networks.  
   - A repeated-measures ANOVA confirmed a robust effect of network on the ISC difference (Narrative minus Non-Narrative).

2. **Cross-Modal RSA (EEG–fMRI):**  
   - Group-level RSA showed a slight positive trend for narratives (r = 0.003) vs. non-narratives (r = –0.009), but the difference was not statistically significant (paired t-test p = 0.148).  
   - Classification accuracy using RSA features was 52.8%, only marginally above chance.  

These findings indicate that while narratives strongly synchronize brain activity across individuals in fMRI, the cross-modal representational alignment (EEG vs. fMRI) remains modest under current analytic parameters.

---

## Virtual Environment

To replicate this analysis, you can use the provided `requirements.txt` in a Python 3.11 environment:

```bash
# Create the virtual environment:
python3.11 -m venv venv

# Activate the virtual environment:
source venv/bin/activate

# Install required packages:
python3 -m pip install -r requirements.txt