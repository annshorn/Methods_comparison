# Methods_comparison

This repository provides a comparison of various fixation detection algorithms in the context of two prediction tasks: radiologist expertise estimation and diagnostic error prediction.

## Repository Structure

**fix_alg/** — Rule-based fixation detection methods, implemented following Salvucci and Goldberg (2000). This folder includes `I-VT.py`, which implements the velocity-threshold identification algorithm as described by Olsen (2012).

**1DCNN-BLSTM-GazeCom/, 1DCNN-BLSTM-HMR/, 1DCNN-LSTM-GazeCom/, 1DCNN-TCM-GazeCom/, 1DCNN-TCM-HMR/** — Deep learning approaches for fixation, saccade, and smooth pursuit classification. These implementations are based on Agtzidis et al. (2016), Startsev et al. (2018), and Elmadjian et al. (2023). The original code and evaluation results are available at http://www.michaeldorr.de/smoothpursuit. The code has been slightly modified to align with our dataset.

**expertise_est/** — Code for radiologist expertise prediction task.

**error_pred/** — Code for diagnostic error prediction task.

**utils/** — Utility scripts including `create_raw_data.py` for data preprocessing.

## Getting Started

Begin by running `utils/create_raw_data.py` to prepare the data. Note that this script was created specifically for our dataset and may require adaptation for other data sources.

## References

D. D. Salvucci and J. H. Goldberg. Identifying fixations and saccades in eye-tracking protocols. In *Proceedings of the 2000 Symposium on Eye Tracking Research & Applications*, ETRA '00, pages 71–78, New York, NY, USA, 2000. Association for Computing Machinery.

A. Olsen. The Tobii I-VT fixation filter algorithm description. *Tobii Technology*, 21:4–19, 2012.

I. Agtzidis, M. Startsev, and M. Dorr. Smooth pursuit detection based on multiple observers. In *Proceedings of the Ninth Biennial ACM Symposium on Eye Tracking Research & Applications*, pages 303–306. ACM, 2016.

M. Startsev, I. Agtzidis, and M. Dorr. 1D CNN with BLSTM for automated classification of fixations, saccades, and smooth pursuits. *Behavior Research Methods*, Nov 2018.

C. Elmadjian, C. Gonzales, R. L. d. Costa, et al. Online eye-movement classification with temporal convolutional networks. *Behavior Research Methods*, 55:3602–3620, 2023.
