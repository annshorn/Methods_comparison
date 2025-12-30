This repository provides a comparison of various fixation detection algorithms in the context of two prediction tasks: radiologist expertise estimation and diagnostic error prediction.

## Repository Structure

**fix_alg/** — Rule-based fixation detection methods, implemented following Salvucci and Goldberg (2000). This folder includes `I-VT.py`, which implements the velocity-threshold identification algorithm as described by Olsen (2012).

**1DCNN-BLSTM-GazeCom/, 1DCNN-BLSTM-HMR/, 1DCNN-LSTM-GazeCom/, 1DCNN-TCM-GazeCom/, 1DCNN-TCM-HMR/** — Deep learning approaches for fixation, saccade, and smooth pursuit classification. These implementations are based on Agtzidis et al. (2016), Startsev et al. (2018), and Elmadjian et al. (2023). The original code and evaluation results are available at http://www.michaeldorr.de/smoothpursuit. The code has been slightly modified to align with our dataset.

**expertise_est/** — Code for radiologist expertise prediction task.

**error_pred/** — Code for diagnostic error prediction task.

**utils/** — Utility scripts including `create_raw_data.py` for data preprocessing.

## Getting Started

Begin by running `utils/create_raw_data.py` to prepare the data. Note that this script was created specifically for our dataset and may require adaptation for other data sources.

## Results

### Reader's Expertise Estimation

Performance metrics (accuracy and AUC) for the expertise estimation task. Fixation points were extracted using ten different fixation identification algorithms to investigate how the choice of algorithm influences performance. Features derived from these points were evaluated using four models commonly applied in gaze analysis. Quality columns represent Z-score-based rankings.

|                       | Random Forest |       |         | SVM   |       |         | LSTM      |       |         | Transformer |       |         |
|-----------------------|---------------|-------|---------|-------|-------|---------|-----------|-------|---------|-------------|-------|---------|
|                       | Acc           | AUC   | Quality | Acc   | AUC   | Quality | Acc       | AUC   | Quality | Acc         | AUC   | Quality |
| I-VT                  | 0.781         | 0.847 | +1.65   | 0.774 | 0.809 | +0.83   | 0.894     | 0.957 | +0.33   | 0.892       | 0.958 | +1.33   |
| I-DT                  | 0.756         | 0.826 | -0.35   | 0.749 | 0.789 | -0.62   | 0.874     | 0.944 | -0.64   | 0.826       | 0.913 | -2.66   |
| I-HMM                 | **0.793**     | **0.865** | **+2.91** | **0.798** | **0.839** | **+2.52** | 0.907 | 0.965 | +0.92   | 0.885       | 0.952 | +0.89   |
| I-MST                 | 0.783         | 0.848 | +1.77   | 0.776 | 0.825 | +1.34   | **0.934** | **0.979** | **+2.1** | 0.891    | 0.953 | +1.15   |
| SP Tool               | 0.728         | 0.804 | -2.48   | 0.69  | 0.728 | -4.43   | 0.793     | 0.877 | -5.01   | 0.807       | 0.888 | -4.32   |
| 1DCNN BLSTM: GazeCom  | 0.72          | 0.794 | -3.25   | 0.727 | 0.774 | -1.84   | 0.859     | 0.927 | -1.58   | 0.868       | 0.944 | -0.01   |
| 1DCNN BLSTM: HMR      | 0.746         | 0.82  | -1.01   | 0.752 | 0.81  | +0.09   | 0.899     | 0.961 | +0.61   | 0.867       | 0.939 | -0.24   |
| 1DCNN LSTM: HMR       | 0.738         | 0.819 | -1.4    | 0.751 | 0.814 | +0.16   | 0.887     | 0.953 | +0.03   | 0.88        | 0.951 | +0.66   |
| TCN: GazeCom          | 0.762         | 0.84  | +0.56   | 0.762 | 0.811 | +0.46   | 0.886     | 0.954 | +0.02   | **0.892**   | **0.96** | **+1.47** |
| TCN: HMR              | 0.771         | 0.846 | +1.17   | 0.775 | 0.829 | +1.44   | 0.905     | 0.961 | +0.73   | 0.889       | 0.956 | +1.2    |

### Diagnostic Error Prediction

Performance metrics (accuracy and AUC) for the diagnostic error prediction task. Fixation points were extracted using ten different fixation identification algorithms. Features derived from these points were evaluated using four models commonly applied in gaze analysis. Quality columns represent Z-score-based rankings.

|                       | Random Forest |       |         | SVM       |       |         | LSTM      |       |         | Transformer |       |         |
|-----------------------|---------------|-------|---------|-----------|-------|---------|-----------|-------|---------|-------------|-------|---------|
|                       | Acc           | AUC   | Quality | Acc       | AUC   | Quality | Acc       | AUC   | Quality | Acc         | AUC   | Quality |
| I-VT                  | **0.621**     | **0.655** | **+1.801** | 0.538  | 0.551 | +0.211  | **0.59**  | **0.622** | **+1.086** | 0.599    | **0.633** | +0.85   |
| I-DT                  | 0.608         | 0.637 | +0.398  | 0.527     | 0.53  | -0.574  | 0.574     | 0.597 | +0.131  | 0.584       | 0.607 | -0.517  |
| I-HMM                 | 0.612         | 0.645 | +0.841  | 0.525     | 0.531 | -0.523  | 0.59      | 0.614 | +0.734  | **0.605**   | 0.624 | +0.615  |
| I-MST                 | 0.615         | 0.646 | +1.004  | 0.549     | 0.567 | +0.733  | 0.59      | 0.611 | +0.655  | 0.589       | 0.614 | +0.215  |
| SP Tool               | 0.609         | 0.641 | +0.573  | 0.535     | 0.541 | -0.082  | 0.577     | 0.609 | +0.446  | 0.592       | 0.632 | +0.821  |
| 1DCNN BLSTM: GazeCom  | 0.608         | 0.638 | +0.391  | 0.527     | 0.535 | -0.369  | 0.574     | 0.601 | +0.177  | 0.597       | 0.627 | +0.597  |
| 1DCNN BLSTM: HMR      | 0.598         | 0.632 | -0.25   | 0.548     | 0.568 | +0.75   | 0.56      | 0.585 | -0.381  | 0.578       | 0.6   | -0.509  |
| 1DCNN LSTM: HMR       | 0.599         | 0.639 | +0.201  | **0.553** | **0.574** | **+1.038** | 0.556  | 0.586 | -0.323  | 0.589       | 0.616 | +0.359  |
| TCN: GazeCom          | 0.608         | 0.644 | +0.706  | 0.545     | 0.56  | +0.42   | 0.575     | 0.596 | +0.089  | 0.593       | 0.626 | +0.651  |
| TCN: HMR              | 0.59          | 0.624 | -1.207  | 0.538     | 0.551 | +0.203  | 0.545     | 0.557 | -0.964  | 0.564       | 0.590 | -1.034  |

## Contact

If you want to test your identification method on our data, please contact:
- bulat@di.ku.dk
- yxyuan@ee.cuhk.edu.hk

## References

D. D. Salvucci and J. H. Goldberg. Identifying fixations and saccades in eye-tracking protocols. In *Proceedings of the 2000 Symposium on Eye Tracking Research & Applications*, ETRA '00, pages 71–78, New York, NY, USA, 2000. Association for Computing Machinery.

A. Olsen. The Tobii I-VT fixation filter algorithm description. *Tobii Technology*, 21:4–19, 2012.

I. Agtzidis, M. Startsev, and M. Dorr. Smooth pursuit detection based on multiple observers. In *Proceedings of the Ninth Biennial ACM Symposium on Eye Tracking Research & Applications*, pages 303–306. ACM, 2016.

M. Startsev, I. Agtzidis, and M. Dorr. 1D CNN with BLSTM for automated classification of fixations, saccades, and smooth pursuits. *Behavior Research Methods*, Nov 2018.

C. Elmadjian, C. Gonzales, R. L. d. Costa, et al. Online eye-movement classification with temporal convolutional networks. *Behavior Research Methods*, 55:3602–3620, 2023.
