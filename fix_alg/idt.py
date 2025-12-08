import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

def calculate_dispersion(points):
    if len(points) == 0:
        return 0
    x_coords, y_coords = points[:, 0], points[:, 1]
    return np.max(x_coords) - np.min(x_coords) + np.max(y_coords) - np.min(y_coords)

def i_dt(dataset, dispersion_threshold, duration_threshold):
    points = dataset
    fixations = []
    i = 0

    while i < len(points) - duration_threshold + 1:
        window = points[i:i + duration_threshold]
        if calculate_dispersion(window) <= dispersion_threshold:
            j = 1
            while i + duration_threshold + j < len(points) and calculate_dispersion(points[i:i + duration_threshold + j]) <= dispersion_threshold:
                j += 1
            window = points[i:i + duration_threshold + j - 1]
            centroid = np.mean(window, axis=0)
            fixation_start_index = i # Commit start index
            fixation_timestamp = points[fixation_start_index, 2]  # It is assumed that the timestamp is in the third column
            fixations.append((centroid[0], centroid[1], fixation_timestamp))
            i += len(window)  # Skip points included in the fixation
        else:
            i += 1  # Move the window

    return fixations
