import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from hmmlearn import hmm

import numpy as np
from hmmlearn import hmm

def calculate_point_to_point_velocities(data):
    """Calculate point-to-point velocities between consecutive spatial points."""
    # spatial_points = points[:, :2]  # Extract only the x, y coordinates
    # velocities = np.diff(spatial_points, axis=0)
    # return np.linalg.norm(velocities, axis=1)
    points = data[['x', 'y']].values
    speeds = []
    for i in range(1, len(points)):
        distance = np.linalg.norm(points[i] - points[i - 1])
        time_diff = data['timestamp'][i] - data['timestamp'][i-1]
        velocity = distance / time_diff if time_diff > 0 else 0
        speeds.append(velocity)
    return np.array(speeds)

def decode_velocities_with_hmm(velocities):
    """Use a two-state HMM to classify points as fixation or saccade."""
    model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000)
    model.fit(velocities.reshape(-1, 1))
    states = model.predict(velocities.reshape(-1, 1))
    return states

def identify_fixations(points, states):
    """Identify fixation points and map each group to a fixation at the centroid, including average timestamp."""
    fixations = []
    current_fixation = []
    for i, point in enumerate(points[:-1]):  # Exclude the last point as there's no velocity for it
        if states[i] == 0:  # Assuming state '0' is fixation
            current_fixation.append(point)
        if states[i] == 1 or i == len(states) - 1:  # Change to saccade or end of list
            if current_fixation:
                fixation_centroid = np.mean(current_fixation, axis=0)
                fixations.append(fixation_centroid)
                current_fixation = []
    return np.array(fixations)

def i_hmm(data):
    """Implement the I-HMM algorithm preserving the timestamp."""
    points = data[['x', 'y', 'timestamp']].values
    velocities = calculate_point_to_point_velocities(data[['x', 'y', 'timestamp']])
    states = decode_velocities_with_hmm(velocities)
    fixations = identify_fixations(points, states)
    return fixations


