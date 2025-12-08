import numpy as np
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN
from collections import defaultdict
import vg
import matplotlib.pyplot as plt

def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    cosine_angle = np.clip(cosine_angle, -1, 1)
    angle = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

def get_fix_points(gaze,
                   px_per_mm=7.21,
                   fix_tr=30,
                   saccade_tr=30,
                   tracker_fr=90,
                   max_time_between_fixations=0.075,
                    max_angle_between_fixations=0.5,
                    min_fixation_duration=0.06
                    ):
    gaze = gaze.copy()
    gaze['timestamp'] -= gaze.loc[0, 'timestamp']
    gaze = gaze.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

    gaze["velocity"] = 0. # deg/sec
    gaze['is_fix_point'] = 0
    for i in range(1, len(gaze) - 1):
        p1 = gaze.loc[i, ['x', 'y', 'left_z']].values
        p2 = gaze.loc[i + 1, ['x', 'y', 'left_z']].values
        p1[:2] /= px_per_mm
        p2[:2] /= px_per_mm

        # t1 = gaze.loc[i, 'timestamp']
        # t2 = gaze.loc[i + 1, 'timestamp']
        # delta_t = (t2 - t1)

        angle = angle_between_vectors(p1, p2)
        velocity = angle * tracker_fr
        # velocity = angle * (1000000 / delta_t) ?

        gaze.loc[i + 1, "velocity"] = velocity
        if velocity < fix_tr:
            gaze.loc[i + 1, 'is_fix_point'] = 1
        elif velocity > saccade_tr:
            gaze.loc[i + 1, 'is_fix_point'] = 0
        else:
            gaze.loc[i + 1, 'is_fix_point'] = -1
    gaze = gaze[2:]

    fix_data = defaultdict(list)
    saccades = gaze[gaze['is_fix_point'] == 0]
    fix_points = gaze[gaze['is_fix_point'] == 1]
    prev_saccade_index = 0
    for saccade_index, _ in saccades.iterrows():
        fix_group = fix_points.loc[prev_saccade_index:saccade_index]
        if len(fix_group) == 0:
            continue

        start_time = fix_group.head(1)['timestamp'].values[0]
        end_time = fix_group.tail(1)['timestamp'].values[0]
        elapsed_time = end_time - start_time

        fix_data['x'].append(fix_group['x'].mean())
        fix_data['y'].append(fix_group['y'].mean())
        fix_data['z'].append(fix_group['left_z'].mean())
        fix_data['std'].append(fix_group[['x', 'y']].std().mean())
        fix_data['elapsed_time'].append(elapsed_time)
        fix_data['start_time'].append(start_time)
        fix_data['end_time'].append(end_time)

        prev_saccade_index = saccade_index
    fix_data = pd.DataFrame().from_dict(fix_data)

    for i in range(1, len(fix_data) - 1):
        delta_time = fix_data.loc[i + 1, "start_time"] - fix_data.loc[i, "end_time"]
        if delta_time < max_time_between_fixations:
            p1 = fix_data.loc[i, ["x", "y", "z"]].values
            p2 = fix_data.loc[i + 1, ["x", "y", "z"]].values

            t1 = (fix_data.loc[i, 'start_time'] + fix_data.loc[i, 'end_time']) / 2
            t2 = (fix_data.loc[i + 1, 'start_time'] + fix_data.loc[i + 1, 'end_time']) / 2
            delta_t = (t2 - t1)

            angle = vg.angle(p1, p2)
            velocity = angle * (1 / delta_t)
            if velocity <= max_angle_between_fixations:
                # merge fixations
                # print("merge")
                mean_columns = ['x', 'y', 'z', 'std']
                max_elapsed_time = max(fix_data.loc[i, "elapsed_time"], fix_data.loc[i+1, "elapsed_time"])
                c1 = fix_data.loc[i, "elapsed_time"] / max_elapsed_time
                c2 = fix_data.loc[i + 1, "elapsed_time"] / max_elapsed_time

                for column in mean_columns:
                    # fix_data.loc[i, "x"] = np.mean(fix_data.loc[i, column], fix_data.loc[i+1, column])
                    fix_data.loc[i, column] = c1 * fix_data.loc[i, column] + c2 * fix_data.loc[i+1, column]

                fix_data.loc[i, "elapsed_time"] = fix_data.loc[i, "elapsed_time"] + fix_data.loc[i + 1, 'elapsed_time']
                fix_data.loc[i, "end_time"] = fix_data.loc[i + 1, 'end_time']
                fix_data.drop(fix_data.index[i])

    fix_data = fix_data[fix_data['elapsed_time'] > min_fixation_duration].reset_index(drop=True)
    return fix_data
