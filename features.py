import numpy as np
import pandas as pd
import math
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc
from skimage.transform import resize

# Minimum norm threshold to avoid division by zero in angle calculations
MIN_NORM_THRESHOLD = 1e-10
    
class accumulated_gaze_map_statistics:
    """Computes statistics from accumulated gaze heatmaps."""

    def __init__(self, case, image_file):
        self.image = mpimg.imread(image_file)
        self.case = case

    def compute(self):
        """Placeholder for computation logic."""
        raise NotImplementedError("compute() method not yet implemented")

    def __one_iteration(self, heatmap):
        return np.sum(heatmap >= 0.5) / (heatmap.shape[0] * heatmap.shape[1])

class gaze_distance_map:
    """
    Compute Euclidean distance transform from gaze fixation points.

    Creates a map where each pixel contains the distance to the nearest
    fixation point. Uses transposed array convention [x, y] to match
    the point coordinate format.

    Attributes:
        scale: Scaling factor for coordinate transformation.
        size: Scaled dimensions [width, height] of the distance map.
        dist_map: The computed distance transform (None until compute_map called).
    """

    def __init__(self, scale, size_orig):
        """
        Initialize the gaze distance map calculator.

        Args:
            scale: Scaling factor to apply to coordinates.
            size_orig: Original size as [width, height].
        """
        self.scale = scale
        self.size = [int(round(size_orig[0] * scale)), int(round(size_orig[1] * scale))]
        self.dist_map = None
        
    def compute_map(self, points):
        """
        Compute the distance transform from fixation points.

        Creates a seed array with 0s at fixation locations and 1s elsewhere,
        then computes the Euclidean distance transform.

        Note: Uses transposed [x, y] array indexing convention.

        Args:
            points: Array of (x, y) fixation coordinates.
        """
        seed_array = np.ones((self.size[0], self.size[1]))
        for point in points:
            # Skip invalid (off-screen) points
            if point[0] < 0 or point[1] < 0:
                continue
            # Mark fixation location as 0 (using [x, y] indexing)
            x_idx = min(seed_array.shape[0] - 1, int(round(point[0] * self.scale)))
            y_idx = min(seed_array.shape[1] - 1, int(round(point[1] * self.scale)))
            seed_array[x_idx, y_idx] = 0
        # Compute Euclidean distance to nearest fixation
        self.dist_map = ndimage.distance_transform_edt(seed_array)

    def get_value(self, point):
        """
        Get the distance value at a specific point.

        Args:
            point: Coordinate as (x, y).

        Returns:
            Distance to nearest fixation at this point.
            Returns mean distance if point is off-screen.
        """
        # Return mean for invalid points
        if point[0] < 0 or point[1] < 0:
            return np.mean(self.dist_map)
        # Look up distance value (using [x, y] indexing)
        x_idx = min(self.dist_map.shape[0] - 1, int(round(point[0] * self.scale)))
        y_idx = min(self.dist_map.shape[1] - 1, int(round(point[1] * self.scale)))
        return self.dist_map[x_idx, y_idx]

    def compute_map_over_mask(self, mask):
        """
        Compute distance statistics within a masked region.

        Args:
            mask: Binary mask in standard [row, col] format.
                  Will be transposed to match dist_map convention.

        Returns:
            Tuple of (total_distance, mean_distance) within the mask.
        """
        # Transpose mask to match dist_map's [x, y] convention
        mask_sw = np.swapaxes(mask, 0, 1)
        # Resize mask to match distance map dimensions
        mask_scale = ndimage.zoom(
            mask_sw,
            [self.dist_map.shape[0] / mask_sw.shape[0],
             self.dist_map.shape[1] / mask_sw.shape[1]],
            order=0  # Nearest neighbor interpolation for binary mask
        )
        
        # Compute distance values within mask
        value = self.dist_map * mask_scale
        total_value = np.sum(value)
        mask_sum = np.sum(mask_scale)
        #value = np.swapaxes(value, 0, 1)
        #imgplot = plt.imshow(value)
        #plt.show()

        # Avoid division by zero for empty masks
        if mask_sum == 0:
            return total_value, 0.0
        return total_value, total_value / mask_sum

    def compute_visits_of_mask(points, mask):
        """
        Count fixation points that fall within a masked region.

        Args:
            points: Array of (x, y) fixation coordinates.
            mask: Binary mask in standard [row, col] format.
                  Region of interest is marked with 0.

        Returns:
            Number of fixations within the mask region.
        """
        count = 0
        for point in points:
            # Skip invalid points
            if point[0] < 0 or point[1] < 0:
                continue
            # TODO: double check here, original code: -- checked, new version is correct.
            # if mask[min(mask.shape[0] - 1, int(point[0])), min(mask.shape[1] - 1, int(point[1]))] == 0:
            #     count += 1
            # Convert (x, y) to [row, col] = [y, x] for standard mask
            row = min(mask.shape[0] - 1, int(point[1]))
            col = min(mask.shape[1] - 1, int(point[0]))
            # Count if point is in region of interest (marked as 0)
            if mask[row, col] == 0:
                count += 1
        #plt.imshow(mask)
        #plt.plot(points[:, 0], points[:, 1], marker = 'o', color = "red", linewidth = 0.5)
        #plt.show()
        return count

class gaze_lung_coverage_map:
    """
    Track gaze coverage and transitions between lung regions.

    Analyzes how gaze moves between left and right lung areas,
    counting switches between regions.

    Attributes:
        image_L: Binary mask for left lung region.
        image_R: Binary mask for right lung region.
        scale: Scaling factor applied to masks.
    """

    def __init__(self, mask_L, mask_R, scale = 1):
        """
        Initialize lung coverage tracker.

        Args:
            mask_L: Binary mask for left lung (standard [row, col] format).
            mask_R: Binary mask for right lung (standard [row, col] format).
            scale: Optional scaling factor for the masks.
        """
        if scale != 1:
            # Resize masks using skimage (scipy.misc.imresize is deprecated)
            # self.image_L = scipy.misc.imresize(mask_L, [int(round(mask_L.shape[0] * scale)),
            #                                             int(round(mask_L.shape[1] * scale))], mode = 'L')
            # self.image_R = scipy.misc.imresize(mask_R, [int(round(mask_R.shape[0] * scale)),
            #                                             int(round(mask_R.shape[1] * scale))], mode = 'L')
            new_shape = (
                int(round(mask_L.shape[0] * scale)),
                int(round(mask_L.shape[1] * scale))
            )
            self.image_L = resize(
                                    mask_L, new_shape, order=0, preserve_range=True
                                ).astype(mask_L.dtype)
            self.image_R = resize(
                                    mask_R, new_shape, order=0, preserve_range=True
                                ).astype(mask_R.dtype)
        else:
            self.image_L = mask_L
            self.image_R = mask_R
        
        self.scale = scale

    def __inside_image(self, shape, point):
        """
        Check if a point is within image boundaries.

        Args:
            shape: Image shape as (rows, cols).
            point: Scaled point coordinates.

        Returns:
            True if point is within bounds, False otherwise.
        """
        if (point[0] < 0) or (point[1] < 0):
            return False
        if (point[0] >= shape[0]) or (point[1] >= shape[1]):
            return False
        return True

    def compute_switches(self, points, size_orig):
        """
        Count gaze transitions between left and right lung regions.

        A switch is counted when gaze moves from one lung region to
        another (including transitions through non-lung areas).

        Args:
            points: Array of (x, y) gaze coordinates.
            size_orig: Original image size (unused, kept for compatibility).

        Returns:
            Number of switches between lung regions.
        """
        last = -1 # -1 = no previous region, 0 = outside, 1 = left, 2 = right
        switches = 0
        for point in points:
            current = 0 # Default: outside both lungs
            if self.__inside_image(self.image_L.shape, point * self.scale):
                row = min(
                            self.image_L.shape[0] - 1,
                            int(point[1] * self.scale)
                        )
                col = min(
                            self.image_L.shape[1] - 1,
                            int(point[0] * self.scale)
                        )
                if self.image_L[row, col] > 0:
                    current = 1  # Left lung
                if self.image_R[row, col] > 0:
                    current = 2  # Right lung
            # Count transition if region changed (ignore initial state)
            if (last != current) and (last != -1):
                switches += 1
            last = current
        return switches

class gaze_statistics_case:
    """
    Compute comprehensive gaze statistics for a single case/trial.

    Extracts various features from fixation sequences including:
    - Saccade length and angle statistics
    - Spatial coverage metrics
    - Temporal dynamics (acceleration, steps back)
    - Lung-specific coverage and switching behavior

    Attributes:
        fixations: Array of (x, y) fixation coordinates.
        histo_features: Distance thresholds for histogram binning.
        cut_off1: Primary distance threshold for close fixations.
        cut_off2: Secondary threshold (2x cut_off1).
        scale: Scaling factor for distance computations.
    """

    def __init__(self, fixations, cut_off, histo_features = None, scale = 0.1):
        """
        Initialize the gaze statistics calculator.

        Args:
            fixations: Array of (x, y) fixation coordinates.
            cut_off: Distance threshold for identifying close fixations.
            histo_features: Distance bins for histogram features.
                           Default: [50, 100, 200, 300, 5000]
            scale: Scaling factor for distance computations.
        """
        if histo_features is None:
            histo_features = [50, 100, 200, 300, 5000]
        self.fixations = np.asarray(fixations)
        self.histo_features = [value * scale for value in histo_features]
        self.cut_off1 =     cut_off
        self.cut_off2 = 2 * cut_off
        self.scale = scale

    def __compute_average_length(self):
        """
        Compute mean and standard deviation of saccade lengths.

        Returns:
            Tuple of (mean_length, std_length).
            Returns (0.0, 0.0) if fewer than 2 fixations.
        """
        if self.fixations.shape[0] < 2:
            return 0.0, 0.0
        
        dist = []
        for i in range(0, self.fixations.shape[0] - 1):
            u = self.fixations[i]
            v = self.fixations[i + 1]
            dist.append(np.linalg.norm(u - v))
        return np.mean(dist), np.std(dist)

    def __compute_average_angle_locked(self):
        """
        Compute average angle difference between consecutive saccade pairs.

        Measures consistency of saccade directions by comparing
        angles of successive saccade pairs.

        Returns:
            Average absolute angle difference in radians.
            Returns 0.0 if fewer than 4 fixations.
        """
        if self.fixations.shape[0] < 4:
            return 0.0
        
        aver_ang = 0
        valid_count = 0
        for i in range(self.fixations.shape[0] - 3):
            u1 = self.fixations[i]
            u2 = self.fixations[i + 1]
            u3 = self.fixations[i + 2]
            u4 = self.fixations[i + 3]

            # Compute saccade vectors
            v1 = u2 - u1
            v2 = u3 - u2
            v3 = u4 - u3

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            norm3 = np.linalg.norm(v3)

            # Skip degenerate cases (zero-length saccades)
            if (norm1 < MIN_NORM_THRESHOLD or
                    norm2 < MIN_NORM_THRESHOLD or
                    norm3 < MIN_NORM_THRESHOLD):
                continue

            # Compute angles with clipping to handle floating point errors
            cos_ang1 = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
            cos_ang2 = np.clip(np.dot(v2, v3) / (norm2 * norm3), -1.0, 1.0)

            ang1 = math.acos(cos_ang1)
            ang2 = math.acos(cos_ang2)

            aver_ang += np.abs(ang1 - ang2)
            valid_count += 1

        return aver_ang / max(1, valid_count)

    def __compute_average_angle(self):
        """
        Compute mean and std of angles between saccade vectors.

        Calculates the cosine of angles between vectors connecting
        non-adjacent fixation pairs.

        Returns:
            Tuple of (mean_cosine, std_cosine).
            Returns (0.0, 0.0) if fewer than 4 fixations.
        """
        if self.fixations.shape[0] < 4:
            return 0.0, 0.0

        angs = []
        # Start from i=1 to avoid negative indexing (i-1 when i=0)
        for i in range(1, self.fixations.shape[0] - 2):
            u1 = self.fixations[i + 1] - self.fixations[i]
            u2 = self.fixations[i + 2] - self.fixations[i - 1]

            norm1 = np.linalg.norm(u1)
            norm2 = np.linalg.norm(u2)

            # Skip degenerate cases
            if norm1 < MIN_NORM_THRESHOLD or norm2 < MIN_NORM_THRESHOLD:
                continue
            
            # Compute cosine with clipping for numerical stability
            ang1 = np.clip(np.dot(u1, u2) / (norm1 * norm2), -1.0, 1.0)
            angs.append(ang1)

        if len(angs) == 0:
            return 0.0, 0.0
        
        return np.mean(angs), np.std(angs)

    def __compute_average_dist2(self):
        """
        Compute average ratio of distances for fixation quadruplets.

        Measures how distance changes across successive fixations.

        Returns:
            Average ratio of dist(f[i], f[i+3]) / dist(f[i], f[i+2]).
            Returns 0.0 if fewer than 4 fixations.
        """
        if self.fixations.shape[0] < 4:
            return 0.0

        aver_add = 0
        valid_count = 0

        for i in range(self.fixations.shape[0] - 3):
            u1 = self.fixations[i]
            u3 = self.fixations[i + 2]
            u4 = self.fixations[i + 3]

            dist_13 = np.linalg.norm(u1 - u3)

            # Skip degenerate cases
            if dist_13 < MIN_NORM_THRESHOLD:
                continue

            aver_add += np.linalg.norm(u1 - u4) / dist_13
            valid_count += 1

        return aver_add / max(1, valid_count)
    
    def __compute_accumulation_gain(self, image):
        """
        Compute how much new area is explored by each successive fixation.

        For each fixation, measures the distance to the nearest
        previously visited location.

        Args:
            image: The stimulus image (used for dimensions).

        Returns:
            List of distance values, one per fixation (excluding first).
        """
        if len(self.fixations) < 2:
            return []

        histo = []
        gaze_mapper = gaze_distance_map(
            self.scale,
            [image.shape[1], image.shape[0]]  # [width, height]
        )

        for i in range(1, len(self.fixations)):
            # Build distance map from all previous fixations
            gaze_mapper.compute_map(self.fixations[0:i, :])
            # Get distance of current fixation to nearest previous one
            dist_value = gaze_mapper.get_value(self.fixations[i, :]) / self.scale
            histo.append(dist_value)
        return histo

    def __compute_close_fixations(self):
        """
        Count fixations within cutoff distances of previous fixation.

        Returns:
            Tuple of (count_within_cutoff1, count_within_cutoff2).
        """
        count_cut_off1 = 0
        count_cut_off2 = 0
        # Start from i=1 to compare with previous fixation
        for i in range(1, len(self.fixations)):
            dist = np.linalg.norm(self.fixations[i] - self.fixations[i - 1])
            if dist < self.cut_off1:
                count_cut_off1 += 1
            if dist < self.cut_off2:
                count_cut_off2 += 1
        return count_cut_off1, count_cut_off2

    def __compute_steps_back(self):
        """
        Compute proportion of fixations that move closer to earlier point.

        A "step back" occurs when the current fixation is closer to
        fixation[i-2] than to fixation[i-1].

        Returns:
            Proportion of fixations that are steps back.
            Returns 0.0 if fewer than 3 fixations.
        """
        if len(self.fixations) < 3:
            return 0.0
        
        count_back = 0

        for i in range(2, len(self.fixations)):
            dist1 = np.linalg.norm(self.fixations[i] - self.fixations[i - 1])
            dist2 = np.linalg.norm(self.fixations[i] - self.fixations[i - 2])

            if dist1 > dist2:
                count_back += 1

        return count_back / (len(self.fixations) - 2)


    def __compute_organ_coverage(self, image, mask_L, mask_R):
        """
        Compute gaze coverage statistics over lung masks.

        Args:
            image: The stimulus image (used for dimensions).
            mask_L: Binary mask for left lung.
            mask_R: Binary mask for right lung.

        Returns:
            Tuple of (abs_dist_L, mean_dist_L, abs_dist_R, mean_dist_R,
                     visits_L, visits_R).
        """
        gaze_mapper = gaze_distance_map(0.1, [image.shape[1], image.shape[0]])
        gaze_mapper.compute_map(self.fixations)
        
        # Compute coverage metrics for each lung
        count_mask_abs_L, count_mask_L = gaze_mapper.compute_map_over_mask(mask_L)
        count_mask_abs_R, count_mask_R = gaze_mapper.compute_map_over_mask(mask_R)

        # Count direct visits to each lung region
        visits_mask_L = gaze_distance_map.compute_visits_of_mask(
                                                                    self.fixations, mask_L
                                                                )
        visits_mask_R = gaze_distance_map.compute_visits_of_mask(
                                                                    self.fixations, mask_R
                                                                )
        # inversed_map = np.swapaxes(gaze_mapper.dist_map, 0, 1)
        # inversed_map = scipy.ndimage.zoom(inversed_map, [mask_L.shape[0] / inversed_map.shape[0], 
        #                                                  mask_L.shape[1] / inversed_map.shape[1]], order = 0)

        #plt.imshow(inversed_map)
        #plt.plot(self.fixations[:, 0], self.fixations[:, 1], marker = 'o', color = "red", linewidth = 0.5)
        #plt.show()
        return count_mask_abs_L, count_mask_L, \
                count_mask_abs_R, count_mask_R, \
                visits_mask_L, visits_mask_R

    def __compute_histo_features(self, histo):
        """
        Bin accumulation gain values into distance categories.

        Args:
            histo: List of distance values from __compute_accumulation_gain.

        Returns:
            Array of counts for each histogram bin.
        """
        count = np.zeros(len(self.histo_features))
        for value in histo:
            for j, threshold in enumerate(self.histo_features):
                if value <= threshold / self.scale:
                    count[j] += 1
                    break

        return count

    def __get_total_length(self, data_frame):
        """
        Calculate total viewing time in seconds.

        Args:
            data_frame: DataFrame or array with one row per sample.

        Returns:
            Total time in seconds based on sampling rate.
        """
        #data_frame = pd.read_csv(file_gaze_data)
        #start_time = datetime.fromtimestamp(data_frame['timestamp'].values[0])
        #end_time   = datetime.fromtimestamp(data_frame['timestamp'].values[-1])
        return len(data_frame) * 0.01 #len(data_frame['timestamp'].values) * 0.01

    def __visualize_fixations(self, image):
        """
        Display fixation scanpath overlaid on the stimulus image.

        Args:
            image: The stimulus image to display.
        """
        plt.imshow(image)
        #plt.imshow(self.heatmap.get_heatmap(), alpha = 0.2)
        #plt.scatter(self.fixations['x'].values, self.fixations['y'].values, 
        #            marker = "x", color = "red", s = 100)
        plt.imshow(image)
        plt.plot(
            self.fixations[:, 0],
            self.fixations[:, 1],
            marker='o',
            color="red",
            linewidth=0.5
        )
        plt.title("Fixation Scanpath")
        plt.show()

    def __compute_acceleration(self):
        """
        Compute acceleration (change in saccade velocity) over time.

        Returns:
            List of acceleration values (absolute speed differences).
            Returns empty list if fewer than 3 fixations.
        """
        if len(self.fixations) < 3:
            return []
        
        # Compute speeds (distances between consecutive fixations)
        speeds = []
        for i in range(1, len(self.fixations)):
            speeds.append(
                np.linalg.norm(self.fixations[i] - self.fixations[i - 1])
            )

        # Compute acceleration (change in speed)
        acceleration = []
        for i in range(1, len(speeds)):
            acceleration.append(np.abs(speeds[i] - speeds[i - 1]))

        return acceleration

    def get_all_statistics(self, image, mask_L, mask_R, file_gaze_data):
        """
        Compute all gaze statistics for this case.

        Args:
            image: The stimulus image (numpy array).
            mask_L: Binary mask for left lung region.
            mask_R: Binary mask for right lung region.
            file_gaze_data: DataFrame with raw gaze data (for timing).

        Returns:
            List of computed statistics:
                [0-1]   fix_length_mean, fix_length_std
                [2-3]   angle_mean, angle_std
                [4]     num_fixations
                [5]     lung_switches
                [6]     total_time_sec
                [7-8]   accumulation_gain_mean, accumulation_gain_std
                [9-10]  acceleration_mean, acceleration_std
                [11-12] close_fixations_cutoff1, close_fixations_cutoff2
                [13-14] visits_left_lung, visits_right_lung
                [15-16] coverage_abs_left, coverage_norm_left
                [17-18] coverage_abs_right, coverage_norm_right
                [19+]   histogram_counts, histogram_percentages
        """
        #self.__visualize_fixations(image)

        # Fix statistics
        dist, dist_std = self.__compute_average_length()
        ang, ang_std = self.__compute_average_angle()

        # Accumulation gain (exploration efficiency)
        histo = self.__compute_accumulation_gain(image)
        histo_count = self.__compute_histo_features(histo)
        histo_count_per = histo_count / max(1, len(histo))

        # Close fixations (potential re-examinations)
        count_cut_off1, count_cut_off2 = self.__compute_close_fixations()

        # Organ coverage
        count_mask_abs_L, count_mask_L, \
         count_mask_abs_R, count_mask_R, \
         visits_mask_L, visits_mask_R = self.__compute_organ_coverage(
                                                                        image, mask_L, mask_R
                                                                    )
        # Temporal dynamics
        # Note: steps_back computed for potential future use
        _steps_back = self.__compute_steps_back()  # noqa: F841
        acceleration = self.__compute_acceleration()

        # Handle empty lists gracefully
        accel_mean = np.mean(acceleration) if acceleration else 0.0
        accel_std = np.std(acceleration) if acceleration else 0.0
        histo_mean = np.mean(histo) if histo else 0.0
        histo_std = np.std(histo) if histo else 0.0

        # Lung switching behavior
        gaze_switches = gaze_lung_coverage_map(mask_L, mask_R)
        switches = gaze_switches.compute_switches(self.fixations, mask_L.shape)

        # Old code
        #self.__visualize_fixations(image)
        # dist, dist_std = self.__compute_average_length()
        # ang, ang_std = self.__compute_average_angle()
        # histo = self.__compute_accumulation_gain(image)

        # histo_count = self.__compute_histo_features(histo)
        # histo_count_per = histo_count / max(1, len(histo))
        # count_cut_off1, count_cut_off2 = self.__compute_close_fixations()
        # count_mask_abs_L, count_mask_L, count_mask_abs_R, count_mask_R, visits_mask_L, visits_mask_R = self.__compute_organ_coverage(image, mask_L, mask_R)
        # steps_back = self.__compute_steps_back()

        # acceleration = self.__compute_acceleration()
        #plt.plot(acceleration, marker = 'o', color = "red", linewidth = 0.5)
        #plt.show()

        # Compile all features
        return [
                    dist, dist_std,                    # [0-1] Fix length
                    ang, ang_std,                      # [2-3] Angle statistics
                    self.fixations.shape[0],           # [4] Number of fixations
                    switches,                          # [5] Lung switches
                    self.__get_total_length(file_gaze_data),  # [6] Total time
                    histo_mean, histo_std,             # [7-8] Accumulation gain
                    accel_mean, accel_std,             # [9-10] Acceleration
                    count_cut_off1, count_cut_off2,    # [11-12] Close fixations
                    visits_mask_L, visits_mask_R,      # [13-14] Lung visits
                    count_mask_abs_L, count_mask_L,    # [15-16] Left coverage
                    count_mask_abs_R, count_mask_R     # [17-18] Right coverage
                ] + histo_count.tolist() + histo_count_per.tolist()

# =============================================================================
# Feature Names (for documentation and output labeling)
# =============================================================================

FEATURE_COLUMNS = [
    'fixation_dist', 'fixation_dist_std',
    'fixation_angle', 'fixation_angle_std',
    '#_fixations', 'switches_between_objects', 'total_length',
    'info_gain_per_fixation_mean', 'info_gain_per_fixation_std',
    'acceleration_mean', 'acceleration_std',
    '#_fixation_below_75pix', '#_fixation_below_150pix',
    'visits_lung_L', 'visits_lung_R',
    'gaze_coverage_abs_L', 'gaze_coverage_L',
    'gaze_coverage_abs_R', 'gaze_coverage_R',
    '#_fixations_infogain_below_50pix',
    '#_fixations_infogain_below_100pix',
    '#_fixations_infogain_below_200pix',
    '#_fixations_infogain_below_300pix',
    '#_fixations_infogain_below_5000pix',
    '%_fixations_infogain_below_50pix',
    '%_fixations_infogain_below_100pix',
    '%_fixations_infogain_below_200pix',
    '%_fixations_infogain_below_300pix',
    '%_fixations_infogain_below_5000pix'
]