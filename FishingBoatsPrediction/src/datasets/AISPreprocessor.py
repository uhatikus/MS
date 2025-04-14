from dataclasses import dataclass
import pandas as pd
import glob
from typing import List, Dict
import plotly.express as px
import pickle
import os

import numpy as np
from pyproj import CRS, Transformer
from utils import set_seed

set_seed(42)

# Define the Coordinate Reference Systems
crs_geodetic = CRS.from_epsg(4326)  # WGS84
crs_ecef = CRS.from_epsg(4978)      # ECEF
# Create transformers
transformer_to_ecef = Transformer.from_crs(crs_geodetic, crs_ecef, always_xy=True)
transformer_to_geodetic = Transformer.from_crs(crs_ecef, crs_geodetic, always_xy=True)

SPEED_MAX = 30  # knot
MAX_DEGREES = 360
@dataclass
class AISColumnNames:
    Date: str = "Date"
    Sampled_Date: str = "Sampled_Date"
    Latitude: str = "Latitude"
    Longitude: str = "Longitude"
    Pseudo_Longitude: str = "Pseudo_Longitude"
    SOG: str = "SOG"
    COG: str = "COG"
    Heading: str = "Heading"
    
    n_Latitude: str = "norm Latitude"
    n_Longitude: str = "norm Longitude"
    n_SOG: str = "norm SOG"
    n_COG: str = "norm COG"
    n_Heading: str = "norm Heading"
    
    is_synthetic: str = "is_synthetic"
    to_predict: str = "to_predict"
    
class AISPreprocessor:
    def __init__(self,
                 dataset_path: str,
                 target_freq_in_minutes: int = 10,
                 trajectory_sequence_len: int = 16, #96
                 max_trajectory_sequence_len_to_predict: int = 8,
                 min_trajectory_sequence_len_to_predict: int = 4,
                 rotate_trajetories: bool = True,
                 shitf_trajectories: bool = False,
                 cols: AISColumnNames = AISColumnNames(),
                 latlon_scale: float = None, 
                 synthetic_ratio: float = 0.7,
                 prediction_buffer: int = 1, 
                 validation_size_ratio: float = 0.15, # from 0 to 1
                 test_size_ratio: float = 0.15, # from 0 to 1
                 output_dir: str = "results",
                 dataset_dir: str = "dataset"
                 ):
        
        self.dataset_path: str = dataset_path        
        self.dynamic_data_files = glob.glob(self.dataset_path)
        
        self.target_freq: str = f"{target_freq_in_minutes}min"
        self.sample_T: pd.Timedelta = pd.Timedelta(minutes=target_freq_in_minutes)
        
        self.trajectory_sequence_len: int = trajectory_sequence_len
        self.rotate_trajetories: bool = rotate_trajetories
        self.shitf_trajectories: bool = shitf_trajectories
        self.synthesize_flag: bool = self.rotate_trajetories or self.shitf_trajectories
        self.cols: AISColumnNames = cols
        self.latlon_scale: float = latlon_scale
        self.synthetic_ratio: float = synthetic_ratio
        self.n_synthetic: float = self.synthetic_ratio/(1-self.synthetic_ratio) # per one trajectory, can be rational => probability is used in that case  
        self.n_synthetic_per_trajectory = int(self.n_synthetic)
        self.n_synthetic_probabilistic = self.n_synthetic % 1
        
        self.prediction_buffer = prediction_buffer
        self.max_trajectory_sequence_len_to_predict = max_trajectory_sequence_len_to_predict
        self.min_trajectory_sequence_len_to_predict = min_trajectory_sequence_len_to_predict
        
        self.validation_size_ratio = validation_size_ratio
        self.test_size_ratio = test_size_ratio
        
        self.min_lat, self.max_lat, self.min_lon, self.max_lon, self.max_pseudo_lon = None, None, None, None, None
        self.local_lat_scale, self.local_lon_scale , self.local_pseudo_lon_scale = None, None, None
        
        self.feature_cols = [self.cols.n_Latitude, self.cols.n_Longitude, self.cols.n_SOG, self.cols.n_COG]
        
        self.output_dir = output_dir
        self.dataset_dir = dataset_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
    
    def run(self):
        all_boats_trajectories: Dict[str, pd.DataFrame] = self.read_all_boats_trajectories() # Dict {mmsi: pd.DataFrame with trajectory}
        good_trajectories: List[pd.DataFrame] = self.get_preprocessed_all_boats_trajectories(all_boats_trajectories.values())
        
        self.get_latlon_box(good_trajectories)
        
        for i, t in enumerate(good_trajectories):
            self.display_trajectory(t, i, base_path=f"{self.output_dir}/sample_ship_trajectory_")
            break
        
        if self.synthesize_flag:
            synthesized_trajectories: List[pd.DataFrame] = self.get_synthesized_trajectories(good_trajectories)
        
        good_trajectories.extend(synthesized_trajectories)
        
        self.get_latlon_box(good_trajectories)
        normalized_trajectories: List[pd.DataFrame] = self.get_normalized_trajectories(good_trajectories)
        
        normalized_trajectories: List[pd.DataFrame] = self.create_prediction_masks(normalized_trajectories)

        for i, t in enumerate(good_trajectories):
            self.display_trajectory(t, i, base_path=f"{self.output_dir}/sample_ship_trajectory_preprocessed_")
            break
        
        # for i, t in enumerate(normalized_trajectories):
        #     t.to_csv(f"data/preprocessed_data/{i}.csv", index=False)
        #     # break
            
        self.save_model_dataset(normalized_trajectories)
            
    def save_model_dataset(self, trajectories):
        
        def prepare_trajectory(t: pd.DataFrame):
            X = t[self.feature_cols + [self.cols.to_predict]].values.astype(float)
            return t[self.cols.Sampled_Date].min(), X
        
        data = [prepare_trajectory(t) for t in trajectories]
        
        data.sort(key=lambda x: x[0])
        
        total_size = len(data)
        validation_size = int(total_size * self.validation_size_ratio)
        test_size = int(total_size * self.validation_size_ratio)
        train_size = total_size - validation_size - test_size
        
        train_data = data[:train_size]
        validation_data = data[train_size:train_size + validation_size]
        test_data = data[train_size + validation_size:]
                    
        def create_3d_arrays(data_list):
            # Assume all DataFrames have same number of timesteps
            n_rows = data_list[0][1].shape[0]
            n_cols = data_list[0][1].shape[1]
            n_samples = len(data_list)
            
            X_3d = np.zeros((n_samples, n_rows, n_cols))
            
            for i, (_, X) in enumerate(data_list):
                X_3d[i] = X
                
            return X_3d
        
        X_train = create_3d_arrays(train_data)
        X_validation = create_3d_arrays(validation_data)
        X_test = create_3d_arrays(test_data)
        
        def save_pickle(data, filename):
            with open(filename, 'wb') as f:
                pickle.dump(data, f)

        save_pickle(X_train, f'{self.dataset_dir}/X_train.pkl')
        save_pickle(X_validation, f'{self.dataset_dir}/X_validation.pkl')
        save_pickle(X_test, f'{self.dataset_dir}/X_test.pkl')
        
        # print("Train dates:", [x[0] for x in train_data])
        # print("Validation dates:", [x[0] for x in validation_data])
        # print("Test dates:", [x[0] for x in test_data])
        print("\nShapes:")
        print("X_train shape:", X_train.shape)
        print("X_validation shape:", X_validation.shape)
        print("X_test shape:", X_test.shape)
        
        with open(f'{self.dataset_dir}/metadata.txt', 'w') as f:
            f.write(f"X_train shape: {X_train.shape}\n")
            f.write(f"X_validation shape: {X_validation.shape}\n")
            f.write(f"X_test shape: {X_test.shape}\n")
            
            f.write(f"Train dates: {[x[0] for x in train_data]}\n")
            f.write(f"Validation dates: {[x[0] for x in validation_data]}\n")
            f.write(f"Test dates: {[x[0] for x in test_data]}\n")
    
    def read_all_boats_trajectories(self):
        all_boats_trajectories = {}
        
        for dynamic_data_file in self.dynamic_data_files:
            print(f"Reading {dynamic_data_file}...")
            df_dynamic = pd.read_csv(dynamic_data_file)
            data_grouped = df_dynamic.groupby("MMSI")
            for mmsi, data in data_grouped:
                if mmsi not in all_boats_trajectories:
                    all_boats_trajectories[mmsi] = data.copy()  # Create a copy to avoid SettingWithCopyWarning
                else:
                    all_boats_trajectories[mmsi] = pd.concat([all_boats_trajectories[mmsi], data], ignore_index=True)
            print("Done!")
            # break
            
        return all_boats_trajectories
            
    def get_max_pseudo_lon(self, trajectories: List[pd.DataFrame]):
        if len(trajectories) == 0:
            raise Exception("There are no trajectories in the input")
        
        self.max_pseudo_lon = -181
        
        for trajectory in trajectories:
            cur_max_pseudo_lon = trajectory[self.cols.Pseudo_Longitude].max()
            if cur_max_pseudo_lon > self.max_lon:
                self.max_pseudo_lon = cur_max_pseudo_lon
        
    def validate_latlon_scale(self):
        if self.max_pseudo_lon is None:
            raise Exception("self.max_pseudo_lon is None. Should be calculated for validate_latlon_scale")
        calculated_latlon_scale = max(self.max_lat - self.min_lat, self.max_pseudo_lon - self.min_lon )
        if self.latlon_scale is None:
            self.latlon_scale = calculated_latlon_scale
        if self.latlon_scale < calculated_latlon_scale:
            raise Exception(f"Calculated latlon_scale {calculated_latlon_scale} is bigger than given latlon_scale {self.latlon_scale}!!! That's not possible")
        print(f"self.latlon_scale: {self.latlon_scale}")
        
    def get_latlon_box(self, trajectories: List[pd.DataFrame]):
        if len(trajectories) == 0:
            raise Exception("There are no trajectories in the input")
        self.min_lat = 91
        self.max_lat = -91
        self.min_lon = 181
        self.max_lon = -181
        for trajectory in trajectories:
            cur_lat_min = trajectory[self.cols.Latitude].min()
            cur_lon_min = trajectory[self.cols.Longitude].min()
            cur_lat_max = trajectory[self.cols.Latitude].max()
            cur_lon_max = trajectory[self.cols.Longitude].max()
            if cur_lat_max > self.max_lat:
                self.max_lat = cur_lat_max
            if cur_lat_min < self.min_lat:
                self.min_lat = cur_lat_min
            if cur_lon_max > self.max_lon:
                self.max_lon = cur_lon_max
            if cur_lon_min < self.min_lon:
                self.min_lon = cur_lon_min
        info = f"""
self.min_lat: {self.min_lat}
self.max_lat: {self.max_lat}
self.min_lon: {self.min_lon}
self.max_lon: {self.max_lon}"""
        print(info)
        
    def get_local_latlon_scales(self, trajectories: List[pd.DataFrame]):
        if len(trajectories) == 0:
            raise Exception("There are no trajectories in the input")
        
        self.local_lat_scale, self.local_lon_scale, self.local_pseudo_lon_scale = 0, 0, 0
        for trajectory in trajectories:
            cur_lat_min = trajectory[self.cols.Latitude].min()
            cur_lon_min = trajectory[self.cols.Longitude].min()
            cur_pseudo_lon_min = trajectory[self.cols.Pseudo_Longitude].min()
            cur_lat_max = trajectory[self.cols.Latitude].max()
            cur_lon_max = trajectory[self.cols.Longitude].max()
            cur_pseudo_lon_max = trajectory[self.cols.Pseudo_Longitude].max()
            
            cur_local_lat_scale = cur_lat_max - cur_lat_min
            cur_local_lon_scale = cur_lon_max - cur_lon_min
            cur_local_pseudo_lon_scale = cur_pseudo_lon_max - cur_pseudo_lon_min
            
            if cur_local_lat_scale > self.local_lat_scale:
                self.local_lat_scale = cur_local_lat_scale
            
            if cur_local_lon_scale > self.local_lon_scale:
                self.local_lon_scale = cur_local_lon_scale
                
            if cur_local_pseudo_lon_scale > self.local_pseudo_lon_scale:
                self.local_pseudo_lon_scale = cur_local_pseudo_lon_scale
                
        info = f"""
self.local_lat_scale: {self.local_lat_scale}
self.local_lon_scale: {self.local_lon_scale}
self.local_pseudo_lon_scale: {self.local_pseudo_lon_scale}
"""
        print(info)
        
    def get_preprocessed_all_boats_trajectories(self, trajectories: List[pd.DataFrame]): 
        if len(trajectories) == 0:
            raise Exception("There are no trajectories in the input")
        pre_good_trajectories: List[pd.DataFrame] = []
        good_trajectories: List[pd.DataFrame] = []
        for trajectory in trajectories:
            sampled_trajectory = self.get_sampled_trajectory(trajectory)
            trajectory_sequences = self.get_trajectory_sequences(sampled_trajectory)
            # filter short trajectories
            # TODO: uncomment current_good_trajectories = [seq for seq in trajectory_sequences if len(seq) >= self.trajectory_sequence_len]
            current_good_trajectories = [seq for seq in trajectory_sequences if len(seq) >= 2*self.trajectory_sequence_len]
            pre_good_trajectories.extend(current_good_trajectories)
            if len(pre_good_trajectories) > 0:
                break
        
        for t in pre_good_trajectories:
            t[self.cols.is_synthetic] = False
            
            n_complete_chunks = len(t) // self.trajectory_sequence_len
            current_good_trajectories = [t[i*self.trajectory_sequence_len:(i+1)*self.trajectory_sequence_len] for i in range(n_complete_chunks)]
            good_trajectories.extend(current_good_trajectories)
            
        return good_trajectories

    def normalize_trajectory(self, trajectory: pd.DataFrame):
        
        cur_lat_min = trajectory[self.cols.Latitude].min()
        cur_pseudo_lon_min = trajectory[self.cols.Pseudo_Longitude].min()
        
        # from real lat/lon to values with range [0,1]    
        trajectory[self.cols.n_Latitude] = (trajectory[self.cols.Latitude] - cur_lat_min) / self.local_lat_scale
        trajectory[self.cols.n_Longitude] = (trajectory[self.cols.Pseudo_Longitude] - cur_pseudo_lon_min) / self.local_pseudo_lon_scale
        trajectory[self.cols.n_SOG] = ((trajectory[self.cols.SOG] / SPEED_MAX < 1) * trajectory[self.cols.SOG] / SPEED_MAX) + (trajectory[self.cols.SOG] / SPEED_MAX > 1) 
        trajectory[self.cols.n_COG] = trajectory[self.cols.COG] / MAX_DEGREES
        trajectory[self.cols.n_Heading] = ((trajectory[self.cols.Heading] / MAX_DEGREES < 1) * trajectory[self.cols.Heading] / MAX_DEGREES) + 511 * (trajectory[self.cols.Heading] / MAX_DEGREES > 1) 
        
        # center self.cols.n_Latitude and self.cols.n_Longitude around 0.5
        trajectory[self.cols.n_Latitude] = trajectory[self.cols.n_Latitude] + (0.5 - trajectory[self.cols.n_Latitude].max() / 2)
        trajectory[self.cols.n_Longitude] = trajectory[self.cols.n_Longitude] + (0.5 - trajectory[self.cols.n_Longitude].max() / 2)
        return trajectory

    def get_normalized_trajectories(self, trajectories: List[pd.DataFrame]) -> List[pd.DataFrame]:
        # add Pseudo Longitude 
        for trajectory in trajectories:
            # calculate Pseudo Longitude, which represents the real distance traveled by the boat. 
            # After that operation for both Latutude and Longitude 1 degree ~ 111 km
            trajectory[self.cols.Pseudo_Longitude] = self.min_lon + (trajectory[self.cols.Longitude] - self.min_lon) * np.cos(np.radians(trajectory[self.cols.Latitude]))
        self.get_max_pseudo_lon(trajectories)
        self.validate_latlon_scale()
        
        self.get_local_latlon_scales(trajectories)
        
        normalized_trajectories = [self.normalize_trajectory(trajectory) for trajectory in trajectories]
        return normalized_trajectories
    
    def create_prediction_masks(self, trajectories: List[pd.DataFrame]) -> List[pd.DataFrame]:
        
        for trajectory in trajectories:
            n_predict = np.random.randint(self.min_trajectory_sequence_len_to_predict, self.max_trajectory_sequence_len_to_predict + 1)
            start_range = range(self.prediction_buffer, self.trajectory_sequence_len - n_predict - self.prediction_buffer + 1)
            start_idx = np.random.choice(start_range)
            
            mask = np.zeros(self.trajectory_sequence_len, dtype=bool)  # All False by default
            mask[start_idx:start_idx + n_predict] = True
            trajectory[self.cols.to_predict] = mask
        
        return trajectories

    def synthesize_trajectory(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        synthetic_trajectory = trajectory.copy()
        
        if self.rotate_trajetories:
            angle_deg = np.random.uniform(0, MAX_DEGREES)
            synthetic_trajectory =  self.rotate_trajectory(synthetic_trajectory, angle_deg)
        
        if self.shitf_trajectories:
            lat_shift, lon_shift = self.get_random_bounded_latlon_shifts(synthetic_trajectory)
            synthetic_trajectory = self.shift_trajectory(synthetic_trajectory, lat_shift, lon_shift)
        
        synthetic_trajectory[self.cols.is_synthetic] = True
        
        return synthetic_trajectory
                
    
    def get_synthesized_trajectories(self, trajectories: List[pd.DataFrame]) -> List[pd.DataFrame]:
        synthetic_trajectories: List[pd.DataFrame] = []
        
        # rotate or/and shift trajectories to make more rtajectories
        for t in trajectories:
            for i in range(self.n_synthetic_per_trajectory):
                synthetic_trajectories.append(self.synthesize_trajectory(t))
            if np.random.random() < self.n_synthetic_probabilistic:
                synthetic_trajectories.append(self.synthesize_trajectory(t))
            
        return synthetic_trajectories
    
    def get_sampled_trajectory(self, trajectory: pd.DataFrame) ->  pd.DataFrame:
        trajectory[self.cols.Date] = pd.to_datetime(trajectory[self.cols.Date])
        trajectory = trajectory.set_index(self.cols.Date)
        trajectory = trajectory.sort_index()

        # add first and last steps of trajectory which are divisible by 10 minutes
        first = trajectory.iloc[:1].copy()
        first.index = [trajectory.index.min().floor(self.target_freq)]
        last = trajectory.iloc[-1:].copy()
        last.index = [trajectory.index.max().ceil(self.target_freq)]
        trajectory = pd.concat([first, trajectory, last])

        # Define exact 10-minute sampling times
        start_time = trajectory.index.min().floor("h")  # Round down to the nearest hour
        end_time = trajectory.index.max().ceil("h")  # Round up to the nearest hour
        sampling_times = pd.date_range(start_time, end_time, freq=self.target_freq)

        # Filter only timestamps where at least one real record exists within ±10 minutes
        valid_sampling_times = [t for t in sampling_times if any(abs(trajectory.index - t) <= pd.Timedelta(minutes=10))]

        trajectory = trajectory[~trajectory.index.duplicated(keep='first')]
        trajectory_interpolated = trajectory.reindex(trajectory.index.union(valid_sampling_times)).sort_index()

        # Perform linear interpolation
        trajectory_interpolated = trajectory_interpolated.interpolate(method="time")

        # Keep only the sampled timestamps and drop any remaining NaNs
        trajectory_sampled = trajectory_interpolated.loc[valid_sampling_times].dropna().reset_index()
        trajectory_sampled.rename(columns={"index": self.cols.Sampled_Date}, inplace=True)
        return trajectory_sampled
    
    def get_trajectory_sequences(self, trajectory_sampled: pd.DataFrame, time_column_name=None) -> List[pd.DataFrame]:
        if time_column_name is None:
            time_column_name = self.cols.Sampled_Date
        trajectory_sequences: List[pd.DataFrame] = []  # To store the sequences
        current_sequence = pd.DataFrame(columns=trajectory_sampled.columns) # DF To track the current sequence

        # Iterate through the timestamps
        for i in range(len(trajectory_sampled) - 1):
            if trajectory_sampled[time_column_name][i + 1] - trajectory_sampled[time_column_name][i] == self.sample_T:
                # If the difference is 10 minutes, add the current timestamp to the sequence
                if len(current_sequence) == 0:
                    current_sequence = trajectory_sampled.iloc[[i]] # Add the first timestamp of the sequence
                current_sequence = pd.concat([current_sequence, trajectory_sampled.iloc[[i+1]]], ignore_index=True)  # Add the next timestamp
            else:
                # If the difference is not 10 minutes, end the current sequence
                if len(current_sequence) != 0:
                    trajectory_sequences.append(current_sequence)  # Store the completed sequence
                    current_sequence = pd.DataFrame(columns=trajectory_sampled.columns)   # Reset the current sequence

        # Handle the last sequence if it ends at the last timestamp
        if len(current_sequence) != 0:
            trajectory_sequences.append(current_sequence)
            
        return trajectory_sequences
    
    @staticmethod
    def geodetic_to_ecef(lat, lon, alt=0):
        """Convert geodetic coordinates to ECEF coordinates."""
        x, y, z = transformer_to_ecef.transform(lon, lat, alt)
        return np.array([x, y, z])

    @staticmethod
    def ecef_to_geodetic(x, y, z):
        """Convert ECEF coordinates to geodetic coordinates (lat, lon, alt)."""
        lon, lat, alt = transformer_to_geodetic.transform(x, y, z)
        return lat, lon, alt

    @staticmethod
    def enu_matrix(lat, lon):
        """
        Compute the rotation matrix from ECEF to local ENU coordinates at a given geodetic point.
        The rows are the unit vectors for East, North, and Up, respectively.
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # East vector
        east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0])
        # North vector
        north = np.array([-np.sin(lat_rad)*np.cos(lon_rad),
                        -np.sin(lat_rad)*np.sin(lon_rad),
                        np.cos(lat_rad)])
        # Up vector
        up = np.array([np.cos(lat_rad)*np.cos(lon_rad),
                    np.cos(lat_rad)*np.sin(lon_rad),
                    np.sin(lat_rad)])
        
        return np.vstack((east, north, up))

    def rotate_trajectory(self, 
                          trajectory: pd.DataFrame, 
                          angle_deg: float # should be in degrees from 0 to 360
                          ) -> pd.DataFrame:
        # https://www.mdpi.com/2076-3417/14/3/1067
        """
        Rotate ship trajectory around a pivot point (midrange of Latitude and Longitude)
        by a given angle (in degrees) about the pivot's local up (vertical) direction.
        """
        angle_deg = angle_deg % MAX_DEGREES
        
        pivot_lat = (trajectory[self.cols.Latitude].max() + trajectory[self.cols.Latitude].min())/2
        pivot_lon = (trajectory[self.cols.Longitude].max() + trajectory[self.cols.Longitude].min())/2

        # Get pivot ECEF coordinates
        pivot_ecef = AISPreprocessor.geodetic_to_ecef(pivot_lat, pivot_lon)
        
        # Compute ENU transformation matrix at the pivot
        enu_mat = AISPreprocessor.enu_matrix(pivot_lat, pivot_lon)  # Rows: [east, north, up]
        
        # Convert ship trajectory to ECEF coordinates
        ecef_coords = np.array([
            AISPreprocessor.geodetic_to_ecef(lat, lon) for lat, lon in zip(trajectory[self.cols.Latitude], trajectory[self.cols.Longitude])
        ])
        
        # Translate to pivot frame
        relative_ecef = ecef_coords - pivot_ecef  # shape (n, 3)
        
        # Transform relative coordinates to local ENU frame
        # (Since enu_mat's rows are unit vectors, we multiply on the left)
        enu_coords = (enu_mat @ relative_ecef.T).T  # shape (n, 3)
        
        # Define a 2D rotation matrix for the horizontal (east, north) plane
        theta = np.radians(angle_deg)
        rot_2d = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        
        # Apply rotation to east and north components; leave up unchanged
        enu_rotated = enu_coords.copy()
        enu_rotated[:, :2] = (rot_2d @ enu_coords[:, :2].T).T
        
        # Convert rotated ENU coordinates back to ECEF
        # Since enu_mat is orthonormal, its transpose converts from ENU back to ECEF
        rotated_relative_ecef = (enu_mat.T @ enu_rotated.T).T  # shape (n, 3)
        
        # Translate back to the original ECEF frame
        new_ecef_coords = rotated_relative_ecef + pivot_ecef
        
        # Convert back to geodetic coordinates
        new_lat_lon = np.array([AISPreprocessor.ecef_to_geodetic(x, y, z)[:2] for x, y, z in new_ecef_coords])
        trajectory[[self.cols.Latitude, self.cols.Longitude]] = new_lat_lon
        trajectory[self.cols.COG] = (trajectory[self.cols.COG] + angle_deg) % MAX_DEGREES
        # rotate Heading only if it is not equal to 511 (reference: https://api.vtexplorer.com/docs/response-ais.html)
        trajectory[self.cols.Heading] = (trajectory[self.cols.Heading] != 511) * (trajectory[self.cols.Heading] + angle_deg) % MAX_DEGREES + (trajectory[self.cols.Heading] == 511) * 511
        
        return trajectory

    def get_random_bounded_latlon_shifts(self, trajectory):
        min_lat, max_lat = trajectory[self.cols.Latitude].min(), trajectory[self.cols.Latitude].max()
        min_lon, max_lon = trajectory[self.cols.Longitude].min(), trajectory[self.cols.Longitude].max()
        
        shift_lat_bounds = (self.min_lat - min_lat, self.max_lat - max_lat)
        shift_lon_bounds = (self.min_lon - min_lon, self.max_lon - max_lon)
        
        lat_shift = np.random.uniform(shift_lat_bounds[0], shift_lat_bounds[1])
        lon_shift = np.random.uniform(shift_lon_bounds[0], shift_lon_bounds[1])
        
        return lat_shift, lon_shift
    
    def shift_trajectory(self, trajectory, lat_shift, lon_shift):
        """Shift trajectory within lat_shift and lon_shift"""
        trajectory[self.cols.Latitude] += lat_shift
        trajectory[self.cols.Longitude] += lon_shift
        return trajectory
    
    def display_trajectory(self, trajectory: pd.DataFrame, i=0, base_path="data/good_trajectories/ship_trajectory_"):
        # is_synthetic = False
        # if self.cols.is_synthetic in list(trajectory.columns):
        #     is_synthetic = trajectory[self.cols.is_synthetic][0]
        
        fig = px.line_mapbox(
            trajectory,
            lat=self.cols.Latitude,
            lon=self.cols.Longitude,
            color = self.cols.is_synthetic,
            title=f"Ship Trajectory {i}",
            hover_data=[self.cols.SOG, self.cols.COG, self.cols.Heading]
        )
        
            # Calculate bounding box
        min_lat = trajectory[self.cols.Latitude].min()
        max_lat = trajectory[self.cols.Latitude].max()
        min_lon = trajectory[self.cols.Longitude].min()
        max_lon = trajectory[self.cols.Longitude].max()
        
        center_lat = (min_lat+max_lat)/2
        center_lon = (min_lon+max_lon)/2

        # Calculate zoom level based on bounding box
        delta_lat = max_lat - min_lat
        delta_lon = max_lon - min_lon

        # Simple heuristic for zoom level (adjust as needed)
        zoom_level = 13 - max(np.log2(delta_lat * 100), np.log2(delta_lon * 100))

        # Ensure zoom level is within reasonable bounds
        zoom_level = max(1, min(18, zoom_level))  # Adjust 1 and 18 as needed for your data.

        # Configure the map style
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=zoom_level+1,
            mapbox_center={"lat": center_lat, "lon": center_lon},
            height=600,
        )

        fig.write_html(f"{base_path}{i}.html")
        fig.write_image(f"{base_path}{i}.png")
        
        if self.cols.n_Latitude in trajectory.columns:
            fig = px.line(trajectory, x=self.cols.n_Longitude, y=self.cols.n_Latitude, title='Trajectory')
            fig.update_xaxes(range=[0, 1])
            fig.update_yaxes(range=[0, 1])
            fig.write_image(f"{base_path}{i}_norm.png")

if __name__ == "__main__":
    aisp = AISPreprocessor(dataset_path = 'data/fishing_boats_dynamic/Dynamic_*.csv')
    aisp.run()