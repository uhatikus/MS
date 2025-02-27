import pandas as pd
import glob

import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial.transform import Rotation as R

# Define the Coordinate Reference Systems
crs_geodetic = CRS.from_epsg(4326)  # WGS84
crs_ecef = CRS.from_epsg(4978)      # ECEF
# Create transformers
transformer_to_ecef = Transformer.from_crs(crs_geodetic, crs_ecef, always_xy=True)
transformer_to_geodetic = Transformer.from_crs(crs_ecef, crs_geodetic, always_xy=True)

class AISPreprocessor:
    def __init__(self,
                 dataset_path = '../../data/fishing_boats_dynamic/Dynamic_*.csv',
                 target_freq_in_minutes = 10,
                 min_trajectory_sequence_len = 10,
                 rotate_trajetories = True,
                 shitf_trajectories = False):
        self.dataset_path = dataset_path
        self.dynamic_data_files = glob.glob(self.dataset_path)
        self.all_boats_trajectories = {}
        self.good_trajectories = []
        self.normalized_trajectories = []
        
        self.min_lat, self.max_lat, self.min_lon, self.max_lon = None, None, None, None
        
        self.target_freq = f"{target_freq_in_minutes}min"
        self.sample_T = pd.Timedelta(minutes=target_freq_in_minutes)
        self.min_trajectory_sequence_len = min_trajectory_sequence_len
        
        self.rotate_trajetories = rotate_trajetories
        self.shitf_trajectories = shitf_trajectories
    
    def read_all_boats_trajectories(self):
        for dynamic_data_file in self.dynamic_data_files:
            print(f"Reading {dynamic_data_file}...")
            df_dynamic = pd.read_csv(dynamic_data_file)
            data_grouped = df_dynamic.groupby("MMSI")
            for mmsi, data in data_grouped:
                if mmsi not in self.all_boats_trajectories:
                    self.all_boats_trajectories[mmsi] = data.copy()  # Create a copy to avoid SettingWithCopyWarning
                else:
                    self.all_boats_trajectories[mmsi] = pd.concat([self.all_boats_trajectories[mmsi], data], ignore_index=True)
            print("Done!")
            
    def get_latlon_box(self):
        if len(self.good_trajectories) == 0:
            raise Exception("There are no good_trajectories")
        self.min_lat = 91
        self.max_lat = -91
        self.min_lon = 181
        self.max_lon = -181
        for trajectory in self.good_trajectories:
            lat = trajectory["Latitude"]
            lon = trajectory["Longitude"]
            if lat > self.max_lat:
                self.max_lat = lat
            if lat < self.min_lat:
                self.min_lat = lat
            if lon > self.max_lon:
                self.max_lon = lon
            if lon < self.min_lon:
                self.min_lon = lon
            
            
    
    def preprocess_all_boats_trajectories(self):
        for trajectory in self.all_boats_trajectories.values():
            sampled_trajectory = self.get_sampled_trajectory(trajectory)
            trajectory_sequences = self.get_trajectory_sequences(sampled_trajectory)
            
            current_good_trajectories = [seq for seq in trajectory_sequences if len(seq) >= self.min_trajectory_sequence_len]
            self.good_trajectories.extend(current_good_trajectories)
            
        self.get_latlon_box()
    
    def normalize_trajectory(self, trajectory):
        pass
            
    def normalize_trajectories(self, trajectories):
        self.normalized_trajectories = [self.normalize_trajectory(trajectory) for trajectory in trajectories]
            
    def synthesize_more_trajectories(self, trajectories):
        pass
    
    def get_sampled_trajectory(self, trajectory):
        trajectory["Date"] = pd.to_datetime(trajectory["Date"])
        trajectory = trajectory.set_index("Date")
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

        # Reindex only valid sampling times
        trajectory_interpolated = trajectory.reindex(trajectory.index.union(valid_sampling_times)).sort_index()

        # Perform linear interpolation
        trajectory_interpolated = trajectory_interpolated.interpolate(method="time")

        # Keep only the sampled timestamps and drop any remaining NaNs
        trajectory_sampled = trajectory_interpolated.loc[valid_sampling_times].dropna().reset_index()
        trajectory_sampled.rename(columns={"index": "Sampled_Date"}, inplace=True)
        return trajectory_sampled
    
    def get_trajectory_sequences(self, trajectory_sampled, time_column_name="Sampled_Date"):
        trajectory_sequences = []  # To store the sequences
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

    def rotate_trajectory(self, trajectory, angle_deg):
        """
        Rotate ship trajectory around a pivot point (given in geodetic coordinates)
        by a given angle (in degrees) about the pivot's local up (vertical) direction.
        """
        angle_deg = 180
        pivot_lat = (trajectory['Latitude'].max() + trajectory['Latitude'].min())/2
        pivot_lon = (trajectory['Longitude'].max() + trajectory['Longitude'].min())/2

        # Get pivot ECEF coordinates
        pivot_ecef = AISPreprocessor.geodetic_to_ecef(pivot_lat, pivot_lon)
        
        # Compute ENU transformation matrix at the pivot
        enu_mat = AISPreprocessor.enu_matrix(pivot_lat, pivot_lon)  # Rows: [east, north, up]
        
        # Convert ship trajectory to ECEF coordinates
        ecef_coords = np.array([
            AISPreprocessor.geodetic_to_ecef(lat, lon) for lat, lon in zip(trajectory['Latitude'], trajectory['Longitude'])
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
        trajectory[['Latitude', 'Longitude']] = new_lat_lon
        trajectory["COG"] = (trajectory["COG"] + angle_deg) % 360
        trajectory["Heading ROTATED"] = (trajectory["Heading"] != 511) * (trajectory["Heading"] + angle_deg) % 360 + (trajectory["Heading"] == 511) * 511
        return trajectory

    # @staticmethod
    # def shift_trajectory(df, lat_bounds=(30, 40), lon_bounds=(120, 130)):
    #     """Shift trajectory if needed to ensure it stays within lat/lon bounds."""
    #     min_lat, max_lat = df['Latitude'].min(), df['Latitude'].max()
    #     min_lon, max_lon = df['Longitude'].min(), df['Longitude'].max()
    #     lat_shift = 0 if lat_bounds[0] <= min_lat and max_lat <= lat_bounds[1] else lat_bounds[0] - min_lat
    #     lon_shift = 0 if lon_bounds[0] <= min_lon and max_lon <= lon_bounds[1] else lon_bounds[0] - min_lon
        
    #     df['Latitude'] += lat_shift
    #     df['Longitude'] += lon_shift
    #     return df