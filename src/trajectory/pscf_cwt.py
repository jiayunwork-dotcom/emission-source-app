import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PSCFResult:
    pscf_values: np.ndarray
    n_ij: np.ndarray
    m_ij: np.ndarray
    lat_edges: np.ndarray
    lon_edges: np.ndarray
    weighted: bool
    threshold_percentile: float


@dataclass
class CWTResult:
    cwt_values: np.ndarray
    n_ij: np.ndarray
    lat_edges: np.ndarray
    lon_edges: np.ndarray
    weighted: bool


class TrajectoryAnalyzer:
    def __init__(
        self,
        lat_range: Tuple[float, float] = (15.0, 55.0),
        lon_range: Tuple[float, float] = (70.0, 140.0),
        grid_resolution: float = 0.5,
    ):
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.grid_resolution = grid_resolution
        self.lat_edges = np.arange(lat_range[0], lat_range[1] + grid_resolution, grid_resolution)
        self.lon_edges = np.arange(lon_range[0], lon_range[1] + grid_resolution, grid_resolution)
        self.n_lat = len(self.lat_edges) - 1
        self.n_lon = len(self.lon_edges) - 1

    def _get_grid_indices(self, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lat_idx = np.floor((lat - self.lat_range[0]) / self.grid_resolution).astype(int)
        lon_idx = np.floor((lon - self.lon_range[0]) / self.grid_resolution).astype(int)
        valid = (lat_idx >= 0) & (lat_idx < self.n_lat) & (lon_idx >= 0) & (lon_idx < self.n_lon)
        return lat_idx, lon_idx, valid

    def compute_pscf(
        self,
        trajectories: List[pd.DataFrame],
        concentrations: np.ndarray,
        threshold_percentile: float = 75.0,
        apply_weighting: bool = True,
    ) -> PSCFResult:
        n_trajectories = len(trajectories)

        n_ij = np.zeros((self.n_lat, self.n_lon), dtype=float)
        m_ij = np.zeros((self.n_lat, self.n_lon), dtype=float)

        threshold = np.percentile(concentrations, threshold_percentile)
        is_high_pollution = concentrations >= threshold

        for i in range(n_trajectories):
            traj = trajectories[i]
            if 'lat' not in traj.columns or 'lon' not in traj.columns:
                continue

            lat = traj['lat'].values
            lon = traj['lon'].values

            lat_idx, lon_idx, valid = self._get_grid_indices(lat, lon)

            valid_indices = np.where(valid)[0]
            if len(valid_indices) > 0:
                grid_cells = set(zip(lat_idx[valid_indices], lon_idx[valid_indices]))
                for li, lj in grid_cells:
                    n_ij[li, lj] += 1
                    if is_high_pollution[i]:
                        m_ij[li, lj] += 1

        pscf = np.zeros_like(n_ij)
        valid_mask = n_ij > 0
        pscf[valid_mask] = m_ij[valid_mask] / n_ij[valid_mask]

        if apply_weighting:
            avg_n = np.mean(n_ij[n_ij > 0]) if np.any(n_ij > 0) else 1.0
            weight_threshold = 3.0 * avg_n

            weights = np.ones_like(n_ij)
            low_weight_mask = (n_ij > 0) & (n_ij < weight_threshold)
            weights[low_weight_mask] = n_ij[low_weight_mask] / weight_threshold

            pscf = pscf * weights
        else:
            weights = np.ones_like(n_ij)

        return PSCFResult(
            pscf_values=pscf,
            n_ij=n_ij,
            m_ij=m_ij,
            lat_edges=self.lat_edges,
            lon_edges=self.lon_edges,
            weighted=apply_weighting,
            threshold_percentile=threshold_percentile,
        )

    def compute_cwt(
        self,
        trajectories: List[pd.DataFrame],
        concentrations: np.ndarray,
        time_step_hours: float = 1.0,
        apply_weighting: bool = True,
    ) -> CWTResult:
        n_trajectories = len(trajectories)

        n_ij = np.zeros((self.n_lat, self.n_lon), dtype=float)
        cwt = np.zeros((self.n_lat, self.n_lon), dtype=float)

        for i in range(n_trajectories):
            traj = trajectories[i]
            if 'lat' not in traj.columns or 'lon' not in traj.columns:
                continue

            lat = traj['lat'].values
            lon = traj['lon'].values
            n_points = len(lat)

            lat_idx, lon_idx, valid = self._get_grid_indices(lat, lon)

            valid_indices = np.where(valid)[0]
            if len(valid_indices) > 0:
                tau = time_step_hours * np.ones(n_points)

                for idx in valid_indices:
                    li = lat_idx[idx]
                    lj = lon_idx[idx]
                    n_ij[li, lj] += tau[idx]
                    cwt[li, lj] += concentrations[i] * tau[idx]

        valid_mask = n_ij > 0
        cwt[valid_mask] = cwt[valid_mask] / n_ij[valid_mask]

        if apply_weighting:
            avg_n = np.mean(n_ij[n_ij > 0]) if np.any(n_ij > 0) else 1.0
            weight_threshold = 3.0 * avg_n

            weights = np.ones_like(n_ij)
            low_weight_mask = (n_ij > 0) & (n_ij < weight_threshold)
            weights[low_weight_mask] = n_ij[low_weight_mask] / weight_threshold

            cwt = cwt * weights

        return CWTResult(
            cwt_values=cwt,
            n_ij=n_ij,
            lat_edges=self.lat_edges,
            lon_edges=self.lon_edges,
            weighted=apply_weighting,
        )

    def load_trajectories_from_csv(self, filepath: str) -> List[pd.DataFrame]:
        df = pd.read_csv(filepath)
        trajectories = []
        if 'trajectory_id' in df.columns:
            for traj_id, group in df.groupby('trajectory_id'):
                trajectories.append(group.reset_index(drop=True))
        else:
            trajectories.append(df)
        return trajectories

    def generate_synthetic_trajectories(
        self,
        n_trajectories: int = 50,
        start_lat: float = 39.9,
        start_lon: float = 116.4,
        duration_hours: int = 72,
        time_step_hours: int = 1,
        random_seed: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        if random_seed is not None:
            np.random.seed(random_seed)

        trajectories = []
        n_steps = duration_hours // time_step_hours

        for i in range(n_trajectories):
            speed = 5.0 + np.random.rand() * 15.0
            direction = np.random.rand() * 360.0
            direction_rad = np.radians(direction)

            lat = np.zeros(n_steps)
            lon = np.zeros(n_steps)
            height = np.zeros(n_steps)

            lat[0] = start_lat
            lon[0] = start_lon
            height[0] = 100.0 + np.random.rand() * 200.0

            for t in range(1, n_steps):
                turn_angle = np.random.normal(0, 5.0)
                direction_rad += np.radians(turn_angle)

                dx = speed * np.sin(direction_rad) * time_step_hours / 111.0
                dy = speed * np.cos(direction_rad) * time_step_hours / 111.0

                lat[t] = lat[t-1] + dy
                lon[t] = lon[t-1] + dx / np.cos(np.radians(lat[t]))
                height[t] = max(50.0, height[t-1] + np.random.normal(0, 20.0))

            times = pd.date_range('2024-01-01', periods=n_steps, freq=f'{time_step_hours}h')

            traj_df = pd.DataFrame({
                'time': times,
                'trajectory_id': i,
                'lat': lat,
                'lon': lon,
                'height': height,
            })
            trajectories.append(traj_df)

        return trajectories
