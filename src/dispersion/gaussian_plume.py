import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class EmissionSource:
    name: str
    source_strength: float
    effective_height: float
    x: float
    y: float


@dataclass
class DispersionResult:
    concentration_field: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    max_concentration: float
    max_location: Tuple[float, float]
    source_contributions: Dict[str, np.ndarray]
    source_contribution_at_max: Dict[str, float]
    centerline_concentrations: Dict[str, Tuple[np.ndarray, np.ndarray]]
    parameters: Dict


class GaussianPlumeModel:
    STABILITY_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F']

    PG_PARAMS = {
        'y': {
            'A': {'a': 213.0, 'b': 0.894, 'c': 440.8, 'd': 1.66, 'e': -1.0},
            'B': {'a': 156.0, 'b': 0.894, 'c': 108.2, 'd': 1.098, 'e': 0.0},
            'C': {'a': 104.0, 'b': 0.894, 'c': 61.0, 'd': 0.911, 'e': 0.0},
            'D': {'a': 68.0, 'b': 0.894, 'c': 33.5, 'd': 0.805, 'e': 0.0},
            'E': {'a': 50.5, 'b': 0.894, 'c': 22.0, 'd': 0.707, 'e': 0.0},
            'F': {'a': 34.0, 'b': 0.894, 'c': 14.35, 'd': 0.634, 'e': 0.0},
        },
        'z': {
            'A': {'a': 453.85, 'b': 2.1166, 'c': 440.8, 'd': 1.66, 'e': -1.0},
            'B': {'a': 106.15, 'b': 1.0569, 'c': 108.2, 'd': 1.098, 'e': 0.0},
            'C': {'a': 61.14, 'b': 0.9147, 'c': 61.0, 'd': 0.911, 'e': 0.0},
            'D': {'a': 30.49, 'b': 0.7306, 'c': 33.5, 'd': 0.805, 'e': 0.0},
            'E': {'a': 21.53, 'b': 0.6784, 'c': 22.0, 'd': 0.707, 'e': 0.0},
            'F': {'a': 13.86, 'b': 0.6319, 'c': 14.35, 'd': 0.634, 'e': 0.0},
        }
    }

    def __init__(
        self,
        wind_speed: float = 3.0,
        wind_direction: float = 225.0,
        stability_class: str = 'D',
        mixing_height: float = 800.0,
        temperature: float = 20.0,
        domain_size: float = 20.0,
        grid_resolution: float = 0.2,
        background_concentration: float = 5.0,
    ):
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.stability_class = stability_class.upper()
        self.mixing_height = mixing_height
        self.temperature = temperature
        self.domain_size = domain_size
        self.grid_resolution = grid_resolution
        self.background_concentration = background_concentration
        self.sources: List[EmissionSource] = []

    def add_source(self, source: EmissionSource):
        self.sources.append(source)

    def set_sources(self, sources: List[EmissionSource]):
        self.sources = sources

    def _calculate_sigma_y(self, x_km: float) -> float:
        x_m = x_km * 1000.0
        if x_m <= 0:
            return 0.1
        params = self.PG_PARAMS['y'][self.stability_class]
        if x_m < 1000:
            sigma = params['a'] * (x_m / 1000.0) ** params['b']
        else:
            sigma = params['c'] * (x_m / 1000.0) ** params['d'] + params['e']
        return max(sigma, 0.1)

    def _calculate_sigma_z(self, x_km: float) -> float:
        x_m = x_km * 1000.0
        if x_m <= 0:
            return 0.1
        params = self.PG_PARAMS['z'][self.stability_class]
        if x_m < 1000:
            sigma = params['a'] * (x_m / 1000.0) ** params['b']
        else:
            sigma = params['c'] * (x_m / 1000.0) ** params['d'] + params['e']
        sigma = min(sigma, self.mixing_height)
        return max(sigma, 0.1)

    def _gaussian_concentration(
        self,
        Q: float,
        u: float,
        sigma_y: float,
        sigma_z: float,
        y: float,
        z: float,
        H: float,
    ) -> float:
        if u <= 0 or sigma_y <= 0 or sigma_z <= 0:
            return 0.0

        term1 = Q / (2 * np.pi * u * sigma_y * sigma_z)
        y_term = np.exp(-0.5 * (y / sigma_y) ** 2)

        if H < 1.0:
            z_term = 2 * np.exp(-0.5 * (z / sigma_z) ** 2)
        else:
            h = self.mixing_height
            if h <= 0:
                z_direct = np.exp(-0.5 * ((z - H) / sigma_z) ** 2)
                z_reflect = np.exp(-0.5 * ((z + H) / sigma_z) ** 2)
                z_term = z_direct + z_reflect
            else:
                z_term = 0.0
                for n in range(-3, 4):
                    z_direct = np.exp(-0.5 * ((z - H + 2 * n * h) / sigma_z) ** 2)
                    z_reflect = np.exp(-0.5 * ((z + H + 2 * n * h) / sigma_z) ** 2)
                    z_term += z_direct + z_reflect

        return term1 * y_term * z_term

    def _rotate_coordinates(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        source_x: float,
        source_y: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        dx = x_grid - source_x
        dy = y_grid - source_y

        theta_rad = np.radians(270.0 - self.wind_direction)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        x_downwind = dx * cos_theta + dy * sin_theta
        y_crosswind = -dx * sin_theta + dy * cos_theta

        return x_downwind, y_crosswind

    def simulate(self, selected_source: Optional[str] = None) -> DispersionResult:
        n_points = int(self.domain_size / self.grid_resolution) + 1
        x_coords = np.linspace(-self.domain_size / 2, self.domain_size / 2, n_points)
        y_coords = np.linspace(-self.domain_size / 2, self.domain_size / 2, n_points)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        total_concentration = np.zeros_like(x_grid)
        source_contributions: Dict[str, np.ndarray] = {}
        centerline_concentrations: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        sources_to_process = self.sources
        if selected_source is not None:
            sources_to_process = [s for s in self.sources if s.name == selected_source]

        for source in sources_to_process:
            source_field = np.zeros_like(x_grid)

            x_down, y_cross = self._rotate_coordinates(
                x_grid, y_grid, source.x, source.y
            )

            for i in range(x_grid.shape[0]):
                for j in range(x_grid.shape[1]):
                    x_d = x_down[i, j]
                    y_c = y_cross[i, j]

                    if x_d <= 0:
                        continue

                    sigma_y = self._calculate_sigma_y(x_d)
                    sigma_z = self._calculate_sigma_z(x_d)

                    Q_ug_s = source.source_strength * 1e6
                    conc = self._gaussian_concentration(
                        Q_ug_s, self.wind_speed, sigma_y, sigma_z,
                        y_c * 1000.0, 0.0, source.effective_height
                    )
                    source_field[i, j] = conc

            source_contributions[source.name] = source_field
            total_concentration += source_field

            cl_distances = np.linspace(0, self.domain_size, 200)
            cl_concentrations = np.zeros_like(cl_distances)
            for idx, d in enumerate(cl_distances):
                if d <= 0:
                    continue
                sigma_y = self._calculate_sigma_y(d)
                sigma_z = self._calculate_sigma_z(d)
                Q_ug_s = source.source_strength * 1e6
                cl_concentrations[idx] = self._gaussian_concentration(
                    Q_ug_s, self.wind_speed, sigma_y, sigma_z,
                    0.0, 0.0, source.effective_height
                )
            centerline_concentrations[source.name] = (cl_distances, cl_concentrations)

        total_concentration += self.background_concentration

        max_idx = np.unravel_index(np.argmax(total_concentration), total_concentration.shape)
        max_concentration = total_concentration[max_idx]
        max_x = x_grid[max_idx]
        max_y = y_grid[max_idx]

        source_contribution_at_max = {}
        for name, field in source_contributions.items():
            source_contribution_at_max[name] = field[max_idx]

        parameters = {
            'wind_speed': self.wind_speed,
            'wind_direction': self.wind_direction,
            'stability_class': self.stability_class,
            'mixing_height': self.mixing_height,
            'temperature': self.temperature,
            'domain_size': self.domain_size,
            'grid_resolution': self.grid_resolution,
            'background_concentration': self.background_concentration,
        }

        return DispersionResult(
            concentration_field=total_concentration,
            x_grid=x_grid,
            y_grid=y_grid,
            max_concentration=max_concentration,
            max_location=(max_x, max_y),
            source_contributions=source_contributions,
            source_contribution_at_max=source_contribution_at_max,
            centerline_concentrations=centerline_concentrations,
            parameters=parameters,
        )

    def simulate_with_weights(
        self,
        weights: Dict[str, float],
        selected_source: Optional[str] = None,
    ) -> DispersionResult:
        n_points = int(self.domain_size / self.grid_resolution) + 1
        x_coords = np.linspace(-self.domain_size / 2, self.domain_size / 2, n_points)
        y_coords = np.linspace(-self.domain_size / 2, self.domain_size / 2, n_points)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        total_concentration = np.zeros_like(x_grid)
        source_contributions: Dict[str, np.ndarray] = {}
        centerline_concentrations: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        sources_to_process = self.sources
        if selected_source is not None:
            sources_to_process = [s for s in self.sources if s.name == selected_source]

        for source in sources_to_process:
            w = weights.get(source.name, 1.0)
            adjusted_strength = source.source_strength * w
            source_field = np.zeros_like(x_grid)

            x_down, y_cross = self._rotate_coordinates(
                x_grid, y_grid, source.x, source.y
            )

            for i in range(x_grid.shape[0]):
                for j in range(x_grid.shape[1]):
                    x_d = x_down[i, j]
                    y_c = y_cross[i, j]

                    if x_d <= 0:
                        continue

                    sigma_y = self._calculate_sigma_y(x_d)
                    sigma_z = self._calculate_sigma_z(x_d)

                    Q_ug_s = adjusted_strength * 1e6
                    conc = self._gaussian_concentration(
                        Q_ug_s, self.wind_speed, sigma_y, sigma_z,
                        y_c * 1000.0, 0.0, source.effective_height
                    )
                    source_field[i, j] = conc

            source_contributions[source.name] = source_field
            total_concentration += source_field

            cl_distances = np.linspace(0, self.domain_size, 200)
            cl_concentrations = np.zeros_like(cl_distances)
            for idx, d in enumerate(cl_distances):
                if d <= 0:
                    continue
                sigma_y = self._calculate_sigma_y(d)
                sigma_z = self._calculate_sigma_z(d)
                Q_ug_s = adjusted_strength * 1e6
                cl_concentrations[idx] = self._gaussian_concentration(
                    Q_ug_s, self.wind_speed, sigma_y, sigma_z,
                    0.0, 0.0, source.effective_height
                )
            centerline_concentrations[source.name] = (cl_distances, cl_concentrations)

        total_concentration += self.background_concentration

        max_idx = np.unravel_index(np.argmax(total_concentration), total_concentration.shape)
        max_concentration = total_concentration[max_idx]
        max_x = x_grid[max_idx]
        max_y = y_grid[max_idx]

        source_contribution_at_max = {}
        for name, field in source_contributions.items():
            source_contribution_at_max[name] = field[max_idx]

        parameters = {
            'wind_speed': self.wind_speed,
            'wind_direction': self.wind_direction,
            'stability_class': self.stability_class,
            'mixing_height': self.mixing_height,
            'temperature': self.temperature,
            'domain_size': self.domain_size,
            'grid_resolution': self.grid_resolution,
            'background_concentration': self.background_concentration,
            'weights': weights,
        }

        return DispersionResult(
            concentration_field=total_concentration,
            x_grid=x_grid,
            y_grid=y_grid,
            max_concentration=max_concentration,
            max_location=(max_x, max_y),
            source_contributions=source_contributions,
            source_contribution_at_max=source_contribution_at_max,
            centerline_concentrations=centerline_concentrations,
            parameters=parameters,
        )

    def compute_receptor_concentrations(
        self,
        receptor_points: List[Dict],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        if weights is None:
            weights = {s.name: 1.0 for s in self.sources}

        results = {}
        for rp in receptor_points:
            rp_name = rp.get('name', f"({rp['x']:.1f},{rp['y']:.1f})")
            rx, ry = rp['x'], rp['y']
            source_contribs = {}

            for source in self.sources:
                w = weights.get(source.name, 1.0)
                adjusted_strength = source.source_strength * w

                dx = rx - source.x
                dy = ry - source.y
                theta_rad = np.radians(270.0 - self.wind_direction)
                cos_theta = np.cos(theta_rad)
                sin_theta = np.sin(theta_rad)
                x_downwind = dx * cos_theta + dy * sin_theta
                y_crosswind = -dx * sin_theta + dy * cos_theta

                if x_downwind <= 0:
                    source_contribs[source.name] = 0.0
                    continue

                sigma_y = self._calculate_sigma_y(x_downwind)
                sigma_z = self._calculate_sigma_z(x_downwind)
                Q_ug_s = adjusted_strength * 1e6
                conc = self._gaussian_concentration(
                    Q_ug_s, self.wind_speed, sigma_y, sigma_z,
                    y_crosswind * 1000.0, 0.0, source.effective_height
                )
                source_contribs[source.name] = conc

            total = sum(source_contribs.values()) + self.background_concentration
            source_contribs['总浓度'] = total
            results[rp_name] = source_contribs

        return results

    def get_influence_radius(
        self,
        source_name: str,
        threshold: Optional[float] = None,
    ) -> float:
        if threshold is None:
            threshold = self.background_concentration

        source = next((s for s in self.sources if s.name == source_name), None)
        if source is None:
            return 0.0

        distances = np.linspace(0.01, self.domain_size, 500)
        for d in distances:
            sigma_y = self._calculate_sigma_y(d)
            sigma_z = self._calculate_sigma_z(d)
            Q_ug_s = source.source_strength * 1e6
            conc = self._gaussian_concentration(
                Q_ug_s, self.wind_speed, sigma_y, sigma_z,
                0.0, 0.0, source.effective_height
            )
            if conc < threshold:
                return d

        return self.domain_size
