import numpy as np
import pandas as pd
from typing import List, Tuple
import os


def generate_sample_data(
    n_samples: int = 100,
    n_stations: int = 3,
    output_dir: str = "data",
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)

    component_cols = [
        'SO4', 'NO3', 'NH4', 'Cl', 'Na', 'K', 'Ca', 'Mg',
        'OC', 'EC',
        'Al', 'Si', 'Fe', 'Zn', 'Pb', 'Mn', 'Cu', 'V', 'Ni', 'As',
        'Cd', 'Cr', 'Co', 'Se', 'Ba', 'Sr', 'Ti',
        'levoglucosan',
    ]

    source_profiles = {
        'coal': {
            'SO4': 0.20, 'NO3': 0.05, 'NH4': 0.03, 'Cl': 0.02,
            'Na': 0.005, 'K': 0.01, 'Ca': 0.03, 'Mg': 0.008,
            'OC': 0.08, 'EC': 0.04,
            'Al': 0.05, 'Si': 0.15, 'Fe': 0.04, 'Zn': 0.003,
            'Pb': 0.002, 'Mn': 0.002, 'Cu': 0.001, 'V': 0.003,
            'Ni': 0.001, 'As': 0.002, 'Se': 0.0005,
            'Cd': 0.0003, 'Cr': 0.0005, 'Co': 0.0002,
            'Ba': 0.001, 'Sr': 0.0003, 'Ti': 0.003,
            'levoglucosan': 0.001,
        },
        'vehicle': {
            'SO4': 0.03, 'NO3': 0.08, 'NH4': 0.02, 'Cl': 0.01,
            'Na': 0.005, 'K': 0.008, 'Ca': 0.005, 'Mg': 0.003,
            'OC': 0.25, 'EC': 0.20,
            'Al': 0.005, 'Si': 0.01, 'Fe': 0.01, 'Zn': 0.01,
            'Pb': 0.003, 'Mn': 0.005, 'Cu': 0.008, 'V': 0.001,
            'Ni': 0.002, 'As': 0.0005, 'Se': 0.0001,
            'Cd': 0.0002, 'Cr': 0.0003, 'Co': 0.0001,
            'Ba': 0.005, 'Sr': 0.0002, 'Ti': 0.001,
            'levoglucosan': 0.0005,
        },
        'dust': {
            'SO4': 0.02, 'NO3': 0.01, 'NH4': 0.005, 'Cl': 0.01,
            'Na': 0.02, 'K': 0.015, 'Ca': 0.15, 'Mg': 0.03,
            'OC': 0.05, 'EC': 0.02,
            'Al': 0.10, 'Si': 0.25, 'Fe': 0.06, 'Zn': 0.002,
            'Pb': 0.001, 'Mn': 0.003, 'Cu': 0.001, 'V': 0.0005,
            'Ni': 0.0005, 'As': 0.0002, 'Se': 0.0001,
            'Cd': 0.0001, 'Cr': 0.0002, 'Co': 0.0001,
            'Ba': 0.001, 'Sr': 0.001, 'Ti': 0.008,
            'levoglucosan': 0.0002,
        },
        'secondary': {
            'SO4': 0.35, 'NO3': 0.25, 'NH4': 0.15, 'Cl': 0.01,
            'Na': 0.005, 'K': 0.01, 'Ca': 0.005, 'Mg': 0.003,
            'OC': 0.10, 'EC': 0.005,
            'Al': 0.002, 'Si': 0.005, 'Fe': 0.002, 'Zn': 0.001,
            'Pb': 0.0005, 'Mn': 0.0002, 'Cu': 0.0002, 'V': 0.0001,
            'Ni': 0.0001, 'As': 0.0002, 'Se': 0.0001,
            'Cd': 0.0001, 'Cr': 0.0001, 'Co': 0.00005,
            'Ba': 0.0002, 'Sr': 0.0001, 'Ti': 0.0005,
            'levoglucosan': 0.0002,
        },
    }

    station_names = ['站点A', '站点B', '站点C'][:n_stations]

    filepaths = []

    for station_idx in range(n_stations):
        station_name = station_names[station_idx]
        np.random.seed(station_idx)

        source_contribs = np.zeros((n_samples, 4))
        source_contribs[:, 0] = np.random.gamma(2, 10, n_samples)
        source_contribs[:, 1] = np.random.gamma(1.5, 8, n_samples)
        source_contribs[:, 2] = np.random.gamma(2, 6, n_samples)
        source_contribs[:, 3] = np.random.gamma(2, 12, n_samples)

        source_contribs[:, 0] *= 15
        source_contribs[:, 1] *= 12
        source_contribs[:, 2] *= 8
        source_contribs[:, 3] *= 20

        data = np.zeros((n_samples, len(component_cols)))

        for i, comp in enumerate(component_cols):
            vals = np.zeros(n_samples)
            for j, source in enumerate(['coal', 'vehicle', 'dust', 'secondary']):
                frac = source_profiles[source].get(comp, 0)
                vals += source_contribs[:, j] * frac
            noise = np.random.normal(0, 0.1, n_samples)
            vals = vals * (1 + noise)
            vals = np.maximum(vals, 0.001)
            data[:, i] = vals

        start_date = pd.Timestamp('2024-01-01')
        dates = pd.date_range(start_date, periods=n_samples, freq='D')

        df = pd.DataFrame(data, columns=component_cols)
        df.insert(0, 'time', dates)
        df.insert(1, 'station', station_name)

        filepath = os.path.join(output_dir, f'{station_name}_pm25_components.csv')
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        filepaths.append(filepath)

        print(f"Generated: {filepath}")

    return filepaths


def generate_sample_source_spectrum(output_dir: str = "data") -> str:
    os.makedirs(output_dir, exist_ok=True)

    components = [
        'SO4', 'NO3', 'NH4', 'Cl', 'Na', 'K', 'Ca', 'Mg',
        'OC', 'EC', 'Al', 'Si', 'Fe', 'Zn', 'Pb',
    ]

    fractions = [
        0.15, 0.10, 0.08, 0.02, 0.01, 0.02, 0.05, 0.01,
        0.12, 0.05, 0.03, 0.08, 0.02, 0.005, 0.003,
    ]

    uncertainties = [f * 0.1 for f in fractions]

    df = pd.DataFrame({
        'component': components,
        'fraction': fractions,
        'uncertainty': uncertainties,
    })

    filepath = os.path.join(output_dir, 'custom_source_example.csv')
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"Generated: {filepath}")
    return filepath


def generate_sample_trajectories(
    n_trajectories: int = 50,
    output_dir: str = "data",
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)

    all_trajectories = []
    start_lat = 39.9
    start_lon = 116.4
    duration_hours = 72
    time_step = 1
    n_steps = duration_hours // time_step

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

            dx = speed * np.sin(direction_rad) * time_step / 111.0
            dy = speed * np.cos(direction_rad) * time_step / 111.0

            lat[t] = lat[t-1] + dy
            lon[t] = lon[t-1] + dx / np.cos(np.radians(lat[t]))
            height[t] = max(50.0, height[t-1] + np.random.normal(0, 20.0))

        times = pd.date_range('2024-01-01 00:00', periods=n_steps, freq='h')

        traj_df = pd.DataFrame({
            'trajectory_id': i,
            'time': times,
            'lat': lat,
            'lon': lon,
            'height': height,
        })
        all_trajectories.append(traj_df)

    all_df = pd.concat(all_trajectories, ignore_index=True)
    filepath = os.path.join(output_dir, 'sample_trajectories.csv')
    all_df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"Generated: {filepath}")
    return filepath


if __name__ == '__main__':
    generate_sample_data()
    generate_sample_source_spectrum()
    generate_sample_trajectories()
    print("\nAll sample data generated successfully!")
