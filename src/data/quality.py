import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional


class DataQualityChecker:
    def __init__(self, outlier_threshold: float = 5.0):
        self.outlier_threshold = outlier_threshold
        self.quality_report = {}

    def check(self, df: pd.DataFrame, component_cols: List[str]) -> Tuple[pd.DataFrame, Dict]:
        result_df = df.copy()
        report = {
            'total_samples': len(df),
            'missing_values': {},
            'negative_values': {},
            'outliers': {},
            'valid_samples_after_qc': 0,
        }

        for col in component_cols:
            if col not in result_df.columns:
                continue

            missing_mask = result_df[col].isna()
            report['missing_values'][col] = int(missing_mask.sum())

            negative_mask = result_df[col] < 0
            report['negative_values'][col] = int(negative_mask.sum())
            result_df.loc[negative_mask, col] = np.nan

            valid_data = result_df[col].dropna()
            if len(valid_data) > 0:
                mean_val = valid_data.mean()
                std_val = valid_data.std()
                outlier_mask = result_df[col] > mean_val * self.outlier_threshold
                report['outliers'][col] = int(outlier_mask.sum())
                result_df.loc[outlier_mask, col] = np.nan

        valid_mask = result_df[component_cols].notna().all(axis=1)
        report['valid_samples_after_qc'] = int(valid_mask.sum())

        self.quality_report = report
        return result_df, report

    def get_report_summary(self) -> str:
        if not self.quality_report:
            return "No quality check performed yet."
        r = self.quality_report
        lines = [
            f"Total samples: {r['total_samples']}",
            f"Valid samples after QC: {r['valid_samples_after_qc']}",
        ]
        return "\n".join(lines)


def calculate_uncertainty(
    concentration: np.ndarray,
    detection_limit: float,
    relative_uncertainty: float = 0.1,
    dl_fraction: float = 1.0 / 3.0,
    below_dl_fraction: float = 5.0 / 6.0,
) -> np.ndarray:
    concentration = np.asarray(concentration, dtype=float)
    uncertainty = np.zeros_like(concentration)

    below_dl = concentration < detection_limit
    above_dl = ~below_dl

    uncertainty[below_dl] = detection_limit * below_dl_fraction
    uncertainty[above_dl] = (
        concentration[above_dl] * relative_uncertainty
        + detection_limit * dl_fraction
    )

    return uncertainty


def get_detection_limits(component_cols: List[str]) -> Dict[str, float]:
    default_dl = {
        'SO4': 0.05, 'NO3': 0.05, 'NH4': 0.03, 'Cl': 0.03,
        'Na': 0.02, 'K': 0.02, 'Ca': 0.02, 'Mg': 0.01,
        'OC': 0.5, 'EC': 0.2,
        'Al': 0.01, 'Si': 0.05, 'Fe': 0.01, 'Zn': 0.005,
        'Pb': 0.002, 'Mn': 0.002, 'Cu': 0.002, 'V': 0.001,
        'Ni': 0.001, 'As': 0.001, 'Cd': 0.001, 'Cr': 0.001,
        'Co': 0.001, 'Se': 0.001, 'Ba': 0.003, 'Sr': 0.001,
        'Ti': 0.005, 'levoglucosan': 0.01, 'heptane': 0.005,
        'heptadecane': 0.005,
    }
    return {col: default_dl.get(col, 0.01) for col in component_cols}


def get_season(date) -> str:
    month = date.month
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'


def add_season_column(df: pd.DataFrame, time_col: str = 'time') -> pd.DataFrame:
    result = df.copy()
    result['season'] = pd.to_datetime(result[time_col]).apply(get_season)
    return result
