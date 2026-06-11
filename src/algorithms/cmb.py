import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings


class CMBResult:
    def __init__(
        self,
        source_contributions: np.ndarray,
        source_uncertainties: np.ndarray,
        predicted_concentrations: np.ndarray,
        residuals: np.ndarray,
        chi_square: float,
        r_squared: float,
        condition_number: float,
        source_names: List[str],
        component_names: List[str],
    ):
        self.source_contributions = source_contributions
        self.source_uncertainties = source_uncertainties
        self.predicted_concentrations = predicted_concentrations
        self.residuals = residuals
        self.chi_square = chi_square
        self.r_squared = r_squared
        self.condition_number = condition_number
        self.source_names = source_names
        self.component_names = component_names

    def get_contribution_dataframe(self) -> Dict:
        avg_contrib = np.mean(self.source_contributions, axis=0)
        avg_uncert = np.mean(self.source_uncertainties, axis=0)
        total = np.sum(avg_contrib)
        percentages = (avg_contrib / total * 100) if total > 0 else np.zeros_like(avg_contrib)
        return {
            'source': self.source_names,
            'contribution': avg_contrib,
            'uncertainty': avg_uncert,
            'percentage': percentages,
        }


class CMBSolver:
    def __init__(
        self,
        source_names: List[str],
        component_names: List[str],
        source_matrix: np.ndarray,
        source_uncertainty_matrix: Optional[np.ndarray] = None,
    ):
        self.source_names = source_names
        self.component_names = component_names
        self.source_matrix = source_matrix
        self.source_uncertainty_matrix = source_uncertainty_matrix
        self.condition_number = self._compute_condition_number()

    def _compute_condition_number(self) -> float:
        U, S, Vt = np.linalg.svd(self.source_matrix)
        if np.min(S) < 1e-10:
            return np.inf
        return float(np.max(S) / np.min(S))

    def check_collinearity(self, threshold: float = 20.0) -> Tuple[bool, str]:
        cond_num = self.condition_number
        if cond_num > threshold:
            return True, f"条件数为 {cond_num:.2f}，超过阈值 {threshold}，源谱可能存在共线性问题。"
        return False, f"条件数为 {cond_num:.2f}，源谱线性无关性良好。"

    def solve(
        self,
        concentrations: np.ndarray,
        uncertainties: np.ndarray,
    ) -> CMBResult:
        n_samples = concentrations.shape[0]
        n_sources = len(self.source_names)
        n_components = len(self.component_names)

        source_contributions = np.zeros((n_samples, n_sources))
        source_uncertainties = np.zeros((n_samples, n_sources))
        predicted_concentrations = np.zeros((n_samples, n_components))
        residuals = np.zeros((n_samples, n_components))

        chi_square_total = 0.0
        total_samples = 0

        for i in range(n_samples):
            conc = concentrations[i]
            uncert = uncertainties[i]

            valid_mask = (uncert > 0) & (~np.isnan(conc)) & (~np.isnan(uncert))
            if np.sum(valid_mask) < n_sources:
                source_contributions[i, :] = np.nan
                source_uncertainties[i, :] = np.nan
                predicted_concentrations[i, :] = np.nan
                residuals[i, :] = np.nan
                continue

            F = self.source_matrix[valid_mask, :]
            c = conc[valid_mask]
            u = uncert[valid_mask]

            W = np.diag(1.0 / u**2)
            FtWF = F.T @ W @ F
            FtWc = F.T @ W @ c

            try:
                g = np.linalg.solve(FtWF, FtWc)
            except np.linalg.LinAlgError:
                g = np.linalg.lstsq(FtWF, FtWc, rcond=None)[0]

            cov_g = np.linalg.inv(FtWF)
            g_uncert = np.sqrt(np.diag(cov_g))

            pred_all = self.source_matrix @ g
            res_all = conc - pred_all

            chi2 = np.sum((res_all[valid_mask] / u) ** 2)
            chi_square_total += chi2
            total_samples += 1

            source_contributions[i, :] = g
            source_uncertainties[i, :] = g_uncert
            predicted_concentrations[i, :] = pred_all
            residuals[i, :] = res_all

        valid_mask = ~np.isnan(source_contributions[:, 0])
        if np.any(valid_mask):
            ss_res = np.sum(residuals[valid_mask] ** 2)
            ss_tot = np.sum((concentrations[valid_mask] - np.mean(concentrations[valid_mask], axis=0)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            r_squared = 0.0

        chi_square_avg = chi_square_total / total_samples if total_samples > 0 else 0.0

        return CMBResult(
            source_contributions=source_contributions,
            source_uncertainties=source_uncertainties,
            predicted_concentrations=predicted_concentrations,
            residuals=residuals,
            chi_square=chi_square_avg,
            r_squared=r_squared,
            condition_number=self.condition_number,
            source_names=self.source_names,
            component_names=self.component_names,
        )
