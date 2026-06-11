import numpy as np
from typing import Tuple, Dict, List, Optional


class PCAMLRResult:
    def __init__(
        self,
        source_contributions: np.ndarray,
        loadings: np.ndarray,
        explained_variance: np.ndarray,
        cumulative_variance: np.ndarray,
        n_components: int,
        residuals: np.ndarray,
        r_squared: float,
        component_names: List[str],
        source_names: Optional[List[str]] = None,
    ):
        self.source_contributions = source_contributions
        self.loadings = loadings
        self.explained_variance = explained_variance
        self.cumulative_variance = cumulative_variance
        self.n_components = n_components
        self.residuals = residuals
        self.r_squared = r_squared
        self.component_names = component_names
        self.source_names = source_names if source_names else [f'PC {i+1}' for i in range(n_components)]

    def get_contribution_dataframe(self) -> Dict:
        avg_contrib = np.mean(np.abs(self.source_contributions), axis=0)
        total = np.sum(avg_contrib)
        percentages = (avg_contrib / total * 100) if total > 0 else np.zeros_like(avg_contrib)
        return {
            'source': self.source_names,
            'contribution': avg_contrib,
            'percentage': percentages,
        }

    def get_loadings_dataframe(self) -> Dict:
        return {
            'components': self.component_names,
            'factors': self.source_names,
            'loadings': self.loadings,
        }


class PCAMLRSolver:
    def __init__(
        self,
        component_names: List[str],
        variance_threshold: float = 0.8,
        max_components: Optional[int] = None,
    ):
        self.component_names = component_names
        self.variance_threshold = variance_threshold
        self.max_components = max_components

    def _pca(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        n_samples, n_features = X.shape
        X_centered = X - np.mean(X, axis=0)

        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        cumulative_variance = np.cumsum(explained_variance_ratio)

        n_components = np.searchsorted(cumulative_variance, self.variance_threshold) + 1
        if self.max_components is not None:
            n_components = min(n_components, self.max_components)
        n_components = min(n_components, n_features, n_samples)

        loadings = eigenvectors[:, :n_components] * np.sqrt(eigenvalues[:n_components])
        scores = X_centered @ eigenvectors[:, :n_components]
        eigenvecs = eigenvectors[:, :n_components]
        eigvals = eigenvalues[:n_components]

        return scores, loadings, explained_variance_ratio, eigenvecs, eigvals, n_components

    def _mlr(self, y: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        n_samples = len(y)
        X_aug = np.column_stack([np.ones(n_samples), scores])

        try:
            beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(X_aug.shape[1])

        y_pred = X_aug @ beta
        residuals = y - y_pred

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return beta, r_squared, residuals

    def solve(self, X: np.ndarray, total_mass: Optional[np.ndarray] = None) -> PCAMLRResult:
        if total_mass is None:
            total_mass = np.sum(X, axis=1)

        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(total_mass)
        X_valid = X[valid_mask]
        total_valid = total_mass[valid_mask]
        mean_X = np.mean(X_valid, axis=0)

        scores, loadings, explained_variance_ratio, eigenvecs, eigvals, n_components = self._pca(X_valid)

        beta, r_squared, residuals = self._mlr(total_valid, scores)

        X_centered_full = X - mean_X
        scores_full = X_centered_full @ eigenvecs
        source_contributions_full = scores_full * beta[1:][np.newaxis, :]

        residuals_full = np.full(X.shape[0], np.nan)
        y_pred_full = beta[0] + scores_full @ beta[1:]
        residuals_full[valid_mask] = total_valid - y_pred_full[valid_mask]

        cumulative_variance = np.cumsum(explained_variance_ratio[:n_components])

        return PCAMLRResult(
            source_contributions=source_contributions_full,
            loadings=loadings,
            explained_variance=explained_variance_ratio[:n_components],
            cumulative_variance=cumulative_variance,
            n_components=n_components,
            residuals=residuals_full,
            r_squared=r_squared,
            component_names=self.component_names,
        )

    def _rotate_varimax(self, loadings: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        n_features, n_factors = loadings.shape
        rotation = np.eye(n_factors)

        for _ in range(max_iter):
            loadings_rot = loadings @ rotation
            d = np.diag(loadings_rot.T @ loadings_rot) / n_features
            b = loadings_rot * (loadings_rot ** 2 - d[np.newaxis, :])
            B = loadings.T @ b
            U, S, Vt = np.linalg.svd(B)
            rotation_new = U @ Vt

            if np.allclose(rotation, rotation_new, atol=tol):
                break
            rotation = rotation_new

        return loadings @ rotation

    def solve_with_varimax(self, X: np.ndarray, total_mass: Optional[np.ndarray] = None) -> PCAMLRResult:
        result = self.solve(X, total_mass)

        loadings_rot = self._rotate_varimax(result.loadings)

        n_samples = X.shape[0]
        X_centered = X - np.mean(X, axis=0)

        return PCAMLRResult(
            source_contributions=result.source_contributions,
            loadings=loadings_rot,
            explained_variance=result.explained_variance,
            cumulative_variance=result.cumulative_variance,
            n_components=result.n_components,
            residuals=result.residuals,
            r_squared=result.r_squared,
            component_names=self.component_names,
            source_names=result.source_names,
        )
