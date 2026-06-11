import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings


class PMFResult:
    def __init__(
        self,
        G: np.ndarray,
        F: np.ndarray,
        residuals: np.ndarray,
        Q: float,
        Q_expected: float,
        iterations: int,
        converged: bool,
        n_factors: int,
        component_names: List[str],
        source_names: Optional[List[str]] = None,
        bootstrap_results: Optional[Dict] = None,
        disp_results: Optional[Dict] = None,
    ):
        self.G = G
        self.F = F
        self.residuals = residuals
        self.Q = Q
        self.Q_expected = Q_expected
        self.Q_ratio = Q / Q_expected if Q_expected > 0 else float('inf')
        self.iterations = iterations
        self.converged = converged
        self.n_factors = n_factors
        self.component_names = component_names
        self.source_names = source_names if source_names else [f'Factor {i+1}' for i in range(n_factors)]
        self.bootstrap_results = bootstrap_results
        self.disp_results = disp_results

    def get_contribution_dataframe(self) -> Dict:
        avg_contrib = np.mean(self.G, axis=0)
        total = np.sum(avg_contrib)
        percentages = (avg_contrib / total * 100) if total > 0 else np.zeros_like(avg_contrib)
        return {
            'source': self.source_names,
            'contribution': avg_contrib,
            'percentage': percentages,
        }

    def get_factor_profiles_dataframe(self) -> Dict:
        profiles = []
        for i in range(self.n_factors):
            profiles.append(self.F[i, :])
        return {
            'components': self.component_names,
            'factors': self.source_names,
            'profiles': np.array(profiles).T,
        }


class PMFSolver:
    def __init__(
        self,
        component_names: List[str],
        n_factors: int,
        max_iterations: int = 500,
        convergence_tolerance: float = 1e-4,
        random_seed: Optional[int] = None,
    ):
        self.component_names = component_names
        self.n_factors = n_factors
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def _calculate_Q(self, X: np.ndarray, G: np.ndarray, F: np.ndarray, U: np.ndarray) -> float:
        residuals = X - G @ F
        weighted = residuals / U
        return float(np.sum(weighted ** 2))

    def _calculate_Q_expected(self, X: np.ndarray) -> float:
        n_samples, n_components = X.shape
        n_data_points = n_samples * n_components
        n_params = n_samples * self.n_factors + self.n_factors * n_components
        dof = max(n_data_points - n_params, 1)
        return float(dof)

    def _nnls(self, A: np.ndarray, b: np.ndarray, max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
        n, k = A.shape
        x = np.zeros(k)
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        x = np.maximum(x, 0)
        for _ in range(max_iter):
            residual = A @ x - b
            gradient = A.T @ residual
            inactive = x > 0
            if np.all(x == 0):
                idx = np.argmin(gradient)
                x[idx] = 1e-10
                continue
            if not np.any(inactive):
                break
            step = np.zeros_like(x)
            step[inactive] = np.linalg.lstsq(A[:, inactive], b, rcond=None)[0]
            step = np.maximum(step, 0) - x
            if np.all(np.abs(step) < tol):
                break
            alpha = 1.0
            while alpha > 1e-10:
                x_new = x + alpha * step
                if np.all(x_new >= -1e-10):
                    x = np.maximum(x_new, 0)
                    break
                alpha *= 0.5
        return x

    def _update_G(self, X: np.ndarray, G: np.ndarray, F: np.ndarray, U: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        F_T = F.T
        for i in range(n_samples):
            u_col = U[i, :]
            A = F_T / u_col[:, np.newaxis]
            b = X[i, :] / u_col
            G[i, :] = self._nnls(A, b)
        return G

    def _update_F(self, X: np.ndarray, G: np.ndarray, F: np.ndarray, U: np.ndarray) -> np.ndarray:
        n_components = X.shape[1]
        for j in range(n_components):
            A = G / U[:, j, np.newaxis]
            b = X[:, j] / U[:, j]
            F[:, j] = self._nnls(A, b)
        return F

    def _initialize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples, n_components = X.shape
        avg_conc = np.mean(X, axis=0)
        F = np.random.rand(self.n_factors, n_components) * avg_conc[np.newaxis, :] * 0.5
        F = F + 0.01 * np.mean(X)
        G = np.random.rand(n_samples, self.n_factors) * np.mean(X) / self.n_factors
        G = G + 0.01 * np.mean(X)
        return G, F

    def solve(self, X: np.ndarray, U: np.ndarray) -> PMFResult:
        n_samples, n_components = X.shape

        Q_expected = self._calculate_Q_expected(X)
        G, F = self._initialize(X)

        Q_prev = self._calculate_Q(X, G, F, U)
        converged = False
        iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            G = self._update_G(X, G, F, U)
            F = self._update_F(X, G, F, U)
            Q_current = self._calculate_Q(X, G, F, U)
            delta_Q = abs(Q_prev - Q_current) / Q_prev if Q_prev > 0 else 1.0
            if delta_Q < self.convergence_tolerance:
                converged = True
                break
            Q_prev = Q_current

        residuals = X - G @ F
        return PMFResult(
            G=G,
            F=F,
            residuals=residuals,
            Q=Q_prev,
            Q_expected=Q_expected,
            iterations=iteration,
            converged=converged,
            n_factors=self.n_factors,
            component_names=self.component_names,
        )

    def bootstrap(
        self,
        X: np.ndarray,
        U: np.ndarray,
        n_bootstrap: int = 100,
        block_size: int = 7,
    ) -> Dict:
        n_samples = X.shape[0]
        n_blocks = n_samples // block_size

        base_result = self.solve(X, U)
        base_G = base_result.G.copy()
        base_F = base_result.F.copy()

        bootstrap_G = []
        bootstrap_F = []
        successes = 0

        for _ in range(n_bootstrap):
            block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
            sample_indices = []
            for b in block_indices:
                start = b * block_size
                end = min(start + block_size, n_samples)
                sample_indices.extend(range(start, end))
            sample_indices = np.array(sample_indices[:n_samples])

            X_boot = X[sample_indices, :]
            U_boot = U[sample_indices, :]

            try:
                boot_result = self.solve(X_boot, U_boot)
                boot_G, boot_F = self._match_factors(base_G, base_F, boot_result.G, boot_result.F)
                bootstrap_G.append(boot_G)
                bootstrap_F.append(boot_F)
                successes += 1
            except Exception:
                pass

        if successes == 0:
            return {
                'n_bootstrap': n_bootstrap,
                'successes': 0,
                'stability_ratio': 0.0,
                'G_std': np.zeros_like(base_G),
                'F_std': np.zeros_like(base_F),
            }

        bootstrap_G = np.array(bootstrap_G)
        bootstrap_F = np.array(bootstrap_F)

        G_std = np.std(bootstrap_G, axis=0)
        F_std = np.std(bootstrap_F, axis=0)

        stability_ratio = successes / n_bootstrap

        return {
            'n_bootstrap': n_bootstrap,
            'successes': successes,
            'stability_ratio': stability_ratio,
            'G_std': G_std,
            'F_std': F_std,
            'bootstrap_G': bootstrap_G,
            'bootstrap_F': bootstrap_F,
        }

    def _match_factors(
        self,
        base_G: np.ndarray,
        base_F: np.ndarray,
        boot_G: np.ndarray,
        boot_F: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_factors = self.n_factors
        correlation_matrix = np.zeros((n_factors, n_factors))

        for i in range(n_factors):
            for j in range(n_factors):
                base_profile = base_F[i, :]
                boot_profile = boot_F[j, :]
                if np.std(base_profile) > 0 and np.std(boot_profile) > 0:
                    corr = np.corrcoef(base_profile, boot_profile)[0, 1]
                    correlation_matrix[i, j] = abs(corr)
                else:
                    correlation_matrix[i, j] = 0.0

        best_match = np.full(n_factors, -1, dtype=int)
        used = set()

        for i in range(n_factors):
            best_corr = -1
            best_j = -1
            for j in range(n_factors):
                if j not in used and correlation_matrix[i, j] > best_corr:
                    best_corr = correlation_matrix[i, j]
                    best_j = j
            if best_j >= 0:
                best_match[i] = best_j
                used.add(best_j)

        matched_G = np.zeros_like(boot_G)
        matched_F = np.zeros_like(boot_F)

        for i in range(n_factors):
            if best_match[i] >= 0:
                matched_G[:, i] = boot_G[:, best_match[i]]
                matched_F[i, :] = boot_F[best_match[i], :]
            else:
                matched_G[:, i] = boot_G[:, i]
                matched_F[i, :] = boot_F[i, :]

        return matched_G, matched_F

    def disp_analysis(
        self,
        X: np.ndarray,
        U: np.ndarray,
        base_result: PMFResult,
        n_rotations: int = 20,
        fpeak_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> Dict:
        n_factors = self.n_factors
        fpeak_values = np.linspace(fpeak_range[0], fpeak_range[1], n_rotations)

        results = []
        base_Q = base_result.Q

        for fpeak in fpeak_values:
            if abs(fpeak) < 1e-6:
                results.append({
                    'fpeak': fpeak,
                    'G': base_result.G.copy(),
                    'F': base_result.F.copy(),
                    'Q': base_Q,
                    'dQ': 0.0,
                })
                continue

            G_rot = base_result.G.copy()
            F_rot = base_result.F.copy()

            for i in range(n_factors):
                for j in range(n_factors):
                    if i != j:
                        G_rot[:, i] += fpeak * G_rot[:, j]
                        F_rot[i, :] -= fpeak * F_rot[j, :]

            G_rot = np.maximum(G_rot, 0)
            F_rot = np.maximum(F_rot, 0)

            Q_rot = self._calculate_Q(X, G_rot, F_rot, U)
            dQ = (Q_rot - base_Q) / base_Q * 100

            results.append({
                'fpeak': fpeak,
                'G': G_rot,
                'F': F_rot,
                'Q': Q_rot,
                'dQ': dQ,
            })

        return {
            'fpeak_values': fpeak_values,
            'results': results,
            'base_Q': base_Q,
        }

    def run_full_analysis(
        self,
        X: np.ndarray,
        U: np.ndarray,
        do_bootstrap: bool = True,
        do_disp: bool = True,
        n_bootstrap: int = 100,
        block_size: int = 7,
    ) -> PMFResult:
        base_result = self.solve(X, U)

        if do_bootstrap:
            base_result.bootstrap_results = self.bootstrap(X, U, n_bootstrap=n_bootstrap, block_size=block_size)

        if do_disp:
            base_result.disp_results = self.disp_analysis(X, U, base_result)

        return base_result


def determine_optimal_factors(
    X: np.ndarray,
    U: np.ndarray,
    component_names: List[str],
    min_factors: int = 2,
    max_factors: int = 10,
    target_Q_ratio: float = 1.0,
) -> Dict:
    results = []
    Q_values = []
    Q_ratios = []

    for n_f in range(min_factors, max_factors + 1):
        solver = PMFSolver(component_names=component_names, n_factors=n_f, random_seed=42)
        result = solver.solve(X, U)
        results.append(result)
        Q_values.append(result.Q)
        Q_ratios.append(result.Q_ratio)

    delta_Q = []
    for i in range(1, len(Q_values)):
        dq = (Q_values[i-1] - Q_values[i]) / Q_values[i-1] * 100
        delta_Q.append(dq)

    return {
        'n_factors_range': list(range(min_factors, max_factors + 1)),
        'Q_values': Q_values,
        'Q_ratios': Q_ratios,
        'delta_Q': delta_Q,
        'results': results,
    }
