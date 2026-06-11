import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Optional, Tuple
from io import BytesIO


COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


class Visualizer:
    def __init__(self, style: str = 'default'):
        self.style = style
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def _fig_to_bytes(self, fig) -> BytesIO:
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    def pie_chart(self, labels: List[str], values: np.ndarray, title: str = "源贡献占比") -> BytesIO:
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = COLORS[:len(labels)]

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 10},
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('equal')

        return self._fig_to_bytes(fig)

    def stacked_area_chart(
        self,
        time_index: pd.DatetimeIndex,
        contributions: np.ndarray,
        source_names: List[str],
        title: str = "源贡献时间序列",
        ylabel: str = "浓度 (μg/m³)",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = COLORS[:len(source_names)]

        ax.stackplot(
            time_index,
            contributions.T,
            labels=source_names,
            colors=colors,
            alpha=0.8,
        )

        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.autofmt_xdate()
        plt.tight_layout()

        return self._fig_to_bytes(fig)

    def factor_profile_bar_chart(
        self,
        component_names: List[str],
        profiles: np.ndarray,
        factor_names: List[str],
        title: str = "因子谱图",
        normalize: bool = True,
    ) -> BytesIO:
        n_factors = len(factor_names)
        n_cols = min(3, n_factors)
        n_rows = (n_factors + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_factors == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, (name, profile) in enumerate(zip(factor_names, profiles)):
            ax = axes[i]

            prof = profile.copy()
            if normalize:
                total = np.sum(profile)
                if total > 0:
                    prof = profile / total * 100

            colors = plt.cm.viridis(np.linspace(0, 1, len(component_names)))
            bars = ax.bar(range(len(component_names)), prof, color=colors)
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.set_ylabel('质量分数 (%)' if normalize else '浓度')
            ax.set_xticks(range(len(component_names)))
            ax.set_xticklabels(component_names, rotation=45, ha='right', fontsize=7)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        return self._fig_to_bytes(fig)

    def residual_scatter_plot(
        self,
        measured: np.ndarray,
        predicted: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        component_names: Optional[List[str]] = None,
        title: str = "残差分析",
    ) -> BytesIO:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        residuals = measured - predicted

        ax1 = axes[0]
        ax1.scatter(measured.flatten(), predicted.flatten(), alpha=0.5, s=20)
        min_val = min(measured.min(), predicted.min())
        max_val = max(measured.max(), predicted.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1线')
        ax1.set_xlabel('实测浓度', fontsize=12)
        ax1.set_ylabel('预测浓度', fontsize=12)
        ax1.set_title('实测 vs 预测', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        if uncertainties is not None and component_names is not None:
            n_components = len(component_names)
            for i in range(n_components):
                res_norm = residuals[:, i] / uncertainties[:, i]
                ax2.scatter(
                    np.full_like(res_norm, i),
                    res_norm,
                    alpha=0.5,
                    s=15,
                    label=component_names[i],
                )
            ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax2.axhline(y=2, color='r', linestyle='--', linewidth=0.8, alpha=0.7)
            ax2.axhline(y=-2, color='r', linestyle='--', linewidth=0.8, alpha=0.7)
            ax2.set_xticks(range(n_components))
            ax2.set_xticklabels(component_names, rotation=45, ha='right', fontsize=8)
            ax2.set_ylabel('归一化残差 (残差/不确定度)', fontsize=12)
        else:
            ax2.hist(residuals.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax2.axvline(x=0, color='r', linestyle='--', label='零残差')
            ax2.set_xlabel('残差', fontsize=12)
            ax2.set_ylabel('频数', fontsize=12)
            ax2.legend()

        ax2.set_title('残差分布', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        return self._fig_to_bytes(fig)

    def bootstrap_boxplot(
        self,
        bootstrap_contributions: np.ndarray,
        source_names: List[str],
        title: str = "Bootstrap稳定性分析",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(10, 6))

        n_bootstrap = bootstrap_contributions.shape[0]
        avg_contribs = np.mean(bootstrap_contributions, axis=1)

        bp = ax.boxplot(avg_contribs, labels=source_names, patch_artist=True)

        colors = COLORS[:len(source_names)]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('平均贡献浓度', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticklabels(source_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        return self._fig_to_bytes(fig)

    def heatmap_2d(
        self,
        values: np.ndarray,
        lat_edges: np.ndarray,
        lon_edges: np.ndarray,
        title: str = "潜在源区分布",
        cmap: str = 'YlOrRd',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(10, 8))

        lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
        lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

        im = ax.pcolormesh(
            lon_centers, lat_centers, values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading='auto',
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('PSCF值' if 'PSCF' in title else 'CWT值', fontsize=11)

        ax.set_xlabel('经度 (°)', fontsize=12)
        ax.set_ylabel('纬度 (°)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return self._fig_to_bytes(fig)

    def grouped_bar_chart(
        self,
        categories: List[str],
        values: Dict[str, np.ndarray],
        title: str = "多站点源贡献对比",
        ylabel: str = "贡献占比 (%)",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(12, 6))

        n_categories = len(categories)
        n_groups = len(values)
        width = 0.8 / n_groups

        group_names = list(values.keys())
        colors = COLORS[:n_groups]

        x = np.arange(n_categories)

        for i, (name, vals) in enumerate(values.items()):
            offset = (i - n_groups / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=name, color=colors[i], alpha=0.85)

        ax.set_xlabel('源类', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        return self._fig_to_bytes(fig)

    def season_bar_chart(
        self,
        seasons: List[str],
        contributions: Dict[str, np.ndarray],
        title: str = "季节源贡献统计",
        ylabel: str = "贡献占比 (%)",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(10, 6))

        source_names = list(contributions.keys())
        n_sources = len(source_names)
        n_seasons = len(seasons)

        data = np.zeros((n_seasons, n_sources))
        for i, source in enumerate(source_names):
            data[:, i] = contributions[source]

        bottom = np.zeros(n_seasons)
        colors = COLORS[:n_sources]

        for i in range(n_sources):
            ax.bar(
                range(n_seasons),
                data[:, i],
                bottom=bottom,
                label=source_names[i],
                color=colors[i],
                alpha=0.85,
            )
            bottom += data[:, i]

        ax.set_xlabel('季节', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(n_seasons))
        ax.set_xticklabels(seasons)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        return self._fig_to_bytes(fig)

    def q_convergence_plot(
        self,
        iterations: List[int],
        q_values: List[float],
        title: str = "PMF收敛曲线",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(iterations, q_values, 'b-', linewidth=2, label='Q值')
        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel('Q值', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()

        return self._fig_to_bytes(fig)

    def source_spectrum_comparison(
        self,
        component_names: List[str],
        spectra: Dict[str, np.ndarray],
        title: str = "源谱对比",
        normalize: bool = True,
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(14, 6))

        n_components = len(component_names)
        n_sources = len(spectra)
        width = 0.8 / n_sources

        source_names = list(spectra.keys())
        colors = COLORS[:n_sources]

        x = np.arange(n_components)

        for i, (name, spec) in enumerate(spectra.items()):
            s = spec.copy()
            if normalize:
                total = np.sum(spec)
                if total > 0:
                    s = spec / total * 100
            offset = (i - n_sources / 2 + 0.5) * width
            ax.bar(x + offset, s, width, label=name, color=colors[i], alpha=0.85)

        ax.set_xlabel('化学组分', fontsize=12)
        ax.set_ylabel('质量分数 (%)' if normalize else '质量分数', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(component_names, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        return self._fig_to_bytes(fig)
