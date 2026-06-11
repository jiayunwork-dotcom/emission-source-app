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

    def sensitivity_heatmap(
        self,
        component_names: List[str],
        source_names: List[str],
        sensitivity_matrix: np.ndarray,
        title: str = "源谱灵敏度分析",
        high_threshold: float = 15.0,
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(max(8, len(source_names) * 1.2), max(6, len(component_names) * 0.5)))

        vmax = max(abs(sensitivity_matrix.min()), abs(sensitivity_matrix.max()))
        vmax = max(vmax, high_threshold)

        cmap = plt.cm.RdBu_r
        bounds = np.linspace(-vmax, vmax, 101)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(
            sensitivity_matrix,
            cmap=cmap,
            norm=norm,
            aspect='auto',
            interpolation='nearest',
        )

        for i in range(len(component_names)):
            for j in range(len(source_names)):
                val = sensitivity_matrix[i, j]
                text_color = 'white' if abs(val) > high_threshold / 2 else 'black'
                font_weight = 'bold' if abs(val) > high_threshold else 'normal'
                ax.text(
                    j, i, f'{val:.1f}%',
                    ha='center', va='center',
                    color=text_color,
                    fontweight=font_weight,
                    fontsize=8,
                )

        ax.set_xticks(range(len(source_names)))
        ax.set_xticklabels(source_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(component_names)))
        ax.set_yticklabels(component_names, fontsize=9)
        ax.set_xlabel('源类', fontsize=11)
        ax.set_ylabel('组分', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('贡献变化百分比 (%)', fontsize=10)

        high_mask = np.abs(sensitivity_matrix) > high_threshold
        if np.any(high_mask):
            for i, j in zip(*np.where(high_mask)):
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor='black', linewidth=2,
                )
                ax.add_patch(rect)

        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def algorithm_comparison_bar(
        self,
        source_names: List[str],
        algorithm_results: Dict[str, np.ndarray],
        title: str = "多算法源贡献对比",
        ylabel: str = "贡献占比 (%)",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(12, 6))

        n_sources = len(source_names)
        n_algorithms = len(algorithm_results)
        width = 0.8 / n_algorithms

        algo_names = list(algorithm_results.keys())
        colors = COLORS[:n_algorithms]

        x = np.arange(n_sources)

        for i, (name, contribs) in enumerate(algorithm_results.items()):
            total = np.sum(contribs)
            percentages = contribs / total * 100 if total > 0 else contribs
            offset = (i - n_algorithms / 2 + 0.5) * width
            ax.bar(x + offset, percentages, width, label=name, color=colors[i], alpha=0.85)

        ax.set_xlabel('源类', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(source_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def algorithm_scatter_comparison(
        self,
        contribs_a: np.ndarray,
        contribs_b: np.ndarray,
        source_names: List[str],
        label_a: str = "算法A",
        label_b: str = "算法B",
        title: str = "算法结果对比",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(8, 8))

        total_a = np.sum(contribs_a)
        total_b = np.sum(contribs_b)
        pct_a = contribs_a / total_a * 100 if total_a > 0 else contribs_a
        pct_b = contribs_b / total_b * 100 if total_b > 0 else contribs_b

        max_val = max(pct_a.max(), pct_b.max()) * 1.1
        min_val = 0

        colors = COLORS[:len(source_names)]
        for i, (name, pa, pb) in enumerate(zip(source_names, pct_a, pct_b)):
            ax.scatter(pa, pb, s=100, color=colors[i], label=name, alpha=0.8, edgecolors='black', linewidth=0.5)
            ax.annotate(name, (pa, pb), xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='45°线', linewidth=1.5, alpha=0.7)

        ax.set_xlabel(f'{label_a} 贡献占比 (%)', fontsize=12)
        ax.set_ylabel(f'{label_b} 贡献占比 (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def trend_alert_plot(
        self,
        times: pd.DatetimeIndex,
        contributions: np.ndarray,
        source_names: List[str],
        alert_indices: Optional[List[int]] = None,
        alert_sources: Optional[List[str]] = None,
        title: str = "源贡献时间趋势与预警",
        ylabel: str = "浓度 (μg/m³)",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = COLORS[:len(source_names)]

        for i, (name, color) in enumerate(zip(source_names, colors)):
            linewidth = 2
            alpha = 0.8
            if alert_sources and name in alert_sources:
                linewidth = 3
                alpha = 1.0
            ax.plot(times, contributions[:, i], label=name, color=color, linewidth=linewidth, alpha=alpha)

        if alert_indices is not None and len(alert_indices) > 0:
            for idx in alert_indices:
                if idx < len(times):
                    ax.axvline(x=times[idx], color='red', linestyle='--', alpha=0.5, linewidth=1)
                    ax.scatter(times[idx], contributions[idx, :].max(), color='red', s=150, zorder=5, marker='*', edgecolors='black', linewidth=0.5)

        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def anomaly_correlation_bars(
        self,
        component_name: str,
        correlated_components: List[str],
        correlation_values: List[float],
        title: str = "异常组分关联分析",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(8, 5))

        colors = ['#ff6b6b' if abs(r) > 0.7 else '#4ecdc4' if abs(r) > 0.5 else '#45b7d1' for r in correlation_values]

        bars = ax.barh(range(len(correlated_components)), correlation_values, color=colors, alpha=0.85)
        ax.set_yticks(range(len(correlated_components)))
        ax.set_yticklabels(correlated_components, fontsize=10)
        ax.set_xlabel('相关系数', fontsize=11)
        ax.set_title(f'{component_name} - 关联组分', fontsize=13, fontweight='bold')
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='强相关 (|r|>0.7)')
        ax.axvline(x=-0.7, color='red', linestyle='--', alpha=0.5)
        ax.set_xlim(-1.1, 1.1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')

        for bar, val in zip(bars, correlation_values):
            width = bar.get_width()
            ax.text(width + 0.02 if width >= 0 else width - 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}',
                    va='center',
                    ha='left' if width >= 0 else 'right',
                    fontsize=9)

        plt.tight_layout()
        return self._fig_to_bytes(fig)
