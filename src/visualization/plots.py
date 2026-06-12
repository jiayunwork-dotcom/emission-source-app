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

    def scenario_comparison_bar(
        self,
        scenario_names: List[str],
        current_concentration: float,
        expected_concentrations: List[float],
        title: str = "减排情景对比",
        ylabel: str = "PM2.5浓度 (μg/m³)",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(max(10, len(scenario_names) * 2.5), 6))

        n_scenarios = len(scenario_names)
        width = 0.35
        x = np.arange(n_scenarios)

        current_vals = [current_concentration] * n_scenarios

        bars1 = ax.bar(x - width/2, current_vals, width, label='当前浓度',
                       color='#ff7f0e', alpha=0.85)
        bars2 = ax.bar(x + width/2, expected_concentrations, width, label='预期浓度',
                       color='#2ca02c', alpha=0.85)

        for bar, val in zip(bars2, expected_concentrations):
            height = bar.get_height()
            reduction = current_concentration - val
            if reduction > 0:
                ax.annotate(f'-{reduction:.1f}\n(-{reduction/current_concentration*100:.1f}%)',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9, color='red', fontweight='bold')

        ax.set_xlabel('减排情景', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=30, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=35, color='red', linestyle='--', alpha=0.7, linewidth=1, label='国家二级标准(35μg/m³)')
        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def emission_pie_chart(
        self,
        industry_names: List[str],
        emissions: List[float],
        title: str = "各行业PM2.5排放占比",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = COLORS[:len(industry_names)]

        total = sum(emissions)
        if total > 0:
            percentages = [e / total * 100 for e in emissions]
        else:
            percentages = [0] * len(emissions)

        wedges, texts, autotexts = ax.pie(
            emissions,
            labels=industry_names,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 10},
            explode=[0.05] * len(industry_names),
        )

        for at, pct in zip(autotexts, percentages):
            if pct < 3:
                at.set_visible(False)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('equal')

        legend_labels = [f"{name}: {emission:.2f}吨/年" for name, emission in zip(industry_names, emissions)]
        ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)

        return self._fig_to_bytes(fig)

    def validation_gauge_chart(
        self,
        industry_names: List[str],
        deviation_rates: List[float],
        statuses: List[str],
        title: str = "清单质量平衡校验",
    ) -> BytesIO:
        n_industries = len(industry_names)
        n_cols = min(3, n_industries)
        n_rows = (n_industries + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_industries == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, (name, dev, status) in enumerate(zip(industry_names, deviation_rates, statuses)):
            ax = axes[i]

            if status == 'green':
                color = '#2ca02c'
            elif status == 'yellow':
                color = '#ff7f0e'
            else:
                color = '#d62728'

            angles = np.linspace(0, np.pi, 100)
            x_circle = np.cos(angles)
            y_circle = np.sin(angles)

            ax.plot(x_circle, y_circle, 'gray', linewidth=1, alpha=0.3)

            warning_range = 0.3
            danger_range = 0.5

            angle_green_start = np.pi * (0.5 + warning_range)
            angle_green_end = np.pi * (0.5 - warning_range)
            angles_green = np.linspace(angle_green_start, angle_green_end, 50)
            ax.fill_between(np.cos(angles_green), np.sin(angles_green), alpha=0.2, color='#2ca02c')

            angle_yellow1_start = np.pi
            angle_yellow1_end = np.pi * (0.5 + warning_range)
            angles_yellow1 = np.linspace(angle_yellow1_start, angle_yellow1_end, 30)
            ax.fill_between(np.cos(angles_yellow1), np.sin(angles_yellow1), alpha=0.3, color='#ff7f0e')

            angle_yellow2_start = np.pi * (0.5 - warning_range)
            angle_yellow2_end = 0
            angles_yellow2 = np.linspace(angle_yellow2_start, angle_yellow2_end, 30)
            ax.fill_between(np.cos(angles_yellow2), np.sin(angles_yellow2), alpha=0.3, color='#ff7f0e')

            normalized_dev = np.clip(dev / 100.0, -1, 1)
            needle_angle = np.pi * (0.5 - normalized_dev * 0.5)

            ax.plot([0, 0.85 * np.cos(needle_angle)], [0, 0.85 * np.sin(needle_angle)],
                    color=color, linewidth=3, zorder=5)
            ax.scatter([0.85 * np.cos(needle_angle)], [0.85 * np.sin(needle_angle)],
                       color=color, s=100, zorder=6)

            ax.set_ylim(-0.1, 1.1)
            ax.set_xlim(-1.1, 1.1)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_xticklabels(['-100%', '-50%', '0%', '50%', '100%'])
            ax.set_yticks([])
            ax.set_title(f"{name}\n偏差: {dev:.1f}%", fontsize=11, fontweight='bold', color=color)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.axhline(y=0, color='gray', alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        return self._fig_to_bytes(fig)

    def emission_bar_chart(
        self,
        industry_names: List[str],
        emissions: List[float],
        title: str = "各行业PM2.5排放量",
        ylabel: str = "排放量 (吨/年)",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(max(10, len(industry_names) * 1.5), 6))
        colors = COLORS[:len(industry_names)]

        bars = ax.bar(industry_names, emissions, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

        for bar, val in zip(bars, emissions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel('行业', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticklabels(industry_names, rotation=30, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        total = sum(emissions)
        ax.text(0.98, 0.98, f'总排放量: {total:.2f} 吨/年',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def emission_factor_curve(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        x_label: str,
        y_label: str,
        title: str = "排放因子曲线",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(x_values, y_values, 'b-', linewidth=2, label='排放因子')
        ax.fill_between(x_values, y_values, alpha=0.3, color='#1f77b4')

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def reduction_measures_bar(
        self,
        scenario_names: List[str],
        industry_reductions: Dict[str, List[float]],
        title: str = "各行业减排贡献",
        ylabel: str = "减排量 (吨/年)",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(max(12, len(scenario_names) * 2.5), 6))

        n_scenarios = len(scenario_names)
        n_industries = len(industry_reductions)
        width = 0.8 / max(1, n_industries)

        industry_names = list(industry_reductions.keys())
        colors = COLORS[:n_industries]

        x = np.arange(n_scenarios)
        bottom = np.zeros(n_scenarios)

        for i, (industry, reductions) in enumerate(industry_reductions.items()):
            ax.bar(x, reductions, width, bottom=bottom, label=industry,
                   color=colors[i], alpha=0.85)
            bottom += np.array(reductions)

        ax.set_xlabel('减排情景', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=30, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def sensitivity_line_chart(
        self,
        perturbation_ratios: List[float],
        sensitivity_data: Dict[str, List[float]],
        title: str = "敏感性分析曲线",
        xlabel: str = "参数变化比例 (%)",
        ylabel: str = "PM2.5浓度变化百分比 (%)",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(10, 7))

        curve_labels = list(sensitivity_data.keys())
        colors = COLORS[:len(curve_labels)]

        for i, (label, change_pcts) in enumerate(sensitivity_data.items()):
            ax.plot(
                perturbation_ratios,
                change_pcts,
                'o-',
                label=label,
                color=colors[i],
                linewidth=2,
                markersize=6,
                alpha=0.9,
            )

        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='基准点(0%)')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        x_ticks = list(range(int(min(perturbation_ratios)), int(max(perturbation_ratios)) + 1, 10))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{t}%' for t in x_ticks])

        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def tornado_chart(
        self,
        param_labels: List[str],
        impact_magnitudes: List[float],
        title: str = "参数敏感性龙卷风图",
        xlabel: str = "PM2.5浓度变化绝对值 (%)",
    ) -> BytesIO:
        sorted_indices = np.argsort(impact_magnitudes)
        sorted_labels = [param_labels[i] for i in sorted_indices]
        sorted_magnitudes = [impact_magnitudes[i] for i in sorted_indices]

        n_params = len(sorted_labels)
        fig_height = max(5, n_params * 0.6 + 2)
        fig, ax = plt.subplots(figsize=(10, fig_height))

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_params))

        y_pos = np.arange(n_params)
        bars = ax.barh(y_pos, sorted_magnitudes, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

        for bar, mag in zip(bars, sorted_magnitudes):
            width = bar.get_width()
            ax.text(
                width + max(sorted_magnitudes) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{mag:.2f}%',
                va='center',
                ha='left',
                fontsize=10,
                fontweight='bold',
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_labels, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        max_val = max(sorted_magnitudes) if sorted_magnitudes else 1
        ax.set_xlim(0, max_val * 1.25)

        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def concentration_heatmap(
        self,
        concentration_field: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        sources: Optional[List[Dict]] = None,
        wind_direction: Optional[float] = None,
        title: str = "PM2.5浓度空间分布",
        cmap: str = 'YlOrRd',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(10, 8))

        if vmax is None:
            vmax = np.percentile(concentration_field, 99)
        if vmin is None:
            vmin = concentration_field.min()

        im = ax.contourf(
            x_grid, y_grid, concentration_field,
            levels=20,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extend='both',
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('PM2.5浓度 (μg/m³)', fontsize=11)

        if sources is not None:
            markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
            colors_src = COLORS[:len(sources)]
            for i, src in enumerate(sources):
                marker = markers[i % len(markers)]
                ax.scatter(
                    src['x'], src['y'],
                    marker=marker,
                    s=150,
                    color=colors_src[i],
                    edgecolors='black',
                    linewidth=1.5,
                    zorder=10,
                    label=src['name'],
                )
                ax.annotate(
                    src['name'],
                    (src['x'], src['y']),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                )
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

        if wind_direction is not None:
            theta_rad = np.radians(wind_direction)
            arrow_x = np.cos(theta_rad) * 3
            arrow_y = np.sin(theta_rad) * 3
            ax.annotate(
                '',
                xy=(arrow_x, arrow_y),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2.5),
            )
            ax.text(
                arrow_x * 1.2, arrow_y * 1.2,
                f'风向 {wind_direction:.0f}°',
                color='blue',
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            )

        ax.set_xlabel('东西向距离 (km)', fontsize=12)
        ax.set_ylabel('南北向距离 (km)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def concentration_heatmap_comparison(
        self,
        conc_field_1: np.ndarray,
        x_grid_1: np.ndarray,
        y_grid_1: np.ndarray,
        conc_field_2: np.ndarray,
        x_grid_2: np.ndarray,
        y_grid_2: np.ndarray,
        title_1: str = "工况1",
        title_2: str = "工况2",
        sources: Optional[List[Dict]] = None,
        cmap: str = 'YlOrRd',
    ) -> BytesIO:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        vmax = max(np.percentile(conc_field_1, 99), np.percentile(conc_field_2, 99))
        vmin = min(conc_field_1.min(), conc_field_2.min())

        for ax, conc, xg, yg, title in zip(
            axes,
            [conc_field_1, conc_field_2],
            [x_grid_1, x_grid_2],
            [y_grid_1, y_grid_2],
            [title_1, title_2],
        ):
            im = ax.contourf(
                xg, yg, conc,
                levels=20,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extend='both',
            )
            ax.set_xlabel('东西向距离 (km)', fontsize=11)
            ax.set_ylabel('南北向距离 (km)', fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle='--')

            if sources is not None:
                markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
                colors_src = COLORS[:len(sources)]
                for i, src in enumerate(sources):
                    marker = markers[i % len(markers)]
                    ax.scatter(
                        src['x'], src['y'],
                        marker=marker,
                        s=100,
                        color=colors_src[i],
                        edgecolors='black',
                        linewidth=1,
                        zorder=10,
                    )

        cbar = fig.colorbar(im, ax=axes, shrink=0.8)
        cbar.set_label('PM2.5浓度 (μg/m³)', fontsize=11)

        fig.suptitle('气象参数对比分析', fontsize=15, fontweight='bold')
        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def centerline_decay_curve(
        self,
        distances: np.ndarray,
        concentrations: np.ndarray,
        source_name: str,
        background_concentration: float = 5.0,
        influence_radius: Optional[float] = None,
        title: Optional[str] = None,
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            distances, concentrations,
            'b-', linewidth=2.5, label='中心线浓度',
        )
        ax.fill_between(
            distances, concentrations,
            alpha=0.3, color='#1f77b4',
        )

        ax.axhline(
            y=background_concentration,
            color='red',
            linestyle='--',
            linewidth=1.5,
            alpha=0.8,
            label=f'背景浓度 ({background_concentration:.1f} μg/m³)',
        )

        if influence_radius is not None and influence_radius < distances.max():
            ax.axvline(
                x=influence_radius,
                color='green',
                linestyle='--',
                linewidth=1.5,
                alpha=0.8,
                label=f'影响半径 ({influence_radius:.1f} km)',
            )
            ax.scatter(
                [influence_radius], [background_concentration],
                color='green', s=100, zorder=5,
                edgecolors='black', linewidth=1,
            )

        ax.set_xlabel('下风向距离 (km)', fontsize=12)
        ax.set_ylabel('PM2.5浓度 (μg/m³)', fontsize=12)
        if title is None:
            title = f'{source_name} - 下风向中心线浓度衰减'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, distances.max())
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def concentration_heatmap_with_receptors(
        self,
        concentration_field: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        sources: Optional[List[Dict]] = None,
        receptor_points: Optional[List[Dict]] = None,
        shutdown_sources: Optional[List[str]] = None,
        wind_direction: Optional[float] = None,
        title: str = "PM2.5浓度空间分布",
        cmap: str = 'YlOrRd',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(10, 8))

        if vmax is None:
            vmax = np.percentile(concentration_field, 99)
        if vmin is None:
            vmin = concentration_field.min()

        im = ax.contourf(
            x_grid, y_grid, concentration_field,
            levels=20,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extend='both',
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('PM2.5浓度 (μg/m³)', fontsize=11)

        if sources is not None:
            markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
            colors_src = COLORS[:len(sources)]
            shutdown_set = set(shutdown_sources or [])
            for i, src in enumerate(sources):
                marker = markers[i % len(markers)]
                if src['name'] in shutdown_set:
                    ax.scatter(
                        src['x'], src['y'],
                        marker='o',
                        s=200,
                        facecolors='none',
                        edgecolors='gray',
                        linewidth=2,
                        linestyle='dashed',
                        zorder=10,
                    )
                    ax.annotate(
                        src['name'] + '(已关停)',
                        (src['x'], src['y']),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=9,
                        fontweight='bold',
                        color='gray',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='gray', linestyle='dashed'),
                    )
                    circle = plt.Circle((src['x'], src['y']), 0.3, fill=False,
                                        edgecolor='gray', linestyle='--', linewidth=1.5)
                    ax.add_patch(circle)
                else:
                    ax.scatter(
                        src['x'], src['y'],
                        marker=marker,
                        s=150,
                        color=colors_src[i],
                        edgecolors='black',
                        linewidth=1.5,
                        zorder=10,
                        label=src['name'],
                    )
                    ax.annotate(
                        src['name'],
                        (src['x'], src['y']),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=9,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    )
            active_sources = [s for s in sources if s['name'] not in shutdown_set]
            if active_sources:
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

        if receptor_points is not None and len(receptor_points) > 0:
            for i, rp in enumerate(receptor_points):
                ax.scatter(
                    rp['x'], rp['y'],
                    marker='*',
                    s=300,
                    color='#FF00FF',
                    edgecolors='black',
                    linewidth=1.5,
                    zorder=15,
                    label=f"受体{i+1}" if i == 0 else None,
                )
                ax.annotate(
                    rp.get('name', f'R{i+1}'),
                    (rp['x'], rp['y']),
                    xytext=(8, -15),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    color='#FF00FF',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#FF00FF'),
                )

        if wind_direction is not None:
            theta_rad = np.radians(wind_direction)
            arrow_x = np.cos(theta_rad) * 3
            arrow_y = np.sin(theta_rad) * 3
            ax.annotate(
                '',
                xy=(arrow_x, arrow_y),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2.5),
            )
            ax.text(
                arrow_x * 1.2, arrow_y * 1.2,
                f'风向 {wind_direction:.0f}°',
                color='blue',
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            )

        ax.set_xlabel('东西向距离 (km)', fontsize=12)
        ax.set_ylabel('南北向距离 (km)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def radar_chart(
        self,
        categories: List[str],
        values_dict: Dict[str, List[float]],
        title: str = "各源贡献构成比例",
    ) -> BytesIO:
        n_cats = len(categories)
        if n_cats < 3:
            n_cats = 3
            categories = categories + [''] * (n_cats - len(categories))

        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]

        n_receptors = len(values_dict)
        fig, axes = plt.subplots(1, max(n_receptors, 1), figsize=(5 * max(n_receptors, 1), 5),
                                  subplot_kw=dict(polar=True))
        if n_receptors == 1:
            axes = [axes]

        receptor_names = list(values_dict.keys())
        for idx, (rp_name, values) in enumerate(values_dict.items()):
            ax = axes[idx]
            vals = list(values[:n_cats])
            while len(vals) < n_cats:
                vals.append(0)
            vals += vals[:1]

            ax.plot(angles, vals, 'o-', linewidth=2, color=COLORS[idx % len(COLORS)])
            ax.fill(angles, vals, alpha=0.25, color=COLORS[idx % len(COLORS)])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=8)
            ax.set_title(rp_name, fontsize=11, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def time_series_max_conc_chart(
        self,
        hours: List[int],
        max_concentrations: List[float],
        current_hour: int,
        title: str = "最大地面浓度时序变化",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(12, 5))

        played_hours = [h for h in hours if h <= current_hour]
        played_concs = [max_concentrations[h] for h in played_hours]
        future_hours = [h for h in hours if h > current_hour]
        future_concs = [max_concentrations[h] for h in future_hours]

        if len(played_hours) > 1:
            ax.plot(played_hours, played_concs, '-o', linewidth=2, markersize=5,
                    label='已播放', color='#1f77b4')
        elif len(played_hours) == 1:
            ax.scatter(played_hours, played_concs, s=60, color='#1f77b4', zorder=5)

        if len(future_hours) > 0:
            all_future_hours = played_hours[-1:] + future_hours if played_hours else future_hours
            all_future_concs = played_concs[-1:] + future_concs if played_concs else future_concs
            ax.plot(all_future_hours, all_future_concs, '--o', linewidth=1.5, markersize=4,
                    label='未播放', color='#aaaaaa', alpha=0.7)

        if current_hour in hours:
            idx = hours.index(current_hour)
            ax.scatter([current_hour], [max_concentrations[idx]], s=120, color='red',
                       zorder=10, edgecolors='black', linewidth=1.5)
            ax.annotate(f'{max_concentrations[idx]:.1f}',
                        (current_hour, max_concentrations[idx]),
                        xytext=(5, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold', color='red')

        ax.set_xlabel('时刻 (时)', fontsize=12)
        ax.set_ylabel('最大地面浓度 (μg/m³)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(hours)
        ax.set_xticklabels([f'{h}:00' for h in hours], fontsize=8, rotation=45)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def weighted_contribution_change_bar(
        self,
        source_names: List[str],
        original_pcts: List[float],
        weighted_pcts: List[float],
        title: str = "加权后各源贡献百分比变化",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(source_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, original_pcts, width, label='原始(100%)', color='#1f77b4', alpha=0.85)
        bars2 = ax.bar(x + width/2, weighted_pcts, width, label='加权后', color='#ff7f0e', alpha=0.85)

        for i, (orig, weighted) in enumerate(zip(original_pcts, weighted_pcts)):
            change = weighted - orig
            if abs(change) > 0.01:
                color = '#d62728' if change > 0 else '#2ca02c'
                sign = '+' if change > 0 else ''
                ax.text(i + width/2, weighted + 0.3, f'{sign}{change:.1f}%',
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)

        ax.set_xlabel('排放源', fontsize=12)
        ax.set_ylabel('区域平均浓度贡献 (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(source_names, rotation=30, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def source_contribution_bar(
        self,
        source_names: List[str],
        contributions: List[float],
        title: str = "各源最大浓度点贡献占比",
    ) -> BytesIO:
        fig, ax = plt.subplots(figsize=(10, 6))

        total = sum(contributions)
        if total > 0:
            percentages = [c / total * 100 for c in contributions]
        else:
            percentages = [0] * len(contributions)

        colors = COLORS[:len(source_names)]
        bars = ax.bar(source_names, percentages, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

        for bar, pct, contrib in zip(bars, percentages, contributions):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{pct:.1f}%\n({contrib:.2f} μg/m³)',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
            )

        ax.set_xlabel('排放源', fontsize=12)
        ax.set_ylabel('贡献占比 (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticklabels(source_names, rotation=30, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return self._fig_to_bytes(fig)
