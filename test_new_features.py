import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.generate_sample_data import generate_sample_data
from src.data.quality import DataQualityChecker, calculate_uncertainty, get_detection_limits
from src.sources.library import SourceSpectrumLibrary
from src.algorithms.cmb import CMBSolver
from src.algorithms.pmf import PMFSolver
from src.algorithms.pca_mlr import PCAMLRSolver
from src.visualization.plots import Visualizer

print("=" * 60)
print("测试新功能模块")
print("=" * 60)

print("\n1. 生成测试数据...")
data_dir = "data"
generate_sample_data(n_samples=80, n_stations=2, output_dir=data_dir)

dfs = []
for fname in os.listdir(data_dir):
    if fname.endswith('_pm25_components.csv'):
        filepath = os.path.join(data_dir, fname)
        dfs.append(pd.read_csv(filepath))

df = pd.concat(dfs, ignore_index=True)
df['time'] = pd.to_datetime(df['time'])
print(f"   ✓ 数据加载完成: {len(df)} 条记录")

component_cols = [col for col in df.columns if col not in ['time', 'station', 'season']]
print(f"   ✓ 组分列表: {len(component_cols)} 个组分")

print("\n2. 测试数据异常溯源功能...")
checker = DataQualityChecker(outlier_threshold=5.0)
cleaned_df, qc_report = checker.check(df, component_cols)
print(f"   ✓ 质检完成")
print(f"   ✓ 异常高值总数: {sum(qc_report['outliers'].values())}")

original_df = df.copy()
total_outliers = sum(qc_report['outliers'].values())
if total_outliers > 0:
    anomaly_results = []
    for comp in component_cols:
        outlier_count = qc_report['outliers'].get(comp, 0)
        if outlier_count > 0:
            valid_data = original_df[comp].dropna()
            mean_val = valid_data.mean()
            outlier_mask = original_df[comp] > mean_val * 5.0
            outlier_indices = original_df[outlier_mask].index.tolist()
            
            for idx in outlier_indices[:3]:
                row = original_df.loc[idx]
                time_val = row.get('time', 'N/A')
                station_val = row.get('station', 'N/A')
                
                other_cols = [c for c in component_cols if c != comp]
                
                if 'station' in original_df.columns:
                    same_station_mask = original_df['station'] == station_val
                    if 'time' in original_df.columns:
                        time_window = pd.Timedelta(hours=2)
                        row_time = pd.to_datetime(row['time'])
                        time_mask = (pd.to_datetime(original_df['time']) >= row_time - time_window) & \
                                   (pd.to_datetime(original_df['time']) <= row_time + time_window)
                        segment_mask = same_station_mask & time_mask
                    else:
                        segment_mask = same_station_mask
                else:
                    segment_mask = pd.Series([True] * len(original_df))
                
                segment_data = original_df[segment_mask]
                
                correlations = {}
                for other_comp in other_cols:
                    if segment_data[[comp, other_comp]].notna().all(axis=1).sum() >= 3:
                        corr = segment_data[comp].corr(segment_data[other_comp])
                        if not pd.isna(corr):
                            correlations[other_comp] = corr
                
                sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                top3 = sorted_corrs[:3]
                
                anomaly_results.append({
                    'component': comp,
                    'time': time_val,
                    'station': station_val,
                    'value': row[comp],
                    'mean': mean_val,
                    'ratio': row[comp] / mean_val if mean_val > 0 else 0,
                    'top_correlations': top3,
                })
    
    print(f"   ✓ 找到 {len(anomaly_results)} 个异常点的关联分析")
    if anomaly_results:
        for a in anomaly_results[:2]:
            print(f"     - {a['component']} at {a['time']}: 关联组分 = {[c[0] for c in a['top_correlations']]}")

print("\n3. 测试源谱灵敏度分析...")
library = SourceSpectrumLibrary()
source_names = library.get_all_names()
selected_sources = source_names[:4]

detection_limits = get_detection_limits(component_cols)
uncertainty_matrix = np.zeros((len(cleaned_df), len(component_cols)))
for i, col in enumerate(component_cols):
    dl = detection_limits.get(col, 0.01)
    concentrations = cleaned_df[col].values
    uncertainty_matrix[:, i] = calculate_uncertainty(concentrations, dl, relative_uncertainty=0.1)

X = cleaned_df[component_cols].values
U = uncertainty_matrix
valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(U).any(axis=1) & (U > 0).all(axis=1)
X_valid = X[valid_mask]
U_valid = U[valid_mask]

source_matrix = library.get_source_matrix(selected_sources, component_cols)
source_uncert = library.get_uncertainty_matrix(selected_sources, component_cols)

base_solver = CMBSolver(
    source_names=selected_sources,
    component_names=component_cols,
    source_matrix=source_matrix,
    source_uncertainty_matrix=source_uncert,
)
base_result = base_solver.solve(X_valid, U_valid)
valid_contrib = base_result.source_contributions[~np.isnan(base_result.source_contributions[:, 0])]
base_avg_contrib = np.mean(valid_contrib, axis=0)
print(f"   ✓ 基础CMB解析完成")

n_components = len(component_cols)
n_sources = len(selected_sources)
sensitivity_matrix = np.zeros((n_components, n_sources))
perturbation_percent = 10.0

for i, comp in enumerate(component_cols):
    for direction in [1, -1]:
        perturbed_matrix = source_matrix.copy()
        perturbation = perturbation_percent / 100.0 * direction
        for j in range(n_sources):
            if perturbed_matrix[i, j] > 0:
                perturbed_matrix[i, j] *= (1 + perturbation)

        perturbed_solver = CMBSolver(
            source_names=selected_sources,
            component_names=component_cols,
            source_matrix=perturbed_matrix,
            source_uncertainty_matrix=source_uncert,
        )
        perturbed_result = perturbed_solver.solve(X_valid, U_valid)
        valid_perturbed = perturbed_result.source_contributions[~np.isnan(perturbed_result.source_contributions[:, 0])]
        
        if len(valid_perturbed) > 0:
            perturbed_avg = np.mean(valid_perturbed, axis=0)
            for k in range(n_sources):
                if base_avg_contrib[k] > 0:
                    change_pct = (perturbed_avg[k] - base_avg_contrib[k]) / base_avg_contrib[k] * 100
                    sensitivity_matrix[i, k] = max(sensitivity_matrix[i, k], abs(change_pct))

high_sensitivity = (np.abs(sensitivity_matrix) > 15).sum()
print(f"   ✓ 灵敏度分析完成, 高敏点 (>15%): {high_sensitivity}")

print("\n4. 测试多算法交叉验证...")
algorithms = ["PMF", "CMB", "PCA-MLR"]
results = {}
source_names_map = {}

for algo in algorithms[:2]:
    if algo == "PMF":
        solver = PMFSolver(
            component_names=component_cols,
            n_factors=4,
            max_iterations=500,
            random_seed=42,
        )
        result = solver.solve(X_valid, U_valid)
        avg_contrib = np.mean(result.G, axis=0)
        results[algo] = {'avg_contrib': avg_contrib, 'source_names': result.source_names}
        source_names_map[algo] = result.source_names
        print(f"   ✓ PMF完成, 因子数: {result.n_factors}")
    
    elif algo == "CMB":
        solver = CMBSolver(
            source_names=selected_sources,
            component_names=component_cols,
            source_matrix=source_matrix,
            source_uncertainty_matrix=source_uncert,
        )
        result = solver.solve(X_valid, U_valid)
        valid_contrib = result.source_contributions[~np.isnan(result.source_contributions[:, 0])]
        avg_contrib = np.mean(valid_contrib, axis=0)
        results[algo] = {'avg_contrib': avg_contrib, 'source_names': result.source_names}
        source_names_map[algo] = result.source_names
        print(f"   ✓ CMB完成, 源类数: {len(selected_sources)}")
    
    elif algo == "PCA-MLR":
        total_mass = np.sum(X_valid, axis=1)
        solver = PCAMLRSolver(
            component_names=component_cols,
            variance_threshold=0.8,
        )
        result = solver.solve(X_valid, total_mass)
        avg_contrib = np.mean(np.abs(result.source_contributions), axis=0)
        results[algo] = {'avg_contrib': avg_contrib, 'source_names': result.source_names}
        source_names_map[algo] = result.source_names
        print(f"   ✓ PCA-MLR完成, 主成分数: {result.n_components}")

if len(results) >= 2:
    algo_a, algo_b = list(results.keys())[:2]
    common_ab = list(set(source_names_map[algo_a]) & set(source_names_map[algo_b]))
    if len(common_ab) >= 2:
        contribs_a = []
        contribs_b = []
        for src in common_ab:
            idx_a = source_names_map[algo_a].index(src)
            idx_b = source_names_map[algo_b].index(src)
            contribs_a.append(results[algo_a]['avg_contrib'][idx_a])
            contribs_b.append(results[algo_b]['avg_contrib'][idx_b])
        contribs_a = np.array(contribs_a)
        contribs_b = np.array(contribs_b)
        
        total_a = np.sum(contribs_a)
        total_b = np.sum(contribs_b)
        pct_a = contribs_a / total_a * 100 if total_a > 0 else contribs_a
        pct_b = contribs_b / total_b * 100 if total_b > 0 else contribs_b
        
        corr = np.corrcoef(pct_a, pct_b)[0, 1]
        rmse = np.sqrt(np.mean((pct_a - pct_b) ** 2))
        print(f"   ✓ {algo_a} vs {algo_b}: 相关系数={corr:.4f}, RMSE={rmse:.2f}%")

print("\n5. 测试趋势预警功能...")
if 'time' in df.columns:
    solver = PMFSolver(
        component_names=component_cols,
        n_factors=4,
        max_iterations=500,
        random_seed=42,
    )
    result = solver.solve(X_valid, U_valid)
    
    G_full = np.full((len(df), result.n_factors), np.nan)
    G_full[valid_mask] = result.G
    
    times = pd.to_datetime(df['time'])
    daily_contrib = pd.DataFrame(G_full, columns=result.source_names)
    daily_contrib['date'] = times.dt.date
    daily_avg = daily_contrib.groupby('date').mean()
    
    consecutive_days = 3
    growth_threshold = 0.2
    
    alerts = []
    for source in result.source_names:
        series = daily_avg[source].dropna()
        if len(series) >= consecutive_days + 1:
            for i in range(consecutive_days - 1, len(series)):
                consecutive_growth = True
                growth_rates = []
                for j in range(i - consecutive_days + 1, i + 1):
                    if j > 0 and series.iloc[j-1] > 0:
                        growth_rate = (series.iloc[j] - series.iloc[j-1]) / series.iloc[j-1]
                        growth_rates.append(growth_rate)
                        if growth_rate < growth_threshold:
                            consecutive_growth = False
                            break
                    elif j > 0:
                        consecutive_growth = False
                        break
                
                if consecutive_growth and len(growth_rates) == consecutive_days:
                    avg_growth = np.mean(growth_rates) * 100
                    alert_date = series.index[i]
                    alerts.append({
                        'time': str(alert_date),
                        'source': source,
                        'growth': avg_growth,
                        'consecutive_days': consecutive_days,
                    })
    
    print(f"   ✓ 预警检测完成, 触发预警数: {len(alerts)}")
    if alerts:
        for alert in alerts[:3]:
            print(f"     - {alert['source']} at {alert['time']}: 平均增幅 {alert['growth']:.1f}%")

print("\n6. 测试可视化功能...")
viz = Visualizer()

try:
    img1 = viz.sensitivity_heatmap(
        component_cols[:10],
        selected_sources,
        sensitivity_matrix[:10, :],
        title="测试灵敏度热力图",
    )
    print("   ✓ 灵敏度热力图生成成功")
except Exception as e:
    print(f"   ✗ 灵敏度热力图失败: {e}")

try:
    if len(results) >= 2:
        algo_results = {k: v['avg_contrib'] for k, v in results.items()}
        first_algo = list(results.keys())[0]
        img2 = viz.algorithm_comparison_bar(
            results[first_algo]['source_names'],
            algo_results,
            title="测试算法对比柱状图",
        )
        print("   ✓ 算法对比柱状图生成成功")
except Exception as e:
    print(f"   ✗ 算法对比柱状图失败: {e}")

try:
    if len(results) >= 2:
        algo_a, algo_b = list(results.keys())[:2]
        common_ab = list(set(source_names_map[algo_a]) & set(source_names_map[algo_b]))
        if len(common_ab) > 0:
            contribs_a = np.array([results[algo_a]['avg_contrib'][source_names_map[algo_a].index(s)] for s in common_ab])
            contribs_b = np.array([results[algo_b]['avg_contrib'][source_names_map[algo_b].index(s)] for s in common_ab])
            img3 = viz.algorithm_scatter_comparison(
                contribs_a,
                contribs_b,
                common_ab,
                label_a=algo_a,
                label_b=algo_b,
            )
            print("   ✓ 算法散点对比图生成成功")
except Exception as e:
    print(f"   ✗ 算法散点对比图失败: {e}")

try:
    if 'time' in df.columns:
        img4 = viz.trend_alert_plot(
            times[:50],
            G_full[:50, :],
            result.source_names,
            alert_indices=[10, 20],
            alert_sources=[result.source_names[0]],
        )
        print("   ✓ 趋势预警图生成成功")
except Exception as e:
    print(f"   ✗ 趋势预警图失败: {e}")

try:
    if anomaly_results:
        a = anomaly_results[0]
        if a['top_correlations']:
            comp_names = [c[0] for c in a['top_correlations']]
            corr_values = [c[1] for c in a['top_correlations']]
            img5 = viz.anomaly_correlation_bars(
                a['component'],
                comp_names,
                corr_values,
            )
            print("   ✓ 异常关联分析图生成成功")
except Exception as e:
    print(f"   ✗ 异常关联分析图失败: {e}")

print("\n" + "=" * 60)
print("所有新功能测试完成!")
print("=" * 60)
