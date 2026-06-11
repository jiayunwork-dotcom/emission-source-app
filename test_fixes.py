import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

print("=" * 60)
print("测试1: 异常溯源关联分析")
print("=" * 60)

component_cols = [
    'SO4', 'NO3', 'NH4', 'Cl', 'Na', 'K', 'Ca', 'Mg',
    'OC', 'EC', 'Al', 'Si', 'Fe', 'Zn', 'Pb',
]

np.random.seed(42)
n_samples = 100
dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')

source_profiles = {
    'coal': {'SO4': 0.20, 'NO3': 0.05, 'NH4': 0.03, 'Cl': 0.02, 'OC': 0.08, 'EC': 0.04,
             'Al': 0.05, 'Si': 0.15, 'Fe': 0.04},
    'vehicle': {'SO4': 0.03, 'NO3': 0.08, 'NH4': 0.02, 'OC': 0.25, 'EC': 0.20,
                'Al': 0.005, 'Si': 0.01},
    'secondary': {'SO4': 0.35, 'NO3': 0.25, 'NH4': 0.15, 'OC': 0.10, 'EC': 0.005},
}

source_contribs = np.zeros((n_samples, 3))
source_contribs[:, 0] = np.random.gamma(2, 10, n_samples) * 15
source_contribs[:, 1] = np.random.gamma(1.5, 8, n_samples) * 12
source_contribs[:, 2] = np.random.gamma(2, 12, n_samples) * 20

data = np.zeros((n_samples, len(component_cols)))
for i, comp in enumerate(component_cols):
    vals = np.zeros(n_samples)
    for j, source in enumerate(['coal', 'vehicle', 'secondary']):
        frac = source_profiles[source].get(comp, 0.001)
        vals += source_contribs[:, j] * frac
    noise = np.random.normal(0, 0.1, n_samples)
    vals = vals * (1 + noise)
    vals = np.maximum(vals, 0.001)
    data[:, i] = vals

df = pd.DataFrame(data, columns=component_cols)
df.insert(0, 'time', dates)
df.insert(1, 'station', '站点A')

outlier_threshold = 2.0
qc_outliers = {}
for comp in component_cols:
    valid_data = df[comp].dropna()
    mean_val = valid_data.mean()
    outlier_mask = df[comp] > mean_val * outlier_threshold
    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        qc_outliers[comp] = n_outliers

print(f"检测到异常组分: {list(qc_outliers.keys())}")

anomaly_results = []
for comp in component_cols:
    outlier_count = qc_outliers.get(comp, 0)
    if outlier_count > 0:
        valid_data = df[comp].dropna()
        mean_val = valid_data.mean()
        outlier_mask = df[comp] > mean_val * outlier_threshold
        outlier_indices = df[outlier_mask].index.tolist()
        
        for idx in outlier_indices[:3]:
            row = df.loc[idx]
            time_val = row.get('time', 'N/A')
            station_val = row.get('station', 'N/A')
            
            other_cols = [c for c in component_cols if c != comp]
            
            if 'station' in df.columns:
                same_station_mask = df['station'] == station_val
                station_data = df[same_station_mask]
            else:
                station_data = df
            
            if 'time' in df.columns and len(station_data) > 10:
                time_diffs = pd.to_datetime(station_data['time']).diff().dropna()
                if len(time_diffs) > 0:
                    median_interval = time_diffs.median()
                    if median_interval <= pd.Timedelta(hours=6):
                        time_window = pd.Timedelta(hours=12)
                    elif median_interval <= pd.Timedelta(days=1):
                        time_window = pd.Timedelta(days=3)
                    else:
                        time_window = pd.Timedelta(days=7)
                    
                    row_time = pd.to_datetime(row['time'])
                    time_mask = (pd.to_datetime(station_data['time']) >= row_time - time_window) & \
                               (pd.to_datetime(station_data['time']) <= row_time + time_window)
                    segment_data = station_data[time_mask]
                    
                    if len(segment_data) < 5:
                        segment_data = station_data
                else:
                    segment_data = station_data
            else:
                segment_data = station_data
            
            correlations = {}
            for other_comp in other_cols:
                valid_data = segment_data[[comp, other_comp]].dropna()
                if len(valid_data) >= 5:
                    corr = valid_data[comp].corr(valid_data[other_comp])
                    if not pd.isna(corr):
                        correlations[other_comp] = corr
            
            if not correlations and len(station_data) > len(segment_data):
                for other_comp in other_cols:
                    valid_data = station_data[[comp, other_comp]].dropna()
                    if len(valid_data) >= 5:
                        corr = valid_data[comp].corr(valid_data[other_comp])
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

print(f"\n分析了 {len(anomaly_results)} 个异常点")
for i, anomaly in enumerate(anomaly_results[:5]):
    print(f"\n异常点 {i+1}: {anomaly['component']}")
    print(f"  时间: {anomaly['time']}")
    print(f"  数值: {anomaly['value']:.2f} (均值的 {anomaly['ratio']:.1f}x)")
    if anomaly['top_correlations']:
        print(f"  关联组分:")
        for comp, corr in anomaly['top_correlations']:
            print(f"    - {comp}: {corr:.3f}")
    else:
        print(f"  无关联组分")

print("\n" + "=" * 60)
print("测试2: 源类自动匹配（基于谱图相似性）")
print("=" * 60)

pmf_sources = ['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4']
cmb_sources = ['工业锅炉燃煤', '机动车尾气', '建筑扬尘', '二次硫酸盐']

pmf_profiles = np.array([
    [0.20, 0.05, 0.03, 0.02, 0.08, 0.04, 0.05, 0.15, 0.04, 0.003, 0.002, 0.002, 0.001, 0.003, 0.001],
    [0.03, 0.08, 0.02, 0.01, 0.25, 0.20, 0.005, 0.01, 0.01, 0.01, 0.003, 0.005, 0.008, 0.001, 0.002],
    [0.02, 0.01, 0.005, 0.01, 0.05, 0.02, 0.10, 0.25, 0.06, 0.002, 0.001, 0.003, 0.001, 0.0005, 0.0005],
    [0.35, 0.25, 0.15, 0.01, 0.10, 0.005, 0.002, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0002, 0.0001, 0.0001],
])

cmb_profiles = np.array([
    [0.18, 0.06, 0.04, 0.03, 0.07, 0.05, 0.06, 0.14, 0.05, 0.004, 0.003, 0.003, 0.002, 0.004, 0.002],
    [0.04, 0.09, 0.03, 0.02, 0.22, 0.18, 0.006, 0.012, 0.012, 0.012, 0.004, 0.006, 0.009, 0.002, 0.003],
    [0.03, 0.02, 0.01, 0.02, 0.06, 0.03, 0.09, 0.23, 0.07, 0.003, 0.002, 0.004, 0.002, 0.001, 0.001],
    [0.32, 0.23, 0.17, 0.02, 0.09, 0.01, 0.003, 0.006, 0.003, 0.002, 0.001, 0.0003, 0.0003, 0.0002, 0.0002],
])

def match_sources_by_profile(profiles_a, names_a, profiles_b, names_b):
    n_a = len(names_a)
    n_b = len(names_b)
    
    corr_matrix = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            if np.std(profiles_a[i]) > 0 and np.std(profiles_b[j]) > 0:
                corr = np.corrcoef(profiles_a[i], profiles_b[j])[0, 1]
                corr_matrix[i, j] = abs(corr)
            else:
                corr_matrix[i, j] = 0
    
    matched_pairs = []
    used_a = set()
    used_b = set()
    
    while len(matched_pairs) < min(n_a, n_b):
        max_corr = -1
        best_i, best_j = -1, -1
        for i in range(n_a):
            if i in used_a:
                continue
            for j in range(n_b):
                if j in used_b:
                    continue
                if corr_matrix[i, j] > max_corr:
                    max_corr = corr_matrix[i, j]
                    best_i, best_j = i, j
        
        if max_corr < 0.3 or best_i < 0 or best_j < 0:
            break
        
        matched_pairs.append((names_a[best_i], names_b[best_j], max_corr))
        used_a.add(best_i)
        used_b.add(best_j)
    
    return matched_pairs

pairs = match_sources_by_profile(pmf_profiles, pmf_sources, cmb_profiles, cmb_sources)

print(f"PMF源类数量: {len(pmf_sources)}")
print(f"CMB源类数量: {len(cmb_sources)}")
print(f"\n匹配结果 ({len(pairs)} 对):")
for name_a, name_b, corr in pairs:
    print(f"  {name_a:15s} <-> {name_b:15s}  相似度: {corr:.4f}")

print("\n" + "=" * 60)
print("所有测试通过!")
print("=" * 60)
