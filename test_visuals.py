import sys
sys.path.insert(0, '.')
from src.visualization.plots import Visualizer
import numpy as np
import pandas as pd

viz = Visualizer()

print("测试可视化功能...")

# 1. 测试趋势预警图
times = pd.date_range('2024-01-01', periods=50, freq='D')
contributions = np.random.rand(50, 4)
source_names = ['Source1', 'Source2', 'Source3', 'Source4']

img = viz.trend_alert_plot(
    times, contributions, source_names,
    alert_indices=[10, 20],
    alert_sources=['Source1'],
)
print('✓ 趋势预警图成功')

# 2. 测试算法散点对比图
img2 = viz.algorithm_scatter_comparison(
    np.array([10, 20, 30]),
    np.array([12, 18, 33]),
    ['A', 'B', 'C'],
    label_a='PMF', label_b='CMB'
)
print('✓ 算法散点对比图成功')

# 3. 测试灵敏度热力图
img3 = viz.sensitivity_heatmap(
    ['Comp1', 'Comp2', 'Comp3'],
    ['Source1', 'Source2'],
    np.array([[5.0, 20.0], [10.0, 8.0], [3.0, 25.0]]),
    high_threshold=15.0,
)
print('✓ 灵敏度热力图成功')

# 4. 测试异常关联分析图
img4 = viz.anomaly_correlation_bars(
    'SO4',
    ['NO3', 'NH4', 'OC'],
    [0.85, 0.72, 0.45],
)
print('✓ 异常关联分析图成功')

# 5. 测试算法对比柱状图
img5 = viz.algorithm_comparison_bar(
    ['Source1', 'Source2', 'Source3'],
    {
        'PMF': np.array([30, 40, 30]),
        'CMB': np.array([25, 45, 30]),
        'PCA-MLR': np.array([35, 35, 30]),
    },
)
print('✓ 算法对比柱状图成功')

print('')
print('所有可视化功能测试通过!')
