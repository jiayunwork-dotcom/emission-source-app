import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from src.data.quality import DataQualityChecker, calculate_uncertainty, get_detection_limits
from src.sources.library import SourceSpectrumLibrary
from src.algorithms.cmb import CMBSolver
from src.algorithms.pmf import PMFSolver
from src.algorithms.pca_mlr import PCAMLRSolver
from src.trajectory.pscf_cwt import TrajectoryAnalyzer
from src.visualization.plots import Visualizer
from src.report.pdf_generator import ReportGenerator

def test_data_quality():
    print("=" * 50)
    print("测试数据质检模块")
    print("=" * 50)

    component_cols = ['SO4', 'NO3', 'NH4', 'OC', 'EC', 'Al', 'Si', 'Fe']

    import pandas as pd
    np.random.seed(42)
    n = 100
    data = {
        'time': pd.date_range('2024-01-01', periods=n, freq='D'),
        'station': ['站点A'] * n,
    }
    for col in component_cols:
        vals = np.abs(np.random.randn(n) * 5 + 10)
        vals[10] = np.nan
        vals[20] = -1
        vals[30] = 1000
        data[col] = vals

    df = pd.DataFrame(data)

    checker = DataQualityChecker(outlier_threshold=5.0)
    cleaned_df, qc_report = checker.check(df, component_cols)

    print(f"原始样本数: {qc_report['total_samples']}")
    print(f"有效样本数: {qc_report['valid_samples_after_qc']}")
    print(f"缺失值总数: {sum(qc_report['missing_values'].values())}")
    print(f"负值总数: {sum(qc_report['negative_values'].values())}")
    print(f"异常高值总数: {sum(qc_report['outliers'].values())}")

    detection_limits = get_detection_limits(component_cols)
    print(f"检出限配置: {len(detection_limits)}个组分")

    uncertainty_matrix = np.zeros((len(cleaned_df), len(component_cols)))
    for i, col in enumerate(component_cols):
        dl = detection_limits.get(col, 0.01)
        concentrations = cleaned_df[col].values
        uncertainty_matrix[:, i] = calculate_uncertainty(concentrations, dl)

    print(f"不确定度矩阵形状: {uncertainty_matrix.shape}")
    print("✅ 数据质检模块测试通过\n")


def test_source_library():
    print("=" * 50)
    print("测试源谱库管理模块")
    print("=" * 50)

    library = SourceSpectrumLibrary()
    source_names = library.get_all_names()
    print(f"内置源谱数量: {len(source_names)}")
    print(f"源谱列表: {source_names}")

    component_cols = ['SO4', 'NO3', 'NH4', 'Cl', 'OC', 'EC', 'Al', 'Si', 'Fe', 'Zn', 'Pb']
    source_matrix = library.get_source_matrix(
        ['工业锅炉燃煤', '机动车尾气', '道路扬尘', '二次气溶胶'],
        component_cols
    )
    print(f"源谱矩阵形状: {source_matrix.shape}")

    spec = library.get_spectrum('工业锅炉燃煤')
    print(f"燃煤源谱组分数: {len(spec.components)}")

    print("✅ 源谱库模块测试通过\n")


def test_cmb():
    print("=" * 50)
    print("测试CMB算法")
    print("=" * 50)

    component_cols = ['SO4', 'NO3', 'NH4', 'Cl', 'OC', 'EC', 'Al', 'Si', 'Fe', 'Zn', 'Pb']
    library = SourceSpectrumLibrary()
    source_names = ['工业锅炉燃煤', '机动车尾气', '道路扬尘', '二次气溶胶']

    source_matrix = library.get_source_matrix(source_names, component_cols)

    cmb_solver = CMBSolver(
        source_names=source_names,
        component_names=component_cols,
        source_matrix=source_matrix,
    )

    is_col, msg = cmb_solver.check_collinearity()
    print(f"共线性检查: {msg}")
    print(f"条件数: {cmb_solver.condition_number:.2f}")

    np.random.seed(42)
    n_samples = 50
    true_contribs = np.random.gamma(2, 10, (n_samples, 4))
    X = true_contribs @ source_matrix.T
    X += np.random.normal(0, 0.5, X.shape)
    X = np.maximum(X, 0.01)

    U = X * 0.1 + 0.05

    result = cmb_solver.solve(X, U)
    print(f"Chi-square: {result.chi_square:.4f}")
    print(f"R²: {result.r_squared:.4f}")

    contrib_dict = result.get_contribution_dataframe()
    print("源贡献结果:")
    for name, contrib, pct in zip(contrib_dict['source'], contrib_dict['contribution'], contrib_dict['percentage']):
        print(f"  {name}: {contrib:.4f} ({pct:.2f}%)")

    print("✅ CMB算法测试通过\n")


def test_pmf():
    print("=" * 50)
    print("测试PMF算法")
    print("=" * 50)

    component_cols = ['SO4', 'NO3', 'NH4', 'Cl', 'OC', 'EC', 'Al', 'Si', 'Fe', 'Zn', 'Pb']

    np.random.seed(42)
    n_samples = 50

    true_F = np.array([
        [0.20, 0.05, 0.03, 0.02, 0.08, 0.04, 0.05, 0.15, 0.04, 0.003, 0.002],
        [0.03, 0.08, 0.02, 0.01, 0.25, 0.20, 0.005, 0.01, 0.01, 0.01, 0.003],
        [0.02, 0.01, 0.005, 0.01, 0.05, 0.02, 0.10, 0.25, 0.06, 0.002, 0.001],
        [0.35, 0.25, 0.15, 0.01, 0.10, 0.005, 0.002, 0.005, 0.002, 0.001, 0.0005],
    ])

    true_G = np.random.gamma(2, 5, (n_samples, 4)) + 2

    X = true_G @ true_F
    X += np.random.normal(0, 0.3, X.shape)
    X = np.maximum(X, 0.01)

    U = X * 0.1 + 0.05

    solver = PMFSolver(
        component_names=component_cols,
        n_factors=4,
        max_iterations=200,
        convergence_tolerance=1e-4,
        random_seed=42,
    )

    result = solver.solve(X, U)
    print(f"Q值: {result.Q:.2f}")
    print(f"Q期望值: {result.Q_expected:.2f}")
    print(f"Q/Qexpected: {result.Q_ratio:.4f}")
    print(f"迭代次数: {result.iterations}")
    print(f"是否收敛: {result.converged}")

    contrib_dict = result.get_contribution_dataframe()
    print("源贡献结果:")
    for name, contrib, pct in zip(contrib_dict['source'], contrib_dict['contribution'], contrib_dict['percentage']):
        print(f"  {name}: {contrib:.4f} ({pct:.2f}%)")

    print("\n测试Bootstrap (简化版, 10次)...")
    boot_results = solver.bootstrap(X, U, n_bootstrap=10, block_size=3)
    print(f"Bootstrap成功率: {boot_results['stability_ratio']*100:.1f}% ({boot_results['successes']}/{boot_results['n_bootstrap']})")

    print("✅ PMF算法测试通过\n")


def test_pca_mlr():
    print("=" * 50)
    print("测试PCA-MLR算法")
    print("=" * 50)

    component_cols = ['SO4', 'NO3', 'NH4', 'Cl', 'OC', 'EC', 'Al', 'Si', 'Fe', 'Zn', 'Pb']

    np.random.seed(42)
    n_samples = 50

    true_F = np.array([
        [0.20, 0.05, 0.03, 0.02, 0.08, 0.04, 0.05, 0.15, 0.04, 0.003, 0.002],
        [0.03, 0.08, 0.02, 0.01, 0.25, 0.20, 0.005, 0.01, 0.01, 0.01, 0.003],
        [0.02, 0.01, 0.005, 0.01, 0.05, 0.02, 0.10, 0.25, 0.06, 0.002, 0.001],
        [0.35, 0.25, 0.15, 0.01, 0.10, 0.005, 0.002, 0.005, 0.002, 0.001, 0.0005],
    ])

    true_G = np.random.gamma(2, 5, (n_samples, 4)) + 2
    X = true_G @ true_F
    X += np.random.normal(0, 0.3, X.shape)
    X = np.maximum(X, 0.01)

    total_mass = np.sum(X, axis=1)

    solver = PCAMLRSolver(
        component_names=component_cols,
        variance_threshold=0.8,
    )

    result = solver.solve(X, total_mass)
    print(f"提取主成分数: {result.n_components}")
    print(f"累计方差贡献率: {result.cumulative_variance[-1]*100:.2f}%")
    print(f"R²: {result.r_squared:.4f}")

    result_varimax = solver.solve_with_varimax(X, total_mass)
    print(f"Varimax旋转后主成分数: {result_varimax.n_components}")

    print("✅ PCA-MLR算法测试通过\n")


def test_trajectory():
    print("=" * 50)
    print("测试后向轨迹与潜在源区分析")
    print("=" * 50)

    analyzer = TrajectoryAnalyzer(
        lat_range=(20.0, 50.0),
        lon_range=(90.0, 135.0),
        grid_resolution=1.0,
    )

    trajectories = analyzer.generate_synthetic_trajectories(
        n_trajectories=20,
        start_lat=39.9,
        start_lon=116.4,
        duration_hours=48,
        random_seed=42,
    )

    print(f"生成轨迹数量: {len(trajectories)}")
    print(f"单条轨迹点数: {len(trajectories[0])}")

    concentrations = np.random.rand(20) * 50 + 10

    pscf_result = analyzer.compute_pscf(
        trajectories,
        concentrations,
        threshold_percentile=75.0,
        apply_weighting=True,
    )

    print(f"PSCF最大值: {pscf_result.pscf_values.max():.4f}")
    print(f"PSCF非零网格数: {np.sum(pscf_result.pscf_values > 0)}")

    cwt_result = analyzer.compute_cwt(
        trajectories,
        concentrations,
        time_step_hours=1.0,
        apply_weighting=True,
    )

    print(f"CWT最大值: {cwt_result.cwt_values.max():.4f}")
    print(f"CWT非零网格数: {np.sum(cwt_result.cwt_values > 0)}")

    print("✅ 轨迹分析模块测试通过\n")


def test_visualization():
    print("=" * 50)
    print("测试可视化模块")
    print("=" * 50)

    viz = Visualizer()

    source_names = ['燃煤', '机动车', '扬尘', '二次气溶胶']
    values = np.array([25.3, 18.7, 12.4, 30.1])

    img = viz.pie_chart(source_names, values, title="测试饼图")
    print(f"饼图生成成功, 大小: {len(img.getvalue())} bytes")

    import pandas as pd
    times = pd.date_range('2024-01-01', periods=50, freq='D')
    contribs = np.random.rand(50, 4) * 10 + 5

    img = viz.stacked_area_chart(times, contribs, source_names, title="测试时序图")
    print(f"时序图生成成功, 大小: {len(img.getvalue())} bytes")

    component_names = ['SO4', 'NO3', 'NH4', 'OC', 'EC', 'Al', 'Si', 'Fe']
    profiles = np.random.rand(4, 8)
    profiles = profiles / profiles.sum(axis=1, keepdims=True)

    img = viz.factor_profile_bar_chart(component_names, profiles, source_names, title="测试因子谱图")
    print(f"因子谱图生成成功, 大小: {len(img.getvalue())} bytes")

    print("✅ 可视化模块测试通过\n")


def test_report_generator():
    print("=" * 50)
    print("测试PDF报告生成")
    print("=" * 50)

    report_gen = ReportGenerator()

    data_info = {
        'n_stations': 2,
        'n_samples': 100,
        'n_components': 20,
        'time_range': '2024-01-01 ~ 2024-04-10',
    }

    source_spectra_info = {
        'source_names': ['工业锅炉燃煤', '机动车尾气', '道路扬尘', '二次气溶胶'],
        'descriptions': {
            '工业锅炉燃煤': '工业锅炉燃煤排放',
            '机动车尾气': '机动车尾气排放',
            '道路扬尘': '道路扬尘',
            '二次气溶胶': '二次气溶胶生成',
        },
        'components': ['SO4', 'NO3', 'NH4', 'OC', 'EC'],
        'fractions': {
            '工业锅炉燃煤': {'SO4': 0.2, 'NO3': 0.05, 'NH4': 0.03, 'OC': 0.08, 'EC': 0.04},
            '机动车尾气': {'SO4': 0.03, 'NO3': 0.08, 'NH4': 0.02, 'OC': 0.25, 'EC': 0.2},
            '道路扬尘': {'SO4': 0.02, 'NO3': 0.01, 'NH4': 0.005, 'OC': 0.05, 'EC': 0.02},
            '二次气溶胶': {'SO4': 0.35, 'NO3': 0.25, 'NH4': 0.15, 'OC': 0.1, 'EC': 0.005},
        },
    }

    results = {
        'source_names': ['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4'],
        'contributions': {'Factor 1': 15.2, 'Factor 2': 12.1, 'Factor 3': 8.5, 'Factor 4': 18.3},
        'percentages': {'Factor 1': 28.1, 'Factor 2': 22.4, 'Factor 3': 15.7, 'Factor 4': 33.8},
        'diagnostics': {
            'Q值': 1250.5,
            'Q期望值': 1200.0,
            'Q/Qexpected': 1.042,
            '迭代次数': 350,
            '是否收敛': '是',
        },
        'n_factors': 4,
    }

    qc_info = {
        'total_samples': 100,
        'valid_samples_after_qc': 92,
    }

    pdf_buffer = report_gen.generate_report(
        data_info=data_info,
        source_spectra_info=source_spectra_info,
        results=results,
        algorithm='PMF',
        qc_info=qc_info,
        charts=None,
    )

    print(f"PDF报告生成成功, 大小: {len(pdf_buffer.getvalue())} bytes")

    print("✅ PDF报告生成模块测试通过\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  工业废气排放源解析系统 - 核心模块测试")
    print("=" * 60 + "\n")

    try:
        test_data_quality()
        test_source_library()
        test_cmb()
        test_pmf()
        test_pca_mlr()
        test_trajectory()
        test_visualization()
        test_report_generator()

        print("=" * 60)
        print("🎉 所有模块测试通过！")
        print("=" * 60)
    except Exception as e:
        import traceback
        print(f"\n❌ 测试失败: {e}")
        traceback.print_exc()
        sys.exit(1)
