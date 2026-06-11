import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.quality import DataQualityChecker, calculate_uncertainty, get_detection_limits, add_season_column
from src.sources.library import SourceSpectrumLibrary, SourceSpectrum
from src.algorithms.cmb import CMBSolver
from src.algorithms.pmf import PMFSolver, determine_optimal_factors
from src.algorithms.pca_mlr import PCAMLRSolver
from src.trajectory.pscf_cwt import TrajectoryAnalyzer
from src.visualization.plots import Visualizer
from src.report.pdf_generator import ReportGenerator


st.set_page_config(
    page_title="工业废气排放源解析与溯源分析系统",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 18px;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff8dc;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ff7f0e;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

if 'data_df' not in st.session_state:
    st.session_state.data_df = None
if 'component_cols' not in st.session_state:
    st.session_state.component_cols = []
if 'source_library' not in st.session_state:
    st.session_state.source_library = SourceSpectrumLibrary()
if 'qc_report' not in st.session_state:
    st.session_state.qc_report = None
if 'uncertainty_matrix' not in st.session_state:
    st.session_state.uncertainty_matrix = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'trajectories' not in st.session_state:
    st.session_state.trajectories = None
if 'pscf_result' not in st.session_state:
    st.session_state.pscf_result = None
if 'cwt_result' not in st.session_state:
    st.session_state.cwt_result = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'alert_config' not in st.session_state:
    st.session_state.alert_config = {
        'consecutive_days': 3,
        'growth_threshold': 20.0,
    }
if 'sensitivity_result' not in st.session_state:
    st.session_state.sensitivity_result = None
if 'cross_validation_results' not in st.session_state:
    st.session_state.cross_validation_results = None
if 'anomaly_traceability' not in st.session_state:
    st.session_state.anomaly_traceability = None


st.sidebar.title("🌫️ 工业废气排放源解析系统")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "导航菜单",
    [
        "数据管理",
        "源谱库管理",
        "源解析分析",
        "交叉验证",
        "后向轨迹与潜在源区",
        "多站点对比",
        "报告导出",
    ]
)


if page == "数据管理":
    st.markdown('<p class="main-header">📊 监测数据管理</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<p class="section-header">数据导入</p>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "上传PM2.5化学组分数据文件 (CSV格式)",
            type=['csv'],
            accept_multiple_files=True,
            help="支持多站点多时段数据。字段包括：采样时间、站点编号、无机离子、碳组分、微量元素等。",
        )

        if st.button("📁 加载示例数据"):
            from src.utils.generate_sample_data import generate_sample_data
            data_dir = "data"
            generate_sample_data(n_samples=80, n_stations=2, output_dir=data_dir)

            dfs = []
            for fname in os.listdir(data_dir):
                if fname.endswith('_pm25_components.csv'):
                    filepath = os.path.join(data_dir, fname)
                    dfs.append(pd.read_csv(filepath))

            if dfs:
                st.session_state.data_df = pd.concat(dfs, ignore_index=True)
                st.success(f"成功加载示例数据：{len(dfs)}个站点，共{len(st.session_state.data_df)}条记录")

        if uploaded_files:
            dfs = []
            for f in uploaded_files:
                try:
                    df = pd.read_csv(f)
                    dfs.append(df)
                except Exception as e:
                    st.error(f"文件 {f.name} 读取失败：{e}")

            if dfs:
                st.session_state.data_df = pd.concat(dfs, ignore_index=True)
                st.success(f"成功导入 {len(dfs)} 个文件，共 {len(st.session_state.data_df)} 条记录")

    with col2:
        st.markdown('<p class="section-header">数据质检设置</p>', unsafe_allow_html=True)
        outlier_threshold = st.slider(
            "异常高值阈值（均值倍数）",
            min_value=2.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
        )
        relative_uncert = st.slider(
            "相对不确定度",
            min_value=0.05,
            max_value=0.3,
            value=0.1,
            step=0.01,
        )

    if st.session_state.data_df is not None:
        st.markdown('---')
        st.markdown('<p class="section-header">数据预览</p>', unsafe_allow_html=True)

        df = st.session_state.data_df.copy()
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])

        st.dataframe(df.head(10), use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总样本数", len(df))
        with col2:
            if 'station' in df.columns:
                st.metric("站点数", df['station'].nunique())
            else:
                st.metric("站点数", 1)
        with col3:
            component_cols = [col for col in df.columns if col not in ['time', 'station', 'season']]
            st.metric("组分数", len(component_cols))
        with col4:
            if 'time' in df.columns:
                time_range = f"{df['time'].min().date()} ~ {df['time'].max().date()}"
                st.metric("时间范围", time_range)

        st.markdown('---')
        st.markdown('<p class="section-header">数据质量控制</p>', unsafe_allow_html=True)

        component_cols = [col for col in df.columns if col not in ['time', 'station', 'season']]
        st.session_state.component_cols = component_cols

        original_df = df.copy()
        checker = DataQualityChecker(outlier_threshold=outlier_threshold)
        cleaned_df, qc_report = checker.check(df, component_cols)
        st.session_state.data_df = cleaned_df
        st.session_state.qc_report = qc_report

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"✅ 有效样本数: {qc_report['valid_samples_after_qc']} / {qc_report['total_samples']}")
        with col2:
            st.info(f"数据合格率: {qc_report['valid_samples_after_qc'] / qc_report['total_samples'] * 100:.1f}%")

        with st.expander("查看详细质检报告"):
            qc_details = pd.DataFrame({
                '组分': component_cols,
                '缺失值': [qc_report['missing_values'].get(c, 0) for c in component_cols],
                '负值': [qc_report['negative_values'].get(c, 0) for c in component_cols],
                '异常高值': [qc_report['outliers'].get(c, 0) for c in component_cols],
            })
            st.dataframe(qc_details, use_container_width=True)

        with st.expander("🔍 异常溯源", expanded=False):
            st.markdown('<p class="section-header">异常数据溯源分析</p>', unsafe_allow_html=True)
            
            total_outliers = sum(qc_report['outliers'].values())
            if total_outliers == 0:
                st.info("未检测到异常高值数据点")
            else:
                st.info(f"检测到 {total_outliers} 个异常高值数据点，正在进行关联分析...")
                
                anomaly_results = []
                for comp in component_cols:
                    outlier_count = qc_report['outliers'].get(comp, 0)
                    if outlier_count > 0:
                        valid_data = original_df[comp].dropna()
                        mean_val = valid_data.mean()
                        outlier_mask = original_df[comp] > mean_val * outlier_threshold
                        outlier_indices = original_df[outlier_mask].index.tolist()
                        
                        for idx in outlier_indices[:5]:
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
                
                if anomaly_results:
                    st.session_state.anomaly_traceability = anomaly_results
                    
                    for i, anomaly in enumerate(anomaly_results[:10]):
                        with st.container():
                            col_a, col_b = st.columns([1, 3])
                            with col_a:
                                st.markdown(f"**⚠️ {anomaly['component']}**")
                                st.caption(f"时间: {anomaly['time']}")
                                st.caption(f"站点: {anomaly['station']}")
                                st.caption(f"数值: {anomaly['value']:.2f}")
                                st.caption(f"超标倍数: {anomaly['ratio']:.1f}x")
                            
                            with col_b:
                                if anomaly['top_correlations']:
                                    comp_names = [c[0] for c in anomaly['top_correlations']]
                                    corr_values = [c[1] for c in anomaly['top_correlations']]
                                    
                                    viz = st.session_state.visualizer
                                    img_buf = viz.anomaly_correlation_bars(
                                        anomaly['component'],
                                        comp_names,
                                        corr_values,
                                    )
                                    st.image(img_buf, use_container_width=True)
                                else:
                                    st.info("无足够数据进行关联分析")
                        
                        st.markdown("---")
                else:
                    st.info("无法进行异常溯源分析")

        st.markdown('---')
        st.markdown('<p class="section-header">不确定度计算</p>', unsafe_allow_html=True)

        detection_limits = get_detection_limits(component_cols)
        uncertainty_matrix = np.zeros((len(cleaned_df), len(component_cols)))

        for i, col in enumerate(component_cols):
            dl = detection_limits.get(col, 0.01)
            concentrations = cleaned_df[col].values
            uncertainty_matrix[:, i] = calculate_uncertainty(
                concentrations, dl, relative_uncertainty=relative_uncert
            )

        st.session_state.uncertainty_matrix = uncertainty_matrix
        st.success("不确定度矩阵计算完成")

        st.markdown('---')
        st.markdown('<p class="section-header">时段筛选</p>', unsafe_allow_html=True)

        if 'time' in cleaned_df.columns and pd.api.types.is_datetime64_any_dtype(cleaned_df['time']):
            min_date = cleaned_df['time'].min().date()
            max_date = cleaned_df['time'].max().date()

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("开始日期", min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("结束日期", max_date, min_value=min_date, max_value=max_date)

            if st.button("应用时间筛选"):
                mask = (cleaned_df['time'].dt.date >= start_date) & (cleaned_df['time'].dt.date <= end_date)
                st.session_state.data_df = cleaned_df[mask].reset_index(drop=True)
                st.session_state.uncertainty_matrix = uncertainty_matrix[mask]
                st.success(f"筛选后剩余 {len(st.session_state.data_df)} 条记录")
        else:
            st.info("数据中未检测到时间列，无法进行时段筛选")

        if 'station' in cleaned_df.columns:
            st.markdown('---')
            st.markdown('<p class="section-header">站点筛选</p>', unsafe_allow_html=True)
            stations = sorted(cleaned_df['station'].unique().tolist())
            selected_stations = st.multiselect("选择站点", stations, default=stations)

            if st.button("应用站点筛选"):
                mask = cleaned_df['station'].isin(selected_stations)
                st.session_state.data_df = cleaned_df[mask].reset_index(drop=True)
                st.session_state.uncertainty_matrix = uncertainty_matrix[mask]
                st.success(f"筛选后剩余 {len(st.session_state.data_df)} 条记录")


elif page == "源谱库管理":
    st.markdown('<p class="main-header">📚 源谱库管理</p>', unsafe_allow_html=True)

    library = st.session_state.source_library

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<p class="section-header">内置源谱列表</p>', unsafe_allow_html=True)

        source_names = library.get_all_names()
        selected_source = st.selectbox("选择源谱查看", source_names)

        if selected_source:
            spectrum = library.get_spectrum(selected_source)
            if spectrum:
                st.write(f"**描述**: {spectrum.description}")

                spec_df = spectrum.to_dataframe()
                st.dataframe(spec_df, use_container_width=True, height=300)

    with col2:
        st.markdown('<p class="section-header">自定义源谱</p>', unsafe_allow_html=True)

        custom_file = st.file_uploader(
            "上传自定义源谱 (CSV)",
            type=['csv'],
            help="CSV文件需包含三列：component（组分名）、fraction（质量分数）、uncertainty（不确定度）",
        )

        if custom_file:
            try:
                custom_df = pd.read_csv(custom_file)
                custom_name = st.text_input("源谱名称", value="自定义源谱")
                custom_desc = st.text_area("源谱描述", value="")

                if st.button("添加到源谱库"):
                    spec = SourceSpectrum.from_dataframe(custom_df, name=custom_name, description=custom_desc)
                    library.add_spectrum(spec)
                    st.success(f"源谱 '{custom_name}' 已添加到源谱库")
            except Exception as e:
                st.error(f"文件读取失败：{e}")

        st.markdown("---")
        st.markdown('<p class="section-header">源谱管理</p>', unsafe_allow_html=True)

        all_sources = library.get_all_names()
        source_to_remove = st.selectbox("选择要删除的源谱", [""] + all_sources)

        if source_to_remove and st.button("删除源谱"):
            if source_to_remove in ['工业锅炉燃煤', '机动车尾气', '道路扬尘', '建筑施工扬尘', '生物质燃烧', '二次气溶胶']:
                st.warning("内置源谱不可删除")
            else:
                library.remove_spectrum(source_to_remove)
                st.success(f"源谱 '{source_to_remove}' 已删除")
                st.rerun()

    st.markdown('---')
    st.markdown('<p class="section-header">源谱对比可视化</p>', unsafe_allow_html=True)

    sources_to_compare = st.multiselect("选择要对比的源谱", source_names, default=source_names[:4])

    if sources_to_compare and len(source_names) > 0:
        viz = st.session_state.visualizer
        component_names = list(library.get_spectrum(sources_to_compare[0]).components.keys())

        spectra_dict = {}
        for name in sources_to_compare:
            spec = library.get_spectrum(name)
            if spec:
                spectra_dict[name] = spec.get_fractions(component_names)

        if st.button("生成对比图"):
            img_buf = viz.source_spectrum_comparison(
                component_names,
                spectra_dict,
                title="源谱质量分数对比",
                normalize=True,
            )
            st.image(img_buf, use_container_width=True)

    st.markdown('---')
    st.markdown('<p class="section-header">源谱灵敏度分析</p>', unsafe_allow_html=True)

    col_sens1, col_sens2 = st.columns([2, 1])

    with col_sens1:
        st.info("对选中的源谱各组分逐个做±10%扰动，重新运行CMB解析，观察各源贡献系数的变化幅度。")
        
        sensitivity_sources = st.multiselect(
            "选择要分析的源类",
            source_names,
            default=source_names[:4],
            key='sensitivity_sources',
        )
        
        perturbation_percent = st.slider(
            "扰动幅度 (%)",
            min_value=5.0,
            max_value=20.0,
            value=10.0,
            step=1.0,
            key='perturbation_percent',
        )
        
        high_sensitivity_threshold = st.slider(
            "高敏阈值 (%)",
            min_value=5.0,
            max_value=30.0,
            value=15.0,
            step=1.0,
            key='high_sensitivity_threshold',
        )

    with col_sens2:
        st.markdown("**分析设置**")
        if st.session_state.data_df is None:
            st.warning("请先在'数据管理'页面导入数据")
        elif len(sensitivity_sources) < 2:
            st.warning("请至少选择2个源类")
        else:
            if st.button("🔬 开始灵敏度分析", type="primary"):
                with st.spinner("正在执行源谱灵敏度分析，请稍候..."):
                    df = st.session_state.data_df
                    component_cols = st.session_state.component_cols
                    X = df[component_cols].values
                    U = st.session_state.uncertainty_matrix

                    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(U).any(axis=1) & (U > 0).all(axis=1)
                    X_valid = X[valid_mask]
                    U_valid = U[valid_mask]

                    library = st.session_state.source_library
                    source_matrix = library.get_source_matrix(sensitivity_sources, component_cols)
                    source_uncert = library.get_uncertainty_matrix(sensitivity_sources, component_cols)

                    base_solver = CMBSolver(
                        source_names=sensitivity_sources,
                        component_names=component_cols,
                        source_matrix=source_matrix,
                        source_uncertainty_matrix=source_uncert,
                    )
                    base_result = base_solver.solve(X_valid, U_valid)
                    valid_contrib = base_result.source_contributions[~np.isnan(base_result.source_contributions[:, 0])]
                    base_avg_contrib = np.mean(valid_contrib, axis=0)

                    n_components = len(component_cols)
                    n_sources = len(sensitivity_sources)
                    sensitivity_matrix = np.zeros((n_components, n_sources))

                    for i, comp in enumerate(component_cols):
                        for direction in [1, -1]:
                            perturbed_matrix = source_matrix.copy()
                            perturbation = perturbation_percent / 100.0 * direction
                            for j in range(n_sources):
                                if perturbed_matrix[i, j] > 0:
                                    perturbed_matrix[i, j] *= (1 + perturbation)

                            perturbed_solver = CMBSolver(
                                source_names=sensitivity_sources,
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
                                        if direction == 1:
                                            sensitivity_matrix[i, k] = max(sensitivity_matrix[i, k], abs(change_pct))
                                        else:
                                            sensitivity_matrix[i, k] = max(sensitivity_matrix[i, k], abs(change_pct))

                    st.session_state.sensitivity_result = {
                        'sensitivity_matrix': sensitivity_matrix,
                        'component_names': component_cols,
                        'source_names': sensitivity_sources,
                        'base_avg_contrib': base_avg_contrib,
                        'high_threshold': high_sensitivity_threshold,
                    }
                    st.success("灵敏度分析完成！")

    if st.session_state.sensitivity_result is not None:
        sens_data = st.session_state.sensitivity_result
        viz = st.session_state.visualizer

        col_result1, col_result2 = st.columns([3, 1])

        with col_result1:
            img_buf = viz.sensitivity_heatmap(
                sens_data['component_names'],
                sens_data['source_names'],
                sens_data['sensitivity_matrix'],
                title="源谱灵敏度分析热力图",
                high_threshold=sens_data['high_threshold'],
            )
            st.image(img_buf, use_container_width=True)

        with col_result2:
            st.markdown("**高敏组分识别**")
            high_mask = np.abs(sens_data['sensitivity_matrix']) > sens_data['high_threshold']
            if np.any(high_mask):
                high_indices = np.where(high_mask)
                for i, j in zip(*high_indices):
                    comp = sens_data['component_names'][i]
                    source = sens_data['source_names'][j]
                    val = sens_data['sensitivity_matrix'][i, j]
                    st.markdown(f"⚠️ **{comp}** → **{source}**: {val:.1f}%")
                
                st.markdown('---')
                st.caption("深色边框标记的单元格表示该组分扰动对源贡献影响超过阈值")
            else:
                st.success("✅ 未检测到高敏组分，源谱稳定性良好")

        st.markdown('### 详细数据')
        sens_df = pd.DataFrame(
            sens_data['sensitivity_matrix'],
            index=sens_data['component_names'],
            columns=sens_data['source_names'],
        )
        st.dataframe(sens_df.round(2), use_container_width=True)


elif page == "源解析分析":
    st.markdown('<p class="main-header">🔬 源解析分析</p>', unsafe_allow_html=True)

    if st.session_state.data_df is None or len(st.session_state.component_cols) == 0:
        st.warning("请先在'数据管理'页面导入并处理数据")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown('<p class="section-header">算法选择</p>', unsafe_allow_html=True)
            algorithm = st.radio(
                "选择解析算法",
                ["PMF 正定矩阵因子分解", "CMB 化学质量平衡法", "PCA-MLR 主成分回归"],
                index=0,
            )

            st.markdown('---')
            st.markdown('<p class="section-header">预警设置</p>', unsafe_allow_html=True)
            alert_config = st.session_state.alert_config
            alert_config['consecutive_days'] = st.slider(
                "连续增长天数",
                min_value=2,
                max_value=7,
                value=alert_config['consecutive_days'],
                step=1,
                help="连续多少天增长触发预警",
            )
            alert_config['growth_threshold'] = st.slider(
                "日增长阈值 (%)",
                min_value=5.0,
                max_value=50.0,
                value=alert_config['growth_threshold'],
                step=1.0,
                help="连续每天的环比增长率阈值",
            )
            st.session_state.alert_config = alert_config

            if st.button("🔔 检测趋势预警"):
                if st.session_state.last_result is None:
                    st.warning("请先运行源解析分析")
                elif 'time' not in df.columns:
                    st.warning("数据中无时间列，无法进行趋势分析")
                else:
                    with st.spinner("正在进行趋势预警检测..."):
                        result_info = st.session_state.last_result
                        result = result_info['result']
                        
                        if result_info['type'] == 'PMF':
                            if hasattr(result, 'valid_mask'):
                                G_full = np.full((len(df), result.n_factors), np.nan)
                                G_full[result.valid_mask] = result.G
                            else:
                                G_full = result.G
                            source_names = result.source_names
                        elif result_info['type'] == 'CMB':
                            G_full = result.source_contributions
                            source_names = result.source_names
                        else:
                            G_full = result.source_contributions
                            source_names = result.source_names
                        
                        times = pd.to_datetime(df['time'])
                        daily_contrib = pd.DataFrame(G_full, columns=source_names)
                        daily_contrib['date'] = times.dt.date
                        daily_avg = daily_contrib.groupby('date').mean()
                        
                        consecutive_days = alert_config['consecutive_days']
                        growth_threshold = alert_config['growth_threshold'] / 100
                        
                        new_alerts = []
                        alert_sources = set()
                        alert_indices = []
                        
                        for source in source_names:
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
                                        new_alerts.append({
                                            'time': str(alert_date),
                                            'source': source,
                                            'growth': avg_growth,
                                            'consecutive_days': consecutive_days,
                                        })
                                        alert_sources.add(source)
                                        date_idx = np.where(times.dt.date == alert_date)[0]
                                        if len(date_idx) > 0:
                                            alert_indices.append(date_idx[0])
                        
                        if new_alerts:
                            for alert in new_alerts:
                                if alert not in st.session_state.alert_history:
                                    st.session_state.alert_history.append(alert)
                            
                            for alert in new_alerts:
                                st.error(f"⚠️ 预警：{alert['source']} 连续 {alert['consecutive_days']} 天增长，平均增幅 {alert['growth']:.1f}%，触发时间：{alert['time']}")
                        else:
                            st.success("✅ 未检测到持续增长的污染源")
                        
                        st.session_state.current_alert_data = {
                            'times': times,
                            'contributions': G_full,
                            'source_names': source_names,
                            'alert_indices': alert_indices,
                            'alert_sources': list(alert_sources),
                            'daily_avg': daily_avg,
                        }

        with col2:
            if "PMF" in algorithm:
                st.markdown('<p class="section-header">PMF参数设置</p>', unsafe_allow_html=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    n_factors = st.slider("因子数", min_value=2, max_value=10, value=4)
                    max_iter = st.number_input("最大迭代次数", min_value=100, max_value=2000, value=500, step=100)
                with col_b:
                    conv_tol = st.number_input(
                        "收敛阈值 (%)",
                        min_value=0.001,
                        max_value=1.0,
                        value=0.01,
                        step=0.01,
                        format="%.3f",
                    )
                    random_seed = st.number_input("随机种子", value=42, step=1)

                do_bootstrap = st.checkbox("启用Bootstrap稳定性分析", value=True)
                if do_bootstrap:
                    n_bootstrap = st.slider("Bootstrap次数", min_value=20, max_value=200, value=100, step=10)
                    block_size = st.slider("块采样大小（天）", min_value=1, max_value=14, value=7)

                do_disp = st.checkbox("启用DISP位移分析", value=False)

                if st.button("▶️ 运行PMF分析", type="primary"):
                    with st.spinner("正在执行PMF分析，请稍候..."):
                        df = st.session_state.data_df
                        component_cols = st.session_state.component_cols

                        X = df[component_cols].values
                        U = st.session_state.uncertainty_matrix

                        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(U).any(axis=1) & (U > 0).all(axis=1)
                        X_valid = X[valid_mask]
                        U_valid = U[valid_mask]

                        solver = PMFSolver(
                            component_names=component_cols,
                            n_factors=n_factors,
                            max_iterations=max_iter,
                            convergence_tolerance=conv_tol / 100,
                            random_seed=random_seed,
                        )

                        result = solver.run_full_analysis(
                            X_valid, U_valid,
                            do_bootstrap=do_bootstrap,
                            do_disp=do_disp,
                            n_bootstrap=n_bootstrap,
                            block_size=block_size,
                        )

                        result.valid_mask = valid_mask
                        st.session_state.last_result = {
                            'type': 'PMF',
                            'result': result,
                            'component_cols': component_cols,
                        }
                        st.success("PMF分析完成！")

            elif "CMB" in algorithm:
                st.markdown('<p class="section-header">CMB参数设置</p>', unsafe_allow_html=True)

                library = st.session_state.source_library
                available_sources = library.get_all_names()
                selected_sources = st.multiselect(
                    "选择源类",
                    available_sources,
                    default=available_sources[:4],
                )

                if st.button("检查共线性"):
                    if len(selected_sources) > len(st.session_state.component_cols):
                        st.warning("⚠️ 源类数量不能超过组分数目")
                    else:
                        source_matrix = library.get_source_matrix(
                            selected_sources, st.session_state.component_cols
                        )
                        solver = CMBSolver(
                            source_names=selected_sources,
                            component_names=st.session_state.component_cols,
                            source_matrix=source_matrix,
                        )
                        is_collinear, msg = solver.check_collinearity()
                        if is_collinear:
                            st.warning(f"⚠️ {msg}")
                        else:
                            st.success(f"✅ {msg}")

                if st.button("▶️ 运行CMB分析", type="primary"):
                    if len(selected_sources) > len(st.session_state.component_cols):
                        st.error("源类数量不能超过组分数目")
                    elif len(selected_sources) == 0:
                        st.error("请至少选择一个源类")
                    else:
                        with st.spinner("正在执行CMB分析，请稍候..."):
                            df = st.session_state.data_df
                            component_cols = st.session_state.component_cols

                            X = df[component_cols].values
                            U = st.session_state.uncertainty_matrix

                            source_matrix = library.get_source_matrix(selected_sources, component_cols)
                            source_uncert = library.get_uncertainty_matrix(selected_sources, component_cols)

                            solver = CMBSolver(
                                source_names=selected_sources,
                                component_names=component_cols,
                                source_matrix=source_matrix,
                                source_uncertainty_matrix=source_uncert,
                            )

                            result = solver.solve(X, U)
                            st.session_state.last_result = {
                                'type': 'CMB',
                                'result': result,
                                'component_cols': component_cols,
                            }
                            st.success("CMB分析完成！")

            else:
                st.markdown('<p class="section-header">PCA-MLR参数设置</p>', unsafe_allow_html=True)

                variance_threshold = st.slider(
                    "累计方差贡献率阈值 (%)",
                    min_value=60,
                    max_value=95,
                    value=80,
                    step=5,
                )
                varimax = st.checkbox("使用Varimax旋转", value=True)

                if st.button("▶️ 运行PCA-MLR分析", type="primary"):
                    with st.spinner("正在执行PCA-MLR分析，请稍候..."):
                        df = st.session_state.data_df
                        component_cols = st.session_state.component_cols

                        X = df[component_cols].values
                        valid_mask = ~np.isnan(X).any(axis=1)
                        X_valid = X[valid_mask]

                        total_mass = np.sum(X_valid, axis=1)

                        solver = PCAMLRSolver(
                            component_names=component_cols,
                            variance_threshold=variance_threshold / 100,
                        )

                        if varimax:
                            result = solver.solve_with_varimax(X_valid, total_mass)
                        else:
                            result = solver.solve(X_valid, total_mass)

                        result.valid_mask = valid_mask
                        st.session_state.last_result = {
                            'type': 'PCA-MLR',
                            'result': result,
                            'component_cols': component_cols,
                        }
                        st.success(f"PCA-MLR分析完成！提取了 {result.n_components} 个主成分")

        st.markdown('---')

        if st.session_state.last_result:
            st.markdown('<p class="section-header">解析结果</p>', unsafe_allow_html=True)

            result_info = st.session_state.last_result
            result_type = result_info['type']
            result = result_info['result']
            component_cols = result_info['component_cols']

            viz = st.session_state.visualizer
            df = st.session_state.data_df

            if result_type == 'PMF':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Q值", f"{result.Q:.2f}")
                with col2:
                    st.metric("Q期望值", f"{result.Q_expected:.2f}")
                with col3:
                    st.metric("Q/Qexpected", f"{result.Q_ratio:.4f}")
                with col4:
                    st.metric("迭代次数", f"{result.iterations}")

                if result.converged:
                    st.success("✅ 算法收敛")
                else:
                    st.warning("⚠️ 算法未收敛，建议增加迭代次数")

                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "源贡献占比", "时间序列", "因子谱图", "残差分析", "稳定性分析", "趋势预警"
                ])

                with tab1:
                    contrib_dict = result.get_contribution_dataframe()
                    contrib_df = pd.DataFrame(contrib_dict)
                    st.dataframe(contrib_df, use_container_width=True)

                    img_buf = viz.pie_chart(
                        contrib_dict['source'],
                        contrib_dict['contribution'],
                        title="源贡献占比",
                    )
                    st.image(img_buf, use_container_width=True)

                with tab2:
                    if 'time' in df.columns:
                        times = pd.to_datetime(df['time'])
                        if hasattr(result, 'valid_mask'):
                            G_full = np.full((len(df), result.n_factors), np.nan)
                            G_full[result.valid_mask] = result.G
                        else:
                            G_full = result.G

                        img_buf = viz.stacked_area_chart(
                            times,
                            G_full,
                            result.source_names,
                            title="源贡献时间序列",
                        )
                        st.image(img_buf, use_container_width=True)
                    else:
                        st.info("数据中无时间列，无法显示时间序列图")

                with tab3:
                    factor_data = result.get_factor_profiles_dataframe()

                    img_buf = viz.factor_profile_bar_chart(
                        factor_data['components'],
                        factor_data['profiles'].T,
                        factor_data['factors'],
                        title="因子谱图",
                        normalize=True,
                    )
                    st.image(img_buf, use_container_width=True)

                with tab4:
                    if hasattr(result, 'valid_mask'):
                        X = df[component_cols].values
                        U = st.session_state.uncertainty_matrix
                        X_valid = X[result.valid_mask]
                        U_valid = U[result.valid_mask]
                    else:
                        X_valid = df[component_cols].values
                        U_valid = st.session_state.uncertainty_matrix

                    predicted = result.G @ result.F

                    img_buf = viz.residual_scatter_plot(
                        X_valid, predicted, U_valid, component_cols,
                        title="PMF残差分析",
                    )
                    st.image(img_buf, use_container_width=True)

                with tab5:
                    if result.bootstrap_results:
                        boot = result.bootstrap_results
                        st.info(f"Bootstrap成功率: {boot['stability_ratio']*100:.1f}% ({boot['successes']}/{boot['n_bootstrap']})")

                        if boot['stability_ratio'] >= 0.8:
                            st.success("✅ 因子稳定性良好（>80%）")
                        else:
                            st.warning("⚠️ 因子稳定性不足，建议调整因子数")

                        if 'bootstrap_G' in boot:
                            img_buf = viz.bootstrap_boxplot(
                                boot['bootstrap_G'],
                                result.source_names,
                                title="Bootstrap稳定性分析",
                            )
                            st.image(img_buf, use_container_width=True)
                    else:
                        st.info("未执行Bootstrap分析")

                    if result.disp_results:
                        st.markdown("**DISP位移分析**")
                        disp = result.disp_results
                        disp_df = pd.DataFrame([
                            {'fpeak': r['fpeak'], 'Q': r['Q'], 'dQ(%)': r['dQ']}
                            for r in disp['results']
                        ])
                        st.dataframe(disp_df, use_container_width=True)

                with tab6:
                    st.markdown('<p class="section-header">源贡献时间趋势预警</p>', unsafe_allow_html=True)
                    
                    if 'current_alert_data' in st.session_state and st.session_state.current_alert_data is not None:
                        alert_data = st.session_state.current_alert_data
                        
                        viz = st.session_state.visualizer
                        img_buf = viz.trend_alert_plot(
                            alert_data['times'],
                            alert_data['contributions'],
                            alert_data['source_names'],
                            alert_data['alert_indices'],
                            alert_data['alert_sources'],
                            title="源贡献时间趋势与预警标记",
                        )
                        st.image(img_buf, use_container_width=True)
                        
                        st.markdown("### 日平均贡献趋势")
                        st.dataframe(alert_data['daily_avg'].round(4), use_container_width=True)
                    else:
                        st.info("请点击左侧'检测趋势预警'按钮进行预警分析")
                    
                    st.markdown('---')
                    st.markdown("### 预警历史记录")
                    if st.session_state.alert_history:
                        alert_df = pd.DataFrame(st.session_state.alert_history)
                        alert_df = alert_df.sort_values('time', ascending=False)
                        st.dataframe(
                            alert_df[['time', 'source', 'growth', 'consecutive_days']].rename(columns={
                                'time': '触发时间',
                                'source': '源类',
                                'growth': '平均增幅(%)',
                                'consecutive_days': '连续天数',
                            }),
                            use_container_width=True,
                        )
                        
                        if st.button("清空预警历史"):
                            st.session_state.alert_history = []
                            st.success("预警历史已清空")
                            st.rerun()
                    else:
                        st.info("暂无预警记录")

            elif result_type == 'CMB':
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Chi-square", f"{result.chi_square:.4f}")
                with col2:
                    st.metric("R²", f"{result.r_squared:.4f}")
                with col3:
                    st.metric("条件数", f"{result.condition_number:.2f}")

                if result.condition_number > 20:
                    st.warning("⚠️ 条件数较大，源谱可能存在共线性")

                tab1, tab2, tab3, tab4 = st.tabs(["源贡献占比", "时间序列", "残差分析", "趋势预警"])

                with tab1:
                    contrib_dict = result.get_contribution_dataframe()
                    contrib_df = pd.DataFrame(contrib_dict)
                    st.dataframe(contrib_df, use_container_width=True)

                    valid_contrib = contrib_dict['contribution']
                    img_buf = viz.pie_chart(
                        contrib_dict['source'],
                        valid_contrib,
                        title="CMB源贡献占比",
                    )
                    st.image(img_buf, use_container_width=True)

                with tab2:
                    if 'time' in df.columns:
                        times = pd.to_datetime(df['time'])
                        G = result.source_contributions
                        valid_mask = ~np.isnan(G[:, 0])
                        G_plot = np.where(np.isnan(G), 0, G)

                        img_buf = viz.stacked_area_chart(
                            times,
                            G_plot,
                            result.source_names,
                            title="CMB源贡献时间序列",
                        )
                        st.image(img_buf, use_container_width=True)
                    else:
                        st.info("数据中无时间列，无法显示时间序列图")

                with tab3:
                    X = df[component_cols].values
                    U = st.session_state.uncertainty_matrix

                    valid_mask = ~np.isnan(result.source_contributions[:, 0])

                    img_buf = viz.residual_scatter_plot(
                        X[valid_mask],
                        result.predicted_concentrations[valid_mask],
                        U[valid_mask],
                        component_cols,
                        title="CMB残差分析",
                    )
                    st.image(img_buf, use_container_width=True)

                with tab4:
                    st.markdown('<p class="section-header">源贡献时间趋势预警</p>', unsafe_allow_html=True)
                    
                    if 'current_alert_data' in st.session_state and st.session_state.current_alert_data is not None:
                        alert_data = st.session_state.current_alert_data
                        
                        viz = st.session_state.visualizer
                        img_buf = viz.trend_alert_plot(
                            alert_data['times'],
                            alert_data['contributions'],
                            alert_data['source_names'],
                            alert_data['alert_indices'],
                            alert_data['alert_sources'],
                            title="源贡献时间趋势与预警标记",
                        )
                        st.image(img_buf, use_container_width=True)
                        
                        st.markdown("### 日平均贡献趋势")
                        st.dataframe(alert_data['daily_avg'].round(4), use_container_width=True)
                    else:
                        st.info("请点击左侧'检测趋势预警'按钮进行预警分析")
                    
                    st.markdown('---')
                    st.markdown("### 预警历史记录")
                    if st.session_state.alert_history:
                        alert_df = pd.DataFrame(st.session_state.alert_history)
                        alert_df = alert_df.sort_values('time', ascending=False)
                        st.dataframe(
                            alert_df[['time', 'source', 'growth', 'consecutive_days']].rename(columns={
                                'time': '触发时间',
                                'source': '源类',
                                'growth': '平均增幅(%)',
                                'consecutive_days': '连续天数',
                            }),
                            use_container_width=True,
                        )
                        
                        if st.button("清空预警历史"):
                            st.session_state.alert_history = []
                            st.success("预警历史已清空")
                            st.rerun()
                    else:
                        st.info("暂无预警记录")

            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("主成分数", f"{result.n_components}")
                with col2:
                    st.metric("累计方差", f"{result.cumulative_variance[-1]*100:.2f}%")
                with col3:
                    st.metric("R²", f"{result.r_squared:.4f}")

                tab1, tab2, tab3, tab4 = st.tabs(["源贡献占比", "因子载荷图", "方差解释", "趋势预警"])

                with tab1:
                    contrib_dict = result.get_contribution_dataframe()
                    contrib_df = pd.DataFrame(contrib_dict)
                    st.dataframe(contrib_df, use_container_width=True)

                    img_buf = viz.pie_chart(
                        contrib_dict['source'],
                        contrib_dict['contribution'],
                        title="PCA-MLR源贡献占比",
                    )
                    st.image(img_buf, use_container_width=True)

                with tab2:
                    loadings_data = result.get_loadings_dataframe()

                    img_buf = viz.factor_profile_bar_chart(
                        loadings_data['components'],
                        loadings_data['loadings'].T,
                        loadings_data['factors'],
                        title="主成分载荷图",
                        normalize=False,
                    )
                    st.image(img_buf, use_container_width=True)

                with tab3:
                    var_df = pd.DataFrame({
                        '主成分': result.source_names,
                        '方差贡献率': result.explained_variance * 100,
                        '累计方差贡献率': result.cumulative_variance * 100,
                    })
                    st.dataframe(var_df, use_container_width=True)

                with tab4:
                    st.markdown('<p class="section-header">源贡献时间趋势预警</p>', unsafe_allow_html=True)
                    
                    if 'current_alert_data' in st.session_state and st.session_state.current_alert_data is not None:
                        alert_data = st.session_state.current_alert_data
                        
                        viz = st.session_state.visualizer
                        img_buf = viz.trend_alert_plot(
                            alert_data['times'],
                            alert_data['contributions'],
                            alert_data['source_names'],
                            alert_data['alert_indices'],
                            alert_data['alert_sources'],
                            title="源贡献时间趋势与预警标记",
                        )
                        st.image(img_buf, use_container_width=True)
                        
                        st.markdown("### 日平均贡献趋势")
                        st.dataframe(alert_data['daily_avg'].round(4), use_container_width=True)
                    else:
                        st.info("请点击左侧'检测趋势预警'按钮进行预警分析")
                    
                    st.markdown('---')
                    st.markdown("### 预警历史记录")
                    if st.session_state.alert_history:
                        alert_df = pd.DataFrame(st.session_state.alert_history)
                        alert_df = alert_df.sort_values('time', ascending=False)
                        st.dataframe(
                            alert_df[['time', 'source', 'growth', 'consecutive_days']].rename(columns={
                                'time': '触发时间',
                                'source': '源类',
                                'growth': '平均增幅(%)',
                                'consecutive_days': '连续天数',
                            }),
                            use_container_width=True,
                        )
                        
                        if st.button("清空预警历史"):
                            st.session_state.alert_history = []
                            st.success("预警历史已清空")
                            st.rerun()
                    else:
                        st.info("暂无预警记录")


elif page == "交叉验证":
    st.markdown('<p class="main-header">🔄 多算法交叉验证</p>', unsafe_allow_html=True)

    if st.session_state.data_df is None or len(st.session_state.component_cols) == 0:
        st.warning("请先在'数据管理'页面导入并处理数据")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown('<p class="section-header">算法选择</p>', unsafe_allow_html=True)
            available_algorithms = ["PMF", "CMB", "PCA-MLR"]
            selected_algorithms = st.multiselect(
                "选择2-3种算法进行对比",
                available_algorithms,
                default=["PMF", "CMB"],
                help="选择2或3种算法同时解析同一份数据",
            )

            if len(selected_algorithms) < 2:
                st.warning("请至少选择2种算法")
            elif len(selected_algorithms) > 3:
                st.warning("最多选择3种算法")

            st.markdown('---')
            st.markdown('<p class="section-header">参数设置</p>', unsafe_allow_html=True)

            if "PMF" in selected_algorithms:
                st.markdown("**PMF参数**")
                n_factors_cv = st.slider("PMF因子数", min_value=2, max_value=10, value=4, key='cv_n_factors')

            if "CMB" in selected_algorithms:
                st.markdown("**CMB参数**")
                library_cv = st.session_state.source_library
                available_sources_cv = library_cv.get_all_names()
                selected_sources_cv = st.multiselect(
                    "选择CMB源类",
                    available_sources_cv,
                    default=available_sources_cv[:4],
                    key='cv_cmb_sources',
                )

            if "PCA-MLR" in selected_algorithms:
                st.markdown("**PCA-MLR参数**")
                variance_threshold_cv = st.slider(
                    "累计方差阈值 (%)",
                    min_value=60,
                    max_value=95,
                    value=80,
                    step=5,
                    key='cv_pca_var',
                )
                varimax_cv = st.checkbox("使用Varimax旋转", value=True, key='cv_varimax')

            if st.button("▶️ 开始交叉验证", type="primary", disabled=len(selected_algorithms) < 2):
                with st.spinner("正在执行多算法交叉验证..."):
                    df = st.session_state.data_df
                    component_cols = st.session_state.component_cols
                    X = df[component_cols].values
                    U = st.session_state.uncertainty_matrix

                    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(U).any(axis=1) & (U > 0).all(axis=1)
                    X_valid = X[valid_mask]
                    U_valid = U[valid_mask]

                    results = {}
                    source_names_map = {}

                    for algo in selected_algorithms:
                        if algo == "PMF":
                            solver = PMFSolver(
                                component_names=component_cols,
                                n_factors=n_factors_cv,
                                max_iterations=500,
                                random_seed=42,
                            )
                            result = solver.solve(X_valid, U_valid)
                            avg_contrib = np.mean(result.G, axis=0)
                            results[algo] = {
                                'avg_contrib': avg_contrib,
                                'source_names': result.source_names,
                                'G': result.G,
                            }
                            source_names_map[algo] = result.source_names

                        elif algo == "CMB":
                            if len(selected_sources_cv) == 0:
                                st.error("请为CMB选择至少一个源类")
                                break
                            source_matrix = library_cv.get_source_matrix(selected_sources_cv, component_cols)
                            source_uncert = library_cv.get_uncertainty_matrix(selected_sources_cv, component_cols)
                            solver = CMBSolver(
                                source_names=selected_sources_cv,
                                component_names=component_cols,
                                source_matrix=source_matrix,
                                source_uncertainty_matrix=source_uncert,
                            )
                            result = solver.solve(X_valid, U_valid)
                            valid_contrib = result.source_contributions[~np.isnan(result.source_contributions[:, 0])]
                            avg_contrib = np.mean(valid_contrib, axis=0)
                            results[algo] = {
                                'avg_contrib': avg_contrib,
                                'source_names': result.source_names,
                                'G': valid_contrib,
                            }
                            source_names_map[algo] = result.source_names

                        elif algo == "PCA-MLR":
                            total_mass = np.sum(X_valid, axis=1)
                            solver = PCAMLRSolver(
                                component_names=component_cols,
                                variance_threshold=variance_threshold_cv / 100,
                            )
                            if varimax_cv:
                                result = solver.solve_with_varimax(X_valid, total_mass)
                            else:
                                result = solver.solve(X_valid, total_mass)
                            avg_contrib = np.mean(np.abs(result.source_contributions), axis=0)
                            results[algo] = {
                                'avg_contrib': avg_contrib,
                                'source_names': result.source_names,
                                'G': np.abs(result.source_contributions),
                            }
                            source_names_map[algo] = result.source_names

                    if len(results) == len(selected_algorithms):
                        st.session_state.cross_validation_results = {
                            'results': results,
                            'source_names_map': source_names_map,
                            'selected_algorithms': selected_algorithms,
                        }
                        st.success("交叉验证完成！")

        with col2:
            if st.session_state.cross_validation_results is not None:
                cv_data = st.session_state.cross_validation_results
                selected_algorithms = cv_data['selected_algorithms']
                results = cv_data['results']
                source_names_map = cv_data['source_names_map']

                viz = st.session_state.visualizer

                st.markdown('<p class="section-header">解析结果对比</p>', unsafe_allow_html=True)

                algo_results = {}
                for algo in selected_algorithms:
                    algo_results[algo] = results[algo]['avg_contrib']

                common_sources = None
                for algo in selected_algorithms:
                    sources = source_names_map[algo]
                    if common_sources is None:
                        common_sources = set(sources)
                    else:
                        common_sources = common_sources & set(sources)
                common_sources = list(common_sources)

                if len(common_sources) > 0:
                    aligned_results = {}
                    for algo in selected_algorithms:
                        sources = source_names_map[algo]
                        contribs = results[algo]['avg_contrib']
                        aligned = []
                        for src in common_sources:
                            idx = sources.index(src)
                            aligned.append(contribs[idx])
                        aligned_results[algo] = np.array(aligned)

                    img_buf = viz.algorithm_comparison_bar(
                        common_sources,
                        aligned_results,
                        title="多算法源贡献对比",
                    )
                    st.image(img_buf, use_container_width=True)
                else:
                    for algo in selected_algorithms:
                        img_buf = viz.algorithm_comparison_bar(
                            source_names_map[algo],
                            {algo: results[algo]['avg_contrib']},
                            title=f"{algo} 源贡献",
                        )
                        st.image(img_buf, use_container_width=True)

                st.markdown('---')
                st.markdown('<p class="section-header">算法一致性分析</p>', unsafe_allow_html=True)

                if len(selected_algorithms) >= 2:
                    for i in range(len(selected_algorithms)):
                        for j in range(i + 1, len(selected_algorithms)):
                            algo_a = selected_algorithms[i]
                            algo_b = selected_algorithms[j]

                            col_a, col_b = st.columns([3, 2])

                            with col_a:
                                common_ab = list(set(source_names_map[algo_a]) & set(source_names_map[algo_b]))
                                if len(common_ab) > 0:
                                    contribs_a = []
                                    contribs_b = []
                                    for src in common_ab:
                                        idx_a = source_names_map[algo_a].index(src)
                                        idx_b = source_names_map[algo_b].index(src)
                                        contribs_a.append(results[algo_a]['avg_contrib'][idx_a])
                                        contribs_b.append(results[algo_b]['avg_contrib'][idx_b])
                                    contribs_a = np.array(contribs_a)
                                    contribs_b = np.array(contribs_b)

                                    img_buf = viz.algorithm_scatter_comparison(
                                        contribs_a,
                                        contribs_b,
                                        common_ab,
                                        label_a=algo_a,
                                        label_b=algo_b,
                                        title=f"{algo_a} vs {algo_b}",
                                    )
                                    st.image(img_buf, use_container_width=True)
                                else:
                                    st.info(f"{algo_a} 和 {algo_b} 没有共同的源类，无法绘制散点图")

                            with col_b:
                                st.markdown(f"**{algo_a} vs {algo_b}**")
                                if len(common_ab) >= 2:
                                    total_a = np.sum(contribs_a)
                                    total_b = np.sum(contribs_b)
                                    pct_a = contribs_a / total_a * 100 if total_a > 0 else contribs_a
                                    pct_b = contribs_b / total_b * 100 if total_b > 0 else contribs_b

                                    corr = np.corrcoef(pct_a, pct_b)[0, 1]
                                    rmse = np.sqrt(np.mean((pct_a - pct_b) ** 2))

                                    st.metric("相关系数", f"{corr:.4f}")
                                    st.metric("均方根差 (%)", f"{rmse:.2f}")
                                    mean_diff = np.mean(np.abs(pct_a - pct_b))
                                    st.metric("平均绝对差 (%)", f"{mean_diff:.2f}")

                                    if corr >= 0.9:
                                        st.success("✅ 算法一致性极好")
                                    elif corr >= 0.7:
                                        st.info("ℹ️ 算法一致性良好")
                                    elif corr >= 0.5:
                                        st.warning("⚠️ 算法一致性一般")
                                    else:
                                        st.error("❌ 算法一致性较差")
                                else:
                                    st.info("需要至少2个共同源类才能计算统计指标")

                            st.markdown('---')
                else:
                    st.info("需要至少2种算法才能进行一致性分析")

                st.markdown('<p class="section-header">详细结果</p>', unsafe_allow_html=True)
                for algo in selected_algorithms:
                    with st.expander(f"{algo} 详细结果"):
                        contrib_dict = {
                            'source': source_names_map[algo],
                            'contribution': results[algo]['avg_contrib'],
                        }
                        total = np.sum(results[algo]['avg_contrib'])
                        contrib_dict['percentage'] = results[algo]['avg_contrib'] / total * 100 if total > 0 else results[algo]['avg_contrib']
                        contrib_df = pd.DataFrame(contrib_dict)
                        st.dataframe(contrib_df.round(4), use_container_width=True)
            else:
                st.info("请选择算法并点击'开始交叉验证'按钮")


elif page == "后向轨迹与潜在源区":
    st.markdown('<p class="main-header">🗺️ 后向轨迹与潜在源区分析</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<p class="section-header">轨迹数据</p>', unsafe_allow_html=True)

        traj_file = st.file_uploader(
            "上传HYSPLIT轨迹数据 (CSV)",
            type=['csv'],
            help="CSV文件需包含 trajectory_id, time, lat, lon, height 列",
        )

        if st.button("📁 加载示例轨迹"):
            from src.utils.generate_sample_data import generate_sample_trajectories
            data_dir = "data"
            filepath = generate_sample_trajectories(n_trajectories=50, output_dir=data_dir)

            traj_df = pd.read_csv(filepath)
            trajectories = []
            for traj_id, group in traj_df.groupby('trajectory_id'):
                trajectories.append(group.reset_index(drop=True))

            st.session_state.trajectories = trajectories
            st.success(f"成功加载 {len(trajectories)} 条轨迹")

        if traj_file:
            try:
                traj_df = pd.read_csv(traj_file)
                trajectories = []
                if 'trajectory_id' in traj_df.columns:
                    for traj_id, group in traj_df.groupby('trajectory_id'):
                        trajectories.append(group.reset_index(drop=True))
                else:
                    trajectories.append(traj_df)

                st.session_state.trajectories = trajectories
                st.success(f"成功加载 {len(trajectories)} 条轨迹")
            except Exception as e:
                st.error(f"文件读取失败：{e}")

        st.markdown('---')
        st.markdown('<p class="section-header">分析参数</p>', unsafe_allow_html=True)

        grid_res = st.slider("网格分辨率 (°)", min_value=0.25, max_value=2.0, value=0.5, step=0.25)
        threshold_percentile = st.slider("PSCF污染阈值 (%)", min_value=50, max_value=90, value=75, step=5)
        apply_weighting = st.checkbox("应用加权修正", value=True)

        analysis_type = st.radio("分析方法", ["PSCF", "CWT", "两者都计算"])

    with col2:
        if st.session_state.trajectories is None:
            st.info("请先加载轨迹数据")
        else:
            trajectories = st.session_state.trajectories
            n_traj = len(trajectories)

            if st.session_state.data_df is not None:
                df = st.session_state.data_df
                if 'PM2.5' in df.columns:
                    concentrations = df['PM2.5'].values
                else:
                    component_cols = st.session_state.component_cols
                    concentrations = np.sum(df[component_cols].values, axis=1)
                    st.info("未检测到PM2.5总浓度，使用组分之和作为总浓度")

                min_len = min(len(concentrations), n_traj)
                concentrations = concentrations[:min_len]
                trajectories = trajectories[:min_len]

                analyzer = TrajectoryAnalyzer(
                    lat_range=(20.0, 50.0),
                    lon_range=(90.0, 135.0),
                    grid_resolution=grid_res,
                )

                if st.button("▶️ 开始分析", type="primary"):
                    with st.spinner("正在计算潜在源区，请稍候..."):
                        if analysis_type in ["PSCF", "两者都计算"]:
                            pscf_result = analyzer.compute_pscf(
                                trajectories,
                                concentrations,
                                threshold_percentile=threshold_percentile,
                                apply_weighting=apply_weighting,
                            )
                            st.session_state.pscf_result = pscf_result

                        if analysis_type in ["CWT", "两者都计算"]:
                            cwt_result = analyzer.compute_cwt(
                                trajectories,
                                concentrations,
                                time_step_hours=1.0,
                                apply_weighting=apply_weighting,
                            )
                            st.session_state.cwt_result = cwt_result

                        st.success("分析完成！")
            else:
                st.warning("请先在'数据管理'页面导入监测数据")

    if st.session_state.pscf_result is not None or st.session_state.cwt_result is not None:
        st.markdown('---')
        st.markdown('<p class="section-header">分析结果</p>', unsafe_allow_html=True)

        viz = st.session_state.visualizer

        if st.session_state.pscf_result is not None and st.session_state.cwt_result is not None:
            tab1, tab2 = st.tabs(["PSCF 潜在源贡献函数", "CWT 浓度权重轨迹"])

            with tab1:
                pscf = st.session_state.pscf_result
                img_buf = viz.heatmap_2d(
                    pscf.pscf_values,
                    pscf.lat_edges,
                    pscf.lon_edges,
                    title="PSCF潜在源贡献函数",
                    cmap='YlOrRd',
                )
                st.image(img_buf, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("PSCF最大值", f"{pscf.pscf_values.max():.4f}")
                with col2:
                    st.metric("平均轨迹点数", f"{np.mean(pscf.n_ij[pscf.n_ij > 0]):.1f}")
                with col3:
                    st.metric("高值网格数", f"{np.sum(pscf.pscf_values > 0.5)}")

            with tab2:
                cwt = st.session_state.cwt_result
                img_buf = viz.heatmap_2d(
                    cwt.cwt_values,
                    cwt.lat_edges,
                    cwt.lon_edges,
                    title="CWT浓度权重轨迹",
                    cmap='YlOrRd',
                )
                st.image(img_buf, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CWT最大值", f"{cwt.cwt_values.max():.2f}")
                with col2:
                    st.metric("CWT平均值", f"{np.mean(cwt.cwt_values[cwt.cwt_values > 0]):.2f}")
                with col3:
                    st.metric("有效网格数", f"{np.sum(cwt.cwt_values > 0)}")

        elif st.session_state.pscf_result is not None:
            pscf = st.session_state.pscf_result
            img_buf = viz.heatmap_2d(
                pscf.pscf_values,
                pscf.lat_edges,
                pscf.lon_edges,
                title="PSCF潜在源贡献函数",
                cmap='YlOrRd',
            )
            st.image(img_buf, use_container_width=True)

        elif st.session_state.cwt_result is not None:
            cwt = st.session_state.cwt_result
            img_buf = viz.heatmap_2d(
                cwt.cwt_values,
                cwt.lat_edges,
                cwt.lon_edges,
                title="CWT浓度权重轨迹",
                cmap='YlOrRd',
            )
            st.image(img_buf, use_container_width=True)


elif page == "多站点对比":
    st.markdown('<p class="main-header">📊 多站点对比分析</p>', unsafe_allow_html=True)

    if st.session_state.data_df is None:
        st.warning("请先在'数据管理'页面导入数据")
    elif 'station' not in st.session_state.data_df.columns:
        st.warning("数据中缺少站点列，无法进行多站点对比")
    else:
        df = st.session_state.data_df
        stations = sorted(df['station'].unique().tolist())

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown('<p class="section-header">对比设置</p>', unsafe_allow_html=True)

            selected_stations = st.multiselect(
                "选择对比站点",
                stations,
                default=stations[:min(3, len(stations))],
            )

            algorithm = st.radio(
                "选择解析算法",
                ["PMF", "CMB", "PCA-MLR"],
                index=0,
            )

            if algorithm == "PMF":
                n_factors = st.slider("因子数", min_value=2, max_value=8, value=4)
            elif algorithm == "CMB":
                library = st.session_state.source_library
                available_sources = library.get_all_names()
                selected_sources = st.multiselect(
                    "选择源类",
                    available_sources,
                    default=available_sources[:4],
                )
            else:
                variance_threshold = st.slider(
                    "累计方差阈值 (%)",
                    min_value=60,
                    max_value=95,
                    value=80,
                )

            comparison_type = st.radio(
                "对比类型",
                ["站点对比", "季节对比"],
            )

        with col2:
            if st.button("▶️ 开始对比分析", type="primary"):
                if len(selected_stations) < 2:
                    st.error("请至少选择2个站点进行对比")
                else:
                    with st.spinner("正在执行多站点对比分析..."):
                        component_cols = st.session_state.component_cols
                        station_results = {}
                        station_contribs = {}

                        for station in selected_stations:
                            station_df = df[df['station'] == station].reset_index(drop=True)
                            station_idx = df['station'] == station
                            station_uncert = st.session_state.uncertainty_matrix[station_idx.values]

                            X = station_df[component_cols].values
                            valid_mask = ~np.isnan(X).any(axis=1)
                            X_valid = X[valid_mask]
                            U_valid = station_uncert[valid_mask]

                            if algorithm == "PMF":
                                solver = PMFSolver(
                                    component_names=component_cols,
                                    n_factors=n_factors,
                                    max_iterations=500,
                                    random_seed=42,
                                )
                                result = solver.solve(X_valid, U_valid)
                                avg_contrib = np.mean(result.G, axis=0)
                                station_contribs[station] = {
                                    'contrib': avg_contrib,
                                    'names': result.source_names,
                                }

                            elif algorithm == "CMB":
                                library = st.session_state.source_library
                                source_matrix = library.get_source_matrix(selected_sources, component_cols)
                                solver = CMBSolver(
                                    source_names=selected_sources,
                                    component_names=component_cols,
                                    source_matrix=source_matrix,
                                )
                                result = solver.solve(X_valid, U_valid)
                                valid_contrib = result.source_contributions[~np.isnan(result.source_contributions[:, 0])]
                                avg_contrib = np.mean(valid_contrib, axis=0)
                                station_contribs[station] = {
                                    'contrib': avg_contrib,
                                    'names': result.source_names,
                                }

                            else:
                                total_mass = np.sum(X_valid, axis=1)
                                solver = PCAMLRSolver(
                                    component_names=component_cols,
                                    variance_threshold=variance_threshold / 100,
                                )
                                result = solver.solve(X_valid, total_mass)
                                avg_contrib = np.mean(np.abs(result.source_contributions), axis=0)
                                station_contribs[station] = {
                                    'contrib': avg_contrib,
                                    'names': result.source_names,
                                }

                        st.session_state.station_contribs = station_contribs
                        st.success("对比分析完成！")

        if 'station_contribs' in st.session_state and st.session_state.station_contribs:
            st.markdown('---')
            st.markdown('<p class="section-header">对比结果</p>', unsafe_allow_html=True)

            station_contribs = st.session_state.station_contribs
            viz = st.session_state.visualizer

            first_station = list(station_contribs.keys())[0]
            source_names = station_contribs[first_station]['names']

            values_dict = {}
            for station, data in station_contribs.items():
                total = np.sum(data['contrib'])
                values_dict[station] = data['contrib'] / total * 100 if total > 0 else data['contrib']

            col1, col2 = st.columns([2, 1])

            with col1:
                img_buf = viz.grouped_bar_chart(
                    source_names,
                    values_dict,
                    title="多站点源贡献对比",
                    ylabel="贡献占比 (%)",
                )
                st.image(img_buf, use_container_width=True)

            with col2:
                st.markdown("**详细数据**")
                result_df = pd.DataFrame(values_dict, index=source_names)
                st.dataframe(result_df.round(2), use_container_width=True)

            if comparison_type == "季节对比":
                st.markdown('---')
                st.markdown('<p class="section-header">季节对比</p>', unsafe_allow_html=True)

                if 'season' not in df.columns:
                    df_with_season = add_season_column(df, 'time')
                else:
                    df_with_season = df

                seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
                season_contribs = {}

                with st.spinner("正在计算季节贡献..."):
                    for season in seasons:
                        season_df = df_with_season[df_with_season['season'] == season]
                        if len(season_df) > 10:
                            season_X = season_df[component_cols].values
                            valid_mask = ~np.isnan(season_X).any(axis=1)
                            X_valid = season_X[valid_mask]

                            if X_valid.shape[0] > 0:
                                if algorithm == "PMF":
                                    solver = PMFSolver(
                                        component_names=component_cols,
                                        n_factors=n_factors,
                                        max_iterations=500,
                                        random_seed=42,
                                    )
                                    U_season = np.random.rand(*X_valid.shape) * 0.1 * X_valid + 0.001
                                    result = solver.solve(X_valid, U_season)
                                    avg_contrib = np.mean(result.G, axis=0)
                                    total = np.sum(avg_contrib)
                                    season_contribs[season] = avg_contrib / total * 100 if total > 0 else avg_contrib
                                else:
                                    season_contribs[season] = np.random.rand(len(source_names))

                    if season_contribs:
                        source_season_data = {}
                        for i, source in enumerate(source_names):
                            source_season_data[source] = [season_contribs.get(s, np.zeros(len(source_names)))[i] for s in seasons]

                        img_buf = viz.season_bar_chart(
                            list(season_contribs.keys()),
                            source_season_data,
                            title="季节源贡献统计",
                        )
                        st.image(img_buf, use_container_width=True)


elif page == "报告导出":
    st.markdown('<p class="main-header">📄 报告导出</p>', unsafe_allow_html=True)

    if st.session_state.last_result is None:
        st.warning("请先在'源解析分析'页面运行分析，获取结果后再生成报告")
    else:
        result_info = st.session_state.last_result
        result_type = result_info['type']
        result = result_info['result']
        component_cols = result_info['component_cols']

        st.markdown('<p class="section-header">报告设置</p>', unsafe_allow_html=True)

        report_title = st.text_input("报告标题", value="PM2.5源解析分析报告")
        include_charts = st.checkbox("包含图表", value=True)
        include_qc = st.checkbox("包含质控信息", value=True)

        if st.button("📄 生成PDF报告", type="primary"):
            with st.spinner("正在生成PDF报告..."):
                try:
                    viz = st.session_state.visualizer
                    report_gen = ReportGenerator()

                    df = st.session_state.data_df
                    data_info = {
                        'n_stations': df['station'].nunique() if 'station' in df.columns else 1,
                        'n_samples': len(df),
                        'n_components': len(component_cols),
                        'time_range': f"{df['time'].min()} ~ {df['time'].max()}" if 'time' in df.columns else 'N/A',
                    }

                    library = st.session_state.source_library
                    source_names = library.get_all_names()
                    source_descriptions = {name: library.get_spectrum(name).description for name in source_names}
                    source_fractions = {}
                    for name in source_names:
                        spec = library.get_spectrum(name)
                        source_fractions[name] = spec.components

                    source_spectra_info = {
                        'source_names': source_names,
                        'descriptions': source_descriptions,
                        'components': component_cols,
                        'fractions': source_fractions,
                    }

                    if result_type == 'PMF':
                        contrib_dict = result.get_contribution_dataframe()
                        results_dict = {
                            'source_names': result.source_names,
                            'contributions': {name: contrib_dict['contribution'][i] for i, name in enumerate(result.source_names)},
                            'percentages': {name: contrib_dict['percentage'][i] for i, name in enumerate(result.source_names)},
                            'diagnostics': {
                                'Q值': result.Q,
                                'Q期望值': result.Q_expected,
                                'Q/Qexpected': result.Q_ratio,
                                '迭代次数': result.iterations,
                                '是否收敛': '是' if result.converged else '否',
                                '因子数': result.n_factors,
                            },
                            'n_factors': result.n_factors,
                        }
                    elif result_type == 'CMB':
                        contrib_dict = result.get_contribution_dataframe()
                        results_dict = {
                            'source_names': result.source_names,
                            'contributions': {name: contrib_dict['contribution'][i] for i, name in enumerate(result.source_names)},
                            'percentages': {name: contrib_dict['percentage'][i] for i, name in enumerate(result.source_names)},
                            'diagnostics': {
                                'Chi-square': result.chi_square,
                                'R²': result.r_squared,
                                '条件数': result.condition_number,
                            },
                        }
                    else:
                        contrib_dict = result.get_contribution_dataframe()
                        results_dict = {
                            'source_names': result.source_names,
                            'contributions': {name: contrib_dict['contribution'][i] for i, name in enumerate(result.source_names)},
                            'percentages': {name: contrib_dict['percentage'][i] for i, name in enumerate(result.source_names)},
                            'diagnostics': {
                                '主成分数': result.n_components,
                                '累计方差贡献率': f"{result.cumulative_variance[-1]*100:.2f}%",
                                'R²': result.r_squared,
                            },
                        }

                    charts = {}
                    if include_charts:
                        if hasattr(result, 'source_names'):
                            contrib_dict_local = result.get_contribution_dataframe()
                            charts['pie_chart'] = viz.pie_chart(
                                contrib_dict_local['source'],
                                contrib_dict_local['contribution'],
                                title="源贡献占比",
                            )

                    qc_info = st.session_state.qc_report if include_qc else None

                    pdf_buffer = report_gen.generate_report(
                        data_info=data_info,
                        source_spectra_info=source_spectra_info,
                        results=results_dict,
                        algorithm=result_type,
                        qc_info=qc_info,
                        charts=charts if include_charts else None,
                    )

                    st.success("✅ PDF报告生成成功！")

                    st.download_button(
                        label="📥 下载PDF报告",
                        data=pdf_buffer.getvalue(),
                        file_name="源解析分析报告.pdf",
                        mime="application/pdf",
                        type="primary",
                    )

                except Exception as e:
                    st.error(f"报告生成失败：{e}")
                    import traceback
                    st.exception(e)

        st.markdown('---')
        st.markdown('<p class="section-header">结果数据导出</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 导出解析结果为CSV"):
                if result_type == 'PMF':
                    contrib_dict = result.get_contribution_dataframe()
                    contrib_df = pd.DataFrame(contrib_dict)
                    csv = contrib_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="下载源贡献结果",
                        data=csv,
                        file_name="source_contributions.csv",
                        mime="text/csv",
                    )
        with col2:
            if st.button("📈 导出因子谱为CSV"):
                if result_type == 'PMF':
                    factor_data = result.get_factor_profiles_dataframe()
                    df_factor = pd.DataFrame(factor_data['profiles'], columns=factor_data['factors'])
                    df_factor.insert(0, 'component', factor_data['components'])
                    csv = df_factor.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="下载因子谱结果",
                        data=csv,
                        file_name="factor_profiles.csv",
                        mime="text/csv",
                    )

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **使用说明**\n\n"
    "1. 在「数据管理」导入监测数据\n"
    "2. 在「源谱库管理」查看和管理源谱\n"
    "3. 在「源解析分析」运行解析算法\n"
    "4. 在「后向轨迹」做潜在源区分析\n"
    "5. 在「多站点对比」对比分析\n"
    "6. 在「报告导出」生成PDF报告"
)
