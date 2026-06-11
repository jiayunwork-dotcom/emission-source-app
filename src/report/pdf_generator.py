import numpy as np
import pandas as pd
from io import BytesIO
from typing import Dict, List, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._define_styles()

    def _define_styles(self):
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=20,
            alignment=1,
        )
        self.heading1_style = ParagraphStyle(
            'CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.darkblue,
        )
        self.heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=8,
            textColor=colors.darkslategray,
        )
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            leading=14,
        )
        self.table_header_style = ParagraphStyle(
            'TableHeader',
            parent=self.styles['Normal'],
            fontSize=9,
            alignment=1,
            textColor=colors.white,
            fontName='Helvetica-Bold',
        )

    def _create_table(self, data: List[List], has_header: bool = True) -> Table:
        table = Table(data, repeatRows=1 if has_header else 0)
        style = TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ])
        if has_header:
            style.add('BACKGROUND', (0, 0), (-1, 0), colors.darkblue)
            style.add('TEXTCOLOR', (0, 0), (-1, 0), colors.white)
            style.add('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
        table.setStyle(style)
        return table

    def generate_report(
        self,
        data_info: Dict,
        source_spectra_info: Dict,
        results: Dict,
        algorithm: str = "PMF",
        qc_info: Optional[Dict] = None,
        charts: Optional[Dict[str, BytesIO]] = None,
    ) -> BytesIO:
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        story = []
        story.append(Paragraph("PM2.5源解析分析报告", self.title_style))
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(f"解析方法: {algorithm}", self.normal_style))
        story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph("一、数据概况", self.heading1_style))
        story = self._add_data_overview(story, data_info, qc_info)

        story.append(PageBreak())

        story.append(Paragraph("二、源谱信息", self.heading1_style))
        story = self._add_source_spectra(story, source_spectra_info)

        story.append(PageBreak())

        story.append(Paragraph("三、解析结果", self.heading1_style))
        story = self._add_results(story, results, algorithm)

        if charts:
            story.append(PageBreak())
            story.append(Paragraph("四、核心图表", self.heading1_style))
            story = self._add_charts(story, charts)

        story.append(PageBreak())
        story.append(Paragraph("五、算法参数与质控诊断", self.heading1_style))
        story = self._add_qc_diagnosis(story, results, qc_info, algorithm)

        doc.build(story)
        buffer.seek(0)
        return buffer

    def _add_data_overview(self, story: List, data_info: Dict, qc_info: Optional[Dict]) -> List:
        story.append(Paragraph("1. 数据基本信息", self.heading2_style))

        data = [
            ['项目', '数值'],
            ['站点数量', str(data_info.get('n_stations', 'N/A'))],
            ['总样本数', str(data_info.get('n_samples', 'N/A'))],
            ['组分数', str(data_info.get('n_components', 'N/A'))],
            ['时间范围', str(data_info.get('time_range', 'N/A'))],
        ]
        story.append(self._create_table(data))
        story.append(Spacer(1, 0.5 * cm))

        if qc_info:
            story.append(Paragraph("2. 数据质量控制", self.heading2_style))
            qc_data = [
                ['质控指标', '结果'],
                ['原始样本数', str(qc_info.get('total_samples', 'N/A'))],
                ['有效样本数', str(qc_info.get('valid_samples_after_qc', 'N/A'))],
                ['缺失值标记', '已完成'],
                ['负值剔除', '已完成'],
                ['异常高值标记', '已完成'],
            ]
            story.append(self._create_table(qc_data))

        return story

    def _add_source_spectra(self, story: List, source_spectra_info: Dict) -> List:
        story.append(Paragraph("1. 源类列表", self.heading2_style))

        source_names = source_spectra_info.get('source_names', [])
        source_data = [['序号', '源类名称', '描述']]
        for i, name in enumerate(source_names, 1):
            desc = source_spectra_info.get('descriptions', {}).get(name, '')
            source_data.append([str(i), name, desc])
        story.append(self._create_table(source_data))
        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph("2. 源谱组分质量分数", self.heading2_style))
        components = source_spectra_info.get('components', [])
        if len(components) > 15:
            components_show = components[:15]
            story.append(Paragraph(f"共 {len(components)} 种组分，显示前15种主要组分：", self.normal_style))
        else:
            components_show = components

        spec_data = [['组分'] + source_names]
        for comp in components_show:
            row = [comp]
            for name in source_names:
                frac = source_spectra_info.get('fractions', {}).get(name, {}).get(comp, 0)
                row.append(f"{frac:.4f}")
            spec_data.append(row)
        story.append(self._create_table(spec_data))

        return story

    def _add_results(self, story: List, results: Dict, algorithm: str) -> List:
        story.append(Paragraph("1. 源贡献汇总", self.heading2_style))

        source_names = results.get('source_names', [])
        result_data = [['源类', '平均贡献浓度 (μg/m³)', '占比 (%)']]
        for name in source_names:
            contrib = results.get('contributions', {}).get(name, 0)
            pct = results.get('percentages', {}).get(name, 0)
            result_data.append([name, f"{contrib:.4f}", f"{pct:.2f}"])
        story.append(self._create_table(result_data))
        story.append(Spacer(1, 0.5 * cm))

        if 'diagnostics' in results:
            story.append(Paragraph("2. 诊断指标", self.heading2_style))
            diag = results['diagnostics']
            diag_data = [['指标', '数值']]
            for key, value in diag.items():
                if isinstance(value, float):
                    diag_data.append([key, f"{value:.4f}"])
                else:
                    diag_data.append([key, str(value)])
            story.append(self._create_table(diag_data))

        return story

    def _add_charts(self, story: List, charts: Dict[str, BytesIO]) -> List:
        chart_order = ['pie_chart', 'time_series', 'factor_profile', 'residual_plot']
        chart_titles = {
            'pie_chart': '源贡献占比饼图',
            'time_series': '源贡献时间序列图',
            'factor_profile': '因子谱图',
            'residual_plot': '残差分析图',
            'bootstrap': 'Bootstrap稳定性图',
            'pscf_map': 'PSCF潜在源区图',
            'cwt_map': 'CWT潜在源区图',
        }

        for chart_key in chart_order:
            if chart_key in charts:
                story.append(Paragraph(chart_titles.get(chart_key, chart_key), self.heading2_style))
                img = Image(charts[chart_key], width=15 * cm, height=10 * cm)
                story.append(img)
                story.append(Spacer(1, 0.5 * cm))

        for key, value in charts.items():
            if key not in chart_order:
                story.append(Paragraph(chart_titles.get(key, key), self.heading2_style))
                img = Image(value, width=15 * cm, height=10 * cm)
                story.append(img)
                story.append(Spacer(1, 0.5 * cm))

        return story

    def _add_qc_diagnosis(self, story: List, results: Dict, qc_info: Optional[Dict], algorithm: str) -> List:
        story.append(Paragraph("1. 算法参数", self.heading2_style))

        if algorithm == 'PMF':
            param_data = [
                ['参数', '取值'],
                ['因子数', str(results.get('n_factors', 'N/A'))],
                ['最大迭代次数', '500'],
                ['收敛阈值', '0.01%'],
                ['Bootstrap次数', '100'],
                ['Bootstrap块大小', '7天'],
            ]
        elif algorithm == 'CMB':
            param_data = [
                ['参数', '取值'],
                ['源类数量', str(len(results.get('source_names', [])))],
                ['条件数阈值', '20'],
            ]
        else:
            param_data = [
                ['参数', '取值'],
                ['方差贡献率阈值', '80%'],
            ]
        story.append(self._create_table(param_data))
        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph("2. 不确定度估计规则", self.heading2_style))
        story.append(Paragraph("• 低于检出限的数据点：不确定度 = 检出限 × 5/6", self.normal_style))
        story.append(Paragraph("• 高于检出限的数据点：不确定度 = 浓度 × 0.1 + 检出限/3", self.normal_style))
        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph("3. 质量控制说明", self.heading2_style))
        story.append(Paragraph("• 缺失值标记：检测并标记所有缺失数据点", self.normal_style))
        story.append(Paragraph("• 负值剔除：将负浓度值设为缺失", self.normal_style))
        story.append(Paragraph("• 异常高值标记：超过均值5倍的值标记为异常", self.normal_style))

        return story
