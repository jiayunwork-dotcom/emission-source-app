import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class IndustryEmissionFactor:
    name: str
    description: str
    unit: str
    base_factor: float
    control_params: Dict = field(default_factory=dict)
    subcategories: Optional[List[str]] = None
    sub_factors: Optional[Dict[str, float]] = None

    def calculate_factor(self, **kwargs) -> float:
        if self.name == "燃煤电厂":
            desulfurization_efficiency = kwargs.get('desulfurization_efficiency', 0.0)
            efficiency = np.clip(desulfurization_efficiency / 100.0, 0.0, 0.99)
            factor = 12.0 * np.exp(-4.6 * efficiency)
            return max(factor, 0.12)

        elif self.name == "工地扬尘":
            coverage_rate = kwargs.get('coverage_rate', 0.0)
            coverage = np.clip(coverage_rate / 100.0, 0.0, 1.0)
            factor = 0.5 * (1 - coverage) + 0.01 * coverage
            return max(factor, 0.01)

        elif self.name == "餐饮油烟":
            purifier_efficiency = kwargs.get('purifier_efficiency', 0.0)
            efficiency = np.clip(purifier_efficiency / 100.0, 0.0, 0.95)
            factor = 0.24 * (1 - efficiency) + 0.012 * efficiency
            return max(factor, 0.012)

        elif self.name == "机动车":
            standard = kwargs.get('emission_standard', '国III')
            annual_vkm = kwargs.get('annual_vkm', 15000)
            factor_per_km = self.sub_factors.get(standard, 0.08)
            return factor_per_km * annual_vkm / 1000.0

        elif self.name == "生物质燃烧":
            crop_type = kwargs.get('crop_type', '稻草')
            return self.sub_factors.get(crop_type, 8.3)

        return self.base_factor


class EmissionFactorLibrary:
    def __init__(self):
        self.factors: Dict[str, IndustryEmissionFactor] = {}
        self._init_default_factors()

    def _init_default_factors(self):
        self.factors["燃煤电厂"] = IndustryEmissionFactor(
            name="燃煤电厂",
            description="电力及供热用煤燃烧排放",
            unit="kg/吨煤",
            base_factor=12.0,
            control_params={'desulfurization_efficiency': '脱硫效率 (%)'},
        )

        self.factors["机动车"] = IndustryEmissionFactor(
            name="机动车",
            description="道路移动源排放",
            unit="g/km",
            base_factor=0.08,
            control_params={'emission_standard': '排放标准', 'annual_vkm': '年均行驶里程(公里)'},
            subcategories=['国III', '国IV', '国V', '国VI'],
            sub_factors={'国III': 0.08, '国IV': 0.05, '国V': 0.03, '国VI': 0.015},
        )

        self.factors["工地扬尘"] = IndustryEmissionFactor(
            name="工地扬尘",
            description="建筑施工场地扬尘排放",
            unit="t/万平方米/年",
            base_factor=0.5,
            control_params={'coverage_rate': '覆盖率 (%)'},
        )

        self.factors["生物质燃烧"] = IndustryEmissionFactor(
            name="生物质燃烧",
            description="农作物秸秆露天焚烧排放",
            unit="g/kg",
            base_factor=8.3,
            control_params={'crop_type': '秸秆类型'},
            subcategories=['稻草', '麦秆', '玉米秸秆'],
            sub_factors={'稻草': 8.3, '麦秆': 7.2, '玉米秸秆': 6.8},
        )

        self.factors["餐饮油烟"] = IndustryEmissionFactor(
            name="餐饮油烟",
            description="餐饮服务业油烟排放",
            unit="kg/万元营业额",
            base_factor=0.24,
            control_params={'purifier_efficiency': '油烟净化器效率 (%)'},
        )

    def get_factor(self, industry_name: str) -> Optional[IndustryEmissionFactor]:
        return self.factors.get(industry_name)

    def get_all_industries(self) -> List[str]:
        return list(self.factors.keys())

    def calculate_emission_factor(self, industry_name: str, **kwargs) -> float:
        factor = self.get_factor(industry_name)
        if factor:
            return factor.calculate_factor(**kwargs)
        return 0.0

    def get_unit(self, industry_name: str) -> str:
        factor = self.get_factor(industry_name)
        return factor.unit if factor else ""
