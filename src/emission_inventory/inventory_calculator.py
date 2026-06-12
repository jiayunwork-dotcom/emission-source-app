import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .emission_factors import EmissionFactorLibrary


@dataclass
class IndustryActivityData:
    industry_name: str
    activity_level: float
    activity_unit: str
    control_efficiency: float = 0.0
    factor_params: Dict = field(default_factory=dict)
    emission_factor: float = 0.0
    emission_amount: float = 0.0


@dataclass
class ValidationResult:
    industry_name: str
    bottom_up: float
    top_down: float
    deviation_rate: float
    status: str


class EmissionInventoryCalculator:
    def __init__(self):
        self.factor_library = EmissionFactorLibrary()
        self.industries: Dict[str, IndustryActivityData] = {}
        self.source_contributions: Dict[str, float] = {}
        self._init_default_industries()

    def _init_default_industries(self):
        default_units = {
            "燃煤电厂": "万吨/年",
            "机动车": "万辆",
            "工地扬尘": "万平方米",
            "生物质燃烧": "万吨/年",
            "餐饮油烟": "万元/年",
        }
        default_activities = {
            "燃煤电厂": 500.0,
            "机动车": 100.0,
            "工地扬尘": 500.0,
            "生物质燃烧": 50.0,
            "餐饮油烟": 50000.0,
        }
        for name in self.factor_library.get_all_industries():
            self.industries[name] = IndustryActivityData(
                industry_name=name,
                activity_level=default_activities.get(name, 0.0),
                activity_unit=default_units.get(name, ""),
            )

    def set_activity_level(self, industry_name: str, activity_level: float,
                           control_efficiency: float = 0.0, **factor_params):
        if industry_name in self.industries:
            self.industries[industry_name].activity_level = activity_level
            self.industries[industry_name].control_efficiency = control_efficiency
            self.industries[industry_name].factor_params = factor_params
            self._calculate_industry_emission(industry_name)

    def _calculate_industry_emission(self, industry_name: str):
        if industry_name not in self.industries:
            return

        industry = self.industries[industry_name]
        factor_params = industry.factor_params

        emission_factor = self.factor_library.calculate_emission_factor(
            industry_name, **factor_params
        )

        activity_level = industry.activity_level
        control_efficiency = np.clip(industry.control_efficiency / 100.0, 0.0, 0.99)

        unit_conversions = {
            "燃煤电厂": 10000.0,
            "机动车": 10000.0,
            "工地扬尘": 1.0,
            "生物质燃烧": 10000.0 * 1000.0,
            "餐饮油烟": 1.0,
        }

        factor_conversions = {
            "燃煤电厂": 0.001,
            "机动车": 0.001,
            "工地扬尘": 1.0,
            "生物质燃烧": 0.000001,
            "餐饮油烟": 0.001,
        }

        base_activity = activity_level * unit_conversions.get(industry_name, 1.0)
        emission_amount = (base_activity * emission_factor *
                          (1 - control_efficiency) *
                          factor_conversions.get(industry_name, 1.0))

        industry.emission_factor = emission_factor
        industry.emission_amount = emission_amount

    def calculate_all_emissions(self):
        for name in self.industries:
            self._calculate_industry_emission(name)

    def get_total_emissions(self) -> float:
        return sum(ind.emission_amount for ind in self.industries.values())

    def get_emissions_dataframe(self) -> pd.DataFrame:
        data = []
        for name, industry in self.industries.items():
            unit = self.factor_library.get_unit(name)
            data.append({
                '行业名': name,
                '活动水平': industry.activity_level,
                '活动水平单位': industry.activity_unit,
                '排放因子': round(industry.emission_factor, 6),
                '排放因子单位': unit,
                '控制效率(%)': industry.control_efficiency,
                '排放量(吨/年)': round(industry.emission_amount, 4),
            })
        return pd.DataFrame(data)

    def set_source_contributions(self, contributions: Dict[str, float]):
        self.source_contributions = contributions

    def _convert_to_ton_per_year(self, concentration_ugm3: float) -> float:
        if concentration_ugm3 <= 0:
            return 0.0
        conversion_factor = 1000.0
        return concentration_ugm3 * conversion_factor

    def validate_inventory(self) -> List[ValidationResult]:
        results = []
        for name, industry in self.industries.items():
            bottom_up = industry.emission_amount
            top_down_concentration = self.source_contributions.get(name, 0.0)
            top_down = self._convert_to_ton_per_year(top_down_concentration)

            if top_down > 0:
                deviation_rate = (bottom_up - top_down) / top_down * 100
            elif bottom_up > 0:
                deviation_rate = 100.0
            else:
                deviation_rate = 0.0

            abs_deviation = abs(deviation_rate)
            if abs_deviation > 50:
                status = 'red'
            elif abs_deviation > 30:
                status = 'yellow'
            else:
                status = 'green'

            results.append(ValidationResult(
                industry_name=name,
                bottom_up=bottom_up,
                top_down=top_down,
                deviation_rate=deviation_rate,
                status=status,
            ))

        return results

    def get_validation_dataframe(self) -> pd.DataFrame:
        results = self.validate_inventory()
        data = []
        for r in results:
            data.append({
                '行业名': r.industry_name,
                '自下而上(吨/年)': round(r.bottom_up, 4),
                '自上而下(吨/年)': round(r.top_down, 4),
                '偏差率(%)': round(r.deviation_rate, 2),
                '状态': r.status,
            })
        return pd.DataFrame(data)

    def get_emission_amount(self, industry_name: str) -> float:
        if industry_name in self.industries:
            return self.industries[industry_name].emission_amount
        return 0.0

    def get_activity_data(self, industry_name: str) -> Optional[IndustryActivityData]:
        return self.industries.get(industry_name)
