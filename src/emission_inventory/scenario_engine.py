import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .inventory_calculator import EmissionInventoryCalculator, IndustryActivityData


@dataclass
class ReductionMeasure:
    industry_name: str
    measure_type: str
    parameter: str
    value: float
    description: str = ""


@dataclass
class Scenario:
    name: str
    description: str = ""
    measures: List[ReductionMeasure] = field(default_factory=list)
    emission_reductions: Dict[str, float] = field(default_factory=dict)
    expected_concentration: float = 0.0
    reduction_percentage: float = 0.0

    def add_measure(self, measure: ReductionMeasure):
        self.measures.append(measure)

    def get_measures_description(self) -> str:
        descriptions = []
        for m in self.measures:
            desc = f"{m.industry_name}: {m.description}"
            descriptions.append(desc)
        return "; ".join(descriptions)


class ScenarioSimulationEngine:
    def __init__(self, inventory_calculator: EmissionInventoryCalculator):
        self.inventory = inventory_calculator
        self.scenarios: Dict[str, Scenario] = {}
        self.current_pm25_concentration: float = 35.0
        self.baseline_emissions: Dict[str, float] = {}

    def set_current_pm25(self, concentration: float):
        self.current_pm25_concentration = concentration

    def update_baseline(self):
        self.inventory.calculate_all_emissions()
        self.baseline_emissions = {
            name: self.inventory.get_emission_amount(name)
            for name in self.inventory.factor_library.get_all_industries()
        }

    def create_scenario(self, name: str, description: str = "") -> Scenario:
        scenario = Scenario(name=name, description=description)
        self.scenarios[name] = scenario
        return scenario

    def add_measure_to_scenario(self, scenario_name: str, measure: ReductionMeasure):
        if scenario_name in self.scenarios:
            self.scenarios[scenario_name].add_measure(measure)

    def _apply_measure_to_activity(self, activity: IndustryActivityData,
                                   measure: ReductionMeasure) -> IndustryActivityData:
        new_activity = IndustryActivityData(
            industry_name=activity.industry_name,
            activity_level=activity.activity_level,
            activity_unit=activity.activity_unit,
            control_efficiency=activity.control_efficiency,
            factor_params=activity.factor_params.copy(),
        )

        if measure.measure_type == "activity_reduction":
            reduction_ratio = measure.value / 100.0
            new_activity.activity_level = activity.activity_level * (1 - reduction_ratio)

        elif measure.measure_type == "control_efficiency":
            new_activity.control_efficiency = measure.value

        elif measure.measure_type == "factor_param":
            new_activity.factor_params[measure.parameter] = measure.value

        return new_activity

    def _calculate_scenario_emissions(self, scenario: Scenario) -> Dict[str, float]:
        scenario_emissions = {}
        for name, base_activity in self.inventory.industries.items():
            current_activity = base_activity
            for measure in scenario.measures:
                if measure.industry_name == name:
                    current_activity = self._apply_measure_to_activity(current_activity, measure)

            factor_params = current_activity.factor_params
            emission_factor = self.inventory.factor_library.calculate_emission_factor(
                name, **factor_params
            )

            activity_level = current_activity.activity_level
            control_efficiency = np.clip(current_activity.control_efficiency / 100.0, 0.0, 0.99)

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

            base_activity_val = activity_level * unit_conversions.get(name, 1.0)
            emission_amount = (base_activity_val * emission_factor *
                              (1 - control_efficiency) *
                              factor_conversions.get(name, 1.0))

            scenario_emissions[name] = emission_amount

        return scenario_emissions

    def _saturation_correction(self, theoretical_reduction_ratio: float) -> float:
        max_reduction_ratio = 0.85
        if theoretical_reduction_ratio <= 0:
            return 0.0
        actual_ratio = max_reduction_ratio * (1 - np.exp(-theoretical_reduction_ratio / 0.3))
        return actual_ratio

    def simulate_scenario(self, scenario_name: str) -> Optional[Scenario]:
        if scenario_name not in self.scenarios:
            return None

        scenario = self.scenarios[scenario_name]
        self.update_baseline()

        baseline_total = sum(self.baseline_emissions.values())
        if baseline_total <= 0:
            return scenario

        scenario_emissions = self._calculate_scenario_emissions(scenario)

        emission_reductions = {}
        for name in self.baseline_emissions:
            baseline = self.baseline_emissions[name]
            scenario_val = scenario_emissions.get(name, baseline)
            reduction = baseline - scenario_val
            emission_reductions[name] = reduction

        scenario.emission_reductions = emission_reductions

        industry_reduction_ratios = []
        for name in self.baseline_emissions:
            baseline = self.baseline_emissions[name]
            if baseline > 0:
                reduction = emission_reductions.get(name, 0.0)
                ratio = reduction / baseline
                industry_reduction_ratios.append(ratio)

        total_theoretical_ratio = sum(industry_reduction_ratios)

        actual_reduction_ratio = self._saturation_correction(total_theoretical_ratio)
        actual_reduction = actual_reduction_ratio * self.current_pm25_concentration

        scenario.expected_concentration = self.current_pm25_concentration - actual_reduction
        scenario.reduction_percentage = (actual_reduction_ratio * 100
                                         if self.current_pm25_concentration > 0 else 0)

        return scenario

    def simulate_all_scenarios(self) -> List[Scenario]:
        results = []
        for name in self.scenarios:
            result = self.simulate_scenario(name)
            if result:
                results.append(result)
        return results

    def get_scenario_results_dataframe(self) -> pd.DataFrame:
        data = []
        for scenario in self.scenarios.values():
            self.simulate_scenario(scenario.name)
            data.append({
                '情景名': scenario.name,
                '描述': scenario.description,
                '减排措施': scenario.get_measures_description(),
                '预期PM2.5浓度(μg/m³)': round(scenario.expected_concentration, 2),
                '削减幅度(%)': round(scenario.reduction_percentage, 2),
            })
        return pd.DataFrame(data)

    def get_comparison_data(self) -> Dict[str, Dict[str, float]]:
        comparison = {}
        for scenario in self.scenarios.values():
            self.simulate_scenario(scenario.name)
            comparison[scenario.name] = {
                'current': self.current_pm25_concentration,
                'expected': scenario.expected_concentration,
                'reduction': self.current_pm25_concentration - scenario.expected_concentration,
            }
        return comparison

    def delete_scenario(self, scenario_name: str) -> bool:
        if scenario_name in self.scenarios:
            del self.scenarios[scenario_name]
            return True
        return False

    def get_scenario(self, scenario_name: str) -> Optional[Scenario]:
        return self.scenarios.get(scenario_name)

    def _saturation_correction_increase(self, theoretical_increase_ratio: float) -> float:
        max_increase_ratio = 2.0
        if theoretical_increase_ratio <= 0:
            return 0.0
        actual_ratio = max_increase_ratio * (1 - np.exp(-theoretical_increase_ratio / 0.3))
        return actual_ratio

    def simulate_single_perturbation(
        self,
        industry_name: str,
        param_type: str,
        perturbation_pct: float,
    ) -> Dict:
        self.update_baseline()
        baseline_total = sum(self.baseline_emissions.values())
        if baseline_total <= 0:
            return None

        scenario_emissions = {}
        for name, base_activity in self.inventory.industries.items():
            current_activity = IndustryActivityData(
                industry_name=base_activity.industry_name,
                activity_level=base_activity.activity_level,
                activity_unit=base_activity.activity_unit,
                control_efficiency=base_activity.control_efficiency,
                factor_params=base_activity.factor_params.copy(),
            )

            if name == industry_name:
                if param_type == "activity_level":
                    current_activity.activity_level = base_activity.activity_level * (1 + perturbation_pct / 100.0)
                elif param_type == "control_efficiency":
                    base_eff = base_activity.control_efficiency
                    eff_change = perturbation_pct / 100.0 * 99.0
                    new_eff = base_eff + eff_change
                    current_activity.control_efficiency = np.clip(new_eff, 0.0, 99.0)

            factor_params = current_activity.factor_params
            emission_factor = self.inventory.factor_library.calculate_emission_factor(
                name, **factor_params
            )

            activity_level = current_activity.activity_level
            control_efficiency = np.clip(current_activity.control_efficiency / 100.0, 0.0, 0.99)

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

            base_activity_val = activity_level * unit_conversions.get(name, 1.0)
            emission_amount = (base_activity_val * emission_factor *
                              (1 - control_efficiency) *
                              factor_conversions.get(name, 1.0))

            scenario_emissions[name] = emission_amount

        baseline_industry_emission = self.baseline_emissions.get(industry_name, 0.0)
        new_industry_emission = scenario_emissions.get(industry_name, baseline_industry_emission)
        industry_emission_change = new_industry_emission - baseline_industry_emission

        total_new_emissions = sum(scenario_emissions.values())
        total_reduction = baseline_total - total_new_emissions

        if baseline_total > 0:
            theoretical_reduction_ratio = total_reduction / baseline_total
        else:
            theoretical_reduction_ratio = 0.0

        if theoretical_reduction_ratio >= 0:
            actual_reduction_ratio = self._saturation_correction(theoretical_reduction_ratio)
            actual_concentration_change = -actual_reduction_ratio * self.current_pm25_concentration
        else:
            theoretical_increase_ratio = -theoretical_reduction_ratio
            actual_increase_ratio = self._saturation_correction_increase(theoretical_increase_ratio)
            actual_concentration_change = actual_increase_ratio * self.current_pm25_concentration

        expected_concentration = self.current_pm25_concentration + actual_concentration_change
        concentration_change_pct = (expected_concentration - self.current_pm25_concentration) / self.current_pm25_concentration * 100 if self.current_pm25_concentration > 0 else 0

        return {
            'perturbation_pct': perturbation_pct,
            'industry_emission': new_industry_emission,
            'industry_emission_change': industry_emission_change,
            'expected_pm25': expected_concentration,
            'concentration_change_pct': concentration_change_pct,
        }

    def run_sensitivity_analysis(
        self,
        analysis_configs: List[Dict],
        perturbation_min: float = -50.0,
        perturbation_max: float = 50.0,
        perturbation_step: float = 10.0,
    ) -> Dict:
        perturbation_points = np.arange(perturbation_min, perturbation_max + perturbation_step / 2, perturbation_step).tolist()
        perturbation_points = [round(p, 2) for p in perturbation_points]

        results = {}
        baseline_pm25 = self.current_pm25_concentration

        for config in analysis_configs:
            industry_name = config['industry_name']
            param_type = config['param_type']
            param_label = config.get('label', f"{industry_name}-{param_type}")

            curve_results = []
            for pct in perturbation_points:
                single_result = self.simulate_single_perturbation(industry_name, param_type, pct)
                if single_result:
                    curve_results.append(single_result)

            results[param_label] = {
                'industry_name': industry_name,
                'param_type': param_type,
                'perturbation_points': perturbation_points,
                'curve_results': curve_results,
            }

        tornado_data = []
        for param_label, data in results.items():
            curve_results = data['curve_results']
            if len(curve_results) >= 2:
                change_pcts = [r['concentration_change_pct'] for r in curve_results]
                impact_magnitude = abs(max(change_pcts) - min(change_pcts))
                avg_slope = 0
                valid_slopes = []
                for i in range(1, len(curve_results)):
                    dx = curve_results[i]['perturbation_pct'] - curve_results[i-1]['perturbation_pct']
                    dy = curve_results[i]['concentration_change_pct'] - curve_results[i-1]['concentration_change_pct']
                    if dx != 0:
                        valid_slopes.append(dy / dx)
                if valid_slopes:
                    avg_slope = np.mean(valid_slopes)

                tornado_data.append({
                    'label': param_label,
                    'impact_magnitude': impact_magnitude,
                    'avg_slope_per_10pct': avg_slope * 10,
                    'industry_name': data['industry_name'],
                    'param_type': data['param_type'],
                })

        tornado_data.sort(key=lambda x: x['impact_magnitude'], reverse=True)

        return {
            'perturbation_points': perturbation_points,
            'baseline_pm25': baseline_pm25,
            'curves': results,
            'tornado_data': tornado_data,
        }
