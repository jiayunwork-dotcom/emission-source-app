import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import copy


class SourceSpectrum:
    def __init__(
        self,
        name: str,
        components: Dict[str, float],
        uncertainties: Optional[Dict[str, float]] = None,
        description: str = "",
    ):
        self.name = name
        self.components = components.copy()
        self.description = description
        if uncertainties is None:
            self.uncertainties = {k: v * 0.1 for k, v in components.items()}
        else:
            self.uncertainties = uncertainties.copy()

    def get_component_names(self) -> List[str]:
        return list(self.components.keys())

    def get_fractions(self, component_names: List[str]) -> np.ndarray:
        result = []
        for name in component_names:
            result.append(self.components.get(name, 0.0))
        return np.array(result)

    def get_uncertainties(self, component_names: List[str]) -> np.ndarray:
        result = []
        for name in component_names:
            result.append(self.uncertainties.get(name, self.components.get(name, 0.0) * 0.1))
        return np.array(result)

    def to_dataframe(self) -> pd.DataFrame:
        data = {
            'component': list(self.components.keys()),
            'fraction': list(self.components.values()),
            'uncertainty': [self.uncertainties.get(k, v * 0.1) for k, v in self.components.items()],
        }
        return pd.DataFrame(data)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str = "", description: str = "") -> 'SourceSpectrum':
        components = dict(zip(df['component'], df['fraction']))
        if 'uncertainty' in df.columns:
            uncertainties = dict(zip(df['component'], df['uncertainty']))
        else:
            uncertainties = None
        return cls(name=name, components=components, uncertainties=uncertainties, description=description)


class SourceSpectrumLibrary:
    def __init__(self):
        self.spectra: Dict[str, SourceSpectrum] = {}
        self._init_builtin_spectra()

    def _init_builtin_spectra(self):
        self.spectra['工业锅炉燃煤'] = self._create_coal_combustion()
        self.spectra['机动车尾气'] = self._create_vehicle_exhaust()
        self.spectra['道路扬尘'] = self._create_road_dust()
        self.spectra['建筑施工扬尘'] = self._create_construction_dust()
        self.spectra['生物质燃烧'] = self._create_biomass_burning()
        self.spectra['二次气溶胶'] = self._create_secondary_aerosol()

    def _create_coal_combustion(self) -> SourceSpectrum:
        components = {
            'SO4': 0.20, 'NO3': 0.05, 'NH4': 0.03, 'Cl': 0.02,
            'Na': 0.005, 'K': 0.01, 'Ca': 0.03, 'Mg': 0.008,
            'OC': 0.08, 'EC': 0.04,
            'Al': 0.05, 'Si': 0.15, 'Fe': 0.04, 'Zn': 0.003,
            'Pb': 0.002, 'Mn': 0.002, 'Cu': 0.001, 'V': 0.003,
            'Ni': 0.001, 'As': 0.002, 'Se': 0.0005,
        }
        return SourceSpectrum(
            name='工业锅炉燃煤',
            components=components,
            description='工业锅炉燃煤排放源谱',
        )

    def _create_vehicle_exhaust(self) -> SourceSpectrum:
        components = {
            'SO4': 0.03, 'NO3': 0.08, 'NH4': 0.02, 'Cl': 0.01,
            'Na': 0.005, 'K': 0.008, 'Ca': 0.005, 'Mg': 0.003,
            'OC': 0.25, 'EC': 0.20,
            'Al': 0.005, 'Si': 0.01, 'Fe': 0.01, 'Zn': 0.01,
            'Pb': 0.003, 'Mn': 0.005, 'Cu': 0.008, 'V': 0.001,
            'Ni': 0.002, 'As': 0.0005, 'Ba': 0.005,
        }
        return SourceSpectrum(
            name='机动车尾气',
            components=components,
            description='机动车尾气排放源谱',
        )

    def _create_road_dust(self) -> SourceSpectrum:
        components = {
            'SO4': 0.02, 'NO3': 0.01, 'NH4': 0.005, 'Cl': 0.01,
            'Na': 0.02, 'K': 0.015, 'Ca': 0.15, 'Mg': 0.03,
            'OC': 0.05, 'EC': 0.02,
            'Al': 0.10, 'Si': 0.25, 'Fe': 0.06, 'Zn': 0.002,
            'Pb': 0.001, 'Mn': 0.003, 'Cu': 0.001, 'V': 0.0005,
            'Ni': 0.0005, 'Ti': 0.008, 'Sr': 0.001,
        }
        return SourceSpectrum(
            name='道路扬尘',
            components=components,
            description='道路扬尘源谱',
        )

    def _create_construction_dust(self) -> SourceSpectrum:
        components = {
            'SO4': 0.01, 'NO3': 0.005, 'NH4': 0.002, 'Cl': 0.005,
            'Na': 0.01, 'K': 0.01, 'Ca': 0.20, 'Mg': 0.04,
            'OC': 0.03, 'EC': 0.01,
            'Al': 0.12, 'Si': 0.30, 'Fe': 0.07, 'Zn': 0.001,
            'Pb': 0.0005, 'Mn': 0.004, 'Cu': 0.0005, 'V': 0.0003,
            'Ni': 0.0003, 'Ti': 0.01, 'Sr': 0.0015,
        }
        return SourceSpectrum(
            name='建筑施工扬尘',
            components=components,
            description='建筑施工扬尘源谱',
        )

    def _create_biomass_burning(self) -> SourceSpectrum:
        components = {
            'SO4': 0.03, 'NO3': 0.04, 'NH4': 0.02, 'Cl': 0.03,
            'Na': 0.01, 'K': 0.08, 'Ca': 0.02, 'Mg': 0.005,
            'OC': 0.30, 'EC': 0.10,
            'Al': 0.01, 'Si': 0.02, 'Fe': 0.01, 'Zn': 0.002,
            'Pb': 0.001, 'Mn': 0.001, 'Cu': 0.0005, 'V': 0.0002,
            'Ni': 0.0002, 'As': 0.0003, 'levoglucosan': 0.05,
        }
        return SourceSpectrum(
            name='生物质燃烧',
            components=components,
            description='生物质燃烧源谱',
        )

    def _create_secondary_aerosol(self) -> SourceSpectrum:
        components = {
            'SO4': 0.35, 'NO3': 0.25, 'NH4': 0.15, 'Cl': 0.01,
            'Na': 0.005, 'K': 0.01, 'Ca': 0.005, 'Mg': 0.003,
            'OC': 0.10, 'EC': 0.005,
            'Al': 0.002, 'Si': 0.005, 'Fe': 0.002, 'Zn': 0.001,
            'Pb': 0.0005, 'Mn': 0.0002, 'Cu': 0.0002, 'V': 0.0001,
            'Ni': 0.0001, 'As': 0.0002,
        }
        return SourceSpectrum(
            name='二次气溶胶',
            components=components,
            description='二次气溶胶生成源谱',
        )

    def get_spectrum(self, name: str) -> Optional[SourceSpectrum]:
        return self.spectra.get(name)

    def get_all_names(self) -> List[str]:
        return list(self.spectra.keys())

    def add_spectrum(self, spectrum: SourceSpectrum):
        self.spectra[spectrum.name] = spectrum

    def remove_spectrum(self, name: str):
        if name in self.spectra:
            del self.spectra[name]

    def update_spectrum(self, name: str, components: Dict[str, float], uncertainties: Optional[Dict[str, float]] = None):
        if name in self.spectra:
            self.spectra[name].components = components.copy()
            if uncertainties is not None:
                self.spectra[name].uncertainties = uncertainties.copy()

    def get_source_matrix(self, source_names: List[str], component_names: List[str]) -> np.ndarray:
        matrix = []
        for name in source_names:
            if name in self.spectra:
                matrix.append(self.spectra[name].get_fractions(component_names))
            else:
                raise ValueError(f"Source spectrum '{name}' not found in library.")
        return np.array(matrix).T

    def get_uncertainty_matrix(self, source_names: List[str], component_names: List[str]) -> np.ndarray:
        matrix = []
        for name in source_names:
            if name in self.spectra:
                matrix.append(self.spectra[name].get_uncertainties(component_names))
            else:
                raise ValueError(f"Source spectrum '{name}' not found in library.")
        return np.array(matrix).T

    def load_custom_spectrum(self, df: pd.DataFrame, name: str, description: str = "") -> SourceSpectrum:
        spectrum = SourceSpectrum.from_dataframe(df, name=name, description=description)
        self.add_spectrum(spectrum)
        return spectrum

    def to_dataframe(self) -> pd.DataFrame:
        all_components = set()
        for spec in self.spectra.values():
            all_components.update(spec.components.keys())
        all_components = sorted(list(all_components))

        data = {'component': all_components}
        for name, spec in self.spectra.items():
            data[name] = [spec.components.get(c, 0.0) for c in all_components]
        return pd.DataFrame(data)
