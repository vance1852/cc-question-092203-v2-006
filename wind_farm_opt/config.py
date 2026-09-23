"""配置管理模块。

用于从JSON/YAML文件加载配置，或通过命令行参数构建配置。

机型配置支持三种等价的声明方式：

1. 单一机型（旧格式，结果与历史版本完全一致）::

    {"n_turbines": 15, "turbine_model": "V126-3.45MW"}

2. 按稳定型号声明多种机型及数量::

    {"fleet": {"turbine_types": [
        {"model": "V126-3.45MW", "count": 10},
        {"model": "V164-9.5MW", "count": 5}
    ]}}

   自定义机型可在条目内直接给出 ``spec``（转子直径、轮毂高度、
   额定功率等）。

3. 逐机清单，每台机组拥有场站范围内唯一的稳定编号::

    {"fleet": {"turbines": [
        {"id": "WT-01", "model": "V126-3.45MW"},
        {"id": "WT-02", "model": "V164-9.5MW", "position": [100.0, 200.0]}
    ], "catalog": {"MyTurbine": {...}}}}

   ``position`` 可省略（由基线/优化流程布置）；一旦给出，则清单中
   所有机组都必须给出位置。
"""

import json
from dataclasses import dataclass, field
from typing import Optional, List, Any

import numpy as np

from .core.turbine import (
    Turbine,
    create_turbine_by_name,
    create_turbine_from_spec,
)
from .core.wind_resource import WindResource, create_default_wind_resource
from .core.wake import JensenWake, GaussianWake, WakeModel
from .constraints.boundary import (
    SiteBoundary,
    create_rectangular_boundary,
    create_hexagonal_boundary,
    create_irregular_boundary,
)
from .constraints.spacing import pairwise_min_spacings
from .economy.costs import TurbineCostModel, cost_model_from_spec


BUILTIN_MODELS = ("V164-9.5MW", "V126-3.45MW")


class ConfigError(ValueError):
    """配置错误。

    消息中始终包含可操作的修复建议，便于直接定位 JSON 配置问题。
    """


@dataclass
class OptimizationConfig:
    """优化算法配置。"""
    algorithm: str = "ga"
    population_size: int = 40
    max_iterations: int = 80
    min_spacing_multiple: float = 5.0
    seed: Optional[int] = 42


@dataclass
class VisualizationConfig:
    """可视化配置。"""
    save_dir: str = "output"
    save_plots: bool = True
    show_plots: bool = False
    plot_wake_heatmap: bool = True


@dataclass
class EconomicConfig:
    """经济性分析配置。"""
    electricity_price: float = 0.45
    discount_rate: float = 0.06
    enable_analysis: bool = True


@dataclass
class WindFarmConfig:
    """完整的风电场分析配置。"""
    n_turbines: int = 15
    turbine_model: str = "V126-3.45MW"
    wake_model: str = "jensen"
    wake_decay: float = 0.07
    superposition_method: str = "sum_of_squares"

    boundary_type: str = "rectangular"
    boundary_params: dict = field(default_factory=lambda: {
        "width": 4000,
        "height": 4000,
        "center_x": 0,
        "center_y": 0,
    })

    wind_resource_type: str = "default"
    wind_resource_params: dict = field(default_factory=lambda: {
        "num_sectors": 12,
        "dominant_direction": 270.0,
        "mean_speed": 8.5,
    })

    # 机队声明（多机型）。fleet_mode: "single" | "types" | "manifest"
    fleet_mode: str = "single"
    turbine_types: List[dict] = field(default_factory=list)
    turbine_units: List[dict] = field(default_factory=list)
    turbine_catalog: dict = field(default_factory=dict)

    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    economic: EconomicConfig = field(default_factory=EconomicConfig)

    # ------------------------------------------------------------------
    # 加载 / 保存
    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, filepath: str) -> "WindFarmConfig":
        """从JSON文件加载配置。"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        opt_config = OptimizationConfig(**data.get("optimization", {}))
        vis_config = VisualizationConfig(**data.get("visualization", {}))
        econ_config = EconomicConfig(**data.get("economic", {}))

        config = cls(
            n_turbines=data.get("n_turbines", 15),
            turbine_model=data.get("turbine_model", "V126-3.45MW"),
            wake_model=data.get("wake_model", "jensen"),
            wake_decay=data.get("wake_decay", 0.07),
            superposition_method=data.get("superposition_method", "sum_of_squares"),
            boundary_type=data.get("boundary_type", "rectangular"),
            boundary_params=data.get("boundary_params", {}),
            wind_resource_type=data.get("wind_resource_type", "default"),
            wind_resource_params=data.get("wind_resource_params", {}),
            optimization=opt_config,
            visualization=vis_config,
            economic=econ_config,
        )

        fleet = data.get("fleet")
        if fleet is not None:
            config.load_fleet_block(fleet)

        return config

    def load_fleet_block(self, fleet: dict) -> None:
        """解析 ``fleet`` 配置块并确定机队声明模式。"""
        if not isinstance(fleet, dict):
            raise ConfigError("fleet 必须是一个对象，请检查配置文件格式")

        catalog = fleet.get("catalog", {}) or {}
        if not isinstance(catalog, dict):
            raise ConfigError("fleet.catalog 必须是 {型号名: 机型参数} 的对象")

        has_types = bool(fleet.get("turbine_types"))
        has_units = bool(fleet.get("turbines"))
        if has_types and has_units:
            raise ConfigError(
                "fleet.turbine_types（按型号数量声明）与 fleet.turbines"
                "（逐机清单）不能同时使用，请二选一"
            )
        if not has_types and not has_units:
            raise ConfigError(
                "fleet 块必须包含 turbine_types 或 turbines 之一，"
                "或删除 fleet 块改用 n_turbines + turbine_model"
            )

        self.turbine_catalog = dict(catalog)

        if has_types:
            self.fleet_mode = "types"
            self.turbine_types = [dict(t) for t in fleet["turbine_types"]]
            self.turbine_units = []
        else:
            self.fleet_mode = "manifest"
            self.turbine_units = [dict(u) for u in fleet["turbines"]]
            self.turbine_types = []

        self.validate()

    def to_json(self, filepath: str) -> None:
        """保存配置到JSON文件。"""
        data = {
            "n_turbines": self.n_turbines,
            "turbine_model": self.turbine_model,
            "wake_model": self.wake_model,
            "wake_decay": self.wake_decay,
            "superposition_method": self.superposition_method,
            "boundary_type": self.boundary_type,
            "boundary_params": self.boundary_params,
            "wind_resource_type": self.wind_resource_type,
            "wind_resource_params": self.wind_resource_params,
            "optimization": self.optimization.__dict__,
            "visualization": self.visualization.__dict__,
            "economic": self.economic.__dict__,
        }

        if self.fleet_mode == "types":
            data["fleet"] = {"turbine_types": self.turbine_types}
            if self.turbine_catalog:
                data["fleet"]["catalog"] = self.turbine_catalog
        elif self.fleet_mode == "manifest":
            data["fleet"] = {"turbines": self.turbine_units}
            if self.turbine_catalog:
                data["fleet"]["catalog"] = self.turbine_catalog

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def _known_models(self) -> list[str]:
        return list(BUILTIN_MODELS) + list(self.turbine_catalog.keys())

    def _check_model_known(self, model: str, where: str) -> None:
        if model not in BUILTIN_MODELS and model not in self.turbine_catalog:
            raise ConfigError(
                f"{where}引用了未知机型 '{model}'。"
                f"已知机型: {', '.join(self._known_models())}。"
                f"若为自定义机型，请在 fleet.catalog 中声明其参数"
                f"（rotor_diameter、rated_power_kw 等）"
            )

    def validate(self) -> None:
        """校验整机配置，失败时抛出带修复建议的 :class:`ConfigError`。"""
        # 先校验 catalog 中所有自定义机型的物理参数
        for catalog_model, catalog_spec in self.turbine_catalog.items():
            self._validate_custom_spec(catalog_model, catalog_spec)

        if self.n_turbines <= 0:
            raise ConfigError(
                f"n_turbines 必须为正整数，当前为 {self.n_turbines}"
            )
        if self.optimization.min_spacing_multiple <= 0:
            raise ConfigError(
                "optimization.min_spacing_multiple 必须为正数，当前为 "
                f"{self.optimization.min_spacing_multiple}"
            )

        if self.fleet_mode == "single":
            self._check_model_known(self.turbine_model, "turbine_model")
            return

        if self.fleet_mode == "types":
            if not self.turbine_types:
                raise ConfigError("turbine_types 不能为空")
            total = 0
            for idx, entry in enumerate(self.turbine_types):
                model = entry.get("model")
                if not model:
                    raise ConfigError(
                        f"turbine_types[{idx}] 缺少 'model' 字段"
                    )
                count = entry.get("count")
                if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                    raise ConfigError(
                        f"机型 '{model}' 的 count 必须为正整数，当前为 {count!r}"
                    )
                if "spec" in entry:
                    self._validate_custom_spec(model, entry["spec"])
                    self.turbine_catalog.setdefault(model, entry["spec"])
                self._check_model_known(model, f"turbine_types[{idx}]")
                total += count

            self.n_turbines = total
            # 混合机型时 turbine_model 仅作占位标签
            models = [e["model"] for e in self.turbine_types]
            self.turbine_model = models[0] if len(models) == 1 else "mixed"
            return

        if self.fleet_mode == "manifest":
            if not self.turbine_units:
                raise ConfigError("fleet.turbines 逐机清单不能为空")

            seen_ids = set()
            positions_present = 0
            for idx, unit in enumerate(self.turbine_units):
                uid = unit.get("id")
                model = unit.get("model")
                if not uid or not isinstance(uid, str):
                    raise ConfigError(
                        f"turbines[{idx}] 缺少稳定编号 'id'（字符串）"
                    )
                if uid in seen_ids:
                    raise ConfigError(
                        f"机组编号 '{uid}' 重复：逐机清单中的 id 必须唯一。"
                        f"请为第 {idx + 1} 台机组更换编号"
                    )
                seen_ids.add(uid)
                if not model:
                    raise ConfigError(
                        f"机组 '{uid}' 缺少 'model' 字段"
                    )
                self._check_model_known(model, f"机组 '{uid}'")

                pos = unit.get("position")
                if pos is not None:
                    if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
                        raise ConfigError(
                            f"机组 '{uid}' 的 position 必须是 [x, y] 两个坐标"
                        )
                    positions_present += 1

            if 0 < positions_present < len(self.turbine_units):
                missing = [
                    u["id"] for u in self.turbine_units
                    if u.get("position") is None
                ]
                raise ConfigError(
                    "逐机清单中只有部分机组给出了 position（"
                    f"已给 {positions_present}/{len(self.turbine_units)}）。"
                    "请为全部机组补充坐标，或全部省略交由基线/优化布置；"
                    f"缺少坐标的机组: {', '.join(missing[:5])}"
                    + (" ..." if len(missing) > 5 else "")
                )

            self.n_turbines = len(self.turbine_units)
            models = sorted({u["model"] for u in self.turbine_units})
            self.turbine_model = models[0] if len(models) == 1 else "mixed"
            return

        raise ConfigError(f"未知的机队声明模式: {self.fleet_mode}")

    def _validate_custom_spec(self, model: str, spec: Any) -> None:
        if not isinstance(spec, dict):
            raise ConfigError(f"机型 '{model}' 的 spec 必须是对象")
        if "rotor_diameter" not in spec and "power_curve" not in spec:
            raise ConfigError(
                f"自定义机型 '{model}' 需提供 rotor_diameter（以及 "
                f"rated_power_kw），或直接提供 power_curve"
            )
        # 交给物理构造器做数值校验，错误信息原样透传
        try:
            create_turbine_from_spec({**spec, "name": model})
        except ValueError as exc:
            raise ConfigError(f"自定义机型 '{model}' 参数无效: {exc}") from exc

    # ------------------------------------------------------------------
    # 机组实例化
    # ------------------------------------------------------------------
    def create_turbines(self) -> list[Turbine]:
        """根据配置创建互相独立的风机实例列表。

        每台机组都是独立对象（功率曲线数组各自持有副本），并带有
        场站范围内唯一的稳定编号 ``turbine_id``。机组顺序稳定：

        - single 模式：按编号 WT-01 ... WT-N 排列；
        - types 模式：按配置中型号声明顺序排列，同型号内顺序编号；
        - manifest 模式：严格按清单顺序排列。
        """
        self.validate()

        if self.fleet_mode == "single":
            width = max(2, len(str(self.n_turbines)))
            return [
                create_turbine_by_name(
                    self.turbine_model,
                    turbine_id=f"WT-{i + 1:0{width}d}",
                    extra_models=self.turbine_catalog,
                )
                for i in range(self.n_turbines)
            ]

        if self.fleet_mode == "types":
            turbines: list[Turbine] = []
            width = max(2, len(str(self.n_turbines)))
            seq = 0
            for entry in self.turbine_types:
                model = entry["model"]
                for k in range(entry["count"]):
                    seq += 1
                    turbines.append(
                        create_turbine_by_name(
                            model,
                            turbine_id=f"WT-{seq:0{width}d}",
                            extra_models=self.turbine_catalog,
                        )
                    )
            return turbines

        # manifest
        turbines = []
        for unit in self.turbine_units:
            turbines.append(
                create_turbine_by_name(
                    unit["model"],
                    turbine_id=unit["id"],
                    extra_models=self.turbine_catalog,
                )
            )
        return turbines

    def manifest_positions(self) -> Optional[np.ndarray]:
        """返回逐机清单给定的坐标；未给坐标时返回 None。"""
        if self.fleet_mode != "manifest":
            return None
        if not self.turbine_units or self.turbine_units[0].get("position") is None:
            return None
        return np.array(
            [[float(p) for p in u["position"]] for u in self.turbine_units],
            dtype=np.float64,
        )

    def fleet_composition(self, turbines: Optional[list[Turbine]] = None) -> list[dict]:
        """返回按型号汇总的机队构成明细。"""
        if turbines is None:
            turbines = self.create_turbines()

        order: list[str] = []
        groups: dict[str, list[Turbine]] = {}
        for t in turbines:
            if t.name not in groups:
                groups[t.name] = []
                order.append(t.name)
            groups[t.name].append(t)

        composition = []
        for model in order:
            units = groups[model]
            composition.append({
                "model": model,
                "count": len(units),
                "rotor_diameter": float(units[0].rotor_diameter),
                "hub_height": float(units[0].hub_height),
                "rated_power_kw": float(units[0].rated_power),
                "capacity_mw": float(len(units) * units[0].rated_power / 1e3),
                "turbine_ids": [t.turbine_id for t in units],
            })
        return composition

    def build_cost_models(self) -> dict[str, TurbineCostModel]:
        """为机队中的每个型号构建造价模型（自定义型号取 catalog 中的成本覆盖）。"""
        composition = self.fleet_composition()
        models = {}
        for item in composition:
            model = item["model"]
            spec = self.turbine_catalog.get(model)
            models[model] = cost_model_from_spec(model, spec)
        return models

    def spacing_matrix(
        self,
        turbines: Optional[list[Turbine]] = None,
    ) -> np.ndarray:
        """计算机队的逐对最小间距矩阵。"""
        if turbines is None:
            turbines = self.create_turbines()
        diameters = np.array([t.rotor_diameter for t in turbines])
        return pairwise_min_spacings(
            diameters, self.optimization.min_spacing_multiple
        )

    # ------------------------------------------------------------------
    # 物理模型构造
    # ------------------------------------------------------------------
    def create_wake_model(self) -> WakeModel:
        """根据配置创建尾流模型。"""
        if self.wake_model.lower() == "jensen":
            return JensenWake(wake_decay=self.wake_decay)
        elif self.wake_model.lower() == "gaussian":
            return GaussianWake(wake_decay=0.035)
        else:
            raise ValueError(f"未知的尾流模型: {self.wake_model}")

    def create_boundary(self) -> SiteBoundary:
        """根据配置创建场地边界。"""
        bp = self.boundary_params
        if self.boundary_type.lower() == "rectangular":
            return create_rectangular_boundary(
                width=bp.get("width", 4000),
                height=bp.get("height", 4000),
                center_x=bp.get("center_x", 0),
                center_y=bp.get("center_y", 0),
            )
        elif self.boundary_type.lower() == "hexagonal":
            return create_hexagonal_boundary(
                radius=bp.get("radius", 2500),
                center_x=bp.get("center_x", 0),
                center_y=bp.get("center_y", 0),
            )
        elif self.boundary_type.lower() == "irregular":
            return create_irregular_boundary()
        elif self.boundary_type.lower() == "custom":
            vertices = np.array(bp["vertices"], dtype=np.float64)
            return SiteBoundary(vertices)
        else:
            raise ValueError(f"未知的边界类型: {self.boundary_type}")

    def create_wind_resource(self) -> WindResource:
        """根据配置创建风资源。"""
        wrp = self.wind_resource_params
        if self.wind_resource_type.lower() == "default":
            return create_default_wind_resource(
                num_sectors=wrp.get("num_sectors", 12),
                dominant_direction=wrp.get("dominant_direction", 270.0),
                mean_speed=wrp.get("mean_speed", 8.5),
            )
        elif self.wind_resource_type.lower() == "uniform":
            from .core.wind_resource import create_simple_wind_resource
            return create_simple_wind_resource(
                num_sectors=wrp.get("num_sectors", 12),
                uniform=True,
                mean_speed=wrp.get("mean_speed", 8.0),
            )
        else:
            raise ValueError(f"未知的风资源类型: {self.wind_resource_type}")


def create_sample_config() -> WindFarmConfig:
    """创建示例配置。"""
    return WindFarmConfig(
        n_turbines=12,
        turbine_model="V126-3.45MW",
        wake_model="jensen",
        wake_decay=0.07,
        boundary_type="rectangular",
        boundary_params={"width": 3500, "height": 3500, "center_x": 0, "center_y": 0},
        wind_resource_type="default",
        wind_resource_params={"num_sectors": 12, "dominant_direction": 270.0, "mean_speed": 8.5},
        optimization=OptimizationConfig(
            algorithm="ga",
            population_size=30,
            max_iterations=50,
            min_spacing_multiple=5.0,
            seed=42,
        ),
    )
