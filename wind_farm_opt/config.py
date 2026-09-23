"""配置管理模块。

用于从JSON/YAML文件加载配置，或通过命令行参数构建配置。

机组声明支持三种等价形式：

1. 旧版单一机型（保持向后兼容）::

    {"n_turbines": 15, "turbine_model": "V126-3.45MW"}

2. 按机型声明数量（稳定编号按声明顺序连续分配）::

    {"turbine_fleet": [
        {"model": "V126-3.45MW", "count": 10, "id_prefix": "OLD"},
        {"model": "V164-9.5MW", "count": 5, "id_prefix": "NEW"}
    ]}

3. 逐机清单（每台机组独立条目，可携带稳定编号与预设位置）::

    {"turbine_fleet": [
        {"id": "WTG-001", "model": "V126-3.45MW"},
        {"id": "WTG-002", "model": "V164-9.5MW", "position": [1200.0, 800.0]}
    ]}

非内置型号可在条目中直接给出气动参数与（可选）单位造价。
"""

import json
from dataclasses import dataclass, field
from typing import Optional, List, Any

import numpy as np

from .core.turbine import (
    Turbine,
    clone_turbine,
    create_default_turbine,
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
from .constraints.spacing import compute_pairwise_min_spacings, check_min_spacing_pairwise


BUILTIN_TURBINE_MODELS = ("V164-9.5MW", "V126-3.45MW")


class ConfigError(ValueError):
    """配置错误。

    消息中携带可操作的修复建议，供命令行直接展示给用户。
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
    turbine_fleet: Optional[List[dict]] = None
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

        fleet_spec = data.get("turbine_fleet")
        explicit_n = data.get("n_turbines")

        if fleet_spec is not None:
            fleet_total = count_fleet_units(fleet_spec)
            if explicit_n is not None and explicit_n != fleet_total:
                raise ConfigError(
                    f"机型数量与声明不匹配: turbine_fleet 共声明 {fleet_total} 台，"
                    f"但 n_turbines={explicit_n}。请将 n_turbines 改为 "
                    f"{fleet_total}，或从配置中删除 n_turbines（机队模式下会"
                    "自动按清单计数）。"
                )
            n_turbines = fleet_total
        else:
            n_turbines = explicit_n if explicit_n is not None else 15

        return cls(
            n_turbines=n_turbines,
            turbine_model=data.get("turbine_model", "V126-3.45MW"),
            turbine_fleet=fleet_spec,
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
        if self.turbine_fleet is not None:
            data["turbine_fleet"] = self.turbine_fleet
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 机队构造
    # ------------------------------------------------------------------

    def create_turbines(self) -> list[Turbine]:
        """根据配置创建风机列表。

        多机型模式下每台机组都是独立实例并携带稳定编号；
        旧版单一机型模式返回同一型号的若干独立克隆，数值行为与旧版一致。
        """
        if self.turbine_fleet is not None:
            turbines = build_fleet_turbines(self.turbine_fleet)
            if len(turbines) != self.n_turbines:
                # 正常情况下 from_json 已保证一致；直接构造 dataclass 时兜底。
                raise ConfigError(
                    f"机型数量与 n_turbines 不匹配: 清单声明 {len(turbines)} 台，"
                    f"n_turbines={self.n_turbines}。请保持二者一致，或删除 "
                    "n_turbines 由清单自动计数。"
                )
            return turbines

        template = create_default_turbine(self.turbine_model)
        return [
            clone_turbine(template, turbine_id=f"WTG-{i + 1:03d}")
            for i in range(self.n_turbines)
        ]

    def fleet_positions(self) -> Optional[np.ndarray]:
        """逐机清单携带预设位置时返回位置数组，否则返回 None。"""
        if not self.turbine_fleet:
            return None
        if not any("position" in entry for entry in self.turbine_fleet):
            return None
        if not all("position" in entry for entry in self.turbine_fleet):
            given = [
                entry.get("id", f"第 {i + 1} 条")
                for i, entry in enumerate(self.turbine_fleet)
                if "position" in entry
            ]
            raise ConfigError(
                "逐机清单中只有部分机组提供了 position（已提供: "
                f"{', '.join(given)}）：要么为每台机组都给出 position，"
                "要么全部省略（由基线/优化自动布置）。"
            )
        positions = []
        for entry in self.turbine_fleet:
            pos = entry["position"]
            try:
                positions.append([float(pos[0]), float(pos[1])])
            except (TypeError, IndexError, KeyError, ValueError) as exc:
                raise ConfigError(
                    f"机组 {entry.get('id', '?')} 的 position 必须是 [x, y] "
                    f"两个数值，当前为 {pos!r}。"
                ) from exc
        return np.asarray(positions, dtype=np.float64)

    def validate_manifest_positions(self, boundary: SiteBoundary) -> np.ndarray:
        """校验逐机清单中的预设位置。

        检查位置数量、重复机位、是否越界以及成对间距，错误信息中包含
        对应机组的稳定编号与修复建议。

        Returns
        -------
        np.ndarray
            校验通过的位置数组 (N, 2)
        """
        if not self.turbine_fleet:
            raise ConfigError("当前配置不是逐机清单模式，没有可校验的预设位置")

        turbines = self.create_turbines()
        ids = [t.turbine_id for t in turbines]
        positions = self.fleet_positions()
        if positions is None:
            raise ConfigError(
                "逐机清单中仅部分机组提供了 position：要么为每台机组都给出 "
                "position，要么全部省略（由基线/优化自动布置）。"
            )
        if positions.shape[0] != len(turbines):
            raise ConfigError(
                f"机型数量与位置数量不匹配: {len(turbines)} 台机组仅提供了 "
                f"{positions.shape[0]} 个位置。"
            )

        seen: dict[tuple[float, float], str] = {}
        for (x, y), tid in zip(positions, ids):
            key = (round(float(x), 6), round(float(y), 6))
            if key in seen:
                raise ConfigError(
                    f"机位重复: 机组 {tid} 与 {seen[key]} 位于同一位置 "
                    f"({x:.1f}, {y:.1f})，请为其分配不同坐标。"
                )
            seen[key] = tid

        outside = [
            tid for tid, pt in zip(ids, positions) if not boundary.contains_point(pt)
        ]
        if outside:
            raise ConfigError(
                f"以下 {len(outside)} 台机组的预设位置超出场地边界: "
                f"{', '.join(outside)}。请调整坐标或扩大场地边界。"
            )

        diameters = np.array([t.rotor_diameter for t in turbines])
        matrix = compute_pairwise_min_spacings(
            diameters, self.optimization.min_spacing_multiple
        )
        valid, violations = check_min_spacing_pairwise(positions, matrix)
        if not valid:
            details = []
            for i, j in violations[:5]:
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                details.append(
                    f"{ids[i]}<->{ids[j]}: 实际 {dist:.0f} m < "
                    f"要求 {matrix[i, j]:.0f} m"
                )
            more = "" if len(violations) <= 5 else f"（另有 {len(violations) - 5} 对）"
            raise ConfigError(
                "预设位置不满足按各自直径计算的成对间距规则："
                + "；".join(details) + more +
                "。请增大机对间距或降低 optimization.min_spacing_multiple。"
            )

        return positions

    def fleet_summary(self) -> dict[str, dict[str, float]]:
        """按型号汇总机组数量与装机容量。"""
        summary: dict[str, dict[str, float]] = {}
        for turbine in self.create_turbines():
            item = summary.setdefault(
                turbine.name, {"count": 0, "capacity_mw": 0.0}
            )
            item["count"] += 1
            item["capacity_mw"] += turbine.rated_power / 1e3
        return summary

    def create_cost_models(self) -> dict[str, Any]:
        """按型号创建造价模型（含清单中的自定义造价覆盖）。"""
        from .economy.costs import (
            TurbineCostModel,
            get_default_turbine_cost,
            DEFAULT_CAPITAL_COST_PER_MW,
            DEFAULT_INSTALLATION_COST_PER_MW,
            DEFAULT_OM_COST_PER_MW_PER_YEAR,
        )

        cost_keys = (
            "capital_cost_per_MW",
            "installation_cost_per_MW",
            "o_and_m_cost_per_MW_per_year",
        )

        def _build_cost_model(name: str, entry: dict) -> TurbineCostModel:
            base = get_default_turbine_cost(name)
            overrides = {k: float(entry[k]) for k in cost_keys if k in entry}
            if not overrides:
                return base
            return TurbineCostModel(
                turbine_model=name,
                capital_cost_per_MW=overrides.get(
                    "capital_cost_per_MW", base.capital_cost_per_MW),
                installation_cost_per_MW=overrides.get(
                    "installation_cost_per_MW", base.installation_cost_per_MW),
                o_and_m_cost_per_MW_per_year=overrides.get(
                    "o_and_m_cost_per_MW_per_year",
                    base.o_and_m_cost_per_MW_per_year),
                design_lifetime=base.design_lifetime,
            )

        models: dict[str, TurbineCostModel] = {}
        seen_cost_specs: dict[str, dict] = {}
        for entry in self.turbine_fleet or []:
            name = entry["model"]
            cost_spec = {
                k: float(entry[k]) for k in cost_keys if k in entry
            }
            if name in seen_cost_specs and seen_cost_specs[name] != cost_spec:
                raise ConfigError(
                    f"机型 {name} 在不同条目中给出了互相矛盾的造价参数"
                    f"（{seen_cost_specs[name]} vs {cost_spec}）。"
                    "同名机型的造价定义必须一致；若确有不同造价，请改用不同 model 名称。"
                )
            seen_cost_specs[name] = cost_spec
            if name in models:
                continue
            if name in BUILTIN_TURBINE_MODELS:
                models[name] = _build_cost_model(name, entry)
            else:
                models[name] = TurbineCostModel(
                    turbine_model=name,
                    capital_cost_per_MW=float(
                        entry.get("capital_cost_per_MW", DEFAULT_CAPITAL_COST_PER_MW)
                    ),
                    installation_cost_per_MW=float(
                        entry.get("installation_cost_per_MW", DEFAULT_INSTALLATION_COST_PER_MW)
                    ),
                    o_and_m_cost_per_MW_per_year=float(
                        entry.get("o_and_m_cost_per_MW_per_year", DEFAULT_OM_COST_PER_MW_PER_YEAR)
                    ),
                )
        if not models:
            models[self.turbine_model] = get_default_turbine_cost(self.turbine_model)
        return models

    # ------------------------------------------------------------------
    # 其它组件
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


# ----------------------------------------------------------------------
# 机队清单解析
# ----------------------------------------------------------------------

def count_fleet_units(fleet_spec: list[dict]) -> int:
    """统计机队清单中的机组总数（不构造气动对象，便于提前校验）。"""
    total = 0
    for idx, entry in enumerate(fleet_spec):
        if not isinstance(entry, dict) or "model" not in entry:
            raise ConfigError(
                f"turbine_fleet 第 {idx + 1} 条记录格式无效：每个条目必须是包含 "
                '"model" 键的对象，例如 {"model": "V126-3.45MW", "count": 5}。'
            )
        count = entry.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ConfigError(
                f"机型 {entry['model']} 的 count 必须是不小于 1 的整数，"
                f"当前为 {count!r}。"
            )
        total += count
    if total == 0:
        raise ConfigError("turbine_fleet 为空：至少需要声明 1 台机组。")
    return total


def _build_template(entry: dict) -> Turbine:
    """根据清单条目构造一个机型模板（内置型号或自定义参数）。"""
    model = entry["model"]
    if not isinstance(model, str) or not model:
        raise ConfigError(f"机型名称必须是非空字符串，当前为 {model!r}。")

    custom_keys = (
        "rotor_diameter", "hub_height", "thrust_coefficient",
    )
    has_custom = any(k in entry for k in custom_keys)

    if not has_custom:
        if model not in BUILTIN_TURBINE_MODELS:
            raise ConfigError(
                f"未知机型 {model!r}：内置型号为 "
                f"{', '.join(BUILTIN_TURBINE_MODELS)}。若使用自定义机型，"
                "请在条目中补充 rotor_diameter、hub_height、thrust_coefficient "
                "以及 rated_power_kw/cut_in_speed/rated_wind_speed/cut_out_speed"
                "（或直接提供 power_curve）。"
            )
        return create_default_turbine(model)

    missing = [
        k for k in ("rotor_diameter", "hub_height", "thrust_coefficient")
        if k not in entry
    ]
    if missing:
        raise ConfigError(
            f"自定义机型 {model} 缺少必要参数: {', '.join(missing)}。"
        )

    power_curve = entry.get("power_curve")
    if power_curve is not None:
        power_curve = np.asarray(power_curve, dtype=np.float64)

    try:
        return create_turbine_from_spec(
            name=model,
            hub_height=float(entry["hub_height"]),
            rotor_diameter=float(entry["rotor_diameter"]),
            thrust_coefficient=float(entry["thrust_coefficient"]),
            rated_power_kw=(float(entry["rated_power_kw"])
                            if entry.get("rated_power_kw") is not None else None),
            cut_in_speed=entry.get("cut_in_speed"),
            rated_wind_speed=entry.get("rated_wind_speed"),
            cut_out_speed=entry.get("cut_out_speed"),
            power_curve=power_curve,
        )
    except ValueError as exc:
        raise ConfigError(f"自定义机型 {model} 构造失败: {exc}") from exc


def build_fleet_turbines(fleet_spec: list[dict]) -> list[Turbine]:
    """按机队清单构造互相独立、带稳定编号的机组实例。

    编号规则：

    - 逐机条目的 ``id`` 直接使用；
    - ``count`` 条目按 ``id_prefix``（默认 ``WTG``）加全场连续序号，
      序号在同一前缀组内从 1 开始；未给 ``id_prefix`` 时使用全场连续
      序号，保证声明顺序不变时编号稳定。
    """
    count_fleet_units(fleet_spec)

    turbines: list[Turbine] = []
    used_ids: set[str] = set()
    global_seq = 0
    prefix_seqs: dict[str, int] = {}
    model_signatures: dict[str, tuple] = {}

    def _model_signature(entry: dict, template: Turbine) -> tuple:
        if any(k in entry for k in ("rotor_diameter", "hub_height", "thrust_coefficient")):
            return (
                round(float(entry["rotor_diameter"]), 6),
                round(float(entry["hub_height"]), 6),
                round(float(entry["thrust_coefficient"]), 6),
                round(float(template.rated_power), 6),
            )
        return ("builtin",)

    def register_id(turbine_id: str, model: str) -> str:
        if turbine_id in used_ids:
            raise ConfigError(
                f"机组编号重复: {turbine_id!r} 在机型 {model} 的条目中再次出现。"
                "请检查 turbine_fleet 中各 id / id_prefix 是否冲突，"
                "并为每台机组分配唯一编号。"
            )
        used_ids.add(turbine_id)
        return turbine_id

    for entry in fleet_spec:
        model = entry["model"]
        count = int(entry.get("count", 1))
        explicit_id = entry.get("id")
        id_prefix = entry.get("id_prefix")
        has_position = "position" in entry

        if has_position and count > 1:
            raise ConfigError(
                f"机型 {model} 的条目同时给出 count={count} 与 position: "
                "按数量声明的条目不能携带单机位置。请改为逐机清单（每台机组"
                "一条记录并各自给出 id 与 position）。"
            )
        if explicit_id is not None and count > 1:
            raise ConfigError(
                f"机型 {model} 的条目同时给出 id={explicit_id!r} 与 "
                f"count={count}：多台机组不能共用一个编号。请改用 id_prefix "
                "声明编号前缀，或拆成逐机清单。"
            )

        template = _build_template(entry)

        signature = _model_signature(entry, template)
        if model in model_signatures and model_signatures[model] != signature:
            raise ConfigError(
                f"机型 {model} 在不同条目中给出了互相矛盾的气动参数。"
                "同名机型只能有一套直径/轮毂高/推力/功率定义；若确为不同机型，"
                "请改用不同的 model 名称。"
            )
        model_signatures[model] = signature

        for k in range(count):
            global_seq += 1

            if explicit_id is not None:
                turbine_id = str(explicit_id)
            elif id_prefix is not None:
                prefix_seqs[id_prefix] = prefix_seqs.get(id_prefix, 0) + 1
                turbine_id = f"{id_prefix}-{prefix_seqs[id_prefix]:02d}"
            else:
                turbine_id = f"WTG-{global_seq:03d}"

            register_id(turbine_id, model)
            turbine = clone_turbine(template, turbine_id=turbine_id)
            if has_position:
                pos = entry["position"]
                try:
                    turbine.position = (float(pos[0]), float(pos[1]))
                except (TypeError, IndexError, ValueError) as exc:
                    raise ConfigError(
                        f"机组 {turbine_id} 的 position 必须是 [x, y] 两个数值，"
                        f"当前为 {pos!r}。"
                    ) from exc
            turbines.append(turbine)

    return turbines


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
