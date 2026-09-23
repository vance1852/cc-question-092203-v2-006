"""风机模型定义。"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class Turbine:
    """风机参数类。

    Parameters
    ----------
    name : str
        风机名称/型号
    hub_height : float
        轮毂高度 (m)
    rotor_diameter : float
        转子直径 (m)
    thrust_coefficient : float
        推力系数 Ct (0-1)
    power_curve : np.ndarray
        功率曲线，形状为 (N, 2)，第一列为风速 (m/s)，第二列为功率 (kW)
    cut_in_speed : float
        切入风速 (m/s)，从功率曲线自动推断
    rated_speed : float
        额定风速 (m/s)，从功率曲线自动推断
    cut_out_speed : float
        切出风速 (m/s)，从功率曲线自动推断
    rated_power : float
        额定功率 (kW)，从功率曲线自动推断
    position : Optional[Tuple[float, float]]
        风机位置 (x, y) (m)，可选
    turbine_id : Optional[str]
        机组实例的稳定编号（场站范围内唯一），用于区分同型号的不同机组
    """

    name: str
    hub_height: float
    rotor_diameter: float
    thrust_coefficient: float
    power_curve: np.ndarray
    position: Optional[Tuple[float, float]] = None
    turbine_id: Optional[str] = None

    cut_in_speed: float = field(init=False)
    rated_speed: float = field(init=False)
    cut_out_speed: float = field(init=False)
    rated_power: float = field(init=False)

    def __post_init__(self) -> None:
        self.power_curve = np.asarray(self.power_curve, dtype=np.float64)
        if self.power_curve.ndim != 2 or self.power_curve.shape[1] != 2:
            raise ValueError("功率曲线必须是形状为 (N, 2) 的数组")

        self._derive_parameters()

        if not (0.0 < self.thrust_coefficient <= 1.0):
            raise ValueError(f"推力系数必须在 (0, 1] 范围内，当前为 {self.thrust_coefficient}")

    def _derive_parameters(self) -> None:
        """从功率曲线推导出切入、额定、切出风速和额定功率。"""
        ws = self.power_curve[:, 0]
        pw = self.power_curve[:, 1]

        positive_idx = np.where(pw > 0.0)[0]
        if len(positive_idx) == 0:
            raise ValueError("功率曲线中没有正功率点")

        self.cut_in_speed = float(ws[positive_idx[0]])
        self.rated_power = float(np.max(pw))

        rated_idx = np.where(pw >= self.rated_power * 0.999)[0]
        if len(rated_idx) > 0:
            self.rated_speed = float(ws[rated_idx[0]])
        else:
            self.rated_speed = float(ws[np.argmax(pw)])

        self.cut_out_speed = float(ws[-1])

    def power(self, wind_speed: float | np.ndarray) -> float | np.ndarray:
        """根据风速计算功率。

        Parameters
        ----------
        wind_speed : float | np.ndarray
            风速 (m/s)

        Returns
        -------
        float | np.ndarray
            功率 (kW)
        """
        ws = np.asarray(wind_speed, dtype=np.float64)
        result = np.interp(ws, self.power_curve[:, 0], self.power_curve[:, 1],
                          left=0.0, right=0.0)
        return result if ws.ndim > 0 else float(result)

    @property
    def rotor_area(self) -> float:
        """风轮扫掠面积 (m^2)。"""
        return np.pi * (self.rotor_diameter / 2.0) ** 2


def create_default_turbine(model: str = "V164-9.5MW") -> Turbine:
    """创建默认风机模型。

    Parameters
    ----------
    model : str
        风机型号，可选 "V164-9.5MW" 或 "V126-3.45MW"

    Returns
    -------
    Turbine
        风机实例
    """
    if model == "V164-9.5MW":
        speeds = np.arange(0.0, 31.0, 1.0)
        powers = np.zeros_like(speeds)
        cut_in = 4.0
        rated = 11.5
        cut_out = 25.0

        for i, ws in enumerate(speeds):
            if ws < cut_in or ws > cut_out:
                powers[i] = 0.0
            elif ws <= rated:
                powers[i] = 9500.0 * ((ws - cut_in) / (rated - cut_in)) ** 3
            else:
                powers[i] = 9500.0

        power_curve = np.column_stack([speeds, powers])

        return Turbine(
            name="V164-9.5MW",
            hub_height=105.0,
            rotor_diameter=164.0,
            thrust_coefficient=0.80,
            power_curve=power_curve,
        )

    elif model == "V126-3.45MW":
        speeds = np.arange(0.0, 26.0, 1.0)
        powers = np.zeros_like(speeds)
        cut_in = 3.5
        rated = 12.0
        cut_out = 25.0

        for i, ws in enumerate(speeds):
            if ws < cut_in or ws > cut_out:
                powers[i] = 0.0
            elif ws <= rated:
                powers[i] = 3450.0 * ((ws - cut_in) / (rated - cut_in)) ** 3
            else:
                powers[i] = 3450.0

        power_curve = np.column_stack([speeds, powers])

        return Turbine(
            name="V126-3.45MW",
            hub_height=87.0,
            rotor_diameter=126.0,
            thrust_coefficient=0.82,
            power_curve=power_curve,
        )

    else:
        raise ValueError(f"未知的风机型号: {model}")


def clone_turbine(prototype: Turbine, turbine_id: Optional[str] = None) -> Turbine:
    """基于原型机创建一个互相独立的机组实例。

    与直接复用同一个对象不同，本函数会复制功率曲线数组，保证后续给
    某台机组设置位置或修改参数不会影响其他机组。

    Parameters
    ----------
    prototype : Turbine
        原型风机
    turbine_id : Optional[str]
        新机组的稳定编号

    Returns
    -------
    Turbine
        与原型参数相同但彼此独立的风机实例
    """
    return Turbine(
        name=prototype.name,
        hub_height=prototype.hub_height,
        rotor_diameter=prototype.rotor_diameter,
        thrust_coefficient=prototype.thrust_coefficient,
        power_curve=np.array(prototype.power_curve, dtype=np.float64, copy=True),
        position=prototype.position,
        turbine_id=turbine_id,
    )


def create_turbine_from_spec(spec: dict) -> Turbine:
    """根据型号参数字典创建自定义风机。

    Parameters
    ----------
    spec : dict
        风机参数，支持字段：

        - ``name``: 型号名称（必填）
        - ``hub_height``: 轮毂高度 (m)（默认 90.0）
        - ``rotor_diameter``: 转子直径 (m)（必填）
        - ``thrust_coefficient``: 推力系数（默认 0.82）
        - ``rated_power_kw``: 额定功率 (kW)（必填）
        - ``cut_in_speed`` / ``rated_speed`` / ``cut_out_speed``:
          切入、额定、切出风速 (m/s)，默认 3.5/12.0/25.0
        - ``power_curve``: 自定义功率曲线 [[风速, kW], ...]，优先于
          上述三次方曲线参数

    Returns
    -------
    Turbine
        风机实例
    """
    if "name" not in spec:
        raise ValueError("自定义机型缺少必填字段 'name'")
    name = str(spec["name"])

    if "rotor_diameter" not in spec:
        raise ValueError(f"机型 {name} 缺少必填字段 'rotor_diameter'")

    if "power_curve" in spec and spec["power_curve"] is not None:
        power_curve = np.asarray(spec["power_curve"], dtype=np.float64)
        if power_curve.ndim != 2 or power_curve.shape[1] != 2:
            raise ValueError(f"机型 {name} 的 power_curve 必须是 (N, 2) 数组")
    else:
        missing = [k for k in ("rotor_diameter", "rated_power_kw") if k not in spec]
        if missing:
            raise ValueError(
                f"机型 {name} 缺少必填字段: {', '.join(missing)}"
                "（或直接提供 power_curve）"
            )

        cut_in = float(spec.get("cut_in_speed", 3.5))
        rated_ws = float(spec.get("rated_speed", 12.0))
        cut_out = float(spec.get("cut_out_speed", 25.0))
        rated_power_kw = float(spec["rated_power_kw"])

        if not (cut_in < rated_ws < cut_out):
            raise ValueError(
                f"机型 {name} 的风速参数需满足 "
                f"cut_in({cut_in}) < rated({rated_ws}) < cut_out({cut_out})"
            )

        speeds = np.arange(0.0, cut_out + 1.0, 1.0)
        powers = np.zeros_like(speeds)
        for i, ws in enumerate(speeds):
            if ws < cut_in or ws > cut_out:
                powers[i] = 0.0
            elif ws <= rated_ws:
                powers[i] = rated_power_kw * ((ws - cut_in) / (rated_ws - cut_in)) ** 3
            else:
                powers[i] = rated_power_kw
        power_curve = np.column_stack([speeds, powers])

    return Turbine(
        name=name,
        hub_height=float(spec.get("hub_height", 90.0)),
        rotor_diameter=float(spec["rotor_diameter"]),
        thrust_coefficient=float(spec.get("thrust_coefficient", 0.82)),
        power_curve=power_curve,
        turbine_id=spec.get("turbine_id"),
    )


def create_turbine_by_name(
    model: str,
    turbine_id: Optional[str] = None,
    extra_models: Optional[dict] = None,
) -> Turbine:
    """按型号名称创建独立机组实例。

    先在内置型号库中查找，找不到时再从自定义型号表 ``extra_models``
    中按规格创建。

    Parameters
    ----------
    model : str
        型号名称
    turbine_id : Optional[str]
        机组稳定编号
    extra_models : Optional[dict]
        自定义型号规格表 {型号名: 规格dict}

    Returns
    -------
    Turbine
        独立的风机实例
    """
    try:
        prototype = create_default_turbine(model)
    except ValueError:
        if extra_models and model in extra_models:
            return create_turbine_from_spec(
                {**extra_models[model], "name": model, "turbine_id": turbine_id}
            )
        known = ["V164-9.5MW", "V126-3.45MW"]
        if extra_models:
            known.extend(extra_models.keys())
        raise ValueError(
            f"未知的风机型号: {model}。已知型号: {', '.join(known)}"
        )
    return clone_turbine(prototype, turbine_id=turbine_id)
