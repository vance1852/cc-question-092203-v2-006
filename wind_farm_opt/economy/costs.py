"""风电场经济性评估。

简化的度电成本(LCOE)计算模型。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TurbineCostModel:
    """风机造价模型。

    Parameters
    ----------
    turbine_model : str
        风机型号名称
    capital_cost_per_MW : float
        单位容量造价 (万元/MW)
    installation_cost_per_MW : float
        安装费用 (万元/MW)
    o_and_m_cost_per_MW_per_year : float
        年运维费用 (万元/MW/year)
    design_lifetime : float
        设计寿命 (年)
    """

    turbine_model: str
    capital_cost_per_MW: float
    installation_cost_per_MW: float
    o_and_m_cost_per_MW_per_year: float
    design_lifetime: float = 25.0


@dataclass
class FarmCostModel:
    """风电场整体造价模型。

    Parameters
    ----------
    site_development_cost : float
        场地开发费用 (万元)
    grid_connection_cost_per_MW : float
        并网费用 (万元/MW)
    access_road_cost : float
        道路建设费用 (万元)
    decommissioning_cost_per_MW : float
        退役拆除费用 (万元/MW)
    discount_rate : float
        折现率 (0-1)
    inflation_rate : float
        通货膨胀率 (0-1)
    """

    site_development_cost: float = 5000.0
    grid_connection_cost_per_MW: float = 300.0
    access_road_cost: float = 2000.0
    decommissioning_cost_per_MW: float = 100.0
    discount_rate: float = 0.06
    inflation_rate: float = 0.025


@dataclass
class EconomicResult:
    """经济性评估结果。

    Parameters
    ----------
    total_installed_capacity : float
        总装机容量 (MW)
    net_aep : float
        净年发电量 (GWh/year)
    annual_revenue : float
        年收益 (万元/year)
    lcoe : float
        度电成本 (元/kWh)
    total_capital_cost : float
        总初始投资 (万元)
    total_om_cost_annual : float
        年运维费用 (万元/year)
    npv : Optional[float]
        净现值 (万元)
    irr : Optional[float]
        内部收益率 (%)
    payback_period : Optional[float]
        投资回收期 (年)
    cost_breakdown : dict[str, float]
        成本分项 (万元)
    """

    total_installed_capacity: float
    net_aep: float
    annual_revenue: float
    lcoe: float
    total_capital_cost: float
    total_om_cost_annual: float
    npv: Optional[float]
    irr: Optional[float]
    payback_period: Optional[float]
    cost_breakdown: dict[str, float]
    model_breakdown: Optional[list[dict]] = None


class EconomicAnalyzer:
    """风电场经济性分析器。"""

    def __init__(
        self,
        turbine_cost: TurbineCostModel,
        farm_cost: FarmCostModel,
        electricity_price: float = 0.45,
        turbine_costs: Optional[dict[str, TurbineCostModel]] = None,
    ) -> None:
        """
        Parameters
        ----------
        turbine_cost : TurbineCostModel
            风机造价模型（单一机型，或作为未在 turbine_costs 中
            声明的型号的回退）
        farm_cost : FarmCostModel
            风电场造价模型
        electricity_price : float
            上网电价 (元/kWh)
        turbine_costs : Optional[dict[str, TurbineCostModel]]
            多机型场景下各型号的造价模型 {型号名: 造价模型}；
            省略时仅包含 turbine_cost 对应的单一型号
        """
        self.turbine_cost = turbine_cost
        self.farm_cost = farm_cost
        self.electricity_price = electricity_price
        if turbine_costs is None:
            turbine_costs = {turbine_cost.turbine_model: turbine_cost}
        self.turbine_costs = dict(turbine_costs)

    def _cost_for(self, model: str) -> TurbineCostModel:
        """返回指定型号的造价模型，未知型号回退到单一机型造价模型。"""
        return self.turbine_costs.get(model, self.turbine_cost)

    def compute_capital_cost(
        self,
        n_turbines: int,
        rated_power_per_turbine_MW: float,
    ) -> tuple[float, dict[str, float]]:
        """计算初始投资。

        Parameters
        ----------
        n_turbines : int
            风机台数
        rated_power_per_turbine_MW : float
            单台风机额定功率 (MW)

        Returns
        -------
        tuple[float, dict[str, float]]
            - 总初始投资 (万元)
            - 成本分项明细
        """
        total_capacity = n_turbines * rated_power_per_turbine_MW

        turbine_capital = n_turbines * rated_power_per_turbine_MW * self.turbine_cost.capital_cost_per_MW
        turbine_installation = n_turbines * rated_power_per_turbine_MW * self.turbine_cost.installation_cost_per_MW
        grid_connection = total_capacity * self.farm_cost.grid_connection_cost_per_MW
        site_dev = self.farm_cost.site_development_cost
        access_road = self.farm_cost.access_road_cost

        total = (
            turbine_capital
            + turbine_installation
            + grid_connection
            + site_dev
            + access_road
        )

        breakdown = {
            "风机设备": turbine_capital,
            "风机安装": turbine_installation,
            "并网工程": grid_connection,
            "场地开发": site_dev,
            "道路建设": access_road,
        }

        return total, breakdown

    def compute_annual_om_cost(
        self,
        n_turbines: int,
        rated_power_per_turbine_MW: float,
    ) -> float:
        """计算年运维费用。

        Parameters
        ----------
        n_turbines : int
            风机台数
        rated_power_per_turbine_MW : float
            单台风机额定功率 (MW)

        Returns
        -------
        float
            年运维费用 (万元/year)
        """
        total_capacity = n_turbines * rated_power_per_turbine_MW
        return total_capacity * self.turbine_cost.o_and_m_cost_per_MW_per_year

    def compute_annual_revenue(self, net_aep_GWh: float) -> float:
        """计算年发电收益。

        Parameters
        ----------
        net_aep_GWh : float
            净年发电量 (GWh/year)

        Returns
        -------
        float
            年收益 (万元/year)
        """
        return net_aep_GWh * 1e6 * self.electricity_price / 1e4

    def compute_lcoe(
        self,
        total_capital_cost: float,
        total_om_cost_annual: float,
        net_aep_GWh: float,
        lifetime: Optional[float] = None,
    ) -> float:
        """计算度电成本(LCOE)。

        LCOE = 总费用现值 / 总发电量现值

        Parameters
        ----------
        total_capital_cost : float
            初始投资 (万元)
        total_om_cost_annual : float
            年运维费用 (万元/year)
        net_aep_GWh : float
            年净发电量 (GWh/year)
        lifetime : Optional[float]
            寿命期 (年)，默认使用风机设计寿命

        Returns
        -------
        float
            LCOE (元/kWh)
        """
        if lifetime is None:
            lifetime = self.turbine_cost.design_lifetime

        r = self.farm_cost.discount_rate
        i = self.farm_cost.inflation_rate
        r_real = (1 + r) / (1 + i) - 1

        annuity_factor = (1 - (1 + r_real) ** (-lifetime)) / r_real

        total_cost_pv = total_capital_cost + total_om_cost_annual * annuity_factor
        total_energy_pv = net_aep_GWh * 1e6 * annuity_factor

        if total_energy_pv <= 0:
            return np.inf

        lcoe_yuan_per_kwh = (total_cost_pv * 1e4) / total_energy_pv

        return float(lcoe_yuan_per_kwh)

    def compute_npv(
        self,
        total_capital_cost: float,
        annual_revenue: float,
        annual_om_cost: float,
        lifetime: Optional[float] = None,
    ) -> float:
        """计算净现值(NPV)。

        Parameters
        ----------
        total_capital_cost : float
            初始投资 (万元)
        annual_revenue : float
            年收益 (万元/year)
        annual_om_cost : float
            年运维费用 (万元/year)
        lifetime : Optional[float]
            寿命期 (年)

        Returns
        -------
        float
            NPV (万元)
        """
        if lifetime is None:
            lifetime = self.turbine_cost.design_lifetime

        r = self.farm_cost.discount_rate
        i = self.farm_cost.inflation_rate
        r_real = (1 + r) / (1 + i) - 1

        net_cash_flow = annual_revenue - annual_om_cost
        annuity_factor = (1 - (1 + r_real) ** (-lifetime)) / r_real

        npv = -total_capital_cost + net_cash_flow * annuity_factor

        return float(npv)

    def compute_payback_period(
        self,
        total_capital_cost: float,
        annual_revenue: float,
        annual_om_cost: float,
    ) -> Optional[float]:
        """计算静态投资回收期。

        Parameters
        ----------
        total_capital_cost : float
            初始投资 (万元)
        annual_revenue : float
            年收益 (万元/year)
        annual_om_cost : float
            年运维费用 (万元/year)

        Returns
        -------
        Optional[float]
            投资回收期 (年)，若无法收回则返回 None
        """
        net_annual = annual_revenue - annual_om_cost
        if net_annual <= 0:
            return None
        return float(total_capital_cost / net_annual)

    def compute_irr(
        self,
        total_capital_cost: float,
        annual_revenue: float,
        annual_om_cost: float,
        lifetime: Optional[float] = None,
    ) -> Optional[float]:
        """计算内部收益率(IRR)。

        使用二分法求解。

        Parameters
        ----------
        total_capital_cost : float
            初始投资 (万元)
        annual_revenue : float
            年收益 (万元/year)
        annual_om_cost : float
            年运维费用 (万元/year)
        lifetime : Optional[float]
            寿命期 (年)

        Returns
        -------
        Optional[float]
            IRR (%)，若无法求解则返回 None
        """
        if lifetime is None:
            lifetime = self.turbine_cost.design_lifetime

        net_cash_flow = annual_revenue - annual_om_cost

        def npv_at_rate(rate):
            annuity = (1 - (1 + rate) ** (-lifetime)) / rate if rate > 0 else lifetime
            return -total_capital_cost + net_cash_flow * annuity

        low, high = -0.99, 1.0
        if npv_at_rate(low) * npv_at_rate(high) > 0:
            return None

        for _ in range(100):
            mid = (low + high) / 2
            npv_mid = npv_at_rate(mid)
            if abs(npv_mid) < 1e-6:
                return float(mid * 100)
            if npv_at_rate(low) * npv_mid < 0:
                high = mid
            else:
                low = mid

        return float(((low + high) / 2) * 100)

    def analyze(
        self,
        n_turbines: int,
        rated_power_per_turbine_MW: float,
        net_aep_GWh: float,
    ) -> EconomicResult:
        """进行完整的经济性分析。

        Parameters
        ----------
        n_turbines : int
            风机台数
        rated_power_per_turbine_MW : float
            单台风机额定功率 (MW)
        net_aep_GWh : float
            净年发电量 (GWh/year)

        Returns
        -------
        EconomicResult
            经济性分析结果
        """
        total_capacity = n_turbines * rated_power_per_turbine_MW

        total_capital_cost, cost_breakdown = self.compute_capital_cost(
            n_turbines, rated_power_per_turbine_MW
        )
        annual_om_cost = self.compute_annual_om_cost(
            n_turbines, rated_power_per_turbine_MW
        )
        annual_revenue = self.compute_annual_revenue(net_aep_GWh)

        lcoe = self.compute_lcoe(
            total_capital_cost, annual_om_cost, net_aep_GWh
        )
        npv = self.compute_npv(
            total_capital_cost, annual_revenue, annual_om_cost
        )
        irr = self.compute_irr(
            total_capital_cost, annual_revenue, annual_om_cost
        )
        payback = self.compute_payback_period(
            total_capital_cost, annual_revenue, annual_om_cost
        )

        return EconomicResult(
            total_installed_capacity=float(total_capacity),
            net_aep=float(net_aep_GWh),
            annual_revenue=float(annual_revenue),
            lcoe=float(lcoe),
            total_capital_cost=float(total_capital_cost),
            total_om_cost_annual=float(annual_om_cost),
            npv=npv,
            irr=irr,
            payback_period=payback,
            cost_breakdown=cost_breakdown,
        )

    def analyze_fleet(
        self,
        fleet_items: list[dict],
        net_aep_GWh: float,
    ) -> EconomicResult:
        """多机型机队的完整经济性分析。

        各型号按自身单位容量造价、安装费、年运维费分别核算后汇总；
        并网费按总装机容量核算，场地开发与道路建设为固定费用。
        寿命期取各型号设计寿命的最大值，年运维费用现值则按各型号
        自身寿命分别折现后求和。

        Parameters
        ----------
        fleet_items : list[dict]
            机队构成，每个元素为
            ``{"model": 型号名, "count": 台数,
               "rated_power_mw": 单台额定功率(MW)}``
        net_aep_GWh : float
            净年发电量 (GWh/year)

        Returns
        -------
        EconomicResult
            经济性分析结果，``model_breakdown`` 中含分型号明细
        """
        r = self.farm_cost.discount_rate
        inflation = self.farm_cost.inflation_rate
        r_real = (1 + r) / (1 + inflation) - 1

        def annuity(years: float) -> float:
            return (1 - (1 + r_real) ** (-years)) / r_real

        total_capacity = 0.0
        turbine_capital = 0.0
        turbine_installation = 0.0
        annual_om_cost = 0.0
        om_cost_pv = 0.0
        lifetimes: list[float] = []
        model_breakdown: list[dict] = []

        for item in fleet_items:
            model = item["model"]
            count = int(item["count"])
            rated_mw = float(item["rated_power_mw"])
            capacity = count * rated_mw
            cm = self._cost_for(model)

            cap = capacity * cm.capital_cost_per_MW
            inst = capacity * cm.installation_cost_per_MW
            om_annual = capacity * cm.o_and_m_cost_per_MW_per_year

            total_capacity += capacity
            turbine_capital += cap
            turbine_installation += inst
            annual_om_cost += om_annual
            om_cost_pv += om_annual * annuity(cm.design_lifetime)
            lifetimes.append(cm.design_lifetime)

            model_breakdown.append({
                "model": model,
                "count": count,
                "rated_power_mw": rated_mw,
                "capacity_mw": float(capacity),
                "capital_cost": float(cap),
                "installation_cost": float(inst),
                "annual_om_cost": float(om_annual),
                "capital_cost_per_MW": float(cm.capital_cost_per_MW),
                "installation_cost_per_MW": float(cm.installation_cost_per_MW),
                "o_and_m_cost_per_MW_per_year": float(cm.o_and_m_cost_per_MW_per_year),
                "design_lifetime": float(cm.design_lifetime),
            })

        grid_connection = total_capacity * self.farm_cost.grid_connection_cost_per_MW
        site_dev = self.farm_cost.site_development_cost
        access_road = self.farm_cost.access_road_cost

        total_capital_cost = (
            turbine_capital + turbine_installation
            + grid_connection + site_dev + access_road
        )
        cost_breakdown = {
            "风机设备": turbine_capital,
            "风机安装": turbine_installation,
            "并网工程": grid_connection,
            "场地开发": site_dev,
            "道路建设": access_road,
        }

        annual_revenue = self.compute_annual_revenue(net_aep_GWh)
        farm_lifetime = max(lifetimes) if lifetimes else self.turbine_cost.design_lifetime

        energy_pv = net_aep_GWh * 1e6 * annuity(farm_lifetime)
        if energy_pv <= 0:
            lcoe = np.inf
        else:
            lcoe = (total_capital_cost + om_cost_pv) * 1e4 / energy_pv

        npv = self.compute_npv(
            total_capital_cost, annual_revenue, annual_om_cost, farm_lifetime
        )
        irr = self.compute_irr(
            total_capital_cost, annual_revenue, annual_om_cost, farm_lifetime
        )
        payback = self.compute_payback_period(
            total_capital_cost, annual_revenue, annual_om_cost
        )

        return EconomicResult(
            total_installed_capacity=float(total_capacity),
            net_aep=float(net_aep_GWh),
            annual_revenue=float(annual_revenue),
            lcoe=float(lcoe),
            total_capital_cost=float(total_capital_cost),
            total_om_cost_annual=float(annual_om_cost),
            npv=npv,
            irr=irr,
            payback_period=payback,
            cost_breakdown=cost_breakdown,
            model_breakdown=model_breakdown,
        )


def get_default_turbine_cost(model: str = "V164-9.5MW") -> TurbineCostModel:
    """获取默认风机造价模型。

    Parameters
    ----------
    model : str
        风机型号

    Returns
    -------
    TurbineCostModel
        风机造价模型
    """
    if model == "V164-9.5MW":
        return TurbineCostModel(
            turbine_model="V164-9.5MW",
            capital_cost_per_MW=650.0,
            installation_cost_per_MW=80.0,
            o_and_m_cost_per_MW_per_year=18.0,
            design_lifetime=25.0,
        )
    elif model == "V126-3.45MW":
        return TurbineCostModel(
            turbine_model="V126-3.45MW",
            capital_cost_per_MW=580.0,
            installation_cost_per_MW=70.0,
            o_and_m_cost_per_MW_per_year=15.0,
            design_lifetime=25.0,
        )
    else:
        raise ValueError(f"未知的风机型号: {model}")


# 自定义机型的默认造价系数（在内置小机型水平上取保守默认值）
DEFAULT_CUSTOM_CAPITAL_COST_PER_MW = 600.0
DEFAULT_CUSTOM_INSTALLATION_COST_PER_MW = 75.0
DEFAULT_CUSTOM_OM_COST_PER_MW_PER_YEAR = 16.0


def cost_model_from_spec(
    model: str,
    spec: Optional[dict] = None,
) -> TurbineCostModel:
    """根据型号名称（及可选的自定义机型规格）构建造价模型。

    自定义机型可在规格中通过 ``capital_cost_per_MW``、
    ``installation_cost_per_MW``、``o_and_m_cost_per_MW_per_year``、
    ``design_lifetime`` 覆盖默认造价系数；未覆盖时使用混合机型的
    保守默认值。

    Parameters
    ----------
    model : str
        风机型号
    spec : Optional[dict]
        自定义机型规格（config 中的 catalog 条目）

    Returns
    -------
    TurbineCostModel
        该型号的造价模型
    """
    try:
        return get_default_turbine_cost(model)
    except ValueError:
        pass

    spec = spec or {}
    return TurbineCostModel(
        turbine_model=model,
        capital_cost_per_MW=float(
            spec.get("capital_cost_per_MW", DEFAULT_CUSTOM_CAPITAL_COST_PER_MW)
        ),
        installation_cost_per_MW=float(
            spec.get("installation_cost_per_MW", DEFAULT_CUSTOM_INSTALLATION_COST_PER_MW)
        ),
        o_and_m_cost_per_MW_per_year=float(
            spec.get(
                "o_and_m_cost_per_MW_per_year",
                DEFAULT_CUSTOM_OM_COST_PER_MW_PER_YEAR,
            )
        ),
        design_lifetime=float(spec.get("design_lifetime", 25.0)),
    )


def get_default_farm_cost() -> FarmCostModel:
    """获取默认风电场造价模型。"""
    return FarmCostModel(
        site_development_cost=5000.0,
        grid_connection_cost_per_MW=300.0,
        access_road_cost=2000.0,
        decommissioning_cost_per_MW=100.0,
        discount_rate=0.06,
        inflation_rate=0.025,
    )
