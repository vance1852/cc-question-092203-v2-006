"""命令行接口。

提供完整的风电场机位布局评估和优化流程。
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np

from .config import WindFarmConfig, ConfigError, create_sample_config
from .core.turbine import Turbine, create_turbine_by_name
from .core.wind_resource import WindResource
from .core.wake import WakeModel
from .constraints.boundary import SiteBoundary
from .constraints.spacing import pairwise_min_spacings
from .farm.aep import AEPCalculator, FarmResult
from .optimization.baseline import generate_grid_layout, verify_layout_feasible
from .optimization.ga import GeneticAlgorithm, GAConfig
from .optimization.pso import ParticleSwarmOptimizer, PSOConfig
from .economy.costs import (
    EconomicAnalyzer,
    EconomicResult,
    get_default_farm_cost,
)
from .visualization.plotting import (
    plot_farm_layout,
    plot_wind_rose,
    plot_convergence,
    plot_aep_vs_turbines,
    plot_turbine_loss_bar,
    plot_comparison,
    plot_wake_heatmap,
)


class WindFarmOptimizerCLI:
    """风电场优化命令行接口主类。"""

    def __init__(self, config: WindFarmConfig) -> None:
        self.config = config
        self._setup_output_dir()

        self.turbines = config.create_turbines()
        self.boundary = config.create_boundary()
        self.wind_resource = config.create_wind_resource()
        self.wake_model = config.create_wake_model()

        self.rotor_diameters = np.array([t.rotor_diameter for t in self.turbines])
        self.rated_powers = np.array([t.rated_power for t in self.turbines])
        self.thrust_coefficients = np.array([t.thrust_coefficient for t in self.turbines])
        # 每一对机组基于各自平均直径的最小间距矩阵
        self.spacing_matrix = config.spacing_matrix(self.turbines)
        self.turbine_ids = [t.turbine_id for t in self.turbines]
        self.turbine_models = [t.name for t in self.turbines]
        self.fleet_composition = config.fleet_composition(self.turbines)
        self._model_info = {
            item["model"]: {
                "rotor_diameter": item["rotor_diameter"],
                "rated_power_kw": item["rated_power_kw"],
            }
            for item in self.fleet_composition
        }

        self.aep_calc = AEPCalculator(
            turbines=self.turbines,
            wind_resource=self.wind_resource,
            wake_model=self.wake_model,
            wake_superposition=config.superposition_method,
        )

        self.baseline_positions: Optional[np.ndarray] = None
        self.baseline_result: Optional[FarmResult] = None
        self.optimized_positions: Optional[np.ndarray] = None
        self.optimized_result: Optional[FarmResult] = None
        self.optimize_result = None
        self.economic_result: Optional[EconomicResult] = None
        self.sweep_results: Optional[dict] = None
        self._manifest_positions_fixed = False

    def _setup_output_dir(self) -> None:
        """创建输出目录。"""
        output_dir = self.config.visualization.save_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f"输出目录: {os.path.abspath(output_dir)}")

    def _print_header(self, title: str) -> None:
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)

    def _print_result_summary(self, result: FarmResult, label: str = "") -> None:
        """打印计算结果摘要。"""
        print(f"\n--- {label} 结果 ---")
        print(f"  装机容量:    {result.total_installed_capacity:.2f} MW")
        print(f"  理论AEP:     {result.gross_aep/1e3:.2f} GWh/年")
        print(f"  净AEP:       {result.net_aep/1e3:.2f} GWh/年")
        print(f"  尾流损失:    {result.total_wake_loss/1e3:.2f} GWh/年 ({result.wake_loss_pct:.2f}%)")
        print(f"  容量系数:    {result.capacity_factor:.2f}%")
        print(f"  风机台数:    {len(result.turbine_results)}")
        if result.model_summary:
            print("  机型构成:")
            for model, info in result.model_summary.items():
                print(
                    f"    {model}: {info['count']} 台, "
                    f"Ø{info['rotor_diameter']:.0f} m, "
                    f"{info['rated_power_kw']/1e3:.2f} MW/台, "
                    f"小计 {info['capacity_mw']:.2f} MW, "
                    f"净AEP {info['net_aep_mwh']/1e3:.2f} GWh/年"
                )

        max_loss_turb = max(result.turbine_results, key=lambda x: x.wake_loss_pct)
        max_label = max_loss_turb.turbine_id or f"#{max_loss_turb.turbine_idx}"
        print(f"  最大损失机组: {max_label} ({max_loss_turb.name}, "
              f"{max_loss_turb.wake_loss_pct:.2f}%)")
        if max_loss_turb.dominant_wake_source is not None:
            src = result.turbine_results[max_loss_turb.dominant_wake_source]
            src_label = src.turbine_id or f"#{src.turbine_idx}"
            print(f"    主要影响源: {src_label} ({src.name})")

    def _print_fleet_composition(self) -> None:
        """打印机队型号与容量构成。"""
        if len(self.fleet_composition) == 1:
            item = self.fleet_composition[0]
            print(f"  机型: {item['model']} × {item['count']} 台 "
                  f"(Ø{item['rotor_diameter']:.0f} m, "
                  f"{item['rated_power_kw']/1e3:.2f} MW/台, "
                  f"合计 {item['capacity_mw']:.2f} MW)")
        else:
            print("  混装机队:")
            for item in self.fleet_composition:
                print(
                    f"    - {item['model']} × {item['count']} 台 "
                    f"(Ø{item['rotor_diameter']:.0f} m, "
                    f"{item['rated_power_kw']/1e3:.2f} MW/台, "
                    f"小计 {item['capacity_mw']:.2f} MW)"
                )
            total_cap = sum(i["capacity_mw"] for i in self.fleet_composition)
            print(f"    合计 {len(self.turbines)} 台, {total_cap:.2f} MW")

    def _label_for(self, i: int) -> str:
        """机组在图表/文本中的稳定标签。"""
        return self.turbine_ids[i] if self.turbine_ids[i] is not None else f"#{i}"

    def run_baseline(self) -> None:
        """运行基线（规则网格布局）评估。"""
        self._print_header("步骤 1/6: 生成并评估基线布局")

        rng = np.random.default_rng(self.config.optimization.seed)

        manifest_positions = self.config.manifest_positions()
        if manifest_positions is not None:
            # 逐机清单显式给出了全部坐标：直接评估该固定布局
            if manifest_positions.shape[0] != len(self.turbines):
                raise ConfigError(
                    f"清单中坐标数量 ({manifest_positions.shape[0]}) 与机组数量 "
                    f"({len(self.turbines)}) 不匹配，请逐机核对 position"
                )
            self.baseline_positions = manifest_positions
            feasible, problems = verify_layout_feasible(
                self.baseline_positions, self.boundary, self.spacing_matrix
            )
            if not feasible:
                detail = "\n  ".join(problems[:10])
                more = "" if len(problems) <= 10 else f"\n  ...另有 {len(problems) - 10} 处"
                raise ConfigError(
                    "逐机清单给定的坐标不满足场地或间距约束：\n  "
                    + detail + more
                    + "\n请调整坐标，或删除 position 交由基线/优化布置"
                )
            print(f"使用逐机清单给定的 {len(self.turbines)} 台机组固定坐标")
            self._manifest_positions_fixed = True
        else:
            self._manifest_positions_fixed = False
            self.baseline_positions = generate_grid_layout(
                boundary=self.boundary,
                n_turbines=len(self.turbines),
                rotor_diameters=self.rotor_diameters,
                min_multiple=self.config.optimization.min_spacing_multiple,
                rng=rng,
                spacing_matrix=self.spacing_matrix,
            )
            print(f"已生成 {len(self.turbines)} 台机组的网格布局")

        self._print_fleet_composition()
        self._print_pairwise_spacing_summary()

        self.baseline_result = self.aep_calc.compute_farm_aep(self.baseline_positions)
        self._print_result_summary(self.baseline_result, "基线布局")

    def _print_pairwise_spacing_summary(self) -> None:
        """打印逐对间距规则概况。"""
        mult = self.config.optimization.min_spacing_multiple
        unique_d = np.unique(self.rotor_diameters)
        if len(unique_d) == 1:
            req = mult * unique_d[0]
            print(f"  间距规则: {mult:.1f} × Ø{unique_d[0]:.0f} m = {req:.0f} m（全场统一）")
        else:
            off_diag = self.spacing_matrix[~np.eye(len(self.turbines), dtype=bool)]
            print(
                f"  间距规则: {mult:.1f} × (Ø_i + Ø_j)/2 逐对计算，"
                f"范围 {off_diag.min():.0f} ~ {off_diag.max():.0f} m"
            )

    def run_optimization(self) -> None:
        """运行机位优化。"""
        self._print_header("步骤 2/6: 执行机位布局优化")

        if self._manifest_positions_fixed:
            print("逐机清单已显式给定全部机组坐标，跳过位置优化（按固定布局评估）。")
            return

        fit_fn = self.aep_calc.evaluate_layout

        algo = self.config.optimization.algorithm.lower()

        if algo == "ga":
            ga_config = GAConfig(
                population_size=self.config.optimization.population_size,
                max_generations=self.config.optimization.max_iterations,
                min_spacing_multiple=self.config.optimization.min_spacing_multiple,
                seed=self.config.optimization.seed,
            )
            optimizer = GeneticAlgorithm(
                n_turbines=len(self.turbines),
                rotor_diameters=self.rotor_diameters,
                boundary=self.boundary,
                fitness_fn=fit_fn,
                config=ga_config,
                spacing_matrix=self.spacing_matrix,
            )
        elif algo == "pso":
            pso_config = PSOConfig(
                swarm_size=self.config.optimization.population_size,
                max_iterations=self.config.optimization.max_iterations,
                min_spacing_multiple=self.config.optimization.min_spacing_multiple,
                seed=self.config.optimization.seed,
            )
            optimizer = ParticleSwarmOptimizer(
                n_turbines=len(self.turbines),
                rotor_diameters=self.rotor_diameters,
                boundary=self.boundary,
                fitness_fn=fit_fn,
                config=pso_config,
                spacing_matrix=self.spacing_matrix,
            )
        else:
            raise ValueError(f"未知的优化算法: {algo}")

        print(f"使用优化算法: {algo.upper()}")
        self.optimize_result = optimizer.optimize(verbose=True)

        self.optimized_positions = self.optimize_result.best_positions
        self.optimized_result = self.aep_calc.compute_farm_aep(self.optimized_positions)

        print("\n--- 优化后结果 ---")
        self._print_result_summary(self.optimized_result, "优化后布局")

        if self.baseline_result is not None:
            improvement = (
                (self.optimized_result.net_aep - self.baseline_result.net_aep)
                / self.baseline_result.net_aep
                * 100
            )
            loss_reduction = (
                (self.baseline_result.wake_loss_pct - self.optimized_result.wake_loss_pct)
                / self.baseline_result.wake_loss_pct
                * 100
            )
            print(f"\n--- 优化提升 ---")
            print(f"  发电量提升:   {improvement:+.2f}%")
            print(f"  尾流损失减少: {loss_reduction:+.2f}%")
            print(f"  额外发电量:   {(self.optimized_result.net_aep - self.baseline_result.net_aep)/1e3:+.2f} GWh/年")

    def run_economic_analysis(self) -> None:
        """运行经济性分析。"""
        if not self.config.economic.enable_analysis:
            return

        self._print_header("步骤 3/6: 经济性分析")

        if self.optimized_result is None:
            print("警告: 未进行优化，使用基线布局进行经济性分析")
            result = self.baseline_result
        else:
            result = self.optimized_result

        cost_models = self.config.build_cost_models()
        first_model = self.fleet_composition[0]["model"]
        turbine_cost = cost_models[first_model]
        farm_cost = get_default_farm_cost()
        farm_cost.discount_rate = self.config.economic.discount_rate

        analyzer = EconomicAnalyzer(
            turbine_cost=turbine_cost,
            farm_cost=farm_cost,
            electricity_price=self.config.economic.electricity_price,
            turbine_costs=cost_models,
        )

        if len(self.fleet_composition) == 1:
            # 单一机型：沿用原核算路径，结果与旧版本完全一致
            item = self.fleet_composition[0]
            rated_power_MW = item["rated_power_kw"] / 1e3
            self.economic_result = analyzer.analyze(
                n_turbines=item["count"],
                rated_power_per_turbine_MW=rated_power_MW,
                net_aep_GWh=result.net_aep / 1e3,
            )
        else:
            fleet_items = [
                {
                    "model": item["model"],
                    "count": item["count"],
                    "rated_power_mw": item["rated_power_kw"] / 1e3,
                }
                for item in self.fleet_composition
            ]
            self.economic_result = analyzer.analyze_fleet(
                fleet_items=fleet_items,
                net_aep_GWh=result.net_aep / 1e3,
            )

        print(f"\n--- 经济性分析结果（基于优化后布局） ---")
        print(f"  上网电价:      {self.config.economic.electricity_price:.2f} 元/kWh")
        print(f"  折现率:        {self.config.economic.discount_rate*100:.1f}%")
        print(f"  初始投资:      {self.economic_result.total_capital_cost/1e4:.2f} 亿元")
        print(f"  年运维费用:    {self.economic_result.total_om_cost_annual:.1f} 万元/年")
        print(f"  年发电收益:    {self.economic_result.annual_revenue:.1f} 万元/年")
        print(f"  度电成本:      {self.economic_result.lcoe:.3f} 元/kWh")

        if self.economic_result.npv is not None:
            print(f"  净现值(NPV):   {self.economic_result.npv/1e4:+.2f} 亿元")
        if self.economic_result.irr is not None:
            print(f"  内部收益率:    {self.economic_result.irr:.2f}%")
        if self.economic_result.payback_period is not None:
            print(f"  投资回收期:    {self.economic_result.payback_period:.1f} 年")

        print(f"\n  成本构成:")
        for item, cost in self.economic_result.cost_breakdown.items():
            pct = cost / self.economic_result.total_capital_cost * 100
            print(f"    {item}: {cost/1e4:.2f} 亿元 ({pct:.1f}%)")

        if self.economic_result.model_breakdown:
            print(f"\n  分型号明细:")
            for mb in self.economic_result.model_breakdown:
                print(
                    f"    {mb['model']}: {mb['count']} 台 × {mb['rated_power_mw']:.2f} MW "
                    f"= {mb['capacity_mw']:.2f} MW; "
                    f"设备 {mb['capital_cost']:.0f} 万元, "
                    f"安装 {mb['installation_cost']:.0f} 万元, "
                    f"年运维 {mb['annual_om_cost']:.0f} 万元/年"
                )

    def run_turbine_sweep(self, min_turbines: int = 5, max_turbines: int = 25, step: int = 2) -> None:
        """运行风机台数扫描分析（仅单一机型配置）。"""
        self._print_header("步骤 4/6: 风机台数扫描分析")

        if self.config.fleet_mode != "single" or len(self.fleet_composition) != 1:
            print("跳过台数扫描：当前为多机型/逐机清单配置，台数扫描仅支持单一机型。")
            print("如需分析不同规模，请调整 fleet 配置后分别运行。")
            return

        print(f"扫描范围: {min_turbines} ~ {max_turbines} 台，步长 {step}")
        print("此分析将为不同台数快速评估布局与经济性")

        sweep_data = {
            "n_turbines": [],
            "aep": [],
            "lcoe": [],
        }

        rng = np.random.default_rng(self.config.optimization.seed)

        # 保存现场状态，扫描结束后完整恢复，避免污染后续图表与结果保存
        saved = {
            "turbines": self.turbines,
            "rotor_diameters": self.rotor_diameters,
            "rated_powers": self.rated_powers,
            "thrust_coefficients": self.thrust_coefficients,
            "spacing_matrix": self.spacing_matrix,
            "aep_calc": self.aep_calc,
        }

        model = self.fleet_composition[0]["model"]
        cost_models = self.config.build_cost_models()
        farm_cost = get_default_farm_cost()
        analyzer = EconomicAnalyzer(
            turbine_cost=cost_models[model],
            farm_cost=farm_cost,
            electricity_price=self.config.economic.electricity_price,
            turbine_costs=cost_models,
        )

        base_turbine = self.turbines[0]

        try:
            for n in range(min_turbines, max_turbines + 1, step):
                print(f"\n  分析 {n} 台机组...")
                width = max(2, len(str(n)))
                self.turbines = [
                    create_turbine_by_name(
                        model, turbine_id=f"WT-{k + 1:0{width}d}",
                        extra_models=self.config.turbine_catalog,
                    )
                    for k in range(n)
                ]
                self.rotor_diameters = np.array([t.rotor_diameter for t in self.turbines])
                self.rated_powers = np.array([t.rated_power for t in self.turbines])
                self.thrust_coefficients = np.array([t.thrust_coefficient for t in self.turbines])
                self.spacing_matrix = pairwise_min_spacings(
                    self.rotor_diameters,
                    self.config.optimization.min_spacing_multiple,
                )

                self.aep_calc = AEPCalculator(
                    turbines=self.turbines,
                    wind_resource=self.wind_resource,
                    wake_model=self.wake_model,
                    wake_superposition=self.config.superposition_method,
                )

                try:
                    positions = generate_grid_layout(
                        boundary=self.boundary,
                        n_turbines=n,
                        rotor_diameters=self.rotor_diameters,
                        min_multiple=self.config.optimization.min_spacing_multiple,
                        rng=rng,
                        spacing_matrix=self.spacing_matrix,
                    )

                    result = self.aep_calc.compute_farm_aep(positions)

                    rated_power_MW = base_turbine.rated_power / 1e3
                    econ_result = analyzer.analyze(
                        n_turbines=n,
                        rated_power_per_turbine_MW=rated_power_MW,
                        net_aep_GWh=result.net_aep / 1e3,
                    )

                    sweep_data["n_turbines"].append(n)
                    sweep_data["aep"].append(result.net_aep)
                    sweep_data["lcoe"].append(econ_result.lcoe)

                    print(f"    净AEP: {result.net_aep/1e3:.1f} GWh, LCOE: {econ_result.lcoe:.3f} 元/kWh")
                except Exception as e:
                    print(f"    跳过: {e}")
        finally:
            self.turbines = saved["turbines"]
            self.rotor_diameters = saved["rotor_diameters"]
            self.rated_powers = saved["rated_powers"]
            self.thrust_coefficients = saved["thrust_coefficients"]
            self.spacing_matrix = saved["spacing_matrix"]
            self.aep_calc = saved["aep_calc"]

        self.sweep_results = sweep_data

    def run_visualization(self) -> None:
        """生成所有可视化图表。"""
        self._print_header("步骤 5/6: 生成可视化图表")

        save_dir = self.config.visualization.save_dir
        save = self.config.visualization.save_plots
        show = self.config.visualization.show_plots

        if save:
            print("图表将保存到:", os.path.abspath(save_dir))

        plot_wind_rose(
            wind_resource=self.wind_resource,
            title="项目场址风玫瑰图",
            save_path=os.path.join(save_dir, "wind_rose.png") if save else None,
            show=show,
        )

        labels = [self._label_for(i) for i in range(len(self.turbines))]
        mixed_models = self.turbine_models if len(set(self.turbine_models)) > 1 else None

        if self.baseline_positions is not None and self.baseline_result is not None:
            baseline_losses = np.array([tr.wake_loss_pct for tr in self.baseline_result.turbine_results])
            plot_farm_layout(
                positions=self.baseline_positions,
                boundary=self.boundary,
                rotor_diameters=self.rotor_diameters,
                turbine_losses=baseline_losses,
                turbine_names=labels,
                turbine_models=mixed_models,
                model_info=self._model_info,
                title="基线布局 - 尾流损失分布",
                save_path=os.path.join(save_dir, "baseline_layout.png") if save else None,
                show=show,
            )

            plot_turbine_loss_bar(
                farm_result=self.baseline_result,
                title="基线布局 - 各机组尾流损失",
                save_path=os.path.join(save_dir, "baseline_losses.png") if save else None,
                show=show,
            )

        if self.optimized_positions is not None and self.optimized_result is not None:
            opt_losses = np.array([tr.wake_loss_pct for tr in self.optimized_result.turbine_results])
            plot_farm_layout(
                positions=self.optimized_positions,
                boundary=self.boundary,
                rotor_diameters=self.rotor_diameters,
                turbine_losses=opt_losses,
                turbine_names=labels,
                turbine_models=mixed_models,
                model_info=self._model_info,
                title="优化后布局 - 尾流损失分布",
                save_path=os.path.join(save_dir, "optimized_layout.png") if save else None,
                show=show,
            )

            plot_turbine_loss_bar(
                farm_result=self.optimized_result,
                title="优化后布局 - 各风机尾流损失",
                save_path=os.path.join(save_dir, "optimized_losses.png") if save else None,
                show=show,
            )

        if self.optimize_result is not None and self.baseline_result is not None:
            plot_convergence(
                optimize_result=self.optimize_result,
                baseline_aep=self.baseline_result.net_aep,
                title="优化收敛曲线",
                save_path=os.path.join(save_dir, "convergence.png") if save else None,
                show=show,
            )

        if self.baseline_result is not None and self.optimized_result is not None:
            plot_comparison(
                baseline_result=self.baseline_result,
                optimized_result=self.optimized_result,
                title="优化前后关键指标对比",
                save_path=os.path.join(save_dir, "comparison.png") if save else None,
                show=show,
            )

        if self.sweep_results is not None:
            plot_aep_vs_turbines(
                n_turbines_list=self.sweep_results["n_turbines"],
                aep_list=self.sweep_results["aep"],
                lcoe_list=self.sweep_results["lcoe"],
                title="风机台数优化分析",
                save_path=os.path.join(save_dir, "aep_vs_turbines.png") if save else None,
                show=show,
            )

        if self.config.visualization.plot_wake_heatmap and self.optimized_positions is not None:
            dominant_dir = self.wind_resource.directions[np.argmax(self.wind_resource.frequencies)]
            plot_wake_heatmap(
                positions=self.optimized_positions,
                boundary=self.boundary,
                wake_model=self.wake_model,
                wind_direction=dominant_dir,
                rotor_diameters=self.rotor_diameters,
                thrust_coefficients=self.thrust_coefficients,
                title=f"主风向({dominant_dir:.0f}°)尾流速度亏损分布",
                save_path=os.path.join(save_dir, "wake_heatmap.png") if save else None,
                show=show,
            )

    def save_results(self) -> None:
        """保存所有结果到JSON文件。"""
        self._print_header("步骤 6/6: 保存结果数据")

        output_dir = self.config.visualization.save_dir

        results = {
            "config": {
                "n_turbines": self.config.n_turbines,
                "turbine_model": self.config.turbine_model,
                "fleet_mode": self.config.fleet_mode,
                "wake_model": self.config.wake_model,
                "min_spacing_multiple": self.config.optimization.min_spacing_multiple,
            },
            "fleet_composition": self.fleet_composition,
            "site": {
                "area_km2": float(self.boundary.area / 1e6),
                "mean_wind_speed": float(self.wind_resource.overall_mean_speed),
            },
        }

        def _turbine_losses(farm_result):
            return [
                {
                    "idx": tr.turbine_idx,
                    "turbine_id": tr.turbine_id,
                    "model": tr.name,
                    "rated_power_kw": tr.rated_power_kw,
                    "rotor_diameter": tr.rotor_diameter,
                    "gross_aep_mwh": tr.gross_aep,
                    "net_aep_mwh": tr.net_aep,
                    "wake_loss_pct": float(tr.wake_loss_pct),
                    "capacity_factor_pct": float(tr.capacity_factor),
                    "dominant_source": tr.dominant_wake_source,
                }
                for tr in farm_result.turbine_results
            ]

        if self.baseline_result is not None:
            results["baseline"] = {
                "positions": self.baseline_positions.tolist() if self.baseline_positions is not None else None,
                "gross_aep_gwh": float(self.baseline_result.gross_aep / 1e3),
                "net_aep_gwh": float(self.baseline_result.net_aep / 1e3),
                "wake_loss_pct": float(self.baseline_result.wake_loss_pct),
                "capacity_factor": float(self.baseline_result.capacity_factor),
                "model_summary": self.baseline_result.model_summary,
                "turbine_losses": _turbine_losses(self.baseline_result),
            }

        if self.optimized_result is not None:
            results["optimized"] = {
                "positions": self.optimized_positions.tolist() if self.optimized_positions is not None else None,
                "gross_aep_gwh": float(self.optimized_result.gross_aep / 1e3),
                "net_aep_gwh": float(self.optimized_result.net_aep / 1e3),
                "wake_loss_pct": float(self.optimized_result.wake_loss_pct),
                "capacity_factor": float(self.optimized_result.capacity_factor),
                "model_summary": self.optimized_result.model_summary,
                "turbine_losses": _turbine_losses(self.optimized_result),
            }

        if self.economic_result is not None:
            results["economic"] = {
                "total_capital_cost_yiyuan": float(self.economic_result.total_capital_cost / 1e4),
                "annual_revenue_wanyuan": float(self.economic_result.annual_revenue),
                "lcoe_yuan_per_kwh": float(self.economic_result.lcoe),
                "npv_yiyuan": float(self.economic_result.npv / 1e4) if self.economic_result.npv is not None else None,
                "irr_pct": float(self.economic_result.irr) if self.economic_result.irr is not None else None,
                "payback_years": float(self.economic_result.payback_period) if self.economic_result.payback_period is not None else None,
                "model_breakdown": self.economic_result.model_breakdown,
            }

        if self.baseline_result is not None and self.optimized_result is not None:
            results["improvement"] = {
                "aep_improvement_pct": float(
                    (self.optimized_result.net_aep - self.baseline_result.net_aep)
                    / self.baseline_result.net_aep * 100
                ),
                "additional_aep_gwh": float(
                    (self.optimized_result.net_aep - self.baseline_result.net_aep) / 1e3
                ),
                "loss_reduction_pct": float(
                    (self.baseline_result.wake_loss_pct - self.optimized_result.wake_loss_pct)
                    / self.baseline_result.wake_loss_pct * 100
                ),
            }

        if self.sweep_results is not None:
            results["turbine_sweep"] = {
                "n_turbines": self.sweep_results["n_turbines"],
                "aep_mwh": self.sweep_results["aep"],
                "lcoe_yuan_per_kwh": self.sweep_results["lcoe"],
            }

        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        config_path = os.path.join(output_dir, "config.json")
        self.config.to_json(config_path)

        print(f"结果已保存到: {os.path.abspath(results_path)}")
        print(f"配置已保存到: {os.path.abspath(config_path)}")

    def run_full_analysis(
        self,
        run_baseline: bool = True,
        run_opt: bool = True,
        run_econ: bool = True,
        run_sweep: bool = False,
        run_viz: bool = True,
        save: bool = True,
    ) -> None:
        """运行完整分析流程。"""
        start_time = time.time()

        self._print_header("风电场机位布局优化分析")
        if self.config.fleet_mode == "single":
            print(f"  风机: {self.config.turbine_model} × {self.config.n_turbines} 台")
        else:
            print(f"  机队声明模式: {self.config.fleet_mode}，共 {self.config.n_turbines} 台")
            self._print_fleet_composition()
        print(f"  尾流模型: {self.config.wake_model}")
        print(f"  平均风速: {self.wind_resource.overall_mean_speed:.2f} m/s")
        print(f"  场地面积: {self.boundary.area / 1e6:.2f} km²")

        if run_baseline:
            self.run_baseline()

        if run_opt:
            self.run_optimization()

        if run_econ:
            self.run_economic_analysis()

        if run_sweep:
            self.run_turbine_sweep(
                min_turbines=getattr(self, '_min_turbines', 5),
                max_turbines=getattr(self, '_max_turbines', 25),
            )

        if run_viz:
            self.run_visualization()

        if save:
            self.save_results()

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  全部分析完成! 耗时: {elapsed:.1f} 秒")
        print(f"{'='*60}\n")


def build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="风电场机位布局优化工具 - 尾流计算、布局优化、经济性评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置运行完整分析
  python -m wind_farm_opt

  # 从配置文件运行
  python -m wind_farm_opt --config my_config.json

  # 自定义参数运行
  python -m wind_farm_opt --n-turbines 20 --turbine V164-9.5MW --wake-model gaussian

  # 仅评估不优化
  python -m wind_farm_opt --no-optimization

  # 启用风机台数扫描
  python -m wind_farm_opt --sweep --min-turbines 10 --max-turbines 30
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径(JSON格式)",
    )

    parser.add_argument(
        "--n-turbines",
        type=int,
        default=None,
        help="风机台数",
    )

    parser.add_argument(
        "--turbine",
        type=str,
        default=None,
        choices=["V126-3.45MW", "V164-9.5MW"],
        help="风机型号",
    )

    parser.add_argument(
        "--wake-model",
        type=str,
        default=None,
        choices=["jensen", "gaussian"],
        help="尾流模型: jensen 或 gaussian",
    )

    parser.add_argument(
        "--wake-decay",
        type=float,
        default=None,
        help="尾流衰减系数 (Jensen模型)",
    )

    parser.add_argument(
        "--boundary",
        type=str,
        default=None,
        choices=["rectangular", "hexagonal", "irregular"],
        help="场地边界类型",
    )

    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="矩形场地宽度 (m)",
    )

    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="矩形场地高度 (m)",
    )

    parser.add_argument(
        "--min-spacing",
        type=float,
        default=None,
        help="最小间距倍数（转子直径倍数）",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["ga", "pso"],
        help="优化算法: ga(遗传算法) 或 pso(粒子群)",
    )

    parser.add_argument(
        "--population",
        type=int,
        default=None,
        help="种群/粒子群大小",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="最大迭代代数",
    )

    parser.add_argument(
        "--no-optimization",
        action="store_true",
        help="仅评估基线布局，不执行优化",
    )

    parser.add_argument(
        "--no-economic",
        action="store_true",
        help="跳过经济性分析",
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="启用风机台数扫描分析",
    )

    parser.add_argument(
        "--min-turbines",
        type=int,
        default=5,
        help="台数扫描最小值",
    )

    parser.add_argument(
        "--max-turbines",
        type=int,
        default=25,
        help="台数扫描最大值",
    )

    parser.add_argument(
        "--electricity-price",
        type=float,
        default=None,
        help="上网电价 (元/kWh)",
    )

    parser.add_argument(
        "--discount-rate",
        type=float,
        default=None,
        help="折现率 (0-1)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="不生成图表",
    )

    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="显示图表窗口",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子",
    )

    parser.add_argument(
        "--generate-config",
        type=str,
        default=None,
        help="生成示例配置文件并退出",
    )

    return parser


def main() -> int:
    """主函数入口。"""
    parser = build_argparser()
    args = parser.parse_args()

    if args.generate_config:
        config = create_sample_config()
        config.to_json(args.generate_config)
        print(f"示例配置已生成: {os.path.abspath(args.generate_config)}")
        return 0

    try:
        if args.config:
            config = WindFarmConfig.from_json(args.config)
        else:
            config = create_sample_config()

        # 多机型 fleet 配置与单机覆盖参数互斥，避免数量/型号语义冲突
        fleet_overrides = []
        if args.n_turbines is not None:
            fleet_overrides.append("--n-turbines")
        if args.turbine is not None:
            fleet_overrides.append("--turbine")
        if fleet_overrides and config.fleet_mode in ("types", "manifest"):
            raise ConfigError(
                f"参数 {'、'.join(fleet_overrides)} 仅适用于单一机型配置，"
                "不能与配置文件中的 fleet 块同时使用。请在 fleet 块内"
                "修改机型/数量，或删除 fleet 块。"
            )

        if args.n_turbines is not None:
            config.n_turbines = args.n_turbines
        if args.turbine is not None:
            config.turbine_model = args.turbine
        if args.wake_model is not None:
            config.wake_model = args.wake_model
        if args.wake_decay is not None:
            config.wake_decay = args.wake_decay
        if args.boundary is not None:
            config.boundary_type = args.boundary
        if args.width is not None:
            config.boundary_params["width"] = args.width
        if args.height is not None:
            config.boundary_params["height"] = args.height
        if args.min_spacing is not None:
            config.optimization.min_spacing_multiple = args.min_spacing
        if args.algorithm is not None:
            config.optimization.algorithm = args.algorithm
        if args.population is not None:
            config.optimization.population_size = args.population
        if args.iterations is not None:
            config.optimization.max_iterations = args.iterations
        if args.seed is not None:
            config.optimization.seed = args.seed
        if args.electricity_price is not None:
            config.economic.electricity_price = args.electricity_price
        if args.discount_rate is not None:
            config.economic.discount_rate = args.discount_rate
        if args.output_dir is not None:
            config.visualization.save_dir = args.output_dir
        if args.no_plots:
            config.visualization.save_plots = False
        if args.show_plots:
            config.visualization.show_plots = True
        if args.no_economic:
            config.economic.enable_analysis = False

        # 命令行覆盖后重新校验（如 n_turbines 合法性、型号一致性）
        config.validate()

        cli = WindFarmOptimizerCLI(config)
        cli._min_turbines = args.min_turbines
        cli._max_turbines = args.max_turbines
    except ConfigError as e:
        print(f"\n配置错误: {e}", file=sys.stderr)
        return 2

    try:
        cli.run_full_analysis(
            run_baseline=True,
            run_opt=not args.no_optimization,
            run_econ=not args.no_economic,
            run_sweep=args.sweep,
            run_viz=not args.no_plots,
            save=True,
        )
        return 0
    except ConfigError as e:
        print(f"\n配置错误: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
