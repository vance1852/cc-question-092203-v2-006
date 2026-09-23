"""快速测试脚本 - 用于验证核心功能。"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

print("=" * 60)
print("风电场布局优化工具 - 快速测试")
print("=" * 60)

print("\n1. 测试风机模型...")
from wind_farm_opt.core.turbine import create_default_turbine
turb = create_default_turbine("V126-3.45MW")
print(f"   ✓ 风机: {turb.name}")
print(f"   ✓ 额定功率: {turb.rated_power/1e3:.2f} MW")
print(f"   ✓ 转子直径: {turb.rotor_diameter:.1f} m")
print(f"   ✓ 切入风速: {turb.cut_in_speed:.1f} m/s")
print(f"   ✓ 额定风速: {turb.rated_speed:.1f} m/s")
print(f"   ✓ 切出风速: {turb.cut_out_speed:.1f} m/s")

print("\n2. 测试风资源模型...")
from wind_farm_opt.core.wind_resource import create_default_wind_resource
wr = create_default_wind_resource(num_sectors=12, dominant_direction=270.0, mean_speed=8.5)
print(f"   ✓ 扇区数: {wr.num_sectors}")
print(f"   ✓ 加权平均风速: {wr.overall_mean_speed:.2f} m/s")
freq_sort_idx = np.argsort(wr.frequencies)[::-1]
print(f"   ✓ 主风向: {wr.directions[freq_sort_idx[0]]:.0f}° ({wr.frequencies[freq_sort_idx[0]]*100:.1f}%)")

print("\n3. 测试尾流模型...")
from wind_farm_opt.core.wake import JensenWake, GaussianWake, superpose_wakes
jensen = JensenWake(wake_decay=0.07)
gaussian = GaussianWake(wake_decay=0.035)

dists = np.array([5.0, 10.0, 15.0]) * turb.rotor_diameter
d0 = turb.rotor_diameter
ct = turb.thrust_coefficient

jensen_def = jensen.velocity_deficit(dists, d0, ct)
gauss_def = gaussian.velocity_deficit(dists, d0, ct)
print(f"   ✓ Jensen尾流模型 (5D/10D/15D): {jensen_def[0]:.3f}, {jensen_def[1]:.3f}, {jensen_def[2]:.3f}")
print(f"   ✓ 高斯尾流模型 (5D/10D/15D): {gauss_def[0]:.3f}, {gauss_def[1]:.3f}, {gauss_def[2]:.3f}")

deficits = np.array([0.1, 0.05, 0.08])
total_def = superpose_wakes(deficits, method="sum_of_squares")
print(f"   ✓ 平方和叠加: {total_def:.4f} (线性叠加: {deficits.sum():.4f})")

print("\n4. 测试场地边界...")
from wind_farm_opt.constraints.boundary import (
    create_rectangular_boundary,
    create_irregular_boundary,
)
boundary = create_rectangular_boundary(width=4000, height=4000)
print(f"   ✓ 矩形场地面积: {boundary.area/1e6:.2f} km²")

irregular = create_irregular_boundary()
print(f"   ✓ 不规则场地面积: {irregular.area/1e6:.2f} km²")

test_point = np.array([1000.0, 1000.0])
test_outside = np.array([-5000.0, -5000.0])
print(f"   ✓ 点在边界内: {boundary.contains_point(test_point)}")
print(f"   ✓ 点在边界外: {boundary.contains_point(test_outside)}")

print("\n5. 测试间距约束...")
from wind_farm_opt.constraints.spacing import (
    check_min_spacing,
    compute_min_spacing_from_diameters,
)
diameters = np.array([turb.rotor_diameter, turb.rotor_diameter])
min_space = compute_min_spacing_from_diameters(diameters, min_multiple=5.0)
print(f"   ✓ 最小间距 (5D): {min_space:.1f} m")

good_positions = np.array([[0, 0], [1000, 0]])
bad_positions = np.array([[0, 0], [500, 0]])
valid, violations = check_min_spacing(good_positions, min_space)
print(f"   ✓ 合格布局检查: {valid}")
valid, violations = check_min_spacing(bad_positions, min_space)
print(f"   ✓ 违规布局检查: {valid}, 违规对数: {len(violations)}")

print("\n6. 测试AEP计算（12台风机，50x50网格分辨率）...")
from wind_farm_opt.farm.aep import AEPCalculator
from wind_farm_opt.optimization.baseline import generate_grid_layout

n_turb = 12
turbines = [create_default_turbine("V126-3.45MW") for _ in range(n_turb)]
rotor_diameters = np.array([t.rotor_diameter for t in turbines])

boundary = create_rectangular_boundary(3500, 3500)
rng = np.random.default_rng(42)

positions = generate_grid_layout(boundary, n_turb, rotor_diameters, min_multiple=5.0, rng=rng)
print(f"   ✓ 生成 {n_turb} 台风机网格布局")

wake_model = JensenWake(0.07)
aep_calc = AEPCalculator(
    turbines=turbines,
    wind_resource=wr,
    wake_model=wake_model,
    wake_superposition="sum_of_squares",
    speed_step=1.0,
)

result = aep_calc.compute_farm_aep(positions)
print(f"   ✓ 装机容量: {result.total_installed_capacity:.2f} MW")
print(f"   ✓ 理论AEP: {result.gross_aep:.2f} MWh/年")
print(f"   ✓ 净AEP: {result.net_aep:.2f} MWh/年")
print(f"   ✓ 尾流损失: {result.wake_loss_pct:.2f}%")
print(f"   ✓ 容量系数: {result.capacity_factor:.2f}%")

max_loss = max(result.turbine_results, key=lambda x: x.wake_loss_pct)
print(f"   ✓ 最大损失风机: #{max_loss.turbine_idx} ({max_loss.wake_loss_pct:.1f}%)")
if max_loss.dominant_wake_source is not None:
    print(f"     主要影响源: #{max_loss.dominant_wake_source}")

print("\n7. 测试优化算法（小规模快速测试）...")
from wind_farm_opt.optimization.ga import GeneticAlgorithm, GAConfig

ga_config = GAConfig(
    population_size=10,
    max_generations=5,
    min_spacing_multiple=5.0,
    seed=42,
)

fitness_fn = aep_calc.evaluate_layout
ga = GeneticAlgorithm(
    n_turbines=n_turb,
    rotor_diameters=rotor_diameters,
    boundary=boundary,
    fitness_fn=fitness_fn,
    config=ga_config,
)

opt_result = ga.optimize(verbose=False)
print(f"   ✓ 遗传算法优化完成")
print(f"   ✓ 最优净AEP: {opt_result.best_fitness:.2f} MWh")
print(f"   ✓ 基线净AEP: {result.net_aep:.2f} MWh")
improvement = (opt_result.best_fitness - result.net_aep) / result.net_aep * 100
print(f"   ✓ 提升: {improvement:+.2f}%")

print("\n8. 测试经济性分析...")
from wind_farm_opt.economy.costs import (
    EconomicAnalyzer,
    get_default_turbine_cost,
    get_default_farm_cost,
)

turb_cost = get_default_turbine_cost("V126-3.45MW")
farm_cost = get_default_farm_cost()
analyzer = EconomicAnalyzer(turb_cost, farm_cost, electricity_price=0.45)

econ_result = analyzer.analyze(
    n_turbines=n_turb,
    rated_power_per_turbine_MW=turb.rated_power/1e3,
    net_aep_GWh=opt_result.best_fitness/1e3,
)
print(f"   ✓ 度电成本(LCOE): {econ_result.lcoe:.3f} 元/kWh")
print(f"   ✓ 初始投资: {econ_result.total_capital_cost/1e4:.2f} 亿元")
print(f"   ✓ 年收益: {econ_result.annual_revenue:.0f} 万元")
if econ_result.payback_period:
    print(f"   ✓ 投资回收期: {econ_result.payback_period:.1f} 年")
if econ_result.irr:
    print(f"   ✓ 内部收益率: {econ_result.irr:.2f}%")

print("\n9. 测试可视化模块...")
from wind_farm_opt.visualization.plotting import (
    plot_farm_layout,
    plot_wind_rose,
    plot_convergence,
    plot_turbine_loss_bar,
)

os.makedirs("test_output", exist_ok=True)

plot_wind_rose(wr, save_path="test_output/wind_rose.png", show=False)
print("   ✓ 风玫瑰图已生成")

losses = np.array([tr.wake_loss_pct for tr in result.turbine_results])
plot_farm_layout(
    positions, boundary, rotor_diameters,
    turbine_losses=losses,
    turbine_names=[f"#{i}" for i in range(n_turb)],
    save_path="test_output/layout.png",
    show=False,
)
print("   ✓ 布局图已生成")

plot_convergence(
    opt_result,
    baseline_aep=result.net_aep * 1e3,
    save_path="test_output/convergence.png",
    show=False,
)
print("   ✓ 收敛曲线已生成")

plot_turbine_loss_bar(
    result,
    save_path="test_output/losses.png",
    show=False,
)
print("   ✓ 损失柱状图已生成")

print("\n" + "=" * 60)
print("多机型混装测试")
print("=" * 60)

print("\n10. 测试多机型机队配置与独立实例...")
from wind_farm_opt.config import WindFarmConfig, ConfigError, VisualizationConfig
from wind_farm_opt.constraints.spacing import pairwise_min_spacings

_no_plots = VisualizationConfig(save_plots=False)
fleet_cfg = WindFarmConfig(
    boundary_type="rectangular",
    boundary_params={"width": 4000, "height": 3500},
    visualization=_no_plots,
)
fleet_cfg.load_fleet_block({"turbine_types": [
    {"model": "V126-3.45MW", "count": 3},
    {"model": "V164-9.5MW", "count": 2},
]})
fleet_turbines = fleet_cfg.create_turbines()
assert len(fleet_turbines) == 5
assert [t.turbine_id for t in fleet_turbines] == ["WT-01", "WT-02", "WT-03", "WT-04", "WT-05"]
assert [t.name for t in fleet_turbines] == ["V126-3.45MW"] * 3 + ["V164-9.5MW"] * 2
# 实例与功率曲线互相独立
assert fleet_turbines[0] is not fleet_turbines[1]
assert fleet_turbines[0].power_curve is not fleet_turbines[1].power_curve
comp = fleet_cfg.fleet_composition(fleet_turbines)
assert comp[0]["count"] == 3 and comp[1]["count"] == 2
assert abs(comp[0]["capacity_mw"] - 3 * 3.45) < 1e-9
assert abs(comp[1]["capacity_mw"] - 2 * 9.5) < 1e-9
print("   ✓ 按型号数量生成 5 台独立机组，稳定编号连续")
print("   ✓ 机队容量明细: "
      + ", ".join(f"{c['model']}×{c['count']}={c['capacity_mw']:.2f}MW" for c in comp))

# 逐机清单 + 自定义机型
manifest = WindFarmConfig(boundary_params={"width": 4000, "height": 4000},
                          visualization=_no_plots)
manifest.load_fleet_block({
    "turbines": [
        {"id": "OLD-A", "model": "V126-3.45MW"},
        {"id": "BIG-B", "model": "V150-6.0MW"},
    ],
    "catalog": {"V150-6.0MW": {
        "rotor_diameter": 150.0, "rated_power_kw": 6000.0,
        "hub_height": 100.0,
    }},
})
units = manifest.create_turbines()
assert [t.turbine_id for t in units] == ["OLD-A", "BIG-B"]
assert units[1].rotor_diameter == 150.0 and units[1].rated_power == 6000.0
print("   ✓ 逐机清单按自定义编号生成，catalog 自定义机型生效")

print("\n11. 测试基于各自直径的逐对间距规则...")
diameters = np.array([t.rotor_diameter for t in fleet_turbines])
S = pairwise_min_spacings(diameters, min_multiple=5.0)
# 同型号对 = 5D；混合对 = 5*(D1+D2)/2
assert abs(S[0, 1] - 5.0 * 126.0) < 1e-9
assert abs(S[3, 4] - 5.0 * 164.0) < 1e-9
assert abs(S[0, 3] - 5.0 * (126.0 + 164.0) / 2.0) < 1e-9
assert S[0, 3] < S[3, 4], "小-大机组对间距应小于大-大机组对，避免场地浪费"
# 同直径机队退化为旧的全场统一 5D
S_homo = pairwise_min_spacings(np.full(4, 126.0), 5.0)
assert np.allclose(S_homo[S_homo > 0], 5.0 * 126.0)
print(f"   ✓ 小-小 {S[0,1]:.0f} m, 小-大 {S[0,3]:.0f} m, 大-大 {S[3,4]:.0f} m")
print("   ✓ 同型号机队逐对规则与旧的 k·D 完全等价")

print("\n12. 测试多机型 AEP（各机组自身功率/推力特性）...")
from wind_farm_opt.farm.aep import AEPCalculator
from wind_farm_opt.optimization.baseline import generate_grid_layout

f_boundary = fleet_cfg.create_boundary()
f_wr = fleet_cfg.create_wind_resource()
f_rng = np.random.default_rng(7)
f_pos = generate_grid_layout(
    f_boundary, 5, diameters, min_multiple=5.0, rng=f_rng, spacing_matrix=S,
)
f_aep = AEPCalculator(fleet_turbines, f_wr, JensenWake(0.07), speed_step=1.0)
f_result = f_aep.compute_farm_aep(f_pos)
assert abs(f_result.total_installed_capacity - (3 * 3.45 + 2 * 9.5)) < 1e-9
by_model = f_result.model_summary
assert set(by_model) == {"V126-3.45MW", "V164-9.5MW"}
assert by_model["V164-9.5MW"]["count"] == 2
tr = f_result.turbine_results[3]
assert tr.name == "V164-9.5MW" and tr.turbine_id == "WT-04"
assert abs(tr.rated_power_kw - 9500.0) < 1e-9 and tr.rotor_diameter == 164.0
# 同位置单机 AEP 应只取决于自身功率曲线：大机组毛发电量显著更高
small_gross = f_result.turbine_results[0].gross_aep
big_gross = f_result.turbine_results[3].gross_aep
assert big_gross > small_gross * 2.0
print(f"   ✓ 总装机 {f_result.total_installed_capacity:.2f} MW")
print(f"   ✓ 型号汇总保留: " + ", ".join(
    f"{m} {v['count']}台/{v['net_aep_mwh']/1e3:.1f}GWh" for m, v in by_model.items()))
print(f"   ✓ 同布局大机组毛AEP {big_gross:.0f} vs 小机组 {small_gross:.0f} MWh（各用自身功率曲线）")

print("\n13. 测试多机型 GA/PSO 贯穿逐对间距约束...")
from wind_farm_opt.optimization.ga import GeneticAlgorithm, GAConfig
from wind_farm_opt.optimization.pso import ParticleSwarmOptimizer, PSOConfig
from wind_farm_opt.constraints.spacing import check_pairwise_spacing

mixed_ga = GeneticAlgorithm(
    n_turbines=5, rotor_diameters=diameters, boundary=f_boundary,
    fitness_fn=f_aep.evaluate_layout, spacing_matrix=S,
    config=GAConfig(population_size=6, max_generations=3, seed=1),
)
ga_res = mixed_ga.optimize(verbose=False)
ok_ga, _ = check_pairwise_spacing(ga_res.best_positions, S)
assert ok_ga, "GA 最优解必须满足逐对间距"

mixed_pso = ParticleSwarmOptimizer(
    n_turbines=5, rotor_diameters=diameters, boundary=f_boundary,
    fitness_fn=f_aep.evaluate_layout, spacing_matrix=S,
    config=PSOConfig(swarm_size=6, max_iterations=3, seed=1),
)
pso_res = mixed_pso.optimize(verbose=False)
ok_pso, _ = check_pairwise_spacing(pso_res.best_positions, S)
assert ok_pso, "PSO 最优解必须满足逐对间距"
print(f"   ✓ GA 最优 AEP {ga_res.best_fitness/1e3:.2f} GWh，逐对间距全部满足")
print(f"   ✓ PSO 最优 AEP {pso_res.best_fitness/1e3:.2f} GWh，逐对间距全部满足")

print("\n14. 测试多机型经济性分析（分型号成本与容量）...")
from wind_farm_opt.economy.costs import (
    EconomicAnalyzer,
    get_default_turbine_cost,
    get_default_farm_cost,
)
analyzer_mix = EconomicAnalyzer(
    turbine_cost=get_default_turbine_cost("V126-3.45MW"),
    farm_cost=get_default_farm_cost(),
    electricity_price=0.45,
    turbine_costs=fleet_cfg.build_cost_models(),
)
fleet_items = [
    {"model": "V126-3.45MW", "count": 3, "rated_power_mw": 3.45},
    {"model": "V164-9.5MW", "count": 2, "rated_power_mw": 9.5},
]
econ_mix = analyzer_mix.analyze_fleet(fleet_items, f_result.net_aep / 1e3)
assert len(econ_mix.model_breakdown) == 2
assert abs(sum(m["capacity_mw"] for m in econ_mix.model_breakdown)
           - econ_mix.total_installed_capacity) < 1e-9
cap_cost = sum(m["capital_cost"] for m in econ_mix.model_breakdown)
assert cap_cost > 0
print(f"   ✓ LCOE {econ_mix.lcoe:.3f} 元/kWh，分型号明细 {len(econ_mix.model_breakdown)} 条")

print("\n15. 测试配置诊断的可操作性...")
def expect_config_error(label, data):
    import tempfile, json as _json
    p = tempfile.mktemp(suffix=".json")
    _json.dump(data, open(p, "w"))
    try:
        WindFarmConfig.from_json(p)
        raise AssertionError(f"{label} 应当报错")
    except ConfigError:
        pass

expect_config_error("未知型号", {"fleet": {"turbine_types": [{"model": "NA", "count": 1}]}})
expect_config_error("count非正", {"fleet": {"turbine_types": [{"model": "V126-3.45MW", "count": 0}]}})
expect_config_error("重复编号", {"fleet": {"turbines": [
    {"id": "A", "model": "V126-3.45MW"}, {"id": "A", "model": "V126-3.45MW"}]}})
expect_config_error("部分坐标", {"fleet": {"turbines": [
    {"id": "A", "model": "V126-3.45MW", "position": [0, 0]},
    {"id": "B", "model": "V126-3.45MW"}]}})
print("   ✓ 未知型号 / 非法数量 / 重复编号 / 坐标缺失 均有明确诊断")

# 场地无法容纳在基线生成阶段诊断（RuntimeError，含处置建议）
tiny = WindFarmConfig(n_turbines=50, turbine_model="V164-9.5MW",
                      boundary_params={"width": 500, "height": 500},
                      visualization=_no_plots)
try:
    generate_grid_layout(
        tiny.create_boundary(), 50,
        np.full(50, 164.0), 5.0, rng=np.random.default_rng(0),
        spacing_matrix=tiny.spacing_matrix(),
    )
    raise AssertionError("场地不足应当报错")
except RuntimeError as exc:
    assert "场地无法容纳" in str(exc)
    print(f"   ✓ 场地无法容纳时给出可操作诊断: {str(exc).splitlines()[0]}")

print("\n" + "=" * 60)
print("所有核心测试通过! ✓")
print("=" * 60)
print("\n可以使用以下命令运行完整分析:")
print("  python -m wind_farm_opt --help")
print("  python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50")
print("  python -m wind_farm_opt --config my_config.json")
print("\n多机型配置示例见 README，支持 fleet.turbine_types 与 fleet.turbines 两种声明。")
