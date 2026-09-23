"""多机型机队功能验证脚本（手动运行，不纳入版本控制）。"""
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from wind_farm_opt.config import WindFarmConfig, ConfigError, build_fleet_turbines, count_fleet_units
from wind_farm_opt.constraints.boundary import create_rectangular_boundary
from wind_farm_opt.constraints.spacing import (
    compute_pairwise_min_spacings,
    check_min_spacing_pairwise,
)
from wind_farm_opt.farm.aep import AEPCalculator
from wind_farm_opt.core.wake import JensenWake
from wind_farm_opt.core.wind_resource import create_default_wind_resource
from wind_farm_opt.optimization.baseline import generate_grid_layout, SiteCapacityError
from wind_farm_opt.optimization.ga import GeneticAlgorithm, GAConfig
from wind_farm_opt.optimization.pso import ParticleSwarmOptimizer, PSOConfig
from wind_farm_opt.economy.costs import EconomicAnalyzer, get_default_farm_cost

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


print("1. 稳定编号与独立实例")
fleet = [
    {"model": "V126-3.45MW", "count": 3, "id_prefix": "OLD"},
    {"model": "V164-9.5MW", "count": 2, "id_prefix": "NEW"},
]
turbines = build_fleet_turbines(fleet)
check("机组数量", len(turbines) == 5)
check("编号按前缀分配", [t.turbine_id for t in turbines] == [
    "OLD-01", "OLD-02", "OLD-03", "NEW-01", "NEW-02"])
check("型号顺序", [t.name for t in turbines] == [
    "V126-3.45MW"] * 3 + ["V164-9.5MW"] * 2)
ids_obj = {id(t) for t in turbines}
check("实例互相独立", len(ids_obj) == 5)
check("功率曲线数组独立", len({id(t.power_curve) for t in turbines}) == 5)

print("\n2. 默认编号（无前缀）与逐机清单")
fleet2 = [
    {"id": "A1", "model": "V126-3.45MW"},
    {"id": "B1", "model": "V164-9.5MW"},
]
t2 = build_fleet_turbines(fleet2)
check("显式 id", [t.turbine_id for t in t2] == ["A1", "B1"])
fleet3 = [{"model": "V126-3.45MW", "count": 2}, {"model": "V164-9.5MW", "count": 1}]
t3 = build_fleet_turbines(fleet3)
check("默认编号 WTG-001..", [x.turbine_id for x in t3] == ["WTG-001", "WTG-002", "WTG-003"])

print("\n3. 成对间距矩阵")
d = np.array([126.0, 126.0, 164.0])
m = compute_pairwise_min_spacings(d, 5.0)
check("同型号对 = 5D", abs(m[0, 1] - 630.0) < 1e-9)
check("混合对 = 5×平均D", abs(m[0, 2] - 5 * 145.0) < 1e-9, f"got {m[0,2]}")
check("混合对 < 全场最大5D=820", m[0, 2] < 820.0)
check("对角线为0", m[0, 0] == 0.0 and m[1, 1] == 0.0)
pos = np.array([[0, 0], [630, 0], [1355, 0]])
ok, viol = check_min_spacing_pairwise(pos, m)
check("恰好满足同型/混合型对", ok, f"violations={viol}")
pos2 = np.array([[0, 0], [629, 0], [1355, 0]])
ok2, viol2 = check_min_spacing_pairwise(pos2, m)
check("差1米即违规", not ok2 and len(viol2) == 1 and list(viol2[0]) == [0, 1])

print("\n4. 基线生成（混合机型）+ 机对间距贯穿")
boundary = create_rectangular_boundary(3000, 3000)
diameters = np.array([126.0] * 6 + [164.0] * 3)
rng = np.random.default_rng(7)
positions = generate_grid_layout(boundary, 9, diameters, min_multiple=5.0, rng=rng)
mm = compute_pairwise_min_spacings(diameters, 5.0)
ok, viol = check_min_spacing_pairwise(positions, mm)
check("基线满足成对间距", ok)
check("机组全部在场内", boundary.contains_all(positions).all())
check("位置按编号顺序对齐", positions.shape == (9, 2))
small_pairs_dist = []
for i in range(6):
    for j in range(6, 9):
        small_pairs_dist.append(np.linalg.norm(positions[i] - positions[j]))
check("存在间距小于旧规则820m的混合机对（节省场地证据）",
      min(small_pairs_dist) < 820.0,
      f"min mixed-pair dist={min(small_pairs_dist):.1f}")

print("\n5. 场地无法容纳诊断")
tiny = create_rectangular_boundary(500, 500)
try:
    generate_grid_layout(tiny, 9, diameters, min_multiple=5.0,
                         rng=np.random.default_rng(1))
    check("小场地抛出 SiteCapacityError", False)
except SiteCapacityError as exc:
    check("小场地抛出 SiteCapacityError", True)
    check("诊断含可操作建议", "减少机组" in str(exc) and "min_spacing_multiple" in str(exc),
          str(exc))

print("\n6. 配置诊断：数量不匹配/重复编号/未知机型/重复机位")
def expect_config_error(label, data):
    try:
        cfg = WindFarmConfig.from_json_dict if hasattr(WindFarmConfig, "from_json_dict") else None
        import io
        path = "/tmp/_bad_cfg.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        c = WindFarmConfig.from_json(path)
        c.create_turbines()
        check(label, False, "未报错")
    except ConfigError as exc:
        check(label, True)
        print(f"        消息: {str(exc)[:110]}...")

expect_config_error("数量不匹配", {
    "n_turbines": 4,
    "turbine_fleet": [{"model": "V126-3.45MW", "count": 3}],
})
expect_config_error("重复编号", {
    "turbine_fleet": [
        {"id": "X", "model": "V126-3.45MW"},
        {"id": "X", "model": "V164-9.5MW"},
    ],
})
expect_config_error("未知机型无参数", {
    "turbine_fleet": [{"model": "MY-6MW", "count": 2}],
})
expect_config_error("count + id 冲突", {
    "turbine_fleet": [{"id": "X", "model": "V126-3.45MW", "count": 2}],
})
expect_config_error("count 非法", {
    "turbine_fleet": [{"model": "V126-3.45MW", "count": 0}],
})

# 重复机位 / 越界 / 间距不足（通过 validate_manifest_positions）
c = WindFarmConfig(
    turbine_fleet=[
        {"id": "A", "model": "V126-3.45MW", "position": [0.0, 0.0]},
        {"id": "B", "model": "V126-3.45MW", "position": [0.0, 0.0]},
    ],
    n_turbines=2,
)
try:
    c.validate_manifest_positions(create_rectangular_boundary(2000, 2000))
    check("重复机位诊断", False)
except ConfigError:
    check("重复机位诊断", True)

c = WindFarmConfig(
    turbine_fleet=[
        {"id": "A", "model": "V126-3.45MW", "position": [-5000.0, 0.0]},
        {"id": "B", "model": "V126-3.45MW", "position": [0.0, 0.0]},
    ],
    n_turbines=2,
)
try:
    c.validate_manifest_positions(create_rectangular_boundary(2000, 2000))
    check("越界诊断", False)
except ConfigError:
    check("越界诊断", True)

c = WindFarmConfig(
    turbine_fleet=[
        {"id": "A", "model": "V126-3.45MW", "position": [-500.0, 0.0]},
        {"id": "B", "model": "V164-9.5MW", "position": [100.0, 0.0]},
    ],
    n_turbines=2,
)
try:
    c.validate_manifest_positions(create_rectangular_boundary(4000, 4000))
    check("清单间距不足诊断", False)
except ConfigError as exc:
    check("清单间距不足诊断", True and "A<->B" in str(exc), str(exc)[:100])

print("\n7. 自定义机型")
custom = [{"model": "EXT-6MW", "count": 2, "id_prefix": "EXT",
           "rotor_diameter": 170.0, "hub_height": 110.0, "thrust_coefficient": 0.81,
           "rated_power_kw": 6000.0, "cut_in_speed": 3.0,
           "rated_wind_speed": 11.0, "cut_out_speed": 25.0,
           "capital_cost_per_MW": 700.0}]
tc = build_fleet_turbines(custom)
check("自定义机型参数", tc[0].rotor_diameter == 170.0 and tc[0].rated_power == 6000.0)
check("自定义编号", tc[0].turbine_id == "EXT-01")

print("\n8. AEP 使用各机组自己的功率/推力/直径")
wr = create_default_wind_resource(num_sectors=12, dominant_direction=270.0, mean_speed=8.5)
mixed = build_fleet_turbines([
    {"model": "V126-3.45MW", "count": 4, "id_prefix": "S"},
    {"model": "V164-9.5MW", "count": 2, "id_prefix": "L"},
])
diam = np.array([t.rotor_diameter for t in mixed])
b = create_rectangular_boundary(3500, 3500)
pos = generate_grid_layout(b, 6, diam, 5.0, rng=np.random.default_rng(3))
calc = AEPCalculator(mixed, wr, JensenWake(0.07), speed_step=1.0)
res = calc.compute_farm_aep(pos)
check("总容量 = 4×3.45+2×9.5 = 32.8", abs(res.total_installed_capacity - 32.8) < 1e-6,
      f"{res.total_installed_capacity}")
names = {tr.name for tr in res.turbine_results}
check("逐机结果保留型号", names == {"V126-3.45MW", "V164-9.5MW"})
check("逐机结果保留编号", [tr.turbine_id for tr in res.turbine_results] ==
      ["S-01", "S-02", "S-03", "S-04", "L-01", "L-02"])
large = [tr for tr in res.turbine_results if tr.name == "V164-9.5MW"]
check("大机组单机毛发电量更高", large[0].gross_aep >
      max(tr.gross_aep for tr in res.turbine_results if tr.name == "V126-3.45MW") * 1.5)
check("model_summary 按型号汇总",
      res.model_summary["V164-9.5MW"]["count"] == 2
      and abs(res.model_summary["V164-9.5MW"]["capacity_mw"] - 19.0) < 1e-6)

print("\n9. GA / PSO 成对间距贯穿")
for algo_cls, cfg, label in [
    (GeneticAlgorithm, GAConfig(population_size=6, max_generations=3, seed=11), "GA"),
    (ParticleSwarmOptimizer, PSOConfig(swarm_size=6, max_iterations=3, seed=11), "PSO"),
]:
    opt = algo_cls(n_turbines=6, rotor_diameters=diam, boundary=b,
                   fitness_fn=calc.evaluate_layout, config=cfg)
    out = opt.optimize(verbose=False)
    ok, viol = check_min_spacing_pairwise(out.best_positions, opt.min_spacing_matrix)
    check(f"{label} 最优解满足成对间距", ok and b.contains_all(out.best_positions).all())
    # 最终种群所有个体修复后也应满足
    pop = out.final_population.reshape(-1, 6, 2)
    all_ok = all(
        check_min_spacing_pairwise(p, opt.min_spacing_matrix)[0]
        and b.contains_all(p).all() for p in pop
    )
    check(f"{label} 最终种群全部可行", all_ok)

print("\n10. 经济分析按型号分别计价")
cfg_models = {t.name: None for t in mixed}
from wind_farm_opt.economy.costs import get_default_turbine_cost
cost_models = {n: get_default_turbine_cost(n) for n in cfg_models}
analyzer = EconomicAnalyzer(cost_models["V126-3.45MW"], get_default_farm_cost())
powers = np.array([t.rated_power / 1e3 for t in mixed])
models = [t.name for t in mixed]
er = analyzer.analyze_fleet(powers, res.net_aep / 1e3, cost_models, models)
expected_capex = (4 * 3.45 * (580 + 70)) + (2 * 9.5 * (650 + 80)) + \
    32.8 * 300 + 5000 + 2000
check("混合造价正确", abs(er.total_capital_cost - expected_capex) < 1e-6,
      f"{er.total_capital_cost} vs {expected_capex}")
check("型号明细", set(er.model_breakdown) == {"V126-3.45MW", "V164-9.5MW"})
check("明细容量", abs(er.model_breakdown["V164-9.5MW"]["capacity_mw"] - 19.0) < 1e-6)

# 单一机型 analyze 与 analyze_fleet 等价
er2 = analyzer.analyze(4, 3.45, 100.0)
er3 = analyzer.analyze_fleet(np.full(4, 3.45), 100.0)
check("旧 analyze 与 fleet 等价", abs(er2.total_capital_cost - er3.total_capital_cost) < 1e-9
      and abs(er2.lcoe - er3.lcoe) < 1e-12)

print("\n11. 旧配置等价（JSON 往返 + 编号自动分配）")
sample = WindFarmConfig(n_turbines=5, turbine_model="V126-3.45MW")
ts = sample.create_turbines()
check("旧配置克隆独立", len({id(x) for x in ts}) == 5)
check("旧配置自动编号", [x.turbine_id for x in ts] == [f"WTG-{i:03d}" for i in range(1, 6)])
check("旧配置型号统一", all(x.name == "V126-3.45MW" for x in ts))

print("\n12. 修复过程 enforce_pairwise 推开量按机对要求")
from wind_farm_opt.constraints.spacing import enforce_min_spacing_pairwise
p = np.array([[0.0, 0.0], [700.0, 0.0]], dtype=np.float64)
mm2 = compute_pairwise_min_spacings(np.array([126.0, 164.0]), 5.0)  # 要求725
fixed = enforce_min_spacing_pairwise(p, mm2, create_rectangular_boundary(4000, 4000),
                                     rng=np.random.default_rng(0))
dist = np.linalg.norm(fixed[0] - fixed[1])
check("修复后满足该机对要求", dist >= 725.0 - 1e-6, f"dist={dist}")
check("修复不会过度推到820以外太多", dist < 820.0, f"dist={dist}")

print(f"\n{'='*60}")
print(f"结果: {passed} 通过, {failed} 失败")
print('='*60)
sys.exit(1 if failed else 0)
