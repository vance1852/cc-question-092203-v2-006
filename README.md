# 风电场布局优化工具

这个项目用于估算风电场的年发电量，并比较不同风机布局和尾流模型的结果。项目包含风机与风资源模型、场地边界和间距约束、遗传算法与粒子群优化、经济性分析以及无界面图表输出。

## 安装

建议使用 Python 3.10 或更新版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 可以使用 `.venv\\Scripts\\Activate.ps1` 激活环境。

## 快速验证

```bash
python quick_test.py
```

快速验证会覆盖模型、约束、年发电量、优化、经济性和图表生成，并在 `test_output/` 写入临时图片。该目录不会纳入版本控制。

## 完整分析

```bash
python -m wind_farm_opt --help
python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50 --output-dir output
```

也可以先生成配置文件，再通过 `--config` 运行：

```bash
python -m wind_farm_opt --generate-config my_config.json
python -m wind_farm_opt --config my_config.json
```

所有运行结果默认写入 `output/`，可以用 `--no-plots` 跳过图表生成。命令行使用无界面绘图后端，适合容器和服务器环境。

## 多机型扩建（稳定编号与成对间距）

扩建场景需要在旧有小转子机组之间插入更大转子的机型。配置支持三种等价的机组声明：

**1. 旧版单一机型（保持向后兼容，数值结果与历史版本一致）**

```json
{"n_turbines": 15, "turbine_model": "V126-3.45MW"}
```

**2. 按型号声明数量（`count` + 稳定编号前缀）**

```json
{
  "turbine_fleet": [
    {"model": "V126-3.45MW", "count": 8, "id_prefix": "OLD"},
    {"model": "V164-9.5MW", "count": 4, "id_prefix": "NEW"}
  ]
}
```

同前缀机组自动编号为 `OLD-01 … OLD-08`；不给 `id_prefix` 时全场连续编号
`WTG-001 …`。编号只取决于声明顺序与数量，重复运行保持稳定。

**3. 逐机清单（每台机组一条，可携带稳定编号与预设机位）**

```json
{
  "turbine_fleet": [
    {"id": "OLD-01", "model": "V126-3.45MW"},
    {"id": "NEW-01", "model": "V164-9.5MW", "position": [1200.0, 800.0]}
  ]
}
```

非内置机型可在条目内直接声明气动与造价参数：`rotor_diameter`、
`hub_height`、`thrust_coefficient`、`rated_power_kw`、`cut_in_speed`、
`rated_wind_speed`、`cut_out_speed`（或直接给出 `power_curve`），
以及可选的 `capital_cost_per_MW` / `installation_cost_per_MW` /
`o_and_m_cost_per_MW_per_year`。

完整示例见 `examples/mixed_fleet_config.json` 与
`examples/per_turbine_manifest.json`，也可通过命令行直接传入：

```bash
python -m wind_farm_opt --fleet examples/mixed_fleet_config.json
# 或直接传 JSON 字符串
python -m wind_farm_opt --fleet '[{"model":"V126-3.45MW","count":8},{"model":"V164-9.5MW","count":4}]'
```

### 成对间距规则

每一对机组 (i, j) 的最小机位间距按各自直径计算：

```
最小间距(i,j) = min_spacing_multiple × (D_i + D_j) / 2
```

同型号机组退化为 `倍数 × D`；大小机对只按平均直径要求，不再被全场最大
直径统一拉开。该规则贯穿基线网格生成、约束检查、GA、PSO 与修复过程；
AEP 中每台机组使用自己的功率曲线与推力系数，经济分析、逐机结果
（`results.json` 的 `turbine_results`）与布局图均保留型号、稳定编号与
容量明细。

### 配置诊断

下列错误会在计算前以退出码 2 报出并给出修复建议：

- 机型数量与 `n_turbines`/位置数量不匹配；
- 机组编号重复、`count` 条目误用单机 `id`/`position`；
- 未知机型且缺少自定义参数、`count` 非法、清单为空；
- 预设机位重复、越界或不满足该机对的间距要求；
- 场地无法容纳机组组合（提示减少机组、降低间距倍数或扩大场地）。
