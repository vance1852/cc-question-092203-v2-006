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

## 多机型混装（扩建方案）

工具支持在同一场站中混装不同转子直径、不同容量的机型。每台机组都是
互相独立的实例，拥有场站范围内唯一的稳定编号；任意两台机组之间的最小
间距按各自转子直径的平均值计算：

```
s_ij = min_spacing_multiple × (D_i + D_j) / 2
```

同型号机队（`D_i = D_j = D`）时该规则退化为经典的 `k·D`，因此旧的
单一机型配置结果与历史版本完全一致；小机组之间不再被全场最大直径
强制拉开。AEP 使用每台机组自己的功率曲线与推力系数，经济性分析、
逐机结果和布局图均保留型号与容量明细。

### 方式一：按型号声明数量

```json
{
  "fleet": {
    "turbine_types": [
      {"model": "V126-3.45MW", "count": 10},
      {"model": "V164-9.5MW", "count": 4}
    ]
  }
}
```

机组按声明顺序获得稳定编号（WT-01、WT-02 …）。

### 方式二：逐机清单（可带固定坐标）

```json
{
  "fleet": {
    "turbines": [
      {"id": "OLD-01", "model": "V126-3.45MW", "position": [-1000.0, -800.0]},
      {"id": "NEW-01", "model": "V150-6.0MW",  "position": [  500.0,  600.0]}
    ],
    "catalog": {
      "V150-6.0MW": {
        "rotor_diameter": 150.0,
        "hub_height": 100.0,
        "thrust_coefficient": 0.83,
        "rated_power_kw": 6000.0,
        "cut_in_speed": 3.5,
        "rated_speed": 11.5,
        "cut_out_speed": 25.0
      }
    }
  }
}
```

- `position` 全部省略时，机组由基线网格和 GA/PSO 自动布置；
- 全部给出时，直接评估该固定布局（跳过位置优化），并校验场地与
  逐对间距；
- 只给一部分会报错并列出缺少坐标的机组。

`catalog` 中也可声明自定义机型的造价系数
（`capital_cost_per_MW`、`installation_cost_per_MW`、
`o_and_m_cost_per_MW_per_year`）。内置型号为 `V126-3.45MW` 与
`V164-9.5MW`，自定义机型还可以直接给出 `power_curve`
（形如 `[[风速, kW], ...]`）替代三次方曲线。

### 诊断

以下情况会在加载或基线阶段给出带修复建议的错误：未知机型、数量不是
正整数、机组编号重复、只给了部分坐标、坐标不满足场地/间距约束、
两种 fleet 声明混用、以及场地无法容纳所申报的机组数量。

