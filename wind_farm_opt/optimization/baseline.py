"""基线布局生成（规则网格）。"""

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..constraints.spacing import (
    check_pairwise_spacing,
    enforce_min_spacing,
    pairwise_min_spacings,
)


def _grid_min_spacing(spacing_matrix: np.ndarray) -> float:
    """网格几何所需的标量间距：取逐对矩阵的最大值。

    单一机型时即 k*D，与历史行为一致；混装时由最大的一对
    （最大直径机组之间）决定网格尺度，但候选点合法性检查仍按
    逐对矩阵执行，小机组之间可以靠得更近。
    """
    return float(np.max(spacing_matrix))


def generate_grid_layout(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    aspect_ratio: float = 1.0,
    rng: np.random.Generator | None = None,
    spacing_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """生成规则网格布局作为优化基线。

    Parameters
    ----------
    boundary : SiteBoundary
        场地边界
    n_turbines : int
        风机台数
    rotor_diameters : np.ndarray
        每台风机的转子直径
    min_multiple : float
        最小间距倍数
    aspect_ratio : float
        网格纵横比 (列数/行数)
    rng : Optional[np.random.Generator]
        随机数生成器
    spacing_matrix : Optional[np.ndarray]
        预计算的逐对最小间距矩阵 (N, N)；省略时按
        ``min_multiple * (D_i + D_j) / 2`` 现算。每一对机组
        都按各自直径的间距规则布置。

    Returns
    -------
    np.ndarray
        网格布局位置 (n_turbines, 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    if spacing_matrix is None:
        spacing_matrix = pairwise_min_spacings(rotor_diameters, min_multiple)
    min_spacing = _grid_min_spacing(spacing_matrix)

    n_rows = max(1, int(np.round(np.sqrt(n_turbines / aspect_ratio))))
    n_cols = max(1, int(np.ceil(n_turbines / n_rows)))

    x_min, x_max = boundary.x_min, boundary.x_max
    y_min, y_max = boundary.y_min, boundary.y_max

    margin = min_spacing * 0.5
    x_range = x_max - x_min - 2 * margin
    y_range = y_max - y_min - 2 * margin

    spacing_x = min(x_range / max(n_cols - 1, 1), min_spacing * 1.5)
    spacing_y = min(y_range / max(n_rows - 1, 1), min_spacing * 1.5)

    positions = []

    start_x = x_min + margin + (x_range - spacing_x * (n_cols - 1)) / 2.0
    start_y = y_min + margin + (y_range - spacing_y * (n_rows - 1)) / 2.0

    count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if count >= n_turbines:
                break
            x = start_x + col * spacing_x
            y = start_y + row * spacing_y
            pos = np.array([x, y])
            if boundary.contains_point(pos):
                positions.append(pos)
                count += 1

    positions = _fill_remaining_positions(
        positions, n_turbines, boundary, spacing_matrix, rng
    )

    positions = np.array(positions, dtype=np.float64)

    feasible, problems = verify_layout_feasible(positions, boundary, spacing_matrix)
    if not feasible:
        try:
            positions = enforce_min_spacing(positions, spacing_matrix, boundary, rng)
            feasible, problems = verify_layout_feasible(positions, boundary, spacing_matrix)
        except RuntimeError:
            feasible = False

    if not feasible:
        raise _site_capacity_error(n_turbines, spacing_matrix, problems)

    return positions


def _site_capacity_error(
    n_turbines: int,
    spacing_matrix: np.ndarray,
    problems: list[str],
) -> RuntimeError:
    """构造“场地无法容纳机组”的可操作诊断。"""
    off_diag = spacing_matrix[~np.eye(n_turbines, dtype=bool)]
    detail = "\n  ".join(problems[:5])
    more = "" if len(problems) <= 5 else f"\n  ...另有 {len(problems) - 5} 处"
    return RuntimeError(
        f"场地无法容纳 {n_turbines} 台机组：按逐对间距规则（要求 "
        f"{off_diag.min():.0f} ~ {off_diag.max():.0f} m）布置后仍有约束冲突。\n  "
        f"{detail}{more}\n"
        f"建议：减少机组数量、降低 min_spacing_multiple、扩大场地，"
        f"或改用更多小直径机型。"
    )


def _fill_remaining_positions(
    positions: list[np.ndarray],
    n_turbines: int,
    boundary: SiteBoundary,
    spacing_matrix: np.ndarray,
    rng: np.random.Generator,
    max_batches: int = 200,
) -> list[np.ndarray]:
    """对网格未能容纳的机组，按逐对间距规则在场地内随机补点。"""
    batches = 0
    while len(positions) < n_turbines:
        batches += 1
        if batches > max_batches:
            placed = len(positions)
            min_req = float(np.max(spacing_matrix))
            raise RuntimeError(
                f"场地无法容纳 {n_turbines} 台机组：仅布置出 {placed} 台后 "
                f"就找不到满足最小间距（最大 {min_req:.0f} m）的位置。"
                f"请缩小场地边界内的机组数量、减小 "
                f"min_spacing_multiple（当前按平均直径倍数），或扩大场地。"
            )
        candidates = boundary.sample_random_points(n_turbines * 2, rng)
        for cand in candidates:
            if len(positions) >= n_turbines:
                break
            trial = positions + [cand]
            pos_arr = np.array(trial) if positions else np.array([cand])[None, :]
            valid, _ = check_pairwise_spacing(pos_arr, spacing_matrix[:len(trial), :len(trial)])
            if valid and boundary.contains_point(cand):
                positions.append(cand)
    return positions


def generate_staggered_grid_layout(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    dominant_direction: float = 270.0,
    rng: np.random.Generator | None = None,
    spacing_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """生成交错网格布局（错位排列，减少主风向下的尾流）。

    Parameters
    ----------
    boundary : SiteBoundary
        场地边界
    n_turbines : int
        风机台数
    rotor_diameters : np.ndarray
        每台风机的转子直径
    min_multiple : float
        最小间距倍数
    dominant_direction : float
        主风向（度），用于确定交错方向
    rng : Optional[np.random.Generator]
        随机数生成器
    spacing_matrix : Optional[np.ndarray]
        预计算的逐对最小间距矩阵 (N, N)

    Returns
    -------
    np.ndarray
        交错网格布局位置 (n_turbines, 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    if spacing_matrix is None:
        spacing_matrix = pairwise_min_spacings(rotor_diameters, min_multiple)
    min_spacing = _grid_min_spacing(spacing_matrix)

    n_rows = max(1, int(np.sqrt(n_turbines)))
    n_cols = max(1, int(np.ceil(n_turbines / n_rows)))

    x_min, x_max = boundary.x_min, boundary.x_max
    y_min, y_max = boundary.y_min, boundary.y_max

    margin = min_spacing * 0.5
    x_range = x_max - x_min - 2 * margin
    y_range = y_max - y_min - 2 * margin

    spacing_x = max(x_range / max(n_cols - 1, 1), min_spacing * 1.2)
    spacing_y = max(y_range / max(n_rows - 1, 1), min_spacing * 1.2)

    positions = []

    start_x = x_min + margin + (x_range - spacing_x * (n_cols - 1)) / 2.0
    start_y = y_min + margin + (y_range - spacing_y * (n_rows - 1)) / 2.0

    count = 0
    for row in range(n_rows):
        offset = spacing_x / 2.0 if row % 2 == 1 else 0.0
        for col in range(n_cols):
            if count >= n_turbines:
                break
            x = start_x + col * spacing_x + offset
            y = start_y + row * spacing_y
            pos = np.array([x, y])
            if boundary.contains_point(pos):
                positions.append(pos)
                count += 1

    positions = _fill_remaining_positions(
        positions, n_turbines, boundary, spacing_matrix, rng
    )

    positions = np.array(positions, dtype=np.float64)

    feasible, problems = verify_layout_feasible(positions, boundary, spacing_matrix)
    if not feasible:
        try:
            positions = enforce_min_spacing(
                positions, spacing_matrix, boundary, rng
            )
            feasible, problems = verify_layout_feasible(positions, boundary, spacing_matrix)
        except RuntimeError:
            feasible = False

    if not feasible:
        raise _site_capacity_error(n_turbines, spacing_matrix, problems)

    return positions


def verify_layout_feasible(
    positions: np.ndarray,
    boundary: SiteBoundary,
    spacing_matrix: np.ndarray,
) -> tuple[bool, list[str]]:
    """检查最终布局是否满足全部硬约束，返回问题清单。"""
    problems = []
    inside = boundary.contains_all(positions)
    for i, ok in enumerate(inside):
        if not ok:
            problems.append(f"机组 #{i} 位于场地边界之外")
    valid, violations = check_pairwise_spacing(positions, spacing_matrix)
    if not valid:
        for i, j in violations:
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            problems.append(
                f"机组 #{i} 与 #{j} 间距 {dist:.1f} m "
                f"小于要求的 {spacing_matrix[i, j]:.1f} m"
            )
    return len(problems) == 0, problems
