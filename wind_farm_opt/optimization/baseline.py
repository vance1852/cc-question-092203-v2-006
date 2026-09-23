"""基线布局生成（规则网格）。

均匀直径机队沿用历史网格算法（数值与旧版一致）；混合直径机队使用
按机对各自直径计算的成对间距，且大直径机组优先入网格、小直径机组
补入空隙，避免全场按最大直径统一拉大间距。
"""

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..constraints.spacing import (
    check_min_spacing,
    check_min_spacing_pairwise,
    compute_min_spacing_from_diameters,
    compute_pairwise_min_spacings,
    enforce_min_spacing,
    enforce_min_spacing_pairwise,
)


class SiteCapacityError(RuntimeError):
    """场地无法容纳给定机组组合。

    错误消息中包含机组数量、直径范围、成对间距要求与场地尺寸，
    便于用户决定减少机组、缩小间距倍数或扩大场地。
    """


def _all_diameters_equal(rotor_diameters: np.ndarray) -> bool:
    diameters = np.asarray(rotor_diameters, dtype=np.float64)
    if diameters.size <= 1:
        return True
    return bool(np.allclose(diameters, diameters[0], rtol=0.0, atol=1e-9))


# ----------------------------------------------------------------------
# 旧版均匀直径网格（保持数值与随机序列兼容）
# ----------------------------------------------------------------------

def _legacy_grid(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float,
    aspect_ratio: float,
    staggered: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    min_spacing = compute_min_spacing_from_diameters(rotor_diameters, min_multiple)

    if staggered:
        n_rows = max(1, int(np.sqrt(n_turbines)))
    else:
        n_rows = max(1, int(np.round(np.sqrt(n_turbines / aspect_ratio))))
    n_cols = max(1, int(np.ceil(n_turbines / n_rows)))

    x_min, x_max = boundary.x_min, boundary.x_max
    y_min, y_max = boundary.y_min, boundary.y_max

    margin = min_spacing * 0.5
    x_range = x_max - x_min - 2 * margin
    y_range = y_max - y_min - 2 * margin

    if staggered:
        spacing_x = max(x_range / max(n_cols - 1, 1), min_spacing * 1.2)
        spacing_y = max(y_range / max(n_rows - 1, 1), min_spacing * 1.2)
    else:
        spacing_x = min(x_range / max(n_cols - 1, 1), min_spacing * 1.5)
        spacing_y = min(y_range / max(n_rows - 1, 1), min_spacing * 1.5)

    positions = []

    start_x = x_min + margin + (x_range - spacing_x * (n_cols - 1)) / 2.0
    start_y = y_min + margin + (y_range - spacing_y * (n_rows - 1)) / 2.0

    count = 0
    for row in range(n_rows):
        offset = spacing_x / 2.0 if (staggered and row % 2 == 1) else 0.0
        for col in range(n_cols):
            if count >= n_turbines:
                break
            x = start_x + col * spacing_x + offset
            y = start_y + row * spacing_y
            pos = np.array([x, y])
            if boundary.contains_point(pos):
                positions.append(pos)
                count += 1

    while len(positions) < n_turbines:
        candidates = boundary.sample_random_points(n_turbines * 2, rng)
        for cand in candidates:
            if len(positions) >= n_turbines:
                break
            pos_arr = np.array(positions + [cand]) if positions else np.array([cand])
            valid, _ = check_min_spacing(pos_arr, min_spacing)
            if valid and boundary.contains_point(cand):
                positions.append(cand)

    positions = np.array(positions, dtype=np.float64)

    valid, _ = check_min_spacing(positions, min_spacing)
    inside = boundary.contains_all(positions).all()

    if not (valid and inside):
        try:
            positions = enforce_min_spacing(positions, min_spacing, boundary, rng)
        except RuntimeError:
            diameters = np.asarray(rotor_diameters, dtype=np.float64)
            raise SiteCapacityError(
                _capacity_message(boundary, diameters, min_multiple)
            )

    valid, _ = check_min_spacing(positions, min_spacing)
    if not valid:
        diameters = np.asarray(rotor_diameters, dtype=np.float64)
        raise SiteCapacityError(
            _capacity_message(boundary, diameters, min_multiple)
        )

    return positions


# ----------------------------------------------------------------------
# 混合直径成对间距网格
# ----------------------------------------------------------------------

def _mixed_grid(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float,
    aspect_ratio: float,
    staggered: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    diameters = np.asarray(rotor_diameters, dtype=np.float64)
    min_spacing_matrix = compute_pairwise_min_spacings(diameters, min_multiple)

    n_rows = max(1, int(np.round(np.sqrt(n_turbines / aspect_ratio))))
    n_cols = max(1, int(np.ceil(n_turbines / n_rows)))

    x_min, x_max = boundary.x_min, boundary.x_max
    y_min, y_max = boundary.y_min, boundary.y_max

    margin = 0.5 * float(np.max(min_spacing_matrix))
    x_range = x_max - x_min - 2 * margin
    y_range = y_max - y_min - 2 * margin

    # 大直径机组优先占据网格槽位，小直径机组补入剩余空间。
    order = np.argsort(-diameters, kind="stable")

    slot_index: dict[tuple[int, int], int] = {}
    seq = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if seq >= n_turbines:
                break
            slot_index[(row, col)] = int(order[seq])
            seq += 1

    def gap_between(slots: list[tuple[int, int]]) -> float:
        reqs = [
            min_spacing_matrix[slot_index[a], slot_index[b]]
            for a, b in slots
            if a in slot_index and b in slot_index
        ]
        return max(float(np.max(reqs)), 1.0) if reqs else float(np.max(min_spacing_matrix))

    col_gaps = np.array([
        gap_between([((r, c), (r, c + 1)) for r in range(n_rows)])
        for c in range(max(n_cols - 1, 1))
    ])
    row_gaps = np.array([
        gap_between([((r, c), (r + 1, c)) for c in range(n_cols)])
        for r in range(max(n_rows - 1, 1))
    ])

    total_width = float(np.sum(col_gaps[:n_cols - 1])) if n_cols > 1 else 0.0
    total_height = float(np.sum(row_gaps[:n_rows - 1])) if n_rows > 1 else 0.0

    def build_coordinates(cg: np.ndarray, rg: np.ndarray):
        """由间隙构造各槽位坐标（含交错偏移），返回 slot->坐标。"""
        xs = np.zeros(n_cols)
        for c in range(1, n_cols):
            xs[c] = xs[c - 1] + cg[c - 1]
        ys = np.zeros(n_rows)
        for r in range(1, n_rows):
            ys[r] = ys[r - 1] + rg[r - 1]
        coords = {}
        for (r, c), idx in slot_index.items():
            offset = cg[max(r - 1, 0)] / 2.0 if (staggered and r % 2 == 1) else 0.0
            coords[(r, c)] = np.array([xs[c] + offset, ys[r]])
        return coords, xs, ys

    def grid_pairwise_feasible(coords: dict) -> tuple[bool, float]:
        """检查网格内全部已占槽位（含对角机对）的成对间距。"""
        slots = list(coords)
        worst = 1.0
        feasible = True
        for a_i in range(len(slots)):
            for b_i in range(a_i + 1, len(slots)):
                a, b = slots[a_i], slots[b_i]
                dist = float(np.linalg.norm(coords[a] - coords[b]))
                req = float(min_spacing_matrix[slot_index[a], slot_index[b]])
                if dist < req:
                    feasible = False
                    worst = max(worst, req / max(dist, 1e-9))
        return feasible, worst

    # 相邻间隙只保证正上下/左右；逐轮整体放大间隙直到对角机对也满足。
    fit_iterations = 0
    coords: dict = {}
    total_width = 0.0
    total_height = 0.0
    while fit_iterations < 50:
        candidate, xs, ys = build_coordinates(col_gaps, row_gaps)
        feasible, worst = grid_pairwise_feasible(candidate)
        cand_width = (
            float(max(p[0] for p in candidate.values())
                  - min(p[0] for p in candidate.values()))
            if candidate else 0.0
        )
        cand_height = float(ys[-1] - ys[0]) if n_rows > 1 else 0.0
        if feasible:
            coords = candidate
            total_width = cand_width
            total_height = cand_height
            break
        if cand_width * worst > x_range or cand_height * worst > y_range:
            # 放大后必然超出可用范围，放弃网格，全部走拒绝采样。
            break
        col_gaps = col_gaps * (worst * 1.001)
        row_gaps = row_gaps * (worst * 1.001)
        fit_iterations += 1

    positions_in_order: list[np.ndarray] = []
    placed_global: list[int] = []
    placed_seqs: set[int] = set()

    if coords and total_width <= x_range and total_height <= y_range:
        start_x = x_min + margin + (x_range - total_width) / 2.0
        start_y = y_min + margin + (y_range - total_height) / 2.0
        origin = np.array([start_x, start_y])

        seq = 0
        for row in range(n_rows):
            for col in range(n_cols):
                if seq >= n_turbines:
                    break
                if (row, col) not in coords:
                    seq += 1
                    continue
                pos = coords[(row, col)] + origin
                if boundary.contains_point(pos):
                    positions_in_order.append(pos)
                    placed_global.append(int(order[seq]))
                    placed_seqs.add(seq)
                seq += 1

    # 网格未覆盖（含落在边界外）的机组用拒绝采样补点。
    pending = [int(order[s]) for s in range(n_turbines) if s not in placed_seqs]
    message = _capacity_message(boundary, diameters, min_multiple)
    for idx in pending:
        placed = False
        try:
            for _ in range(2000):
                cand = boundary.sample_random_points(1, rng, max_attempts=100)[0]
                if all(
                    np.linalg.norm(cand - pos) >= min_spacing_matrix[idx, j]
                    for pos, j in zip(positions_in_order, placed_global)
                ):
                    positions_in_order.append(cand)
                    placed_global.append(idx)
                    placed = True
                    break
        except RuntimeError as exc:
            # 采样器在狭小/凹多边形内连续失败时，统一转成容量诊断。
            raise SiteCapacityError(
                f"在场地内为机组 #{idx} 采样机位失败：{exc}。{message}"
            ) from exc
        if not placed:
            raise SiteCapacityError(
                f"机组 #{idx} 在拒绝采样 2000 次后仍找不到满足成对间距的机位。"
                + message
            )

    positions = np.zeros((n_turbines, 2), dtype=np.float64)
    for pos, global_idx in zip(positions_in_order, placed_global):
        positions[global_idx] = pos

    valid, _ = check_min_spacing_pairwise(positions, min_spacing_matrix)
    inside = boundary.contains_all(positions)
    if not (valid and inside.all()):
        try:
            positions = enforce_min_spacing_pairwise(
                positions, min_spacing_matrix, boundary, rng
            )
        except RuntimeError as exc:
            raise SiteCapacityError(
                _capacity_message(boundary, diameters, min_multiple)
            ) from exc

    valid, _ = check_min_spacing_pairwise(positions, min_spacing_matrix)
    if not valid:
        raise SiteCapacityError(
            _capacity_message(boundary, diameters, min_multiple)
        )

    return positions


def _capacity_message(
    boundary: SiteBoundary,
    diameters: np.ndarray,
    min_multiple: float,
) -> str:
    matrix = compute_pairwise_min_spacings(diameters, min_multiple)
    positive = matrix[matrix > 0]
    if positive.size > 0:
        spacing_range = f"{positive.min():.0f}~{positive.max():.0f} m"
    else:
        spacing_range = "无机对间距要求"
    return (
        f"场地无法容纳 {len(diameters)} 台机组"
        f"（直径 {diameters.min():.0f}~{diameters.max():.0f} m，"
        f"间距倍数 {min_multiple:g}，成对间距要求 "
        f"{spacing_range}，"
        f"场地 {boundary.x_max - boundary.x_min:.0f} m × "
        f"{boundary.y_max - boundary.y_min:.0f} m、面积 "
        f"{boundary.area / 1e6:.2f} km²）。请尝试：减少机组数量、"
        "降低 min_spacing_multiple、扩大场地，或改用更大的场地边界。"
    )


def generate_grid_layout(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    aspect_ratio: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """生成规则网格布局作为优化基线。

    每一对机组的间距按各自转子直径的平均值乘以 ``min_multiple`` 计算；
    同型号机队等价于 ``min_multiple × D`` 并沿用历史网格算法。

    Parameters
    ----------
    boundary : SiteBoundary
        场地边界
    n_turbines : int
        风机台数
    rotor_diameters : np.ndarray
        每台风机的转子直径（按机组稳定编号顺序）
    min_multiple : float
        最小间距倍数（相对于机对平均直径）
    aspect_ratio : float
        网格纵横比 (列数/行数)
    rng : Optional[np.random.Generator]
        随机数生成器

    Returns
    -------
    np.ndarray
        网格布局位置 (n_turbines, 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    if _all_diameters_equal(rotor_diameters):
        return _legacy_grid(
            boundary, n_turbines, rotor_diameters,
            min_multiple, aspect_ratio, staggered=False, rng=rng,
        )

    return _mixed_grid(
        boundary, n_turbines, rotor_diameters,
        min_multiple, aspect_ratio, staggered=False, rng=rng,
    )


def generate_staggered_grid_layout(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    dominant_direction: float = 270.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """生成交错网格布局（错位排列，减少主风向下的尾流）。

    间距规则与 :func:`generate_grid_layout` 相同，按机对各自直径计算。

    Parameters
    ----------
    boundary : SiteBoundary
        场地边界
    n_turbines : int
        风机台数
    rotor_diameters : np.ndarray
        每台风机的转子直径（按机组稳定编号顺序）
    min_multiple : float
        最小间距倍数
    dominant_direction : float
        主风向（度），用于确定交错方向
    rng : Optional[np.random.Generator]
        随机数生成器

    Returns
    -------
    np.ndarray
        交错网格布局位置 (n_turbines, 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    if _all_diameters_equal(rotor_diameters):
        return _legacy_grid(
            boundary, n_turbines, rotor_diameters,
            min_multiple, aspect_ratio=1.0, staggered=True, rng=rng,
        )

    return _mixed_grid(
        boundary, n_turbines, rotor_diameters,
        min_multiple, aspect_ratio=1.0, staggered=True, rng=rng,
    )
