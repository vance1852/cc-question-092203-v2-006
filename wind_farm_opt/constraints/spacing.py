"""风机间距约束。"""

import numpy as np


def compute_pairwise_min_spacings(
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
) -> np.ndarray:
    """根据各机组转子直径计算机对最小间距矩阵。

    每一对机组 (i, j) 的最小间距为
    ``min_multiple * (D_i + D_j) / 2``，即按两台机组各自直径的
    平均值计算；同型号机组退化为 ``min_multiple * D``。

    Parameters
    ----------
    rotor_diameters : np.ndarray
        每台风机的转子直径，形状为 (N,)
    min_multiple : float
        最小间距倍数（相对于两机转子直径的平均值）

    Returns
    -------
    np.ndarray
        成对最小间距矩阵，形状为 (N, N)，对角线为 0
    """
    diameters = np.asarray(rotor_diameters, dtype=np.float64)
    mean_diameters = 0.5 * (diameters[:, np.newaxis] + diameters[np.newaxis, :])
    matrix = min_multiple * mean_diameters
    np.fill_diagonal(matrix, 0.0)
    return matrix


def check_min_spacing_pairwise(
    positions: np.ndarray,
    min_spacing_matrix: np.ndarray,
) -> tuple[bool, np.ndarray]:
    """检查所有风机对之间的间距是否满足各自的成对最小距离要求。

    Parameters
    ----------
    positions : np.ndarray
        风机位置，形状为 (N_turbines, 2)
    min_spacing_matrix : np.ndarray
        成对最小间距矩阵，形状为 (N_turbines, N_turbines)

    Returns
    -------
    tuple[bool, np.ndarray]
        - 是否所有间距都满足要求
        - 不满足要求的风机对索引数组，形状为 (M, 2)，M 为违规对数
    """
    n = positions.shape[0]
    violations = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min_spacing_matrix[i, j]:
                violations.append([i, j])

    if violations:
        return False, np.array(violations, dtype=int)
    else:
        return True, np.zeros((0, 2), dtype=int)


def check_min_spacing(
    positions: np.ndarray,
    min_distance: float,
) -> tuple[bool, np.ndarray]:
    """检查所有风机对之间的间距是否满足统一的最小距离要求。

    Parameters
    ----------
    positions : np.ndarray
        风机位置，形状为 (N_turbines, 2)
    min_distance : float
        最小允许间距 (m)

    Returns
    -------
    tuple[bool, np.ndarray]
        - 是否所有间距都满足要求
        - 不满足要求的风机对索引数组，形状为 (M, 2)，M 为违规对数
    """
    n = positions.shape[0]
    matrix = np.full((n, n), float(min_distance), dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    return check_min_spacing_pairwise(positions, matrix)


def compute_min_spacing_from_diameters(
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
) -> float:
    """根据转子直径计算全场统一的最小间距（取最大直径的倍数）。

    保留用于旧的单一机型流程；多机型场景应使用
    :func:`compute_pairwise_min_spacings` 得到按机组对区分的间距。

    Parameters
    ----------
    rotor_diameters : np.ndarray
        每台风机的转子直径
    min_multiple : float
        最小间距倍数（相对于转子直径）

    Returns
    -------
    float
        最小间距 (m)
    """
    return float(min_multiple * np.max(rotor_diameters))


def compute_pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """计算所有风机对之间的距离矩阵。

    Parameters
    ----------
    positions : np.ndarray
        风机位置，形状为 (N, 2)

    Returns
    -------
    np.ndarray
        距离矩阵，形状为 (N, N)，对角线为 0
    """
    positions = np.asarray(positions, dtype=np.float64)
    delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    return np.linalg.norm(delta, axis=-1)


def enforce_min_spacing_pairwise(
    positions: np.ndarray,
    min_spacing_matrix: np.ndarray,
    boundary,
    rng: np.random.Generator | None = None,
    max_iterations: int = 1000,
) -> np.ndarray:
    """尝试通过移动风机满足各机组对各自的最小间距约束。

    当有风机对间距不足时，将它们沿连线方向推开，推开量对应该机对
    在 ``min_spacing_matrix`` 中的要求值。

    Parameters
    ----------
    positions : np.ndarray
        初始风机位置，形状为 (N, 2)
    min_spacing_matrix : np.ndarray
        成对最小间距矩阵，形状为 (N, N)
    boundary : SiteBoundary
        场地边界
    rng : Optional[np.random.Generator]
        随机数生成器
    max_iterations : int
        最大迭代次数

    Returns
    -------
    np.ndarray
        调整后的风机位置
    """
    if rng is None:
        rng = np.random.default_rng()

    positions = np.array(positions, dtype=np.float64, copy=True)
    n = positions.shape[0]

    for _ in range(max_iterations):
        valid, violations = check_min_spacing_pairwise(positions, min_spacing_matrix)
        if valid:
            break

        for i, j in violations:
            required = float(min_spacing_matrix[i, j])
            vec = positions[j] - positions[i]
            dist = np.linalg.norm(vec)
            if dist < 1e-12:
                vec = rng.standard_normal(2)
                dist = np.linalg.norm(vec)
            vec_norm = vec / dist

            push = (required - dist) / 2.0 + 1e-6
            positions[i] -= vec_norm * push
            positions[j] += vec_norm * push

        for k in range(n):
            if not boundary.contains_point(positions[k]):
                positions[k] = boundary.project_to_boundary(positions[k])
                perturbation = rng.uniform(-5.0, 5.0, 2)
                positions[k] += perturbation
                if not boundary.contains_point(positions[k]):
                    positions[k] = boundary.project_to_boundary(positions[k])

    valid, _ = check_min_spacing_pairwise(positions, min_spacing_matrix)
    inside = boundary.contains_all(positions)
    if not (valid and inside.all()):
        raise RuntimeError("无法通过调整满足间距和边界约束")

    return positions


def enforce_min_spacing(
    positions: np.ndarray,
    min_distance: float,
    boundary,
    rng: np.random.Generator | None = None,
    max_iterations: int = 1000,
) -> np.ndarray:
    """尝试通过移动风机来满足统一的最小间距约束。

    当有风机对间距不足时，将它们沿连线方向推开。

    Parameters
    ----------
    positions : np.ndarray
        初始风机位置，形状为 (N, 2)
    min_distance : float
        最小间距 (m)
    boundary : SiteBoundary
        场地边界
    rng : Optional[np.random.Generator]
        随机数生成器
    max_iterations : int
        最大迭代次数

    Returns
    -------
    np.ndarray
        调整后的风机位置
    """
    n = positions.shape[0]
    matrix = np.full((n, n), float(min_distance), dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    return enforce_min_spacing_pairwise(
        positions, matrix, boundary, rng=rng, max_iterations=max_iterations
    )
