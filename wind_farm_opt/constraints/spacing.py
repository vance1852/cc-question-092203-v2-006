"""风机间距约束。"""

import numpy as np


def pairwise_min_spacings(
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
) -> np.ndarray:
    """计算机组两两之间的最小间距矩阵。

    每一对机组的最小间距按各自转子直径的算术平均值确定：

    .. math::
        s_{ij} = k \\cdot (D_i + D_j) / 2

    同型号机组（D_i = D_j = D）时退化为经典的 ``k * D``，因此旧的
    单一机型配置会得到完全等价的结果；混装不同转子直径的机型时，
    小机组之间不再被全场最大直径强制拉开。

    Parameters
    ----------
    rotor_diameters : np.ndarray
        每台风机的转子直径，形状为 (N,)
    min_multiple : float
        最小间距倍数（相对于两台机组的平均转子直径）

    Returns
    -------
    np.ndarray
        成对最小间距矩阵，形状为 (N, N)，对角线为 0
    """
    diameters = np.asarray(rotor_diameters, dtype=np.float64)
    if diameters.ndim != 1:
        raise ValueError("rotor_diameters 必须是一维数组")
    if np.any(diameters <= 0.0):
        raise ValueError("转子直径必须为正数")
    if min_multiple <= 0.0:
        raise ValueError("最小间距倍数必须为正数")

    mean_diameter = 0.5 * (diameters[:, np.newaxis] + diameters[np.newaxis, :])
    return min_multiple * mean_diameter


def check_pairwise_spacing(
    positions: np.ndarray,
    spacing_matrix: np.ndarray,
) -> tuple[bool, np.ndarray]:
    """按逐对间距矩阵检查机组间距。

    Parameters
    ----------
    positions : np.ndarray
        风机位置，形状为 (N, 2)
    spacing_matrix : np.ndarray
        成对最小间距矩阵 (N, N)

    Returns
    -------
    tuple[bool, np.ndarray]
        - 是否所有间距都满足要求
        - 不满足要求的风机对索引数组，形状为 (M, 2)
    """
    positions = np.asarray(positions, dtype=np.float64)
    spacing_matrix = np.asarray(spacing_matrix, dtype=np.float64)
    n = positions.shape[0]
    violations = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < spacing_matrix[i, j]:
                violations.append([i, j])

    if violations:
        return False, np.array(violations, dtype=int)
    return True, np.zeros((0, 2), dtype=int)


def check_min_spacing(
    positions: np.ndarray,
    min_distance: float,
) -> tuple[bool, np.ndarray]:
    """检查所有风机对之间的间距是否满足统一最小距离要求。

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
    n = np.asarray(positions).shape[0]
    spacing_matrix = np.full((n, n), float(min_distance), dtype=np.float64)
    np.fill_diagonal(spacing_matrix, 0.0)
    return check_pairwise_spacing(positions, spacing_matrix)


def compute_min_spacing_from_diameters(
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
) -> float:
    """根据转子直径计算全场统一的最小间距（取最大直径的倍数）。

    保留用于兼容单一机型流程；混装机型应改用
    :func:`pairwise_min_spacings`。

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


def enforce_min_spacing(
    positions: np.ndarray,
    min_distance,
    boundary,
    rng: np.random.Generator | None = None,
    max_iterations: int = 1000,
) -> np.ndarray:
    """尝试通过移动风机来满足间距约束。

    当有风机对间距不足时，将它们沿连线方向按各自所需间距推开。
    ``min_distance`` 既可以是标量（全场统一间距），也可以是由
    :func:`pairwise_min_spacings` 生成的逐对间距矩阵——此时不同型号
    的机组对会按各自平均直径的倍数被推开。

    Parameters
    ----------
    positions : np.ndarray
        初始风机位置，形状为 (N, 2)
    min_distance : float | np.ndarray
        最小间距 (m)，标量或 (N, N) 间距矩阵
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

    if np.isscalar(min_distance):
        spacing_matrix = np.full((n, n), float(min_distance), dtype=np.float64)
    else:
        spacing_matrix = np.asarray(min_distance, dtype=np.float64)
    np.fill_diagonal(spacing_matrix, 0.0)

    for _ in range(max_iterations):
        valid, violations = check_pairwise_spacing(positions, spacing_matrix)
        if valid:
            break

        for i, j in violations:
            required = spacing_matrix[i, j]
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

    valid, _ = check_pairwise_spacing(positions, spacing_matrix)
    inside = boundary.contains_all(positions)
    if not (valid and inside.all()):
        raise RuntimeError("无法通过调整满足间距和边界约束")

    return positions
