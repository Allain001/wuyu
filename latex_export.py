"""
MatrixVis - 核心矩阵运算模块
包含：LU分解、高斯-约当消元、QR迭代、线性方程组求解等
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

def compute_determinant_lu(matrix: np.ndarray) -> Dict:
    """
    使用LU分解计算行列式（带部分主元）
    
    Args:
        matrix: 输入方阵
        
    Returns:
        dict: 包含行列式值、分解步骤、中间状态
    """
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("行列式计算需要方阵")
    
    A = matrix.astype(float).copy()
    L = np.eye(n)
    P = np.eye(n)  # 置换矩阵
    steps = []
    intermediate_states = [A.copy()]
    
    det_sign = 1  # 记录行交换次数的符号
    
    for k in range(n-1):
        # 部分主元选择
        max_idx = np.argmax(np.abs(A[k:, k])) + k
        
        if max_idx != k:
            A[[k, max_idx]] = A[[max_idx, k]]
            P[[k, max_idx]] = P[[max_idx, k]]
            if k > 0:
                L[[k, max_idx], :k] = L[[max_idx, k], :k]
            det_sign *= -1
            
            steps.append({
                'step': k,
                'type': 'pivot',
                'description': f'第{k+1}步：部分主元选择，交换第{k+1}行与第{max_idx+1}行',
                'matrix': A.copy(),
                'formula': f'P_{{{k},{max_idx}}} A'
            })
            intermediate_states.append(A.copy())
        
        # 检查主元是否为零
        if abs(A[k, k]) < 1e-10:
            return {
                'value': 0,
                'steps': steps,
                'intermediate_states': intermediate_states,
                'L': L,
                'U': A,
                'P': P
            }
        
        # 计算乘子
        for i in range(k+1, n):
            L[i, k] = A[i, k] / A[k, k]
            A[i, k:] -= L[i, k] * A[k, k:]
        
        steps.append({
            'step': k,
            'type': 'elimination',
            'description': f'第{k+1}步：消去第{k+1}列下方元素',
            'matrix': A.copy(),
            'formula': f'L_{{{k+1}}} = I - l_{{{k+1}}} e_{{{k+1}}}^T'
        })
        intermediate_states.append(A.copy())
    
    # 计算行列式
    det = det_sign * np.prod(np.diag(A))
    
    return {
        'value': det,
        'steps': steps,
        'intermediate_states': intermediate_states,
        'L': L,
        'U': A,
        'P': P
    }

def compute_inverse_gauss_jordan(matrix: np.ndarray) -> Dict:
    """
    使用高斯-约当消元法计算逆矩阵
    
    Args:
        matrix: 输入方阵
        
    Returns:
        dict: 包含逆矩阵、消元步骤
    """
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("逆矩阵计算需要方阵")
    
    # 构造增广矩阵 [A|I]
    A = matrix.astype(float).copy()
    I = np.eye(n)
    augmented = np.hstack([A, I])
    
    steps = [{
        'step': 0,
        'description': '初始增广矩阵 [A|I]',
        'matrix': augmented.copy()
    }]
    
    for i in range(n):
        # 主元选择
        pivot = augmented[i, i]
        if abs(pivot) < 1e-10:
            raise ValueError("矩阵是奇异的，不存在逆矩阵")
        
        # 归一化当前行
        augmented[i] /= pivot
        
        steps.append({
            'step': i+1,
            'description': f'第{i+1}行归一化：除以主元 {pivot:.4f}',
            'matrix': augmented.copy()
        })
        
        # 消去其他行
        for j in range(n):
            if j != i:
                factor = augmented[j, i]
                augmented[j] -= factor * augmented[i]
                
                steps.append({
                    'step': i+1,
                    'description': f'消去第{j+1}行的第{i+1}列元素',
                    'matrix': augmented.copy()
                })
    
    # 提取逆矩阵
    inverse = augmented[:, n:]
    
    return {
        'matrix': inverse,
        'steps': steps,
        'augmented_final': augmented
    }

def compute_eigenvalue_qr(matrix: np.ndarray, max_iter: int = 1000, tol: float = 1e-10) -> Dict:
    """
    使用QR迭代算法计算特征值
    
    Args:
        matrix: 输入方阵
        max_iter: 最大迭代次数
        tol: 收敛阈值
        
    Returns:
        dict: 包含特征值、特征向量、收敛历史
    """
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("特征值计算需要方阵")
    
    A = matrix.astype(float).copy()
    
    # 首先化为Hessenberg形式（简化QR迭代）
    # 这里使用简化的实现
    
    convergence = []
    eigenvectors = np.eye(n)
    
    for iter_count in range(max_iter):
        # QR分解
        Q, R = np.linalg.qr(A)
        
        # 更新矩阵
        A_new = R @ Q
        
        # 计算误差
        off_diagonal = np.sqrt(np.sum(A_new**2) - np.sum(np.diag(A_new)**2))
        convergence.append(off_diagonal)
        
        # 累积特征向量
        eigenvectors = eigenvectors @ Q
        
        # 检查收敛
        if off_diagonal < tol:
            break
        
        A = A_new
    
    # 提取特征值（对角线元素）
    eigenvalues = np.diag(A)
    
    # 排序
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return {
        'values': eigenvalues,
        'vectors': eigenvectors,
        'iterations': iter_count + 1,
        'convergence': convergence,
        'final_matrix': A
    }

def solve_linear_system(A: np.ndarray, b: np.ndarray) -> Dict:
    """
    求解线性方程组 Ax = b
    
    Args:
        A: 系数矩阵
        b: 右侧向量
        
    Returns:
        dict: 包含解、消元步骤
    """
    m, n = A.shape
    
    # 构造增广矩阵
    augmented = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    
    steps = [{
        'step': 0,
        'description': '初始增广矩阵 [A|b]',
        'matrix': augmented.copy()
    }]
    
    rank_A = 0
    rank_aug = 0
    
    # 高斯消元
    for i in range(min(m, n)):
        # 找主元
        max_row = i
        for j in range(i+1, m):
            if abs(augmented[j, i]) > abs(augmented[max_row, i]):
                max_row = j
        
        if abs(augmented[max_row, i]) < 1e-10:
            continue
        
        # 交换行
        if max_row != i:
            augmented[[i, max_row]] = augmented[[max_row, i]]
        
        # 归一化
        pivot = augmented[i, i]
        augmented[i] /= pivot
        
        # 消去
        for j in range(m):
            if j != i and abs(augmented[j, i]) > 1e-10:
                factor = augmented[j, i]
                augmented[j] -= factor * augmented[i]
        
        rank_A += 1
        steps.append({
            'step': i+1,
            'description': f'第{i+1}步消元',
            'matrix': augmented.copy()
        })
    
    # 检查解的情况
    rank_aug = np.linalg.matrix_rank(augmented)
    
    if rank_A < rank_aug:
        solution_type = '无解'
        x = None
    elif rank_A < n:
        solution_type = '无穷多解'
        x = augmented[:, :-1]  # 返回行最简型
    else:
        solution_type = '唯一解'
        x = augmented[:, -1]
    
    return {
        'x': x,
        'type': solution_type,
        'rank_A': rank_A,
        'rank_aug': rank_aug,
        'steps': steps,
        'rref': augmented
    }

def compute_rank(matrix: np.ndarray) -> Dict:
    """
    计算矩阵的秩
    
    Args:
        matrix: 输入矩阵
        
    Returns:
        dict: 包含秩、行最简型
    """
    A = matrix.astype(float).copy()
    m, n = A.shape
    
    rank = 0
    row = 0
    
    for col in range(n):
        if row >= m:
            break
        
        # 找主元
        max_row = row
        for i in range(row+1, m):
            if abs(A[i, col]) > abs(A[max_row, col]):
                max_row = i
        
        if abs(A[max_row, col]) < 1e-10:
            continue
        
        # 交换
        A[[row, max_row]] = A[[max_row, row]]
        
        # 归一化
        A[row] /= A[row, col]
        
        # 消去
        for i in range(m):
            if i != row:
                A[i] -= A[i, col] * A[row]
        
        rank += 1
        row += 1
    
    return {
        'rank': rank,
        'rref': A,
        'nullity': n - rank
    }

def compute_all(matrix: np.ndarray) -> Dict:
    """
    批量计算所有运算
    
    Args:
        matrix: 输入矩阵
        
    Returns:
        dict: 包含所有计算结果
    """
    results = {
        'matrix': matrix,
        'determinant': None,
        'inverse': None,
        'eigenvalues': None,
        'rank': None
    }
    
    # 行列式
    try:
        results['determinant'] = compute_determinant_lu(matrix)
    except Exception as e:
        results['determinant_error'] = str(e)
    
    # 逆矩阵
    try:
        results['inverse'] = compute_inverse_gauss_jordan(matrix)
    except Exception as e:
        results['inverse_error'] = str(e)
    
    # 特征值
    try:
        results['eigenvalues'] = compute_eigenvalue_qr(matrix)
    except Exception as e:
        results['eigenvalue_error'] = str(e)
    
    # 秩
    try:
        results['rank'] = compute_rank(matrix)
    except Exception as e:
        results['rank_error'] = str(e)
    
    return results
