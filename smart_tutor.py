"""
MatrixVis - 知识图谱模块
基于NetworkX构建线性代数知识图谱
"""

import streamlit as st
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
import random

# 预定义的知识点数据
CONCEPTS = {
    'matrix_basic': {
        'name': '矩阵基础',
        'level': '入门',
        'time': 2,
        'description': '矩阵的定义、表示、基本运算（加法、数乘）',
        'depends': [],
        'color': '#4CAF50',
        'icon': '📊'
    },
    'matrix_mult': {
        'name': '矩阵乘法',
        'level': '入门',
        'time': 3,
        'description': '矩阵乘法的定义、性质、分块乘法',
        'depends': ['matrix_basic'],
        'color': '#4CAF50',
        'icon': '✖️'
    },
    'determinant': {
        'name': '行列式',
        'level': '基础',
        'time': 4,
        'description': '行列式的定义、性质、计算方法（展开、化三角）',
        'depends': ['matrix_basic'],
        'color': '#2196F3',
        'icon': '🔢'
    },
    'inverse_matrix': {
        'name': '逆矩阵',
        'level': '基础',
        'time': 3,
        'description': '逆矩阵的定义、性质、求法（公式法、初等变换）',
        'depends': ['determinant'],
        'color': '#2196F3',
        'icon': '🔄'
    },
    'matrix_rank': {
        'name': '矩阵的秩',
        'level': '基础',
        'time': 3,
        'description': '秩的定义、性质、计算方法',
        'depends': ['matrix_basic'],
        'color': '#2196F3',
        'icon': '📈'
    },
    'linear_equations': {
        'name': '线性方程组',
        'level': '核心',
        'time': 5,
        'description': '高斯消元法、解的判定、解的结构',
        'depends': ['inverse_matrix', 'matrix_rank'],
        'color': '#FF9800',
        'icon': '📐'
    },
    'vector_space': {
        'name': '向量空间',
        'level': '核心',
        'time': 4,
        'description': '向量空间、子空间、基与维数',
        'depends': ['matrix_basic'],
        'color': '#FF9800',
        'icon': '📍'
    },
    'linear_transform': {
        'name': '线性变换',
        'level': '核心',
        'time': 4,
        'description': '线性变换的定义、矩阵表示、核与像',
        'depends': ['vector_space', 'matrix_mult'],
        'color': '#FF9800',
        'icon': '🔀'
    },
    'eigenvalue': {
        'name': '特征值与特征向量',
        'level': '核心',
        'time': 6,
        'description': '特征值、特征向量的定义、计算、性质',
        'depends': ['linear_equations', 'linear_transform'],
        'color': '#FF9800',
        'icon': '⚡'
    },
    'diagonalization': {
        'name': '矩阵对角化',
        'level': '进阶',
        'time': 4,
        'description': '相似矩阵、对角化条件、对角化方法',
        'depends': ['eigenvalue'],
        'color': '#9C27B0',
        'icon': '📉'
    },
    'jordan_form': {
        'name': 'Jordan标准型',
        'level': '高阶',
        'time': 8,
        'description': 'Jordan块、Jordan标准型的求法',
        'depends': ['diagonalization'],
        'color': '#9C27B0',
        'icon': '📋'
    },
    'quadratic_form': {
        'name': '二次型',
        'level': '进阶',
        'time': 5,
        'description': '二次型的矩阵表示、标准形、正定性',
        'depends': ['eigenvalue'],
        'color': '#9C27B0',
        'icon': '📊'
    },
    'svd': {
        'name': '奇异值分解',
        'level': '应用',
        'time': 5,
        'description': 'SVD的定义、计算、应用',
        'depends': ['eigenvalue'],
        'color': '#E91E63',
        'icon': '🔍'
    },
    'pca': {
        'name': '主成分分析',
        'level': '应用',
        'time': 3,
        'description': 'PCA原理、计算步骤、实际应用',
        'depends': ['svd'],
        'color': '#E91E63',
        'icon': '📉'
    },
    'least_squares': {
        'name': '最小二乘法',
        'level': '应用',
        'time': 4,
        'description': '最小二乘问题、正规方程、几何意义',
        'depends': ['linear_equations', 'svd'],
        'color': '#E91E63',
        'icon': '📏'
    },
    'numerical_linear': {
        'name': '数值线性代数',
        'level': '高阶',
        'time': 10,
        'description': 'LU分解、QR分解、迭代法',
        'depends': ['linear_equations', 'svd'],
        'color': '#607D8B',
        'icon': '💻'
    }
}

def build_knowledge_graph() -> nx.DiGraph:
    """
    构建线性代数知识图谱
    
    Returns:
        NetworkX有向图
    """
    G = nx.DiGraph()
    
    # 添加节点
    for key, data in CONCEPTS.items():
        G.add_node(key, **data)
    
    # 添加边（前置知识关系）
    for key, data in CONCEPTS.items():
        for dep in data.get('depends', []):
            G.add_edge(dep, key, relation='前置知识', weight=2)
    
    # 添加额外的知识关联
    additional_edges = [
        ('determinant', 'eigenvalue', '应用'),
        ('matrix_rank', 'linear_equations', '判定'),
        ('inverse_matrix', 'linear_transform', '应用'),
        ('diagonalization', 'quadratic_form', '应用'),
        ('svd', 'numerical_linear', '基础'),
    ]
    
    for src, dst, rel in additional_edges:
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst, relation=rel, weight=1)
    
    return G

def recommend_learning_path(user_history: List[Dict], G: nx.DiGraph) -> List[str]:
    """
    基于用户计算历史推荐学习路径
    
    Args:
        user_history: 用户计算历史
        G: 知识图谱
        
    Returns:
        推荐的学习路径（节点列表）
    """
    if not user_history:
        # 默认从基础开始
        return ['matrix_basic', 'determinant', 'inverse_matrix']
    
    # 统计用户操作类型
    op_types = [h.get('type', '') for h in user_history]
    
    # 映射运算类型到知识点
    type_to_concept = {
        '📊 行列式 (LU分解可视化)': 'determinant',
        '🔄 逆矩阵 (高斯-约当消元)': 'inverse_matrix',
        '⚡ 特征值 (QR迭代+几何解释)': 'eigenvalue',
        '📐 线性方程组 (完整消元过程)': 'linear_equations',
        '🔍 矩阵秩 (行最简型)': 'matrix_rank',
        '📈 全部运算 (批量计算)': 'determinant'
    }
    
    # 找出用户最常操作的类型
    from collections import Counter
    op_counts = Counter(op_types)
    most_common = op_counts.most_common(1)[0][0] if op_counts else ''
    
    current_concept = type_to_concept.get(most_common, 'matrix_basic')
    
    # 找到从基础到当前知识点的路径
    try:
        path = nx.shortest_path(G, 'matrix_basic', current_concept)
    except nx.NetworkXNoPath:
        path = ['matrix_basic']
    
    # 找下一步推荐（后继节点）
    successors = list(G.successors(current_concept))
    
    # 按学习时间和难度排序
    successors.sort(key=lambda x: (G.nodes[x].get('time', 0), 
                                    ['入门', '基础', '核心', '进阶', '高阶', '应用'].index(
                                        G.nodes[x].get('level', '入门'))))
    
    # 返回路径 + 推荐
    return path + successors[:2]

def visualize_knowledge_graph(G: nx.DiGraph, highlight_path: List[str] = None) -> go.Figure:
    """
    可视化知识图谱
    
    Args:
        G: 知识图谱
        highlight_path: 要高亮显示的路径
        
    Returns:
        Plotly图形对象
    """
    # 使用spring layout布局
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 创建边
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(G.edges[edge[0], edge[1]].get('relation', ''))
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # 创建节点
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    node_symbols = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_data = G.nodes[node]
        name = node_data.get('name', node)
        level = node_data.get('level', '')
        time = node_data.get('time', 0)
        icon = node_data.get('icon', '📚')
        
        node_text.append(f"{icon} <b>{name}</b><br>难度: {level}<br>预计学习: {time}小时")
        
        # 颜色
        color = node_data.get('color', '#607D8B')
        if highlight_path and node in highlight_path:
            color = '#FF5722'  # 高亮颜色
        node_color.append(color)
        
        # 大小
        size = 30 + len(list(G.successors(node))) * 5
        node_size.append(size)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[G.nodes[n].get('name', n) for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=10),
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=2,
            line_color='white'
        )
    )
    
    # 创建图形
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text='🗺️ 线性代数知识图谱',
                font=dict(size=20)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500
        )
    )
    
    return fig

def get_prerequisites(G: nx.DiGraph, concept: str) -> List[str]:
    """获取某知识点的前置知识"""
    return list(nx.ancestors(G, concept))

def get_learning_order(G: nx.DiGraph, target: str) -> List[str]:
    """获取学习某知识点的推荐顺序"""
    try:
        # 拓扑排序
        topo_order = list(nx.topological_sort(G))
        
        # 找到目标位置
        target_idx = topo_order.index(target)
        
        # 返回从开始到目标的路径
        return topo_order[:target_idx + 1]
    except:
        return []

def estimate_learning_time(G: nx.DiGraph, path: List[str]) -> int:
    """估计学习路径所需时间"""
    total_time = 0
    for node in path:
        total_time += G.nodes[node].get('time', 0)
    return total_time

def get_difficulty_distribution(G: nx.DiGraph) -> Dict[str, int]:
    """获取知识点难度分布"""
    levels = ['入门', '基础', '核心', '进阶', '高阶', '应用']
    distribution = {level: 0 for level in levels}
    
    for node in G.nodes():
        level = G.nodes[node].get('level', '入门')
        distribution[level] = distribution.get(level, 0) + 1
    
    return distribution
