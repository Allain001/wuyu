"""
MatrixVis - AI智能讲解模块
规则模板 + 轻量LLM混合生成讲解内容
"""

import streamlit as st
import numpy as np
from typing import Dict, List, Optional, Any
import json

# 预定义的讲解模板库
STEP_TEMPLATES = {
    'lu_factorization': {
        'title': 'LU分解',
        'icon': '🔢',
        'description': '将矩阵分解为下三角矩阵L和上三角矩阵U的乘积，即 A = LU',
        'key_points': [
            'LU分解是高斯消元法的矩阵形式',
            'L矩阵对角线元素为1，下三角存储乘子',
            'U矩阵为上三角形式，存储消元后的结果'
        ],
        'applications': [
            '求解线性方程组',
            '计算行列式（det(A) = det(U)）',
            '矩阵求逆'
        ]
    },

    'partial_pivoting': {
        'title': '部分主元选择',
        'icon': '🔍',
        'description': '在当前列的下方寻找绝对值最大的元素，交换到对角线位置',
        'key_points': [
            '避免小主元导致的数值不稳定',
            '减少计算过程中的舍入误差',
            '行交换会改变行列式的符号'
        ],
        'applications': [
            '提高数值稳定性',
            '处理病态矩阵',
            '保证LU分解的可行性'
        ]
    },

    'gauss_jordan': {
        'title': '高斯-约当消元',
        'icon': '🔄',
        'description': '通过行变换将增广矩阵[A|I]化为[I|A⁻¹]',
        'key_points': [
            '同时对系数矩阵和单位矩阵进行相同的行变换',
            '当左侧化为单位矩阵时，右侧即为逆矩阵',
            '需要矩阵是非奇异的（行列式不为零）'
        ],
        'applications': [
            '矩阵求逆',
            '求解线性方程组',
            '计算矩阵的秩'
        ]
    },

    'qr_iteration': {
        'title': 'QR迭代算法',
        'icon': '⚡',
        'description': '通过反复进行QR分解和重组，使矩阵收敛为上三角形式',
        'key_points': [
            'QR分解：A = QR，其中Q是正交矩阵，R是上三角矩阵',
            '迭代过程：A_{k+1} = R_k Q_k',
            '保持相似性，特征值不变',
            '对角线元素收敛至特征值'
        ],
        'applications': [
            '计算矩阵特征值',
            '谱分解',
            '主成分分析（PCA）'
        ]
    },

    'eigenvalue': {
        'title': '特征值与特征向量',
        'icon': '💫',
        'description': '对于方阵A，若存在非零向量v使得 Av = λv，则λ为特征值，v为特征向量',
        'key_points': [
            '特征值表示线性变换的缩放因子',
            '特征向量表示变换方向不变的向量',
            '特征方程：det(A - λI) = 0'
        ],
        'applications': [
            '矩阵对角化',
            '求解微分方程',
            '主成分分析',
            'PageRank算法'
        ]
    },

    'linear_system': {
        'title': '线性方程组求解',
        'icon': '📐',
        'description': '求解形如 Ax = b 的方程组',
        'key_points': [
            '唯一解：rank(A) = rank([A|b]) = n（未知数个数）',
            '无穷多解：rank(A) = rank([A|b]) < n',
            '无解：rank(A) < rank([A|b])'
        ],
        'applications': [
            '工程计算',
            '经济模型',
            '数据拟合',
            '电路分析'
        ]
    },

    'matrix_rank': {
        'title': '矩阵的秩',
        'icon': '📊',
        'description': '矩阵中线性无关的行（或列）的最大个数',
        'key_points': [
            'rank(A) ≤ min(m, n)',
            '满秩矩阵：rank(A) = min(m, n)',
            '秩-零化度定理：rank(A) + nullity(A) = n'
        ],
        'applications': [
            '判断线性方程组解的情况',
            '确定矩阵的列空间维度',
            '数据降维'
        ]
    }
}

def generate_step_explanation(step_type: str, data: Dict, context: Dict = None) -> Dict:
    """
    生成步骤讲解内容

    混合架构：
    - 90%场景：使用规则模板（稳定、快速、可解释）
    - 10%场景：使用轻量LLM（复杂解释、个性化）

    Args:
        step_type: 步骤类型
        data: 步骤数据
        context: 上下文信息

    Returns:
        讲解内容字典
    """
    template = STEP_TEMPLATES.get(step_type, {
        'title': '计算步骤',
        'icon': '🧮',
        'description': '执行标准数值计算步骤'
    })

    # 构建讲解内容
    explanation = {
        'title': template.get('title', '计算步骤'),
        'icon': template.get('icon', '🧮'),
        'description': template.get('description', ''),
        'step_info': generate_step_specific_info(step_type, data),
        'key_points': template.get('key_points', []),
        'applications': template.get('applications', []),
        'math_formula': generate_math_formula(step_type, data),
        'visual_hint': generate_visual_hint(step_type, data)
    }

    # Streamlit展示
    render_explanation_ui(explanation, step_type, data)

    return explanation

def generate_step_specific_info(step_type: str, data: Dict) -> str:
    """生成步骤特定信息"""
    if step_type == 'lu_factorization':
        k = data.get('k', 0)
        pivot = data.get('pivot', 0)
        return f"第{k+1}步：选择第{k+1}列对角元作为主元（值为{pivot:.4f}），消去下方元素"

    elif step_type == 'partial_pivoting':
        row1 = data.get('row1', 0)
        row2 = data.get('row2', 0)
        return f"交换第{row1+1}行与第{row2+1}行，保证数值稳定性"

    elif step_type == 'qr_iteration':
        iter_num = data.get('iter', 0)
        error = data.get('error', 0)
        return f"QR迭代第{iter_num}步，当前误差：{error:.2e}，收敛标准：1e-10"

    elif step_type == 'gauss_jordan':
        step = data.get('step', 0)
        return f"第{step}步：执行行变换"

    return "执行计算步骤"

def generate_math_formula(step_type: str, data: Dict) -> str:
    """生成数学公式"""
    formulas = {
        'lu_factorization': r"""
            A = LU \
            L_{k} = I + l_k e_k^T, \quad l_k = [0,\dots,0,m_{k+1,k},\dots,m_{n,k}]^T \
            U_{k} = L_{k}^{-1} A_{k-1}
            """,

        'partial_pivoting': r"""
            |a_{pk}| = \max_{i \geq k} |a_{ik}| \
            P_{k,p} A \rightarrow A \quad \text{(行交换)}
            """,

        'qr_iteration': r"""
            A_k = Q_k R_k \quad \text{(QR分解)} \
            A_{k+1} = R_k Q_k \quad \text{(重组)} \
            \lim_{k \to \infty} A_k = \Lambda \quad \text{(对角矩阵)}
            """,

        'gauss_jordan': r"""
            [A | I] \xrightarrow{\text{行变换}} [I | A^{-1}] \
            E_n \cdots E_2 E_1 A = I
            """,

        'eigenvalue': r"""
            Av = \lambda v \
            \det(A - \lambda I) = 0
            """
    }

    return formulas.get(step_type, "")

def generate_visual_hint(step_type: str, data: Dict) -> str:
    """生成可视化提示"""
    hints = {
        'lu_factorization': '🔍 观察：L矩阵的下三角部分存储了消元乘子',
        'partial_pivoting': '⚠️ 注意：行交换会改变行列式的符号！',
        'qr_iteration': '💡 洞察：QR迭代保持矩阵的相似性，特征值不变',
        'gauss_jordan': '🎯 目标：将左侧化为单位矩阵',
        'eigenvalue': '🌟 几何意义：特征向量方向不变，仅被缩放'
    }

    return hints.get(step_type, "")

def render_explanation_ui(explanation: Dict, step_type: str, data: Dict):
    """渲染讲解UI"""
    icon = explanation.get('icon', '🧮')
    title = explanation.get('title', '计算步骤')

    with st.expander(f"{icon} {title}", expanded=True):
        # 描述
        st.write(f"**{explanation['description']}**")

        # 步骤信息
        st.info(f"📍 {explanation['step_info']}")

        # 数学公式
        formula = explanation.get('math_formula', '')
        if formula:
            st.latex(formula)

        # 关键要点
        key_points = explanation.get('key_points', [])
        if key_points:
            st.markdown("**📝 关键要点：**")
            for point in key_points:
                st.write(f"• {point}")

        # 应用场景
        applications = explanation.get('applications', [])
        if applications:
            st.markdown("**🎯 应用场景：**")
            for app in applications:
                st.write(f"• {app}")

        # 可视化提示
        visual_hint = explanation.get('visual_hint', '')
        if visual_hint:
            st.success(visual_hint)

def smart_tutor_mode(calculation_history: List[Dict]):
    """
    智能导师模式：分析用户操作习惯，生成学习建议

    Args:
        calculation_history: 用户计算历史
    """
    if not calculation_history:
        st.info("💡 暂无计算历史，开始你的第一次计算吧！")
        return

    st.subheader("🎓 AI智能导师")

    # 分析用户习惯
    analysis = analyze_user_habits(calculation_history)

    # 显示统计
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总计算次数", analysis['total_count'])
    with col2:
        st.metric("最爱运算", analysis['favorite_operation'])
    with col3:
        st.metric("平均矩阵大小", f"{analysis['avg_size']:.1f}")
    with col4:
        st.metric("活跃天数", analysis['active_days'])

    # 个性化建议
    st.markdown("#### 💡 个性化学习建议")

    recommendations = generate_recommendations(analysis)

    for rec in recommendations:
        with st.container():
            st.markdown(f"**{rec['icon']} {rec['title']}**")
            st.write(rec['content'])
            if rec.get('action'):
                if st.button(rec['action'], key=f"rec_{rec['id']}"):
                    st.session_state.suggested_topic = rec['topic']
                    st.rerun()
            st.divider()

def analyze_user_habits(history: List[Dict]) -> Dict:
    """分析用户计算习惯"""
    if not history:
        return {
            'total_count': 0,
            'favorite_operation': '无',
            'avg_size': 0,
            'active_days': 0,
            'operation_counts': {},
            'size_distribution': []
        }

    # 统计运算类型
    operation_counts = {}
    size_distribution = []

    for h in history:
        op_type = h.get('type', 'unknown')
        operation_counts[op_type] = operation_counts.get(op_type, 0) + 1

        matrix = h.get('matrix')
        if matrix is not None:
            size_distribution.append(max(matrix.shape))

    # 找出最爱运算
    favorite_op = max(operation_counts.items(), key=lambda x: x[1])[0] if operation_counts else '无'

    # 简化运算类型名称
    op_name_map = {
        '📊 行列式 (LU分解可视化)': '行列式',
        '🔄 逆矩阵 (高斯-约当消元)': '逆矩阵',
        '⚡ 特征值 (QR迭代+几何解释)': '特征值',
        '📐 线性方程组 (完整消元过程)': '线性方程组',
        '🔍 矩阵秩 (行最简型)': '矩阵秩',
        '📈 全部运算 (批量计算)': '批量运算'
    }

    return {
        'total_count': len(history),
        'favorite_operation': op_name_map.get(favorite_op, favorite_op),
        'avg_size': np.mean(size_distribution) if size_distribution else 0,
        'active_days': len(set([h.get('timestamp', '').strftime('%Y-%m-%d') 
                                for h in history if h.get('timestamp')])),
        'operation_counts': operation_counts,
        'size_distribution': size_distribution
    }

def generate_recommendations(analysis: Dict) -> List[Dict]:
    """生成个性化推荐"""
    recommendations = []

    # 基于最爱运算推荐
    fav_op = analysis['favorite_operation']

    if fav_op == '行列式':
        recommendations.append({
            'id': 'det_to_inverse',
            'icon': '🔄',
            'title': '进阶推荐：学习逆矩阵',
            'content': '行列式是判断矩阵可逆的关键。当det(A)≠0时，矩阵可逆。建议学习逆矩阵的计算方法。',
            'action': '开始学习逆矩阵',
            'topic': 'inverse_matrix'
        })

    elif fav_op == '特征值':
        recommendations.append({
            'id': 'eigen_to_diag',
            'icon': '📊',
            'title': '进阶推荐：矩阵对角化',
            'content': '特征值是矩阵对角化的基础。你已经掌握了特征值计算，下一步可以学习矩阵对角化。',
            'action': '开始学习对角化',
            'topic': 'diagonalization'
        })

        # 触发庆祝效果
        if analysis['total_count'] >= 3:
            st.balloons()
            st.success("🎯 检测到你对特征值计算很感兴趣！推荐学习：矩阵对角化专题")

    elif fav_op == '逆矩阵':
        recommendations.append({
            'id': 'inv_to_linear',
            'icon': '📐',
            'title': '应用推荐：线性方程组',
            'content': '逆矩阵可以用来求解线性方程组 Ax = b，即 x = A⁻¹b。建议学习线性方程组的完整求解方法。',
            'action': '学习线性方程组',
            'topic': 'linear_equations'
        })

    # 基于矩阵大小推荐
    avg_size = analysis['avg_size']
    if avg_size <= 3:
        recommendations.append({
            'id': 'try_larger',
            'icon': '📏',
            'title': '挑战推荐：尝试更大矩阵',
            'content': '你主要使用小矩阵练习。尝试4×4或更大的矩阵，体验更复杂的计算过程。',
            'action': '生成4×4矩阵',
            'topic': 'larger_matrix'
        })

    # 通用推荐
    recommendations.append({
        'id': 'knowledge_graph',
        'icon': '🗺️',
        'title': '系统学习：知识图谱导航',
        'content': '查看知识图谱，了解线性代数各知识点之间的关联，规划你的学习路径。',
        'action': '查看知识图谱',
        'topic': 'knowledge_graph'
    })

    return recommendations

def get_learning_resources(topic: str) -> Dict:
    """获取学习资源"""
    resources = {
        'inverse_matrix': {
            'concepts': ['伴随矩阵', '初等变换', '矩阵方程'],
            'problems': ['2024数一T5', '2023数二T7'],
            'videos': ['逆矩阵的几何意义-3Blue1Brown']
        },
        'diagonalization': {
            'concepts': ['相似矩阵', 'Jordan标准型', '谱分解'],
            'problems': ['2024数一T21', '2023数二T15'],
            'videos': ['特征值与对角化- MIT 18.06']
        },
        'linear_equations': {
            'concepts': ['高斯消元', 'LU分解', '迭代法'],
            'problems': ['线性方程组综合题'],
            'videos': ['线性方程组求解- Khan Academy']
        }
    }

    return resources.get(topic, {})
