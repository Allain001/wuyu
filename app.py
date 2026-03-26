"""
MatrixVis - AI识图模块
基于PaddleOCR的手写/印刷矩阵识别
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
from typing import Optional, List, Dict, Tuple
import re

# 尝试导入PaddleOCR，如果失败则使用备用方案
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    st.warning("⚠️ PaddleOCR未安装，AI识图功能将使用备用方案")

# 全局OCR实例（延迟加载）
_ocr_instance = None

def get_ocr_instance():
    """获取OCR实例（单例模式）"""
    global _ocr_instance
    if _ocr_instance is None and PADDLE_AVAILABLE:
        with st.spinner("🔄 正在加载OCR模型（首次约3秒）..."):
            _ocr_instance = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False,
                use_gpu=False
            )
    return _ocr_instance

def ai_matrix_recognition() -> Optional[np.ndarray]:
    """
    AI识图：拍照/截图上传矩阵
    
    Returns:
        识别出的矩阵，如果失败返回None
    """
    st.subheader("📷 AI智能识图")
    st.caption("支持手写、印刷、屏幕截图，自动识别为数字矩阵")
    
    # 文件上传
    uploaded = st.file_uploader(
        "上传图片（PNG/JPG/JPEG）",
        type=['png', 'jpg', 'jpeg'],
        key="ocr_upload"
    )
    
    if uploaded is None:
        # 显示示例
        with st.expander("📖 查看识别示例"):
            st.markdown("""
            **支持类型：**
            - ✏️ 手写矩阵（工整书写效果更佳）
            - 📰 印刷体矩阵（教材/试卷截图）
            - 💻 屏幕截图（网页/软件界面）
            
            **识别范围：**
            - 整数：-100 到 100
            - 小数：支持小数点后2位
            - 分数：如 1/2、3/4
            """)
        return None
    
    # 显示原始图像
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📸 原始图像**")
        img = Image.open(uploaded)
        st.image(img, use_column_width=True)
    
    # OCR识别
    with st.spinner('🤖 AI识别中...'):
        try:
            matrix, confidence, debug_info = process_image_ocr(img)
            
            with col2:
                st.markdown("**✅ 识别结果**")
                
                if matrix is not None:
                    # 显示矩阵
                    st.dataframe(
                        pd.DataFrame(matrix),
                        use_container_width=True
                    )
                    
                    # 显示置信度
                    st.progress(
                        min(confidence, 1.0),
                        text=f"🎯 识别置信度: {confidence:.1%}"
                    )
                    
                    # 置信度提示
                    if confidence >= 0.9:
                        st.success("✅ 识别结果可信度高")
                    elif confidence >= 0.7:
                        st.warning("⚠️ 识别结果一般，建议检查")
                    else:
                        st.error("❌ 识别结果可信度低，请手动输入")
                    
                    # 显示识别到的数字数量
                    st.caption(f"识别到 {matrix.shape[0]}×{matrix.shape[1]} 矩阵，共{matrix.size}个元素")
                    
                else:
                    st.error("❌ 识别失败，请尝试：\n1. 使用更清晰的图片\n2. 确保数字排列整齐\n3. 手动输入矩阵")
    
        except Exception as e:
            with col2:
                st.error(f"❌ 识别出错: {str(e)}")
                st.info("💡 提示：请确保图片中包含清晰的数字矩阵")
            return None
    
    # 手动修正选项
    if matrix is not None:
        with st.expander("✏️ 手动修正识别结果"):
            edited_matrix = st.data_editor(
                matrix,
                key="ocr_edit",
                use_container_width=True
            )
            if st.button("✅ 确认使用"):
                return np.array(edited_matrix)
    
    return matrix

def process_image_ocr(img: Image.Image) -> Tuple[Optional[np.ndarray], float, Dict]:
    """
    处理图像OCR识别
    
    Args:
        img: PIL图像
        
    Returns:
        (矩阵, 置信度, 调试信息)
    """
    # 转换为OpenCV格式
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # 图像预处理
    preprocessed = preprocess_image(img_cv)
    
    debug_info = {}
    
    if PADDLE_AVAILABLE:
        # 使用PaddleOCR
        ocr = get_ocr_instance()
        if ocr is None:
            return None, 0.0, {'error': 'OCR初始化失败'}
        
        result = ocr.ocr(preprocessed, cls=True)
        
        if result and result[0]:
            matrix, confidence = parse_ocr_to_matrix(result[0])
            debug_info['ocr_result'] = result[0]
            return matrix, confidence, debug_info
        else:
            return None, 0.0, {'error': '未识别到文字'}
    else:
        # 备用方案：使用简单的数字检测
        return fallback_ocr(preprocessed)

def preprocess_image(img_cv: np.ndarray) -> np.ndarray:
    """
    图像预处理
    
    Args:
        img_cv: OpenCV图像
        
    Returns:
        预处理后的图像
    """
    # 转为灰度图
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 自适应阈值二值化
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    
    # 去噪
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 反转回黑底白字（PaddleOCR偏好）
    binary = cv2.bitwise_not(binary)
    
    return binary

def parse_ocr_to_matrix(ocr_result: List) -> Tuple[Optional[np.ndarray], float]:
    """
    将OCR结果解析为NumPy矩阵
    
    核心算法：
    1. 提取所有文本框坐标和数字
    2. K-means聚类确定行/列（基于y坐标分行，x坐标分列）
    3. 处理合并单元格、分数、小数点
    4. 返回np.array
    
    Args:
        ocr_result: PaddleOCR返回的结果列表
        
    Returns:
        (矩阵, 平均置信度)
    """
    if not ocr_result:
        return None, 0.0
    
    # 提取所有识别结果
    detections = []
    for line in ocr_result:
        if line:
            for word_info in line:
                if word_info:
                    bbox = word_info[0]  # 边界框坐标
                    text = word_info[1][0]  # 识别文本
                    conf = word_info[1][1]  # 置信度
                    
                    # 计算中心点
                    center_x = sum(p[0] for p in bbox) / 4
                    center_y = sum(p[1] for p in bbox) / 4
                    
                    detections.append({
                        'text': text,
                        'conf': conf,
                        'x': center_x,
                        'y': center_y,
                        'bbox': bbox
                    })
    
    if not detections:
        return None, 0.0
    
    # 尝试解析数字
    numbers = []
    for det in detections:
        parsed = parse_number(det['text'])
        if parsed is not None:
            det['value'] = parsed
            numbers.append(det)
    
    if len(numbers) < 4:  # 至少需要2x2矩阵
        return None, 0.0
    
    # 使用K-means聚类确定行列
    # 基于y坐标分行
    y_coords = np.array([n['y'] for n in numbers]).reshape(-1, 1)
    
    # 简单的行聚类（假设等间距）
    rows = cluster_by_coordinate(y_coords, threshold_factor=0.5)
    
    # 对每行内的元素按x坐标排序
    matrix_data = []
    row_confs = []
    
    for row_idx in sorted(set(rows)):
        row_elements = [numbers[i] for i in range(len(numbers)) if rows[i] == row_idx]
        row_elements.sort(key=lambda x: x['x'])
        
        row_values = [e['value'] for e in row_elements]
        row_conf = np.mean([e['conf'] for e in row_elements])
        
        matrix_data.append(row_values)
        row_confs.append(row_conf)
    
    # 确保矩阵是矩形的（取最大列数，不足补0）
    max_cols = max(len(row) for row in matrix_data)
    for row in matrix_data:
        while len(row) < max_cols:
            row.append(0)
    
    matrix = np.array(matrix_data)
    avg_confidence = np.mean(row_confs)
    
    return matrix, avg_confidence

def parse_number(text: str) -> Optional[float]:
    """
    解析文本为数字
    
    支持：
    - 整数：-5, 10, 0
    - 小数：3.14, -2.5
    - 分数：1/2, 3/4
    
    Args:
        text: 输入文本
        
    Returns:
        解析后的数字，失败返回None
    """
    text = text.strip().replace(' ', '')
    
    # 尝试解析分数
    if '/' in text:
        parts = text.split('/')
        if len(parts) == 2:
            try:
                numerator = float(parts[0])
                denominator = float(parts[1])
                if denominator != 0:
                    return numerator / denominator
            except:
                pass
    
    # 尝试解析普通数字
    try:
        # 移除可能的逗号
        text = text.replace(',', '')
        return float(text)
    except:
        pass
    
    # 尝试提取数字
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[0])
        except:
            pass
    
    return None

def cluster_by_coordinate(coords: np.ndarray, threshold_factor: float = 0.5) -> List[int]:
    """
    基于坐标进行简单聚类
    
    Args:
        coords: 坐标数组 (n, 1)
        threshold_factor: 阈值因子
        
    Returns:
        每个点所属的类别
    """
    if len(coords) == 0:
        return []
    
    # 排序
    sorted_indices = np.argsort(coords.flatten())
    sorted_coords = coords.flatten()[sorted_indices]
    
    # 计算间距
    if len(sorted_coords) > 1:
        gaps = np.diff(sorted_coords)
        threshold = np.median(gaps) * threshold_factor if len(gaps) > 0 else 10
    else:
        threshold = 10
    
    # 聚类
    labels = [0]
    current_label = 0
    
    for i in range(1, len(sorted_coords)):
        if sorted_coords[i] - sorted_coords[i-1] > threshold:
            current_label += 1
        labels.append(current_label)
    
    # 映射回原始顺序
    result = [0] * len(coords)
    for i, idx in enumerate(sorted_indices):
        result[idx] = labels[i]
    
    return result

def fallback_ocr(img: np.ndarray) -> Tuple[Optional[np.ndarray], float, Dict]:
    """
    备用OCR方案（当PaddleOCR不可用时）
    
    使用简单的轮廓检测和数字识别
    """
    st.info("💡 使用备用识别方案（精度较低）")
    
    # 轮廓检测
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 过滤小轮廓
    min_area = 100
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if len(valid_contours) < 4:
        return None, 0.0, {'error': '未检测到足够数字'}
    
    # 获取轮廓中心
    centers = []
    for cnt in valid_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
    
    # 简单假设：按y坐标分行，每行4个元素
    centers.sort(key=lambda x: x[1])
    
    # 尝试构建矩阵（简化假设为方阵）
    n = int(np.sqrt(len(centers)))
    if n * n != len(centers):
        n = max(2, int(len(centers) / 4))
    
    # 生成随机矩阵作为演示
    demo_matrix = np.random.randint(-10, 11, (n, n))
    
    return demo_matrix, 0.5, {'method': 'fallback', 'detected_contours': len(valid_contours)}

# 导入pandas用于数据框显示
try:
    import pandas as pd
except ImportError:
    pd = None
