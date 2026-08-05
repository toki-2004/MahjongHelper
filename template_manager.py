# -*- coding: utf-8 -*-
"""模板管理 v2：彩色（BGR）模板 + 多样本平均构建，识别域统一。"""
import os
import cv2
import numpy as np

TEMPLATE_PATH = "templates"
CONFIG_PATH = "config.json"
TEMPLATE_SIZE = (141, 96)  # (高, 宽)，所有模板统一尺寸

# 牌映射（供 vision.py / main.py）
ID_TO_SUIT_VAL = {}
for i in range(9):
    ID_TO_SUIT_VAL[i] = (0, i + 1)          # 万
for i in range(9, 18):
    ID_TO_SUIT_VAL[i] = (1, i - 8)          # 条
for i in range(18, 27):
    ID_TO_SUIT_VAL[i] = (2, i - 17)         # 筒
for i in range(27, 34):
    ID_TO_SUIT_VAL[i] = (3, 0)              # 字牌
VAL_TO_ID = {v: k for k, v in ID_TO_SUIT_VAL.items()}


def preprocess_tile(img):
    """
    牌面统一预处理（BGR）：
    1. 双边滤波降噪；
    2. LAB 通道 CLAHE 增强对比度（保留颜色）；
    3. 近白色像素置为纯白，去除底纹杂色（彩色花色不受影响）。
    """
    if img is None or img.size == 0:
        return img
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    denoised = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=5)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    out[gray >= 200] = 255
    return out


def _to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _resize_to_template(img):
    h, w = TEMPLATE_SIZE
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def load_templates():
    """从 TEMPLATE_PATH 加载 34 张模板并预处理，构建匹配矩阵。"""
    templates = []
    for i in range(34):
        path = os.path.join(TEMPLATE_PATH, f"{i}.png")
        img = None
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:  # 灰度图兼容
                g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if g is not None:
                    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if img is not None:
            templates.append((i, _resize_to_template(preprocess_tile(img))))
        else:
            templates.append((i, np.full((TEMPLATE_SIZE[0], TEMPLATE_SIZE[1], 3), 255, np.uint8)))
    rebuild_template_matrix(templates)
    return templates


def rebuild_template_matrix(tpl_list=None):
    """
    预计算模板的归一化向量矩阵（BGR 三通道拼接后去均值、单位化），
    等价于彩色 CCOEFF_NORMED 匹配。
    """
    global TEMPLATE_MATRIX, TEMPLATE_SIZE
    if tpl_list is None:
        tpl_list = templates
    if not tpl_list:
        return
    h, w = TEMPLATE_SIZE
    rows = []
    for _, tpl in tpl_list:
        vec = tpl.astype(np.float32).reshape(-1)
        vec = vec - vec.mean()
        n = np.linalg.norm(vec)
        rows.append(vec / n if n > 1e-6 else np.zeros_like(vec))
    TEMPLATE_MATRIX = np.array(rows, dtype=np.float32)
    TEMPLATE_SIZE = (h, w)


def update_template(tile_id, img):
    """在线修正：用新样本直接替换对应模板（预处理后保存并重新加载）。"""
    path = os.path.join(TEMPLATE_PATH, f"{tile_id}.png")
    new = _resize_to_template(preprocess_tile(_to_bgr(img)))
    cv2.imwrite(path, new)
    global templates
    templates = load_templates()
    return True


def build_template_library(origin_dir="templates_origin", samples_by_id=None, target_dir=None):
    """
    构建模板库：每张牌取 原始模板 + 全部标记样本 的均值，写入 target_dir。
    samples_by_id: {tile_id: [BGR ROI, ...]}
    """
    target_dir = target_dir or TEMPLATE_PATH
    os.makedirs(target_dir, exist_ok=True)
    built = []
    for i in range(34):
        refs = []
        op = os.path.join(origin_dir, f"{i}.png")
        if os.path.exists(op):
            img = cv2.imread(op, cv2.IMREAD_COLOR)
            if img is None:
                g = cv2.imread(op, cv2.IMREAD_GRAYSCALE)
                img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if g is not None else None
            if img is not None:
                refs.append(_resize_to_template(preprocess_tile(img)))
        if samples_by_id and i in samples_by_id:
            for s in samples_by_id[i]:
                if s is not None and s.size > 0:
                    refs.append(_resize_to_template(preprocess_tile(_to_bgr(s))))
        if refs:
            avg = np.mean(np.stack(refs), axis=0).astype(np.uint8)
            cv2.imwrite(os.path.join(target_dir, f"{i}.png"), avg)
            built.append((i, len(refs)))
    return built


def get_tile_name(tile_id):
    if tile_id < 0:
        return "?"
    suit, val = ID_TO_SUIT_VAL.get(tile_id, (3, 0))
    suit_map = {0: "w", 1: "s", 2: "p", 3: ""}
    if suit == 3:
        names = ["东", "南", "西", "北", "中", "发", "白"]
        return names[tile_id - 27] if 0 <= tile_id - 27 < 7 else "字"
    return f"{val}{suit_map[suit]}"


# 全局模板变量（供 vision.py 使用）
templates = load_templates()
