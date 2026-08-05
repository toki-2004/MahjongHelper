import os
import cv2
import numpy as np
import json

TEMPLATE_PATH = "templates"
CONFIG_PATH = "config.json"

# 牌映射（同 vision.py）
ID_TO_SUIT_VAL = {}
for i in range(9):
    ID_TO_SUIT_VAL[i] = (0, i+1)          # 万
for i in range(9, 18):
    ID_TO_SUIT_VAL[i] = (1, i-8)          # 索
for i in range(18, 27):
    ID_TO_SUIT_VAL[i] = (2, i-17)         # 饼
for i in range(27, 34):
    ID_TO_SUIT_VAL[i] = (3, 0)            # 字牌

# 反转映射
VAL_TO_ID = {v: k for k, v in ID_TO_SUIT_VAL.items()}  # 注意：字牌v=(3,0)会冲突，但后面会用其它方式

# 加载模板（返回列表）
def load_templates():
    templates = []
    for i in range(34):
        path = os.path.join(TEMPLATE_PATH, f"{i}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.append((i, img))
            else:
                print(f"警告：{path} 读取失败")
        else:
            # 模板不存在时仅使用内存占位并警告，不写入磁盘，
            # 避免在任意工作目录生成垃圾模板文件
            placeholder = np.ones((70, 50), dtype=np.uint8) * 255
            templates.append((i, placeholder))
            print(f"生成占位模板：{i}.png")
    rebuild_template_matrix(templates)
    return templates

def rebuild_template_matrix(tpl_list=None):
    """预计算模板的归一化向量矩阵，用于快速相关匹配（等价于 TM_CCOEFF_NORMED）。"""
    global TEMPLATE_MATRIX, TEMPLATE_SIZE
    if tpl_list is None:
        tpl_list = templates
    if not tpl_list:
        return
    size = tpl_list[0][1].shape
    rows = []
    for _, tpl in tpl_list:
        vec = tpl.astype(np.float32).reshape(-1)
        vec = vec - vec.mean()
        n = np.linalg.norm(vec)
        rows.append(vec / n if n > 1e-6 else np.zeros_like(vec))
    TEMPLATE_MATRIX = np.array(rows, dtype=np.float32)
    TEMPLATE_SIZE = size

# 更新单个模板（在线替换）
def update_template(tile_id, roi_gray):
    """
    用新样本 roi_gray 直接替换对应模板，保存并重新加载
    """
    path = os.path.join(TEMPLATE_PATH, f"{tile_id}.png")
    new = roi_gray
    # 与当前模板列表保持尺寸一致，避免破坏矩阵预计算
    if TEMPLATE_SIZE is not None:
        tpl_h, tpl_w = TEMPLATE_SIZE
        if new.shape != (tpl_h, tpl_w):
            new = cv2.resize(new, (tpl_w, tpl_h))
    elif os.path.exists(path):
        old = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if old is not None and new.shape != old.shape:
            new = cv2.resize(new, (old.shape[1], old.shape[0]))
    cv2.imwrite(path, new)
    # 重新加载整个模板列表（或只更新内存中的）
    global templates
    templates = load_templates()
    return True

# 获取牌名（用于显示）
def get_tile_name(tile_id):
    if tile_id < 0:
        return '?'
    suit, val = ID_TO_SUIT_VAL.get(tile_id, (3, 0))
    suit_map = {0: 'w', 1: 's', 2: 'p', 3: ''}
    if suit == 3:
        # 字牌特殊映射：ID 27~33 对应 东南西北中发白
        names = ['东', '南', '西', '北', '中', '发', '白']
        return names[tile_id - 27] if 0 <= tile_id-27 < 7 else '字'
    return f"{val}{suit_map[suit]}"

# 全局模板变量（供vision模块使用）
templates = load_templates()
