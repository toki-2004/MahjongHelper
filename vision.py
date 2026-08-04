import cv2
import numpy as np
import os
import template_manager as tm
from template_manager import ID_TO_SUIT_VAL, get_tile_name

OUTPUT_PATH = "debug_output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

def match_template_with_scores(roi):
    """
    向量化模板匹配，与 cv2.matchTemplate(TM_CCOEFF_NORMED) 数学等价（误差 < 1e-5）。
    返回 [(tile_id, score), ...] 按分数降序。
    """
    tpl_h, tpl_w = tm.TEMPLATE_SIZE
    if roi.shape != (tpl_h, tpl_w):
        roi_resized = cv2.resize(roi, (tpl_w, tpl_h))
    else:
        roi_resized = roi
    vec = roi_resized.astype(np.float32).reshape(-1)
    vec = vec - vec.mean()
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return [(tid, 0.0) for tid in range(len(tm.TEMPLATE_MATRIX))]
    scores = (tm.TEMPLATE_MATRIX @ vec) / norm
    order = np.argsort(-scores)
    return [(int(tid), float(scores[tid])) for tid in order]

def viterbi_correct(tile_candidates):
    """
    仅基于动态规划选择最优序列，不加任何强制修正。
    惩罚花色突变，但保留原始候选结果。
    """
    n = len(tile_candidates)
    if n == 0:
        return []
    top_k = 5
    candidates = []
    for cand_list in tile_candidates:
        if cand_list:
            candidates.append(cand_list[:top_k])
        else:
            candidates.append([])

    dp = []
    choices = []
    for i in range(n):
        dp.append([-1e9] * len(candidates[i]) if candidates[i] else [])
        choices.append([-1] * len(candidates[i]) if candidates[i] else [])
        for j, (tid, conf) in enumerate(candidates[i]):
            # -1 表示该槽位未识别，按字牌处理且不产生惩罚
            if tid < 0:
                suit_i, val_i = (3, 0)
            else:
                suit_i, val_i = ID_TO_SUIT_VAL[tid]
            if i == 0:
                dp[i][j] = conf
            else:
                best_prev = -1e9
                best_idx = -1
                for k, (prev_tid, prev_conf) in enumerate(candidates[i-1]):
                    if prev_tid < 0:
                        suit_prev, val_prev = (3, 0)
                    else:
                        suit_prev, val_prev = ID_TO_SUIT_VAL[prev_tid]
                    if suit_i == suit_prev and suit_i != 3:
                        diff = abs(val_i - val_prev)
                        penalty = 100 if diff > 2 else 0
                    else:
                        penalty = 0
                    score = dp[i-1][k] + conf - penalty
                    if score > best_prev:
                        best_prev = score
                        best_idx = k
                dp[i][j] = best_prev
                choices[i][j] = best_idx

    if not dp[-1]:
        return []
    last_idx = np.argmax(dp[-1])
    seq_ids = []
    for i in range(n-1, -1, -1):
        seq_ids.append(candidates[i][last_idx][0])
        if i > 0 and choices[i][last_idx] != -1:
            last_idx = choices[i][last_idx]
    seq_ids.reverse()
    return seq_ids

def extract_regions_by_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 120])
    upper = np.array([180, 80, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > 2000:
            rects.append((x, y, w, h))
    rects.sort(key=lambda r: r[2]*r[3], reverse=True)
    return rects

def _scan_region(region_gray, num_tiles, tile_width, offset):
    """按给定分档数和偏移扫描，返回 (candidates, confs, ok)。"""
    cands = []
    confs = []
    width = region_gray.shape[1]
    for i in range(num_tiles):
        left = offset + i * tile_width
        right = left + tile_width
        if right > width:
            return [], [], False
        roi = region_gray[:, left:right]
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            return [], [], False
        scores = match_template_with_scores(roi)
        if scores and scores[0][1] > 0.3:
            cands.append(scores[:3])
            confs.append(scores[0][1])
        else:
            cands.append([])
            confs.append(0.0)
    return cands, confs, True

def _count_bonus(num_tiles):
    """麻将手牌主行通常是 13~14 张，给轻微偏好，仅在置信度接近时起作用。"""
    if num_tiles in (13, 14):
        return 0.015
    if num_tiles in (12, 15):
        return 0.007
    return 0.0

def recognize_region(gray, x, y, w, h):
    """
    在区域中确定牌数并做对齐偏移搜索：
    对每个分档数在多个偏移下扫描，以“平均匹配置信度优先、有效槽数次之”
    选最优组合，避免仅按 0 偏移导致正确分档被淘汰。
    返回 (tile_candidates, count, tile_width, offset)。
    """
    region_gray = gray[y:y+h, x:x+w]
    best_key = None
    best = None
    for num_tiles in range(10, 19):
        tile_width = w // num_tiles
        if tile_width < 20:
            continue
        step = max(4, tile_width // 4)
        for offset in range(0, tile_width, step):
            cands, confs, ok = _scan_region(region_gray, num_tiles, tile_width, offset)
            if not ok:
                continue
            valid = sum(1 for c in confs if c > 0)
            key = (float(np.mean(confs)) + _count_bonus(num_tiles), valid)
            if best_key is None or key > best_key:
                best_key = key
                best = (num_tiles, offset, tile_width, cands)
    if best is None:
        return None, None, None, None

    num_tiles, offset, tile_width, tile_candidates = best
    # 未识别的槽位用 -1 占位，避免被误报为 1w
    for i in range(len(tile_candidates)):
        if not tile_candidates[i]:
            tile_candidates[i] = [(-1, 0.0)]
    valid = sum(1 for c in tile_candidates if c[0][0] >= 0)
    return tile_candidates, valid, tile_width, offset

def recognize_tiles_from_image(img_bgr):
    """
    输入: BGR图像 (numpy array)
    返回: (tile_ids, boxes, debug_img)
        tile_ids: 识别出的牌ID列表（从左到右，可能包含占位）
        boxes: 每个牌对应的矩形框 (x, y, w, h)
        debug_img: 带绘制框的图像（用于显示）
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = extract_regions_by_color(img_bgr)
    if not rects:
        return [], [], img_bgr

    # 主区域
    x, y, w, h = rects[0]
    tile_candidates, count, tile_width, offset = recognize_region(gray, x, y, w, h)
    if tile_candidates is None:
        return [], [], img_bgr

    # 第二区域（摸到的牌）
    x2, y2, w2, h2 = None, None, None, None
    if len(rects) > 1:
        x2, y2, w2, h2 = rects[1]
        if x2 > x + w and abs(y2 - y) < 20:
            gray2 = gray[y2:y2+h2, x2:x2+w2]
            if gray2.size > 0:
                tpl_h, tpl_w = tm.TEMPLATE_SIZE
                gray2_resized = cv2.resize(gray2, (tpl_w, tpl_h))
                scores = match_template_with_scores(gray2_resized)
                if scores:
                    tile_candidates.append(scores[:3])
                else:
                    tile_candidates.append([])

    corrected_ids = viterbi_correct(tile_candidates)

    # 构建boxes
    boxes = []
    for i, tid in enumerate(corrected_ids):
        if i < len(corrected_ids) - 1 and i < len(tile_candidates) - 1:
            left = x + offset + i * tile_width
            right = left + tile_width
            boxes.append((left, y, tile_width, h))
        else:
            # 第二区域或末尾
            if x2 is not None:
                boxes.append((x2, y2, w2, h2))
            else:
                last_left = x + offset + (len(corrected_ids)-1) * tile_width
                boxes.append((last_left, y, tile_width, h))

    # 绘制调试图
    debug_img = img_bgr.copy()
    for i, (bx, by, bw, bh) in enumerate(boxes):
        cv2.rectangle(debug_img, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
        label = get_tile_name(corrected_ids[i]) if i < len(corrected_ids) else '?'
        cv2.putText(debug_img, label, (bx, by-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return corrected_ids, boxes, debug_img
