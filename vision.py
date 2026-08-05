# -*- coding: utf-8 -*-
"""
视觉识别模块 v2

- 彩色（BGR）模板匹配：同时利用花色颜色与形状，比灰度匹配更准；
- 牌面精确定位：按白色牌面连通域逐张裁剪，牌间间隙不入框；
- 维特比序列平滑：结合花色/点数连续性修正偶发误识；
- 识别结果图：debug_tag 非空时保存到 debug_output/result_{debug_tag}.png。
"""
import os
import cv2
import numpy as np
import template_manager as tm
from template_manager import ID_TO_SUIT_VAL, get_tile_name

OUTPUT_PATH = "debug_output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

MIN_CONF = 0.30  # 最低可接受匹配分数


def _corr2d(a, b):
    """两个等尺寸 BGR 图像的归一化相关（多通道）。"""
    pa = a.astype(np.float32) - a.astype(np.float32).mean()
    qa = b.astype(np.float32) - b.astype(np.float32).mean()
    return float((pa * qa).sum() / (np.linalg.norm(pa) * np.linalg.norm(qa) + 1e-9))


def match_template_with_scores(roi):
    """
    彩色模板匹配：把 ROI 与 34 张模板做多通道归一化相关
    （等价于 BGR 拼接向量的 CCOEFF_NORMED）。
    要求输入为已经过 preprocess_tile 预处理的 BGR 牌面 ROI（调用方统一处理）。
    返回 [(tile_id, score), ...] 按分数降序。
    """
    tpl_h, tpl_w = tm.TEMPLATE_SIZE
    if roi.ndim == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    if roi.shape[:2] != (tpl_h, tpl_w):
        roi = cv2.resize(roi, (tpl_w, tpl_h), interpolation=cv2.INTER_AREA)

    vec = roi.astype(np.float32).reshape(-1)
    vec = vec - vec.mean()
    norm = np.linalg.norm(vec)
    if tm.TEMPLATE_MATRIX is None or tm.TEMPLATE_MATRIX.shape[1] != vec.shape[0]:
        tm.rebuild_template_matrix()
    if norm < 1e-6:
        return [(tid, 0.0) for tid in range(len(tm.TEMPLATE_MATRIX))]

    scores = (tm.TEMPLATE_MATRIX @ vec) / norm
    order = np.argsort(-scores)
    return [(int(tid), float(scores[tid])) for tid in order]


def match_template_with_scores_fast(roi):
    """
    粗粒度灰度快速匹配：仅用于布局搜索（分档数/偏移），
    计算量约为全彩匹配的 1/30，最终判定由样本级匹配负责。
    """
    h, w = tm.COARSE_SIZE
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    small = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    vec = small.astype(np.float32).reshape(-1)
    vec = vec - vec.mean()
    norm = np.linalg.norm(vec)
    if tm.COARSE_MATRIX is None or tm.COARSE_MATRIX.shape[1] != vec.shape[0]:
        tm.rebuild_coarse_matrix()
    if norm < 1e-6:
        return [(tid, 0.0) for tid in range(len(tm.COARSE_MATRIX))]
    scores = (tm.COARSE_MATRIX @ vec) / norm
    order = np.argsort(-scores)
    return [(int(tid), float(scores[tid])) for tid in order]


def match_template_with_scores_exemplar(roi, top_k=8):
    """
    样本级匹配（kNN 式）：快速匹配取 top_k 候选后，用每张牌的全部对齐样本
    重新打分并取该牌最高分。平均模板会模糊细节，样本级匹配对临界牌（如 4w/8w）
    判别力更强。无样本时回退到平均模板分数。
    """
    base = match_template_with_scores(roi)
    if not base:
        return base
    tpl_h, tpl_w = tm.TEMPLATE_SIZE
    if roi.ndim == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    if roi.shape[:2] != (tpl_h, tpl_w):
        roi = cv2.resize(roi, (tpl_w, tpl_h), interpolation=cv2.INTER_AREA)
    vec = roi.astype(np.float32).reshape(-1)
    vec = vec - vec.mean()
    norm = np.linalg.norm(vec)

    best = {}
    if tm.EXEMPLAR_MATRIX is not None and norm > 1e-6:
        candidate_ids = set(tid for tid, _ in base[:top_k])
        rows_sel = [i for i, tid in enumerate(tm.EXEMPLAR_IDS) if tid in candidate_ids]
        if rows_sel:
            scores = (tm.EXEMPLAR_MATRIX[rows_sel] @ vec) / norm
            for s, ri in zip(scores, rows_sel):
                tid = int(tm.EXEMPLAR_IDS[ri])
                if s > best.get(tid, -1e9):
                    best[tid] = float(s)

    # 平均模板分数兜底（无样本的牌或分数更高时）
    for tid, s in base:
        if tid not in best or s > best[tid]:
            best[tid] = s
    return sorted(best.items(), key=lambda x: -x[1])


def viterbi_correct(tile_candidates):
    """
    基于动态规划选择最优序列：惩罚同一花色内点数突变，
    保留原始候选分数，输出识别最一致的序列。
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
            if tid < 0:
                suit_i, val_i = (3, 0)
            else:
                suit_i, val_i = ID_TO_SUIT_VAL[tid]
            if i == 0:
                dp[i][j] = conf
            else:
                best_prev = -1e9
                best_idx = -1
                for k, (prev_tid, prev_conf) in enumerate(candidates[i - 1]):
                    if prev_tid < 0:
                        suit_prev, val_prev = (3, 0)
                    else:
                        suit_prev, val_prev = ID_TO_SUIT_VAL[prev_tid]
                    if suit_i == suit_prev and suit_i != 3:
                        diff = abs(val_i - val_prev)
                        penalty = 100 if diff > 2 else 0
                    else:
                        penalty = 0
                    score = dp[i - 1][k] + conf - penalty
                    if score > best_prev:
                        best_prev = score
                        best_idx = k
                dp[i][j] = best_prev
                choices[i][j] = best_idx

    if not dp[-1]:
        return []
    last_idx = np.argmax(dp[-1])
    seq_ids = []
    for i in range(n - 1, -1, -1):
        seq_ids.append(candidates[i][last_idx][0])
        if i > 0 and choices[i][last_idx] != -1:
            last_idx = choices[i][last_idx]
    seq_ids.reverse()
    return seq_ids


def extract_regions_by_color(img):
    """按白色/浅色掩码提取牌行区域，返回按面积降序的 (x, y, w, h)。"""
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
    rects.sort(key=lambda r: r[2] * r[3], reverse=True)
    return rects


def _template_box(region_h, box_w):
    """按模板宽高比计算居中裁剪框，返回 (top, bottom)，避免缩放形变。"""
    tpl_h, tpl_w = tm.TEMPLATE_SIZE
    if tpl_h is None or tpl_w is None or box_w <= 0:
        return 0, region_h
    box_h = max(1, int(round(box_w * tpl_h / tpl_w)))
    top = max(0, (region_h - box_h) // 2)
    bottom = min(region_h, top + box_h)
    return top, bottom


def _tile_face_rects(region_gray, num_tiles, offset, tile_width):
    """
    按槽位定位每张牌的实际白色牌面，返回 [(x0, x1, top, bottom), ...]。
    牌面按白色连通域定位（大花色在牌面内部不会劈开白区），
    矩形只包住牌面并外扩少量边距，间隙留在框外。
    """
    region_h, region_w = region_gray.shape
    mask = (region_gray >= 200).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    comps = []  # [bbox_left, bbox_right, center_x]
    for i in range(1, n):
        sx, sy, sw, sh, area = stats[i]
        if sw >= 8 and sh >= 20 and area >= 200:
            comps.append([sx, sx + sw, (sx + sx + sw) // 2])

    rects = []
    margin = max(1, int(round(tile_width * 0.03)))
    for i in range(num_tiles):
        left = offset + i * tile_width
        right = left + tile_width
        # 收集中心落在本槽位内的白色碎片（严格归属，避免跨槽双算），取并集作为牌面
        in_pitch = [c for c in comps if left <= c[2] < right]
        if in_pitch:
            x0 = max(0, min(c[0] for c in in_pitch) - margin)
            x1 = min(region_w, max(c[1] for c in in_pitch) + margin)
        else:
            x0, x1 = left, min(region_w, right)
        if x1 - x0 < max(20, int(tile_width * 0.45)):
            # 未找到可靠牌面（如牌面偏暗）：退回完整槽位，覆盖整张牌
            x0, x1 = left, min(region_w, right)
        if x1 <= x0:
            x1 = min(region_w, x0 + 1)
        top, bottom = _template_box(region_h, x1 - x0)
        rects.append((x0, x1, top, bottom))
    return rects


def _normalize_rects(rects, region_w, region_h, common_w=None, clamp=True):
    """
    把同一张图内的识别框统一为相同宽高（取各牌面宽度的中位数），
    每框中心仍对准各自牌面，保证同一张图内所有绿框大小完全一致。
    """
    widths = [x1 - x0 for x0, x1, _, _ in rects if x1 > x0]
    if not widths:
        return rects
    w = common_w if common_w else int(np.median(widths))
    w = max(w, 20)
    top, bottom = _template_box(region_h, w)
    out = []
    for x0, x1, _, _ in rects:
        cx = (x0 + x1) // 2
        nx0 = cx - w // 2
        nx1 = nx0 + w
        if clamp:
            if nx0 < 0:
                nx0, nx1 = 0, w
            if nx1 > region_w:
                nx1 = region_w
                nx0 = max(0, nx1 - w)
        out.append((nx0, nx1, top, bottom))
    return out


def _scan_region(region_bgr, num_tiles, tile_width, offset):
    """按给定分档数和偏移扫描（槽位 ROI，BGR），返回 (candidates, confs, ok)。"""
    cands = []
    confs = []
    region_h, width = region_bgr.shape[:2]
    top, bottom = _template_box(region_h, tile_width)
    for i in range(num_tiles):
        left = offset + i * tile_width
        right = left + tile_width
        if right > width:
            return [], [], False
        roi = region_bgr[top:bottom, left:right]
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            return [], [], False
        scores = match_template_with_scores_fast(roi)
        if scores and scores[0][1] > MIN_CONF:
            cands.append(scores[:3])
            confs.append(scores[0][1])
        else:
            cands.append([])
            confs.append(0.0)
    return cands, confs, True


def _count_bonus(num_tiles):
    """手牌主行通常 13~14 张，仅在置信度接近时起偏好作用。"""
    if num_tiles in (13, 14):
        return 0.015
    if num_tiles in (12, 15):
        return 0.007
    return 0.0


def _tile_row_extent(region_gray):
    """
    估算牌行的横向范围（白色牌面的最小左缘到最大右缘），
    用于排除框选区域内牌行两侧的空白，避免少牌时被分档切碎。
    返回 (x0, x1)；找不到牌面时返回 None。
    """
    mask = (region_gray >= 180).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    lefts = []
    rights = []
    for i in range(1, n):
        sx, sy, sw, sh, area = stats[i]
        if sw >= 12 and sh >= 20 and area >= 300:
            lefts.append(sx)
            rights.append(sx + sw)
    if not lefts:
        return None
    return min(lefts), max(rights)


def recognize_region(region_bgr, region_gray, region_w, region_h):
    """
    在已预处理的主区域内搜索最佳分档数与偏移。
    返回 (tile_candidates, valid, tile_width, offset)。
    """
    # 只在实际牌行范围内分档（排除两侧空白），支持 3~18 张
    extent = _tile_row_extent(region_gray)
    if extent is None:
        extent = (0, region_w)
    ex0, ex1 = extent
    row_w = ex1 - ex0

    best_key = None
    best = None
    for num_tiles in range(3, 19):
        tile_width = row_w // num_tiles
        if tile_width < 20:
            continue
        step = max(4, tile_width // 4)
        for off in range(0, tile_width, step):
            cands, confs, ok = _scan_region(region_bgr, num_tiles, tile_width, ex0 + off)
            if not ok:
                continue
            valid = sum(1 for c in confs if c > 0)
            key = (float(np.mean(confs)) + _count_bonus(num_tiles), valid)
            if best_key is None or key > best_key:
                best_key = key
                best = (num_tiles, ex0 + off, tile_width, cands)
    if best is None:
        return None, None, None, None

    num_tiles, offset, tile_width, tile_candidates = best
    for i in range(len(tile_candidates)):
        if not tile_candidates[i]:
            tile_candidates[i] = [(-1, 0.0)]
    valid = sum(1 for c in tile_candidates if c[0][0] >= 0)
    return tile_candidates, valid, tile_width, offset


def recognize_tiles_from_image(img_bgr, debug_tag=None):
    """
    识别整张截图中的手牌。
    返回 (tile_ids, boxes, debug_img)。
    debug_tag 非空时把带框识别图保存到 debug_output/result_{debug_tag}.png。
    """
    rects = extract_regions_by_color(img_bgr)
    if not rects:
        return [], [], img_bgr

    # 主区域：预处理一次，BGR 用于匹配，灰度用于牌面定位
    x, y, w, h = rects[0]
    region_bgr = tm.preprocess_tile(img_bgr[y:y + h, x:x + w])
    region_gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)

    tile_candidates, count, tile_width, offset = recognize_region(region_bgr, region_gray, w, h)
    if tile_candidates is None:
        return [], [], img_bgr

    # 按实际牌面精确定位（间隙不入框）
    main_rects = _tile_face_rects(region_gray, len(tile_candidates), offset, tile_width)

    # 第二区域（摸到的牌）
    x2, y2, w2, h2 = None, None, None, None
    region2 = None
    second_rect = None
    if len(rects) > 1:
        x2, y2, w2, h2 = rects[1]
        if x2 > x + w and abs(y2 - y) < 20:
            region2 = tm.preprocess_tile(img_bgr[y2:y2 + h2, x2:x2 + w2])
            gray2 = cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY)
            if gray2.size > 0:
                second_rect = _tile_face_rects(gray2, 1, 0, w2)[0]

    # 统一识别框尺寸：同一张图内所有牌大小一致（取各牌面宽度中位数）
    all_rects = list(main_rects) + ([second_rect] if second_rect is not None else [])
    widths = [x1 - x0 for x0, x1, _, _ in all_rects if x1 > x0]
    common_w = min(int(np.median(widths)), tile_width) if widths else tile_width
    main_rects = _normalize_rects(main_rects, w, h, common_w)
    if second_rect is not None:
        second_rect = _normalize_rects([second_rect], w2, h2, common_w, clamp=False)[0]

    # 用统一识别框重新匹配主区域
    tile_candidates = []
    for x0, x1, top, bottom in main_rects:
        roi = region_bgr[top:bottom, x0:x1]
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            tile_candidates.append([])
            continue
        scores = match_template_with_scores_exemplar(roi)
        if scores and scores[0][1] > MIN_CONF:
            tile_candidates.append(scores[:3])
        else:
            tile_candidates.append([])
    for i in range(len(tile_candidates)):
        if not tile_candidates[i]:
            tile_candidates[i] = [(-1, 0.0)]

    # 第二区域匹配与成框
    second_box = None
    if second_rect is not None:
        rx0, rx1, top2, bottom2 = second_rect
        crop_x0 = max(0, min(rx0, w2 - 1))
        crop_x1 = min(w2, max(rx1, crop_x0 + 1))
        scores = match_template_with_scores_exemplar(region2[top2:bottom2, crop_x0:crop_x1])
        if scores:
            tile_candidates.append(scores[:3])
        else:
            tile_candidates.append([])
        second_box = (x2 + rx0, y2 + top2, rx1 - rx0, bottom2 - top2)

    # 逐槽位取最高分候选（彩色匹配已足够准确，序列平滑反而会引入错误）
    corrected_ids = [c[0][0] if c and c[0][0] >= 0 else -1 for c in tile_candidates]

    # 构建 boxes（主区域按实际牌面定位，间隙留在框外；第二区域单独成框）
    boxes = []
    for x0, x1, top, bottom in main_rects:
        boxes.append((x + x0, y + top, x1 - x0, bottom - top))
    if second_box is not None:
        boxes.append(second_box)

    # 绘制调试图
    debug_img = img_bgr.copy()
    for i, (bx, by, bw, bh) in enumerate(boxes):
        cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        if i < len(corrected_ids):
            label = get_tile_name(corrected_ids[i])
            cv2.putText(debug_img, label, (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if i < len(tile_candidates) and tile_candidates[i] and tile_candidates[i][0][0] >= 0:
                score = f"{tile_candidates[i][0][1]:.2f}"
                cv2.putText(debug_img, score, (bx + bw - 45, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        else:
            cv2.putText(debug_img, "?", (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if debug_tag:
        out_path = os.path.join(OUTPUT_PATH, f"result_{debug_tag}.png")
        cv2.imwrite(out_path, debug_img)
    return corrected_ids, boxes, debug_img
