# -*- coding: utf-8 -*-
"""视觉识别模块 v3

识别链路与模板采集工具（template_capture.py）完全一致：
- 几何检测定位每张牌（tile_detect.detect_tile_boxes，不依赖任何模板）；
- 原始像素匹配（不做预处理，保留电子麻将画面的全部判别细节）；
- 样本级模板匹配（kNN 式）+ 置信度与二三名差距双重闸门。
"""

import os

import cv2
import numpy as np

import template_manager as tm
from template_manager import get_tile_name
from tile_detect import detect_tile_boxes

OUTPUT_PATH = "debug_output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

MIN_CONF = 0.45    # 接受识别的最低置信度
CONF_GAP = 0.03    # 第一名与第二名的最小差距，防止临界牌误判


def _resize_tile(roi):
    """Only resize to template size; keep raw pixels."""
    h, w = tm.TEMPLATE_SIZE
    if roi.ndim == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    if roi.shape[:2] != (h, w):
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
    return roi


def match_template_with_scores(roi):
    """
    Color template matching: normalized correlation against every template
    (equivalent to CCOEFF_NORMED on the flattened BGR vector).
    """
    roi = _resize_tile(roi)
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


def match_template_with_scores_exemplar(roi, top_k=8):
    """
    Sample-level matching (kNN style): take the top_k candidates from the
    average templates, then rescore against every aligned sample of those
    classes and keep the per-class best score. Falls back to template scores.
    """
    base = match_template_with_scores(roi)
    if not base:
        return base
    roi = _resize_tile(roi)
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
    for tid, s in base:
        if tid not in best or s > best[tid]:
            best[tid] = s
    return sorted(best.items(), key=lambda x: -x[1])


def recognize_tiles_from_image(img_bgr, debug_tag=None):
    """
    Recognize every tile in a screenshot / selected region.

    Same pipeline as the template capture tool: geometry detection for tile
    positions, then raw-pixel sample-level matching per tile. Returns
    (tile_ids, boxes, debug_img); boxes are relative to the input image.
    """
    boxes, _ = detect_tile_boxes(img_bgr)
    if not boxes:
        return [], [], img_bgr

    ids = []
    debug_img = img_bgr.copy()
    for bx, by, bw, bh in boxes:
        roi = img_bgr[by:by + bh, bx:bx + bw]
        if roi.size == 0:
            ids.append(-1)
            continue
        scores = match_template_with_scores_exemplar(roi)
        best_id, best_conf = -1, 0.0
        if scores:
            best_id, best_conf = scores[0]
            second_conf = scores[1][1] if len(scores) > 1 else -1.0
            if not (best_id >= 0 and best_conf >= MIN_CONF
                    and (best_conf - second_conf) >= CONF_GAP):
                best_id = -1
        ids.append(best_id)

        cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        label = get_tile_name(best_id)
        cv2.putText(debug_img, label, (bx, by - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if scores:
            cv2.putText(debug_img, "%.2f" % scores[0][1], (bx + bw - 50, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    if debug_tag:
        out_path = os.path.join(OUTPUT_PATH, "result_%s.png" % debug_tag)
        cv2.imwrite(out_path, debug_img)
    return ids, boxes, debug_img
