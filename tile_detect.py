# -*- coding: utf-8 -*-
"""Geometry-only tile detection (no template dependency).

Locates individual mahjong tiles in a screenshot region using the white
tile face: HSV white mask -> connected components -> size/aspect filter ->
same-row filter -> center-gap clustering -> unified box normalization.

Works without any template image so it can be used to build the template
library from scratch. Returns boxes plus the detected tile count.
"""

import cv2
import numpy as np

# 主行手牌 + 摸牌后可能出现的牌数
VALID_TILE_COUNTS = {14, 13, 11, 10, 8, 7, 5, 4, 2, 1}

_WHITE_LOWER = np.array([0, 0, 150])
_WHITE_UPPER = np.array([180, 120, 255])

_MIN_W = 25
_MIN_H = 35
_RATIO_MIN = 0.45
_RATIO_MAX = 0.95


def _white_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, _WHITE_LOWER, _WHITE_UPPER)


def _tile_blocks(mask):
    """Return candidate tile-face blocks as (cx, x0, x1, y0, y1), sorted by cx."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for s in stats[1:]:
        x, y, w, h = int(s[0]), int(s[1]), int(s[2]), int(s[3])
        if w < _MIN_W or h < _MIN_H:
            continue
        ratio = w / float(h)
        if ratio < _RATIO_MIN or ratio > _RATIO_MAX:
            continue
        out.append((x + w // 2, x, x + w, y, y + h))
    out.sort(key=lambda b: b[0])
    return out


def _same_row(blocks):
    """Keep blocks whose vertical center is close to the median (main row only)."""
    if not blocks:
        return []
    heights = [b[4] - b[3] for b in blocks]
    med_h = float(np.median(heights))
    centers = [(b[3] + b[4]) // 2 for b in blocks]
    med_y = float(np.median(centers))
    tol = max(0.5 * med_h, 8.0)
    return [b for b, cy in zip(blocks, centers) if abs(cy - med_y) <= tol]


def _segments(blocks):
    """Split a row into segments at gaps clearly larger than the median
    center distance (e.g. a drawn tile sitting a bit apart)."""
    if len(blocks) <= 1:
        return [blocks]
    centers = [b[0] for b in blocks]
    diffs = np.diff(centers)
    med_d = float(np.median(diffs))
    tol = max(1.3 * med_d, 12.0)
    segs = [[blocks[0]]]
    for i in range(1, len(blocks)):
        if diffs[i - 1] > tol:
            segs.append([blocks[i]])
        else:
            segs[-1].append(blocks[i])
    return segs


def _merge_blocks(bs):
    x0 = min(b[1] for b in bs)
    x1 = max(b[2] for b in bs)
    y0 = min(b[3] for b in bs)
    y1 = max(b[4] for b in bs)
    return (x0, x1, y0, y1)


def _median_step(main, blocks):
    """Median center-to-center step inside the main segment."""
    diffs = [main[i][0] - main[i - 1][0] for i in range(1, len(main))]
    if diffs:
        return float(np.median(diffs))
    if len(blocks) >= 2:
        centers = [b[0] for b in blocks]
        return float(np.median(np.diff(centers)))
    return 0.0


def _merge_right_neighbor(main, segs, blocks):
    """Merge the segment right next to the main row when it is a drawn tile
    (a short segment sitting a bit further right). The total count must stay
    inside the valid set, otherwise the segment is treated as a side set
    (chi/peng) and left out.
    """
    idx = segs.index(main)
    if idx + 1 >= len(segs):
        return main
    right = segs[idx + 1]
    total = len(main) + len(right)
    if total not in VALID_TILE_COUNTS or len(right) > 2:
        return main
    med_d = _median_step(main, blocks)
    gap = right[0][0] - main[-1][0]
    if gap > max(2.0 * med_d, med_d + 40):
        return main
    return sorted(main + right, key=lambda b: b[0])


def _unify_boxes(boxes, region_w, region_h):
    """Normalize every box to the same median size, keeping each box center."""
    if not boxes:
        return []
    widths = [x1 - x0 for x0, x1, _, _ in boxes]
    heights = [y1 - y0 for _, _, y0, y1 in boxes]
    w = max(int(round(np.median(widths))), 10)
    h = max(int(round(np.median(heights))), 10)
    out = []
    for x0, x1, y0, y1 in boxes:
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        nx0 = cx - w // 2
        nx1 = nx0 + w
        ny0 = cy - h // 2
        ny1 = ny0 + h
        if nx0 < 0:
            nx0, nx1 = 0, w
        if nx1 > region_w:
            nx1 = region_w
            nx0 = max(0, nx1 - w)
        if ny0 < 0:
            ny0, ny1 = 0, h
        if ny1 > region_h:
            ny1 = region_h
            ny0 = max(0, ny1 - h)
        out.append((nx0, ny0, nx1 - nx0, ny1 - ny0))
    return out


def detect_tile_boxes(bgr):
    """Detect individual tile boxes in a BGR region.

    Returns (boxes, count) where boxes is a list of (x, y, w, h) relative
    to the region, or ([], 0) when nothing reliable is found. The returned
    boxes are the main row only (side sets such as chi/peng are filtered
    by row alignment and cluster size).
    """
    if bgr is None or bgr.size == 0:
        return [], 0
    region_h, region_w = bgr.shape[:2]
    mask = _white_mask(bgr)
    blocks = _same_row(_tile_blocks(mask))
    if not blocks:
        return [], 0
    segs = _segments(blocks)
    main = max(segs, key=len)
    chosen = list(main)
    if len(chosen) not in VALID_TILE_COUNTS:
        # 主段数量不合法时，尝试并入另一个段（如摸牌）凑成合法牌数
        for seg in segs:
            if seg is main:
                continue
            candidate = sorted(chosen + seg, key=lambda b: b[0])
            if len(candidate) in VALID_TILE_COUNTS:
                chosen = candidate
                break
    else:
        # 主段合法时仍尝试并入右侧紧邻的摸牌段（如 13 手牌 + 1 摸牌）
        chosen = _merge_right_neighbor(main, segs, blocks)
    boxes = [_merge_blocks([b]) for b in chosen]
    boxes = _unify_boxes(boxes, region_w, region_h)
    return boxes, len(boxes)
