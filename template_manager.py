# -*- coding: utf-8 -*-
"""模板管理 v2：彩色（BGR）模板 + 多样本平均构建，识别域统一。"""
import os
import cv2
import numpy as np

TEMPLATE_PATH = "templates"
CONFIG_PATH = "config.json"
TEMPLATE_SIZE = (141, 96)  # (高, 宽)，所有模板统一尺寸
COARSE_SIZE = (48, 32)     # 布局搜索用的粗粒度灰度尺寸（高, 宽）

# 同会话内每张牌的修正次数（用于模板滚动平均，重启后重置）
_sample_counts = {}
# 已提示过缺失的模板（每个进程只提示一次，避免反复刷屏）
_missing_warned = set()

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
ID_TO_SUIT_VAL[34] = (0, 5)                 # 红5万
ID_TO_SUIT_VAL[35] = (1, 5)                 # 红5索
ID_TO_SUIT_VAL[36] = (2, 5)                 # 红5筒
TOTAL_TILE_TYPES = 37
VAL_TO_ID = {v: k for k, v in ID_TO_SUIT_VAL.items() if k < 34}


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


def _align_image(new, ref, max_shift=3):
    """
    将 new 平移配准到 ref（相位相关，亚像素），返回对齐后的图像。
    对齐不可靠（偏移过大或对齐后相关性反而下降）时返回 None，由调用方决定。
    只修正轻微错位（默认 ±3px）：重复花纹（如 9p 多排圆点）相位相关可能
    锁定到错误周期产生大位移，超出范围一律视为不可靠。
    """
    if new is None or ref is None or new.shape != ref.shape:
        return None
    g_new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    try:
        (dx, dy), _ = cv2.phaseCorrelate(g_ref, g_new)
    except cv2.error:
        return None
    if abs(dx) > max_shift or abs(dy) > max_shift:
        return None

    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned = cv2.warpAffine(new, m, (new.shape[1], new.shape[0]),
                             borderMode=cv2.BORDER_REPLICATE)
    # 校验：对齐后与参考的相关性应不低于对齐前
    def corr(p, q):
        pa = p.astype(np.float32) - p.astype(np.float32).mean()
        qa = q.astype(np.float32) - q.astype(np.float32).mean()
        return float((pa * qa).sum() / (np.linalg.norm(pa) * np.linalg.norm(qa) + 1e-9))
    if corr(aligned, ref) < corr(new, ref) - 1e-6:
        return None
    return aligned


def _corr_bgr(a, b):
    """BGR 多通道归一化相关（等价 CCOEFF_NORMED 向量版本）。"""
    pa = a.astype(np.float32) - a.astype(np.float32).mean()
    qa = b.astype(np.float32) - b.astype(np.float32).mean()
    denom = np.linalg.norm(pa) * np.linalg.norm(qa)
    return float((pa * qa).sum() / (denom + 1e-9))


def _pick_representative(refs):
    """Pick one real sample as the class template instead of averaging.

    Samples are shift-aligned to the first one, then the sample closest to
    the per-pixel median image is chosen. This keeps the template sharp and
    never produces the blurry average the old pipeline suffered from.
    """
    if not refs:
        return None
    if len(refs) == 1:
        return refs[0]
    base = refs[0]
    aligned = [base]
    for s in refs[1:]:
        a = _align_image(s, base)
        aligned.append(a if a is not None else s)
    stack = np.stack(aligned).astype(np.float32)
    med = np.median(stack, axis=0)
    best, best_corr = None, -1.0
    for s in aligned:
        c = _corr_bgr(s, med)
        if c > best_corr:
            best_corr, best = c, s
    return best


def load_templates():
    """从 TEMPLATE_PATH 加载 34 张模板并预处理，构建匹配矩阵。"""
    templates = []
    for i in range(TOTAL_TILE_TYPES):
        path = os.path.join(TEMPLATE_PATH, f"{i}.png")
        img = None
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:  # 灰度图兼容
                g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if g is not None:
                    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if img is not None:
            # 模板文件保存时已预处理，加载不再二次处理，避免变亮/过度增强
            templates.append((i, _resize_to_template(img)))
        else:
            if i not in _missing_warned:
                print(f"模板缺失：{i}.png（当前为占位，等待采集）")
                _missing_warned.add(i)
            templates.append((i, np.full((TEMPLATE_SIZE[0], TEMPLATE_SIZE[1], 3), 255, np.uint8)))
    rebuild_template_matrix(templates)
    return templates


def load_samples():
    """
    加载每张牌的全部对齐样本（templates/samples/{tile_id}/*.png），
    构建示例矩阵供样本级匹配使用（kNN 式，取每类最高分）。
    """
    global SAMPLES, EXEMPLAR_MATRIX, EXEMPLAR_IDS
    SAMPLES = {}
    base = os.path.join(TEMPLATE_PATH, "samples")
    if not os.path.isdir(base):
        EXEMPLAR_MATRIX = None
        EXEMPLAR_IDS = []
        return SAMPLES
    for d in sorted(os.listdir(base)):
        if not d.isdigit():
            continue
        tid = int(d)
        paths = sorted(f for f in os.listdir(os.path.join(base, d)) if f.endswith(".png"))
        imgs = []
        for p in paths:
            img = cv2.imread(os.path.join(base, d, p), cv2.IMREAD_COLOR)
            if img is not None:
                # 样本文件保存时已预处理，加载不再二次处理
                imgs.append(_resize_to_template(img))
        if imgs:
            SAMPLES[tid] = imgs

    rows = []
    ids = []
    for tid in sorted(SAMPLES):
        for img in SAMPLES[tid]:
            vec = img.astype(np.float32).reshape(-1)
            vec = vec - vec.mean()
            n = np.linalg.norm(vec)
            rows.append(vec / n if n > 1e-6 else np.zeros_like(vec))
            ids.append(tid)
    EXEMPLAR_MATRIX = np.array(rows, dtype=np.float32) if rows else None
    EXEMPLAR_IDS = ids
    return SAMPLES


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
    rebuild_coarse_matrix(tpl_list)


def rebuild_coarse_matrix(tpl_list=None):
    """
    构建粗粒度灰度模板矩阵，用于布局搜索（只求快速确定分档数/偏移，
    最终判定仍用全彩样本级匹配）。
    """
    global COARSE_MATRIX
    if tpl_list is None:
        tpl_list = templates
    if not tpl_list:
        return
    h, w = COARSE_SIZE
    rows = []
    for _, tpl in tpl_list:
        gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        vec = small.astype(np.float32).reshape(-1)
        vec = vec - vec.mean()
        n = np.linalg.norm(vec)
        rows.append(vec / n if n > 1e-6 else np.zeros_like(vec))
    COARSE_MATRIX = np.array(rows, dtype=np.float32)


def update_template(tile_id, img):
    """
    在线修正/模板采集：用新样本预处理后保存。
    同一会话内同一张牌多次修正时做滚动平均（加权），避免单次样本覆盖导致模板不稳。
    """
    global templates
    path = os.path.join(TEMPLATE_PATH, f"{tile_id}.png")
    os.makedirs(TEMPLATE_PATH, exist_ok=True)
    new = _resize_to_template(_to_bgr(img))
    n = _sample_counts.get(tile_id, 0)
    old = cv2.imread(path, cv2.IMREAD_COLOR) if n > 0 and os.path.exists(path) else None
    if old is not None:
        old_is_gray = (np.max(old[..., 2]) - np.min(old[..., 0]) < 6 and
                       np.abs(old[..., 0].astype(int) - old[..., 1]).mean() < 2 and
                       np.abs(old[..., 1].astype(int) - old[..., 2]).mean() < 2)
        if old_is_gray:
            # 旧模板是灰度（历史版本采集）：直接替换为彩色，避免混色
            n = 0
        elif old.shape[:2] == new.shape[:2]:
            # 加权平均前先做平移配准，消除采集错位造成的重影
            aligned = _align_image(new, old)
            if aligned is not None:
                new = aligned
            avg = (old.astype(np.float32) * n + new.astype(np.float32)) / (n + 1)
            new = np.clip(avg, 0, 255).astype(np.uint8)
    _sample_counts[tile_id] = n + 1
    # imencode+tofile 代替 imwrite：OpenCV 的 imwrite 在中文路径下会静默失败
    ok, buf = cv2.imencode(".png", new)
    if ok:
        buf.tofile(path)
    templates = load_templates()
    return True


def build_template_library(origin_dir="templates_origin", samples_by_id=None, target_dir=None):
    """
    构建模板库：每张牌从 原始模板 + 全部标记样本 中挑选一张代表样本
    （不叠化平均，保持锐利细节），写入 target_dir。
    samples_by_id: {tile_id: [BGR ROI, ...]}
    """
    target_dir = target_dir or TEMPLATE_PATH
    os.makedirs(target_dir, exist_ok=True)
    built = []
    for i in range(TOTAL_TILE_TYPES):
        refs = []
        if origin_dir:
            op = os.path.join(origin_dir, f"{i}.png")
            if os.path.exists(op):
                img = cv2.imread(op, cv2.IMREAD_COLOR)
                if img is None:
                    g = cv2.imread(op, cv2.IMREAD_GRAYSCALE)
                    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if g is not None else None
                if img is not None:
                    refs.append(_resize_to_template(img))
        if samples_by_id and i in samples_by_id:
            for s in samples_by_id[i]:
                if s is not None and s.size > 0:
                    refs.append(_resize_to_template(_to_bgr(s)))
        if refs:
            best = _pick_representative(refs)
            # 同上：中文路径下 imwrite 会静默失败
            ok, buf = cv2.imencode(".png", best)
            if ok:
                buf.tofile(os.path.join(target_dir, f"{i}.png"))
            built.append((i, len(refs)))
    return built


def rebuild_templates_from_samples(target_dir=None):
    """
    From templates/samples/{tile_id}/*.png rebuild templates/{tile_id}.png
    without averaging. Returns [(tile_id, sample_count), ...].
    """
    target_dir = target_dir or TEMPLATE_PATH
    base = os.path.join(TEMPLATE_PATH, "samples")
    if not os.path.isdir(base):
        return []
    samples_by_id = {}
    for d in sorted(os.listdir(base)):
        if not d.isdigit():
            continue
        tid = int(d)
        paths = sorted(f for f in os.listdir(os.path.join(base, d)) if f.endswith(".png"))
        imgs = []
        for p in paths:
            img = cv2.imread(os.path.join(base, d, p), cv2.IMREAD_COLOR)
            if img is not None:
                imgs.append(_resize_to_template(img))
        if imgs:
            samples_by_id[tid] = imgs
    return build_template_library(origin_dir=None, samples_by_id=samples_by_id,
                                  target_dir=target_dir)


def get_tile_name(tile_id):
    if tile_id < 0:
        return "?"
    if tile_id == 34:
        return "红5w"
    if tile_id == 35:
        return "红5s"
    if tile_id == 36:
        return "红5p"
    suit, val = ID_TO_SUIT_VAL.get(tile_id, (3, 0))
    suit_map = {0: "w", 1: "s", 2: "p", 3: ""}
    if suit == 3:
        names = ["东", "南", "西", "北", "中", "发", "白"]
        return names[tile_id - 27] if 0 <= tile_id - 27 < 7 else "字"
    return f"{val}{suit_map[suit]}"


# 全局变量（供 vision.py 使用）
SAMPLES = {}
EXEMPLAR_MATRIX = None
EXEMPLAR_IDS = []
COARSE_MATRIX = None
templates = load_templates()
load_samples()
