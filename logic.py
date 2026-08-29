# -*- coding: utf-8 -*-
"""日麻切牌建议 v2：向听数 + 有效进张。

- 向听数：普通形（4 面子 + 1 雀头）、七对子、国士无双三路取最小；
- 有效进张：摸到后向听数下降的牌（听牌时即和牌张），按 4 - 手牌持有 计
  剩余张数（暂不扣牌河，副露/牌河留作后续增量）；
- 红5（宝牌）：牌型上等价于普通 5，但推荐切牌时在向听/进张相同的前提下
  优先保留红5。
"""

from functools import lru_cache

ID_TO_NAME = {
    0: '1w', 1: '2w', 2: '3w', 3: '4w', 4: '5w', 5: '6w', 6: '7w', 7: '8w', 8: '9w',
    9: '1s', 10: '2s', 11: '3s', 12: '4s', 13: '5s', 14: '6s', 15: '7s', 16: '8s', 17: '9s',
    18: '1p', 19: '2p', 20: '3p', 21: '4p', 22: '5p', 23: '6p', 24: '7p', 25: '8p', 26: '9p',
    27: 'e', 28: 's', 29: 'w', 30: 'n', 31: 'z', 32: 'f', 33: 'b',
    34: '红5w', 35: '红5s', 36: '红5p',
}

# 红5 对应的普通 5 位置
RED5_TO_BASE = {34: 4, 35: 13, 36: 22}


def _to_counts(tile_ids):
    """把 37 类 id 转成 34 类普通计数；返回 (counts, red5_ids)。"""
    counts = [0] * 34
    red5 = []
    for t in tile_ids:
        if t < 0:
            continue
        if t in RED5_TO_BASE:
            counts[RED5_TO_BASE[t]] += 1
            red5.append(t)
        else:
            counts[t] += 1
    return tuple(counts), red5


# ---------- 单花色拆牌：求最大 (面子, 搭子) ----------

@lru_cache(maxsize=65536)
def _suit_mt(c):
    """c: 9 长度计数（不含字牌）。返回 Pareto 最优的 (m, t) 集合。"""
    if sum(c) == 0:
        return ((0, 0),)

    # 找到第一张非空位置
    i = next(i for i, x in enumerate(c) if x > 0)
    results = []
    c = list(c)

    # 刻子
    if c[i] >= 3:
        c[i] -= 3
        results.extend((m + 1, t) for m, t in _suit_mt(tuple(c)))
        c[i] += 3
    # 顺子
    if i + 2 < 9 and c[i] > 0 and c[i + 1] > 0 and c[i + 2] > 0:
        c[i] -= 1
        c[i + 1] -= 1
        c[i + 2] -= 1
        results.extend((m + 1, t) for m, t in _suit_mt(tuple(c)))
        c[i] += 1
        c[i + 1] += 1
        c[i + 2] += 1
    # 对子（搭子）
    if c[i] >= 2:
        c[i] -= 2
        results.extend((m, t + 1) for m, t in _suit_mt(tuple(c)))
        c[i] += 2
    # 相邻搭子
    if i + 1 < 9 and c[i] > 0 and c[i + 1] > 0:
        c[i] -= 1
        c[i + 1] -= 1
        results.extend((m, t + 1) for m, t in _suit_mt(tuple(c)))
        c[i] += 1
        c[i + 1] += 1
    # 间张搭子
    if i + 2 < 9 and c[i] > 0 and c[i + 2] > 0:
        c[i] -= 1
        c[i + 2] -= 1
        results.extend((m, t + 1) for m, t in _suit_mt(tuple(c)))
        c[i] += 1
        c[i + 2] += 1
    # 单张
    c[i] -= 1
    results.extend((m, t) for m, t in _suit_mt(tuple(c)))

    # Pareto 最优：m 越大越好；m 相同 t 越大越好
    best = {}
    for m, t in results:
        if m not in best or t > best[m]:
            best[m] = t
    out = []
    max_t = -1
    for m in sorted(best, reverse=True):
        if best[m] > max_t:
            out.append((m, best[m]))
            max_t = best[m]
    return tuple(out)


def _honor_mt(c):
    """字牌：刻子 / 对子 / 单张。返回 (m, t) 单组最优。"""
    m = t = 0
    for x in c:
        m += x // 3
        r = x % 3
        t += r // 2
    return m, t


@lru_cache(maxsize=65536)
def _max_mt(counts):
    """34 计数 → (m, t)：全部花色合并后的 Pareto 最优。"""
    results = [(0, 0)]
    # 万 / 索 / 筒
    for start in (0, 9, 18):
        c = counts[start:start + 9]
        nxt = []
        for m, t in results:
            for sm, st in _suit_mt(c):
                nxt.append((m + sm, t + st))
        results = nxt
    # 字牌
    hm, ht = _honor_mt(counts[27:34])
    results = [(m + hm, t + ht) for m, t in results]
    best = {}
    for m, t in results:
        if m not in best or t > best[m]:
            best[m] = t
    out = []
    max_t = -1
    for m in sorted(best, reverse=True):
        if best[m] > max_t:
            out.append((m, best[m]))
            max_t = best[m]
    return tuple(out)


def _shanten_normal(counts):
    """普通形向听：枚举雀头 + 无雀头分支。"""
    best = 8
    for i in range(34):
        if counts[i] >= 2:
            c = list(counts)
            c[i] -= 2
            c = tuple(c)
            for m, t in _max_mt(c):
                if m + t > 4:
                    t = 4 - m
                s = 8 - 2 * m - t - 1
                if s < best:
                    best = s
    for m, t in _max_mt(counts):
        if m + t > 4:
            t = 4 - m
        s = 8 - 2 * m - t
        if s < best:
            best = s
    return best


def _shanten_chiitoi(counts):
    """七对子向听：6 - 对子数 + max(0, 7 - 种类数)（13 张）。

    种类不足 7 种（存在多张刻子牌型）时单靠对子数会低估向听，
    需补种类数修正项，否则可能给出偏乐观的七对建议。
    """
    pairs = sum(1 for x in counts if x >= 2)
    kinds = sum(1 for x in counts if x >= 1)
    return 6 - pairs + max(0, 7 - kinds)


def _shanten_kokushi(counts):
    """国士无双向听。"""
    yaochu = [0, 8, 9, 17, 18, 26] + list(range(27, 34))
    kinds = sum(1 for i in yaochu if counts[i] > 0)
    has_pair = any(counts[i] >= 2 for i in yaochu)
    return 13 - kinds - (1 if has_pair else 0)


@lru_cache(maxsize=65536)
def calculate_shanten(counts):
    """34 计数（13 张）→ 最小向听数。14 张和牌形返回 -1。"""
    total = sum(counts)
    if total == 14:
        # 和牌判定：4 面子 + 1 雀头
        for i in range(34):
            if counts[i] >= 2:
                c = list(counts)
                c[i] -= 2
                for m, t in _max_mt(tuple(c)):
                    if m == 4 and t == 0:
                        return -1
    if total not in (13, 14):
        return 99  # 副露等非常规牌数，暂不支持
    return min(_shanten_normal(counts), _shanten_chiitoi(counts),
               _shanten_kokushi(counts))


def _tile_name(tid):
    return ID_TO_NAME.get(tid, "?")


def effective_tiles(counts):
    """13 张 → 有效进张列表 [(普通牌id, 剩余张数)]，摸到后向听下降（听牌即和牌）。"""
    s = calculate_shanten(counts)
    if s == 99:
        return [], 0
    out = []
    for i in range(34):
        if counts[i] >= 4:
            continue
        c = list(counts)
        c[i] += 1
        if calculate_shanten(tuple(c)) < s:
            out.append((i, 4 - counts[i]))
    return out, sum(n for _, n in out)


def _score_discard(counts_after, red5_discarded):
    """打分：向听越低越好，同向听进张越多越好，红5 保留优先。"""
    s = calculate_shanten(counts_after)
    if s == 99:
        return None
    _, effective = effective_tiles(counts_after)
    return (s, -effective, 1 if red5_discarded else 0)


def decide_discard(tile_ids):
    """给出切牌建议。

    tile_ids 长度 14：推荐打出一张；长度 13：返回当前向听信息（不打牌）。
    返回 (best_idx, info)，info 为 dict：
      shanten / effective / effective_tiles / mode / red5
    best_idx 为 -1 表示不推荐打出（13 张或副露等非常规牌数）。
    """
    counts, red5 = _to_counts(tile_ids)
    n = len(tile_ids)

    if n == 14:
        best = None
        best_idx = -1
        for i, t in enumerate(tile_ids):
            if t < 0:
                continue
            c = list(counts)
            if t in RED5_TO_BASE:
                base = RED5_TO_BASE[t]
                c[base] -= 1
                red5_discarded = True
            else:
                c[t] -= 1
                red5_discarded = False
            sc = _score_discard(tuple(c), red5_discarded)
            if sc is None:
                continue
            if best is None or sc < best:
                best = sc
                best_idx = i
        if best_idx < 0:
            return -1, None
        s, neg_eff, _ = best
        c_after = list(counts)
        t = tile_ids[best_idx]
        if t in RED5_TO_BASE:
            c_after[RED5_TO_BASE[t]] -= 1
        else:
            c_after[t] -= 1
        eff_list, eff_total = effective_tiles(tuple(c_after))
        return best_idx, {
            'shanten': s,
            'effective': eff_total,
            'effective_tiles': [(_tile_name(i), rem) for i, rem in eff_list],
            'mode': 'discard',
            'red5': bool(red5),
        }

    if n == 13:
        s = calculate_shanten(counts)
        if s == 99:
            return -1, None
        eff_list, eff_total = effective_tiles(counts)
        return -1, {
            'shanten': s,
            'effective': eff_total,
            'effective_tiles': [(_tile_name(i), rem) for i, rem in eff_list],
            'mode': 'wait',
            'red5': bool(red5),
        }

    return -1, None
