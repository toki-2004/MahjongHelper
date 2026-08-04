# logic.py
ID_TO_NAME = {
    0:'1w',1:'2w',2:'3w',3:'4w',4:'5w',5:'6w',6:'7w',7:'8w',8:'9w',
    9:'1s',10:'2s',11:'3s',12:'4s',13:'5s',14:'6s',15:'7s',16:'8s',17:'9s',
    18:'1p',19:'2p',20:'3p',21:'4p',22:'5p',23:'6p',24:'7p',25:'8p',26:'9p',
    27:'e',28:'s',29:'w',30:'n',31:'z',32:'f',33:'b'
}

def get_suit_val(id):
    if id <= 8: return (0, id)      # 万
    elif id <= 17: return (1, id-9)  # 索
    elif id <= 26: return (2, id-18) # 饼
    else: return (3, -1)             # 字牌

def _w_partial(v):
    # 搭子权重：1-2 边缘（0-based 为 1-2）权重低，其余为 3
    return 1 if v == 1 else 3

def _build_alloc(max_r=12):
    """预计算每个牌值上剩余 r4 张牌时的分配方案 (得分, 新顺子数, 搭子数, 间张数)。"""
    alloc = []
    for v in range(9):
        w = _w_partial(v)
        table = {}
        for r4 in range(max_r + 1):
            opts = []
            for trip in range(r4 // 3 + 1):
                r5 = r4 - 3 * trip
                for pr in range(r5 // 2 + 1):
                    r6 = r5 - 2 * pr
                    for s in range(r6 + 1):
                        r7 = r6 - s
                        for q in range(r7 + 1):
                            r8 = r7 - q
                            for z in range(r8 + 1):
                                opts.append((100 * trip + 10 * pr, s, q, z))
            table[r4] = opts
        alloc.append(table)
    return alloc

_ALLOC = _build_alloc()

def _score_suit(cnt):
    """
    单花色动态规划，等价于原 DFS 的最大不相交组合权重：
    状态 (a, b, p, g1, g2) 分别表示：
      a  - 已用 2 张、待当前牌完成的顺子
      b  - 已用 1 张、待当前和下一张的顺子
      p  - 已用 1 张、待当前牌完成的搭子
      g1 - 已用 1 张、待当前牌完成的间张
      g2 - 已用 1 张、待下一张完成的间张
    """
    dp = {(0, 0, 0, 0, 0): 0}
    for v in range(9):
        c = cnt[v]
        ndp = {}
        table = _ALLOC[v]
        w_prev = _w_partial(v - 1)
        for (a, b, p, g1, g2), sc in dp.items():
            for t_a in range(min(a, c) + 1):
                r = c - t_a
                base = sc + 100 * t_a
                for t_p in range(min(p, r) + 1):
                    r2 = r - t_p
                    base2 = base + w_prev * t_p
                    for t_g in range(min(g1, r2) + 1):
                        r3 = r2 - t_g
                        base3 = base2 + t_g
                        for t_b in range(min(b, r3) + 1):
                            r4 = r3 - t_b
                            nb = t_b
                            if r4 > 12:
                                r4 = 12
                            for gain, s, q, z in table[r4]:
                                key = (nb, s, q, g2, z)
                                nsc = base3 + gain
                                if nsc > ndp.get(key, -1):
                                    ndp[key] = nsc
        dp = ndp
    return max(dp.values())

def calculate_score(tile_ids):
    """按牌数统计后用 DP 求最大组合权重，避免原 DFS 的指数级搜索。"""
    suits = [[0] * 9 for _ in range(3)]
    honors = [0] * 7
    for t in tile_ids:
        if t <= 8:
            suits[0][t] += 1
        elif t <= 17:
            suits[1][t - 9] += 1
        elif t <= 26:
            suits[2][t - 18] += 1
        else:
            honors[t - 27] += 1

    total = 0
    for cnt in suits:
        total += _score_suit(cnt)
    for h in honors:
        best = 0
        for t in range(h // 3 + 1):
            r = h - 3 * t
            s = 100 * t + 10 * (r // 2)
            if s > best:
                best = s
        total += best
    return total

def decide_discard(tile_ids):
    best_score = -1
    best_idx = -1
    for i in range(len(tile_ids)):
        remaining = [tile_ids[j] for j in range(len(tile_ids)) if j != i]
        if len(remaining) < 2:
            continue
        score = calculate_score(remaining)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx, best_score
