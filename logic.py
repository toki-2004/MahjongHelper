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

def calculate_score(tile_ids):
    n = len(tile_ids)
    groups = []

    # 三张组合：刻子 / 顺子
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a,b,c = tile_ids[i], tile_ids[j], tile_ids[k]
                s1,v1 = get_suit_val(a)
                s2,v2 = get_suit_val(b)
                s3,v3 = get_suit_val(c)
                if a == b == c:
                    groups.append((100, [i,j,k]))
                elif s1 == s2 == s3 and s1 != 3:
                    vals = sorted([v1,v2,v3])
                    if vals[0]+1 == vals[1] and vals[1]+1 == vals[2]:
                        groups.append((100, [i,j,k]))

    # 两张组合：对子、相邻、间隔
    for i in range(n):
        for j in range(i+1, n):
            a,b = tile_ids[i], tile_ids[j]
            s1,v1 = get_suit_val(a)
            s2,v2 = get_suit_val(b)
            if a == b:
                groups.append((10, [i,j]))
            elif s1 == s2 and s1 != 3:
                diff = abs(v1-v2)
                if diff == 1:
                    # 边缘相邻 (1-2 或 8-9) 权重降低
                    if (v1 == 1 and v2 == 2) or (v1 == 2 and v2 == 1) or \
                       (v1 == 8 and v2 == 9) or (v1 == 9 and v2 == 8):
                        groups.append((1, [i,j]))
                    else:
                        groups.append((3, [i,j]))
                elif diff == 2:
                    groups.append((1, [i,j]))

    # DFS 求最大总权重
    best_score = 0
    def dfs(start_idx, used_mask, current_score):
        nonlocal best_score
        if current_score > best_score:
            best_score = current_score
        for idx in range(start_idx, len(groups)):
            weight, indices = groups[idx]
            conflict = False
            mask = 0
            for idx_in_group in indices:
                if used_mask & (1 << idx_in_group):
                    conflict = True
                    break
                mask |= (1 << idx_in_group)
            if not conflict:
                dfs(idx+1, used_mask | mask, current_score + weight)

    dfs(0, 0, 0)
    return best_score

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