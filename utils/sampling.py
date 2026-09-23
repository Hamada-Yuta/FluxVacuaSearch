import numpy as np
from typing import List, Tuple, Iterator, Optional
import argparse


def sample_m_n(max_sum_of_squares: int, num_samples: int, k: int, flux_range: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    フラックスパラメータ m, n をランダムにサンプリングします。

    Parameters:
        max_sum_of_squares (int): 2乗和の最大値
        num_samples (int): サンプル数
        k (int): フラックスパラメータの数

    Returns:
        List[Tuple[Tuple[int, ...], Tuple[int, ...]]]: サンプリングされた (m, n) のリスト
    """
    samples = set()
    while len(samples) < num_samples:
        m = np.random.randint(-flux_range, flux_range + 1, size=k)
        m[0] = np.random.randint(0, flux_range + 1)  # m1 (m[0]) を非負に設定

        n = np.random.randint(-flux_range, flux_range + 1, size=k)
        n[0] = np.random.randint(0, flux_range + 1)  # n1 (n[0]) を非負に設定

        if np.sum(m**2 + n**2) <= max_sum_of_squares:
            samples.add((tuple(m), tuple(n)))
    return list(samples)


# ===== 決定的順序 =====
def _range_first_nonneg(R: int):
    return range(0, R + 1)

def _range_symmetric(R: int):
    return range(-R, R + 1)

# ===== DP 構築: 残り r 成分で二乗和 s になる通り数 (各成分は対称領域) =====
def build_dp_rest(k: int, R: int, max_sum: int) -> List[List[int]]:
    # squares for symmetric coord: 0^2 (mult=1), 1^2..R^2 (mult=2)
    squares = [i * i for i in range(R + 1)]
    mults = [1] + [2] * R
    dp = [[0] * (max_sum + 1) for _ in range(k + 1)]
    dp[0][0] = 1
    for r in range(1, k + 1):
        prev = dp[r - 1]
        cur = dp[r]
        for s, m in zip(squares, mults):
            upper = max_sum - s
            if upper < 0:
                continue
            pv = prev
            cv = cur
            for t in range(upper + 1):
                if pv[t]:
                    cv[t + s] += pv[t] * m
    return dp  # dp[r][S]

# n 側の「最初の成分だけ非負」な分布 cf[B] = 長さk で二乗和 B の通り数
def build_cf_first_nonneg(k: int, R: int, max_sum: int, dp_rest: List[List[int]]) -> List[int]:
    # first coord: 0..R（重みは常に1）
    cf = [0] * (max_sum + 1)
    dp_tail = dp_rest[k - 1] if k >= 1 else [1] + [0] * max_sum
    for x0 in range(0, R + 1):
        s0 = x0 * x0
        rem = max_sum - s0
        if rem < 0:
            break
        tail = dp_tail
        c = cf
        for t in range(rem + 1):
            v = tail[t]
            if v:
                c[t + s0] += v
    return cf  # cf[B]

# 畳み込み: (a * b)[T] = sum_s a[s] * b[T - s]
def convolve_limited(a: List[int], b: List[int], limit: int) -> List[int]:
    out = [0] * (limit + 1)
    for s, av in enumerate(a):
        if not av:
            continue
        up = limit - s
        if up < 0:
            break
        bv = b
        ov = out
        for t in range(up + 1):
            bt = bv[t]
            if bt:
                ov[s + t] += av * bt
    return out

# ユーティリティ：前方和
def _prefixify(arr: List[int]) -> List[int]:
    out = arr[:]
    s = 0
    for i, v in enumerate(out):
        s += v
        out[i] = s
    return out

# ---- 1) 総件数：単調増加になる正しい定義（≤） ----
def total_pairs_count(k: int, R: int, max_sum: int) -> int:
    dp_rest = build_dp_rest(k, R, max_sum)                            # 既存
    cf_n = build_cf_first_nonneg(k, R, max_sum, dp_rest)              # 既存
    # 残り (k-1) の m と全 n の畳み込み（＝ rem ちょうど）を作り、
    # それを prefix 化して “≤ rem” の個数にする
    conv = convolve_limited(dp_rest[k - 1] if k >= 1 else [1] + [0] * max_sum,
                            cf_n, max_sum)
    conv_prefix = _prefixify(conv)
    total = 0
    for m0 in _range_first_nonneg(R):
        rem = max_sum - m0 * m0
        if rem < 0:
            break
        total += conv_prefix[rem]
    return total

# ---- 2) n の部分列挙：ブロックサイズは prefix を使う ----
def emit_n_range(k: int, R: int, rem: int, dp_rest: List[List[int]],
                 offset: int, take: int) -> Tuple[List[Tuple[int, ...]], int]:
    out: List[Tuple[int, ...]] = []
    cur = [0] * k
    dp_rest_prefix = [_prefixify(row) for row in dp_rest]

    def count_suffix_le(j: int, rem_now: int) -> int:
        r = k - j - 1  # 残り成分数
        if r < 0:
            return 1 if rem_now >= 0 else 0
        if rem_now < 0:
            return 0
        return dp_rest_prefix[r][rem_now]  # “≤ rem_now”

    def dfs(j: int, rem_now: int, off: int, need: int) -> Tuple[int, int]:
        if need == 0:
            return off, 0
        if j == k:
            if off > 0:
                return off - 1, 0
            out.append(tuple(cur))
            return 0, 1
        rng = _range_first_nonneg(R) if j == 0 else _range_symmetric(R)
        emitted = 0
        for x in rng:
            r2 = rem_now - x * x
            if r2 < 0:
                if j == 0:
                    break  # ここは単調
                continue     # 対称レンジなので続行
            block = count_suffix_le(j, r2)  # ここが prefix
            if off >= block:
                off -= block
                continue
            cur[j] = x
            off, got = dfs(j + 1, r2, off, need - emitted)
            emitted += got
            if emitted >= need:
                break
        return off, emitted

    _, got = dfs(0, rem, offset, take)
    return out, got

# ---- 3) (m, n) の部分列挙：入口 i=0、対称レンジは break→continue、leaf は prefix ----
def iter_pairs_range(k: int, R: int, max_sum: int,
                     start: int, limit: int):
    dp_rest = build_dp_rest(k, R, max_sum)
    cf_n = build_cf_first_nonneg(k, R, max_sum, dp_rest)

    # prefix（≤）で扱う
    def _prefixify(arr):
        out, s = arr[:], 0
        for i, v in enumerate(out):
            s += v
            out[i] = s
        return out

    cf_n_prefix = _prefixify(cf_n)
    pairs_prefix = []
    for r in range(k + 1):
        conv = convolve_limited(dp_rest[r], cf_n, max_sum)
        pairs_prefix.append(_prefixify(conv))

    cur_m = [0] * k
    produced = 0  # これまでに yield した件数

    def dfs_m(i: int, sumsq: int, off: int, need: int):
        nonlocal produced
        if need == 0:
            return off
        if i == k:
            rem = max_sum - sumsq
            if rem < 0:
                return off
            total_n = cf_n_prefix[rem]  # n の総数（≤ rem）
            if off >= total_n:
                return off - total_n
            take = min(need, total_n - off)
            n_chunk, got = emit_n_range(k, R, rem, dp_rest, offset=off, take=take)
            for n in n_chunk:
                yield (tuple(cur_m), n)
            produced += got
            return 0

        rng = range(0, R + 1) if i == 0 else range(-R, R + 1)
        base = produced  # このノードに入った時点の出力量を記録
        for x in rng:
            # このノード内で必要数を使い切ったら打ち切り
            if produced - base >= need:
                break

            s = sumsq + x * x
            if s > max_sum:
                if i == 0:
                    break   # 先頭（非負レンジ）は単調なのでbreak
                else:
                    continue  # 対称レンジはまだ続けられる

            r_rem = k - (i + 1)
            rem = max_sum - s
            block = pairs_prefix[r_rem][rem]  # この枝全体の (m,n) 件数（≤）
            if off >= block:
                off -= block
                continue

            cur_m[i] = x
            remain = need - (produced - base)  # この枝に割ける残り必要数
            off = yield from dfs_m(i + 1, s, off, remain)

        return off

    yield from dfs_m(0, 0, start, limit)

def ordered_sample_m_n(max_sum_of_squares: int, num_samples: int, sample_seed: int, k: int, flux_range: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    フラックスパラメータ m, n を決定的順序でサンプリングします。

    Parameters:
        max_sum_of_squares (int): 2乗和の最大値
        num_samples (int): サンプル数
        sample_seed (int): サンプルの開始位置
        k (int): フラックスパラメータの数
        flux_range (int): フラックスパラメータの最大値

    Returns:
        List[Tuple[Tuple[int, ...], Tuple[int, ...]]]: サンプリングされた (m, n) のリスト
    """
    samples = set()
    for m, n in iter_pairs_range(k, flux_range, max_sum_of_squares, sample_seed, num_samples):
        samples.add((m,n))
    return list(samples)


# ===== CLI =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--R", type=int, required=True, help="flux_range")
    ap.add_argument("--max_sum", type=int, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--count", action="store_true", help="総件数だけ表示して終了")
    args = ap.parse_args()

    if args.count:
        total = total_pairs_count(args.k, args.R, args.max_sum)
        print(total)
        return

    for m, n in iter_pairs_range(args.k, args.R, args.max_sum, args.start, args.limit):
        print(f"{m}\t{n}")

if __name__ == "__main__":
    main()
