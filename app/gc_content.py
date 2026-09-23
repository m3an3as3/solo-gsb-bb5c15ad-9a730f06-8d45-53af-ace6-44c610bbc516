"""固定窗口 GC 含量范围计算（只读）。

计算口径
--------
* 序列仅由 IUPAC 碱基组成。对每个碱基按其**可能代表的碱基集合**确定
  GC 贡献的最小/最大值：
  - 最小贡献 = 1 当且仅当该符号的所有可能碱基都属于 {G, C}（确定的
    G/C 与仅含 G/C 的简并符号，如 S=G/C）；其余为 0；
  - 最大贡献 = 1 当且仅当该符号的可能碱基与 {G, C} 有交集（即"可能"
    是 G 或 C，如 R=A/G、N=A/C/G/T）；仅含 A/T 的符号为 0；
  - 确定碱基（A/C/G/T）对上下界贡献相同，因此二者的差全部来自简并
    符号的不确定性。
* 窗口的 GC 百分比 = GC 计数 / 窗口长度 × 100；最小、最大百分比分别
  由最小、最大 GC 计数除以同一个窗口长度得到。
* 窗口长度 W 固定，必须满足 1 <= W <= L（序列长度）；步长 S >= 1，
  缺省取 W（不重叠）。

窗口切分
--------
* 线性序列：起点为 0, S, 2S, ...，只保留完全落在序列内的窗口
  （start + W <= L）；末尾不足一个完整窗口的尾随碱基不单独开窗，
  也不补长。
* 环形序列：窗口可跨越末尾回到开头（base 下标对 L 取模），起点同样
  取 0, S, 2S, ... 且 start < L，即完整一圈被 ceil(L / S) 个起点
  覆盖；W <= L 保证每个窗口最多覆盖一圈，同一物理碱基在任一窗口内
  至多被计入一次。W == L 时仅取起点 0（全长窗）。
* 环形窗口 start + W > L 时 wraps_origin 为 True；起点按升序返回，
  相同请求的结果必然一致。
* 整个计算只在内存中进行，不读写数据库。
"""

from __future__ import annotations

from .iupac import IUPAC_SETS

_GC = frozenset("GC")


def gc_contribution(base: str) -> tuple[int, int]:
    """单个 IUPAC 碱基对 GC 计数的（最小贡献, 最大贡献）。

    最小贡献为 1 当且仅当可能碱基集合是 {G,C} 的子集（至少一个元素
    且全部属于 G/C）；最大贡献为 1 当且仅当可能碱基集合与 {G,C}
    有交集。
    """
    possible = IUPAC_SETS[base]
    minimum = 1 if possible <= _GC else 0
    maximum = 1 if possible & _GC else 0
    return minimum, maximum


def _prefix_counts(sequence: str) -> tuple[list[int], list[int]]:
    """各碱基最小/最大 GC 贡献的前缀和（prefix[i] = 前 i 个碱基之和）。"""
    min_prefix = [0]
    max_prefix = [0]
    min_total = 0
    max_total = 0
    for base in sequence:
        d_min, d_max = gc_contribution(base)
        min_total += d_min
        max_total += d_max
        min_prefix.append(min_total)
        max_prefix.append(max_total)
    return min_prefix, max_prefix


def gc_content_windows(
    sequence: str, window_length: int, step: int, circular: bool
) -> list[dict]:
    """按固定窗口计算 GC 含量范围，返回按起点升序的窗口列表。

    调用方须已完成全部校验（序列非空且为 IUPAC、1 <= window_length
    <= len(sequence)、step >= 1）。环形序列的窗口 base 下标对序列长度
    取模，窗口长度不超过序列长度，因此不会重复计入同一物理碱基。

    每个窗口：{"index", "start", "window_length", "gc_min_count",
    "gc_max_count", "gc_min_percent", "gc_max_percent", "wraps_origin"}。
    """
    length = len(sequence)
    min_prefix, max_prefix = _prefix_counts(sequence)

    def counts(start: int) -> tuple[int, int]:
        """取长度 window_length、起点为 start 的窗口 GC 计数（自动处理跨原点）。"""
        end = start + window_length
        if end <= length:
            return (
                min_prefix[end] - min_prefix[start],
                max_prefix[end] - max_prefix[start],
            )
        # 环形窗口跨末尾：拆成 [start, L) 与 [0, end-L) 两段
        tail = end - length
        return (
            min_prefix[length] - min_prefix[start] + min_prefix[tail],
            max_prefix[length] - max_prefix[start] + max_prefix[tail],
        )

    if circular:
        # 起点 0, S, 2S, ... 且 start < L：恰好覆盖一整圈；
        # W == L 时只有起点 0（全长窗），其余起点与它覆盖同一碱基多重集
        last_start = 1 if window_length == length else length
    else:
        # 线性序列只保留完整窗口，末尾不足一窗的尾随碱基忽略
        last_start = length - window_length + 1

    windows = []
    for index, start in enumerate(range(0, last_start, step)):
        gc_min, gc_max = counts(start)
        windows.append(
            {
                "index": index,
                "start": start,
                "window_length": window_length,
                "gc_min_count": gc_min,
                "gc_max_count": gc_max,
                "gc_min_percent": gc_min / window_length * 100.0,
                "gc_max_percent": gc_max / window_length * 100.0,
                "wraps_origin": circular and start + window_length > length,
            }
        )
    return windows
