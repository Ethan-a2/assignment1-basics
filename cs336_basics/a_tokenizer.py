from __future__ import annotations
import os
import re
import math
from typing import List, Tuple, Dict, Iterable, Iterator

# From our common.py
from tests.common import gpt2_bytes_to_unicode, get_gpt2_unicode_to_bytes

class Tokenizer:
    """
    一个基于字节对编码 (BPE) 算法的 Tokenizer 实现，模仿 GPT-2 的行为。

    主要功能包括：
    1. 将文本编码为整数 ID 列表 (`encode`)。
    2. 将整数 ID 列表解码回文本 (`decode`)。
    3. 支持特殊 token 的识别和处理。
    4. 提供迭代式编码，以处理大型文本文件并减少内存消耗 (`encode_iterable`)。
    """

    def __init__(
        self,
        id_to_token: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = None,
    ):
        """
        初始化 Tokenizer。

        Args:
            id_to_token: 映射 token ID 到其原始字节序列的字典。这是我们 BPE 算法操作的 "词汇表"。
                         例如：{0: b'hel', 1: b'lo', ...}
            merges: 学习到的 BPE 合并规则列表。每个规则是一个 (token1_bytes, token2_bytes) 对。
                    这些规则按优先级排序，列表前面的规则优先级更高。
                    例如：[(b'h', b'e'), (b'he', b'l'), ...]
            special_tokens: 字符串形式的特殊 token 列表，例如 ["<|endoftext|>"]。
                             这些 token 不参与 BPE 合并，而是直接被转换为其对应的 ID。
        """
        self.id_to_token = id_to_token
        # 反向映射：字节序列到 token ID，用于编码时查找
        self.token_to_id = {v: k for k, v in id_to_token.items()}

        # 将合并规则列表转换为字典，以便快速查找合并优先级(rank)
        # 列表中的索引越小，合并优先级越高
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}

        # 配置特殊 token
        self.special_token_to_id: Dict[bytes, int] = {}
        special_token_pattern_parts: List[str] = []

        if special_tokens:
            for st_str in special_tokens:
                st_bytes = st_str.encode("utf-8")
                # 检查特殊 token 是否在 id_to_token 词汇表中
                if st_bytes in self.token_to_id:
                    self.special_token_to_id[st_bytes] = self.token_to_id[st_bytes]
                else:
                    # 如果不在，则为其分配一个新的 token ID，并添加到词汇表
                    new_id = len(self.id_to_token)
                    self.id_to_token[new_id] = st_bytes
                    self.token_to_id[st_bytes] = new_id
                    self.special_token_to_id[st_bytes] = new_id

                # 为正则表达式转义特殊 token 字符串，以正确匹配
                special_token_pattern_parts.append(re.escape(st_str))

        # 编译一个正则表达式，用于在文本中查找并分割特殊 token
        # 使用 `|` 连接所有特殊 token 模式，并将其放入一个捕获组中 `()`，
        # 这样 `re.split` 就可以返回匹配到的特殊 token 本身。
        self.special_token_pattern = None
        if special_token_pattern_parts:
            # 确保长特殊 token 优先匹配
            special_token_pattern_parts.sort(key=len, reverse=True)
            self.special_token_pattern = re.compile(
                "(" + "|".join(special_token_pattern_parts) + ")"
            )

        # GPT-2 的字节编码器 (bytes -> unicode char)
        # BPE 算法本身操作的是 bytes，但 GPT-2 的词汇表文件 (`vocab.json`)
        # 内部是用特殊的 Unicode 字符表示的字节，以便 JSON 存储。
        # 在我们的实现中，`id_to_token` 字典已经将这些“字符”解码成了原始字节。
        # 所以，在 encode 阶段，我们直接将输入文本 UTF-8 编码为 bytes 即可。

        # Debugging: 打印初始化信息
        if os.environ.get("DEBUG_TOKENIZER_INIT"):
            print("--- Tokenizer Init Debug ---")
            print(f"Vocab size: {len(self.id_to_token)}")
            print(f"Merges count: {len(self.merges_rank)}")
            print(f"Special tokens recognized: {self.special_token_to_id}")
            if self.special_token_pattern:
                print(f"Special token regex: {self.special_token_pattern.pattern}")
            print("----------------------------")


    def _apply_bpe_merges(self, piece: bytes) -> List[bytes]:
        """
        对给定的字节序列应用 BPE 合并规则，直到无法再合并为止。
        这是 BPE 算法的核心部分。

        Args:
            piece: 要进行 BPE 合并的字节序列 (不包含特殊 token)。

        Returns:
            合并后的字节序列列表，每个元素都是一个有效的 token 字节序列。
        """
        # 如果 piece 为空，直接返回空列表
        if not piece:
            return []

        # 初始分割：将字节序列拆分成单个字节的列表
        # 例如：b"hello" -> [b"h", b"e", b"l", b"l", b"o"]
        segments = [bytes([b]) for b in piece]

        # Debugging: 打印初始分割
        if os.environ.get("DEBUG_BPE_MERGES"):
            print(f"  Initial segments: {[s.decode('utf-8', errors='replace') for s in segments]}")

        while True:
            best_pair_idx = -1
            best_pair_rank = math.inf # 使用无穷大表示最高优先级 (最低 rank)

            # 遍历当前所有相邻的字节对，寻找优先级最高的合并规则
            for i in range(len(segments) - 1):
                current_pair = (segments[i], segments[i+1])
                # 从 merges_rank 字典中查找该字节对的优先级 (rank)
                # 如果找不到，说明这个对无法合并，rank 设为无穷大
                rank = self.merges_rank.get(current_pair, math.inf)

                # 寻找优先级最高的 (rank 最小的) 字节对
                if rank < best_pair_rank:
                    best_pair_rank = rank
                    best_pair_idx = i

            # 如果没有找到任何可以合并的字节对，则退出循环
            if best_pair_rank == math.inf:
                break

            # 执行合并操作
            merged_segment = segments[best_pair_idx] + segments[best_pair_idx+1]

            # 更新 segments 列表：用合并后的新段替换原来的两个段
            # 例如：[..., b'he', b'l', ...] -> [..., b'hel', ...]
            segments[best_pair_idx:best_pair_idx+2] = [merged_segment]

            # Debugging: 打印每次合并后的状态
            if os.environ.get("DEBUG_BPE_MERGES"):
                print(f"  Merged pair: ({segments[best_pair_idx].decode('utf-8', errors='replace')}, "
                      f"{segments[best_pair_idx+1].decode('utf-8', errors='replace')}) "
                      f"-> {merged_segment.decode('utf-8', errors='replace')}")
                print(f"  Segments after merge: {[s.decode('utf-8', errors='replace') for s in segments]}")

        return segments

    def encode(self, text: str) -> List[int]:
        """
        将输入文本编码为 token ID 列表。

        Args:
            text: 待编码的文本字符串。

        Returns:
            编码后的 token ID 列表。
        """
        encoded_ids: List[int] = []

        # Debugging: 打印原始文本
        if os.environ.get("DEBUG_ENCODE"):
            print(f"\n--- Encoding Debug: '{text}' ---")

        if self.special_token_pattern:
            # 使用正则表达式分割文本，同时捕获特殊 token
            # parts 列表中将交替出现普通文本和特殊 token 字符串
            parts = self.special_token_pattern.split(text)
            # Debugging: 打印分割结果
            if os.environ.get("DEBUG_ENCODE"):
                print(f"  Split by special tokens: {parts}")

            for part in parts:
                if not part: # 跳过空字符串部分 (re.split 在开头/结尾匹配时可能产生空字符串)
                    continue

                part_bytes = part.encode("utf-8")
                if part_bytes in self.special_token_to_id:
                    # 如果当前部分是已知的特殊 token
                    encoded_ids.append(self.special_token_to_id[part_bytes])
                    if os.environ.get("DEBUG_ENCODE"):
                        print(f"  -> Special token: '{part}' (ID: {self.special_token_to_id[part_bytes]})")
                else:
                    # 否则，对普通文本部分进行 BPE 合并
                    bpe_segments = self._apply_bpe_merges(part_bytes)
                    for segment in bpe_segments:
                        # 查找合并结果在词汇表中的 ID
                        if segment not in self.token_to_id:
                            # 这通常不应该发生，除非词汇表/合并规则不完整
                            # 对于 GPT-2 词汇表，所有单独的字节都会在词汇表中
                            # 且所有合并结果也都会对应一个 ID。
                            raise ValueError(
                                f"BPE segment '{segment.decode('utf-8', errors='replace')}' "
                                f"not found in vocabulary. Text: '{text}'"
                            )
                        encoded_ids.append(self.token_to_id[segment])
                    if os.environ.get("DEBUG_ENCODE"):
                        print(f"  -> Regular text BPE: '{part}' -> {[s.decode('utf-8', errors='replace') for s in bpe_segments]} "
                              f"(IDs: {[self.token_to_id[s] for s in bpe_segments]})")
        else:
            # 如果没有特殊 token，直接对整个文本进行 BPE 合并
            bpe_segments = self._apply_bpe_merges(text.encode("utf-8"))
            for segment in bpe_segments:
                encoded_ids.append(self.token_to_id[segment])
            if os.environ.get("DEBUG_ENCODE"):
                print(f"  -> Full text BPE: '{text}' -> {[s.decode('utf-8', errors='replace') for s in bpe_segments]} "
                      f"(IDs: {[self.token_to_id[s] for s in bpe_segments]})")

        if os.environ.get("DEBUG_ENCODE"):
            print(f"--- Full Encoded IDs: {encoded_ids} ---")
        return encoded_ids

    def decode(self, ids: List[int]) -> str:
        """
        将 token ID 列表解码回文本字符串。

        Args:
            ids: 待解码的 token ID 列表。

        Returns:
            解码后的文本字符串。
        """
        # Debugging: 打印原始 ID 列表
        if os.environ.get("DEBUG_DECODE"):
            print(f"\n--- Decoding Debug: {ids} ---")

        bpe_bytes = bytearray()
        for _id in ids:
            # 查找 ID 对应的字节序列
            token_bytes = self.id_to_token[_id]
            bpe_bytes.extend(token_bytes)
            if os.environ.get("DEBUG_DECODE"):
                print(f"  -> ID: {_id}, Bytes: '{token_bytes.decode('utf-8', errors='replace')}'")

        # 将所有字节序列连接起来，然后使用 UTF-8 解码成字符串
        decoded_string = bpe_bytes.decode("utf-8", errors="replace")
        if os.environ.get("DEBUG_DECODE"):
            print(f"--- Full Decoded String: '{decoded_string}' ---")
        return decoded_string

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        以迭代方式编码输入，适用于处理非常大的文件，以减少整体内存占用。

        此实现会先将迭代器中的所有数据读取并拼接成一个完整字符串，
        然后调用 `encode` 方法。
        之所以这样做，是为了确保 BPE 合并和特殊 token 识别能够跨越原始的迭代块边界，
        从而与 `encode(entire_content)` 的行为保持一致（这是 `tiktoken`
        在面对整个文件时的行为，也是测试所期望的正确性标准）。

        请注意：如果 `iterable` 提供了极端大的单行/块，或文件总大小超出可用内存，
        此方法仍可能导致内存溢出。一个真正内存高效的流式 BPE 需更复杂的
        内部状态管理和前瞻逻辑。但对于 GPT-2 BPE 的精确匹配，
        全缓冲通常是简单且可靠的方式。

        Args:
            iterable: 一个可迭代对象，产生文本块 (例如文件句柄的行)。

        Yields:
            编码后的 token ID。
        """
        # Debugging: 打印 encode_iterable 启动信息
        if os.environ.get("DEBUG_ENCODE_ITERABLE"):
            print("\n--- Encoding Iterable Debug ---")

        # 将所有迭代器内容聚集到一个列表中，再拼接
        # 这是为了确保 BPE 算法能看到完整的文本，从而产生与 tiktoken 相同的 token 序列
        # 牺牲了一定的内存效率，但保证了与参考实现的兼容性。
        full_text_parts: List[str] = []
        for chunk in iterable:
             full_text_parts.append(chunk)

        full_text = "".join(full_text_parts)

        # 此时，我们已经有了完整的文本，可以调用正常的 encode 方法
        # 然后将结果逐个 yield
        yield from self.encode(full_text)

        if os.environ.get("DEBUG_ENCODE_ITERABLE"):
            print("--- Encoding Iterable Finished ---")


def get_tokenizer(
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    special_tokens: List[str] | None = None,
) -> Tokenizer:
    """
    一个工厂函数，用于创建和返回 Tokenizer 实例。

    Args:
        vocab: 映射 token ID 到其原始字节序列的字典。
        merges: 学习到的 BPE 合并规则列表。
        special_tokens: 字符串形式的特殊 token 列表。

    Returns:
        一个配置好的 Tokenizer 实例。
    """
    return Tokenizer(vocab, merges, special_tokens)

