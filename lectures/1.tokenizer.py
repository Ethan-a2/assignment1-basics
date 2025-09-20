import regex # 用于高级正则表达式操作，如GPT2_TOKENIZER_REGEX和word_tokenizer中的模式匹配
from dataclasses import dataclass # 用于创建数据类，如BPETokenizerParams
from collections import defaultdict # 用于在BPE训练中方便地计数词对
import tiktoken # 用于获取OpenAI的GPT-2分词器实例

# --- 模拟教育/演示框架的辅助函数和类 ---
# 这些函数和类在原始代码中未定义，但根据其使用方式，它们可能是某个交互式学习环境的一部分。
# 在这里，我们提供简单的实现，以便代码可以运行并聚焦于核心逻辑的注释。

def text(s: str):
    """模拟在演示文稿/笔记本中显示文本的函数。"""
    print(s)

def link(*args, **kwargs):
    """模拟创建可点击链接的函数。
    它接收一个标题和/或一个URL，并返回一个表示链接的字符串。
    """
    if 'title' in kwargs and 'url' in kwargs:
        return f"[{kwargs['title']}]({kwargs['url']})"
    elif 'url' in kwargs:
        return f"({kwargs['url']})"
    elif 'title' in kwargs:
        return f"[{kwargs['title']}]"
    return ""

def youtube_link(url: str):
    """模拟显示YouTube链接的函数。"""
    return link(url=url, title="YouTube Video")

def article_link(url: str):
    """模拟显示文章链接的函数。"""
    return link(url=url, title="Article")

class Tokenizer:
    """分词器的抽象基类。
    定义了所有分词器必须实现的标准接口：
    将字符串编码为整数令牌列表（encode）和将令牌列表解码回字符串（decode）。
    """
    def encode(self, string: str) -> list[int]:
        """将输入字符串编码为整数令牌列表。"""
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        """将整数令牌列表解码回字符串。"""
        raise NotImplementedError

# 外部引用的占位符，如果它们是演示框架的一部分
sennrich_2016 = dict(title="Neural Machine Translation of Rare Words with Subword Units", url="https://arxiv.org/abs/1508.07909")
gpt2 = dict(title="Language Models are Unsupervised Multitask Learners", url="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf")

# --- 核心代码开始 ---

# GPT-2分词器使用的正则表达式模式。
# 这个正则表达式定义了如何将文本预分割成初始片段，然后再进行BPE合并。
# 它是从OpenAI的tiktoken库中获取的，用于匹配标点符号、数字、字母、空格等。
# 例如：
# - `'(?:[sdmt]|ll|ve|re)` 匹配常见的英文缩写，如's, 'd, 'm, 't, 'll, 've, 're。
# - ` ?\p{L}+` 匹配一个可选的前导空格和一连串的Unicode字母。
# - ` ?\p{N}+` 匹配一个可选的前导空格和一连串的Unicode数字。
# - ` ?[^\s\p{L}\p{N}]+` 匹配一个可选的前导空格和除了空白、字母、数字之外的任意非空白字符。
# - `\s+(?!\S)|\s+` 匹配一个或多个空格，用于处理多余的空格或只包含空格的片段。
GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def tokenization():
    """主函数，组织并展示不同类型的分词器及其工作原理。
    它通过调用各种子函数来逐步介绍分词概念和示例。
    """
    # 介绍分词的背景和灵感来源
    text("This unit was inspired by Andrej Karpathy's video on tokenization; check it out! "), youtube_link("https://www.youtube.com/watch?v=zduSFxRajkE")

    # 逐步介绍不同的分词方法
    intro_to_tokenization()     # 分词器简介
    tokenization_examples()     # 分词器示例（使用GPT-2分词器）
    character_tokenizer()       # 基于字符的分词器
    byte_tokenizer()            # 基于字节的分词器
    word_tokenizer()            # 基于单词的分词器
    bpe_tokenizer()             # 字节对编码（BPE）分词器

    # 总结分词的主要概念和观察
    text("## Summary")
    text("- Tokenizer: strings <-> tokens (indices)")
    text("- Character-based, byte-based, word-based tokenization highly suboptimal")
    text("- BPE is an effective heuristic that looks at corpus statistics")
    text("- Tokenization is a necessary evil, maybe one day we'll just do it from bytes...")

@dataclass(frozen=True)
class BPETokenizerParams:
    """BPETokenizer的所有必要参数。
    这是一个数据类，用于封装BPE分词器所需的词汇表和合并规则，
    并且由于 `frozen=True`，它的实例是不可变的。
    """
    vocab: dict[int, bytes]     # 词汇表：将整数令牌映射到其对应的字节序列
                                # 例如：{0: b'h', 1: b'e', 256: b'he'}
    merges: dict[tuple[int, int], int]  # 合并规则：将一对相邻令牌映射到它们合并后的新令牌的索引
                                        # 例如：{(0, 1): 256} 表示令牌0和1合并成令牌256


class CharacterTokenizer(Tokenizer):
    """基于字符的分词器。
    将字符串表示为Unicode码点的序列。每个字符都被编码为一个整数。
    """
    def encode(self, string: str) -> list[int]:
        """将字符串编码为Unicode码点的整数列表。"""
        # 使用 `ord()` 函数获取每个字符的Unicode码点
        return list(map(ord, string))

    def decode(self, indices: list[int]) -> str:
        """根据Unicode码点列表解码回字符串。"""
        # 使用 `chr()` 函数将每个码点转换回字符，然后拼接成字符串
        return "".join(map(chr, indices))


class ByteTokenizer(Tokenizer):
    """基于字节的分词器。
    将字符串表示为UTF-8字节序列，每个字节被编码为一个0-255之间的整数。
    """
    def encode(self, string: str) -> list[int]:
        """将字符串编码为UTF-8字节序列的整数列表。"""
        # 首先将字符串编码为UTF-8字节序列
        string_bytes = string.encode("utf-8")  # @inspect string_bytes (可能是一个调试/检查点标记)
        # 然后将每个字节（0-255）转换为整数
        indices = list(map(int, string_bytes))  # @inspect indices
        return indices

    def decode(self, indices: list[int]) -> str:
        """根据整数（字节）列表解码回字符串。"""
        # 首先将整数列表转换回字节序列
        string_bytes = bytes(indices)  # @inspect string_bytes
        # 然后将字节序列解码为UTF-8字符串
        string = string_bytes.decode("utf-8")  # @inspect string
        return string


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """
    将 `indices` 列表中所有出现的 `pair` 替换为 `new_index`。
    这是BPE算法中核心的合并操作。
    """
    new_indices = []  # @inspect new_indices # 用于存储合并后的新列表
    i = 0  # @inspect i # 当前在 `indices` 列表中的位置
    while i < len(indices):
        # 检查当前位置和下一个位置的令牌是否与 `pair` 匹配
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            # 如果匹配，则将 `new_index` 添加到 `new_indices`
            new_indices.append(new_index)
            i += 2  # 跳过两个已合并的令牌
        else:
            # 如果不匹配，则将当前令牌添加到 `new_indices`
            new_indices.append(indices[i])
            i += 1  # 移动到下一个令牌
    return new_indices


class BPETokenizer(Tokenizer):
    """基于字节对编码（BPE）的分词器，使用给定的合并规则和词汇表。"""
    def __init__(self, params: BPETokenizerParams):
        """
        初始化BPE分词器。
        Args:
            params: 包含词汇表和合并规则的BPETokenizerParams对象。
        """
        self.params = params

    def encode(self, string: str) -> list[int]:
        """
        将字符串编码为BPE令牌的列表。
        这个实现是教育性质的，为了清晰而非效率，它会遍历所有的合并规则。
        实际高效的BPE编码器会使用更复杂的数据结构来优化查找和合并。
        """
        # 初始令牌列表是字符串的UTF-8字节序列，每个字节作为一个令牌
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # 按照合并规则的定义顺序（通常是训练时发现的顺序）逐一应用合并
        # 注意：这是一个非常慢的实现，因为它遍历了所有的合并规则，即使它们可能不出现在 `indices` 中
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index) # 应用合并操作
        return indices

    def decode(self, indices: list[int]) -> str:
        """
        将BPE令牌列表解码回字符串。
        """
        # 遍历令牌列表，从词汇表中查找每个令牌对应的字节序列
        # `self.params.vocab.get` 会安全地获取值，如果令牌不在词汇表中则返回None（虽然理论上不应该发生）
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        # 将所有字节序列连接起来形成一个完整的字节序列，然后解码为UTF-8字符串
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string


def get_compression_ratio(string: str, indices: list[int]) -> float:
    """
    计算给定字符串被分词后的压缩比。
    压缩比定义为原始字符串的字节数除以分词后的令牌数。
    更高的压缩比意味着更少的令牌可以表示相同的信息量。
    """
    # 计算原始字符串的UTF-8字节数
    num_bytes = len(bytes(string, encoding="utf-8"))  # @inspect num_bytes
    # 计算分词后的令牌数量
    num_tokens = len(indices)                       # @inspect num_tokens
    # 避免除零错误
    if num_tokens == 0:
        return float('inf') if num_bytes > 0 else 0.0
    return num_bytes / num_tokens


def get_gpt2_tokenizer():
    """
    获取OpenAI tiktoken库中的GPT-2分词器实例。
    gpt2编码方案是OpenAI的早期模型（如GPT-2、GPT-3）使用的。
    对于较新的模型（如gpt-3.5-turbo或gpt-4），通常使用"cl100k_base"。
    """
    # 代码源：https://github.com/openai/tiktoken
    # 你可以使用 cl100k_base 对应 gpt3.5-turbo 或 gpt4 分词器
    return tiktoken.get_encoding("gpt2")


def intro_to_tokenization():
    """介绍分词的基本概念，包括文本表示、令牌和编码/解码过程。"""
    text("原始文本通常表示为Unicode字符串。")
    string = "Hello, �! 你好!"
    text(f"例如：`{string}`")

    text("语言模型通常对令牌序列（通常用整数索引表示）上的概率分布进行建模。")
    indices = [15496, 11, 995, 0] # 这是一个示例令牌序列
    text(f"例如：`{indices}` 可能是 \"Hello, �!\" 的令牌表示。")

    text("因此，我们需要一个将字符串 *编码* 为令牌的程序。")
    text("我们还需要一个将令牌 *解码* 回字符串的程序。")
    text("一个 "), link(Tokenizer), text(" 是一个实现编码（encode）和解码（decode）方法的类。")
    text("**词汇表大小（Vocabulary size）**是可能令牌（整数）的数量。")


def tokenization_examples():
    """展示GPT-2分词器的工作示例，并观察其特征。"""
    text("为了感受分词器是如何工作的，请尝试这个 "), link(title="交互式网站", url="https://tiktokenizer.vercel.app/?encoder=gpt2")

    text("## 观察")
    text("- 一个单词及其前面的空格通常是同一个令牌的一部分（例如，\" world\"）。")
    text("- 一个单词在句子开头和句子中间的表示可能不同（例如，\"hello hello\" 中第一个\"hello\"和第二个\" hello\"）。")
    text("- 数字通常被分词为每几个数字一组。")

    text("这是OpenAI的GPT-2分词器（tiktoken）的实际应用。")
    tokenizer = get_gpt2_tokenizer()
    string = "Hello, �! 你好!"  # @inspect string # 示例字符串

    text("检查 `encode()` 和 `decode()` 是否能够往返转换，即 `string == decode(encode(string))`：")
    indices = tokenizer.encode(string)  # @inspect indices # 编码字符串
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string # 解码回字符串
    assert string == reconstructed_string # 确保原始字符串和重建的字符串相同
    print(f"原始字符串: '{string}'")
    print(f"编码令牌: {indices}")
    print(f"解码字符串: '{reconstructed_string}'")

    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio # 计算压缩比
    print(f"压缩比 (字节/令牌): {compression_ratio:.2f}")


def character_tokenizer():
    """介绍基于字符的分词器，并讨论其优缺点。"""
    text("## 基于字符的分词")

    text("Unicode字符串是Unicode字符的序列。")
    text("每个字符都可以通过 `ord` 函数转换为一个码点（整数）。")
    assert ord("a") == 97
    # assert ord("�") == 127757
    text("它可以通过 `chr` 函数转换回来。")
    assert chr(97) == "a"
    # assert chr(127757) == "�"

    text("现在我们构建一个 `CharacterTokenizer` 并确保它能往返转换：")
    tokenizer = CharacterTokenizer()
    string = "Hello, �! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string
    print(f"原始字符串: '{string}'")
    print(f"字符编码令牌: {indices}")
    print(f"解码字符串: '{reconstructed_string}'")

    text("大约有15万个Unicode字符。"), link(title="[维基百科]", url="https://en.wikipedia.org/wiki/List_of_Unicode_characters")
    # 这里的词汇表大小是基于示例`indices`中最大码点+1的下界估算，实际可能远大于此
    vocabulary_size = max(indices) + 1  # 这是一个下界估值 @inspect vocabulary_size
    print(f"估算的词汇表大小 (基于示例): {vocabulary_size}")
    text("问题1：这是一个非常大的词汇表。")
    text("问题2：许多字符非常罕见（例如 �），这导致词汇表的使用效率低下。")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio
    print(f"压缩比 (字节/令牌): {compression_ratio:.2f}")


def byte_tokenizer():
    """介绍基于字节的分词器，并讨论其优缺点。"""
    text("## 基于字节的分词")

    text("Unicode字符串可以表示为字节序列，字节可以用0到255之间的整数表示。")
    text("最常见的Unicode编码是 "), link(title="UTF-8", url="https://en.wikipedia.org/wiki/UTF-8")

    text("有些Unicode字符由一个字节表示：")
    assert bytes("a", encoding="utf-8") == b"a"
    text("其他字符需要多个字节：")
    # assert bytes("�", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"

    text("现在我们构建一个 `ByteTokenizer` 并确保它能往返转换：")
    tokenizer = ByteTokenizer()
    string = "Hello, �! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string
    print(f"原始字符串: '{string}'")
    print(f"字节编码令牌: {indices}")
    print(f"解码字符串: '{reconstructed_string}'")

    text("词汇表很小，很理想：一个字节可以表示256个值。")
    vocabulary_size = 256  # @inspect vocabulary_size
    print(f"词汇表大小: {vocabulary_size}")
    text("那压缩率如何？")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio
    assert compression_ratio == 1 # 每个字节一个令牌，所以字节数/令牌数=1
    print(f"压缩比 (字节/令牌): {compression_ratio:.2f}")
    text("压缩比非常糟糕（1:1），这意味着令牌序列会太长。")
    text("鉴于Transformer的上下文长度是有限的（因为自注意力是二次复杂度），这看起来不太好...")


def word_tokenizer():
    """介绍基于单词的分词器，并讨论其挑战。"""
    text("## 基于单词的分词")

    text("另一种方法（更接近于NLP经典做法）是将字符串分割成单词。")
    string = "I'll say supercalifragilisticexpialidocious!"
    print(f"原始字符串: '{string}'")

    # 使用简单的正则表达式分割单词和标点符号
    segments = regex.findall(r"\w+|.", string)  # `\w+` 匹配一个或多个字母数字字符，`.` 匹配其他任意字符 (除了换行)
    print(f"简单分割片段: {segments}") # @inspect segments
    text("这个正则表达式将所有字母数字字符（单词）保持在一起。")

    text("这是一个更复杂的版本（类似于GPT-2的预分词模式）：")
    pattern = GPT2_TOKENIZER_REGEX  # @inspect pattern
    segments = regex.findall(pattern, string)  # @inspect segments
    print(f"GPT-2式分割片段: {segments}")

    text("要将其转换为 `Tokenizer`，我们需要将这些片段映射为整数。")
    text("然后，我们可以为一个预定义的单词集合建立一个从单词到整数的映射。")

    text("但存在一些问题：")
    text("- 单词的数量巨大（类似于Unicode字符）。")
    text("- 许多单词很罕见，模型无法学习到太多关于它们的信息。")
    text("- 这无法自然地提供一个固定的词汇表大小（新词不断出现）。")

    text("在训练期间未见过的新词会被赋予一个特殊的 UNK（未知）令牌，这既不美观，也可能影响困惑度计算。")

    vocabulary_size = "训练数据中不同片段的数量（非常大且不固定）"
    # 对于单词分词，压缩比的计算方式与字节或字符不同，这里只是一个概念性的展示
    # 这里的 `segments` 已经是分词后的“令牌”，所以压缩比实际上是原始字符串字节数/单词数
    num_bytes = len(string.encode('utf-8'))
    num_tokens = len(segments)
    if num_tokens > 0:
        compression_ratio = num_bytes / num_tokens # @inspect compression_ratio
    else:
        compression_ratio = float('inf')
    print(f"估算的词汇表大小: {vocabulary_size}")
    print(f"压缩比 (字节/令牌): {compression_ratio:.2f}")


def bpe_tokenizer():
    """介绍字节对编码（BPE）分词器的工作原理、训练过程和使用方法。"""
    text("## 字节对编码 (BPE)")
    link(title="[维基百科]", url="https://en.wikipedia.org/wiki/Byte_pair_encoding")
    text("BPE算法由Philip Gage于1994年为数据压缩而引入。"), article_link("http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM")
    text("它被改编用于NLP，特别是在神经网络机器翻译中。"), link(sennrich_2016)
    text("(在此之前，论文一直使用基于单词的分词。)")
    text("BPE随后被GPT-2使用。"), link(gpt2)

    text("基本思想：*在原始文本上训练* 分词器，以自动确定词汇表。")
    text("直觉：常见的字符序列由单个令牌表示，罕见的序列由多个令牌表示。")

    text("GPT-2论文使用基于单词的分词将文本打散成初始片段，然后在每个片段上运行原始的BPE算法。")
    text("概述：从每个字节作为一个令牌开始，然后通过反复合并最常见的相邻令牌对来构建词汇表。")

    text("## 训练分词器")
    string = "the cat in the hat"  # @inspect string # 用于训练的示例字符串
    print(f"训练字符串: '{string}'")
    params = train_bpe(string, num_merges=3) # 训练BPE分词器，进行3次合并
    print("\n--- 训练完成 ---")
    print(f"最终词汇表: {params.vocab}")
    print(f"最终合并规则: {params.merges}")


    text("## 使用分词器")
    text("现在，给定一个新的文本，我们可以对其进行编码。")
    tokenizer = BPETokenizer(params) # 使用训练好的参数初始化BPE分词器
    string = "the quick brown fox"  # @inspect string # 用于编码的示例字符串
    print(f"待编码字符串: '{string}'")
    indices = tokenizer.encode(string)  # @inspect indices # 编码字符串
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string # 解码回字符串
    assert string == reconstructed_string # 确保往返转换成功
    print(f"编码令牌: {indices}")
    print(f"解码字符串: '{reconstructed_string}'")

    text("在作业1中，你将在此基础上进行以下改进：")
    text("- `encode()` 目前遍历所有合并规则。优化为只遍历需要进行的合并。")
    text("- 检测并保留特殊令牌（例如，`<|endoftext|>`）。")
    text("- 使用预分词（例如，GPT-2分词器正则表达式）。")
    text("- 尽量使实现尽可能快。")


def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    """
    实现BPE算法的训练过程。
    Args:
        string: 用于训练的原始文本字符串。
        num_merges: 要执行的合并操作的数量。
    Returns:
        BPETokenizerParams: 包含训练好的词汇表和合并规则的对象。
    """
    text("\n--- BPE 训练开始 ---")
    text("从 `string` 的字节列表开始作为初始令牌。")
    # 初始令牌列表是字符串的UTF-8字节表示，每个字节被视为一个令牌
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    print(f"初始令牌 (字节表示): {indices}")

    merges: dict[tuple[int, int], int] = {}  # 存储合并规则：(令牌1, 令牌2) -> 新合并令牌的索引
    # 初始词汇表包含所有单个字节（0-255），映射到它们自身的字节表示
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # 词汇表：索引 -> 字节序列
    print(f"初始词汇表大小: {len(vocab)}")

    for i in range(num_merges):
        print(f"\n--- 第 {i+1} 轮合并 ---")
        text("统计每个相邻令牌对的出现次数。")
        counts = defaultdict(int) # 使用defaultdict方便计数
        # 遍历 `indices` 列表，统计所有相邻的令牌对
        for index1, index2 in zip(indices, indices[1:]):  # 对于列表中的每个相邻对
            counts[(index1, index2)] += 1  # 增加该对的计数 # @inspect counts
        print(f"当前令牌序列: {indices}")
        print(f"当前词汇表示: {[vocab[idx].decode('utf-8', errors='ignore') for idx in indices]}") # 可视化当前令牌表示

        if not counts: # 如果没有可以合并的对，则停止
            print("没有更多的令牌对可以合并。")
            break

        text("找到出现次数最多的令牌对。")
        pair = max(counts, key=counts.get)  # 找到字典中值（计数）最大的键（令牌对） # @inspect pair
        index1, index2 = pair
        print(f"出现次数最多的令牌对: {pair} (计数: {counts[pair]})")

        text("合并该令牌对。")
        # 为新的合并令牌分配一个索引。从256开始，因为0-255已经被字节令牌使用。
        new_index = 256 + i  # @inspect new_index
        merges[pair] = new_index  # 将合并规则添加到 `merges` 字典 # @inspect merges
        # 更新词汇表：将新的合并令牌映射到它所代表的字节序列（两个原始令牌的字节序列拼接）
        vocab[new_index] = vocab[index1] + vocab[index2]  # @inspect vocab
        print(f"合并 {vocab[index1].decode('utf-8', errors='ignore')} + {vocab[index2].decode('utf-8', errors='ignore')} -> {vocab[new_index].decode('utf-8', errors='ignore')} (新索引: {new_index})")

        # 在当前的 `indices` 列表中执行合并操作
        indices = merge(indices, pair, new_index)  # @inspect indices
        print(f"合并后的令牌序列: {indices}")
        print(f"合并后的词汇表示: {[vocab[idx].decode('utf-8', errors='ignore') for idx in indices]}")


    return BPETokenizerParams(vocab=vocab, merges=merges)


def main():
    """程序的入口点。"""
    tokenization()

# 当脚本直接运行时，执行 `main` 函数
if __name__ == "__main__":
    main()
