#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[1]:


import regex as re
from collections import defaultdict
from tqdm.contrib.concurrent import process_map


# ## Step 1, chunk text by special tokens

# In[2]:


def split_by_special(text,
                     special_tokens, # list of special tokens, e.g., ["<|endoftext|>", "<|pad|>", "<|unk|>"]
                     drop_special=True, # whether or not drop the special tokens when using them to split
                     ):
    PAT = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special: PAT = f'({PAT})' # capturing group to keep special tokens
    return re.split(PAT, text)


# In[3]:


special_tokens = ["<|endoftext|>", "<|pad|>", "<|unk|>","<|endoftext|><|endoftext|>"]


# In[4]:


test = "<|pad|>abc<|pad|>"


# In[5]:


split_by_special(test,special_tokens,drop_special=False)


# need to solve the above as there's nothing in the edge

# In[6]:


def split_by_special(text, special_tokens, drop_special=True):
    if not special_tokens:  # if there's no special tokens, return the whole text
        return [text]

    PAT = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special: PAT = f"({PAT})"  # capture group to keep special tokens

    chunks = re.split(PAT, text)
    return [c for c in chunks if c]  # remove empty strings


# In[7]:


split_by_special(test,special_tokens,drop_special=False)


# In[8]:


test = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"


# In[9]:


split_by_special(test,special_tokens,drop_special=False)


# <|endoftext|> appear twice but not as a whole

# In[10]:


def split_by_special(text, special_tokens, drop_special=True):
    if not special_tokens:
        return [text]

    # Sort by descending length to prioritize longer tokens (e.g., "<|endoftext|><|endoftext|>" before "<|endoftext|>")
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    PAT = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special: PAT = f"({PAT})"

    chunks = re.split(PAT, text)
    return [c for c in chunks if c]  # remove empty strings


# In[11]:


split_by_special(test,special_tokens,drop_special=False)
# now appear as a whole


# In[12]:


test =  "Hello, world!<|pad|> Hello, world. <|endoftext|>How are you? <|unk|> I'm fine, thank you! And you?"


# In[13]:


chunks = split_by_special(test,special_tokens)
chunks


# In[14]:


chunk=chunks[0]
chunk


# ## Step 2, pre-tokenize: split chunk into word list by GPT2 pattern and get word counts

# In[15]:


def word2bytes(word):
    "Convert word string to tuple of bytes"
    a = list(word.encode('utf-8'))
    return tuple(bytes([i]) for i in a)


# In[16]:


word2bytes('hello')


# In[17]:


word2bytes('€')


# In[18]:


PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


# In[19]:


def count_word(text):
    "Split text into word bytes using GPT2 pattern and count word bytes frequency."
    word_cnt = defaultdict(int)
    for m in PAT.finditer(text):
        word = m.group(0)
        word_bytes = word2bytes(word)
        if len(word_bytes)>=2:
            word_cnt[word_bytes]+=1
    return word_cnt


# In[20]:


word_cnt = count_word(chunk)
word_cnt


# In[21]:


def merge_dicts(dicts):
    merged = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            merged[k] += v
    return merged


# In[22]:


merge_dicts([word_cnt,word_cnt])


# In[23]:


word_dicts =[count_word(chunk) for chunk in chunks]
word_cnt_all = merge_dicts(word_dicts)


# In[24]:


word_cnt_all


# In[25]:


# parallel
word_dicts = process_map(count_word, chunks,chunksize=1)


# In[26]:


word_cnt_all = merge_dicts(word_dicts)
word_cnt_all


# Note that some word only have single byte (len=1); no pair inside it

# ## Step 3. Get pair count based on word count

# In[27]:


def count_pair(word_cnt):
    pair_cnt = defaultdict(int)
    for word_bytes,cnt in word_cnt.items():
        for pair in zip(word_bytes[:-1],word_bytes[1:]):
            pair_cnt[pair]+=cnt
    return pair_cnt


# In[28]:


pair_cnt = count_pair(word_cnt_all)
pair_cnt


# ## Step 4. Get the max and merge

# In[29]:


def get_max_pair(pair_cnt): return max(pair_cnt.items(),key=lambda x: (x[1],x[0]))[0]


# In[30]:


max_pair = get_max_pair(pair_cnt)


# In[31]:


max_pair


# We need to add the max_pair to both vocab and merges

# In[32]:


def get_basic_vocab(special_tokens):
    vocab={token:bytes([token]) for token in range(256)}

    for i,token in enumerate(special_tokens):
        token_id = 256+i
        vocab[token_id] = token.encode("utf-8")
    return vocab


# In[33]:


special_tokens = ["<|endoftext|>", "<|pad|>", "<|unk|>"]


# In[34]:


vocab = get_basic_vocab(special_tokens)


# In[35]:


base_vocab_size = len(vocab)
base_vocab_size


# In[36]:


vocab_size=270


# In[37]:


n_merges=vocab_size-len(vocab)
n_merges


# In[38]:


def apply_merge(word_bytes,merge):
    merged = merge[0]+merge[1]
    i = 0
    new_word_bytes = []
    while i < len(word_bytes):
        # Check for match
        if i < len(word_bytes) - 1 and word_bytes[i] == merge[0] and word_bytes[i+1] == merge[1]:
            new_word_bytes.append(merged)
            i += 2
        else:
            new_word_bytes.append(word_bytes[i])
            i += 1
    return tuple(new_word_bytes)


# In[39]:


apply_merge(['a','b','c'],('b','c'))


# In[40]:


word_cnt


# In[41]:


def update_cnt(word_cnt,pair_cnt, merge_pair):

    new_word_cnt = defaultdict(int)
    new_pair_cnt = defaultdict(int, pair_cnt) # copy with defaultdict

    for word_bytes,cnt in word_cnt.items():

        #----------for word cnt ---------------

        old_pairs = list(zip(word_bytes[:-1], word_bytes[1:]))

        # Keep the original count if the merge not appear in the key
        if merge_pair not in old_pairs:
            new_word_cnt[word_bytes]+=cnt
            continue

        # Use updated key if merge appear
        new_word = apply_merge(word_bytes,merge_pair)
        new_word_cnt[new_word]+=cnt

        #--------for pair cnt ----------------

        # Decrease all old pair counts
        for pair in old_pairs:
            new_pair_cnt[pair]-=cnt
            if new_pair_cnt[pair] ==0:
                del new_pair_cnt[pair]

        # Count new pairs in the new word
        new_pairs = list(zip(new_word[:-1], new_word[1:]))
        for p in new_pairs:
            new_pair_cnt[p] += cnt

    return new_word_cnt,new_pair_cnt


# In[42]:


word_cnt_new, pair_cnt_new = update_cnt(word_cnt_all,pair_cnt,max_pair)


# In[43]:


word_cnt_new


# In[44]:


pair_cnt_new


# ## Pipeline

# In[45]:


test


# In[46]:


chunks = split_by_special(test,special_tokens)
word_dicts = process_map(count_word, chunks,chunksize=1)

word_cnt = merge_dicts(word_dicts)
pair_cnt = count_pair(word_cnt)

vocab = get_basic_vocab(special_tokens)
base_vocab_size = len(vocab)
vocab_size=265
n_merges=vocab_size-base_vocab_size


# In[47]:


merges = []
for i in range(n_merges):
    max_pair = get_max_pair(pair_cnt)
    vocab[base_vocab_size+i] = max_pair[0]+max_pair[1]
    merges.append(max_pair)
    word_cnt, pair_cnt = update_cnt(word_cnt,pair_cnt,max_pair)


# In[48]:


merges


# In[49]:


list(vocab.items())[base_vocab_size:]


# Wrap them up in functions

# In[50]:


def read_text(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


# In[51]:


def train_bpe(input_path,vocab_size,special_tokens):
    # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    text = read_text(input_path)
    chunks = split_by_special(text,special_tokens)
    word_dicts = process_map(count_word, chunks,chunksize=1)

    word_cnt = merge_dicts(word_dicts)
    pair_cnt = count_pair(word_cnt)

    vocab = get_basic_vocab(special_tokens)
    base_vocab_size = len(vocab)
    n_merges=vocab_size-base_vocab_size

    merges = []
    for i in range(n_merges):
        max_pair = get_max_pair(pair_cnt)
        vocab[base_vocab_size+i] = max_pair[0]+max_pair[1]
        merges.append(max_pair)
        word_cnt, pair_cnt = update_cnt(word_cnt,pair_cnt,max_pair)
    return vocab, merges


# Put it in adapters.py and run `uv run pytest -k test_train_bpe`

# In[52]:


get_ipython().run_cell_magic('time', '', 'vocab,merges = train_bpe(input_path="corpus.en",\n                            vocab_size=1000,\n                            special_tokens=["<|endoftext|>"])\n')


# In[53]:


get_ipython().run_cell_magic('time', '', 'vocab,merges = train_bpe(input_path="corpus.en",\n                            vocab_size=1000,\n                            special_tokens=["<|endoftext|>"])\n')


# In[54]:


def build_occ_tables(word_cnt):
    """
    word_occ[word_id]   = (word_bytes, freq)
    pair_occ[pair]      = {word_id1, word_id2, ...}
    pair_freq[pair]     = 累计频数
    """
    word_occ = {}
    pair_occ = defaultdict(set)
    pair_freq = defaultdict(int)

    for wid, (wbytes, freq) in enumerate(word_cnt.items()):
        word_occ[wid] = (wbytes, freq)
        if len(wbytes) >= 2:
            for pair in zip(wbytes[:-1], wbytes[1:]):
                pair_occ[pair].add(wid)
                pair_freq[pair] += freq
    return word_occ, pair_occ, pair_freq


# In[55]:


word_occ, pair_occ, pair_freq = build_occ_tables(word_cnt)


# In[56]:


pair_freq


# In[57]:


pair_occ


# In[58]:


word_occ


# In[59]:


import regex as re
import heapq
from collections import defaultdict
from tqdm.contrib.concurrent import process_map

PAT_string = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = re.compile(PAT_string)

# ---------- 基础工具 ----------

def read_text(input_path: str) -> str:
    with open(input_path, "r", encoding="utf-8") as f:
        return f.read()

def split_by_special(text, special_tokens, drop_special=True):
    if not special_tokens:
        return [text]
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    pattern = f"({pattern})" if not drop_special else pattern
    return [c for c in re.split(pattern, text) if c]

def word2bytes(word: str):
    return tuple(bytes([b]) for b in word.encode("utf-8"))

def count_word(text: str):
    cnt = defaultdict(int)
    for m in PAT.finditer(text):
        cnt[word2bytes(m.group(0))] += 1
    return cnt

def merge_dicts(dicts):
    merged = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            merged[k] += v
    return merged

def get_basic_vocab(special_tokens):
    vocab = {tid: bytes([tid]) for tid in range(256)}
    for i, tok in enumerate(special_tokens, start=256):
        vocab[i] = tok.encode("utf-8")
    return vocab

def apply_merge(word_bytes, merge_pair):
    merged_sym = merge_pair[0] + merge_pair[1]
    out = []
    i = 0
    while i < len(word_bytes):
        if i + 1 < len(word_bytes) and word_bytes[i:i+2] == list(merge_pair):
            out.append(merged_sym)
            i += 2
        else:
            out.append(word_bytes[i])
            i += 1
    return tuple(out)

# ---------- 新增：occ 表 + 堆 ----------

def build_occ_tables(word_cnt):
    """
    word_occ[word_id]   = (word_bytes, freq)
    pair_occ[pair]      = {word_id1, word_id2, ...}
    pair_freq[pair]     = 累计频数
    """
    word_occ = {}
    pair_occ = defaultdict(set)
    pair_freq = defaultdict(int)

    for wid, (wbytes, freq) in enumerate(word_cnt.items()):
        word_occ[wid] = (wbytes, freq)
        if len(wbytes) >= 2:
            for pair in zip(wbytes[:-1], wbytes[1:]):
                pair_occ[pair].add(wid)
                pair_freq[pair] += freq
    return word_occ, pair_occ, pair_freq

def build_heap(pair_freq):
    heap = [(-freq, pair) for pair, freq in pair_freq.items()]
    heapq.heapify(heap)
    return heap

def get_max_pair_lazy(heap, pair_freq):
    """
    弹出堆顶，若与真实 freq 不符说明过期 -> 丢弃，继续。
    """
    while heap:
        neg_freq, pair = heap[0]
        if pair_freq[pair] != -neg_freq:
            heapq.heappop(heap)      # 过期
            continue
        return pair
    return None  # 所有 pair 都被合并完

def update_after_merge(word_occ, pair_occ, pair_freq, heap, merge_pair):
    """
    只更新受 merge_pair 影响的 pretoken
    """
    affected = list(pair_occ[merge_pair])
    pair_occ[merge_pair].clear()

    # 真实频数用不到了，不再减；lazy 过期策略会自动淘汰旧值
    for wid in affected:
        wbytes, freq = word_occ[wid]

        # 1) 先把旧 word 的所有 pair 在 occ 表里注销
        if len(wbytes) >= 2:
            for p in zip(wbytes[:-1], wbytes[1:]):
                if wid in pair_occ[p]:
                    pair_occ[p].discard(wid)

        # 2) 生成新 word，并重新登记
        new_wbytes = apply_merge(wbytes, merge_pair)
        word_occ[wid] = (new_wbytes, freq)

        if len(new_wbytes) >= 2:
            for p in zip(new_wbytes[:-1], new_wbytes[1:]):
                pair_occ[p].add(wid)
                pair_freq[p] += freq          # 只加不减
                heapq.heappush(heap, (-pair_freq[p], p))

# ---------- 训练主函数 ----------

def train_bpe(input_path, vocab_size, special_tokens=()):
    text = read_text(input_path)
    chunks = split_by_special(text, special_tokens)

    # 计数可以并行
    word_dicts = (
        process_map(count_word, chunks, chunksize=1)
        if len(chunks) >= 4
        else list(map(count_word, chunks))
    )
    word_cnt = merge_dicts(word_dicts)

    # === 初始化 occ & heap ===
    word_occ, pair_occ, pair_freq = build_occ_tables(word_cnt)
    heap = build_heap(pair_freq)

    # === 初始化 vocab ===
    vocab = get_basic_vocab(special_tokens)
    base = len(vocab)
    merges_needed = vocab_size - base
    merges = []

    for i in range(merges_needed):
        best_pair = get_max_pair_lazy(heap, pair_freq)
        if best_pair is None:
            break  # 提前终止：再也找不到可合并的 pair
        merges.append(best_pair)
        vocab[base + i] = best_pair[0] + best_pair[1]

        update_after_merge(word_occ, pair_occ, pair_freq, heap, best_pair)

    return vocab, merges


# ## Encode

# In[60]:


special_tokens


# In[61]:


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# In[62]:


chunks = split_by_special(test,special_tokens,drop_special=False)
chunks


# In[63]:


chunk=chunks[0]


# In[64]:


def split_to_words(text):
    "Split text into words."
    return re.findall(PAT,text)


# In[65]:


word_list = split_to_words(chunk)
word_list


# In[66]:


vocab_to_id = {v:k for k,v in vocab.items()}


# In[67]:


def apply_merges(word_bytes, merges, vocab_to_id):
    "Apply merges based on minimum vocab token id."
    while True:
        pairs = list(zip(word_bytes[:-1], word_bytes[1:]))

        # Collect valid merge candidates with their vocab ID
        candidates = {}
        for pair in pairs:
            if pair in merges:
                merged = pair[0] + pair[1]
                token_id = vocab_to_id.get(merged)
                if token_id is not None:
                    candidates[pair] = token_id

        if not candidates:
            break  # no more mergeable pairs

        # Choose the pair with the **smallest token ID**
        best_pair = min(candidates.items(), key=lambda x: x[1])[0]

        word_bytes = apply_merge(word_bytes, best_pair)

    return word_bytes


# In[68]:


word_bytes=word2bytes(' world')
word_bytes


# In[ ]:


merged_word_bytes = apply_merges(word_bytes,merges,vocab_to_id)
merged_word_bytes


# In[ ]:


def encode_merged(text,merges,vocab_to_id):
    word_list = split_to_words(text)
    tokens=[]
    for word in word_list:
        word_bytes=word2bytes(word)
        merged_word_bytes = apply_merges(word_bytes,merges,vocab_to_id)
        tokens+=[vocab_to_id[i] for i in merged_word_bytes]
    return tokens


# In[ ]:


encode_merged(chunk,merges,vocab_to_id)


# In[ ]:


chunks = split_by_special(test,special_tokens,drop_special=False)
tokens =[]
for chunk in chunks:
    if chunk in special_tokens:
        tokens+=[vocab_to_id[chunk.encode('utf-8')]]
    else:
        tokens+=encode_merged(chunk,merges,vocab_to_id)


# In[ ]:


print(tokens)


# ## Decode

# In[ ]:


def decode(tokens,vocab): return b''.join([vocab[t] for t in tokens]).decode('utf-8',errors='replace')


# In[ ]:


decode(tokens,vocab)


# In[ ]:


# if not indicate errors=replace, will throw an error if decode([128],vocab) as 128 unicode is 10... not start bytes


# In[ ]:


decode([128],vocab) # this time will throw a question mark


# ## Class tokenizer

# In[ ]:


from typing import Iterator, Iterable
import json


# In[ ]:


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [i.encode('utf-8') for i in self.special_tokens]


        self.vocab_to_id={v:k for k,v in vocab.items()}

        # Ensure special tokens are in the vocabulary
        for token_bytes in self.special_tokens_bytes:
            if token_bytes not in self.vocab_to_id:
                # Add to vocab if not already present
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.vocab_to_id[token_bytes] = new_id


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # Load vocab (assumed to be a JSON file: {token_id: byte_string})
        with open(vocab_filepath, 'r', encoding='utf-8') as vf:
            vocab_data = json.load(vf)
            # Optional: convert keys to int if stored as strings
            vocab = {int(k): bytes(v, 'latin1') if isinstance(v, str) else bytes(v)
                     for k, v in vocab_data.items()}

        # Load merges (assumed to be a list of pairs like: "a b")
        with open(merges_filepath, 'r', encoding='utf-8') as mf:
            lines = mf.readlines()
            # Optional: skip headers like "#version: 0.2"
            merge_pairs = [tuple(line.strip().split()) for line in lines if not line.startswith('#') and line.strip()]
            # Convert to byte-pairs
            merges = [(a.encode('utf-8'), b.encode('utf-8')) for a, b in merge_pairs]

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        chunks = split_by_special(text, self.special_tokens, drop_special=False)
        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.vocab_to_id[chunk.encode('utf-8')])
            else:
                tokens.extend(encode_merged(chunk, self.merges, self.vocab_to_id))
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        "Decode a sequence of token IDs into text."
        return b''.join([self.vocab[t] for t in ids]).decode('utf-8',errors='replace')


# In[ ]:


import regex as re
from collections import defaultdict
import heapq
from tqdm.contrib.concurrent import process_map


# In[ ]:


a = defaultdict(set)


# In[ ]:


a['a'].add('b')


# In[ ]:


a


# In[ ]:




