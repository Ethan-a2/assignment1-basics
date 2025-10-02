import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum, reduce
from torch import Tensor
from jaxtyping import Float,Int
import math
from typing import Dict

# --------------------------------------------------------------------------
# 核心构建模块 (Core Building Blocks)
# --------------------------------------------------------------------------

def run_linear(
    d_in: int, d_out: int, weights: torch.Tensor, in_features: torch.Tensor
) -> torch.Tensor:
    """
    执行一个线性变换 (矩阵乘法)。
    
    PyTorch 的 F.linear 函数内部处理了权重的转置，
    即计算 input @ weight.T。
    """
    return F.linear(in_features, weights)


def run_embedding(
    vocab_size: int, d_model: int, weights: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    """
    从权重矩阵中查找 token_ids 对应的嵌入向量。
    """
    return F.embedding(token_ids, weights)


def run_silu(x: torch.Tensor) -> torch.Tensor:
    """
    执行 SiLU (Sigmoid-weighted Linear Unit) 激活函数, 也即 Swish。
    公式: x * sigmoid(x)
    """
    return F.silu(x)


def run_rmsnorm(
    d_model: int, eps: float, weights: torch.Tensor, in_features: torch.Tensor
) -> torch.Tensor:
    """
    执行 Root Mean Square Layer Normalization。
    
    公式: x * (1 / sqrt(mean(x^2) + epsilon)) * weight
    """
    variance = in_features.pow(2).mean(-1, keepdim=True)
    normalized_hidden_states = in_features * torch.rsqrt(variance + eps)
    
    return weights * normalized_hidden_states


def _create_rope_freqs(d_feat: int, max_seq_len: int, theta: float, device: torch.device) -> torch.Tensor:
    """
    创建 RoPE 的频率张量。
    返回 shape: (max_seq_len, d_feat)
    """
    # 计算 theta_i = theta^(-2i / d) for i in [0, d/2)
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_feat, 2, device=device, dtype=torch.float32) / d_feat))
    
    # 创建位置索引 t = [0, 1, ..., max_seq_len-1]
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    
    # 计算 m * theta_i, shape: (max_seq_len, d_feat/2)
    freqs = torch.outer(t, inv_freq)
    
    # 扩展为 (max_seq_len, d_feat)
    # 形式为 [m*theta_0, m*theta_0, m*theta_1, m*theta_1, ...]
    return torch.cat((freqs, freqs), dim=-1)


def run_rope(
    d_feat: int, theta: float, max_seq_len: int, in_query_or_key: torch.Tensor, token_positions: torch.Tensor
) -> torch.Tensor:
    """
    对输入张量应用旋转位置编码 (Rotary Positional Embedding)。
    
    参考实现的逻辑：
    1. 将输入重塑为 (..., d/2, 2)，分离出 (x1, x2) 对
    2. 应用旋转: 
       - rotated_x1 = x1 * cos - x2 * sin
       - rotated_x2 = x1 * sin + x2 * cos
    3. 重新组合
    """
    _d_feat = in_query_or_key.shape[-1]
    
    # 创建频率表
    freqs = _create_rope_freqs(_d_feat, max_seq_len, theta, device=in_query_or_key.device)
    
    # 处理 token_positions 的维度
    # 如果 token_positions 是 1D，需要扩展以匹配 batch 维度
    if token_positions.dim() == 1:
        # 假设 in_query_or_key 的第一个维度是 batch（或者需要广播）
        # token_positions shape: (seq_len,) -> 需要变成可广播的形状
        rope_freqs = freqs[token_positions]  # (seq_len, d_feat)
        # 为 batch 维度添加维度
        rope_freqs = rope_freqs.unsqueeze(0)  # (1, seq_len, d_feat)
    else:
        # token_positions shape: (batch, seq_len)
        rope_freqs = freqs[token_positions]  # (batch, seq_len, d_feat)
    
    # 调整维度以匹配 in_query_or_key
    # in_query_or_key 可能是 (b, h, s, d) 或 (b, s, d)
    while rope_freqs.dim() < in_query_or_key.dim():
        # 在第二个位置插入维度（为 head 维度）
        rope_freqs = rope_freqs.unsqueeze(1)
    
    # 重塑输入为 (..., d/2, 2) 以分离 (x1, x2) 对
    x_reshaped = in_query_or_key.reshape(*in_query_or_key.shape[:-1], -1, 2)
    x1, x2 = x_reshaped.unbind(-1)  # 每个 shape: (..., d/2)
    
    # 计算 cos 和 sin
    cos_freqs = torch.cos(rope_freqs)
    sin_freqs = torch.sin(rope_freqs)
    
    # 提取对应的 cos 和 sin 值
    # rope_freqs 的形式是 [freq_0, freq_0, freq_1, freq_1, ...]
    # 我们需要 [freq_0, freq_1, ...] 对应 x1 和 x2
    cos_reshaped = cos_freqs.reshape(*cos_freqs.shape[:-1], -1, 2)
    sin_reshaped = sin_freqs.reshape(*sin_freqs.shape[:-1], -1, 2)
    
    cos = cos_reshaped[..., 0]  # shape: (..., d/2)
    sin = sin_reshaped[..., 0]  # shape: (..., d/2)
    
    # 应用旋转
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    
    # 重新组合
    rotated = torch.stack((rotated_x1, rotated_x2), dim=-1)
    return rotated.flatten(start_dim=-2)


# --------------------------------------------------------------------------
# 复合模块 (Compound Modules)
# --------------------------------------------------------------------------

def run_scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    执行缩放点积注意力，与参考实现对齐。
    
    Args:
        Q: 查询张量, shape (... q_len, d_k)
        K: 键张量, shape (... k_len, d_k)
        V: 值张量, shape (... k_len, d_v)
        mask: 布尔掩码, shape (... q_len, k_len)。True 的位置将被屏蔽。
    
    注意：参考实现使用 ~mask，这意味着传入的 mask 中 True 表示要保留的位置，
    False 表示要屏蔽的位置。但根据测试，我们的实现应该直接使用 mask（True=屏蔽）。
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(d_k)
    
    if mask is not None:
        # mask 中 True 的位置是需要被屏蔽的
        scores = scores.masked_fill(mask, float("-inf"))
    
    attention_weights = F.softmax(scores, dim=-1)
    output = einsum(attention_weights, V, "... q k, ... k v -> ... q v")
    return output


def run_swiglu(
    d_model: int, d_ff: int, w1_weight: torch.Tensor, w2_weight: torch.Tensor, w3_weight: torch.Tensor, in_features: torch.Tensor
) -> torch.Tensor:
    """
    执行 SwiGLU 前馈网络。
    公式: (SiLU(x @ W1.T) * (x @ W3.T)) @ W2.T
    """
    gate = run_linear(d_model, d_ff, w1_weight, in_features)
    up_proj = run_linear(d_model, d_ff, w3_weight, in_features)
    
    fused_gate = run_silu(gate) * up_proj
    
    down_proj = run_linear(d_ff, d_model, w2_weight, fused_gate)
    return down_proj


def run_multihead_self_attention(
    d_model: int, num_heads: int, q_proj_weight: torch.Tensor, k_proj_weight: torch.Tensor, v_proj_weight: torch.Tensor, o_proj_weight: torch.Tensor, in_features: torch.Tensor
)-> torch.Tensor:
    """
    执行标准的多头自注意力 (不含RoPE)，与参考实现对齐。
    """
    batch_size, seq_len, _ = in_features.shape

    # 1. 投影到 Q, K, V
    q = run_linear(d_model, d_model, q_proj_weight, in_features)
    k = run_linear(d_model, d_model, k_proj_weight, in_features)
    v = run_linear(d_model, d_model, v_proj_weight, in_features)

    # 2. 重塑以分离 head
    q = rearrange(q, "b s (h d) -> b h s d", h=num_heads)
    k = rearrange(k, "b s (h d) -> b h s d", h=num_heads)
    v = rearrange(v, "b s (h d) -> b h s d", h=num_heads)
    
    # 3. 创建因果掩码
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1
    )

    # 4. 执行注意力
    attn_output = run_scaled_dot_product_attention(q, k, v, mask=causal_mask)

    # 5. 合并 head 并进行输出投影
    output = rearrange(attn_output, "b h s d -> b s (h d)")
    return run_linear(d_model, d_model, o_proj_weight, output)


def run_multihead_self_attention_with_rope(
    d_model: int, num_heads: int, max_seq_len: int, theta: float, q_proj_weight: torch.Tensor, k_proj_weight: torch.Tensor, v_proj_weight: torch.Tensor, o_proj_weight: torch.Tensor, in_features: torch.Tensor, token_positions: torch.Tensor
) -> torch.Tensor:
    """
    执行带有 RoPE 的多头自注意力，与参考实现对齐。
    """
    batch_size, seq_len, _ = in_features.shape
    head_dim = d_model // num_heads

    # 1. 投影到 Q, K, V
    q = run_linear(d_model, d_model, q_proj_weight, in_features)
    k = run_linear(d_model, d_model, k_proj_weight, in_features)
    v = run_linear(d_model, d_model, v_proj_weight, in_features)
    
    # 2. 重塑以分离 head
    q = rearrange(q, "b s (h d) -> b h s d", h=num_heads)
    k = rearrange(k, "b s (h d) -> b h s d", h=num_heads)
    v = rearrange(v, "b s (h d) -> b h s d", h=num_heads)

    # 3. 对 Q 和 K 应用 RoPE
    q = run_rope(head_dim, theta, max_seq_len, q, token_positions)
    k = run_rope(head_dim, theta, max_seq_len, k, token_positions)

    # 4. 创建因果掩码
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1
    )

    # 5. 执行注意力
    attn_output = run_scaled_dot_product_attention(q, k, v, mask=causal_mask)

    # 6. 合并 head 并进行输出投影
    output = rearrange(attn_output, "b h s d -> b s (h d)")
    return run_linear(d_model, d_model, o_proj_weight, output)


# --------------------------------------------------------------------------
# Transformer 整体结构 (Full Transformer Structure)
# --------------------------------------------------------------------------

def run_transformer_block(
    d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, weights: Dict[str, torch.Tensor], in_features: torch.Tensor, token_positions: torch.Tensor = None
) -> torch.Tensor:
    """
    执行一个完整的 Transformer Block (Layer)。
    结构: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual
    """
    # 如果没有提供 token_positions，则生成
    if token_positions is None:
        seq_len = in_features.shape[1]
        batch_size = in_features.shape[0]
        token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0).expand(batch_size, -1)
    
    # --- Attention Sub-block ---
    residual = in_features
    normalized_input = run_rmsnorm(
        d_model, eps=1e-5, weights=weights["ln1.weight"], in_features=in_features
    )
    
    attention_output = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=normalized_input,
        token_positions=token_positions,
    )
    
    hidden_states = residual + attention_output
    
    # --- FFN Sub-block ---
    residual = hidden_states
    normalized_hidden_states = run_rmsnorm(
        d_model, eps=1e-5, weights=weights["ln2.weight"], in_features=hidden_states
    )
    
    ffn_output = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
        in_features=normalized_hidden_states,
    )
    
    output = residual + ffn_output
    
    return output


def run_transformer_lm(
    vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, weights: Dict[str, torch.Tensor], in_indices: torch.Tensor
) -> torch.Tensor:
    """
    执行一个完整的 Transformer 语言模型。
    """
    # 1. Token Embeddings
    hidden_states = run_embedding(
        vocab_size, d_model, weights["token_embeddings.weight"], in_indices
    )
    
    # 2. 生成 token positions（一次性生成，传递给所有层）
    seq_len = in_indices.shape[1]
    batch_size = in_indices.shape[0]
    token_positions = torch.arange(seq_len, device=in_indices.device).unsqueeze(0).expand(batch_size, -1)
    
    # 3. Transformer Blocks
    for i in range(num_layers):
        layer_weights = {
            k.replace(f"layers.{i}.", ""): v
            for k, v in weights.items()
            if f"layers.{i}." in k
        }
        hidden_states = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=hidden_states,
            token_positions=token_positions,
        )

    # 4. Final Normalization
    # 尝试不同的权重键名
    norm_weight_key = "norm.weight" if "norm.weight" in weights else "ln_f.weight"
    hidden_states = run_rmsnorm(
        d_model, eps=1e-5, weights=weights[norm_weight_key], in_features=hidden_states
    )
    
    # 5. LM Head
    logits = run_linear(d_model, vocab_size, weights["output.weight"], hidden_states)

    return logits