import torch
import torch.nn.functional as F
from einops import rearrange
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
    # d_in 和 d_out 在这里主要用于类型提示和清晰度，F.linear 会从张量形状推断
    return F.linear(in_features, weights)


def run_embedding(
    vocab_size: int, d_model: int, weights: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    """
    从权重矩阵中查找 token_ids 对应的嵌入向量。
    """
    # vocab_size 和 d_model 参数用于清晰度
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
    # torch.rsqrt(x) is 1/sqrt(x)
    variance = in_features.pow(2).mean(-1, keepdim=True)
    normalized_hidden_states = in_features * torch.rsqrt(variance + eps)
    
    return weights * normalized_hidden_states


def _create_rope_freqs(d_feat: int, max_seq_len: int, theta: float, device: torch.device) -> torch.Tensor:
    """
    创建 RoPE 的频率张量 (cos, sin)。
    这是一个辅助函数，实际应用中通常会缓存这个结果。
    """
    # 计算 theta_i = theta^(-2i / d)
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_feat, 2, device=device, dtype=torch.float32) / d_feat))
    
    # 创建位置索引 t = [0, 1, ..., max_seq_len-1]
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    
    # 计算 m * theta_i
    freqs = torch.outer(t, inv_freq)
    
    # 将 freqs 扩展为 (max_seq_len, d_feat/2) -> (max_seq_len, d_feat)
    # 形式为 [m*theta_0, m*theta_0, m*theta_1, m*theta_1, ...]
    # 这样cos和sin应用后，每个 (cos, sin) 对可以应用于 (x_2i, x_2i+1)
    return torch.cat((freqs, freqs), dim=-1)


def run_rope(
    d_model: int, theta: float, max_seq_len: int, in_query_or_key: torch.Tensor, token_positions: torch.Tensor
) -> torch.Tensor:
    d_feat = in_query_or_key.shape[-1]
    
    # 使用 max_seq_len 来创建 freqs，确保它足够大
    freqs = _create_rope_freqs(d_feat, max_seq_len, theta, device=in_query_or_key.device)
    
    # 从 freqs 中根据实际的 token_positions 选取频率。
    # 假设 token_positions.shape = (batch, seq_len)
    rope_freqs = freqs[token_positions] # 结果 shape = (batch, seq_len, d_feat)
    
    # 调整 rope_freqs 的维度以支持广播 (例如，当 in_query_or_key 有 head 维度时)
    # in_query_or_key.shape: (b, [h], s, d)
    # rope_freqs.shape: (b, s, d)
    # 目标: (b, [1], s, d)

    while rope_freqs.dim() < in_query_or_key.dim():
        rope_freqs = rope_freqs.unsqueeze(1)
        
    x_reshaped = in_query_or_key.reshape(*in_query_or_key.shape[:-1], -1, 2)
    x1, x2 = x_reshaped.unbind(-1)

    cos_freqs = torch.cos(rope_freqs)
    sin_freqs = torch.sin(rope_freqs)
    
    # cos_freqs/sin_freqs need to be split for pairing with x1/x2
    cos = cos_freqs[..., ::2]  # even indices
    sin = sin_freqs[..., ::2]  # even indices
    
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    
    # 5. 合并并恢复原始形状
    rotated = torch.stack((rotated_x1, rotated_x2), dim=-1)
    return rotated.flatten(start_dim=-2)


# --------------------------------------------------------------------------
# 复合模块 (Compound Modules)
# --------------------------------------------------------------------------

# def run_scaled_dot_product_attention(
#     Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None
# ) -> torch.Tensor:
#     """
#     执行缩放点积注意力。
#     """
#     # is_causal=True 会自动应用一个上三角掩码。
#     # 如果提供了自定义 mask，则应设置 is_causal=False。
#     is_causal_flag = mask is None
#     return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=is_causal_flag)

def run_scaled_dot_product_attention(Q, K, V, mask):
    # 计算缩放点积注意力：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    d_k = Q.size(-1)  # Q的最后一个维度，即d_k
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    if mask is not None:
        scores = scores + mask  # 添加mask（通常是负无穷大）
    
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
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


def run_multihead_self_attention(d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features):
    # 计算多头自注意力的输出
    Q = torch.matmul(in_features, q_proj_weight.T)
    K = torch.matmul(in_features, k_proj_weight.T)
    V = torch.matmul(in_features, v_proj_weight.T)

    # 分割成多个头
    Q = Q.view(Q.size(0), num_heads, -1, d_model // num_heads)
    K = K.view(K.size(0), num_heads, -1, d_model // num_heads)
    V = V.view(V.size(0), num_heads, -1, d_model // num_heads)

    # 计算每个头的注意力输出
    attention_output = run_scaled_dot_product_attention(Q, K, V, None)

    # 连接各个头的输出
    attention_output = attention_output.view(attention_output.size(0), -1, d_model)

    # 通过最终的线性变换
    output = torch.matmul(attention_output, o_proj_weight.T)
    return output


# def run_multihead_self_attention(
#     d_model: int, num_heads: int, q_proj_weight: torch.Tensor, k_proj_weight: torch.Tensor, v_proj_weight: torch.Tensor, o_proj_weight: torch.Tensor, in_features: torch.Tensor
# )-> torch.Tensor:
#     """
#     执行标准的多头自注意力 (不含RoPE)。
#     """
#     batch_size, seq_len, _ = in_features.shape
#     d_head = d_model // num_heads

#     # 1. 投影到 Q, K, V
#     q = run_linear(d_model, d_model, q_proj_weight, in_features)
#     k = run_linear(d_model, d_model, k_proj_weight, in_features)
#     v = run_linear(d_model, d_model, v_proj_weight, in_features)

#     # 2. 重塑以分离 head
#     q = rearrange(q, "b s (h d) -> b h s d", h=num_heads)
#     k = rearrange(k, "b s (h d) -> b h s d", h=num_heads)
#     v = rearrange(v, "b s (h d) -> b h s d", h=num_heads)
    
#     # 3. 执行注意力
#     # is_causal=True 适用于自回归语言模型
#     attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

#     # 4. 合并 head 并进行输出投影
#     output = rearrange(attn_output, "b h s d -> b s (h d)")
#     return run_linear(d_model, d_model, o_proj_weight, output)


def run_multihead_self_attention_with_rope(
    d_model: int, num_heads: int, max_seq_len: int, theta: float, q_proj_weight: torch.Tensor, k_proj_weight: torch.Tensor, v_proj_weight: torch.Tensor, o_proj_weight: torch.Tensor, in_features: torch.Tensor, token_positions: torch.Tensor
) -> torch.Tensor:
    """
    执行带有 RoPE 的多头自注意力。
    """
    batch_size, seq_len, _ = in_features.shape
    d_head = d_model // num_heads

    # 1. 投影到 Q, K, V
    q = run_linear(d_model, d_model, q_proj_weight, in_features)
    k = run_linear(d_model, d_model, k_proj_weight, in_features)
    v = run_linear(d_model, d_model, v_proj_weight, in_features)
    
    # 2. 重塑以分离 head
    q = rearrange(q, "b s (h d) -> b h s d", h=num_heads)
    k = rearrange(k, "b s (h d) -> b h s d", h=num_heads)
    v = rearrange(v, "b s (h d) -> b h s d", h=num_heads)

    # 3. 对 Q 和 K 应用 RoPE
    # 注意：RoPE 是在 head 维度上操作的，所以 d_model 参数传 d_head
    q = run_rope(d_head, theta, max_seq_len, q, token_positions)
    k = run_rope(d_head, theta, max_seq_len, k, token_positions)

    # 4. 执行注意力
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # 5. 合并 head 并进行输出投影
    output = rearrange(attn_output, "b h s d -> b s (h d)")
    return run_linear(d_model, d_model, o_proj_weight, output)


# --------------------------------------------------------------------------
# Transformer 整体结构 (Full Transformer Structure)
# --------------------------------------------------------------------------

def run_transformer_block(
    d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, weights: Dict[str, torch.Tensor], in_features: torch.Tensor
) -> torch.Tensor:
    """
    执行一个完整的 Transformer Block (Layer)。
    结构: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual
    """
    seq_len = in_features.shape[1]
    token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0)
    
    # --- Attention Sub-block ---
    residual = in_features
    # Pre-normalization
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
    
    # Residual connection
    hidden_states = residual + attention_output
    
    # --- FFN Sub-block ---
    residual = hidden_states
    # Pre-normalization
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
    
    # Residual connection
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
    
    # 2. Transformer Blocks
    for i in range(num_layers):
        # 提取当前层的权重
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
        )

    # 3. Final Normalization
    hidden_states = run_rmsnorm(
        d_model, eps=1e-5, weights=weights["norm.weight"], in_features=hidden_states
    )
    
    # 4. LM Head (Output Projection)
    # 通常 LM Head 的权重与词嵌入权重共享，但这里我们假设它是一个独立的权重 `output.weight`
    logits = run_linear(d_model, vocab_size, weights["output.weight"], hidden_states)

    return logits
