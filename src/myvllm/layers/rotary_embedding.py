import torch.nn as nn
import torch 

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Handle both 3D varlen (total_tokens, num_heads, head_dim) and 4D batched (B, seq_len, num_heads, head_dim)
    
    # 变长模式：拼在一起只是内存布局
    # 不是把它们当成一个长句子
    # 每个seq的开头位置还是0
    if x.dim() == 3:
        # Varlen mode: (total_tokens, num_heads, head_dim)
        total_tokens, num_heads, head_dim = x.shape
        # cos, sin shape: (total_tokens, head_dim/2)
        # Expand to (total_tokens, 1, head_dim/2) for broadcasting
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # 在 head 的维度 不是seq的维度上切分成两半
        # Split x into two halves along the head dimension
        x1, x2 = x.chunk(2, dim=-1)

        # Apply rotary embedding
        # x1, x2 shape: (total_tokens, num_heads, head_dim/2)
        # cos, sin shape: (total_tokens, 1, head_dim/2)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.cat([out1, out2], dim=-1)
    else:
        # Batched mode: (B, seq_len, num_heads, head_dim)
        B = x.size(0)
        seq_len = x.size(1)
        num_heads = x.size(2)
        head_dim = x.size(-1)

        # Expand cos and sin to match the batch and head dimensions
        # cos, sin shape: (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # Split x into two halves along the head dimension
        x1, x2 = x.chunk(2, dim=-1)

        # Apply rotary embedding with proper broadcasting
        # x1, x2 shape: (B, seq_len, num_heads, head_dim/2)
        # cos, sin shape: (1, seq_len, 1, head_dim/2)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.cat([out1, out2], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self, 
        base:int,
        rotary_embedding: int, 
        max_position: int = 2048,
        is_llama3: bool = False,
        # the following params are only used in llama3.2
        llama3_rope_factor: float = 32.0,
        llama3_rope_high_freq_factor: float = 4.0,
        llama3_rope_low_freq_factor: float = 1.0,
        llama3_rope_original_max_position_embeddings: int = 8192,
    ):
        
        # llama3_rope_high_freq_factor: float = 4.0,
        # llama3_rope_low_freq_factor: float = 1.0,
        # llama3_rope_original_max_position_embeddings: int = 8192,
        # 高频边界 = 8192 / 4 = 2048
        # 低频边界 = 8192 / 1 = 8192    

        super().__init__()
        self.base = base
        # how many dimensions to apply rotary embedding
        # 旋转位置编码的维度
        self.rotary_embedding = rotary_embedding
        # max position that the long context can reach
        self.max_position = max_position
        
        # 不同的维度对，旋转的频率不一样
        # 在这里的实现里，假设d = 8
        # 第 0 对: (x0, x4)
        # 第 1 对: (x1, x5)
        # 第 2 对: (x2, x6)
        # 第 3 对: (x3, x7)
        # 这些维度对的旋转频率一样
        self.inv_freq = 1/(base ** (torch.arange(0, self.rotary_embedding, 2)/self.rotary_embedding))

        if is_llama3:
            # specifically for llama3.2
            import math
            inv_freq = self.inv_freq
            # no smooth if low_freq_factor == high_freq_factor
            
            # wave_len: 大概要经过多少个 token 位置，才转完一整圈
            # 现在把 波长分为 三档
            # wave_len < 2048
            #     高频，保持 inv_freq 不变。
  
            # 2048 <= wave_len <= 8192
            #     中频，平滑缩放。
      
            # wave_len > 8192
            #     低频，把 inv_freq 除以 32。       



            wave_len = 2 * math.pi / inv_freq

            # 对不同阶段的波长进行不同的缩放处理
            if llama3_rope_low_freq_factor == llama3_rope_high_freq_factor:
                inv_freq = torch.where(
                    wave_len < llama3_rope_original_max_position_embeddings / llama3_rope_high_freq_factor,
                    inv_freq,
                    inv_freq / llama3_rope_factor,
                )
            else:
                delta = llama3_rope_high_freq_factor - llama3_rope_low_freq_factor
                smooth = (llama3_rope_original_max_position_embeddings / wave_len - llama3_rope_low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / llama3_rope_factor + smooth
                inv_freq = factor * inv_freq
            self.inv_freq = inv_freq

        positions = torch.arange(self.max_position).float()
        # (max_position, rotary_embedding/2)
        freqs = torch.einsum("i,j -> ij", positions, self.inv_freq)
        # 相当于做outer product 外积
        # 每个 position 在每个维度对上的旋转角度
        # [
        # [0*1, 0*0.1, 0*0.01],
        # [1*1, 1*0.1, 1*0.01],
        # [2*1, 2*0.1, 2*0.01],
        # [3*1, 3*0.1, 3*0.01],
        # ]

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # (max_position, rotary_embedding)
        cos_sin_cache = torch.cat([cos, sin], dim=-1)
        self.register_buffer("cos_sin_cache", cos_sin_cache)

    @torch.compile
    # tell the position index of the token
    # apply rotary embedding to query and key
    def forward(self, positions, query, key):
        cos_sin = self.cos_sin_cache[positions]  # (seq_len, rotary_embedding)
        cos, sin = cos_sin.chunk(2, dim=-1)
        return (
            apply_rotary_pos_emb(query, cos, sin),
            apply_rotary_pos_emb(key, cos, sin)
        )


if __name__ == "__main__":
    base = 5
    # how many dimensions to apply rotary embedding
    rotary_dim = 16
    # maximum position that the long context can reach
    max_position = 100
    print(torch.arange(0, rotary_dim, 2))
    print(base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    print(inv_freq)

    t = torch.arange(max_position).float()

    freqs = torch.einsum("i,j -> ij", t, inv_freq)

    print(freqs.size())

    print(freqs[2])

