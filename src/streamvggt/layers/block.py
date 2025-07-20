import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

class Block(nn.Module):
    """Transformer block with multi-head self-attention and feed-forward network."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "Embed_dim must be divisible by num_heads"
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp_fc1 = nn.Linear(dim, hidden_dim)
        self.mlp_act = nn.GELU()
        self.mlp_fc2 = nn.Linear(hidden_dim, dim)

    def forward(self,
                x: torch.Tensor,
                pos: torch.Tensor = None,
                attn_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                remove_idx: Optional[List[int]] = None,
                past_key_values_length: int = 0,
                patch_start_idx: int = 0,
                prev_new_patch_count: Optional[int] = None,
                **kwargs) -> Tuple:
        B, N, C = x.shape
        x_norm = self.norm1(x)
        Q = self.q_proj(x_norm)
        K_new = self.k_proj(x_norm)
        V_new = self.v_proj(x_norm)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K_new = K_new.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V_new = V_new.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if use_cache:
            # 重载过去的 K/V
            past_k, past_v = (None, None) if past_key_value is None else past_key_value
            if past_k is not None:
                # 先 reshape
                past_k_r = past_k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
                past_v_r = past_v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
                # 裁剪 overlapping 区域
                if remove_idx is not None and prev_new_patch_count is not None:
                    offset = past_key_values_length - prev_new_patch_count - patch_start_idx
                    remove_pos = [offset + patch_start_idx + idx for idx in remove_idx]
                    mask = torch.ones(past_k_r.shape[2], dtype=torch.bool, device=x.device)
                    for idx in remove_pos:
                        if 0 <= idx < mask.shape[0]:
                            mask[idx] = False
                    past_k_r = past_k_r[:, :, mask, :]
                    past_v_r = past_v_r[:, :, mask, :]
                    print(f"[Block] Removed {len(remove_pos)} tokens: past length {past_key_values_length} -> {past_k_r.shape[2]}")
                # 合并 old + new
                K_full = torch.cat([past_k_r, K_new], dim=2)
                V_full = torch.cat([past_v_r, V_new], dim=2)
            else:
                K_full, V_full = K_new, V_new

            # 注意力计算
            scores = torch.matmul(Q, K_full.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, V_full).transpose(1, 2).reshape(B, N, C)
            x = x + self.out_proj(attn_out)
            x = x + self.mlp_fc2(self.mlp_act(self.mlp_fc1(self.norm2(x))))
            # 更新缓存：拼回原始位置
            K_new_flat = K_new.transpose(1, 2).reshape(B, N, C)
            V_new_flat = V_new.transpose(1, 2).reshape(B, N, C)
            if past_k is not None:
                new_k = torch.cat([past_k, K_new_flat], dim=1)
                new_v = torch.cat([past_v, V_new_flat], dim=1)
            else:
                new_k, new_v = K_new_flat, V_new_flat
            #print(f"[Block] Memory rebuilt to length {new_k.shape[1]}")
            return x, (new_k.detach(), new_v.detach())
        else:
            # 标准自注意力
            scores = torch.matmul(Q, K_new.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, V_new).transpose(1, 2).reshape(B, N, C)
            x = x + self.out_proj(attn_out)
            x = x + self.mlp_fc2(self.mlp_act(self.mlp_fc1(self.norm2(x))))
            return x
NestedTensorBlock = Block
