
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from streamvggt.layers.block import Block

def slice_expand_and_flatten(token_tensor: torch.Tensor, B: int, S: int) -> torch.Tensor:
    """
    Process a token tensor of shape (1, 2, X, C) for multiple frames:
    - The first slice (index 0) is for the first frame, second slice (index 1) for all subsequent frames.
    - Expand to batch size B and sequence length S.
    - Flatten to shape (B*S, X, C).
    """
    assert token_tensor.ndim == 4 and token_tensor.shape[1] == 2, "Token tensor must have shape (1, 2, X, C)"
    X = token_tensor.shape[2]
    # Token for first frame
    first = token_tensor[:, 0:1, ...].expand(B, 1, X, token_tensor.shape[-1])  # (B, 1, X, C)
    if S > 1:
        # Token for remaining frames
        others = token_tensor[:, 1:2, ...].expand(B, S-1, X, token_tensor.shape[-1])  # (B, S-1, X, C)
        tokens = torch.cat([first, others], dim=1)  # (B, S, X, C)
    else:
        tokens = first  # (B, 1, X, C)
    tokens = tokens.reshape(B * S, X, token_tensor.shape[-1])  # flatten to (B*S, X, C)
    return tokens

class PatchEmbed(nn.Module):
    """Patch Embedding using a convolutional projection."""
    def __init__(self, img_size=518, patch_size=14, in_chans=3, embed_dim=1024, flatten_embedding=True):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.flatten_embedding = flatten_embedding
        # Convolution to generate patch tokens
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()  # (could use LayerNorm if needed)
        self.number = 0 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        self.number += 1
        x = self.proj(x)  # [B, embed_dim, H_patch, W_patch]
        H_p, W_p = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, H_p*W_p, embed_dim]
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.view(-1, H_p, W_p, x.shape[-1])  # [B, H_patch, W_patch, embed_dim]
        return x

class Aggregator(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0, num_register_tokens=4):
        super().__init__()
        # expose patch_size for inference
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth  # total number of (frame+global) attention sub-blocks
        # Alternating attention sequence (frame then global, repeated)
        self.aa_order = ["frame", "global"]
        assert depth % len(self.aa_order) == 0, "Depth must be multiple of number of attention types"
        self.aa_block_num = depth // len(self.aa_order)  # number of alternating cycles
        # Patch embedding and special tokens
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))            # shape (1,2,1,C)
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))  # shape (1,2,R,C)
        # Index at which patch tokens start in token sequence (after camera + register tokens)
        self.patch_start_idx = 1 + num_register_tokens
        # Transformer blocks for frame and global attention
        self.frame_blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) 
                                           for _ in range(self.aa_block_num)])
        self.global_blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) 
                                            for _ in range(self.aa_block_num)])
        # Pre-computed mean and std for normalization (ImageNet)
        self.register_buffer("_resnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("_resnet_std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        # Keep track of last frame's new patch token count for computing offsets
        self.last_new_patch_count: Optional[int] = None

    def forward(self, 
                images: torch.Tensor,           # [B, S, C, H, W]
                past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
                use_cache: bool = False,
                past_frame_idx: int = 0,
                keep_idx: Optional[List[int]] = None,
                remove_idx: Optional[List[int]] = None) -> Tuple:
        """
        If use_cache is True, process images in streaming mode (one frame at a time) using past_key_values.
        Otherwise, process all frames together (batch mode).
        Returns a tuple containing:
        - output_list: List of intermediate outputs (each of shape [B, S, seq_len, 2*embed_dim])
        - patch_start_idx: index where patch tokens start in token sequence (useful for heads)
        - past_key_values (updated, only if use_cache=True)
        """
        B, S = images.shape[0], images.shape[1]
        # Normalize images
        images = (images - self._resnet_mean) / self._resnet_std
        if not use_cache:
            # Non-streaming: process all frames at once
            images_flat = images.view(B * S, images.shape[2], images.shape[3], images.shape[4])  # [B*S, C, H, W]
            patch_tokens = self.patch_embed(images_flat)  # [B*S, P, C]
            if isinstance(patch_tokens, dict):  # if PatchEmbed returns dict (pretrained ViT usage)
                patch_tokens = patch_tokens["x_norm_patchtokens"]
            _, P, C = patch_tokens.shape
            # Prepare special tokens
            camera_tokens = slice_expand_and_flatten(self.camera_token, B, S)       # [B*S, 1, C]
            register_tokens = slice_expand_and_flatten(self.register_token, B, S)   # [B*S, num_register_tokens, C]
            tokens = torch.cat([camera_tokens, register_tokens, patch_tokens], dim=1)  # [B*S, 1+R+P, C]
            # Position embeddings (if using rotary or absolute, omitted here for brevity)
            pos = None  # (Assume position encoding handled inside Block if needed)
            # Reshape tokens to [B, S, seq_len, C] for alternating attention
            seq_len = tokens.shape[1]
            tokens = tokens.view(B, S, seq_len, C)
            frame_idx = global_idx = 0
            output_list: List[torch.Tensor] = []
            # Alternate frame and global attention blocks
            for _ in range(self.aa_block_num):
                # Frame attention (intra-frame)
                # Flatten frames for frame attention: shape [B*S, seq_len, C]
                frame_in = tokens.view(B * S, seq_len, C)
                frame_out = self.frame_blocks[frame_idx](frame_in)  # returns [B*S, seq_len, C]
                frame_idx += 1
                frame_out = frame_out.view(B, S, seq_len, C)
                # Global attention (inter-frame)
                global_in = frame_out.view(B, S * seq_len, C)      # [B, S*seq_len, C]
                global_out = self.global_blocks[global_idx](global_in)  # returns [B, S*seq_len, C]
                global_idx += 1
                global_out = global_out.view(B, S, seq_len, C)
                # Concatenate frame and global outputs along feature dimension
                concat_out = torch.cat([frame_out, global_out], dim=-1)  # [B, S, seq_len, 2*C]
                output_list.append(concat_out)
                tokens = global_out  # update tokens for next cycle
            return output_list, self.patch_start_idx  # no past_key_values returned in non-caching mode
        else:
            # Streaming mode: one frame at a time (S should be 1)
            assert S == 1, "Streaming mode expects S=1 (one frame per forward call)"
            # Prepare patch tokens for the current frame
            images_flat = images.view(B * S, images.shape[2], images.shape[3], images.shape[4])
            patch_tokens_new = self.patch_embed(images_flat)  # [B, P_new, C] since B*S = B, S=1
            if isinstance(patch_tokens_new, dict):
                patch_tokens_new = patch_tokens_new["x_norm_patchtokens"]
            P_new = patch_tokens_new.shape[1]  # number of patch tokens in current frame
            # Prepare special tokens for this frame (use first set for first frame, second set for subsequent frames)
            if past_frame_idx > 0:
                # Subsequent frame: use second position tokens
                cam_tok = self.camera_token[:, 1:2, ...].expand(B, 1, 1, self.embed_dim)   # (B,1,1,C)
                reg_tok = self.register_token[:, 1:2, ...].expand(B, 1, -1, self.embed_dim) # (B,1,R,C)
            else:
                # First frame: use first position tokens
                cam_tok = self.camera_token[:, 0:1, ...].expand(B, 1, 1, self.embed_dim)   # (B,1,1,C)
                reg_tok = self.register_token[:, 0:1, ...].expand(B, 1, -1, self.embed_dim) # (B,1,R,C)
            cam_tok = cam_tok.reshape(B, 1, self.embed_dim)                           # [B, 1, C]
            reg_tok = reg_tok.reshape(B, self.patch_start_idx - 1, self.embed_dim)    # [B, R, C], where patch_start_idx = 1+R
            tokens_new = torch.cat([cam_tok, reg_tok, patch_tokens_new], dim=1)       # [B, patch_start_idx + P_new, C]
            # Print debug info about patch counts for size consistency
            if self.patch_embed is not None:
                orig_patches = self.patch_embed.num_patches
            else:
                orig_patches = None
            print(f"Original patch count: {orig_patches}, current frame patch count: {P_new}")
            # Ensure past_key_values list is initialized
            if past_key_values is None:
                past_key_values = [None] * self.depth
            # Determine previous frame's new patch count for removal offset calculations
            prev_new_patch_count = self.last_new_patch_count
            # Update last_new_patch_count for this frame
            self.last_new_patch_count = P_new
            output_list: List[torch.Tensor] = []
            frame_idx = global_idx = 0
            new_past_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.depth
            # Process each sub-block
            x = tokens_new  # shape [B, seq_len_new, C]
            for layer in range(self.depth):
                attn_type = self.aa_order[layer % len(self.aa_order)]
                if attn_type == "frame":
                    # Frame attention block (no cross-frame attention)
                    x = self.frame_blocks[frame_idx](x)  # [B, seq_len_new, C]
                    frame_idx += 1
                    frame_intermediate = x.view(B, 1, -1, self.embed_dim)  # save intermediate output
                elif attn_type == "global":
                    # Global attention block (with cached keys from past frames)
                    pkv = past_key_values[layer]  # past (key, value) for this layer
                    out, new_kv = self.global_blocks[global_idx](
                        x, 
                        past_key_value=pkv, 
                        use_cache=True, 
                        remove_idx=remove_idx, 
                        past_key_values_length=(pkv[0].shape[1] if pkv is not None else 0),
                        patch_start_idx=self.patch_start_idx,
                        prev_new_patch_count=prev_new_patch_count
                    )
                    x = out  # updated tokens for current frame
                    new_past_key_values[layer] = new_kv  # store updated key/value for this layer
                    global_idx += 1
                    global_intermediate = x.view(B, 1, -1, self.embed_dim)
                    # Concatenate frame and global intermediate features
                    concat_feat = torch.cat([frame_intermediate, global_intermediate], dim=-1)  # [B,1, seq_len_new, 2*embed_dim]
                    output_list.append(concat_feat)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
            # Return intermediate outputs list, patch_start_idx, and updated caches
            return output_list, self.patch_start_idx, new_past_key_values
