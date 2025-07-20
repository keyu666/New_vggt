import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from streamvggt.models.aggregator import Aggregator
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass

@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None

class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        
        # Global previous image cache for similarity tracking
        self.prev_image = None

        self.SIMILARITY_THRESH = 0.98
        # True=一刀切，False=更精细
        self.USE_DIRECT_TRIM = True

        # Adaptive trim: skip trimming if too many consecutive trims
        self.enable_adaptive_trim = True  # set False to disable adaptive bypass
        self.trim_skip_threshold = 5      # number of consecutive trims to skip next frame
        self._consecutive_trim_count = 0

    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0
    ):
        images = torch.stack(
            [view["img"] for view in views], dim=0
        ).permute(1, 0, 2, 3, 4)    # B S C H W

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if history_info is None:
            history_info = {"token": None}

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf
            predictions["images"] = images

            B, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    'pts3d_in_other_view': predictions['world_points'][:, s],  # [B, H, W, 3]
                    'conf': predictions['world_points_conf'][:, s],  # [B, H, W]

                    'depth': predictions['depth'][:, s],  # [B, H, W, 1]
                    'depth_conf': predictions['depth_conf'][:, s],  # [B, H, W]
                    'camera_pose': predictions['pose_enc'][:, s, :],  # [B, 9]

                    **({'valid_mask': views[s]["valid_mask"]}
                    if 'valid_mask' in views[s] else {}),  # [B, H, W]

                    **({'track': predictions['track'][:, s],  # [B, N, 2]
                        'vis': predictions['vis'][:, s],  # [B, N]
                        'track_conf': predictions['conf'][:, s]}
                    if 'track' in predictions else {})
                }
                ress.append(res)
            return StreamVGGTOutput(ress=ress, views=views)  # [S] [B, C, H, W]
        
    def inference(self, frames, query_points: torch.Tensor = None, past_key_values=None):        
        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        
        all_ress = []
        processed_frames = []

        for i, frame in enumerate(frames):
            images = frame["img"].unsqueeze(0)
            curr_image = images  
            images_copy = images.clone()
            # ===== 相似度裁剪 start =====
            # Adaptive trim bypass logic
            if self.enable_adaptive_trim and self._consecutive_trim_count >= self.trim_skip_threshold:
                # skip trimming this frame
                skip_trim = True
                self._consecutive_trim_count = 0
            else:
                skip_trim = False

            if self.prev_image is not None:
                if not skip_trim:

                    # Use original image copy for similarity comparison to ensure matching dimensions
                    old_img = images_copy.squeeze(1)  # [B, C, H, W]
                    new_img = images_copy.squeeze(1)  # [B, C, H, W]
                    B, C, H, W = new_img.shape
                    # 初始化待保留的行/列索引
                    rows_keep = list(range(H))
                    cols_keep = list(range(W))
                    if self.USE_DIRECT_TRIM:
                        # ===== 新方法：一次性按距离裁剪 =====
                        # 左侧裁剪：匹配 new 最左列在 old 上的 idx，然后裁掉 new 左侧距离
                        new_col = 0
                        # 在 old 图上滑动匹配 new_col
                        match_idx = None
                        for offset in range(W):
                            old_idx = W - 1 - offset
                            new_strip = new_img[0, :, :, new_col].reshape(C, -1)
                            old_strip = old_img[0, :, :, old_idx].reshape(C, -1)
                            if torch.nn.functional.cosine_similarity(new_strip.flatten(), old_strip.flatten(), dim=0).item() >= self.SIMILARITY_THRESH:
                                match_idx = old_idx
                                break
                        if match_idx is not None:
                            # 删除 new 左侧多余列
                            trim = (W - 1) - match_idx
                            cols_keep = cols_keep[trim:]
                        # 右侧裁剪：匹配 new 最右列
                        new_col_r = cols_keep[-1]
                        match_idx_r = None
                        for offset in range(W):
                            old_idx = offset
                            new_strip = new_img[0, :, :, new_col_r].reshape(C, -1)
                            old_strip = old_img[0, :, :, old_idx].reshape(C, -1)
                            if torch.nn.functional.cosine_similarity(new_strip.flatten(), old_strip.flatten(), dim=0).item() >= self.SIMILARITY_THRESH:
                                match_idx_r = old_idx
                                break
                        if match_idx_r is not None:
                            trim_r = match_idx_r
                            cols_keep = cols_keep[:-trim_r] if trim_r > 0 else cols_keep
                        # 如果横向全部被裁完，则跳过
                        if not cols_keep:
                            continue
                    else:
                        # ===== 老方法：滑动循环剥离列 =====
                        for direction in ("left", "right"):
                            while cols_keep:
                                new_col = cols_keep[0] if direction == "left" else cols_keep[-1]
                                new_strip = new_img[0, :, :, new_col].reshape(C, -1)
                                found = False
                                for offset in range(W):
                                    old_idx = (W - 1 - offset) if direction == "left" else offset
                                    old_strip = old_img[0, :, :, old_idx].reshape(C, -1)
                                    if torch.nn.functional.cosine_similarity(new_strip.flatten(), old_strip.flatten(), dim=0).item() >= self.SIMILARITY_THRESH:
                                        cols_keep.pop(0 if direction == "left" else -1)
                                        found = True
                                        break
                                if not found:
                                    break
                        if not cols_keep:
                            continue
                    # ===== 纵向裁剪开始 =====
                    if self.USE_DIRECT_TRIM:
                        # 新方法：
                        new_row = 0
                        match_r_idx = None
                        for offset in range(H):
                            old_idx = H - 1 - offset
                            new_strip = new_img[0, :, new_row, :].reshape(C, -1)
                            old_strip = old_img[0, :, old_idx, :].reshape(C, -1)
                            if torch.nn.functional.cosine_similarity(new_strip.flatten(), old_strip.flatten(), dim=0).item() >= self.SIMILARITY_THRESH:
                                match_r_idx = old_idx
                                break
                        if match_r_idx is not None:
                            trim_top = (H - 1) - match_r_idx
                            rows_keep = rows_keep[trim_top:]
                        new_row_b = rows_keep[-1]
                        match_b_idx = None
                        for offset in range(H):
                            old_idx = offset
                            new_strip = new_img[0, :, new_row_b, :].reshape(C, -1)
                            old_strip = old_img[0, :, old_idx, :].reshape(C, -1)
                            if torch.nn.functional.cosine_similarity(new_strip.flatten(), old_strip.flatten(), dim=0).item() >= self.SIMILARITY_THRESH:
                                match_b_idx = old_idx
                                break
                        if match_b_idx is not None:
                            trim_bot = match_b_idx
                            rows_keep = rows_keep[:-trim_bot] if trim_bot > 0 else rows_keep
                    else:
                        # 老方法：滑动循环剥离行
                        for direction in ("top", "bottom"):
                            while rows_keep:
                                new_row = rows_keep[0] if direction == "top" else rows_keep[-1]
                                new_strip = new_img[0, :, new_row, :].reshape(C, -1)
                                found = False
                                for offset in range(H):
                                    old_idx = (H - 1 - offset) if direction == "top" else offset
                                    old_strip = old_img[0, :, old_idx, :].reshape(C, -1)
                                    if torch.nn.functional.cosine_similarity(new_strip.flatten(), old_strip.flatten(), dim=0).item() >= self.SIMILARITY_THRESH:
                                        rows_keep.pop(0 if direction == "top" else -1)
                                        found = True
                                        break
                                if not found:
                                    break
                    if not rows_keep or not cols_keep:
                        continue
                    # ===== 可选：pad 而非裁剪到 patch 大小整数倍 =====
                    p = self.aggregator.patch_size
                    # 计算裁剪后尺寸
                    h_new = len(rows_keep)
                    w_new = len(cols_keep)
                    # 需要的 padding
                    pad_bottom = (p - (h_new % p)) % p
                    pad_right = (p - (w_new % p)) % p
                    if pad_bottom > 0 or pad_right > 0:
                        # 先执行裁剪得到 cropped from images_copy
                        base = images_copy.squeeze(1)  # [B, C, H, W]
                        base_crop = base[:, :, rows_keep][:, :, :, cols_keep]  # [B, C, h_new, w_new]
                        # 使用反射填充 pad_bottom 行和 pad_right 列
                        # F.pad expects (left, right, top, bottom)
                        import torch.nn.functional as F
                        cropped = F.pad(base_crop, (0, pad_right, 0, pad_bottom), mode='reflect')
                        rows_keep = list(range(h_new + pad_bottom))
                        cols_keep = list(range(w_new + pad_right))
                        frame["img"] = cropped  # [B, C, H', W']
                        curr_image = cropped    # 保持 4D，不再额外 unsqueeze
                    else:
                        # 没有 padding, 按原逻辑裁剪
                        cropped = images_copy.squeeze(1)[:, :, rows_keep][:, :, :, cols_keep]
                        frame["img"] = cropped
                        curr_image = cropped
                    K = frame["camera_intrinsics"][0].clone()
                    # 主点偏移
                    K[0, 2] -= cols_keep[0]
                    K[1, 2] -= rows_keep[0]
                    frame["camera_intrinsics"] = K.unsqueeze(0)
                else:
                    # When skip_trim is True, do nothing (leave frame untrimmed)
                    pass

            # Update consecutive trim counter
            if not skip_trim and self.prev_image is not None and (('rows_keep' in locals() and rows_keep) or ('cols_keep' in locals() and cols_keep)):
                # trimming happened
                self._consecutive_trim_count += 1
            else:
                # no trimming this frame
                self._consecutive_trim_count = 0

            print(f"Comparing prev and curr images at index {i}")
            # 构造供 Aggregator 输入的张量
            if self.prev_image is None:
                # 第一次推理，无裁剪缓存，直接使用原始 images ([B,1,C,H,W])
                agg_input = images
                keep_idx = None
                remove_idx = None
            else:
                # 后续帧使用裁剪后的 curr_image ([B,C,H',W']), 需加回序列维度
                agg_input = curr_image.unsqueeze(1)  # [B,1,C,H',W']
                # 计算保留的 patch 索引列表（行主序）
                B, C, H, W = curr_image.shape
                num_cols = W // self.aggregator.patch_size
                p = self.aggregator.patch_size
                rows_keep = list(range(H))
                cols_keep = list(range(W))
                # Note: rows_keep, cols_keep are re-calculated here to match curr_image, which may have been padded/cropped above
                # If you want to use the exact rows_keep/cols_keep used above, consider storing them in the frame or in a local variable before this
                keep_idx = []
                for r in rows_keep:
                    for c in cols_keep:
                        keep_idx.append(r * num_cols + c)
                total_patches = (H // p) * (W // p)
                remove_idx = [i for i in range(total_patches) if i not in keep_idx]
            # 调用 Aggregator
            aggregator_output = self.aggregator(
                agg_input, 
                past_key_values=past_key_values,
                use_cache=True, 
                past_frame_idx=i,
                keep_idx=keep_idx,
                remove_idx=remove_idx
            )
            
            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output

            # ==== Ensure aggregated_tokens_list length for depth_head ====
            # expected number of blocks = number of global attention cycles
            expected_len = self.aggregator.depth
            # aggregated_tokens may be a single tensor when caching: wrap into list if needed
            if not isinstance(aggregated_tokens, list):
                aggregated_tokens = [aggregated_tokens]
            if len(aggregated_tokens) < expected_len:
                last = aggregated_tokens[-1]
                aggregated_tokens.extend([last] * (expected_len - len(aggregated_tokens)))
            
            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    pose_enc, past_key_values_camera = self.camera_head(aggregated_tokens, past_key_values_camera=past_key_values_camera, use_cache=True)
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens, images=agg_input, patch_start_idx=patch_start_idx
                    )
                    depth = depth[:, 0] 
                    depth_conf = depth_conf[:, 0]
                
                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens, images=agg_input, patch_start_idx=patch_start_idx
                    )
                    pts3d = pts3d[:, 0] 
                    pts3d_conf = pts3d_conf[:, 0]

                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=agg_input, patch_start_idx=patch_start_idx, query_points=query_points
                    )
                    track = track_list[-1][:, 0]  
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            all_ress.append({
                'pts3d_in_other_view': pts3d,
                'conf': pts3d_conf,
                'depth': depth,
                'depth_conf': depth_conf,
                'camera_pose': camera_pose,
                **({'valid_mask': frame["valid_mask"]}
                    if 'valid_mask' in frame else {}),  

                **({'track': track, 
                    'vis': vis,  
                    'track_conf': track_conf}
                if query_points is not None else {})
            })
            # update global prev_image for next iteration
            self.prev_image = curr_image  # already squeezed to [B, C, H, W]
            processed_frames.append(frame)
        
        output = StreamVGGTOutput(ress=all_ress, views=processed_frames)
        return output