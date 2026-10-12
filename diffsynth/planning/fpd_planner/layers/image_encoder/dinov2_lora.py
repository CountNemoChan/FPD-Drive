# SPDX-License-Identifier: Apache-2.0
# Adapted through DrivoR; modified for FPD-Drive inference.
# See LICENSE and NOTICE for license terms and attribution.

# code adapted from https://github.com/mrabiabrn/robustbev
#

import torch
from torch import Tensor
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.layers import resample_abs_pos_embed
from einops import rearrange

class timm_ViT(VisionTransformer):

    def _pos_embed(self, x: torch.Tensor, scene_tokens: torch.Tensor = None) -> torch.Tensor:
        """Apply positional embedding to input."""
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        # concatenate the scene tokens
        if scene_tokens is not None:
            x = torch.cat([scene_tokens,x], dim=1)


        return self.pos_drop(x)


    def forward_features(self, x: torch.Tensor, scene_tokens: torch.Tensor = None, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""

        x = self.patch_embed(x)
        x = self._pos_embed(x, scene_tokens)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if attn_mask is not None:
            # If mask provided, we need to apply blocks one by one
            for blk in self.blocks:
                x = blk(x, attn_mask=attn_mask)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x


class _LoRA_qkv_timm(nn.Module):
    """Q/V LoRA projections with the released checkpoint parameter names."""

    def __init__(self, qkv: nn.Module, rank: int):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.linear_a_q = nn.Linear(self.dim, rank, bias=False)
        self.linear_b_q = nn.Linear(rank, self.dim, bias=False)
        self.linear_a_v = nn.Linear(self.dim, rank, bias=False)
        self.linear_b_v = nn.Linear(rank, self.dim, bias=False)

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv


class LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int):
        super().__init__()
        if r > 0:
            for block in vit_model.blocks:
                block.attn.qkv = _LoRA_qkv_timm(block.attn.qkv, r)
        self.lora_vit = vit_model

    def forward(self, x: Tensor, scene_tokens: torch.Tensor = None) -> Tensor:
        return self.lora_vit.forward_features(x, scene_tokens)


class ImgEncoder(torch.nn.Module):
    """Extract embeddings from images using timm's Dinov2 models"""
    model_names = (
        "timm/vit_small_patch14_dinov2.lvd142m",
        "timm/vit_base_patch14_dinov2.lvd142m",
        "timm/vit_large_patch14_dinov2.lvd142m",
        "timm/vit_giant_patch14_dinov2.lvd142m",
        "timm/vit_small_patch14_reg4_dinov2.lvd142m",
        "timm/vit_base_patch14_reg4_dinov2.lvd142m",
        "timm/vit_large_patch14_reg4_dinov2.lvd142m",
        "timm/vit_giant_patch14_reg4_dinov2.lvd142m",
        "timm/vit_small_patch16_dinov3.lvd1689m",
        "timm/vit_large_patch16_dinov3.lvd1689m"
        )

    def __init__(self, 
                 config,
    ):
        super().__init__()


        model_name = config.model_name
        self.num_prefix_tokens = config.num_scene_tokens
        if model_name not in self.model_names:
            raise ValueError(f"Unknown model name: {repr(model_name)}")
        else:
            print("loading ", model_name)
        pretrained_cfg_overlay = {
            "file": config.model_weights,
        }
        
        in_chans = config.in_chans if "in_chans" in config else 3
        
        # The experiment used the Hub-style `timm/` spelling. Use timm's
        # registered architecture with the explicit local safetensors file;
        # this avoids Hub lookup and does not change the checkpoint structure.
        self.model = timm.create_model(
            model_name.removeprefix('timm/'),
            pretrained=True,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            img_size=(config.image_size[1], config.image_size[0]),
            num_classes=0,
            in_chans=in_chans)
        
        self.model.__class__ = timm_ViT
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.use_lora = config.use_lora

        self.neck = torch.nn.Linear(self.model.num_features,config.tf_d_model)

        self.num_features = self.model.num_features

        # Retain the LoRA modules required by the released checkpoint keys.
        if self.use_lora:
            self.model = LoRA_ViT_timm(self.model, r=config.lora_rank)
        self.model.requires_grad_(False)
        self.model.eval()

        # feature pooling
        self.use_feature_pooling = config.use_feature_pooling
        if self.use_feature_pooling:
            self.pool_proj = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool1d(self.num_prefix_tokens)
            )
        
        # focus front cam
        self.focus_front_cam = config.focus_front_cam
        self.compress_fc = config.compress_fc
        if self.compress_fc:
            self.compress_fc_layer = torch.nn.Linear(3957, self.num_prefix_tokens)


    
    # def forward(self, data_dict):

    def forward(self, img, scene_tokens):
        B, N, C, H, W = img.size()
        img = rearrange(img, 'b n c h w -> (b n) c h w')

        scene_tokens = rearrange(scene_tokens, 'b n t c -> (b n) t c')

        # model inference
        if self.use_lora:
            tokens = self.model(img, scene_tokens)
        else:
            tokens = self.model.forward_features(img, scene_tokens)

        if self.use_feature_pooling:
            B_, T, D = tokens.shape  # (B*N, num_tokens, dim)
            # Project the sequence of tokens to `num_prefix_tokens` summary tokens
            tokens = self.pool_proj(tokens.transpose(1, 2))  # shape: (B*N, D, T) → (B*N, D, num_prefix_tokens)
            # print("self.use_feature_pooling: ", self.use_feature_pooling)
            tokens = tokens.transpose(1, 2)  # → (B*N, num_prefix_tokens, D)
        elif self.focus_front_cam:
            B_, T, D = tokens.shape  # (B*N, num_tokens, dim)
            tokens = rearrange(tokens, '(b n) t c -> b n t c', b=B, n=N)
            # all front-cam tokens (camera index 0): [B, T, D]
            front =  tokens[:, 0,  :,                       :] 
            if self.compress_fc:
                # print("before front.shape ", front.shape)  
                front = self.compress_fc_layer(front.transpose(1,2)).transpose(1,2)
                # print("front.shape ", front.shape)                     
            # first K tokens from every other camera: [B, N-1, K, D] -> [B, (N-1)*K, D]
            others = tokens[:, 1:, :self.num_prefix_tokens, :].reshape(B, -1, D)
            # concatenate per batch, preserving order: front first, then cam1..camN-1 prefixes
            tokens = torch.cat([front, others], dim=1)  # [B, (N-1)*K, D]
        elif self.num_prefix_tokens > 0:
            tokens = tokens[:,:self.num_prefix_tokens]
        else:
            tokens = tokens

        tokens = self.neck(tokens)
        if not self.focus_front_cam:
            tokens = rearrange(tokens, '(b n) t c -> b (n t) c', b=B, n=N)

        return tokens
