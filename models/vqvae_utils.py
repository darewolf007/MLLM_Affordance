"""
Shared VQVAE utilities for affordance prediction schemes.

Provides:
  - VQVAE3D model (copied from ShapeLLM-Omni for standalone use)
  - pointcloud_to_voxel: point cloud → binary occupancy grid
  - vqvae_tokenize: occupancy grid → 1024 discrete token IDs
  - load_vqvae: load pretrained VQVAE weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Minimal reimplementation of VQVAE3D components (from ShapeLLM-Omni)
# so that 3d_llm does not depend on the ShapeLLM-Omni codebase at runtime.
# ---------------------------------------------------------------------------

def _zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class _GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class _ChannelLayerNorm32(nn.LayerNorm):
    def forward(self, x):
        # x: (B, C, *spatial)
        x = x.permute(0, *range(2, x.ndim), 1)  # move C to last
        x = super().forward(x.float()).type(x.dtype)
        x = x.permute(0, x.ndim - 1, *range(1, x.ndim - 1))
        return x


def _norm_layer(norm_type, channels):
    if norm_type == "group":
        return _GroupNorm32(32, channels)
    elif norm_type == "layer":
        return _ChannelLayerNorm32(channels)
    raise ValueError(f"Invalid norm type {norm_type}")


def _pixel_shuffle_3d(x, scale):
    B, C, D, H, W = x.shape
    nC = C // (scale ** 3)
    x = x.reshape(B, nC, scale, scale, scale, D, H, W)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    x = x.reshape(B, nC, D * scale, H * scale, W * scale)
    return x


class _ResBlock3d(nn.Module):
    def __init__(self, channels, out_channels=None, norm_type="layer"):
        super().__init__()
        self.out_channels = out_channels or channels
        self.norm1 = _norm_layer(norm_type, channels)
        self.norm2 = _norm_layer(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = _zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1))
        self.skip_connection = (
            nn.Conv3d(channels, self.out_channels, 1)
            if channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip_connection(x)


class _DownsampleBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 2, stride=2)

    def forward(self, x):
        return self.conv(x)


class _UpsampleBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch * 8, 3, padding=1)

    def forward(self, x):
        return _pixel_shuffle_3d(self.conv(x), 2)


class SparseStructureEncoder(nn.Module):
    """VQVAE Encoder: [B,1,64,64,64] → [B,8,16,16,16]"""

    def __init__(self, in_channels=1, latent_channels=8, num_res_blocks=2,
                 channels=(32, 128, 512), num_res_blocks_middle=2, norm_type="layer"):
        super().__init__()
        channels = list(channels)
        self.input_layer = nn.Conv3d(in_channels, channels[0], 3, padding=1)
        self.blocks = nn.ModuleList()
        for i, ch in enumerate(channels):
            for _ in range(num_res_blocks):
                self.blocks.append(_ResBlock3d(ch, ch))
            if i < len(channels) - 1:
                self.blocks.append(_DownsampleBlock3d(ch, channels[i + 1]))
        self.middle_block = nn.Sequential(
            *[_ResBlock3d(channels[-1]) for _ in range(num_res_blocks_middle)]
        )
        self.out_layer = nn.Sequential(
            _norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], latent_channels * 2, 3, padding=1),
        )

    def forward(self, x, using_out_layer=True):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        h = self.middle_block(h)
        if not using_out_layer:
            return h  # [B, 512, 16, 16, 16]
        h = self.out_layer(h)
        mean, _ = h.chunk(2, dim=1)
        return mean  # [B, 8, 16, 16, 16]


class SparseStructureDecoder(nn.Module):
    """VQVAE Decoder: [B,8,16,16,16] → [B,1,64,64,64]"""

    def __init__(self, out_channels=1, latent_channels=8, num_res_blocks=2,
                 channels=(512, 128, 32), num_res_blocks_middle=2, norm_type="layer"):
        super().__init__()
        channels = list(channels)
        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)
        self.middle_block = nn.Sequential(
            *[_ResBlock3d(channels[0]) for _ in range(num_res_blocks_middle)]
        )
        self.blocks = nn.ModuleList()
        for i, ch in enumerate(channels):
            for _ in range(num_res_blocks):
                self.blocks.append(_ResBlock3d(ch, ch))
            if i < len(channels) - 1:
                self.blocks.append(_UpsampleBlock3d(ch, channels[i + 1]))
        self.out_layer = nn.Sequential(
            _norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1),
        )

    def forward(self, x, using_input_layer=True):
        h = self.input_layer(x) if using_input_layer else x
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)
        h = self.out_layer(h)
        return h


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=8192, embedding_dim=32, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(
            -1 / num_embeddings, 1 / num_embeddings
        )

    def forward(self, z, only_return_indices=False):
        bs, h, w, d, c = z.shape
        z_flat = z.reshape(-1, self.embedding_dim)
        distances = torch.cdist(z_flat, self.embeddings.weight)
        encoding_indices = torch.argmin(distances, dim=1)
        if only_return_indices:
            return encoding_indices.view(bs, h * w * d)
        quantized = self.embeddings(encoding_indices).view(bs, h, w, d, c)
        encoding_indices = encoding_indices.view(bs, h, w, d)
        commitment_loss = F.mse_loss(z, quantized.detach())
        vq_loss = F.mse_loss(quantized, z.detach())
        quantized = z + (quantized - z).detach()
        return quantized, vq_loss, commitment_loss, encoding_indices


class VQVAE3D(nn.Module):
    """
    3D VQ-VAE for sparse structure encoding.
    Input:  binary occupancy [B, 1, 64, 64, 64]
    Encode: → 1024 discrete token IDs (codebook size 8192, dim 32)
    Decode: → reconstructed occupancy [B, 1, 64, 64, 64]
    """

    def __init__(self, num_embeddings=8192):
        super().__init__()
        self.Encoder = SparseStructureEncoder(
            in_channels=1, latent_channels=8, num_res_blocks=2,
            channels=[32, 128, 512], num_res_blocks_middle=2,
        )
        self.Decoder = SparseStructureDecoder(
            out_channels=1, latent_channels=8, num_res_blocks=2,
            channels=[512, 128, 32], num_res_blocks_middle=2,
        )
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=32, beta=0.25
        )

    def Encode(self, x):
        """x: [B, 1, 64, 64, 64] → encoding_indices: [B, 1024]"""
        bs = x.shape[0]
        z = self.Encoder(x)  # [B, 8, 16, 16, 16]
        z = z.permute(0, 2, 3, 4, 1).contiguous()  # [B, 16, 16, 16, 8]
        z = z.view(bs, 8, 8, 16, 32)
        encoding_indices = self.vq(z, only_return_indices=True)
        return encoding_indices  # [B, 1024]

    def Decode(self, encoding_indices):
        """encoding_indices: [B, 1024] → recon: [B, 1, 64, 64, 64]"""
        bs = encoding_indices.shape[0]
        quantized = self.vq.embeddings(encoding_indices)  # [B, 1024, 32]
        quantized = quantized.view(bs, 8, 8, 16, 32)
        z_hat = quantized.view(bs, 16, 16, 16, 8)
        z_hat = z_hat.permute(0, 4, 1, 2, 3).contiguous()  # [B, 8, 16, 16, 16]
        return self.Decoder(z_hat)

    def decode_with_intermediate(self, encoding_indices):
        """
        Decode and also return intermediate 3D feature map from decoder.
        Returns:
            recon: [B, 1, 64, 64, 64]
            intermediate_feat: [B, 32, 64, 64, 64] (after last upsample, before out_layer)
        """
        bs = encoding_indices.shape[0]
        quantized = self.vq.embeddings(encoding_indices)
        quantized = quantized.view(bs, 8, 8, 16, 32)
        z_hat = quantized.view(bs, 16, 16, 16, 8)
        z_hat = z_hat.permute(0, 4, 1, 2, 3).contiguous()

        # Manual forward through decoder to capture intermediate features
        decoder = self.Decoder
        h = decoder.input_layer(z_hat)
        h = decoder.middle_block(h)
        for block in decoder.blocks:
            h = block(h)
        # h is now [B, 32, 64, 64, 64] — the intermediate feature
        intermediate_feat = h.clone()
        h = decoder.out_layer(h)
        return h, intermediate_feat

    def encode_with_multiscale(self, x):
        """
        Encode and return multi-scale intermediate 3D feature maps from encoder.

        Encoder blocks layout (channels=(32, 128, 512), num_res_blocks=2):
          blocks[0..1]: ResBlock(32)   at 64³
          blocks[2]:    Downsample 64³→32³
          blocks[3..4]: ResBlock(128)  at 32³
          blocks[5]:    Downsample 32³→16³
          blocks[6..7]: ResBlock(512)  at 16³

        Args:
            x: [B, 1, 64, 64, 64] — binary occupancy

        Returns:
            multiscale_feats: dict with keys:
                64: [B,  32, 64, 64, 64]  — after ResBlocks at 64³
                32: [B, 128, 32, 32, 32]  — after ResBlocks at 32³
                16: [B, 512, 16, 16, 16]  — after ResBlocks + middle_block at 16³
        """
        encoder = self.Encoder
        h = encoder.input_layer(x)          # [B, 32, 64, 64, 64]

        # blocks[0], blocks[1]: ResBlock(32) at 64³
        h = encoder.blocks[0](h)
        h = encoder.blocks[1](h)
        feat_64 = h                         # [B, 32, 64, 64, 64]

        # blocks[2]: Downsample 64³→32³
        h = encoder.blocks[2](h)
        # blocks[3], blocks[4]: ResBlock(128) at 32³
        h = encoder.blocks[3](h)
        h = encoder.blocks[4](h)
        feat_32 = h                         # [B, 128, 32, 32, 32]

        # blocks[5]: Downsample 32³→16³
        h = encoder.blocks[5](h)
        # blocks[6], blocks[7]: ResBlock(512) at 16³
        h = encoder.blocks[6](h)
        h = encoder.blocks[7](h)
        h = encoder.middle_block(h)
        feat_16 = h                         # [B, 512, 16, 16, 16]

        return {64: feat_64, 32: feat_32, 16: feat_16}

    def decode_with_multiscale(self, encoding_indices):
        """
        Decode and return multi-scale intermediate 3D feature maps from decoder.

        Decoder blocks layout (channels=(512, 128, 32), num_res_blocks=2):
          blocks[0..1]: ResBlock(512)  at 16³
          blocks[2]:    Upsample 16³→32³
          blocks[3..4]: ResBlock(128)  at 32³
          blocks[5]:    Upsample 32³→64³
          blocks[6..7]: ResBlock(32)   at 64³

        Returns:
            multiscale_feats: dict with keys:
                16: [B, 512, 16, 16, 16]  — after middle_block + ResBlocks
                32: [B, 128, 32, 32, 32]  — after first upsample + ResBlocks
                64: [B,  32, 64, 64, 64]  — after second upsample + ResBlocks
        """
        bs = encoding_indices.shape[0]
        quantized = self.vq.embeddings(encoding_indices)
        quantized = quantized.view(bs, 8, 8, 16, 32)
        z_hat = quantized.view(bs, 16, 16, 16, 8)
        z_hat = z_hat.permute(0, 4, 1, 2, 3).contiguous()

        decoder = self.Decoder
        h = decoder.input_layer(z_hat)       # [B, 512, 16, 16, 16]
        h = decoder.middle_block(h)          # [B, 512, 16, 16, 16]

        # blocks[0], blocks[1]: ResBlock(512) at 16³
        h = decoder.blocks[0](h)
        h = decoder.blocks[1](h)
        feat_16 = h                          # [B, 512, 16, 16, 16]

        # blocks[2]: Upsample 16³→32³
        h = decoder.blocks[2](h)
        # blocks[3], blocks[4]: ResBlock(128) at 32³
        h = decoder.blocks[3](h)
        h = decoder.blocks[4](h)
        feat_32 = h                          # [B, 128, 32, 32, 32]

        # blocks[5]: Upsample 32³→64³
        h = decoder.blocks[5](h)
        # blocks[6], blocks[7]: ResBlock(32) at 64³
        h = decoder.blocks[6](h)
        h = decoder.blocks[7](h)
        feat_64 = h                          # [B, 32, 64, 64, 64]

        return {16: feat_16, 32: feat_32, 64: feat_64}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def pointcloud_to_voxel(points, resolution=64):
    """
    Convert point cloud(s) to binary occupancy voxel grid.

    Args:
        points: (B, N, 3) or list of (N, 3) tensors, coordinates in [-0.5, 0.5]
        resolution: voxel grid resolution (default 64)

    Returns:
        voxel: (B, 1, R, R, R) float tensor, binary occupancy
    """
    if isinstance(points, (list, tuple)):
        points = torch.stack(points)  # (B, N, 3)

    device = points.device
    bs = points.shape[0]

    # Clamp to valid range
    points_clamped = points.clamp(-0.5 + 1e-6, 0.5 - 1e-6)

    # Map [-0.5, 0.5] → [0, resolution-1]
    coords = ((points_clamped + 0.5) * resolution).long()
    coords = coords.clamp(0, resolution - 1)

    voxel = torch.zeros(bs, 1, resolution, resolution, resolution,
                        dtype=torch.float32, device=device)
    for i in range(bs):
        voxel[i, 0, coords[i, :, 0], coords[i, :, 1], coords[i, :, 2]] = 1.0

    return voxel


def vqvae_tokenize(vqvae, points, resolution=64):
    """
    End-to-end: point cloud → voxelization → VQVAE encode → 1024 discrete tokens.

    Args:
        vqvae: VQVAE3D model (frozen)
        points: (B, N, 3) tensor, coordinates in [-0.5, 0.5]
        resolution: voxel resolution (default 64)

    Returns:
        token_ids: (B, 1024) long tensor, each in [0, num_embeddings-1]
    """
    voxel = pointcloud_to_voxel(points, resolution)
    with torch.no_grad():
        token_ids = vqvae.Encode(voxel)
    return token_ids


def sample_voxel_features(voxel_feat, point_coords):
    """
    Trilinear interpolation: sample 3D feature volume at point locations.

    Args:
        voxel_feat: (B, C, D, H, W) — 3D feature map
        point_coords: (B, N, 3) — point coordinates in [-0.5, 0.5]

    Returns:
        sampled: (B, N, C) — per-point features
    """
    # grid_sample expects coordinates in [-1, 1]
    grid = point_coords * 2.0  # [-0.5, 0.5] → [-1, 1]
    # grid_sample expects (B, D_out, H_out, W_out, 3), we use (B, 1, 1, N, 3)
    grid = grid.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N, 3)
    sampled = F.grid_sample(
        voxel_feat, grid, mode="bilinear", align_corners=True, padding_mode="zeros"
    )
    # sampled: (B, C, 1, 1, N)
    sampled = sampled.squeeze(2).squeeze(2)  # (B, C, N)
    return sampled.permute(0, 2, 1)  # (B, N, C)


def point_mask_to_voxel(gt_mask, point_coords, resolution=64):
    """
    Convert point-level GT mask to voxel-level GT mask.
    For scheme 4 — voxel affordance supervision.

    Args:
        gt_mask: (B, 1, N) — binary ground truth mask per point
        point_coords: (B, N, 3) — point coordinates in [-0.5, 0.5]
        resolution: voxel resolution

    Returns:
        voxel_mask: (B, 1, R, R, R) — binary voxel mask
    """
    device = gt_mask.device
    bs = gt_mask.shape[0]
    N = point_coords.shape[1]

    coords = ((point_coords.clamp(-0.5 + 1e-6, 0.5 - 1e-6) + 0.5) * resolution).long()
    coords = coords.clamp(0, resolution - 1)

    voxel_mask = torch.zeros(bs, 1, resolution, resolution, resolution,
                             dtype=torch.float32, device=device)
    mask_flat = gt_mask.reshape(bs, N)  # (B, N) — robust to extra dims

    for i in range(bs):
        positive_idx = mask_flat[i] > 0.5
        if positive_idx.any():
            pos_coords = coords[i][positive_idx]
            voxel_mask[i, 0, pos_coords[:, 0], pos_coords[:, 1], pos_coords[:, 2]] = 1.0

    return voxel_mask


def load_vqvae(checkpoint_path, num_embeddings=8192, device="cpu"):
    """
    Load pretrained VQVAE3D model.

    Args:
        checkpoint_path: path to the .bin or .pth file
        num_embeddings: codebook size
        device: target device

    Returns:
        vqvae: VQVAE3D model in eval mode, frozen
    """
    vqvae = VQVAE3D(num_embeddings=num_embeddings)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    vqvae.load_state_dict(state_dict)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False
    return vqvae.to(device)
