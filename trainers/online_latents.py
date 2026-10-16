"""Frozen voxel / pixel VAE encoders run inside the training step, replacing cached latents on disk.

Produces exactly what MinetestLatentCameraAction returns in cached mode: deterministic posterior means
(same as dataset_toolkits/encode_{voxel,pixel}_latents.py), normalized with the dataset's latent_stats.npz.
The encoders are never registered in trainer.models, so they are excluded from the optimizer, EMA,
gradient clipping and checkpoints.
"""
from dataclasses import asdict

import safetensors.torch
import torch
from einops import rearrange
from loguru import logger

from models import vae_pixel, vae_voxel
from pipelines.persist_S_pipeline.model_configs import ResNetEncoderArgs, ViTVaeArgs


def _load_frozen(model, checkpoint_path, strip_prefixes):
    checkpoint = safetensors.torch.load_file(checkpoint_path)
    for prefix in strip_prefixes:
        checkpoint = {k.removeprefix(prefix): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.eval()
    model.requires_grad_(False)
    return model


class OnlineLatentEncoder:

    def __init__(self, voxel_encoder_checkpoint, pixel_vae_checkpoint, normalization, device,
                 batch_size_voxel=2, batch_size_pixel=16):
        # batch_size_voxel: the encoder one-hots each 48^3 grid to 2138 fp32 channels (~1 GB/frame), so keep it
        # small -- peak while encoding 16 frames is ~4.7 GB at 2 vs ~15 GB at 8 (measured on A100)
        self.voxel_encoder = _load_frozen(
            vae_voxel.ResNet3dEncoder(**asdict(ResNetEncoderArgs())), voxel_encoder_checkpoint, ["_orig_mod."]
        ).to(device)
        self.pixel_vae = _load_frozen(
            vae_pixel.ViTVae(**asdict(ViTVaeArgs())), pixel_vae_checkpoint, ["_orig_mod.", "vae."]
        ).to(device)
        self.batch_size_voxel = batch_size_voxel
        self.batch_size_pixel = batch_size_pixel

        assert normalization['voxel_latent'] is not None and normalization['pixel_latent'] is not None, \
            "On-the-fly encoding needs latent_stats.npz in the dataset root."
        self.voxel_mean = normalization['voxel_latent']['mean'].view(1, 1, -1, 1, 1, 1).to(device)
        self.voxel_std = normalization['voxel_latent']['std'].view(1, 1, -1, 1, 1, 1).to(device)
        self.pixel_mean = normalization['pixel_latent']['mean'].view(1, 1, -1, 1, 1).to(device)
        self.pixel_std = normalization['pixel_latent']['std'].view(1, 1, -1, 1, 1).to(device)
        logger.info(f"Loaded frozen voxel encoder from {voxel_encoder_checkpoint} "
                    f"and pixel VAE from {pixel_vae_checkpoint} for on-the-fly latent encoding")

    @torch.no_grad()
    def encode_voxels(self, voxel_classes):
        """(B, T, X, Y, Z) class ids -> (B, T, C, x, y, z) normalized latents."""
        B, T = voxel_classes.shape[:2]
        flat = rearrange(voxel_classes.long(), "B T X Y Z -> (B T) X Y Z")
        z = torch.cat([self.voxel_encoder(flat[i:i + self.batch_size_voxel])
                       for i in range(0, len(flat), self.batch_size_voxel)])
        z = rearrange(z.float(), "(B T) C X Y Z -> B T C X Y Z", B=B, T=T)
        return (z - self.voxel_mean) / self.voxel_std

    @torch.no_grad()
    def encode_pixels(self, images):
        """(B, T, 3, H, W) uint8 (or float in [-1, 1]) -> (B, T, C, h, w) normalized latents."""
        if images.dtype == torch.uint8:
            images = images.float() / 255.0 * 2.0 - 1.0
        B, T = images.shape[:2]
        flat = rearrange(images, "B T C H W -> (B T) C H W")
        z = torch.cat([self.pixel_vae.encode(flat[i:i + self.batch_size_pixel]).mode()
                       for i in range(0, len(flat), self.batch_size_pixel)])
        z = rearrange(z.float(), "(B T) (H W) C -> B T C H W", B=B, T=T,
                      H=self.pixel_vae.seq_h, W=self.pixel_vae.seq_w)
        return (z - self.pixel_mean) / self.pixel_std

    def __call__(self, batch):
        """Fill batch['voxel'] / batch['pixel'] from batch['voxel_classes'] / batch['raw_images'], in place."""
        batch["voxel"] = self.encode_voxels(batch.pop("voxel_classes"))
        batch["pixel"] = self.encode_pixels(batch.pop("raw_images"))
        return batch
