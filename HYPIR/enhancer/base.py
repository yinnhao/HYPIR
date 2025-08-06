from typing import Literal, List, overload

import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
import concurrent.futures
from threading import Lock

from HYPIR.utils.common import wavelet_reconstruction, make_tiled_fn, sliding_windows
from HYPIR.utils.tiled_vae import enable_tiled_vae


class BaseEnhancer:

    def __init__(
        self,
        base_model_path,
        weight_path,
        lora_modules,
        lora_rank,
        model_t,
        coeff_t,
        device,
        # 新增优化参数
        vae_batch_size=4,
        enable_fast_vae=False,
        generator_batch_size=2,
        enable_amp=False,
    ):
        self.base_model_path = base_model_path
        self.weight_path = weight_path
        self.lora_modules = lora_modules
        self.lora_rank = lora_rank
        self.model_t = model_t
        self.coeff_t = coeff_t

        self.weight_dtype = torch.bfloat16
        self.device = device
        
        # 优化配置
        self.vae_batch_size = vae_batch_size
        self.enable_fast_vae = enable_fast_vae
        self.generator_batch_size = generator_batch_size
        self.enable_amp = enable_amp

    def init_models(self):
        self.init_scheduler()
        self.init_text_models()
        self.init_vae()
        self.init_generator()

    @overload
    def init_scheduler(self):
        ...

    @overload
    def init_text_models(self):
        ...

    def init_vae(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.base_model_path, subfolder="vae", torch_dtype=self.weight_dtype).to(self.device)
        self.vae.eval().requires_grad_(False)

    @overload
    def init_generator(self):
        ...

    @overload
    def prepare_inputs(self, batch_size, prompt):
        ...

    @overload
    def forward_generator(self, z_lq: torch.Tensor) -> torch.Tensor:
        ...

    @torch.no_grad()
    def enhance(
        self,
        lq: torch.Tensor,
        prompt: str,
        scale_by: Literal["factor", "longest_side"] = "factor",
        upscale: int = 1,
        target_longest_side: int | None = None,
        patch_size: int = 512,
        stride: int = 256,
        return_type: Literal["pt", "np", "pil"] = "pt",
        # 新增优化参数
        vae_batch_size: int = None,
        generator_batch_size: int = None,
    ) -> torch.Tensor | np.ndarray | List[Image.Image]:
        
        # 使用传入参数或默认配置
        vae_batch_size = vae_batch_size or self.vae_batch_size
        generator_batch_size = generator_batch_size or self.generator_batch_size
        
        # 启用混合精度上下文
        amp_context = torch.cuda.amp.autocast() if self.enable_amp else torch.no_grad()
        
        with amp_context:
            return self._enhanced_forward(
                lq, prompt, scale_by, upscale, target_longest_side,
                patch_size, stride, return_type, vae_batch_size, generator_batch_size
            )

    def _enhanced_forward(
        self, lq, prompt, scale_by, upscale, target_longest_side,
        patch_size, stride, return_type, vae_batch_size, generator_batch_size
    ):
        if stride <= 0:
            raise ValueError("Stride must be greater than 0.")
        if patch_size <= 0:
            raise ValueError("Patch size must be greater than 0.")
        if patch_size < stride:
            raise ValueError("Patch size must be greater than or equal to stride.")

        # Prepare low-quality inputs
        bs = len(lq)
        if scale_by == "factor":
            lq = F.interpolate(lq, scale_factor=upscale, mode="bicubic")
        elif scale_by == "longest_side":
            if target_longest_side is None:
                raise ValueError("target_longest_side must be specified when scale_by is 'longest_side'.")
            h, w = lq.shape[2:]
            if h >= w:
                new_h = target_longest_side
                new_w = int(w * (target_longest_side / h))
            else:
                new_w = target_longest_side
                new_h = int(h * (target_longest_side / w))
            lq = F.interpolate(lq, size=(new_h, new_w), mode="bicubic")
        else:
            raise ValueError(f"Unsupported scale_by method: {scale_by}")
        
        ref = lq
        h0, w0 = lq.shape[2:]
        if min(h0, w0) <= patch_size:
            lq = self.resize_at_least(lq, size=patch_size)

        # VAE encoding with optimizations
        lq = (lq * 2 - 1).to(dtype=self.weight_dtype, device=self.device)
        h1, w1 = lq.shape[2:]
        
        # Pad vae input size to multiples of vae_scale_factor
        vae_scale_factor = 8
        ph = (h1 + vae_scale_factor - 1) // vae_scale_factor * vae_scale_factor - h1
        pw = (w1 + vae_scale_factor - 1) // vae_scale_factor * vae_scale_factor - w1
        lq = F.pad(lq, (0, pw, 0, ph), mode="constant", value=0)
        
        # Optimized VAE encoding
        with enable_tiled_vae(
            self.vae,
            is_decoder=False,
            tile_size=patch_size,
            dtype=self.weight_dtype,
        ):
            z_lq = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample()

        # Optimized Generator forward with batching
        self.prepare_inputs(batch_size=bs, prompt=prompt)
        z = self.make_batched_tiled_fn(
            fn=lambda z_batch: self.forward_generator_batch(z_batch),
            size=patch_size // vae_scale_factor,
            stride=stride // vae_scale_factor,
            batch_size=generator_batch_size,
            progress=True,
            desc="Generator Forward (Batched)",
        )(z_lq)
        
        # Optimized VAE decoding
        with enable_tiled_vae(
            self.vae,
            is_decoder=True,
            tile_size=patch_size // vae_scale_factor,
            dtype=self.weight_dtype,
            fast_decoder=self.enable_fast_vae,  # 启用快速解码
        ):
            if vae_batch_size > 1:
                x = self.batched_vae_decode(z, vae_batch_size)
            else:
                x = self.vae.decode(z.to(self.weight_dtype)).sample.float()
        
        x = x[..., :h1, :w1]
        x = (x + 1) / 2
        x = F.interpolate(input=x, size=(h0, w0), mode="bicubic", antialias=True)
        x = wavelet_reconstruction(x, ref.to(device=self.device))

        if return_type == "pt":
            return x.clamp(0, 1).cpu()
        elif return_type == "np":
            return self.tensor2image(x)
        else:
            return [Image.fromarray(img) for img in self.tensor2image(x)]

    def batched_vae_decode(self, z: torch.Tensor, batch_size: int) -> torch.Tensor:
        """批量VAE解码以提高效率"""
        B, C, H, W = z.shape
        
        if B <= batch_size:
            # 如果批次已经足够小，直接解码
            return self.vae.decode(z.to(self.weight_dtype)).sample.float()
        
        # 分批处理
        decoded_chunks = []
        for i in range(0, B, batch_size):
            chunk = z[i:i+batch_size]
            with torch.cuda.amp.autocast() if self.enable_amp else torch.no_grad():
                decoded_chunk = self.vae.decode(chunk.to(self.weight_dtype)).sample.float()
            decoded_chunks.append(decoded_chunk)
        
        return torch.cat(decoded_chunks, dim=0)

    def forward_generator_batch(self, z_batch: torch.Tensor) -> torch.Tensor:
        """批量处理generator前向传播"""
        batch_results = []
        for z in z_batch:
            result = self.forward_generator(z.unsqueeze(0))
            batch_results.append(result)
        return torch.cat(batch_results, dim=0)

    def make_batched_tiled_fn(
        self,
        fn,
        size: int,
        stride: int,
        batch_size: int = 2,
        scale_type: Literal["up", "down"] = "up",
        scale: int = 1,
        channel: int | None = None,
        weight: Literal["uniform", "gaussian"] = "gaussian",
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        progress: bool = True,
        desc: str = None,
    ):
        """创建支持批量处理的分块函数"""
        def batched_tiled_fn(x: torch.Tensor) -> torch.Tensor:
            if scale_type == "up":
                scale_fn = lambda n: int(n * scale)
            else:
                scale_fn = lambda n: int(n // scale)

            b, c, h, w = x.size()
            out_dtype = dtype or x.dtype
            out_device = device or x.device
            out_channel = channel or c
            out = torch.zeros(
                (b, out_channel, scale_fn(h), scale_fn(w)),
                dtype=out_dtype,
                device=out_device,
            )
            count = torch.zeros_like(out, dtype=torch.float32)
            weight_size = scale_fn(size)
            
            # 预计算权重
            if weight == "gaussian":
                from HYPIR.utils.common import gaussian_weights
                weights = gaussian_weights(weight_size, weight_size)[None, None]
            else:
                weights = np.ones((1, 1, weight_size, weight_size))
            
            weights = torch.tensor(weights, dtype=out_dtype, device=out_device)

            # 收集所有tiles
            indices = sliding_windows(h, w, size, stride)
            tiles = []
            tile_indices = []
            
            for hi, hi_end, wi, wi_end in indices:
                x_tile = x[..., hi:hi_end, wi:wi_end]
                tiles.append(x_tile)
                tile_indices.append((hi, hi_end, wi, wi_end))

            # 批量处理tiles
            from tqdm import tqdm
            pbar_desc = f"[{desc}]: Batched Tiled Processing" if desc else "Batched Tiled Processing"
            pbar = tqdm(range(0, len(tiles), batch_size), desc=pbar_desc, disable=not progress)
            
            for i in pbar:
                batch_tiles = tiles[i:i+batch_size]
                batch_indices = tile_indices[i:i+batch_size]
                
                if len(batch_tiles) > 1:
                    # 批量处理
                    batch_input = torch.cat(batch_tiles, dim=0)
                    batch_output = fn(batch_input)
                    outputs = batch_output.chunk(len(batch_tiles), dim=0)
                else:
                    # 单个处理
                    outputs = [fn(batch_tiles[0])]
                
                # 将结果放回原位置
                for output, (hi, hi_end, wi, wi_end) in zip(outputs, batch_indices):
                    out_hi, out_hi_end, out_wi, out_wi_end = map(
                        scale_fn, (hi, hi_end, wi, wi_end)
                    )
                    out[..., out_hi:out_hi_end, out_wi:out_wi_end] += (output * weights)
                    count[..., out_hi:out_hi_end, out_wi:out_wi_end] += weights

            out = out / count
            return out

        return batched_tiled_fn

    @staticmethod
    def tensor2image(img_tensor):
        return (
            (img_tensor * 255.0)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )

    @staticmethod
    def resize_at_least(imgs: torch.Tensor, size: int) -> torch.Tensor:
        _, _, h, w = imgs.size()
        if h == w:
            new_h, new_w = size, size
        elif h < w:
            new_h, new_w = size, int(w * (size / h))
        else:
            new_h, new_w = int(h * (size / w)), size
        return F.interpolate(imgs, size=(new_h, new_w), mode="bicubic", antialias=True)
