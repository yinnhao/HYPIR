from typing import Literal, List, overload

import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
import concurrent.futures
from threading import Lock
import math

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
        # 优化参数
        vae_batch_size=4,
        enable_fast_vae=False,
        generator_batch_size=2,
        enable_amp=False,
        encoder_batch_size=8,  # 新增：encoder批量大小
        enable_parallel_encode=True,  # 新增：启用并行编码
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
        self.encoder_batch_size = encoder_batch_size
        self.enable_parallel_encode = enable_parallel_encode

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

        # 优化的 VAE encoding
        lq = (lq * 2 - 1).to(dtype=self.weight_dtype, device=self.device)
        h1, w1 = lq.shape[2:]
        
        # Pad vae input size to multiples of vae_scale_factor
        vae_scale_factor = 8
        ph = (h1 + vae_scale_factor - 1) // vae_scale_factor * vae_scale_factor - h1
        pw = (w1 + vae_scale_factor - 1) // vae_scale_factor * vae_scale_factor - w1
        lq = F.pad(lq, (0, pw, 0, ph), mode="constant", value=0)
        
        # 选择编码方式：并行批量 vs 原始tiled
        if self.enable_parallel_encode and max(h1, w1) > patch_size:
            print(f"[Optimized VAE]: Using parallel batch encoding with batch_size={self.encoder_batch_size}")
            z_lq = self.parallel_vae_encode(lq, patch_size, self.encoder_batch_size)
        else:
            # 原始方式，但启用快速编码
            with enable_tiled_vae(
                self.vae,
                is_decoder=False,
                tile_size=patch_size,
                dtype=self.weight_dtype,
                fast_encoder=self.enable_fast_vae,  # 启用快速编码
            ):
                z_lq = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample().to(self.weight_dtype)

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

    def parallel_vae_encode(self, lq: torch.Tensor, patch_size: int, batch_size: int) -> torch.Tensor:
        """
        并行批量VAE编码 - 这是加速的关键！
        """
        B, C, H, W = lq.shape
        device = lq.device
        dtype = lq.dtype
        
        # 如果图像较小，直接编码
        if max(H, W) <= patch_size:
            return self.vae.encode(lq).latent_dist.sample().to(self.weight_dtype)
        
        # 计算padding以确保完美重建
        pad = 32  # VAE encoder需要的padding
        stride = patch_size - pad * 2  # 有效stride
        
        # 分割图像为patches
        patches = []
        patch_coords = []
        
        print(f"[Parallel VAE Encode]: Splitting {H}x{W} image into patches...")
        
        # 生成所有patch坐标
        h_positions = list(range(0, H - patch_size + 1, stride))
        w_positions = list(range(0, W - patch_size + 1, stride))
        
        # 确保覆盖边缘
        if h_positions[-1] + patch_size < H:
            h_positions.append(H - patch_size)
        if w_positions[-1] + patch_size < W:
            w_positions.append(W - patch_size)
        
        # 提取所有patches
        for h_start in h_positions:
            for w_start in w_positions:
                h_end = min(h_start + patch_size, H)
                w_end = min(w_start + patch_size, W)
                
                # 提取patch
                patch = lq[:, :, h_start:h_end, w_start:w_end]
                patches.append(patch)
                patch_coords.append((h_start, h_end, w_start, w_end))
        
        print(f"[Parallel VAE Encode]: Processing {len(patches)} patches in batches of {batch_size}")
        
        # 批量编码patches
        encoded_patches = []
        from tqdm import tqdm
        
        for i in tqdm(range(0, len(patches), batch_size), desc="VAE Encoding Batches"):
            batch_patches = patches[i:i+batch_size]
            
            # 将patches堆叠成batch
            if len(batch_patches) == 1:
                batch_input = batch_patches[0]
            else:
                # 处理不同大小的patches
                max_h = max(p.shape[2] for p in batch_patches)
                max_w = max(p.shape[3] for p in batch_patches)
                
                # Pad所有patches到相同大小
                padded_patches = []
                for patch in batch_patches:
                    ph = max_h - patch.shape[2]
                    pw = max_w - patch.shape[3]
                    if ph > 0 or pw > 0:
                        patch = F.pad(patch, (0, pw, 0, ph), mode='constant', value=0)
                    padded_patches.append(patch)
                
                batch_input = torch.cat(padded_patches, dim=0)
            
            # 批量编码
            with torch.cuda.amp.autocast() if self.enable_amp else torch.no_grad():
                encoded_batch = self.vae.encode(batch_input.to(self.weight_dtype)).latent_dist.sample().to(self.weight_dtype)
            
            # 分离batch结果
            if len(batch_patches) == 1:
                encoded_patches.append(encoded_batch)
            else:
                batch_results = encoded_batch.chunk(len(batch_patches), dim=0)
                
                # 移除padding（如果有的话）
                for j, (result, original_patch) in enumerate(zip(batch_results, batch_patches)):
                    orig_h, orig_w = original_patch.shape[2] // 8, original_patch.shape[3] // 8  # VAE scale factor
                    result = result[:, :, :orig_h, :orig_w]
                    encoded_patches.append(result)
        
        # 重构完整的latent tensor
        return self.reconstruct_encoded_image(encoded_patches, patch_coords, H // 8, W // 8, B)

    def reconstruct_encoded_image(self, encoded_patches, patch_coords, target_h, target_w, batch_size):
        """重构编码后的完整图像"""
        device = encoded_patches[0].device
        C = encoded_patches[0].shape[1]
        
        # 创建输出tensor和权重tensor
        reconstructed = torch.zeros((batch_size, C, target_h, target_w), device=device, dtype=encoded_patches[0].dtype)
        weights = torch.zeros((batch_size, C, target_h, target_w), device=device, dtype=torch.float32)
        
        # 创建高斯权重用于smooth blending
        patch_size_latent = encoded_patches[0].shape[2]
        gaussian_weights = self.create_gaussian_weights(patch_size_latent, patch_size_latent).to(device)
        
        # 将每个patch加回到重构图像中
        for patch, (h_start, h_end, w_start, w_end) in zip(encoded_patches, patch_coords):
            # 转换到latent空间坐标
            lh_start, lh_end = h_start // 8, h_end // 8
            lw_start, lw_end = w_start // 8, w_end // 8
            
            # 确保坐标在范围内
            lh_end = min(lh_end, target_h)
            lw_end = min(lw_end, target_w)
            
            # 调整patch大小以匹配目标区域
            patch_h, patch_w = lh_end - lh_start, lw_end - lw_start
            if patch.shape[2] != patch_h or patch.shape[3] != patch_w:
                patch = F.interpolate(patch, size=(patch_h, patch_w), mode='bilinear', align_corners=False)
            
            # 调整权重大小
            current_weights = F.interpolate(
                gaussian_weights.unsqueeze(0).unsqueeze(0),  # 添加两个维度：(1, 1, H, W)
                size=(patch_h, patch_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0).squeeze(0)  # 移除两个维度：(H, W)
            
            # 累加patch和权重
            reconstructed[:, :, lh_start:lh_end, lw_start:lw_end] += patch * current_weights
            weights[:, :, lh_start:lh_end, lw_start:lw_end] += current_weights
        
        # 归一化
        weights = torch.clamp(weights, min=1e-8)  # 避免除零
        reconstructed = reconstructed / weights
        
        return reconstructed

    def create_gaussian_weights(self, height, width, sigma=None):
        """创建高斯权重用于patch blending"""
        if sigma is None:
            sigma = min(height, width) / 6.0
        
        center_h, center_w = height // 2, width // 2
        h_coords, w_coords = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing='ij'
        )
        
        distances = (h_coords - center_h) ** 2 + (w_coords - center_w) ** 2
        weights = torch.exp(-distances / (2 * sigma ** 2))
        
        return weights

    def smart_batch_size_selection(self, total_patches: int, patch_memory_mb: float) -> int:
        """动态选择最优批量大小"""
        if not torch.cuda.is_available():
            return min(4, total_patches)
        
        # 获取GPU内存信息
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        available_memory = total_memory - torch.cuda.memory_allocated(self.device)
        available_memory_mb = available_memory / (1024 * 1024)
        
        # 保留30%内存作为缓冲
        usable_memory_mb = available_memory_mb * 0.7
        
        # 计算最大可能的批量大小
        max_possible_batch = int(usable_memory_mb // patch_memory_mb)
        
        # 限制在合理范围内
        optimal_batch = min(max_possible_batch, self.encoder_batch_size, total_patches)
        optimal_batch = max(1, optimal_batch)  # 至少为1
        
        print(f"[Memory Management]: Available {available_memory_mb:.0f}MB, "
              f"using batch size {optimal_batch} for {total_patches} patches")
        
        return optimal_batch

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
