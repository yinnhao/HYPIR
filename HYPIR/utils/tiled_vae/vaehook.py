# ------------------------------------------------------------------------
#
#   Ultimate VAE Tile Optimization
#
#   Introducing a revolutionary new optimization designed to make
#   the VAE work with giant images on limited VRAM!
#   Say goodbye to the frustration of OOM and hello to seamless output!
#
# ------------------------------------------------------------------------
#
#   This script is a wild hack that splits the image into tiles,
#   encodes each tile separately, and merges the result back together.
#
#   Advantages:
#   - The VAE can now work with giant images on limited VRAM
#       (~10 GB for 8K images!)
#   - The merged output is completely seamless without any post-processing.
#
#   Drawbacks:
#   - Giant RAM needed. To store the intermediate results for a 4096x4096
#       images, you need 32 GB RAM it consumes ~20GB); for 8192x8192
#       you need 128 GB RAM machine (it consumes ~100 GB)
#   - NaNs always appear in for 8k images when you use fp16 (half) VAE
#       You must use --no-half-vae to disable half VAE for that giant image.
#   - Slow speed. With default tile size, it takes around 50/200 seconds
#       to encode/decode a 4096x4096 image; and 200/900 seconds to encode/decode
#       a 8192x8192 image. (The speed is limited by both the GPU and the CPU.)
#   - The gradient calculation is not compatible with this hack. It
#       will break any backward() or torch.autograd.grad() that passes VAE.
#       (But you can still use the VAE to generate training data.)
#
#   How it works:
#   1) The image is split into tiles.
#       - To ensure perfect results, each tile is padded with 32 pixels
#           on each side.
#       - Then the conv2d/silu/upsample/downsample can produce identical
#           results to the original image without splitting.
#   2) The original forward is decomposed into a task queue and a task worker.
#       - The task queue is a list of functions that will be executed in order.
#       - The task worker is a loop that executes the tasks in the queue.
#   3) The task queue is executed for each tile.
#       - Current tile is sent to GPU.
#       - local operations are directly executed.
#       - Group norm calculation is temporarily suspended until the mean
#           and var of all tiles are calculated.
#       - The residual is pre-calculated and stored and addded back later.
#       - When need to go to the next tile, the current tile is send to cpu.
#   4) After all tiles are processed, tiles are merged on cpu and return.
#
#   Enjoy!
#
#   @author: LI YI @ Nanyang Technological University - Singapore
#   @date: 2023-03-02
#   @license: MIT License
#
#   Please give me a star if you like this project!
#
# -------------------------------------------------------------------------

import gc
from time import time
import math
from tqdm import tqdm

import torch
import torch.version
import torch.nn.functional as F
from einops import rearrange
import os
import sys
sys.path.append(os.getcwd())
import HYPIR.utils.tiled_vae.devices as devices

try:
    import xformers
    import xformers.ops
except ImportError:
    pass

sd_flag = False

def get_recommend_encoder_tile_size():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(
            devices.device).total_memory // 2**20
        if total_memory > 16*1000:
            ENCODER_TILE_SIZE = 3072
        elif total_memory > 12*1000:
            ENCODER_TILE_SIZE = 2048
        elif total_memory > 8*1000:
            ENCODER_TILE_SIZE = 1536
        else:
            ENCODER_TILE_SIZE = 960
    else:
        ENCODER_TILE_SIZE = 512
    return ENCODER_TILE_SIZE


def get_recommend_decoder_tile_size():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(
            devices.device).total_memory // 2**20
        if total_memory > 30*1000:
            DECODER_TILE_SIZE = 256
        elif total_memory > 16*1000:
            DECODER_TILE_SIZE = 192
        elif total_memory > 12*1000:
            DECODER_TILE_SIZE = 128
        elif total_memory > 8*1000:
            DECODER_TILE_SIZE = 96
        else:
            DECODER_TILE_SIZE = 64
    else:
        DECODER_TILE_SIZE = 64
    return DECODER_TILE_SIZE


if 'global const':
    DEFAULT_ENABLED = False
    DEFAULT_MOVE_TO_GPU = False
    DEFAULT_FAST_ENCODER = True
    DEFAULT_FAST_DECODER = True
    DEFAULT_COLOR_FIX = 0
    DEFAULT_ENCODER_TILE_SIZE = get_recommend_encoder_tile_size()
    DEFAULT_DECODER_TILE_SIZE = get_recommend_decoder_tile_size()


# inplace version of silu
def inplace_nonlinearity(x):
    # Test: fix for Nans
    return F.silu(x, inplace=True)

# extracted from ldm.modules.diffusionmodules.model

# from diffusers lib
def attn_forward_new(self, h_):
    batch_size, channel, height, width = h_.shape
    hidden_states = h_.view(batch_size, channel, height * width).transpose(1, 2)

    attention_mask = None
    encoder_hidden_states = None
    batch_size, sequence_length, _ = hidden_states.shape
    attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    query = self.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif self.norm_cross:
        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

    key = self.to_k(encoder_hidden_states)
    value = self.to_v(encoder_hidden_states)

    query = self.head_to_batch_dim(query)
    key = self.head_to_batch_dim(key)
    value = self.head_to_batch_dim(value)

    attention_probs = self.get_attention_scores(query, key, attention_mask)
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = self.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    return hidden_states

def attn_forward(self, h_):
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    # compute attention
    b, c, h, w = q.shape
    q = q.reshape(b, c, h*w)
    q = q.permute(0, 2, 1)   # b,hw,c
    k = k.reshape(b, c, h*w)  # b,c,hw
    w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
    w_ = w_ * (int(c)**(-0.5))
    w_ = torch.nn.functional.softmax(w_, dim=2)

    # attend to values
    v = v.reshape(b, c, h*w)
    w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
    # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
    h_ = torch.bmm(v, w_)
    h_ = h_.reshape(b, c, h, w)

    h_ = self.proj_out(h_)

    return h_


def xformer_attn_forward(self, h_):
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    # compute attention
    B, C, H, W = q.shape
    q, k, v = map(lambda x: rearrange(x, 'b c h w -> b (h w) c'), (q, k, v))

    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(B, t.shape[1], 1, C)
        .permute(0, 2, 1, 3)
        .reshape(B * 1, t.shape[1], C)
        .contiguous(),
        (q, k, v),
    )
    out = xformers.ops.memory_efficient_attention(
        q, k, v, attn_bias=None, op=self.attention_op)

    out = (
        out.unsqueeze(0)
        .reshape(B, 1, out.shape[1], C)
        .permute(0, 2, 1, 3)
        .reshape(B, out.shape[1], C)
    )
    out = rearrange(out, 'b (h w) c -> b c h w', b=B, h=H, w=W, c=C)
    out = self.proj_out(out)
    return out


def attn2task(task_queue, net):
    if False: #isinstance(net, AttnBlock):
        task_queue.append(('store_res', lambda x: x))
        task_queue.append(('pre_norm', net.norm))
        task_queue.append(('attn', lambda x, net=net: attn_forward(net, x)))
        task_queue.append(['add_res', None])
    elif False: #isinstance(net, MemoryEfficientAttnBlock):
        task_queue.append(('store_res', lambda x: x))
        task_queue.append(('pre_norm', net.norm))
        task_queue.append(
            ('attn', lambda x, net=net: xformer_attn_forward(net, x)))
        task_queue.append(['add_res', None])
    else:
        task_queue.append(('store_res', lambda x: x))
        task_queue.append(('pre_norm', net.group_norm))
        task_queue.append(('attn', lambda x, net=net: attn_forward_new(net, x)))
        task_queue.append(['add_res', None])

def resblock2task(queue, block):
    """
    Turn a ResNetBlock into a sequence of tasks and append to the task queue

    @param queue: the target task queue
    @param block: ResNetBlock

    """
    if block.in_channels != block.out_channels:
        if sd_flag:
            if block.use_conv_shortcut:
                queue.append(('store_res', block.conv_shortcut))
            else:
                queue.append(('store_res', block.nin_shortcut))
        else:
            if block.use_in_shortcut:
                queue.append(('store_res', block.conv_shortcut))
            else:
                queue.append(('store_res', block.nin_shortcut))

    else:
        queue.append(('store_res', lambda x: x))
    queue.append(('pre_norm', block.norm1))
    queue.append(('silu', inplace_nonlinearity))
    queue.append(('conv1', block.conv1))
    queue.append(('pre_norm', block.norm2))
    queue.append(('silu', inplace_nonlinearity))
    queue.append(('conv2', block.conv2))
    queue.append(['add_res', None])



def build_sampling(task_queue, net, is_decoder):
    """
    Build the sampling part of a task queue
    @param task_queue: the target task queue
    @param net: the network
    @param is_decoder: currently building decoder or encoder
    """
    if is_decoder:
        # resblock2task(task_queue, net.mid.block_1)
        # attn2task(task_queue, net.mid.attn_1)
        # resblock2task(task_queue, net.mid.block_2)
        # resolution_iter = reversed(range(net.num_resolutions))
        # block_ids = net.num_res_blocks + 1
        # condition = 0
        # module = net.up
        # func_name = 'upsample'
        resblock2task(task_queue, net.mid_block.resnets[0])
        attn2task(task_queue, net.mid_block.attentions[0])
        resblock2task(task_queue, net.mid_block.resnets[1])
        resolution_iter = (range(len(net.up_blocks)))  # range(0,4)
        block_ids = 2 + 1
        condition = len(net.up_blocks) - 1
        module = net.up_blocks
        func_name = 'upsamplers'
    else:
        # resolution_iter = range(net.num_resolutions)
        # block_ids = net.num_res_blocks
        # condition = net.num_resolutions - 1
        # module = net.down
        # func_name = 'downsample'
        resolution_iter = (range(len(net.down_blocks)))  # range(0,4)
        block_ids = 2
        condition = len(net.down_blocks) - 1
        module = net.down_blocks
        func_name = 'downsamplers'


    for i_level in resolution_iter:
        for i_block in range(block_ids):
            resblock2task(task_queue, module[i_level].resnets[i_block])
        if i_level != condition:
            if is_decoder:
                task_queue.append((func_name, module[i_level].upsamplers[0]))
            else:
                task_queue.append((func_name, module[i_level].downsamplers[0]))

    if not is_decoder:
        resblock2task(task_queue, net.mid_block.resnets[0])
        attn2task(task_queue, net.mid_block.attentions[0])
        resblock2task(task_queue, net.mid_block.resnets[1])


def build_task_queue(net, is_decoder):
    """
    Build a single task queue for the encoder or decoder
    @param net: the VAE decoder or encoder network
    @param is_decoder: currently building decoder or encoder
    @return: the task queue
    """
    task_queue = []
    task_queue.append(('conv_in', net.conv_in))

    # construct the sampling part of the task queue
    # because encoder and decoder share the same architecture, we extract the sampling part
    build_sampling(task_queue, net, is_decoder)
    if is_decoder and not sd_flag:
        net.give_pre_end = False
        net.tanh_out = False

    if not is_decoder or not net.give_pre_end:
        if sd_flag:
            task_queue.append(('pre_norm', net.norm_out))
        else:
            task_queue.append(('pre_norm', net.conv_norm_out))
        task_queue.append(('silu', inplace_nonlinearity))
        task_queue.append(('conv_out', net.conv_out))
        if is_decoder and net.tanh_out:
            task_queue.append(('tanh', torch.tanh))

    return task_queue


def clone_task_queue(task_queue):
    """
    Clone a task queue
    @param task_queue: the task queue to be cloned
    @return: the cloned task queue
    """
    return [[item for item in task] for task in task_queue]


def split_into_micro_stages(task_queue):
    """
    在每个 'pre_norm'（或 'apply_norm'）处分段，生成多个微阶段。
    保持算子顺序，仅在 Norm 前后切段；down/upsample、conv_out、tanh 也强制切段。
    返回：List[List[(name, fn_or_tensor)]]
    """
    stages = []
    current = []
    seen_norm_in_stage = False
    for op in task_queue:
        name = op[0]
        if name in ('downsamplers', 'upsamplers', 'conv_out', 'tanh'):
            current.append(op)
            if current:
                stages.append(current)
            current = []
            seen_norm_in_stage = False
            continue

        if name in ('pre_norm', 'apply_norm'):
            if seen_norm_in_stage:
                # 已经有一个 norm，先结束当前阶段，再把本 norm 放到新阶段开始
                if current:
                    stages.append(current)
                current = [op]
            else:
                current.append(op)
                seen_norm_in_stage = True
            continue

        # 其他普通算子/控制算子按序加入
        current.append(op)

    if current:
        stages.append(current)
    return stages


def _run_simple_ops(x, ops):
    for name, fn in ops:
        if name in ('store_res', 'store_res_cpu', 'add_res', 'pre_norm', 'apply_norm'):
            raise RuntimeError('control op should be handled outside of _run_simple_ops')
        x = fn(x)
    return x


def get_post_ops_after_first_norm(ops):
    """
    返回首次出现 'pre_norm' 或 'apply_norm' 之后的剩余算子序列。
    不包含该 norm 本身。
    """
    hit = False
    post = []
    for name, fn in ops:
        if not hit and name in ('pre_norm', 'apply_norm'):
            hit = True
            continue
        if hit:
            post.append((name, fn))
    return post

def get_var_mean(input, num_groups, eps=1e-6):
    """
    Get mean and var for group norm
    """
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c/num_groups)
    input_reshaped = input.contiguous().view(
        1, int(b * num_groups), channel_in_group, *input.size()[2:])
    var, mean = torch.var_mean(
        input_reshaped, dim=[0, 2, 3, 4], unbiased=False)
    return var, mean


def custom_group_norm(input, num_groups, mean, var, weight=None, bias=None, eps=1e-6):
    """
    Custom group norm with fixed mean and var

    @param input: input tensor
    @param num_groups: number of groups. by default, num_groups = 32
    @param mean: mean, must be pre-calculated by get_var_mean
    @param var: var, must be pre-calculated by get_var_mean
    @param weight: weight, should be fetched from the original group norm
    @param bias: bias, should be fetched from the original group norm
    @param eps: epsilon, by default, eps = 1e-6 to match the original group norm

    @return: normalized tensor
    """
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c/num_groups)
    input_reshaped = input.contiguous().view(
        1, int(b * num_groups), channel_in_group, *input.size()[2:])

    # Adapt running stats size: accept stats for one sample (num_groups) or full batch (b*num_groups)
    run_mean = mean
    run_var = var
    if run_mean.dim() != 1:
        run_mean = run_mean.reshape(-1)
    if run_var.dim() != 1:
        run_var = run_var.reshape(-1)
    expected_c = int(b * num_groups)
    if run_mean.shape[0] == num_groups:
        run_mean = run_mean.repeat(b)
        run_var = run_var.repeat(b)
    elif run_mean.shape[0] != expected_c:
        raise RuntimeError(f"custom_group_norm: running_mean size {run_mean.shape[0]} incompatible with expected {expected_c}")
    run_mean = run_mean.to(input_reshaped.dtype).to(input_reshaped.device)
    run_var = run_var.to(input_reshaped.dtype).to(input_reshaped.device)

    out = F.batch_norm(input_reshaped, run_mean, run_var, weight=None, bias=None,
                       training=False, momentum=0, eps=eps)

    out = out.view(b, c, *input.size()[2:])

    # post affine transform
    if weight is not None:
        out *= weight.view(1, -1, 1, 1)
    if bias is not None:
        out += bias.view(1, -1, 1, 1)
    return out


def crop_valid_region(x, input_bbox, target_bbox, is_decoder):
    """
    Crop the valid region from the tile
    @param x: input tile
    @param input_bbox: original input bounding box
    @param target_bbox: output bounding box
    @param scale: scale factor
    @return: cropped tile
    """
    padded_bbox = [i * 8 if is_decoder else i//8 for i in input_bbox]
    margin = [target_bbox[i] - padded_bbox[i] for i in range(4)]
    return x[:, :, margin[2]:x.size(2)+margin[3], margin[0]:x.size(3)+margin[1]]

# ↓↓↓ https://github.com/Kahsolt/stable-diffusion-webui-vae-tile-infer ↓↓↓


def perfcount(fn):
    def wrapper(*args, **kwargs):
        ts = time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(devices.device)
        devices.torch_gc()
        gc.collect()

        ret = fn(*args, **kwargs)

        devices.torch_gc()
        gc.collect()
        if torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated(devices.device) / 2**20
            torch.cuda.reset_peak_memory_stats(devices.device)
            print(
                f'[Tiled VAE]: Done in {time() - ts:.3f}s, max VRAM alloc {vram:.3f} MB')
        else:
            print(f'[Tiled VAE]: Done in {time() - ts:.3f}s')

        return ret
    return wrapper

# copy end :)


class GroupNormParam:
    def __init__(self):
        self.var_list = []
        self.mean_list = []
        self.pixel_list = []
        self.weight = None
        self.bias = None

    def add_tile(self, tile, layer):
        var, mean = get_var_mean(tile, 32)
        # For giant images, the variance can be larger than max float16
        # In this case we create a copy to float32
        if var.dtype == torch.float16 and var.isinf().any():
            fp32_tile = tile.float()
            var, mean = get_var_mean(fp32_tile, 32)
        # ============= DEBUG: test for infinite =============
        # if torch.isinf(var).any():
        #    print('var: ', var)
        # ====================================================
        self.var_list.append(var)
        self.mean_list.append(mean)
        self.pixel_list.append(
            tile.shape[2]*tile.shape[3])
        if hasattr(layer, 'weight'):
            self.weight = layer.weight
            self.bias = layer.bias
        else:
            self.weight = None
            self.bias = None

    def summary(self):
        """
        summarize the mean and var and return a function
        that apply group norm on each tile
        """
        if len(self.var_list) == 0:
            return None
        var = torch.vstack(self.var_list)
        mean = torch.vstack(self.mean_list)
        max_value = max(self.pixel_list)
        pixels = torch.tensor(
            self.pixel_list, dtype=torch.float32, device=var.device) / max_value
        sum_pixels = torch.sum(pixels)
        pixels = pixels.unsqueeze(
            1) / sum_pixels
        var = torch.sum(
            var * pixels, dim=0)
        mean = torch.sum(
            mean * pixels, dim=0)
        return lambda x:  custom_group_norm(x, 32, mean, var, self.weight, self.bias)

    @staticmethod
    def from_tile(tile, norm):
        """
        create a function from a single tile without summary
        """
        var, mean = get_var_mean(tile, 32)
        if var.dtype == torch.float16 and var.isinf().any():
            fp32_tile = tile.float()
            var, mean = get_var_mean(fp32_tile, 32)
            # if it is a macbook, we need to convert back to float16
            if var.device.type == 'mps':
                # clamp to avoid overflow
                var = torch.clamp(var, 0, 60000)
                var = var.half()
                mean = mean.half()
        if hasattr(norm, 'weight'):
            weight = norm.weight
            bias = norm.bias
        else:
            weight = None
            bias = None

        def group_norm_func(x, mean=mean, var=var, weight=weight, bias=bias):
            return custom_group_norm(x, 32, mean, var, weight, bias, 1e-6)
        return group_norm_func


class VAEHook:
    def __init__(self, net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu=False, dtype=None,
                 segmented_parallel: bool = False, cache_pre_norm: bool = False, micro_batch_size: int | None = None):
        self.net = net                  # encoder | decoder
        self.tile_size = tile_size
        self.is_decoder = is_decoder
        self.fast_mode = (fast_encoder and not is_decoder) or (
            fast_decoder and is_decoder)
        self.color_fix = color_fix and not is_decoder
        self.to_gpu = to_gpu
        self.pad = 11 if is_decoder else 32
        self.dtype = dtype
        # segmented-parallel options
        self.segmented_parallel = segmented_parallel
        self.cache_pre_norm = cache_pre_norm
        self.micro_batch_size = micro_batch_size

    def __call__(self, x):
        B, C, H, W = x.shape
        original_device = next(self.net.parameters()).device
        try:
            if self.to_gpu:
                self.net.to(devices.get_optimal_device())
            if max(H, W) <= self.pad * 2 + self.tile_size:
                print("[Tiled VAE]: the input size is tiny and unnecessary to tile.")
                out = self.net.original_forward(x)
            else:
                out = self.vae_tile_forward(x)
            if self.dtype is not None:
                out = out.to(self.dtype)
            return out
        finally:
            self.net.to(original_device)

    def get_best_tile_size(self, lowerbound, upperbound):
        """
        Get the best tile size for GPU memory
        """
        divider = 32
        while divider >= 2:
            remainer = lowerbound % divider
            if remainer == 0:
                return lowerbound
            candidate = lowerbound - remainer + divider
            if candidate <= upperbound:
                return candidate
            divider //= 2
        return lowerbound

    def split_tiles(self, h, w):
        """
        Tool function to split the image into tiles
        @param h: height of the image
        @param w: width of the image
        @return: tile_input_bboxes, tile_output_bboxes
        """
        tile_input_bboxes, tile_output_bboxes = [], []
        tile_size = self.tile_size
        pad = self.pad
        num_height_tiles = math.ceil((h - 2 * pad) / tile_size)
        num_width_tiles = math.ceil((w - 2 * pad) / tile_size)
        # If any of the numbers are 0, we let it be 1
        # This is to deal with long and thin images
        num_height_tiles = max(num_height_tiles, 1)
        num_width_tiles = max(num_width_tiles, 1)

        # Suggestions from https://github.com/Kahsolt: auto shrink the tile size
        real_tile_height = math.ceil((h - 2 * pad) / num_height_tiles)
        real_tile_width = math.ceil((w - 2 * pad) / num_width_tiles)
        real_tile_height = self.get_best_tile_size(real_tile_height, tile_size)
        real_tile_width = self.get_best_tile_size(real_tile_width, tile_size)

        print(f'[Tiled VAE]: split to {num_height_tiles}x{num_width_tiles} = {num_height_tiles*num_width_tiles} tiles. ' +
              f'Optimal tile size {real_tile_width}x{real_tile_height}, original tile size {tile_size}x{tile_size}')

        for i in range(num_height_tiles):
            for j in range(num_width_tiles):
                # bbox: [x1, x2, y1, y2]
                # the padding is is unnessary for image borders. So we directly start from (32, 32)
                input_bbox = [
                    pad + j * real_tile_width,
                    min(pad + (j + 1) * real_tile_width, w),
                    pad + i * real_tile_height,
                    min(pad + (i + 1) * real_tile_height, h),
                ]

                # if the output bbox is close to the image boundary, we extend it to the image boundary
                output_bbox = [
                    input_bbox[0] if input_bbox[0] > pad else 0,
                    input_bbox[1] if input_bbox[1] < w - pad else w,
                    input_bbox[2] if input_bbox[2] > pad else 0,
                    input_bbox[3] if input_bbox[3] < h - pad else h,
                ]

                # scale to get the final output bbox
                output_bbox = [x * 8 if self.is_decoder else x // 8 for x in output_bbox]
                tile_output_bboxes.append(output_bbox)

                # indistinguishable expand the input bbox by pad pixels
                tile_input_bboxes.append([
                    max(0, input_bbox[0] - pad),
                    min(w, input_bbox[1] + pad),
                    max(0, input_bbox[2] - pad),
                    min(h, input_bbox[3] + pad),
                ])

        return tile_input_bboxes, tile_output_bboxes

    @torch.no_grad()
    def estimate_group_norm(self, z, task_queue, color_fix):
        device = z.device
        tile = z
        last_id = len(task_queue) - 1
        while last_id >= 0 and task_queue[last_id][0] != 'pre_norm':
            last_id -= 1
        if last_id <= 0 or task_queue[last_id][0] != 'pre_norm':
            raise ValueError('No group norm found in the task queue')
        # estimate until the last group norm
        for i in range(last_id + 1):
            task = task_queue[i]
            if task[0] == 'pre_norm':
                group_norm_func = GroupNormParam.from_tile(tile, task[1])
                task_queue[i] = ('apply_norm', group_norm_func)
                if i == last_id:
                    return True
                tile = group_norm_func(tile)
            elif task[0] == 'store_res':
                task_id = i + 1
                while task_id < last_id and task_queue[task_id][0] != 'add_res':
                    task_id += 1
                if task_id >= last_id:
                    continue
                task_queue[task_id][1] = task[1](tile)
            elif task[0] == 'add_res':
                tile += task[1].to(device)
                task[1] = None
            elif color_fix and task[0] == 'downsample':
                for j in range(i, last_id + 1):
                    if task_queue[j][0] == 'store_res':
                        task_queue[j] = ('store_res_cpu', task_queue[j][1])
                return True
            else:
                tile = task[1](tile)
            try:
                devices.test_for_nans(tile, "vae")
            except:
                print(f'Nan detected in fast mode estimation. Fast mode disabled.')
                return False

        raise IndexError('Should not reach here')

    # @perfcount
    @torch.no_grad()
    def vae_tile_forward(self, z):
        """
        Decode a latent vector z into an image in a tiled manner.
        @param z: latent vector
        @return: image
        """
        device = next(self.net.parameters()).device
        net = self.net
        tile_size = self.tile_size
        is_decoder = self.is_decoder

        z = z.detach() # detach the input to avoid backprop

        N, height, width = z.shape[0], z.shape[2], z.shape[3]
        net.last_z_shape = z.shape

        # Split the input into tiles and build a task queue for each tile
        print(f'[Tiled VAE]: input_size: {z.shape}, tile_size: {tile_size}, padding: {self.pad}')

        in_bboxes, out_bboxes = self.split_tiles(height, width)

        # Prepare tiles by split the input latents
        tiles = []
        for input_bbox in in_bboxes:
            tile = z[:, :, input_bbox[2]:input_bbox[3], input_bbox[0]:input_bbox[1]].cpu()
            tiles.append(tile)

        num_tiles = len(tiles)
        num_completed = 0

        # Build task queues
        single_task_queue = build_task_queue(net, is_decoder)
        #print(single_task_queue)
        if self.fast_mode:
            # Fast mode: downsample the input image to the tile size,
            # then estimate the group norm parameters on the downsampled image
            scale_factor = tile_size / max(height, width)
            z = z.to(device)
            downsampled_z = F.interpolate(z, scale_factor=scale_factor, mode='nearest-exact')
            # use nearest-exact to keep statictics as close as possible
            print(f'[Tiled VAE]: Fast mode enabled, estimating group norm parameters on {downsampled_z.shape[3]} x {downsampled_z.shape[2]} image')

            # ======= Special thanks to @Kahsolt for distribution shift issue ======= #
            # The downsampling will heavily distort its mean and std, so we need to recover it.
            std_old, mean_old = torch.std_mean(z, dim=[0, 2, 3], keepdim=True)
            std_new, mean_new = torch.std_mean(downsampled_z, dim=[0, 2, 3], keepdim=True)
            downsampled_z = (downsampled_z - mean_new) / std_new * std_old + mean_old
            del std_old, mean_old, std_new, mean_new
            # occasionally the std_new is too small or too large, which exceeds the range of float16
            # so we need to clamp it to max z's range.
            downsampled_z = torch.clamp_(downsampled_z, min=z.min(), max=z.max())
            estimate_task_queue = clone_task_queue(single_task_queue)
            if self.estimate_group_norm(downsampled_z, estimate_task_queue, color_fix=self.color_fix):
                single_task_queue = estimate_task_queue
            del downsampled_z

        task_queues = [clone_task_queue(single_task_queue) for _ in range(num_tiles)]

        # Dummy result
        result = None
        result_approx = None
        #try:
        #    with devices.autocast():
        #        result_approx = torch.cat([F.interpolate(cheap_approximation(x).unsqueeze(0), scale_factor=opt_f, mode='nearest-exact') for x in z], dim=0).cpu()
        #except: pass
        # Free memory of input latent tensor
        del z

        # Segmented parallel: split by norm stages并分组等尺寸tile进行两段式并行
        if self.segmented_parallel:
            full_queue = single_task_queue
            stages = split_into_micro_stages(full_queue)
            use_cuda = torch.cuda.is_available() and device.type == 'cuda'
            # choose micro-batch
            if self.micro_batch_size is not None:
                default_micro = max(1, min(self.micro_batch_size, num_tiles))
            else:
                default_micro = 4 if use_cuda else 1

            # per-tile current features and residual queues
            cur_feats = [t.cpu() for t in tiles]
            residual_queues = [[] for _ in range(num_tiles)]
            result = None
            pre_norm_cache = None  # optional caching per stage

            def group_by_shape(feats):
                groups = {}
                for idx, ft in enumerate(feats):
                    if ft is None:
                        continue
                    key = (int(ft.shape[0]), int(ft.shape[2]), int(ft.shape[3]))
                    groups.setdefault(key, []).append(idx)
                return groups

            def cat_batch(indices):
                # assume same N,H,W per group
                Ns = [cur_feats[i].shape[0] for i in indices]
                if len(set(Ns)) != 1:
                    return None, None, None
                N0 = Ns[0]
                xs = [cur_feats[i].to(device) for i in indices]
                batch = torch.cat(xs, dim=0)  # [len(indices)*N0, C, H, W]
                return batch, N0, indices

            def split_batch(batch, N0, count):
                # returns list of [N0,C,H,W]
                outs = []
                for k in range(count):
                    outs.append(batch[k*N0:(k+1)*N0])
                return outs

            def stage_has_norm(ops):
                for name, _ in ops:
                    if name in ('pre_norm', 'apply_norm'):
                        return True
                return False

            def run_until_norm(x, ops, indices, N0, local_res_q):
                # returns pre_norm_input (batch) and norm layer
                for name, fn in ops:
                    if name in ('pre_norm', 'apply_norm'):
                        return x, fn
                    elif name in ('store_res', 'store_res_cpu'):
                        res_batch = fn(x)
                        parts = split_batch(res_batch, N0, len(indices))
                        for j, idx in enumerate(indices):
                            local_res_q[idx].append(parts[j].detach())
                    elif name == 'add_res':
                        # add from queues
                        adds = []
                        for idx in indices:
                            adds.append(local_res_q[idx].pop(0).to(device))
                        add_batch = torch.cat(adds, dim=0)
                        x = x + add_batch
                    else:
                        x = fn(x)
                return x, None

            def run_full_with_norm(x, ops, indices, N0, norm_apply, last_stage):
                applied = False
                for name, fn in ops:
                    if name in ('pre_norm', 'apply_norm'):
                        x = norm_apply(x)
                        applied = True
                    elif name in ('store_res', 'store_res_cpu'):
                        res_batch = fn(x)
                        parts = split_batch(res_batch, N0, len(indices))
                        for j, idx in enumerate(indices):
                            residual_queues[idx].append(parts[j].detach())
                    elif name == 'add_res':
                        adds = []
                        for idx in indices:
                            adds.append(residual_queues[idx].pop(0).to(device))
                        add_batch = torch.cat(adds, dim=0)
                        x = x + add_batch
                    else:
                        x = fn(x)
                assert applied, 'norm not applied in stage'
                return x

            def run_full_no_norm(x, ops, indices, N0):
                for name, fn in ops:
                    if name in ('store_res', 'store_res_cpu'):
                        res_batch = fn(x)
                        parts = split_batch(res_batch, N0, len(indices))
                        for j, idx in enumerate(indices):
                            residual_queues[idx].append(parts[j].detach())
                    elif name == 'add_res':
                        adds = []
                        for idx in indices:
                            adds.append(residual_queues[idx].pop(0).to(device))
                        add_batch = torch.cat(adds, dim=0)
                        x = x + add_batch
                    elif name in ('pre_norm', 'apply_norm'):
                        # should not happen
                        pass
                    else:
                        x = fn(x)
                return x

            # process each stage
            for s_idx, ops in enumerate(stages):
                groups = group_by_shape(cur_feats)
                if not groups:
                    continue
                has_norm = stage_has_norm(ops)
                # optional cache of pre-norm inputs per tile for this stage
                pre_norm_cache = [None for _ in range(num_tiles)] if (has_norm and self.cache_pre_norm) else None
                # pass 1: collect stats if needed
                if has_norm:
                    grp_param = GroupNormParam()
                    for key, idxs in groups.items():
                        N0, H, W = key
                        micro = min(default_micro, len(idxs))
                        for start in range(0, len(idxs), micro):
                            sub = idxs[start:start+micro]
                            batch, N0k, sub_idx = cat_batch(sub)
                            if batch is None:
                                # fallback per-tile
                                for ii in sub:
                                    xb, _, _ = cat_batch([ii])
                                    local_q = [[] for _ in range(num_tiles)]
                                    pre, norm_layer = run_until_norm(xb, ops, [ii], cur_feats[ii].shape[0], local_q)
                                    grp_param.add_tile(pre, norm_layer)
                                    if pre_norm_cache is not None:
                                        pre_norm_cache[ii] = pre.detach().cpu()
                                continue
                            local_q = [[] for _ in range(num_tiles)]
                            pre, norm_layer = run_until_norm(batch, ops, sub_idx, N0k, local_q)
                            # accumulate per tile
                            parts = split_batch(pre, N0k, len(sub_idx))
                            for j, ii in enumerate(sub_idx):
                                grp_param.add_tile(parts[j], norm_layer)
                                if pre_norm_cache is not None:
                                    pre_norm_cache[ii] = parts[j].detach().cpu()
                            del batch, pre
                    norm_apply = grp_param.summary()

                # pass 2: run stage
                for key, idxs in groups.items():
                    N0, H, W = key
                    micro = min(default_micro, len(idxs))
                    for start in range(0, len(idxs), micro):
                        sub = idxs[start:start+micro]
                        if has_norm and pre_norm_cache is not None:
                            # build batch from cached pre-norm inputs (already on CPU)
                            xs = [pre_norm_cache[ii].to(device) for ii in sub]
                            batch = torch.cat(xs, dim=0)
                            N0k = pre_norm_cache[sub[0]].shape[0]
                            sub_idx = sub
                        else:
                            batch, N0k, sub_idx = cat_batch(sub)
                        if batch is None:
                            # fallback per-tile
                            for ii in sub:
                                xb, _, _ = cat_batch([ii])
                                if has_norm:
                                    if pre_norm_cache is not None:
                                        xb = pre_norm_cache[ii].to(device)
                                        # apply norm then run post ops only
                                        xbn = norm_apply(xb)
                                        # run post-only
                                        post_only = []
                                        hit = False
                                        for name, fn in ops:
                                            if not hit and name in ('pre_norm', 'apply_norm'):
                                                hit = True
                                                continue
                                            if hit:
                                                post_only.append((name, fn))
                                        post = run_full_no_norm(xbn, post_only, [ii], xbn.shape[0])
                                    else:
                                        post = run_full_with_norm(xb, ops, [ii], cur_feats[ii].shape[0], norm_apply, s_idx == len(stages)-1)
                                else:
                                    post = run_full_no_norm(xb, ops, [ii], cur_feats[ii].shape[0])
                                if s_idx == len(stages)-1:
                                    out_tile = post
                                    if result is None:
                                        result = torch.zeros((N, out_tile.shape[1], height * 8 if is_decoder else height // 8, width * 8 if is_decoder else width // 8), device=device, requires_grad=False)
                                    result[:, :, out_bboxes[ii][2]:out_bboxes[ii][3], out_bboxes[ii][0]:out_bboxes[ii][1]] = crop_valid_region(out_tile, in_bboxes[ii], out_bboxes[ii], is_decoder)
                                else:
                                    cur_feats[ii] = post.detach().cpu()
                            continue

                        if has_norm:
                            if pre_norm_cache is not None:
                                # apply norm then post-only
                                bn = norm_apply(batch)
                                post_only = []
                                hit = False
                                for name, fn in ops:
                                    if not hit and name in ('pre_norm', 'apply_norm'):
                                        hit = True
                                        continue
                                    if hit:
                                        post_only.append((name, fn))
                                post = run_full_no_norm(bn, post_only, sub_idx, N0k)
                            else:
                                post = run_full_with_norm(batch, ops, sub_idx, N0k, norm_apply, s_idx == len(stages)-1)
                        else:
                            post = run_full_no_norm(batch, ops, sub_idx, N0k)

                        last_stage = (s_idx == len(stages) - 1)
                        parts = split_batch(post, N0k, len(sub_idx))
                        for j, ii in enumerate(sub_idx):
                            out_tile = parts[j]
                            if last_stage:
                                if result is None:
                                    result = torch.zeros((N, out_tile.shape[1], height * 8 if is_decoder else height // 8, width * 8 if is_decoder else width // 8), device=device, requires_grad=False)
                                result[:, :, out_bboxes[ii][2]:out_bboxes[ii][3], out_bboxes[ii][0]:out_bboxes[ii][1]] = crop_valid_region(out_tile, in_bboxes[ii], out_bboxes[ii], is_decoder)
                            else:
                                cur_feats[ii] = out_tile.detach().cpu()
                        del batch, post

            return result if result is not None else torch.zeros(1, device=device)

        # Task queue execution（原串行路径）
        pbar = tqdm(total=num_tiles * len(task_queues[0]), desc=f"[Tiled VAE]: Executing {'Decoder' if is_decoder else 'Encoder'} Task Queue: ")

        # execute the task back and forth when switch tiles so that we always
        # keep one tile on the GPU to reduce unnecessary data transfer
        forward = True
        interrupted = False
        #state.interrupted = interrupted
        while True:
            #if state.interrupted: interrupted = True ; break

            group_norm_param = GroupNormParam()
            for i in range(num_tiles) if forward else reversed(range(num_tiles)):
                #if state.interrupted: interrupted = True ; break

                tile = tiles[i].to(device)
                input_bbox = in_bboxes[i]
                task_queue = task_queues[i]

                interrupted = False
                while len(task_queue) > 0:
                    #if state.interrupted: interrupted = True ; break

                    # DEBUG: current task
                    # print('Running task: ', task_queue[0][0], ' on tile ', i, '/', num_tiles, ' with shape ', tile.shape)
                    task = task_queue.pop(0)
                    if task[0] == 'pre_norm':
                        group_norm_param.add_tile(tile, task[1])
                        break
                    elif task[0] == 'store_res' or task[0] == 'store_res_cpu':
                        task_id = 0
                        res = task[1](tile)
                        if not self.fast_mode or task[0] == 'store_res_cpu':
                            res = res.cpu()
                        while task_queue[task_id][0] != 'add_res':
                            task_id += 1
                        task_queue[task_id][1] = res
                    elif task[0] == 'add_res':
                        tile += task[1].to(device)
                        task[1] = None
                    else:
                        tile = task[1](tile)
                    pbar.update(1)

                if interrupted: break

                # check for NaNs in the tile.
                # If there are NaNs, we abort the process to save user's time
                #devices.test_for_nans(tile, "vae")

                #print(tiles[i].shape, tile.shape, i, num_tiles)
                if len(task_queue) == 0:
                    tiles[i] = None
                    num_completed += 1
                    if result is None:      # NOTE: dim C varies from different cases, can only be inited dynamically
                        result = torch.zeros((N, tile.shape[1], height * 8 if is_decoder else height // 8, width * 8 if is_decoder else width // 8), device=device, requires_grad=False)
                    result[:, :, out_bboxes[i][2]:out_bboxes[i][3], out_bboxes[i][0]:out_bboxes[i][1]] = crop_valid_region(tile, in_bboxes[i], out_bboxes[i], is_decoder)
                    del tile
                elif i == num_tiles - 1 and forward:
                    forward = False
                    tiles[i] = tile
                elif i == 0 and not forward:
                    forward = True
                    tiles[i] = tile
                else:
                    tiles[i] = tile.cpu()
                    del tile

            if interrupted: break
            if num_completed == num_tiles: break

            # insert the group norm task to the head of each task queue
            group_norm_func = group_norm_param.summary()
            if group_norm_func is not None:
                for i in range(num_tiles):
                    task_queue = task_queues[i]
                    task_queue.insert(0, ('apply_norm', group_norm_func))

        # Done!
        pbar.close()
        return result if result is not None else result_approx.to(device)