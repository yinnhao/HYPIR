from contextlib import contextmanager
from HYPIR.utils.tiled_vae.vaehook import VAEHook


@contextmanager
def enable_tiled_vae(
    vae,
    is_decoder,
    tile_size=256,
    dtype=None,
    fast_decoder=False,  # 新增参数
    fast_encoder=False,  # 新增参数
    segmented_parallel=False,  # 分段并行（按 Norm 分段）
    cache_pre_norm=False,      # 显存足够时：阶段内单遍（缓存 pre_norm 激活）
    micro_batch_size=None,     # 分段并行微批大小
):
    if not is_decoder:
        original_forward = vae.encoder.forward
        model = vae.encoder
    else:
        original_forward = vae.decoder.forward
        model = vae.decoder
    model.original_forward = original_forward

    model.forward = VAEHook(
        model,
        tile_size,
        is_decoder=is_decoder,
        fast_decoder=fast_decoder,  # 启用快速解码
        fast_encoder=fast_encoder,  # 启用快速编码
        color_fix=False,
        to_gpu=False,
        dtype=dtype,
        segmented_parallel=segmented_parallel,
        cache_pre_norm=cache_pre_norm,
        micro_batch_size=micro_batch_size,
    )

    try:
        yield
    finally:
        del model.original_forward
        model.forward = original_forward
