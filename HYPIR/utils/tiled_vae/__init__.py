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
    )

    try:
        yield
    finally:
        del model.original_forward
        model.forward = original_forward
