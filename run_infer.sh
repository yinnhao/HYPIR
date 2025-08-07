# python test.py \
#     --base_model_type sd2 \
#     --base_model_path /path/to/sd2_model \
#     --weight_path /path/to/weights.pth \
#     --model_t 200 \
#     --coeff_t 200 \
#     --lora_rank 256 \
#     --lora_modules "to_k,to_q,to_v,to_out.0,conv,conv1,conv2" \
#     --lq_dir /path/to/input \
#     --output_dir /path/to/output \
#     --captioner empty \
#     --patch_size 512 \
#     --stride 256 \
#     --vae_batch_size 8 \
#     --generator_batch_size 4 \
#     --enable_fast_vae \
#     --enable_amp


# python test.py \
#     --base_model_type sd2 \
#     --base_model_path /path/to/sd2_model \
#     --weight_path /path/to/weights.pth \
#     --model_t 200 \
#     --coeff_t 200 \
#     --lora_rank 256 \
#     --lora_modules "to_k,to_q,to_v,to_out.0,conv,conv1,conv2" \
#     --lq_dir /path/to/input \
#     --output_dir /path/to/output \
#     --captioner empty \
#     --patch_size 512 \
#     --stride 256 \
#     --vae_batch_size 8 \
#     --encoder_batch_size 16 \
#     --generator_batch_size 4 \
#     --enable_fast_vae \
#     --enable_parallel_encode \
#     --enable_amp


# python test.py \
#     --base_model_type sd2 \
#     --base_model_path /path/to/sd2_model \
#     --weight_path /path/to/weights.pth \
#     --model_t 200 \
#     --coeff_t 200 \
#     --lora_rank 256 \
#     --lora_modules "to_k,to_q,to_v,to_out.0,conv,conv1,conv2" \
#     --lq_dir /path/to/input \
#     --output_dir /path/to/output \
#     --captioner empty \
#     --patch_size 512 \
#     --stride 256 \
#     --vae_batch_size 8 \
#     --generator_batch_size 4 \
#     --enable_fast_vae \
#     --enable_amp


# LORA_MODULES_LIST=(to_k to_q to_v to_out.0 conv conv1 conv2 conv_shortcut conv_out proj_in proj_out ff.net.2 ff.net.0.proj)
# IFS=','
# LORA_MODULES="${LORA_MODULES_LIST[*]}"
# echo $LORA_MODULES
# unset IFS
python test.py --base_model_type sd2 --base_model_path stabilityai/stable-diffusion-2-1-base --model_t 200 --coeff_t 200 --lora_rank 256 --lora_modules to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj --weight_path model_pth/HYPIR_sd2.pth --patch_size 512 --stride 256 --lq_dir /root/paddlejob/workspace/env_run/guanfeiqiang/data/test100_rresize512 --scale_by factor --upscale 2 --captioner empty --output_dir ./result --seed 231 --device cuda --vae_batch_size 4 --generator_batch_size 4 --enable_fast_vae --encoder_batch_size 4 --enable_parallel_encode