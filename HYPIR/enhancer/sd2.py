import torch
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig

from HYPIR.enhancer.base import BaseEnhancer


class SD2Enhancer(BaseEnhancer):

    def __init__(self, *args, **kwargs):
        # 提取优化参数
        self.optimization_params = {
            'vae_batch_size': kwargs.pop('vae_batch_size', 4),
            'enable_fast_vae': kwargs.pop('enable_fast_vae', False),
            'generator_batch_size': kwargs.pop('generator_batch_size', 2),
            'enable_amp': kwargs.pop('enable_amp', False),
        }
        
        # 调用父类构造函数，传递优化参数
        super().__init__(*args, **self.optimization_params, **kwargs)

    def init_scheduler(self):
        self.scheduler = DDPMScheduler.from_pretrained(
            self.base_model_path, subfolder="scheduler")

    def init_text_models(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.base_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.base_model_path, subfolder="text_encoder", torch_dtype=self.weight_dtype).to(self.device)
        self.text_encoder.eval().requires_grad_(False)

    def init_generator(self):
        self.G: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            self.base_model_path, subfolder="unet", torch_dtype=self.weight_dtype).to(self.device)
        target_modules = self.lora_modules
        G_lora_cfg = LoraConfig(r=self.lora_rank, lora_alpha=self.lora_rank,
            init_lora_weights="gaussian", target_modules=target_modules)
        self.G.add_adapter(G_lora_cfg)

        print(f"Load model weights from {self.weight_path}")
        state_dict = torch.load(self.weight_path, map_location="cpu", weights_only=False)
        self.G.load_state_dict(state_dict, strict=False)
        
        # 启用编译优化（如果支持）
        if hasattr(torch, 'compile') and self.enable_amp:
            try:
                self.G = torch.compile(self.G, mode="reduce-overhead")
                print("✓ Enabled torch.compile optimization for UNet")
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")

        self.G.eval().requires_grad_(False)

    def prepare_inputs(self, batch_size, prompt):
        bs = batch_size
        txt_ids = self.tokenizer(
            [prompt] * bs,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_embed = self.text_encoder(txt_ids.to(self.device))[0]
        c_txt = {"text_embed": text_embed}
        timesteps = torch.full((bs,), self.model_t, dtype=torch.long, device=self.device)
        self.inputs = dict(
            c_txt=c_txt,
            timesteps=timesteps,
        )

    def forward_generator(self, z_lq):
        z_in = z_lq * self.vae.config.scaling_factor
        
        # 使用混合精度
        with torch.cuda.amp.autocast() if self.enable_amp else torch.no_grad():
            eps = self.G(
                z_in, self.inputs["timesteps"],
                encoder_hidden_states=self.inputs["c_txt"]["text_embed"],
            ).sample
            
        z = self.scheduler.step(eps, self.coeff_t, z_in).pred_original_sample
        z_out = z / self.vae.config.scaling_factor
        return z_out
