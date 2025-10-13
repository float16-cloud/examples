import torch
from diffusers import WanPipeline, AutoencoderKLWan, AutoModel, TorchAoConfig
from diffusers.utils import export_to_video
from transformers import UMT5EncoderModel
print(torch.compiler.list_backends())
quantization_config = TorchAoConfig("int8wo")
dtype = torch.bfloat16
text_encoder = UMT5EncoderModel.from_pretrained("/share_weights/Wan2.2-T2V-A14B-Diffusers", subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map = "cuda")
vae = AutoencoderKLWan.from_pretrained("/share_weights/Wan2.2-T2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32, device_map = "cuda")
transformer = AutoModel.from_pretrained("/share_weights/Wan2.2-T2V-A14B-Diffusers", 
                                        subfolder="transformer", 
                                        torch_dtype=torch.bfloat16,
                                        device_map = "cuda",
                                        quantization_config=quantization_config)
transformer_2 = AutoModel.from_pretrained("/share_weights/Wan2.2-T2V-A14B-Diffusers", 
                                          subfolder="transformer_2", 
                                          torch_dtype=torch.bfloat16,
                                          device_map = "cuda",
                                          quantization_config=quantization_config)
pipe = WanPipeline.from_pretrained("/share_weights/Wan2.2-T2V-A14B-Diffusers",
    text_encoder = text_encoder,
    transformer = transformer,
    transformer_2 = transformer_2,
    vae=vae,
    torch_dtype=dtype
).to("cuda")
pipe.transformer = torch.compile(pipe.transformer, backend="onnxrt")
pipe.transformer_2 = torch.compile(pipe.transformer_2, backend="onnxrt")

height = 720
width = 1280

prompt = """Rocket exhaust plume creating artificial aurora. Camera-Tracking Rocket"""
negative_prompt = "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,画得不好的脸部"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=81,
    guidance_scale=3.0,
    guidance_scale_2=4.0,
    num_inference_steps=40,
).frames[0]

print('saving...')
export_to_video(output, "./rocket_t2v_out.mp4", fps=16)