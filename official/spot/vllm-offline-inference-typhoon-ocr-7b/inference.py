from dataclasses import asdict
import time
from vllm import LLM, SamplingParams
from utils import init_typhoon_ocr_vllm, build_batch_request, extract_text_from_image

MAX_BATCH_SIZE = 64 # Adjust this based on your GPU memory and model requirements, 64 is good for H100 (80GB)

images_anchor_list = []
images_pil_list = []
filename = '../../tmp_data/law.pdf'
page_start = 1 # Starting from page 1
page_stop = 32

extract_time = time.time()
for page_num in range(page_start, page_stop + 1):
    anchor_text, pil_image = extract_text_from_image(filename, page_num)
    images_anchor_list.append(anchor_text)
    images_pil_list.append(pil_image)

print(f"Extracted {len(images_anchor_list)} images and texts in {time.time() - extract_time:.2f} seconds.")

init_model_time = time.time()
req_data = init_typhoon_ocr_vllm(images_anchor_list, "default", MAX_BATCH_SIZE)
engine_args = asdict(req_data.engine_args)

llm = LLM(**engine_args)
print(f"Model initialized in {time.time() - init_model_time:.2f} seconds.")

prompts = req_data.prompts
sampling_params = SamplingParams(
    temperature=0.1, ## Recommended by Typhoon OCR team
    top_p=0.6, ## Recommended by Typhoon OCR team
    repetition_penalty=1.25, ## Typhoon OCR team recommends 1.2, but we use 1.25 to avoid repetition
    max_tokens=2048 ## Max tokens for the output, can be adjusted based on your needs
)

print(f"Running with {len(prompts)} prompts and {len(images_pil_list)} images...")
inputs = build_batch_request(prompts,images_pil_list)

print("Loading model...")
start_time = time.time()
outputs = llm.generate(
    inputs,
    sampling_params=sampling_params,
)

print("-" * 50)
for idx,o in enumerate(outputs):
    if idx >= 10:
        break
    generated_text = o.outputs[0].text
    print(generated_text)
    print("-" * 50)