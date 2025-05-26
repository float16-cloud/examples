from PIL import Image
from io import BytesIO
import base64
from typhoon_ocr.ocr_utils import render_pdf_to_base64png, get_anchor_text
from typing import NamedTuple, Optional, Literal
from vllm import EngineArgs

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None


def init_typhoon_ocr_vllm(extra_info: list[str], mode: Literal['default', 'structure'], max_batch_size) -> ModelRequestData:

    model_name = "../../model-weight/typhoon-ocr-7b"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=32000,
        max_num_seqs=max_batch_size,
        mm_processor_kwargs={
            "min_pixels": 180 * 28 * 28,
            "max_pixels": 2560 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={'image': 1}
    )


    PROMPTS_SYS = {
        "default": lambda base_text: (f"Below is an image of a document page along with its dimensions. "
            f"Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.\n"
            f"If the document contains images, use a placeholder like dummy.png for each image.\n"
            f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
            f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"),
        "structure": lambda base_text: (
            f"Below is an image of a document page, along with its dimensions and possibly some raw textual content previously extracted from it. "
            f"Note that the text extraction may be incomplete or partially missing. Carefully consider both the layout and any available text to reconstruct the document accurately.\n"
            f"Your task is to return the markdown representation of this document, presenting tables in HTML format as they naturally appear.\n"
            f"If the document contains images or figures, analyze them and include the tag <figure>IMAGE_ANALYSIS</figure> in the appropriate location.\n"
            f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
            f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
        ),
    }


    placeholder = "<|image_pad|>"
    prompts = [
        ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
         f"""{PROMPTS_SYS[mode](info)}"""
         f"<|vision_start|>{placeholder}<|vision_end|>"
         f"<|im_end|>\n"
         "<|im_start|>assistant\n") for info in extra_info
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )

def build_batch_request(texts, pil_images):
    inputs = []

    for text, image in zip(texts, pil_images):
        inputs.append({
            "prompt": text,
            "multi_modal_data": {
                "image": image
            },
        })
    
    return inputs

def extract_text_from_image(filename, page_num):
    # Render the first page to base64 PNG and then load it into a PIL image.
    image_base64 = render_pdf_to_base64png(filename, page_num, target_longest_image_dim=1800)

    # Extract anchor text from the PDF (first page)
    anchor_text = get_anchor_text(filename, page_num, pdf_engine="pdfreport", target_length=8000)

    # Retrieve and fill in the prompt template with the anchor_text
    pil_image = Image.open(BytesIO(base64.b64decode(image_base64)))
    return anchor_text, pil_image
