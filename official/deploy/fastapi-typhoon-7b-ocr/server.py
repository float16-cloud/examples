import io
import os
import torch
import base64
import uvicorn
import pdf2image
from PIL import Image
from io import BytesIO
from fastapi import FastAPI
from typing import Callable
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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

def get_prompt(prompt_name: str) -> Callable[[str], str]:
    """
    Fetches the system prompt based on the provided PROMPT_NAME.

    :param prompt_name: The identifier for the desired prompt.
    :return: The system prompt as a string.
    """
    return PROMPTS_SYS.get(prompt_name, lambda x: "Invalid PROMPT_NAME provided.")

class OCRRequest(BaseModel):
    base64_pdf: str

app = FastAPI()
@app.post("/ocr")
async def _ocr(ocr_request: OCRRequest):
    base64_pdf = ocr_request.base64_pdf

    # Render the first page to base64 PNG and then load it into a PIL image.
    pdf_bytes = base64.b64decode(base64_pdf)
    task_type = 'default'
    model_path = '../model-weight/typhoon-ocr-7b'

    # Convert PDF to images
    image_pil_list = pdf2image.convert_from_bytes(pdf_bytes, dpi=72) #Return as Image
    if len(image_pil_list) == 0:
        return {"error": "No images found in the PDF."}
    
    if len(image_pil_list) > 1:
        return {"error": "PDF contains multiple pages, only the first page is supported."}
    
    # Get the first (and only) image
    pil_image = image_pil_list[0]

    # Convert PIL Image to base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Retrieve and fill in the prompt template with the anchor_text
    prompt_template_fn = get_prompt(task_type)
    PROMPT = prompt_template_fn("") # This could be empty

    messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }]

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, 
                                                            attn_implementation="flash_attention_2",
                                                            torch_dtype=torch.bfloat16 ).eval().to(device)
    
    processor = AutoProcessor.from_pretrained(model_path)

    # Apply the chat template and processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    inputs = processor(
            text=[text],
            images=[main_image],
            padding=True,
            return_tensors="pt",
        )
    inputs = {key: value.to(device) for (key, value) in inputs.items()}

    # Generate the output
    output = model.generate(
                    **inputs,
                    temperature=0.1,
                    max_new_tokens=4096,
                    num_return_sequences=1,
                    repetition_penalty=1.2,
                    do_sample=True,
                )
    # Decode the output
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
    )

    return JSONResponse(
        content=text_output[0]
    )

async def main():
    config = uvicorn.Config(
        app, host="0.0.0.0", port=int(os.environ["PORT"])
    )
    server = uvicorn.Server(config)
    await server.serve()