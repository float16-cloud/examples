import base64
from io import BytesIO
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2VisionTower,
)

from exllamav2.generator import (
    ExLlamaV2DynamicGenerator,
    ExLlamaV2Sampler,
)

import os
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel


class OCRRequest(BaseModel):
    base64_image: str

app = FastAPI()
model_directory = "../model-weight/gemma-3-12b-it-exl2-6bpw"

streaming = True
greedy = True

# Util function to get a PIL image from a URL or from a file in the script's directory
def get_image(base64_image):
    image_data = base64.b64decode(base64_image)
    image_buffer = BytesIO(image_data)
    return Image.open(image_buffer)

@app.post("/ocr")
async def _ocr(ocr_request: OCRRequest):
    base64_image = ocr_request.base64_image

    # Initialize model
    config = ExLlamaV2Config(model_directory)
    config.max_seq_len = 8192  # Pixtral default is 1M

    # Load vision model and multimodal projector and initialize preprocessor
    vision_model = ExLlamaV2VisionTower(config)
    vision_model.load(progress = True)

    # Load EXL2 model
    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, max_seq_len = 8192, lazy = True)
    model.load_autosplit(progress = True, cache = cache)
    tokenizer = ExLlamaV2Tokenizer(config)

    # Create generator
    generator = ExLlamaV2DynamicGenerator(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
    )

    image_embeddings = [
        vision_model.get_image_embeddings(
            model = model,
            tokenizer = tokenizer,
            image = get_image(base64_image)
        )
    ]

    placeholders = "\n".join([ie.text_alias for ie in image_embeddings]) + "\n"

    # Define a prompt using the aliases above as placeholders for image tokens. The tokenizer will replace each alias
    # with a range of temporary token IDs, and the model will embed those temporary IDs from their respective sources
    # rather than the model's text embedding table.
    #
    # The temporary IDs are unique for the lifetime of the process and persist as long as a reference is held to the
    # corresponding ExLlamaV2Embedding object. This way, images can be reused between generations, or used multiple
    # for multiple jobs in a batch, and the generator will be able to apply prompt caching and deduplication to image
    # tokens as well as text tokens.
    #
    # Image token IDs are assigned sequentially, however, so two ExLlamaV2Embedding objects created from the same
    # source image will not be recognized as the same image for purposes of prompt caching etc.
    instruction = "OCR รูปต่อไปนี้"
    prompt = (
        "<start_of_turn>user\nYou are a helpful assistant.\n\n\n\n" +
        placeholders +
        "\n" +
        instruction +
        "<end_of_turn>\n" +
        "<start_of_turn>model\n"
    )
    stop_conditions = [tokenizer.single_id("<end_of_turn>")]
    output = generator.generate(
        prompt = prompt,
        max_new_tokens = 500,
        add_bos = True,
        encode_special_tokens = True,
        decode_special_tokens = True,
        stop_conditions = stop_conditions,
        gen_settings = ExLlamaV2Sampler.Settings.greedy() if greedy else None,
        embeddings = image_embeddings,
    )
    return {"message": output}

async def main():
    config = uvicorn.Config(
        app, host="0.0.0.0", port=int(os.environ["PORT"])
    )
    server = uvicorn.Server(config)
    await server.serve()