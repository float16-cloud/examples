import base64
from io import BytesIO
from typing import Optional

import requests
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


class ContentPayload(BaseModel):
    type: str
    text: Optional[str] = None  # Make text optional since image_url type won't have text
    image_url: Optional[str] = None

class MessagePayload(BaseModel):
    role: str
    content: list[ContentPayload]

class VisionRequest(BaseModel):
    messages: list[MessagePayload]  # Changed from list[ContentPayload] to list[MessagePayload]
    model: Optional[str] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40

app = FastAPI()
model_directory = "../model-weight/gemma-3-12b-it-exl2-6bpw"

streaming = True
greedy = True

# Util function to get a PIL image from a URL or from a file in the script's directory
def get_image(url):
    try : 
        print('try to read from url:', url)
        return Image.open(requests.get(url, stream = True).raw)
    except Exception as e:
        print(f"try to read from url failed: {e}")
        if url.startswith("data:image"):
            base64_data = url.split(",")[1]
            image_data = base64.b64decode(base64_data)
            return Image.open(BytesIO(image_data))
        else:
            raise ValueError("Invalid image URL or data format.")

@app.post("/chat/completions")
async def _chat_completions(vision_request: VisionRequest):
    messages = vision_request.messages
    max_tokens = vision_request.max_tokens or 1024
    image_url = None
    if messages[-1].content[-1].type == 'image_url':
        try:
            image_url = messages[-1].content[-1].image_url
            if not image_url:
                return {"error": "Image URL is required for image processing."}
        except Exception as e:
            return {"error": f"Invalid image URL: {e}"}
        
    instruction = None
    if messages[-1].content[0].type == 'text':
        try : 
            instruction = messages[-1].content[0].text
        except Exception as e:
            return {"error": f"Invalid instruction text: {e}"}

    # Initialize model
    config = ExLlamaV2Config(model_directory)
    config.max_seq_len = 1024*32  # Pixtral default is 1M

    # Load vision model and multimodal projector and initialize preprocessor
    vision_model = ExLlamaV2VisionTower(config)
    vision_model.load(progress = True)

    # Load EXL2 model
    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, max_seq_len = 1024*32, lazy = True)
    model.load_autosplit(progress = True, cache = cache)
    tokenizer = ExLlamaV2Tokenizer(config)

    # Create generator
    generator = ExLlamaV2DynamicGenerator(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
    )

    if image_url is None:
        image_embeddings = []
    else : 
        image_embeddings = [
            vision_model.get_image_embeddings(
                model = model,
                tokenizer = tokenizer,
                image = get_image(image_url)
            )
        ]

    placeholders = "\n".join([ie.text_alias for ie in image_embeddings]) + "\n"

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
        max_new_tokens = max_tokens,
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