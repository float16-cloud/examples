import json
import base64
import time
import requests
import llama_cpp
from PIL import Image
from io import BytesIO
from typing import List
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
from datamodel.request_message import (
    MessageInput,
    ModelParamsOpenAI,
    EmbeddingParamsOpenAI,
)


def get_image(url, return_bytes=False):
    try:
        return Image.open(requests.get(url, stream=True).raw)
    except Exception as e:
        if url.startswith("data:image"):
            base64_data = url.split(",")[1]
            image_data = base64.b64decode(base64_data)
            img = Image.open(BytesIO(image_data))

            # Calculate new dimensions divisible by 28
            original_width, original_height = img.size
            new_width = ((original_width + 27) // 28) * 28
            new_height = ((original_height + 27) // 28) * 28

            # Create a new white image with the same mode as the original
            mode = img.mode
            if mode == "RGBA":
                new_img = Image.new(mode, (new_width, new_height), (255, 255, 255, 255))
            elif mode == "L":
                new_img = Image.new(mode, (new_width, new_height), 255)
            else:
                new_img = Image.new(mode, (new_width, new_height), (255, 255, 255))

            # Paste the original image at (0, 0)
            new_img.paste(img, (0, 0))
            if return_bytes:
                buffer = BytesIO()
                new_img.save(buffer, format="PNG")
                return buffer.getvalue()
            else:
                return new_img
        else:
            raise ValueError("Invalid image URL or data format.")


def extract_request_params(message_request: ModelParamsOpenAI):
    stream = False
    json_output = False
    tools = None
    tool_choice = None
    messages = None
    reasoning_effort = None
    model_path = None

    if message_request.messages is not None:
        messages: MessageInput = message_request.messages
        messages = [json.loads(message.model_dump_json()) for message in messages]

    if message_request.tools is not None:
        tools = message_request.tools
        tool_choice = message_request.tool_choice
        if message_request.tool_choice is None:
            tool_choice = "auto"

    if message_request.response_format is not None:
        json_output = True

    if message_request.reasoning_effort is not None:
        reasoning_effort = message_request.reasoning_effort

    if message_request.stream is not None:
        stream = message_request.stream

    if message_request.messages is None:
        return {"error": "messages is None"}

    if message_request.model is not None:
        chat_template = select_chat_template(message_request.model)

    if message_request.model is not None:

        model_to_path = {
            ####### TEXT MODELS ########
            "gemma3-27b": "/share_weights/Gemma3-27B-GGUF/gemma-3-27b-it-q4_0.gguf",
            "gemma3-12b": "/share_weights/Gemma3-12B-GGUF/gemma-3-12b-it-q4_0.gguf",
            "typhoon-gemma3": "/share_weights/typhoon2.1-gemma3-12b-gguf/typhoon2.1-gemma3-12b-q4_k_m.gguf",
            "qwen3-32b-fast": "/share_weights/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf",
            "qwen3-32b-128k": "/share_weights/Qwen3-32B-GGUF/Qwen3-32B-128K-Q8_0.gguf",
            "qwen3-32b": "/share_weights/Qwen3-32B-GGUF/Qwen3-32B-Q8_0.gguf",
            "qwen3-14b-128k-fast": "/share_weights/Qwen3-14B-GGUF/Qwen3-14B-128K-Q4_K_M.gguf",
            "qwen3-14b-128k": "/share_weights/Qwen3-14B-GGUF/Qwen3-14B-128K-Q8_0.gguf",
            "qwen3-14b-fast": "/share_weights/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf",
            "qwen3-14b": "/share_weights/Qwen3-14B-GGUF/Qwen3-14B-Q8_0.gguf",
            "qwen3-a3b-fast": "/share_weights/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf",
            "qwen3-a3b": "/share_weights/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf",
            "qwen3-8b-128k-fast": "/share_weights/Qwen3-8B-GGUF/Qwen3-8B-128K-Q4_K_M.gguf",
            "qwen3-8b-128k": "/share_weights/Qwen3-8B-GGUF/Qwen3-8B-128K-Q8_0.gguf",
            "qwen3-8b-fast": "/share_weights/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
            "qwen3-8b": "/share_weights/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf",
            "qwen3-4b-fast": "/share_weights/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf",
            "qwen3-4b": "/share_weights/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf",
            ####### VISION MODELS ########
            "qwen2.5-vl-7b-gguf": "/share_weights/Qwen2.5-VL-7B-Instruct-GGUF/",
            "qwen2.5-vl-7b": "/share_weights/Qwen2.5-VL-7B-Instruct-exl2/",
            "qwen2.5-vl-32b": "/share_weights/Qwen2.5-VL-32B-Instruct-exl2/",
            "ui-tars-1.5-7b": "/share_weights/UI-TARS-1.5-7B-exl2/",
            "typhoon-ocr-7b": "/share_weights/typhoon-ocr-7b/",
        }

        engine_type = {
            "qwen2.5-vl-7b-gguf": "llamacpp-vision",
            "qwen2.5-vl-7b": "exllamav2",
            "qwen2.5-vl-32b": "exllamav2",
            "ui-tars-1.5-7b": "exllamav2",
            "typhoon-ocr-7b": "hf",
        }

        model_request = message_request.model.lower()
        model_path = model_to_path.get(model_request, None)
        model_type = engine_type.get(model_request, None)

        if model_type is None:
            model_type = "llamacpp"

        if model_path is None:
            raise ValueError(f"Unsupported model: {message_request.model}")

    return (
        messages,
        chat_template,
        tools,
        tool_choice,
        json_output,
        stream,
        reasoning_effort,
        model_path,
        model_type,
    )


def extract_embedding_request_params(request: EmbeddingParamsOpenAI):
    if request.input is None:
        raise ValueError("input is None")

    if request.model is not None:
        model_to_path = {
            "bge-m3": "/share_weights/Bge-m3-GGUF/bge-m3-Q8_0.gguf",
            "qwen3-embedding-0.6b": "/share_weights/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf",
            "qwen3-embedding-4b": "/share_weights/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q8_0.gguf",
            "qwen3-embedding-8b": "/share_weights/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf",
        }

        model_request = request.model.lower()
        embedding_path = model_to_path.get(model_request, None)

        if embedding_path is None:
            raise ValueError(f"Unsupported model: {request.model}")

        model_to_pooling_layer = {
            "bge-m3": -1,
            "qwen3-embedding-0.6b": 3,
            "qwen3-embedding-4b": 3,
            "qwen3-embedding-8b": 3,
        }

        pooling_layer = model_to_pooling_layer.get(model_request, None)

    inputs = request.input
    model = request.model

    return inputs, model, request.encoding_format, embedding_path, pooling_layer


def select_chat_template(model_name: str) -> str:
    if "qwen2.5-vl" in model_name.lower():
        return "qwen2.5-vl"
    elif "qwen" in model_name.lower():
        return "qwen"
    elif "typhoon-gemma3" in model_name.lower():
        return "typhoon-gemma3"
    elif "gemma" in model_name.lower():
        return "gemma"
    elif "llama-3" in model_name.lower():
        return "llama-3"
    elif "ui-tars" in model_name.lower():
        return "qwen"
    elif "moondream2" in model_name.lower():
        return "moondream2"
    elif "typhoon-ocr-7b" in model_name.lower():
        return "qwen"
    else:
        return "default"


def format_messages_to_prompt(
    messages,
    chat_template: str,
    tools=None,
    reasoning_effort=None,
    model_type="llamacpp",
    model=None,
    vision_model=None,
    tokenizer=None,
) -> str:

    if model_type == "llamacpp":
        if chat_template == "qwen":
            prompt = format_qwen(messages, tools, reasoning_effort)
        if chat_template == "gemma":
            prompt = format_gemma(messages, tools)
        if chat_template == "typhoon-gemma3":
            prompt = format_typhoon_gemma(messages, tools, reasoning_effort)
        return prompt, None

    elif model_type == "llamacpp-vision":
        if chat_template == "qwen2.5-vl":
            prompt = format_qwen25_vl(messages, tools, reasoning_effort)
        return prompt, None

    elif model_type == "exllamav2":
        if chat_template == "qwen":
            prompt, image_embeddings = format_qwen_vision(
                messages, model, vision_model, tokenizer
            )
            return prompt, image_embeddings

    elif model_type == "hf":
        if chat_template == "qwen":
            prompt, image_embeddings, processor = format_qwen_vision_hf(messages)
            return prompt, processor


def format_qwen_vision_hf(messages) -> str:
    from transformers import AutoProcessor

    model_path = "/share_weights/typhoon-ocr-7b/"
    device = "cuda"

    processor = AutoProcessor.from_pretrained(model_path)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    main_image = None
    for m in messages:
        if m["role"] == "user":
            if isinstance(m["content"], list):
                for item in m["content"]:
                    if item["type"] == "image_url":
                        image_url = item["image_url"]["url"]
                        main_image = get_image(image_url)
                        break
            elif isinstance(m["content"], str):
                continue

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for (key, value) in inputs.items()}
    return inputs, None, processor


def format_qwen_vision(messages, model, vision_model, tokenizer) -> str:
    image_embeddings = []
    prompt = ""
    for m in messages:
        if m["role"] == "system":
            prompt += f"<|im_start|>system\n{m['content']}<|im_end|>\n"
        elif m["role"] == "user":
            if isinstance(m["content"], str):
                prompt += f"<|im_start|>user\n{m['content']}<|im_end|>\n"
            elif isinstance(m["content"], list):
                for item in m["content"]:
                    if item["type"] == "image_url":
                        image_url = item["image_url"]["url"]
                        placeholders = vision_model.get_image_embeddings(
                            model=model, tokenizer=tokenizer, image=get_image(image_url)
                        )
                        image_embeddings.append(placeholders)
                        prompt += (
                            f"<|im_start|>user\n{placeholders.text_alias}<|im_end|>\n"
                        )

                    if item["type"] == "text":
                        prompt += f"<|im_start|>user\n{item['text']}<|im_end|>\n"

        elif m["role"] == "assistant":
            prompt += f"<|im_start|>assistant\n{m['content']}<|im_end|>\n"

    prompt += "<|im_start|>assistant\n"
    return prompt, image_embeddings


def format_qwen(messages, tools=None, reasoning_effort=None) -> str:
    from transformers import AutoTokenizer

    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    enable_thinking = reasoning_effort is not None

    kwargs = {
        "conversation": messages,
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if tools is not None:
        kwargs["tools"] = tools

    text = tokenizer.apply_chat_template(**kwargs)
    return text


def format_qwen25_vl(messages, tools=None, reasoning_effort=None) -> str:
    from transformers import AutoProcessor

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name)

    # Padding white image for vision model
    for m in messages:
        if m["role"] == "user":
            if isinstance(m["content"], list):
                for item in m["content"]:
                    if item["type"] == "image_url":
                        image_url = item["image_url"]["url"]
                        image_bytes = get_image(image_url, return_bytes=True)
                        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                        prefix_base64 = "data:image/png;base64,"
                        new_image_url = prefix_base64 + image_base64
                        item["image_url"]["url"] = new_image_url
                        break
            elif isinstance(m["content"], str):
                continue

    kwargs = {
        "conversation": messages,
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if tools is not None:
        kwargs["tools"] = tools

    text = processor.apply_chat_template(**kwargs)
    return text


def format_gemma(messages, tools=None) -> str:
    from transformers import AutoTokenizer

    model_name = "unsloth/gemma-3-1b-it"  # Prevent gated model access
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    kwargs = {
        "conversation": messages,
        "tokenize": False,
        "add_generation_prompt": True,
    }

    messages = add_gemma_tools(messages, tools)
    text = tokenizer.apply_chat_template(**kwargs)
    return text


def add_gemma_tools(messages, tools: List[dict]) -> str:
    if not tools:
        return messages

    user_message = messages[-1]["content"]
    tools_prompt = """You have access to functions.
You can call these functions to get information or answer without calling the function.
You are Thai assistant, so you should answer in Thai.

If you decide to invoke any of the function(s),
you MUST put it in the format of
{"name": function name, "parameters": dictionary of argument name and its value}

You SHOULD NOT include any other text in the response if you call a function
[
"""
    for tool in tools:
        tool_text = json.dumps(tool, indent=4)
        tools_prompt += f"{tool_text},\n"
    tools_prompt += "]\n"

    messages[-1]["content"] = tools_prompt + user_message
    return messages


def format_typhoon_gemma(messages, tools=None, reasoning_effort=None) -> str:
    from transformers import AutoTokenizer

    model_name = "scb10x/typhoon2.1-gemma3-12b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    enable_thinking = reasoning_effort is not None

    kwargs = {
        "conversation": messages,
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
        "tools": None,
    }

    text = tokenizer.apply_chat_template(**kwargs)
    return text


def get_inference_engine(model_path, model_type="llamacpp"):

    kwargs = {
        "model_path": model_path,
        "n_gpu_layers": -1,
        "verbose": False,
        "n_ctx": 1024 * 40,
        "seed": 1,
        "flash_attn": True,
        "n_threads": 8,
    }

    if model_type == "llamacpp":
        if "128k" in model_path.lower():
            kwargs["rope_scaling_type"] = llama_cpp.LLAMA_ROPE_SCALING_TYPE_YARN
            kwargs["yarn_orig_ctx"] = 1024 * 32
            kwargs["n_ctx"] = 1024 * 32 * 4  # Set context length to 128k
            kwargs["draft_model"] = LlamaPromptLookupDecoding(
                num_pred_tokens=4, max_ngram_size=2
            )  # Use soft-ngram decoding to accelerate inference

        return Llama(**kwargs), None, None, None

    elif model_type == "llamacpp-vision":
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler

        chat_handler = Qwen25VLChatHandler(
            clip_model_path=f"{model_path}/mmproj-BF16.gguf", verbose=True
        )
        kwargs["chat_handler"] = chat_handler
        kwargs["model_path"] = f"{model_path}/Qwen2.5-VL-7B-Instruct-Q8_0.gguf"
        kwargs["n_ctx"] = 1024 * 32

        return Llama(**kwargs), None, None, None

    elif model_type == "hf":
        from transformers import Qwen2_5_VLForConditionalGeneration

        device = "cuda"

        model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
            .eval()
            .to(device)
        )

        return model, None, None, None

    elif model_type == "exllamav2":
        from exllamav2 import (
            ExLlamaV2,
            ExLlamaV2Config,
            ExLlamaV2Cache,
            ExLlamaV2Tokenizer,
            ExLlamaV2VisionTower,
        )

        from exllamav2.generator import ExLlamaV2DynamicGenerator

        # Initialize model
        config = ExLlamaV2Config(model_path)
        config.max_seq_len = 1024 * 32
        config.vision_min_pixels = 1 * 28 * 28  # Minimum image size for vision models
        config.vision_max_pixels = (
            2560 * 28 * 28
        )  # Maximum image size for vision models
        config.no_graphs = False  # Disable graph mode for vision models

        vision_model = ExLlamaV2VisionTower(config)
        vision_model.load(progress=True)

        # Load EXL2 model
        model = ExLlamaV2(config)
        cache = ExLlamaV2Cache(model, max_seq_len=1024 * 128, lazy=True)
        model.load_autosplit(progress=True, cache=cache)
        tokenizer = ExLlamaV2Tokenizer(config)

        # Create generator
        generator = ExLlamaV2DynamicGenerator(
            model=model, cache=cache, tokenizer=tokenizer
        )
        return generator, vision_model, model, tokenizer
