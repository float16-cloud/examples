# fastapi-streaming-llamacpp-qwen25-7b

## Getting Started

```
float16 example official/deploy/fastapi-streaming-llamacpp-qwen25-7b

huggingface-cli download lmstudio-community/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q8_0.gguf --local-dir ./Qwen2.5-7B-Instruct-GGUF

float16 storage upload -f ./Qwen2.5-7B-Instruct-GGUF -d ""

float16 deploy server.py

curl -X POST "{FUNCTION-URL}/chat" -H "Authorization: Bearer {FLOAT16-ENDPOINT-TOKEN}" -H "Content-Type: application/json" -d '{ "messages": "ขอสูตรไก่ย่างหน่อย" }'
```

## Option A1: Multimodal Server with Qwen2.5-VL (`server_multimodal.py`)

This server uses the `Qwen/Qwen2.5-VL-7B-Instruct` model with the Hugging Face `transformers` library to provide multimodal (image and text) capabilities and an OpenAI-compatible API.

### Model Setup (Qwen2.5-VL-7B-Instruct)

To use this server, you first need to download the `Qwen/Qwen2.5-VL-7B-Instruct` model files.

1.  Ensure you have Git LFS installed:
    ```bash
    git lfs install
    ```

2.  Clone the model repository from Hugging Face:
    ```bash
    git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
    ```
3.  By default, `server_multimodal.py` will expect this model to be in a directory named `Qwen2.5-VL-7B-Instruct` at the same level as the `fastapi-streaming-llamacpp-qwen25-7b` project directory (i.e., `../Qwen2.5-VL-7B-Instruct`). You may need to adjust paths in the script if you place it elsewhere.

### Dependencies (Multimodal Server)

Install the required Python packages:
```bash
pip install -r requirements.txt 
```
(Ensure the `requirements.txt` contains the dependencies listed above for `server_multimodal.py`)

## Option B: Text-Optimized Server with Qwen & exllamav2 (`server_exllamav2.py`)

This server uses a Qwen 7B model (text-focused) with the `exllamav2` library for potentially faster text generation. It aims to provide an OpenAI-compatible API for chat and tool usage. Multimodal (image) input is **not reliably supported** with this setup.

### Model Setup (Qwen 7B for exllamav2 - EXL2/GPTQ format)

`exllamav2` requires models to be in its own EXL2 format or in GPTQ format. It does **not** directly use GGUF models like the one originally mentioned in this project (`Qwen2.5-7B-Instruct-Q8_0.gguf`).

You have a few options:

1.  **Find Pre-converted Models:** Search for Qwen 7B models (e.g., Qwen1.5-7B, or text-based Qwen2.5-7B variants if available) in EXL2 or GPTQ format on Hugging Face. Creators like "TheBloke", "turboderp", "LoneStriker", and "bartowski" often provide such quantized models.
    *   Example search: [https://huggingface.co/models?search=qwen%207b%20exl2](https://huggingface.co/models?search=qwen%207b%20exl2)
    *   Look for models compatible with `exllamav2` version 0.0.11 or higher for broader compatibility.

2.  **Convert a GGUF or HF model:** `exllamav2` provides a `convert.py` script in its repository ([https://github.com/turboderp/exllamav2](https://github.com/turboderp/exllamav2)) that can convert Hugging Face models (FP16/BF16) or certain other formats to EXL2. Converting GGUF directly might require an intermediate step to a compatible Hugging Face format first. This process can be technical.

3.  **Experimental - Multimodal GPTQ with `exllamav2`:**
    The `Qwen/Qwen2.5-VL-7B-Instruct` model *might* have a GPTQ variant available on Hugging Face (e.g., `Qwen/Qwen2.5-VL-7B-Instruct-GPTQ`). While `exllamav2` can load some GPTQ models, it is **highly unlikely to support its multimodal capabilities**. If you attempt this, expect it to function as a text-only model, and it may not load or run correctly. This is purely for experimental purposes as per user interest.

Place the downloaded/converted model in a directory accessible by `server_exllamav2.py` (e.g., `../Qwen-7B-EXL2`). You will need to set the model path in `server_exllamav2.py`.

### Dependencies (exllamav2 Server)

Ensure the dependencies for `server_multimodal.py` are installed (especially `fastapi`, `uvicorn`, `pydantic`). Then, for `exllamav2`, ensure the following are met (refer to `requirements.txt`):
*   `exllamav2`
*   `ninja`
*   A compatible version of PyTorch (e.g., 2.1+). Check `exllamav2` documentation for specifics related to your CUDA version. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (replace `cu118` with your CUDA version e.g. `cu121`).

Install requirements:
```bash
pip install -r requirements.txt
```

## Description

This example demonstrates how to deploy a FastAPI server with a Qwen2.5 7b model using Llamacpp.

This server **support** streaming response and JSON response.

## Libraries 

- None

## GPU Configuration

- H100

## Expected Performance

- Return response in less than 5 seconds

## Profile

- [X - Matichon](https://x.com/KMatiDev1)
- [Matichon - Personal website](https://matichon.me)
- Email: matichon[dot]man[at]float16[dot]cloud
- Open for Work: No