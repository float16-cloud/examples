import json
import os
import uvicorn
from llama_cpp.llama import Llama
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from datamodel.request_message import EmbeddingParamsOpenAI, ModelParamsOpenAI
from function.postprocess import generate_response, tools_generate_response
from function.preprocess import (
    extract_embedding_request_params,
    format_messages_to_prompt,
    extract_request_params,
    get_inference_engine,
)
from function.process import completion

app = FastAPI()


@app.post("/chat/completions")
async def _chat_completions(request: Request):
    body = await request.json()
    request = ModelParamsOpenAI(**body)

    (
        formatted_messages,
        chat_template,
        tools,
        tool_choice,
        json_output,
        stream,
        reasoning_effort,
        model_path,
        model_type,
    ) = extract_request_params(request)

    inference_engine, vision_model, model, tokenizer = get_inference_engine(
        model_path=model_path, model_type=model_type
    )

    if model_type == "llamacpp":
        prompt, image_embeddings = format_messages_to_prompt(
            messages=formatted_messages,
            chat_template=chat_template,
            tools=tools,
            reasoning_effort=reasoning_effort,
            model_type=model_type,
        )

    elif model_type == "llamacpp-vision":  # experimental
        prompt, image_embeddings = format_messages_to_prompt(
            messages=formatted_messages,
            chat_template=chat_template,
            tools=tools,
            reasoning_effort=reasoning_effort,
            model_type=model_type,
        )

    elif model_type == "exllamav2":
        prompt, image_embeddings = format_messages_to_prompt(
            messages=formatted_messages,
            chat_template=chat_template,
            tools=tools,
            reasoning_effort=reasoning_effort,
            model_type=model_type,
            model=model,
            vision_model=vision_model,
            tokenizer=tokenizer,
        )

    elif model_type == "hf":
        prompt, image_embeddings = format_messages_to_prompt(
            messages=formatted_messages,
            chat_template=chat_template,
            tools=tools,
            reasoning_effort=reasoning_effort,
            model_type=model_type,
            model=model,
        )

    if json_output:  # If response_format is provided, we assume JSON output
        stream = False
    try:
        if stream and tools:
            output = completion(
                llm=inference_engine,
                chat_template=chat_template,
                prompt=prompt,
                stream=stream,
                max_tokens=request.max_tokens,
            )
            return StreamingResponse(
                tools_generate_response(output, chat_template),
                media_type="text/event-stream",
            )
        elif stream:
            output = completion(
                llm=inference_engine,
                chat_template=chat_template,
                prompt=prompt,
                stream=stream,
                max_tokens=request.max_tokens,
            )
            return StreamingResponse(
                generate_response(output), media_type="text/event-stream"
            )
        else:
            return JSONResponse(
                content=completion(
                    inference_engine=inference_engine,
                    chat_template=chat_template,
                    prompt=prompt,
                    stream=False,
                    json_output=json_output,
                    response_format=request.response_format,
                    tool_choice=tool_choice,
                    max_tokens=request.max_tokens,
                    model_type=model_type,
                    tokenizer=tokenizer,
                    image_embeddings=image_embeddings,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    messages=formatted_messages,
                ),
                status_code=200,
            )
    except Exception as e:
        return {"error": str(e)}


@app.post("/embeddings")
async def _embeddings(request: EmbeddingParamsOpenAI):

    inputs, _, _, EMBEDDING_PATH, pooling_layer = extract_embedding_request_params(
        request
    )

    embedding = Llama(
        model_path=EMBEDDING_PATH,
        n_gpu_layers=-1,
        verbose=False,
        n_ctx=1024 * 32,
        seed=42,
        embedding=True,
        pooling_layer=pooling_layer,
    )

    return embedding.create_embedding(inputs)


@app.get("/models")
async def _models():
    models = [
        {
            "id": "qwen3-32b-fast",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "text",
            "reasoning": True,
            "max_tokens": 32768,
            "context_length": 32768,
        },
        {
            "id": "qwen3-32b-128k",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "text",
            "reasoning": True,
            "max_tokens": 32768,
            "context_length": 131072,
        },
        {
            "id": "qwen3-32b",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "text",
            "reasoning": True,
            "max_tokens": 32768,
            "context_length": 32768,
        },
        {
            "id": "qwen3-14b-fast",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "text",
            "reasoning": True,
            "max_tokens": 32768,
            "context_length": 32768,
        },
        {
            "id": "qwen3-14b-128k",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "text",
            "reasoning": True,
            "max_tokens": 32768,
            "context_length": 131072,
        },
        {
            "id": "qwen3-14b",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "text",
            "reasoning": True,
            "max_tokens": 32768,
            "context_length": 32768,
        },
        {
            "id": "qwen3-8b-fast",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "text",
            "reasoning": True,
            "max_tokens": 32768,
            "context_length": 32768,
        },
        {
            "id": "qwen3-8b-128k",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "text",
            "reasoning": True,
            "max_tokens": 32768,
            "context_length": 131072,
        },
        {
            "id": "qwen3-8b",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "text",
            "reasoning": True,
            "max_tokens": 32768,
            "context_length": 32768,
        },
        {
            "id": "qwen3-4b-fast",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "text",
            "reasoning": True,
            "max_tokens": 32768,
            "context_length": 32768,
        },
        {
            "id": "qwen3-4b",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "text",
            "reasoning": True,
            "max_tokens": 32768,
            "context_length": 32768,
        },
        {
            "id": "qwen3-a3b-fast",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "text",
            "reasoning": True,
            "max_tokens": 32768,
            "context_length": 32768,
        },
        {
            "id": "qwen3-a3b",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "text",
            "reasoning": True,
            "max_tokens": 32768,
            "context_length": 32768,
        },
        {
            "id": "typhoon-gemma3",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "text",
            "reasoning": True,
            "max_tokens": 8192,
            "context_length": 131072,
        },
        {
            "id": "gemma3-27b",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "text",
            "reasoning": False,
            "max_tokens": 8192,
            "context_length": 131072,
        },
        {
            "id": "gemma3-12b",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "reasoning": False,
            "max_tokens": 8192,
            "context_length": 131072,
        },
        {
            "id": "qwen2.5-vl-7b",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "vlm",
            "reasoning": False,
            "max_tokens": 8192,
            "context_length": 32768,
        },
        {
            "id": "qwen2.5-vl-32b",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "vlm",
            "reasoning": False,
            "max_tokens": 8192,
            "context_length": 32768,
        },
        {
            "id": "bge-m3",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "embedding",
            "multilingual": True,
        },
        {
            "id": "qwen3-embedding-0.6b",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "embedding",
            "multilingual": True,
        },
        {
            "id": "qwen3-embedding-4b",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "embedding",
            "multilingual": True,
        },
        {
            "id": "qwen3-embedding-8b",
            "object": "model",
            "created": 1700000000,
            "owned_by": "float16",
            "type": "embedding",
            "multilingual": True,
        },
    ]

    return JSONResponse(content={"object": "list", "data": models}, status_code=200)


async def main():
    config = uvicorn.Config(app, host="0.0.0.0", port=int(os.environ["PORT"]))
    server = uvicorn.Server(config)
    await server.serve()
