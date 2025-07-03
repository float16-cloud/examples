import json
import os
import uvicorn
from transformers import AutoTokenizer
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from llama_cpp.llama import Llama, LlamaGrammar

app = FastAPI()

@app.post("/chat/completions")
async def _chat_completions(request : Request):
    body = await request.json()
    model_name = "scb10x/llama3.2-typhoon2-3b-instruct"
    model_path = "../gguf-model/my-typhoon2.gguf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_tokens = body.get("max_tokens", 1024)        
    response_format = body.get("response_format", None)

    messages = body.get("messages", [])
    inference_engine = Llama(model_path, n_gpu_layers=-1, verbose=False, n_ctx=1024 * 128, seed=42)

    kwargs = {
        "conversation": messages,
        "tokenize": False,
        "add_generation_prompt": True,
    }

    prompt = tokenizer.apply_chat_template(**kwargs)

    if response_format is not None : 
        json_string = json.dumps(response_format, indent=4)
        json_string = json_string.replace("'", '"').replace("True", "true").replace("False", "false").replace("json_schema","object")
        response_grammar = LlamaGrammar.from_json_schema(json_string)
        kwargs = {
            "prompt": prompt,
            "grammar": response_grammar,
            "max_tokens": max_tokens,
            "stream": False
        }
    else:
        kwargs = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": False
        }

    try:

        output = inference_engine(**kwargs)

        return JSONResponse(
            content=output,
            status_code=200
        )

    except Exception as e:
        return {"error": str(e), "model_name": model_name, "tokenizer": tokenizer, "inference_engine": inference_engine}, 400

async def main():
    config = uvicorn.Config(
        app, host="0.0.0.0", port=int(os.environ["PORT"])
    )
    server = uvicorn.Server(config)
    await server.serve()