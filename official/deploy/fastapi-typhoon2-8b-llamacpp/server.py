import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn

app = FastAPI()

class ChatRequest(BaseModel):
    messages: str

@app.post("/chat/")
async def chat(text_request: ChatRequest):
    llm = Llama(
        model_path="../model/typhoon-8b-cpp/llama3.1-typhoon2-8b-instruct-q8_0.gguf",
        n_gpu_layers=-1,
        verbose=False,
        chat_format='llama-3'
    )

    output = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": "You are an assistant who perfectly describes images."},
            {
                "role": "user",
                "content": text_request.messages
            }
        ]
    )

    return JSONResponse(content=output['choices'][0]['message'])

async def main():
    config = uvicorn.Config(
        app, host="0.0.0.0", port=int(os.environ["PORT"])
    )
    server = uvicorn.Server(config)
    await server.serve()