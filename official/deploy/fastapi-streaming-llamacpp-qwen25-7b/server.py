import json
import os
from typing import Iterator, List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llama_cpp.llama import Llama
from pydantic import BaseModel

app = FastAPI()
MODEL_PATH = "../Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q8_0.gguf"

llm = Llama(
    model_path = MODEL_PATH,
    n_gpu_layers=-1,
    verbose=False,
    chat_format='qwen', # Qwen chat format
    n_ctx=1024 * 32,
    seed=42
)

class MessageInput(BaseModel):
    role: str
    content: str

class ModelParamsOpenAI(BaseModel):
    messages: Optional[List[MessageInput]] = None

def format_messages(messages: List[MessageInput]) -> List[dict]:
    return [{"role": message.role, "content": message.content} for message in messages]

def generate_response(output: Iterator) -> Iterator[str]:
    for response in output:
        if isinstance(response, bytes):
            response['choices'][0]['delta']['content'] = response['choices'][0]['delta']['content'].decode('utf-8')
        yield "data: "+ json.dumps(response) + "\n"
    yield "data: [DONE]"

@app.post("/chat/completions")
async def read_root(message_request : ModelParamsOpenAI):
    if message_request.messages is None:
        return {"error": "messages is None"}

    formatted_messages = format_messages(message_request.messages)

    output = llm.create_chat_completion(
        messages = formatted_messages,
        # response_format={ Uncomment this to use the json object format
        #     "type": "json_object",
        # },
        max_tokens=256,
        stream=True
    )


    if isinstance(output, Iterator):
        return StreamingResponse(generate_response(output), media_type="text/event-stream")
    else:
        return {"error": "Unexpected output format"}

async def main():
    config = uvicorn.Config(
        app, host="0.0.0.0", port=int(os.environ["PORT"])
    )
    server = uvicorn.Server(config)
    await server.serve()