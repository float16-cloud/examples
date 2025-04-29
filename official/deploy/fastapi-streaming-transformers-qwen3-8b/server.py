import json
import os
import time
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import  StreamingResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
from typing import List, Optional

start_load = time.time()
model_name = "../weight-llm/Qwen3-8B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

app = FastAPI()

class MessageInput(BaseModel):
    role: str
    content: str

class ModelParamsOpenAI(BaseModel):
    messages: Optional[List[MessageInput]] = None
    max_token: Optional[int] = 1024

def process_llm(message,max_token):
    global model
    _text_formated = message
    _text_tokenized = tokenizer.apply_chat_template(
        _text_formated,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer.encode(_text_tokenized, return_tensors="pt").to(model.device)
    original_inputs = tokenizer.encode(_text_tokenized, return_tensors="pt").to(model.device)

    count_token = 0
    start = time.time()
    sum_text = ""
    buffer_text = []
    text_len_list = []
    recent_text_len = len(_text_tokenized)
    max_buffer_size = 5
    current_text_len = 0
    print(_text_tokenized)
    print(len(_text_tokenized))

    for i in range(max_token):  # Adjust the range for desired output length
        with torch.no_grad():
            output = model.generate(
                model_inputs,
                max_new_tokens=1,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )


        new_token = output[0][-1]
        new_token_unsqueezed = new_token.unsqueeze(0).unsqueeze(0)
        model_inputs = torch.cat((model_inputs, new_token_unsqueezed), dim=1)
        if new_token.item() == tokenizer.eos_token_id:
            break

        decoded_token = tokenizer.decode(new_token)
        
        decoded_text = tokenizer.batch_decode(model_inputs)[0]
        current_text_len = len(decoded_text)
        text_len_list.append(current_text_len)
        count_token += 1
        print(f"Line : {count_token} , Current text len: {current_text_len}, {decoded_token}")
        if len(buffer_text) < max_buffer_size:
            if recent_text_len == current_text_len:  # Edge case
                try:
                    buffer_text[len(buffer_text)-1] = decoded_text[text_len_list[count_token-3]:]
                except IndexError:
                    buffer_text[len(buffer_text)-1] = decoded_text[text_len_list[count_token-2]:]
            else:
                buffer_text.append(decoded_text[recent_text_len:])
                recent_text_len = current_text_len
        else:
            if recent_text_len == current_text_len:  # handle the case where the text is not complete in one line #TODO: Edit to handle more than 1 line 
                buffer_text[len(buffer_text)-1] = decoded_text[text_len_list[count_token-3]:]
            
            decoded_token = buffer_text[0]
            buffer_text.pop(0)
            buffer_text.append(decoded_text[recent_text_len:])
            recent_text_len = current_text_len
            print(f"Buffer text: {decoded_token}")
            sum_text += decoded_token
            response = {
                "id": 1,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "delta": {
                            "role": 'assistant',
                            "content": decoded_token
                        },
                        "index": 0,
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(response)}\n"
    

    if buffer_text:
        final_text = "".join(buffer_text)
        sum_text += final_text
        response = {
                "id": 1,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "delta": {
                            "role": 'assistant',
                            "content": final_text
                        },
                        "index": 0,
                        "finish_reason": None
                    }
                ]
            }

        yield f"data: {json.dumps(response)}\n"
    
    print(f"Token count: {count_token}")
    print(f"Total time: {time.time() - start} seconds")
    print(f"Average token per sec: {count_token / (time.time() - start)} seconds")
    yield "data: [DONE]\n\n"

@app.post("/chat")
async def chat(text_request: ModelParamsOpenAI):
    return StreamingResponse(process_llm(text_request.messages,text_request.max_token), media_type="text/event-stream")

async def main():
    config = uvicorn.Config(
        app, host="0.0.0.0", port=int(os.environ["PORT"])
    )
    server = uvicorn.Server(config)
    await server.serve()