import os
import time
from typing import Optional
import uuid 
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
import asyncio

start_load = time.time()
model_name = "../weight-llm/typhoon-8b"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side="left")

app = FastAPI()

class ChatRequest(BaseModel):
    messages: str
    max_token : Optional[int] = 512
    

def process_llm(batch_data, batch_id):
    global model
    batch_tokenized = []
    for data in batch_data : 
        _text_formated = [{"role": "user", "content": data}]
        _text_tokenized = tokenizer.apply_chat_template(
            _text_formated,
            tokenize=False,
            add_generation_prompt=True
        )
        batch_tokenized.append(_text_tokenized)

    model_inputs = tokenizer(batch_tokenized, return_tensors="pt",padding=True,truncation=True).to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    result_list = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    result_with_id = dict(zip(batch_id,result_list))
    return result_with_id

class BatchProcessor:
    def __init__(self):
        self.batch = []
        self.batch_id = []
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()
        
    async def add_to_batch(self, data, batch_id):
        async with self.lock:
            self.batch.append(data)
            self.batch_id.append(batch_id)

    async def process_batch(self):
        while True:
            await asyncio.sleep(1)  # Wait for 1 second
            async with self.lock:
                current_batch = self.batch.copy()
                current_batch_id = self.batch_id.copy()
                self.batch.clear()
                self.batch_id.clear()

            if current_batch:
                self.results = process_llm(current_batch,current_batch_id)
                self.event.set()
                self.event.clear()

    async def get_result(self, batch_id):
        return self.results[batch_id]

main_batch = BatchProcessor()

@app.post("/chat")
async def chat(text_request: ChatRequest):
    batch_id = uuid.uuid4()
    await main_batch.add_to_batch(text_request.messages, batch_id)
    await main_batch.event.wait()
    result_text = await main_batch.get_result(batch_id)
    return JSONResponse(content={"response": result_text})

async def main():
    asyncio.create_task(main_batch.process_batch())
    config = uvicorn.Config(
        app, host="0.0.0.0", port=int(os.environ["PORT"])
    )
    server = uvicorn.Server(config)
    await server.serve()