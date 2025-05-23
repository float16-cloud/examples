import os
import json
import asyncio
import uuid
import time
from typing import List, Optional, Union # Optional needs to be imported from typing

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field 

# exllamav2 imports
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import ExLlamaV2StreamingGenerator
from exllamav2.generator import ExLlamaV2Sampler 

import uvicorn

# --- Configuration & Model Loading ---
app = FastAPI()

MODEL_DIRECTORY = os.environ.get("EXLLAMAV2_MODEL_DIRECTORY", "../Qwen-7B-EXL2") 

if not os.path.exists(MODEL_DIRECTORY):
    if MODEL_DIRECTORY == "../Qwen-7B-EXL2":
        error_message = (
            f"Model directory not found: {MODEL_DIRECTORY}. "
            "This is the default path. Please ensure a model in EXL2 or compatible GPTQ format "
            "is present at this location, or set the EXLLAMAV2_MODEL_DIRECTORY environment variable "
            "to point to your model's directory."
        )
    else:
        error_message = (
            f"Model directory not found: {MODEL_DIRECTORY}. "
            "Please check the path provided via the EXLLAMAV2_MODEL_DIRECTORY environment variable."
        )
    raise FileNotFoundError(error_message)

config = ExLlamaV2Config()
config.model_dir = MODEL_DIRECTORY
config.prepare() 

config.max_batch_size = 1 

model = ExLlamaV2(config)
print(f"Loading model: {MODEL_DIRECTORY}")
model.load() 
print(f"Model {MODEL_DIRECTORY} loaded successfully.")

tokenizer = ExLlamaV2Tokenizer(config)
print("Tokenizer initialized.")

cache = ExLlamaV2Cache(model) 
generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
print("ExLlamaV2StreamingGenerator initialized.")
QWEN_EXL2_MODEL_NAME = "qwen-7b-exl2" 

# --- OpenAI-compatible Pydantic Models ---
class OpenAIMessageContentItemText(BaseModel):
    type: str = "text"
    text: str

class OpenAIMessage(BaseModel):
    role: str
    content: Union[str, List[OpenAIMessageContentItemText]]
    name: Optional[str] = None

class FunctionParameters(BaseModel):
    type: str = "object"
    properties: dict
    required: Optional[List[str]] = None

class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: FunctionParameters

class Tool(BaseModel):
    type: str = "function"
    function: FunctionDefinition

class OpenAIChatCompletionRequest(BaseModel):
    model: Optional[str] = QWEN_EXL2_MODEL_NAME
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.9
    token_repetition_penalty: Optional[float] = 1.05
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, dict]] = Field(default="none")

class FunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None 

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall
    
class OpenAIChatCompletionStreamResponseDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

class OpenAIChatCompletionStreamResponseChoice(BaseModel):
    index: int = 0
    delta: OpenAIChatCompletionStreamResponseDelta
    finish_reason: Optional[str] = None

class OpenAIChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAIChatCompletionStreamResponseChoice]

def format_prompt_for_qwen_exl2(messages: List[OpenAIMessage], tools: Optional[List[Tool]], tool_choice: Union[str, dict]) -> str:
    prompt_parts = []
    if tools and tool_choice != "none":
        tools_description_parts = ["You have access to the following tools:"]
        tools_json = [tool.model_dump(exclude_none=True) for tool in tools]
        tools_description_parts.append(json.dumps(tools_json, indent=2))
        tools_description_parts.append("To use a tool, respond *only* with a JSON object matching the following schema, enclosed in triple backticks:")
        tools_description_parts.append("```json\n" + '''{
  "tool_calls": [
    {
      "id": "call_...",
      "type": "function",
      "function": {
        "name": "function_name",
        "arguments": "{...}" 
      }
    }
  ]
}''' + "\n```")
        tools_description_parts.append("If you are not using a tool, respond with text as usual.")
        prompt_parts.append("SYSTEM: " + "\n".join(tools_description_parts))

    for msg in messages:
        role = msg.role.upper()
        content_str = ""
        if isinstance(msg.content, str):
            content_str = msg.content
        elif isinstance(msg.content, list): 
            content_str = " ".join([item.text for item in msg.content if item.type == "text"])
        
        if role == "SYSTEM" and tools and tool_choice != "none" and prompt_parts and prompt_parts[0].startswith("SYSTEM:"):
            if not prompt_parts[0].endswith(content_str): 
                existing_system_prompt = prompt_parts[0][len("SYSTEM: "):]
                prompt_parts[0] = f"SYSTEM: {existing_system_prompt}\n{content_str}"
        else:
            prompt_parts.append(f"{role}: {content_str}")

    prompt_parts.append("ASSISTANT:")
    return "\n".join(prompt_parts)

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatCompletionRequest):
    full_prompt = format_prompt_for_qwen_exl2(request.messages, request.tools, request.tool_choice)
    input_ids = tokenizer.encode(full_prompt)
    if input_ids.shape[-1] >= config.max_seq_len: 
        raise HTTPException(
            status_code=400, 
            detail=f"Prompt is too long ({input_ids.shape[-1]} tokens) after formatting. Maximum sequence length is {config.max_seq_len} tokens."
        )

    sampler_settings = ExLlamaV2Sampler.Settings()
    sampler_settings.temperature = request.temperature
    sampler_settings.top_k = request.top_k
    sampler_settings.top_p = request.top_p
    sampler_settings.token_repetition_penalty = request.token_repetition_penalty

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_timestamp = int(time.time())
    current_model_name = request.model if request.model else QWEN_EXL2_MODEL_NAME

    async def stream_openai_response():
        accumulated_text = ""
        tool_call_detected_and_sent = False
        finish_reason_val = "stop" 
        
        initial_delta = OpenAIChatCompletionStreamResponseDelta(role="assistant", content=None)
        initial_choice = OpenAIChatCompletionStreamResponseChoice(index=0, delta=initial_delta, finish_reason=None)
        initial_chunk = OpenAIChatCompletionStreamResponse(
            id=request_id, created=created_timestamp, model=current_model_name, choices=[initial_choice]
        )
        yield f"data: {initial_chunk.model_dump_json(exclude_none=True)}\n\n"

        generator.begin_stream(input_ids, sampler_settings)
        generated_tokens_count = 0
        
        for i in range(request.max_tokens): 
            if tool_call_detected_and_sent: break 
            chunk_text, eos, _ = generator.stream()
            if not chunk_text and not eos: continue 
            if not chunk_text and eos: 
                finish_reason_val = "stop"
                break
            generated_tokens_count += 1 
            accumulated_text += chunk_text

            if request.tools and request.tool_choice != "none":
                json_block_start = accumulated_text.find("```json")
                if json_block_start != -1:
                    json_block_end = accumulated_text.find("```", json_block_start + 7)
                    if json_block_end != -1:
                        json_str_to_parse = accumulated_text[json_block_start + 7 : json_block_end].strip()
                        try:
                            parsed_json = json.loads(json_str_to_parse)
                            if "tool_calls" in parsed_json and isinstance(parsed_json["tool_calls"], list):
                                tool_calls_list = []
                                for tc_data in parsed_json["tool_calls"]:
                                    raw_arguments = tc_data.get("function", {}).get("arguments", {})
                                    stringified_args = json.dumps(raw_arguments) if isinstance(raw_arguments, dict) else str(raw_arguments)
                                    tool_calls_list.append(ToolCall(
                                        id=tc_data.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                        type="function",
                                        function=FunctionCall(
                                            name=tc_data.get("function", {}).get("name"),
                                            arguments=stringified_args
                                        )
                                    ))
                                if tool_calls_list:
                                    delta = OpenAIChatCompletionStreamResponseDelta(tool_calls=tool_calls_list, content=None)
                                    choice = OpenAIChatCompletionStreamResponseChoice(index=0, delta=delta, finish_reason="tool_calls")
                                    chunk_response = OpenAIChatCompletionStreamResponse(
                                        id=request_id, created=created_timestamp, model=current_model_name, choices=[choice]
                                    )
                                    yield f"data: {chunk_response.model_dump_json(exclude_none=True)}\n\n"
                                    tool_call_detected_and_sent = True
                                    finish_reason_val = "tool_calls"
                                    accumulated_text = "" 
                                    break 
                        except json.JSONDecodeError:
                             if accumulated_text.strip().endswith("```"):
                                 print(f"Warning: Model outputted malformed JSON for tool call: {json_str_to_parse}")
                                 pass 
            if not tool_call_detected_and_sent and chunk_text:
                delta = OpenAIChatCompletionStreamResponseDelta(content=chunk_text)
                choice = OpenAIChatCompletionStreamResponseChoice(index=0, delta=delta, finish_reason=None)
                chunk_response = OpenAIChatCompletionStreamResponse(
                    id=request_id, created=created_timestamp, model=current_model_name, choices=[choice]
                )
                yield f"data: {chunk_response.model_dump_json(exclude_none=True)}\n\n"
            if eos:
                if not tool_call_detected_and_sent: finish_reason_val = "stop"
                break
        
        if not tool_call_detected_and_sent: 
            if eos: 
                finish_reason_val = "stop"
            elif generated_tokens_count >= request.max_tokens: 
                finish_reason_val = "length"
        if not tool_call_detected_and_sent:
            final_delta = OpenAIChatCompletionStreamResponseDelta()
            final_choice = OpenAIChatCompletionStreamResponseChoice(index=0, delta=final_delta, finish_reason=finish_reason_val)
            final_chunk = OpenAIChatCompletionStreamResponse(
                 id=request_id, created=created_timestamp, model=current_model_name, choices=[final_choice]
            )
            yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
        
        yield "data: [DONE]\n\n"

    if request.stream:
        return StreamingResponse(stream_openai_response(), media_type="text/event-stream")
    else:
        raise HTTPException(status_code=501, detail="Non-streaming responses are not yet implemented for exllamav2 server. Please use stream=True.")

async def uvicorn_main():
    port = int(os.environ.get("EXLLAMA_PORT", 8081)) 
    host = os.environ.get("EXLLAMA_HOST", "0.0.0.0")
    reload_flag = os.environ.get("EXLLAMA_RELOAD", "false").lower() == "true"

    uv_config = uvicorn.Config(
        "server_exllamav2:app",  
        host=host, 
        port=port,
        reload=reload_flag,
        log_level="info"
    )
    server = uvicorn.Server(uv_config)
    print(f"Starting Exllamav2 server on {host}:{port}, reload={'enabled' if reload_flag else 'disabled'}")
    await server.serve()

if __name__ == "__main__":
    asyncio.run(uvicorn_main())

# --- Experimental: Attempt to load Qwen2.5-VL-GPTQ ---
# This is for testing user curiosity. Multimodal features are NOT expected to work.
# The model might not even load if its architecture is not compatible with exllamav2's GPTQ loader.
#
# 1. Find or create a Qwen2.5-VL-7B-Instruct-GPTQ model. 
#    It must be in a format exllamav2 understands (e.g., safetensors with .json config,
#    potentially with quantize_config.json). This usually means it was converted specifically for exllamav2
#    or a compatible GPTQ library. Standard Hugging Face GPTQ models might not always work directly
#    if their config files (e.g., quantize_config.json or model.json) are not what exllamav2 expects.
#
# 2. Set VL_MODEL_DIRECTORY = "path/to/your/Qwen2.5-VL-7B-Instruct-GPTQ"
#    (e.g., VL_MODEL_DIRECTORY = "../Qwen2.5-VL-7B-Instruct-GPTQ")
#
# if os.environ.get("LOAD_EXPERIMENTAL_VL_GPTQ", "false").lower() == "true":
#     VL_MODEL_DIRECTORY = os.environ.get("VL_MODEL_DIRECTORY")
#     if VL_MODEL_DIRECTORY and os.path.exists(VL_MODEL_DIRECTORY):
#         print(f"\n--- Experimental Qwen2.5-VL-GPTQ Loading ---")
#         try:
#             vl_config = ExLlamaV2Config()
#             vl_config.model_dir = VL_MODEL_DIRECTORY 
#             vl_config.prepare()
#             # For GPTQ, especially if converted from AutoGPTQ, you might need to specify no scratch space for context
#             # vl_config.max_input_len = vl_config.max_seq_len # Or some other appropriate value
#             # vl_config.max_attention_size = vl_config.max_seq_len ** 2 # Or similar
            
#             print(f"Attempting to load experimental VL model from: {VL_MODEL_DIRECTORY}")
#             vl_model = ExLlamaV2(vl_config)
#             vl_model.load() # Or vl_model.load_autosplit(cache_8bit=True) for large models
#             vl_tokenizer = ExLlamaV2Tokenizer(vl_config)
#             print("Experimental Qwen2.5-VL-GPTQ loaded (text parts only expected).")
#             print("Note: This experimental model is NOT wired up to an endpoint in this script.")
#             # To use it, you would need to create a new ExLlamaV2Cache and ExLlamaV2StreamingGenerator
#             # instance for vl_model and vl_tokenizer, and then create a new FastAPI endpoint.
#         except Exception as e:
#             print(f"Failed to load experimental Qwen2.5-VL-GPTQ: {e}")
#         print(f"--- End Experimental Qwen2.5-VL-GPTQ Loading ---\n")
#     else:
#         if VL_MODEL_DIRECTORY:
#             print(f"Experimental VL_MODEL_DIRECTORY '{VL_MODEL_DIRECTORY}' not found. Skipping VL load.")
#         else:
#             print("Experimental VL_MODEL_DIRECTORY not set. Skipping VL load.")
# --- End Experimental ---
