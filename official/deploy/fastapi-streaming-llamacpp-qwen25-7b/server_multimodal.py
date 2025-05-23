import os
import json
from typing import List, Optional, Dict, Any, Union
import uuid
import time

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.generation.streamers import TextIteratorStreamer
from qwen_vl_utils import process_vision_info # Assuming this is in PYTHONPATH or same directory

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from threading import Thread

# --- Configuration & Model Loading ---
MODEL_PATH = os.environ.get("MODEL_PATH", "../Qwen2.5-VL-7B-Instruct") # Use environment variable or default
QWEN_MODEL_NAME = "qwen2.5-vl-7b-instruct" # Default model name for OpenAI API

try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto", # Uses torch.float16 or torch.bfloat16 if available
        device_map="auto",
        trust_remote_code=True
    )
    # For Flash Attention 2 (optional, requires compatible hardware and installation)
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     MODEL_PATH,
    #     torch_dtype=torch.bfloat16, # or torch.float16
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    #     trust_remote_code=True
    # )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
except Exception as e:
    print(f"Error loading model or processor: {e}")
    # Exit if model loading fails, as the server can't function
    # In a real deployment, you might have a more sophisticated health check
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

app = FastAPI()

# --- OpenAI-compatible Pydantic Models ---

# Request Models
class OpenAIImageURL(BaseModel):
    url: str
    detail: Optional[str] = "auto"

class OpenAIMessageContentItem(BaseModel):
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[OpenAIImageURL] = None

class OpenAIMessage(BaseModel):
    role: str
    content: Union[str, List[OpenAIMessageContentItem]] # OpenAI allows string or list for content
    name: Optional[str] = None # For tool calls later

# Tool-related Pydantic Models
class FunctionParameters(BaseModel):
    type: str = "object"
    properties: dict # JSON Schema properties
    required: Optional[List[str]] = None

class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: FunctionParameters

class Tool(BaseModel):
    type: str = "function"
    function: FunctionDefinition

class OpenAIChatCompletionRequest(BaseModel):
    model: Optional[str] = QWEN_MODEL_NAME # Model name, can be fixed for now
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, dict]] = Field(default="none") # "none", "auto", or {"type": "function", "function": {"name": "my_function"}}
    # Add other common OpenAI params like temperature, top_p later if needed

# Response Models (Streaming & Tool Call related)
class FunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None # JSON string

class ToolCall(BaseModel):
    id: str # Tool call ID, e.g., "call_abc123"
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
    # logprobs: Optional[dict] = None

class OpenAIChatCompletionStreamResponse(BaseModel):
    id: str # Usually a unique ID for the request
    object: str = "chat.completion.chunk"
    created: int # Timestamp
    model: str
    choices: List[OpenAIChatCompletionStreamResponseChoice]
    # system_fingerprint: Optional[str] = None # Later
    # usage: Optional[dict] = None # For non-streaming or final chunk


# --- Old Pydantic Models (to be deprecated or removed) ---
class MultimodalMessageContentItem(BaseModel):
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[str] = None # For image URLs (local paths or http(s) URLs)

class MultimodalMessage(BaseModel):
    role: str
    content: List[MultimodalMessageContentItem]

class MultimodalChatInput(BaseModel):
    messages: List[MultimodalMessage]
    max_new_tokens: Optional[int] = 1024 # Increased default


# --- OpenAI-compatible FastAPI Endpoint ---
@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatCompletionRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # 1. Adapt Input Processing: OpenAIMessage to Qwen format
    
    # System prompt augmentation for tools
    tool_system_prompt_content = ""
    if request.tools and request.tool_choice != "none":
        tools_description_parts = ["You have access to the following tools:"]
        tools_json = [tool.model_dump(exclude_none=True) for tool in request.tools]
        tools_description_parts.append(json.dumps(tools_json, indent=4))
        tools_description_parts.append("To use a tool, respond *only* with a JSON object matching the following schema:")
        tools_description_parts.append('''{
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
}''')
        tools_description_parts.append("If you are not using a tool, respond with text as usual.")
        tool_system_prompt_content = "\n\n".join(tools_description_parts)

    formatted_messages_for_template = []
    messages_for_vision_processing = [] # For process_vision_info

    # Handle prepending or augmenting system message
    # We need to iterate through request.messages once to build both formatted_messages_for_template
    # and messages_for_vision_processing correctly, while also injecting the tool system prompt.

    has_existing_system_message = False
    if request.messages and request.messages[0].role == "system":
        has_existing_system_message = True

    if tool_system_prompt_content:
        if has_existing_system_message:
            # Augment existing system message
            original_system_content = request.messages[0].content
            if isinstance(original_system_content, str):
                augmented_content = original_system_content + "\n\n" + tool_system_prompt_content
                formatted_messages_for_template.append({"role": "system", "content": augmented_content})
            else: # content is list of items, should not happen for system message based on OpenAI spec for system role
                print("Warning: System message content is a list, cannot directly augment with tool prompt. Adding tool prompt as new system message.")
                # Fallback: add tool prompt as a new system message later if complex system content
                # For now, let's assume system content is string if role is system.
                # Or, create a new system message with the tool prompt.
                formatted_messages_for_template.append({"role": "system", "content": tool_system_prompt_content})
                # And then process the original system message as a normal message (which is not ideal)
                # A better approach if system message content is complex:
                # formatted_messages_for_template.append({"role": "system", "content": tool_system_prompt_content})
                # Then process original messages starting from index 0. If messages[0] was system, it will be re-added.
                # This might lead to two system messages.
                # Safest: If existing system message is not simple string, add tool prompt as first system message.
                # Let's refine this:
                # If has_existing_system_message and isinstance(request.messages[0].content, str):
                #    ... augment ...
                # else:
                #    formatted_messages_for_template.append({"role": "system", "content": tool_system_prompt_content})
                # This logic will be part of the loop now.

        else: # No existing system message, prepend tool prompt as a new system message
            formatted_messages_for_template.append({"role": "system", "content": tool_system_prompt_content})

    message_start_index = 0
    if tool_system_prompt_content and has_existing_system_message and isinstance(request.messages[0].content, str):
        # Already handled the augmented system message above
        message_start_index = 1 
        # The first message (original system) was consumed and augmented.

    for i in range(message_start_index, len(request.messages)):
        oai_msg = request.messages[i]
        current_template_content_parts = [] # For apply_chat_template content (list of dicts or string)
        current_vision_content_items = [] # For process_vision_info (list of dicts)
        
        role = oai_msg.role
        current_vision_content_items = [] # For process_vision_info
        
        role = oai_msg.role
        
        # Special handling for the first message if it was an existing system message
        # and we augmented it with tool_system_prompt_content.
        # This block is entered if tool_system_prompt_content was non-empty,
        # there was an existing system message, and its content was a simple string.
        # In this case, formatted_messages_for_template already contains the augmented system message.
        if i == 0 and tool_system_prompt_content and has_existing_system_message and isinstance(request.messages[0].content, str):
            # The original system message content was already incorporated.
            # Now, if this system message also had image content (unlikely for system role but for completeness)
            # we'd process it here for vision. Typically, system messages are text.
            if role == "user": # This condition will likely not be met for a system message.
                # This part of logic might be redundant if system messages are strictly text.
                if isinstance(oai_msg.content, list):
                    for item in oai_msg.content:
                        if item.type == "image_url" and item.image_url and item.image_url.url:
                             current_vision_content_items.append({"type": "image", "image": item.image_url.url})
                if any(item.get("type") == "image" for item in current_vision_content_items):
                    messages_for_vision_processing.append({"role": "user", "content": current_vision_content_items})
            continue # Already processed this message by augmentation.

        # General message processing for messages not handled by augmentation above
        if role not in ["user", "system", "assistant"]:
            print(f"Warning: Unsupported role '{role}' encountered. Skipping message.")
            continue

        if isinstance(oai_msg.content, str):
            current_template_content_parts = [{"type": "text", "text": oai_msg.content}]
            if role == "user":
                 current_vision_content_items.append({"type": "text", "text": oai_msg.content})
        else: # List of content items
            for item in oai_msg.content:
                if item.type == "text" and item.text:
                    current_template_content_parts.append({"type": "text", "text": item.text})
                    if role == "user":
                        current_vision_content_items.append({"type": "text", "text": item.text})
                elif item.type == "image_url" and item.image_url and item.image_url.url:
                    if role == "user":
                        current_template_content_parts.append({"type": "image", "image_url": item.image_url.url})
                        current_vision_content_items.append({"type": "image", "image": item.image_url.url})
                    else:
                        print(f"Warning: Image content found in role '{role}'. Images are typically processed for 'user' role only.")
        
        if not current_template_content_parts:
            # If this was the original system message and it was empty, but we added tool prompt, it's fine.
            # Otherwise, if any other message is empty, skip.
            if not (role == "system" and tool_system_prompt_content and i==0 and has_existing_system_message):
                 print(f"Warning: Message from role '{role}' resulted in no content for template. Skipping.")
                 continue

        if role == "user":
            template_content_to_add = current_template_content_parts
        else: # system or assistant
            # If this is the original system message that we chose not to augment directly (e.g. content was a list)
            # then we need to add its text content here.
            # The tool_system_prompt_content would have been added as a separate system message.
            consolidated_text = " ".join([part["text"] for part in current_template_content_parts if part["type"] == "text"])
            if not consolidated_text.strip():
                if not (role == "system" and tool_system_prompt_content and i==0 and has_existing_system_message):
                    print(f"Warning: Non-user role '{role}' message has no text content after processing. Skipping.")
                    continue
            template_content_to_add = consolidated_text
        
        # Add the processed message to formatted_messages_for_template
        # unless it's an empty system message that was supposed to be augmented
        if template_content_to_add or (role == "system" and i==0 and tool_system_prompt_content and not has_existing_system_message):
            # if it's the system message slot, and we added tool_system_prompt_content, but original system message was empty
            # we still need to add the formatted_message entry for the system role if it wasn't added before.
            # This is covered by the initial addition of tool_system_prompt_content if no existing system message.
            if not (role == "system" and i==0 and tool_system_prompt_content and has_existing_system_message and isinstance(request.messages[0].content, str)):
                 # This check prevents re-adding an augmented system message.
                 # If template_content_to_add is empty (e.g. an empty system message that wasn't augmented), don't add.
                 if template_content_to_add:
                    formatted_messages_for_template.append({"role": role, "content": template_content_to_add})

        if role == "user" and any(item.get("type") == "image" for item in current_vision_content_items):
            messages_for_vision_processing.append({"role": "user", "content": current_vision_content_items})

    if not formatted_messages_for_template:
        # This can happen if all messages were skipped or input was empty.
        # Or if tool_system_prompt_content was the only thing and it got filtered.
        # Ensure there's at least the tool system prompt if that was generated.
        if tool_system_prompt_content and not any(m['role'] == 'system' for m in formatted_messages_for_template):
             formatted_messages_for_template.append({"role": "system", "content": tool_system_prompt_content})
        
        if not formatted_messages_for_template:
             raise HTTPException(status_code=400, detail="No valid message content to process after OpenAIMessage conversion and tool prompt handling.")
        
    try:
        # Make sure tokenizer is used for Qwen-VL template application
        text_prompt_for_model = processor.apply_chat_template(
            formatted_messages_for_template,
            tokenize=True, # Tokenize for Qwen-VL
            add_generation_prompt=True,
            return_tensors="pt" 
        )
        # apply_chat_template with tokenize=True, return_tensors="pt" returns a dict of tensors
        # We need input_ids and attention_mask from this.
        # If images are involved, the processor call later will combine text and image features.
        # For now, let's assume text_prompt_for_model contains input_ids and attention_mask if no images.
        # If images are present, this text_prompt_for_model might just be the tokenized text part.
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying chat template: {str(e)}")

    # 2. Process vision information (if any user messages had images)
    image_pixel_values = None
    image_sizes_for_model = None

    if messages_for_vision_processing:
        try:
            # process_vision_info expects a list of messages.
            # It internally handles downloading URLs if they are http/https.
            # It needs the image_processor to correctly preprocess images.
            # It returns preprocessed image tensors and their original sizes.
            vision_inputs = process_vision_info(messages_for_vision_processing, processor.image_processor, query_type='history')
            
            # vision_inputs is a dict containing 'pixel_values' (list of tensors) and 'image_sizes' (list of lists)
            # We need to stack/prepare these for the model batch if multiple images.
            # For now, assuming process_vision_info prepares them adequately or we handle single image context primarily.
            if vision_inputs and vision_inputs.get('pixel_values'):
                 image_pixel_values = vision_inputs['pixel_values'] # Should be a tensor
                 image_sizes_for_model = vision_inputs.get('image_sizes') # Should be a tensor
                 if isinstance(image_pixel_values, list): # if it returns a list of tensors
                     image_pixel_values = torch.cat(image_pixel_values, dim=0) if image_pixel_values else None
                 if isinstance(image_sizes_for_model, list):
                     image_sizes_for_model = torch.tensor(image_sizes_for_model) if image_sizes_for_model else None


        except Exception as e:
            print(f"Warning: Could not process vision info: {e}. Proceeding with text only if possible.")
            # Not raising HTTPException here, allow text-only fallback if image processing fails
            image_pixel_values = None
            image_sizes_for_model = None
            # If vision processing fails, we should ensure the template didn't include image placeholders
            # that the model would then expect pixel_values for.
            # The current template logic for user messages adds {"type": "image", "image_url": ...}
            # The AutoProcessor might handle this gracefully if pixel_values are missing, or it might error.
            # For robustness, if image_pixel_values is None, we might need to re-run apply_chat_template
            # with only text parts of the messages.
            # However, Qwen's processor when called with text + images=None should work.

    # 3. Prepare final inputs for the model
    model_inputs = {}
    try:
        # The `text_prompt_for_model` from `apply_chat_template` (tokenized)
        # already contains input_ids and attention_mask for the text part.
        if isinstance(text_prompt_for_model, dict): # If apply_chat_template returned dict of tensors
            model_inputs.update(text_prompt_for_model)
        else: # Should not happen if tokenize=True, return_tensors="pt"
            raise ValueError("apply_chat_template did not return tokenized tensors as expected.")

        if image_pixel_values is not None:
            model_inputs['pixel_values'] = image_pixel_values
        if image_sizes_for_model is not None:
            model_inputs['image_sizes'] = image_sizes_for_model
        
        # Move all tensor inputs to model device
        for key, value in model_inputs.items():
            if isinstance(value, torch.Tensor):
                model_inputs[key] = value.to(model.device)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing model inputs: {str(e)}")

    if 'input_ids' not in model_inputs or model_inputs['input_ids'] is None:
         raise HTTPException(status_code=500, detail="Failed to obtain input_ids for the model.")


    # 4. Streaming Logic
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = {
        # "input_ids": model_inputs['input_ids'], # Already in model_inputs
        # "attention_mask": model_inputs.get('attention_mask'), # Already in model_inputs
        **model_inputs, # Spread all prepared inputs
        "streamer": streamer,
        "max_new_tokens": request.max_tokens,
        # Add other common generation parameters if desired from request
        # "temperature": request.temperature if hasattr(request, 'temperature') else 0.7,
        # "top_p": request.top_p if hasattr(request, 'top_p') else 0.9,
    }
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_timestamp = int(time.time())
    current_model_name = request.model if request.model else QWEN_MODEL_NAME

    def generate_openai_stream():
        # Optional: Send an initial chunk with role if your model/setup benefits from it.
        # Some clients expect a role even if content is null for the first chunk.
        # yield f"data: {OpenAIChatCompletionStreamResponse(id=request_id, object='chat.completion.chunk', created=created_timestamp, model=current_model_name, choices=[OpenAIChatCompletionStreamResponseChoice(delta=OpenAIChatCompletionStreamResponseDelta(role='assistant'))]).model_dump_json()}\n\n"
        
        finish_reason_val = "stop" # Default to stop
        try:
            for new_text in streamer:
                if not new_text: continue 
                
                # Check for stop conditions potentially embedded in new_text by model or streamer
                # (e.g. if streamer itself yields a token indicating max_length, though less common here)
                # if new_text == processor.tokenizer.eos_token: # Example, may not be exact
                #     finish_reason_val = "stop"
                #     break # Stop sending more content

                delta = OpenAIChatCompletionStreamResponseDelta(content=new_text)
                choice = OpenAIChatCompletionStreamResponseChoice(delta=delta, finish_reason=None) # finish_reason is null for content chunks
                chunk = OpenAIChatCompletionStreamResponse(
                    id=request_id,
                    created=created_timestamp,
                    model=current_model_name,
                    choices=[choice]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            
            # After loop, determine final finish_reason.
            # If streamer stops due to max_tokens, it should be 'length'.
            # For now, we assume 'stop' unless an error or other condition is detected.
            # This part might need more sophisticated logic to detect 'length' if not directly signaled by generate().

        except Exception as e:
            print(f"Error during response streaming: {e}")
            # Send an error chunk if possible, though spec for SSE errors is less defined for OpenAI format
            # For now, just print and ensure final chunk is sent.
            # finish_reason_val = "error" # Custom, not standard OpenAI
        finally:
            # Final chunk with finish_reason
            final_delta = OpenAIChatCompletionStreamResponseDelta() # Empty delta
            final_choice = OpenAIChatCompletionStreamResponseChoice(delta=final_delta, finish_reason=finish_reason_val)
            final_chunk = OpenAIChatCompletionStreamResponse(
                id=request_id,
                created=created_timestamp,
                model=current_model_name,
                choices=[final_choice]
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

    if request.stream:
        return StreamingResponse(generate_openai_stream(), media_type="text/event-stream")
    else:
        # Non-streaming: collect all tokens and return a single response
        # This requires running the generation without the TextIteratorStreamer, or collecting from it.
        try:
            # Simplified non-streaming: Collect from streamer (not ideal for performance)
            # For true non-streaming, model.generate should be called without streamer.
            full_response_content = []
            for new_text in streamer: # This re-uses the threaded generation
                 full_response_content.append(new_text)
            
            # Wait for thread to finish if it hasn't
            thread.join(timeout=30) # Add a timeout

            # Construct non-streaming response (needs full Pydantic models for non-streaming)
            # For now, returning a simple JSON with collected content
            # This part needs the full OpenAIChatCompletionResponse model definitions
            # and proper assembly of the 'choices', 'usage', etc.
            
            # Placeholder for actual non-streaming response structure
            # Define OpenAIChatCompletionResponse, OpenAIChatCompletionResponseChoice, OpenAIMessage (for assistant response) etc.
            # For now, let's use a simplified dict or raise error.
            
            # Simulating what a non-streaming response would look like with current pieces:
            # collected_content = "".join(full_response_content)
            # response_message = OpenAIMessage(role="assistant", content=collected_content)
            # choice = OpenAIChatCompletionResponseChoice(index=0, message=response_message, finish_reason="stop")
            # usage_stats = OpenAIChatCompletionResponseUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0) # Dummy
            # non_stream_response = OpenAIChatCompletionResponse(
            #     id=request_id,
            #     object="chat.completion",
            #     created=created_timestamp,
            #     model=current_model_name,
            #     choices=[choice],
            #     usage=usage_stats
            # )
            # return non_stream_response

            # For this subtask, raising NotImplementedError for non-streaming is acceptable.
             raise HTTPException(status_code=501, detail="Non-streaming responses are not yet implemented. Please use stream=True.")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in non-streaming generation: {str(e)}")


# --- Old FastAPI Endpoint (kept for reference or internal testing) ---
@app.post("/chat_stream_multimodal")
async def chat_stream_multimodal(chat_input: MultimodalChatInput):
    if not chat_input.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # 1. Reformat messages for process_vision_info and apply_chat_template
    formatted_messages_for_template = []
    messages_for_vision_processing = [] 

    for msg in chat_input.messages:
        if msg.role == "user":
            processed_content_for_template = []
            content_for_vision = []
            has_content = False
            for item in msg.content:
                if item.type == "text" and item.text:
                    processed_content_for_template.append({"type": "text", "text": item.text})
                    content_for_vision.append({"type": "text", "text": item.text})
                    has_content = True
                elif item.type == "image_url" and item.image_url:
                    processed_content_for_template.append({"type": "image", "image": item.image_url}) # Corrected key for Qwen template
                    content_for_vision.append({"type": "image", "image": item.image_url}) 
                    has_content = True
            
            if has_content:
                formatted_messages_for_template.append({"role": "user", "content": processed_content_for_template})
                if any(item.type == "image_url" for item in msg.content): 
                     messages_for_vision_processing.append({"role": "user", "content": content_for_vision})

        elif msg.role in ["system", "assistant"]:
            text_content = ""
            for item in msg.content: # Assuming system/assistant content is a list of text items
                if item.type == "text" and item.text:
                    text_content += item.text + " "
            if text_content.strip():
                 formatted_messages_for_template.append({"role": msg.role, "content": text_content.strip()})
        
    if not formatted_messages_for_template:
        raise HTTPException(status_code=400, detail="No valid message content to process after formatting.")
        
    try:
        # For the old endpoint, apply_chat_template might expect text prompt directly
        # and images are handled separately by the model.generate call.
        # Let's assume the old way was to pass text (potentially with placeholders) 
        # and then image tensors separately.
        
        # Simplified text extraction for old endpoint's template
        text_parts_for_old_template = []
        for m in formatted_messages_for_template:
            if isinstance(m['content'], str):
                text_parts_for_old_template.append(m['content'])
            elif isinstance(m['content'], list): # User message with multimodal content
                for item_content in m['content']:
                    if item_content['type'] == 'text':
                        text_parts_for_old_template.append(item_content['text'])
                    # Image placeholders might have been implicitly handled or not used in text prompt
        
        text_prompt_for_template = processor.apply_chat_template(
            [{"role": "user", "content": " ".join(text_parts_for_old_template)}] if text_parts_for_old_template else [], # Example simplification
            tokenize=False, # Old endpoint might have used tokenize=False
            add_generation_prompt=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying chat template (old endpoint): {str(e)}")

    image_inputs_old = None
    if messages_for_vision_processing: 
        try:
            # process_vision_info returns a dict with 'pixel_values' and 'image_sizes'
            vision_processed_data = process_vision_info(messages_for_vision_processing, processor.image_processor, query_type='history')
            if vision_processed_data and vision_processed_data.get('pixel_values'):
                image_inputs_old = {
                    'pixel_values': vision_processed_data['pixel_values'],
                    'image_sizes': vision_processed_data.get('image_sizes')
                }
        except Exception as e:
            print(f"Warning (old endpoint): Could not process vision info: {e}.")
            image_inputs_old = None

    try:
        # Old endpoint's input preparation
        inputs_dict_old = {}
        if image_inputs_old and image_inputs_old.get('pixel_values') is not None:
             # If images are present, processor is called with text and images
            processed_inputs_old = processor(
                text=[text_prompt_for_template],
                images=image_inputs_old['pixel_values'], # Pass pixel_values directly
                padding=True,
                return_tensors="pt",
            )
            inputs_dict_old.update(processed_inputs_old)
            if image_inputs_old.get('image_sizes') is not None:
                 inputs_dict_old['image_sizes'] = image_inputs_old['image_sizes']
        else: # Text-only
            processed_inputs_old = processor(
                text=[text_prompt_for_template],
                padding=True,
                return_tensors="pt",
            )
            inputs_dict_old.update(processed_inputs_old)

        for key, value in inputs_dict_old.items():
            if isinstance(value, torch.Tensor):
                inputs_dict_old[key] = value.to(model.device)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing model inputs (old endpoint): {str(e)}")

    streamer_old = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs_old = {
        **inputs_dict_old,
        "streamer": streamer_old,
        "max_new_tokens": chat_input.max_new_tokens,
    }
    
    thread_old = Thread(target=model.generate, kwargs=generation_kwargs_old)
    thread_old.start()

    def generate_stream_old():
        try:
            for new_text in streamer_old:
                yield json.dumps({"text": new_text}) + "\n"
        except Exception as e:
            print(f"Error during response streaming (old endpoint): {e}")
            yield json.dumps({"error": "Error during streaming", "detail": str(e)}) + "\n"

    return StreamingResponse(generate_stream_old(), media_type="application/x-ndjson")


# --- Basic Main for Uvicorn ---
async def main():
    port = int(os.environ.get("PORT", 8080)) # Default to 8080 if not set
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Check if running with reload for development
    reload_flag = os.environ.get("UVICORN_RELOAD", "false").lower() == "true"

    config = uvicorn.Config(
        "server_multimodal:app",  # Points to the FastAPI app instance
        host=host, 
        port=port,
        reload=reload_flag, # Enable reload if UVICORN_RELOAD is true
        log_level="info"
    )
    server = uvicorn.Server(config)
    print(f"Starting server on {host}:{port}, reload={'enabled' if reload_flag else 'disabled'}")
    await server.serve()

if __name__ == "__main__":
    import asyncio
    # Example: Set environment variable for testing reload
    # os.environ["UVICORN_RELOAD"] = "true" 
    asyncio.run(main())
