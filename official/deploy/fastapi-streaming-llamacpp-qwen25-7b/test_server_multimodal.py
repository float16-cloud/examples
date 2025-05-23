# test_server_multimodal.py

import json
import base64 # For potential base64 image testing later

# --- Instructions ---
# 1. Start server_multimodal.py:
#    python official/deploy/fastapi-streaming-llamacpp-qwen25-7b/server_multimodal.py
#    (Ensure the Qwen/Qwen2.5-VL-7B-Instruct model is downloaded to ../Qwen2.5-VL-7B-Instruct)

# 2. In a separate terminal, use curl to send requests as shown below.

# --- Test Case 1: Text-Only Chat Completion (Streaming) ---
print("\n--- Test Case 1: Text-Only Chat Completion (Streaming) ---")
payload_text_only = {
    "model": "qwen2.5-vl-7b-instruct",
    "messages": [
        {"role": "user", "content": "Hello, what is your name?"}
    ],
    "stream": True,
    "max_tokens": 50
}
print("Payload:")
print(json.dumps(payload_text_only, indent=2))
print("\nCurl command:")
print(f"curl -X POST http://localhost:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{json.dumps(payload_text_only)}' --no-buffer")
print("\nExpected Output:")
print("A stream of JSON objects, each being an OpenAIChatCompletionStreamResponse.")
print("The 'delta.content' should contain parts of the assistant's response.")
print("The stream should end with a chunk containing 'finish_reason: "stop"' (or "length").")
print("Followed by: data: [DONE]")

# --- Test Case 2: Multimodal Chat Completion (Image URL, Streaming) ---
print("\n\n--- Test Case 2: Multimodal Chat Completion (Image URL, Streaming) ---")
# You might need to find a stable image URL for testing, or host one locally.
# Using a known Qwen demo image URL.
image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
payload_multimodal = {
    "model": "qwen2.5-vl-7b-instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one short sentence."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ],
    "stream": True,
    "max_tokens": 100
}
print("Payload:")
print(json.dumps(payload_multimodal, indent=2))
print("\nCurl command:")
print(f"curl -X POST http://localhost:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{json.dumps(payload_multimodal)}' --no-buffer")
print("\nExpected Output:")
print("A stream of JSON objects, similar to Test Case 1.")
print("The 'delta.content' should contain a description of the image.")

# --- Test Case 3: Tool Use (Streaming) ---
print("\n\n--- Test Case 3: Tool Use (Streaming) ---")
payload_tool_use = {
    "model": "qwen2.5-vl-7b-instruct",
    "messages": [
        {"role": "user", "content": "What is the weather like in Boston?"}
    ],
    "stream": True,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ],
    "tool_choice": "auto", # or "none" or specific tool
    "max_tokens": 150
}
print("Payload:")
print(json.dumps(payload_tool_use, indent=2))
print("\nCurl command:")
print(f"curl -X POST http://localhost:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{json.dumps(payload_tool_use)}' --no-buffer")
print("\nExpected Output:")
print("A stream of JSON objects.")
print("One of the chunks should contain 'delta.tool_calls' with a call to 'get_current_weather'.")
print("The arguments should be a JSON string, e.g., '{\"location\": \"Boston, MA\"}'.")
print("The choice containing the tool_calls should have 'finish_reason: "tool_calls"'.")
print("Followed by: data: [DONE]")

# --- Test Case 4: Tool Use - Forced (Example, if model doesn't pick it up automatically) ---
# This is more about showing how to force a tool if 'auto' doesn't work as expected or for specific testing.
# The model might not always pick the tool with "auto", this depends on its training.
print("\n\n--- Test Case 4: Tool Use - Forced (Illustrative) ---")
payload_tool_use_forced = {
    "model": "qwen2.5-vl-7b-instruct",
    "messages": [
        {"role": "user", "content": "I need weather info for London."},
        # For some models, you might need an assistant message acknowledging the tool use if it was a multi-turn setup.
        # {"role": "assistant", "content": null, "tool_calls": [...] } # This is more for sending tool results back.
    ],
    "stream": True,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ],
    "tool_choice": {"type": "function", "function": {"name": "get_current_weather"}},
    "max_tokens": 150
}
print("Payload (Forced Tool):")
print(json.dumps(payload_tool_use_forced, indent=2))
print("\nCurl command (Forced Tool):")
print(f"curl -X POST http://localhost:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{json.dumps(payload_tool_use_forced)}' --no-buffer")
print("\nExpected Output (Forced Tool):")
print("Similar to Test Case 3, expecting a tool_call to 'get_current_weather'.")
print("Forcing a tool might make the model more reliably call it, but the arguments still depend on the prompt and model understanding.")

print("\n\nNote: The server's default port is 8080. If you've set the PORT environment variable, adjust curl commands accordingly.")
