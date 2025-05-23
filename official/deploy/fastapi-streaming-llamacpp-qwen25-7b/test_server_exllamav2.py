# test_server_exllamav2.py

import json

# --- Instructions ---
# 1. Start server_exllamav2.py:
#    python official/deploy/fastapi-streaming-llamacpp-qwen25-7b/server_exllamav2.py
#    (Ensure you have a Qwen 7B model in EXL2 format in the directory specified by 
#     EXLLAMAV2_MODEL_DIRECTORY, e.g., ../Qwen-7B-EXL2)
#    (The default port is 8081)

# 2. In a separate terminal, use curl to send requests as shown below.

# --- Test Case 1: Text-Only Chat Completion (Streaming) ---
print("\n--- Test Case 1: Text-Only Chat Completion (Streaming) ---")
payload_text_only = {
    "model": "qwen-7b-exl2", # Or your specific model name
    "messages": [
        {"role": "user", "content": "Write a short poem about coding."}
    ],
    "stream": True,
    "max_tokens": 100,
    "temperature": 0.7
}
print("Payload:")
print(json.dumps(payload_text_only, indent=2))
print("\nCurl command:")
print(f"curl -X POST http://localhost:8081/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{json.dumps(payload_text_only)}' --no-buffer")
print("\nExpected Output:")
print("A stream of JSON objects, each being an OpenAIChatCompletionStreamResponse.")
print("The 'delta.content' should contain parts of the assistant's poem.")
print("The stream should end with a chunk containing 'finish_reason: "stop"' (or "length").")
print("Followed by: data: [DONE]")

# --- Test Case 2: Tool Use (Streaming) ---
# This test assumes the model is prompted to return tool calls in ```json ... ``` format
print("\n\n--- Test Case 2: Tool Use (Streaming) ---")
payload_tool_use = {
    "model": "qwen-7b-exl2",
    "messages": [
        {"role": "user", "content": "What is the weather like in San Francisco today?"}
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
    "tool_choice": "auto",
    "max_tokens": 150
}
print("Payload:")
print(json.dumps(payload_tool_use, indent=2))
print("\nCurl command:")
print(f"curl -X POST http://localhost:8081/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{json.dumps(payload_tool_use)}' --no-buffer")
print("\nExpected Output:")
print("A stream of JSON objects.")
print("One of the chunks should contain 'delta.tool_calls' with a call to 'get_current_weather'.")
print("The 'arguments' field in the tool call should be a JSON string, e.g., '{\"location\": \"San Francisco, CA\"}'.")
print("The choice containing the tool_calls should have 'finish_reason: "tool_calls"'.")
print("The model output for the tool call should be wrapped in ```json ... ``` by the model, which the server then parses.")
print("Followed by: data: [DONE]")

print("\n\nNote: The server's default port is 8081 (EXLLAMA_PORT). If you've set a different port, adjust curl commands accordingly.")
print("Ensure the model used by server_exllamav2.py is capable of following tool use instructions based on the prompting strategy.")
