# fastapi-streaming-llamacpp-qwen25-7b

## Getting Started

```
float16 example official/deploy/fastapi-streaming-llamacpp-qwen25-7b

huggingface-cli download lmstudio-community/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q8_0.gguf --local-dir ./Qwen2.5-7B-Instruct-GGUF

float16 storage upload -f ./Qwen2.5-7B-Instruct-GGUF -d ""

float16 deploy server.py

curl -X POST "{FUNCTION-URL}/chat" -H "Authorization: Bearer {FLOAT16-ENDPOINT-TOKEN}" -H "Content-Type: application/json" -d '{ "messages": "ขอสูตรไก่ย่างหน่อย" }'
```

## Description

This example demonstrates how to deploy a FastAPI server with a Qwen2.5 7b model using Llamacpp.

This server **support** streaming response and JSON response.

## Libraries 

- None

## GPU Configuration

- H100

## Expected Performance

- Return response in less than 5 seconds

## Profile

- [X - Matichon](https://x.com/KMatiDev1)
- [Matichon - Personal website](https://matichon.me)
- Email: matichon[dot]man[at]float16[dot]cloud
- Open for Work: No