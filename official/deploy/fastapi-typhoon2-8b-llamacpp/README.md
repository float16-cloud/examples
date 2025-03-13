# fastapi-typhoon2-8b-llamacpp

## Getting Started

```
float16 example official/deploy/fastapi-typhoon2-8b-llamacpp

huggingface-cli download Float16-cloud/llama3.1-typhoon2-8b-instruct-gguf llama3.1-typhoon2-8b-instruct-q8_0.gguf --local-dir ./model/

float16 storage upload -f ./model/llama3.1-typhoon2-8b-instruct-q8_0.gguf -d model/typhoon-8b-cpp

float16 deploy server.py

curl -X POST "{FUNCTION-URL}/chat" -H "Authorization: Bearer {FLOAT16-ENDPOINT-TOKEN}" -H "Content-Type: application/json" -d '{ "messages": "ขอสูตรไก่ย่างหน่อย" }'
```

## Description

This example demonstrates how to deploy a FastAPI server with a Typhoon2 8B model using LlamaCpp.

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