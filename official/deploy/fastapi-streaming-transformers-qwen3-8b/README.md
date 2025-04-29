# fastapi-streaming-transformers-qwen3-8b

## Getting Started

```
float16 example official/deploy/fastapi-streaming-transformers-qwen3-8b

huggingface-cli download Qwen/Qwen3-8B --local-dir ./Qwen3-8B/

float16 storage upload -f ./Qwen3-8B -d weight-llm

float16 deploy server.py
```

## Description

This example demonstrates how to deploy a FastAPI server with a Qwen3 8B model with streaming response.

## Libraries 

- Transformers >= 4.51.1

## GPU Configuration

- H100

## Expected Performance

- Return response in less than 10 seconds

## Profile

- [X - Matichon](https://x.com/KMatiDev1)
- [Matichon - Personal website](https://matichon.me)
- Email: matichon[dot]man[at]float16[dot]cloud
- Open for Work: No