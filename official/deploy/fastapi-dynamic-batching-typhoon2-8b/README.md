# fastpi-dynamic-batching-typhoon2-8b

## Getting Started

```
float16 example official/deploy/fastapi-dynamic-batching-typhoon2-8b

huggingface-cli download scb10x/llama3.1-typhoon2-8b-instruct --local-dir ./typhoon2-8b/

float16 storage upload -f ./typhoon2-8b -d weight-llm

float16 deploy server.py
```

## Description

This example demonstrates how to deploy a FastAPI server with a Typhoon2 8B model using dynamic batching.

Dynamic batching is allows you to send multiple requests in a single process. 

This can reduce the latency of the model.

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