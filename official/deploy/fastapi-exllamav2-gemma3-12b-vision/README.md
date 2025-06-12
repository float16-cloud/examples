# fastapi-exllamav2-gemma3-12b-vision

## Getting Started

```
float16 example official/deploy/fastapi-exllamav2-gemma3-12b-vision

huggingface-cli download turboderp/gemma-3-12b-it-exl2 --revision 6.0bpw --local-dir ./gemma-3-12b-it-exl2-6bpw

float16 storage upload -f ./gemma-3-12b-it-exl2-6bpw -d model-weight

float16 deploy server.py

python3 client.py
```

## Description

This example demonstrates how to deploy a FastAPI server with a Gemma3 12B model with Vision and Instruct.

## Libraries 

- None

## GPU Configuration

- H100

## Expected Performance

- Return response in less than 60 seconds

## Profile

- [X - Matichon](https://x.com/KMatiDev1)
- [Matichon - Personal website](https://matichon.me)
- Email: matichon[dot]man[at]float16[dot]cloud
- Open for Work: No