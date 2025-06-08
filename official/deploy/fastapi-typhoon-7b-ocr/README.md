# fastapi-typhoon-7b-ocr

## Getting Started

```
float16 example official/deploy/fastapi-typhoon-7b-ocr

huggingface-cli download scb10x/typhoon-ocr-7b --local-dir ./typhoon-ocr-7b

float16 storage upload -f ./typhoon-ocr-7b -d weight-llm

float16 deploy server.py

python3 client.py
```

## Description

This example demonstrates how to deploy a FastAPI server with a Typhoon-7b-OCR model with OCR.

## Libraries 

- pdf2image

## GPU Configuration

- H100

## Expected Performance

- Return response in less than 60 seconds

## Profile

- [X - Matichon](https://x.com/KMatiDev1)
- [Matichon - Personal website](https://matichon.me)
- Email: matichon[dot]man[at]float16[dot]cloud
- Open for Work: No