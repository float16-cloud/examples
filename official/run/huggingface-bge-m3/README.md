# huggingface-bge-m3

## Getting Started

```
float16 example official/run/huggingface-bge-m3

huggingface-cli download BAAI/bge-m3 --local-dir bge-m3 --exclude "onnx/*"

float16 storage upload -f bge-m3 -d model-weight

float16 run app.py
```

## Description

This script is a simple example of how to use bge-m3 model from Huggingface.

bge-m3 is a multilingual-model for embedding text into vector space.

## Libraries 

- None

## GPU Configuration

- H100

## Expected Performance

- Return response in less than 1 seconds

## Profile

- [X - Matichon](https://x.com/KMatiDev1)
- [Matichon - Personal website](https://matichon.me)
- Email: matichon[dot]man[at]float16[dot]cloud
- Open for Work: No