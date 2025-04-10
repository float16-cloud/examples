# xgrammar-transformers-qwen25-0.5b-json

## Getting Started

```
float16 example official/run/transformers-xgrammar-qwen25-0.5b-json
huggingface download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./Qwen2.5-0.5B-Instruct
float16 stoage upload -f ./Qwen2.5-0.5B-Instruct -d ""
float16 project install
float16 run app.py
```

## Description

This is a simple script example the do Guided decoding with the smallest Qwen model like Qwen2.5-0.5b.

## Libraries 

- [xgrammar](https://github.com/mlc-ai/xgrammar)

## GPU Configuration

- H100

## Expected Performance

- Return JSON with the following keys:
    - `province`: string
    - `district`: string
    - `name`: string
    - `description`: string

## Profile

- [X - Matichon](https://x.com/KMatiDev1)
- [Matichon - Personal website](https://matichon.me)
- Email: matichon[dot]man[at]float16[dot]cloud
- Open for Work: No