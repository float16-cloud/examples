# llamacpp-guided-decoding-qwen25-0.5b

## Getting Started

```
float16 example official/run/llamacpp-guided-decoding-qwen25-0.5b
huggingface download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_0.gguf --local-dir ./qwen2.5-0.5b-instruct-q4_0-GGUF
float16 stoage upload -f ./qwen2.5-0.5b-instruct-q4_0-GGUF -d ""
float16 run app.py
```

## Description

This is a simple script example the do Guided decoding with the smallest Qwen model like Qwen2.5-0.5b with llamacpp.

## Libraries 

- None

## GPU Configuration

- H100

## Expected Performance
- Return answer with a choice of 3 provinces:
    - `เชียงใหม่`
    - `กรุงเทพ`
    - `ภูเก็ต` 

## Profile

- [X - Matichon](https://x.com/KMatiDev1)
- [Matichon - Personal website](https://matichon.me)
- Email: matichon[dot]man[at]float16[dot]cloud
- Open for Work: No