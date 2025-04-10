# Typhoon2-8b-llamacpp

## Getting Started

```
float16 example official/run/typhoon2-8b-llamacpp

huggingface-cli download Float16-cloud/llama3.1-typhoon2-8b-instruct-gguf llama3.1-typhoon2-8b-instruct-q8_0.gguf --local-dir ./model/

float16 storage upload -f ./model/llama3.1-typhoon2-8b-instruct-q8_0.gguf -d model/typhoon-8b-cpp

float16 run app.py
```

## Description

This script is a simple example of how to use the Typhoon2-8b-llamacpp model with the float16 platform. 
The model is fine-tuned from Llama3.1-8b model with the Typhoon2 dataset.

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