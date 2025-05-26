# vllm-offline-inference-typhoon-ocr-7b

## Getting Started

```
float16 example official/spot/vllm-offline-inference-typhoon-ocr

huggingface-cli download scb10x/typhoon-ocr-7b --local-dir typhoon-ocr-7b

float16 storage upload -f typhoon-ocr-7b -d model-weight

float16 storage upload -f law.pdf -d tmp_data

float16 run inference.py --spot

```

## Description

This example demonstrates how to do batch inference a typhoon-ocr-7b model using vLLM in an offline environment.

## Libraries 

- typhoon-ocr

## GPU Configuration

- H100

## Expected Performance

- BATCH_SIZE=64, Should process around 1 image per second.

## Profile

- [X - Matichon](https://x.com/KMatiDev1)
- [Matichon - Personal website](https://matichon.me)
- Email: matichon[dot]man[at]float16[dot]cloud
- Open for Work: No