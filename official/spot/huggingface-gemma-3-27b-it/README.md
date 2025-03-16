# huggingface-gemma-3-27b-it

## Getting Started

```
float16 example official/spot/huggingface-gemma-3-27b-it

huggingface-cli download google/gemma-3-27b-it --local-dir gemma-3-27b-it

float16 storage upload -f gemma-3-27b-it -d model-weight

float16 storage upload -f test_image.jpg -d tmp_data

float16 run inference.py --spot

```

## Description

This example demonstrates how to inference multi-modal model using Huggingface library with Gemma3 27B model. 

## Libraries 

- git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

## GPU Configuration

- H100

## Expected Performance

- Return response in less than 60 seconds

## Profile

- [X - Matichon](https://x.com/KMatiDev1)
- [Matichon - Personal website](https://matichon.me)
- Email: matichon[dot]man[at]float16[dot]cloud
- Open for Work: No