# huggingface-medgemma-27b

## Getting Started

```
float16 example official/spot/huggingface-medgemma-27b

huggingface-cli download google/medgemma-27b-text-it --local-dir medgemma-27b-text-it

float16 storage upload -f medgemma-27b-text-it -d model-weight

float16 run inference.py --spot
```

## Description

This example demonstrates how to batch-inference LLM model using Huggingface library with medgemma-27b model. 

## Libraries 

- None

## GPU Configuration

- H100

## Expected Performance

- The result will store in /background/{task_id}/output.txt

## Profile

- [X - Matichon](https://x.com/KMatiDev1)
- [Matichon - Personal website](https://matichon.me)
- Email: matichon[dot]man[at]float16[dot]cloud
- Open for Work: No