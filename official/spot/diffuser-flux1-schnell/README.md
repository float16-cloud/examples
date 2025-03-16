# diffuser-flux1-schnell

## Getting Started

```
float16 example official/spot/diffuser-flux1-schnell

huggingface-cli download black-forest-labs/FLUX.1-schnell --local-dir FLUX.1-schnell --exclude "flux1-schnell.safetensors"

float16 storage upload -f FLUX.1-schnell -d model-weight

float16 run inference.py --spot

```

## Description

This example demonstrates how to use text-to-image (FLUX.1-schnell) to generate images from text. 

The model is based on the FLUX.1-schnell model from Black Forest Labs.

## Libraries 

- None

## GPU Configuration

- H100

## Expected Performance

- Return response in less than 20 seconds

## Profile

- [X - Matichon](https://x.com/KMatiDev1)
- [Matichon - Personal website](https://matichon.me)
- Email: matichon[dot]man[at]float16[dot]cloud
- Open for Work: No