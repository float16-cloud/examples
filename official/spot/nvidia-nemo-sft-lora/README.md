# nvidia-nemo-sft-lora

## Getting Started

```
float16 example official/spot/nvidia-nemo-sft-lora

huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir Qwen2.5-0.5B-Instruct
huggingface-cli download pythainlp/wisesight_sentiment wisesight_sentiment/train-00000-of-00001.parquet --local-dir ./ --repo-type dataset
mv ./wisesight_sentiment/train-00000-of-00001.parquet ./wisesight_sentiment/wisesight_sentiment_train

float16 storage upload -f Qwen2.5-0.5B-Instruct -d models
float16 storage upload -f wisesight_sentiment/wisesight_sentiment_train -d datasets

float16 run sft-lora.py --spot
```

## Description

This example demonstrates how to fine-tune a Qwen/Qwen2.5-0.5B model SFT with lora using NVIDIA-NeMo.

Features:
- Resilient to spot interruptions
- Monitoring via WandB

## Libraries 

- None

## GPU Configuration

- H100

## Expected Performance

- Jobs will finish in approximately 5 minutes.

## Profile

- [X - Matichon](https://x.com/KMatiDev1)
- [Matichon - Personal website](https://matichon.me)
- Email: matichon[dot]man[at]float16[dot]cloud
- Open for Work: No