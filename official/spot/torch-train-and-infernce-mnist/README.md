# torch-train-and-infernce-mnist

## Getting Started

```
float16 example official/spot/torch-train-and-infernce-mnist

python3 download-mnist-datasets.py

float16 storage upload -f ./mnist-datasets -d datasets

float16 run train.py --spot

## After training, you can run the inference script

float16 run inference.py

```

## Description

This example demonstrates how to train a PyTorch model with MNIST datasets and run inference.

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