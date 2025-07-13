# E2E-LLM-Bootcamp-Typhoon-1

## Event details

- **Event Name**: E2E-LLM-Bootcamp-Typhoon-1
- **Event Date**: 04-07-2025

Link: [Event Link](https://lu.ma/i2nst25l)

## Event Description

This event is designed to provide a comprehensive introduction to building and deploying end-to-end large language models (LLMs) using Typhoon. Participants will learn how to set up their environment, train models, and deploy them for inference.

---

## Setup

To install and configure the Float16 CLI:

```bash
brew install float16-cloud/float16/cli # MacOS

# Windows (NPM is needed)
# npm install -g @float16/cli

float16 config set --base-url https://api.float16.cloud/
float16 login --token <float16-xxxx> # Replace this with your Float16 token
float16 example community/E2E-LLM-bootcamp-typhoon-1
```

## Training

1. Edit `train.py` to include your WandB API key:

   ```python
   os.environ['WANDB_API_KEY'] = ""  # Replace with your actual WandB API key
   ```

2. Start a new Float16 project and run training:

   ```bash
   float16 project start
   float16 run train.py --spot
   ```

## Inference

1. Edit `infer.py` to point to your LoRA adapter and set the input prompt:

   ```python
   lora_adapter_path = "../checkpoints_typhoon2/nemo2_sft/checkpoints/nemo2_sft--reduced_train_loss=0.0801-epoch=3-step=2175-last/hf_adapter"  # Replace with the actual path to your LoRA adapter path

   input_sent = "อากาศไม่ดีเลยวันนี้"  # Example prompt in Thai
   ```

2. Run the inference script:

   ```bash
   float16 run infer.py
   ```

Alternatively, you can go to Playground and copy code from `infer.py` and run it there.

---

### Morning Session

- [infer.py](./infer.py)

### Afternoon Session

- [merge.py](./merge.py)
- [to-gguf.py](./to-gguf.py)
- [infer-grammar.py](./infer-grammar.py)
- [server.py](./server.py)
- [client.py](./client.py)
