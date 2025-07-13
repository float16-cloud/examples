import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load finetuned checkpoint
finetuned_ckpt_path = "/share_weights/bootcamp/e2e-llm-typhoon-1/llama3.2-typhoon2-3b-instruct"
lora_adapter_path = "../../checkpoints_typhoon2/nemo2_sft/checkpoints/nemo2_sft--reduced_train_loss=0.0801-epoch=3-step=2175-last/hf_adapter" # Should be replaced with the actual path to your LoRA adapter

tokenizer = AutoTokenizer.from_pretrained(finetuned_ckpt_path)
model = AutoModelForCausalLM.from_pretrained(finetuned_ckpt_path)
model = PeftModel.from_pretrained(model, lora_adapter_path)

model.eval()

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Generate text
input_sent = "อากาศไม่ดีเลยวันนี้"
messages = [
            {"role": "user", "content": f"Analyze the sentiment of the following text:\nContext: {input_sent}\nSentiment: "}
        ]

# Apply chat template to get the formatted conversation
formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )


inputs = tokenizer(formatted_text, return_tensors="pt").to(device)
input_length = inputs.input_ids.shape[1]

output = model.generate(**inputs, max_length=100, temperature=0.7, do_sample=True)
new_tokens = output[0][input_length:]
# Decode and print the output
print("\nOutput1: ",tokenizer.decode(new_tokens, skip_special_tokens=False))
print("\nOutput2: ",tokenizer.decode(output[0], skip_special_tokens=False))
