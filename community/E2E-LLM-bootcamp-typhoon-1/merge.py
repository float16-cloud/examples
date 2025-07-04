import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load finetuned checkpoint
finetuned_ckpt_path = "/share_weights/bootcamp/e2e-llm-typhoon-1/llama3.2-typhoon2-3b-instruct"
lora_adapter_path = "../../checkpoints_typhoon2/nemo2_sft/checkpoints/nemo2_sft--reduced_train_loss=0.0557-epoch=3-step=2175-last/hf_adapter" # Should be replaced with the actual path to your LoRA adapter

tokenizer = AutoTokenizer.from_pretrained(finetuned_ckpt_path)
model = AutoModelForCausalLM.from_pretrained(finetuned_ckpt_path)
model = PeftModel.from_pretrained(model, lora_adapter_path)

model = model.merge_and_unload()

# Define a directory to save the merged model and tokenizer
save_directory = "../../merged_model_hf/"

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Save the merged model
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
print("Complete")
