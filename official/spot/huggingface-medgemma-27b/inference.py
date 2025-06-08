from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "../../model-weight/medgemma-27b-text-it"
MAX_TOKENS = 4096 ## Maximum number of tokens to generate in the response.
BATCH_SIZE = 4 ## Number of Parallel Compute but It requires more GPU memory. 4 is recommended for H100 GPU.

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

batched_message = [
    [{"role" : "system", "content" : "You are a helpful assistant."},{"role" : "user", "content" : "แนะนำ Guideline สำหรับการวัดความดันหน่อย?"}],
    [{"role" : "system", "content" : "You are a helpful medical assistant."},{"role" : "user", "content" : "แนะนำ Guideline สำหรับการวัดความดันหน่อย?"}],
    [{"role" : "system", "content" : "คุณคือผู้ช่วยด้านการให้ความรู้ด้านสุขภาพ."},{"role" : "user", "content" : "แนะนำ Guideline สำหรับการวัดความดันหน่อย?"}],
]

batch_tokenized = []
for data in batched_message : 
    _text_tokenized = tokenizer.apply_chat_template(
        data,
        tokenize=False,
        add_generation_prompt=True
    )
    batch_tokenized.append(_text_tokenized)


for i in range(0,len(batch_tokenized),BATCH_SIZE):
    with torch.inference_mode():
        model_inputs = tokenizer(
            batch_tokenized[i:i+BATCH_SIZE], 
            return_tensors="pt", 
            padding="longest", 
            truncation=True,
            pad_to_multiple_of=8,
            max_length=4096
        ).to(model.device)

        generation = model.generate(
            **model_inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )


    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generation)
    ]

    result_list = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    for result in result_list: ## Write the result to a file
        with open("output.txt", "a") as f: 
            f.write(result + "\n" + "-" * 50 + "\n")