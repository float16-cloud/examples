import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import xgrammar as xgr
from pydantic import BaseModel

model_name = "../Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)


_text_formated = [
    {
        "role": "user", "content": "ช่วยแนะนำการท่องเที่ยวในประเทศไทยหน่อย โดยให้ตอบเป็น JSON format"
    }
]
model_inputs = tokenizer.apply_chat_template(
    _text_formated,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

input_len = model_inputs["input_ids"].shape[-1]

tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=151936)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)

class answer_json(BaseModel):
    province : str
    district : str
    name : str
    description : str

compiled_grammar = grammar_compiler.compile_json_schema(answer_json)
xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)


with torch.inference_mode():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128,
        logits_processor=[xgr_logits_processor]
    )
    generated_ids = generated_ids[0][input_len:]

print(tokenizer.decode(generated_ids, skip_special_tokens=True))