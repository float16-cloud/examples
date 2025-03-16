import transformers
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
print(transformers.__version__)

try : 
    model_name = "../../model-weight/gemma-3-27b-it"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    data = "OCR รูปภาพต่อไปนี้"
    _text_formated = [
        {
            "role": "user", "content": [
                {"type": "image", "image": "../../tmp_data/test_image.jpg"},
                {"type": "text", "text": data}
            ]
        }
    ]
    model_inputs = processor.apply_chat_template(
        _text_formated,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    input_len = model_inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024
        )
        generated_ids = generated_ids[0][input_len:]

    with open("finish_gen.txt", "w", encoding="utf-8") as f:
        f.write('finish generate')

    result_list = processor.decode(generated_ids, skip_special_tokens=True)
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(result_list)
except Exception as e:
    with open("error.txt", "w", encoding="utf-8") as f:
        f.write(str(e))