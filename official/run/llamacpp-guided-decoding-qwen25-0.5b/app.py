from llama_cpp.llama import Llama, LlamaGrammar

grammar = """root ::= ("เชียงใหม่"|"กรุงเทพ"|"ภูเก็ต")"""
model_name = "../qwen2.5-0.5b-instruct-q4_0-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf"
user_input = "แนะนำสถานที่ท่องเที่ยวให้หน่อย ทะเลให้หน่อย"

llm = Llama(
    model_path = model_name,
    n_gpu_layers=-1,
    verbose=False,
    chat_format='qwen',
    n_ctx=1024 * 32,
    seed=42
)
grammar = LlamaGrammar.from_string(grammar)

output = llm.create_chat_completion(
    messages = [
        {
            "role": "user",
            "content": user_input
        }
    ],
    # grammar = grammar,
    max_tokens=64,
)
print(f"Output without grammar: {output['choices'][0]['message']['content']}")

output = llm.create_chat_completion(
    messages = [
        {
            "role": "user",
            "content": user_input
        }
    ],
    grammar = grammar,
    max_tokens=64,
)
print(f"Output with grammar: {output['choices'][0]['message']['content']}")