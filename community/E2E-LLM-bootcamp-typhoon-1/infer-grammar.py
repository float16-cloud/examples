from llama_cpp.llama import Llama, LlamaGrammar

grammar = """root ::= ("negative" | "positive" | "neutral") """
model_name = "../gguf-model/my-typhoon2.gguf"

input_sent = "อากาศไม่ดีเลยวันนี้"
user_input = f"Analyze the sentiment of the following text:\nContext: {input_sent}\nSentiment:"

llm = Llama(model_path = model_name, n_gpu_layers=-1, verbose=False, chat_format='llama-3', n_ctx=1024 * 128, seed=42)
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
print(f"\nOutput without grammar: <b>{output['choices'][0]['message']['content']}</b>\n")

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
print(f"Output with grammar: <b>{output['choices'][0]['message']['content']}</b>\n")