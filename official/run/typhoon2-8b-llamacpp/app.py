from llama_cpp import Llama

llm = Llama(
      model_path="../model/typhoon-8b-cpp/llama3.1-typhoon2-8b-instruct-q8_0.gguf",
      n_gpu_layers=-1,
      verbose=False,
      chat_format='llama-3'
)

output = llm.create_chat_completion(
      messages = [
          {"role": "system", "content": "You are an assistant who perfectly describes images."},
          {
              "role": "user",
              "content": "ขอสูตรไก่ย่างหน่อย"
          }
      ]
)