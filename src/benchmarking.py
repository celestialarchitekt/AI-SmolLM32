# Benchmarking on Raspi5

import sys
import time
from llama_cpp import Llama

model_path = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-135M-Instruct-Q8_0.gguf"

print (" ----- Initialising Raspi5 ----- ")

llm = Llama(
        model_path = model_path,
        n_ctx = 8192, # max 8192 for SmolLM2-135
        n_threads = 4, # 1 to 4
        verbose = False
        )

messages = [{"role": "user", "content": "Which is the highest ranking AI model so far?"}]

print(" Running generation benchmark ")

start_time = time.time()
response = llm.create_chat_completion(
        messages = messages,
        max_tokens = 128, 
        temperature = 0.5 # 0.0 to 2.0
        )

end_time = time.time()


usage = response.get("usage", {})
completion_tokens = usage.get("completion_tokens",0)
prompt_tokens = usage.get("prompt_tokens",0)
total_time = end_time - start_time

print("\n ====== Results ===== ")
print(f"Prompt Tokens: {prompt_tokens}")
print(f"Completion Tokens: {completion_tokens}")
print(f"Total Time : {total_time} seconds")




