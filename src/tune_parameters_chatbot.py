
# Tuning Parameters Part 2
# SmolLM2 - 135 here will behave like a chatbot and not like a text completion AI.

import sys
from llama_cpp import Llama

model_path = sys.argv[1] if len(sys.argv)>1 else "SmolLM2-135M-Instruct-Q8_0.gguf"

print(" Init SmolLM2-135 Model ")
messages = [{"role": "user", "content": "explain why running AI locally safer?"}]



print(" ------- Profile 1 ------ ")
llm_fast =Llama(
        model_path,
        n_ctx = 512,
        n_threads = 2,
        verbose = False,
        )


output_fast = llm_fast.create_chat_completion(
        messages = messages,
        max_tokens = 64,
        temperature = 0.1,
        top_p = 0.5,
        repeat_penalty = 1.2
        )

print(" Output: ", output_fast["choices"][0]["message"]["content"])


# ---------------------------------------------------------
print(" ----- Profile 2 ----- ")
llm_creative = Llama(
        model_path,
        n_ctx = 2048,
        n_threads = 4,
        verbose = False
        )

output_creative = llm_creative.create_chat_completion(
    messages = messages,
    max_tokens = 64,
    temperature = 1.1,
    top_p = 0.95,
    repeat_penalty = 1.0
    )


print(" Output: ", output_creative["choices"][0]["message"]["content"])












