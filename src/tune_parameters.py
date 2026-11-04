# SmolLM2 - 135
# Tuning Parameters
# SmolLM2-135 here behaves like a text completion model



import sys
from llama_cpp import Llama

model_path = sys.argv[1] if len(sys.argv)>1 else "SmolLM2-135M-Instruct-Q8_0.gguf"

print("----  Init SmolLM2 - 135 ----- ")

print ("\n" +  "-"*10 + "Profile 1" + "\n") 
llm_fast = Llama(
        model_path = model_path,
        n_ctx = 512,
        n_threads = 4,
        verbose = False
    )


prompt = "explain why running AI locally is more secure"

output_first = llm_fast(
        prompt,
        max_tokens = 64,
        temperature = 0.1, 
        top_p = 0.5,
        repeat_penalty = 1.2
        )

print(f"Prompt: {prompt}")
print("Output: ", output_first["choices"][0]["text"])



print("\n" + "*"*10 + "Profile 2" + "\n")
llm_creative = Llama(
        model_path = model_path,
        n_ctx = 2048,
        n_threads = 4,
        verbose = False
        )


output_creative = llm_creative(
        prompt,
        max_tokens = 64,
        temperature = 1.1,
        top_p = 0.95,
        repeat_penalty = 1.0

    )

print(f"Prompt: {prompt}")
print("Output: ", output_creative["choices"][0]["text"])






