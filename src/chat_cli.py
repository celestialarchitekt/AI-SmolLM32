import sys
from llama_cpp import Llama

model_path = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-135M-Instruct-Q8_0.gguf"

print("Loading model into Pi 5 RAM...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4, # Pi 5 has 4 CPU cores
    verbose=False
)

print("\nModel ready! Type 'exit' to quit.\n" + "-"*40)

messages = [{"role": "system", "content": "You are a helpful assistant running locally on a Raspberry Pi 5."}]

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
        
    messages.append({"role": "user", "content": user_input})
    response = llm.create_chat_completion(messages=messages, temperature=0.7)
    
    reply = response["choices"][0]["message"]["content"]
    print(f"\nSmolLM2: {reply}\n")
    messages.append({"role": "assistant", "content": reply})
