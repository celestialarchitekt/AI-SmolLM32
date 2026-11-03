
import sys
from gguf import GGUFReader

def inspect_gguf(model_path):
    print("=== Inspecting Architecure : {model_path}===")
    reader = GGUFReader(model_path)


    keys_to_show = [
            "general.architecture",
            "llama.block_count",
            "llama.embedding_length",
            "llama.attention.head_count",
            "llama.context_length"
            ]

    for key, field in reader.fields.items():
        if any(k in key for k in keys_to_show):
            val = field.parts[field.data[0]]
            print(f" {key}: {val}")



if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)>1 else "SmolLM2-135M-Instruct-Q8_0.gguf"
    inspect_gguf(path)
