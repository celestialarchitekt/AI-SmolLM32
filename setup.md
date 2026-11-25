# Llama-cpp

Llama.cpp is an open source MIT licensed inference engine written in C/C++, that runs large language models (LLMS) and multimodal models locally. Llama.cpp made inference practical on CPUs without a data center GPU. It uses the GGUF file format. It supports 4 bit and 8 bit quantisation, so models can fit into smaller RAM. Llama.cpp hosts more than 150 models, spanning from Llama, Qwen, Mistral, Gemma, and DeepSeek.

It includes cli tools and llama-server (an lightweight HTTP server with an OpenAI compatible endpoints and a WebUI). 

## Setting up llama.cpp

The setup is made on the Raspi5 running Linux Kali (64bit).

** * Installing Build Tools and Curl
```
sudo apt update && sudo apt install -y git cmake build-essential curl
```

** * Clone and Build llama.cpp
```
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CURL=ON
cmake --build build --config Release -j4
```

** * Download the SmolLM2 GGUF model
```
mkdir -p models
curl -L -o models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf
```

** * Run a test prompt
```
./build/bin/llama-cli \
  -m models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  -p "Give me a one-sentence tip for Linux command-line productivity." \
  -n 64 \
  -t 4
```

 
