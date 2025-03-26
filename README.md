# AIXpress

<div align="center">
  
**Download, serve, and interact with state-of-the-art LLMs through a unified API**

</div>

## 📚 Overview

AIXpress is a powerful, user-friendly platform for downloading, serving, and interacting with Large Language Models (LLMs). It provides a unified API interface to work with multiple LLM architectures while abstracting away the complexities of model management, optimization, and inference.

### Key Features

- **🤖 Multi-Model Support**: Compatible with Llama, Mistral, Gemma, Phi, and many other open-source models
- **📥 Easy Model Management**: Download and manage models with a simple CLI or through the web interface
- **⚡ Optimized Inference**: Automatically applies quantization and optimization techniques based on your hardware
- **🔄 Format Conversion**: Easily convert between GGML, GGUF, SafeTensors, PyTorch formats
- **🌐 RESTful API**: Consistent API for all models with streaming and non-streaming options
- **💻 Web UI**: Built-in interface for testing prompts and comparing model outputs
- **📊 Benchmarking**: Tools to evaluate and compare model performance
- **🔌 Plugin System**: Extend functionality with custom pre/post-processing plugins

## 🚀 Quick Start

### Installation

#### Using pip

```bash
pip install aixpress
```

#### Using Docker

```bash
docker pull PawanRamaMali/aixpress:latest
docker run -p 8000:8000 -v ~/aixpress-models:/app/models PawanRamaMali/aixpress:latest
```

### Basic Usage

1. Download a model:

```bash
aixpress download meta-llama/Llama-3-8B-Instruct
```

2. Start the server:

```bash
aixpress serve
```

3. Send a request:

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3-8B-Instruct",
    "prompt": "Explain quantum computing in simple terms",
    "max_tokens": 250,
    "temperature": 0.7
  }'
```

## 🛠️ Installation Options

### Prerequisites

- Python 3.8+
- For GPU support: CUDA 11.8+ (NVIDIA) or ROCm 5.6+ (AMD)
- 8GB RAM minimum (16GB+ recommended for larger models)

### From Source

```bash
git clone https://github.com/PawanRamaMali/aixpress.git
cd aixpress
pip install -e .
```

### Standalone Binary (Linux/macOS)

```bash
curl -sSL https://get.aixpress.ai | bash
```

## 📋 Detailed Documentation

### Downloading Models

AIXpress simplifies model acquisition from multiple sources:

```bash
# From Hugging Face
aixpress download meta-llama/Llama-3-8B-Instruct

# Specifying format and quantization
aixpress download mistralai/Mistral-7B-Instruct-v0.2 --format gguf --quantization q4_k_m

# From local file
aixpress import /path/to/local/model

# Convert between formats
aixpress convert meta-llama/Llama-3-8B --from pytorch --to gguf --quantization q5_k_m
```

### Running the Server

Start the API server with customizable settings:

```bash
# Basic server
aixpress serve

# With custom configuration
aixpress serve --host 0.0.0.0 --port 8080 --model meta-llama/Llama-3-8B-Instruct --gpu-layers 32
```

### Configuration

Create a configuration file at `~/.aixpress/config.yml`:

```yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 2
  cors_origins: ["*"]

models:
  default: meta-llama/Llama-3-8B-Instruct
  cache_dir: ~/.aixpress/models
  
hardware:
  gpu_layers: -1  # Use all available GPU layers
  threads: 8
  batch_size: 512
  
api:
  enable_metrics: true
  max_tokens_per_request: 4096
  request_timeout: 120
```

### API Reference

AIXpress provides a consistent API across all models:

#### Generate Completion

```http
POST /v1/generate
Content-Type: application/json

{
  "model": "meta-llama/Llama-3-8B-Instruct",
  "prompt": "Write a function to calculate factorial in Python",
  "system_prompt": "You are a helpful coding assistant.",
  "max_tokens": 500,
  "temperature": 0.7,
  "top_p": 0.9,
  "stop": ["\n\n", "```"],
  "stream": false
}
```

#### Stream Completion

Same endpoint with `"stream": true` returns Server-Sent Events (SSE).

#### Chat Completion

```http
POST /v1/chat
Content-Type: application/json

{
  "model": "meta-llama/Llama-3-8B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "max_tokens": 250,
  "temperature": 0.7
}
```

#### Model Management

```http
GET /v1/models
```

Returns a list of all available models with their details.

### Web Interface

AIXpress includes a web UI accessible at `http://localhost:8000/ui` for:

- Interactive prompt testing
- Model comparison
- Parameter tuning
- Performance monitoring

## 🔧 Advanced Usage

### Multiple Model Loading

Load multiple models in the same server instance:

```bash
aixpress serve --models meta-llama/Llama-3-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.2
```

### Custom Prompting Formats

Configure custom prompt templates for different models:

```yaml
# In ~/.aixpress/templates.yml
templates:
  llama-3:
    chat: |
      <|system|>
      {system_prompt}
      <|user|>
      {prompt}
      <|assistant|>
    
  mistral:
    chat: |
      [INST] {system_prompt} [/INST]
      
      [INST] {prompt} [/INST]
```

### GPU Memory Management

Control GPU memory allocation:

```bash
aixpress serve --gpu-layers 32 --max-batch-size 512 --max-context-size 8192
```

### Plugin Development

Create custom plugins to extend AIXpress functionality:

```python
from aixpress.plugins import AIXpressPlugin

class MyCustomPlugin(AIXpressPlugin):
    def on_request(self, request):
        # Modify request before processing
        return request
        
    def on_response(self, response):
        # Modify response before returning to client
        return response

# Register in ~/.aixpress/plugins/my_plugin.py
```

## 🧪 Benchmarking

AIXpress includes tools to benchmark model performance:

```bash
aixpress benchmark --models meta-llama/Llama-3-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.2 --tasks gsm8k,mmlu,hellaswag --quantizations q4_k_m,q5_k_m,q8_0
```

## 📊 Monitoring & Metrics

Enable Prometheus metrics:

```bash
aixpress serve --metrics --metrics-port 9100
```

Grafana dashboard templates are available in the `monitoring` directory.

## 🔐 Security

### API Authentication

Enable API key authentication:

```bash
aixpress serve --auth-enabled --auth-file ~/.aixpress/api_keys.yml
```

Create the API keys file:

```yaml
# ~/.aixpress/api_keys.yml
keys:
  - key: sk-aixpress-abcdefg123456
    name: "Dev Key"
    permissions: ["generate", "chat", "list_models"]
  - key: sk-aixpress-7654321gfedcba
    name: "Admin Key"
    permissions: ["*"]
```

### Model Access Control

Restrict model access by API key:

```yaml
# ~/.aixpress/api_keys.yml
keys:
  - key: sk-aixpress-abcdefg123456
    name: "Restricted Key"
    permissions: ["generate", "chat"]
    models: ["meta-llama/Llama-3-8B-Instruct"]  # Can only access this model
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/PawanRamaMali/aixpress.git
cd aixpress
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest tests/
```

## 📝 License

AIXpress is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

AIXpress builds upon several excellent open-source projects:

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [vLLM](https://github.com/vllm-project/vllm)
- [FastAPI](https://github.com/tiangolo/fastapi)

## 📞 Support

- [GitHub Issues](https://github.com/PawanRamaMali/aixpress/issues)
- [Discord Community](https://discord.gg/aixpress)
- [Documentation](https://docs.aixpress.ai)
