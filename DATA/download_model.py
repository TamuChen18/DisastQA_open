from huggingface_hub import snapshot_download

MODELS = {
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen-3-8b": "Qwen/Qwen3-8B",  # 
    "deepseek-v3-7b": "deepseek-ai/deepseek-llm-7b-chat", 
    "gemma-7b": "google/gemma-7b-it",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Hunyuan-7B-Instruct": "tencent/Hunyuan-7B-Instruct",
    "qwen-3-4b": "Qwen/Qwen3-4B",
    "Hunyuan-4B-Instruct": "tencent/Hunyuan-4B-Instruct",
    "Yi-6B-Chat": "01-ai/Yi-6B-Chat",
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen-2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "phi-2": "microsoft/Phi-2",
    "AceMath-1.5B-Instruct": "nvidia/AceMath-1.5B-Instruct",
    "Falcon3-1B-Instruct": "tiiuae/Falcon3-1B-Instruct",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen-3-0.6b": "Qwen/Qwen3-0.6B",
}


for name, repo_id in MODELS.items():
    print(f"Downloading {name} from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=f"/home/shared/RAG_DATA/DATA/models/{name}",
        local_dir_use_symlinks=False
    )
    print(f"✅ {name} downloaded!\n")
