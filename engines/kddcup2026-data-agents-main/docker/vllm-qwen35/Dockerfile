FROM vllm/vllm-openai:cu130-nightly

# Keep runtime deterministic and avoid HF Xet stalls in constrained networks.
ENV HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

# Ensure latest Qwen3.5-compatible transformer stack.
RUN python3 -m pip install --no-cache-dir -U \
    "transformers>=5.0.0" \
    "huggingface_hub>=0.31.0" \
    "hf-transfer>=0.1.9"

