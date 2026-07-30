This directory vendors the tokenizer files required by
`data_agent_baseline.token_utils`.

Expected layout:

```text
assets/huggingface/
  Qwen3.5-35B-A3B/
    chat_template.jinja
    config.json
    merges.txt
    tokenizer.json
    tokenizer_config.json
    vocab.json
```

When this local tokenizer directory exists, the code loads it directly with
`AutoTokenizer.from_pretrained(<local path>)` and enables Hugging Face offline
mode. This avoids network access during catalog construction.
