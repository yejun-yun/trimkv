import torch
from trimkv.models.qwen3 import TrimKVQwen3ForCausalLM
from trimkv.cache_utils import BatchedDynamicBudgetTrimKVCache
from transformers import AutoTokenizer

model_path = "ngocbh/TrimKV-Qwen3-4B-Math"
download_from = "huggingface"

model = TrimKVQwen3ForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    load_trimkv_weights=True,
    download_from=download_from,
    use_cache=True,
    device_map="cuda",
)
model.config._attn_implementation = 'db_attn_flash_batched'
model.config.compress_memory = True
model.config.memory_size = 256
model.config.buffer_size = 32

tokenizer = AutoTokenizer.from_pretrained(
    model.config.base_model, use_fast=True, padding_side="left"
)
tokenizer.pad_token = tokenizer.eos_token

prompts = [
    """A train travels 120 kilometers in 3 hours at a constant speed. How many kilometers will it travel in 5 hours at the same speed?""",
    """A train travels 120 kilometers in 3 hours at a constant speed. How many kilometers will it travel in 5 hours at the same speed?""",
]

messages_list = [
    [{"role": "user", "content": prompt}]
    for prompt in prompts
]

texts = [
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    for messages in messages_list
]

model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
past_key_values = BatchedDynamicBudgetTrimKVCache(device='cuda')

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768,
    past_key_values=past_key_values,
)

for i, (input_ids, output_ids) in enumerate(zip(model_inputs.input_ids, generated_ids)):
    response_ids = output_ids[len(input_ids):].tolist()
    generated_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip("\n")
    print(f"=== Response {i} ===")
    print(generated_text)
    print()
