import torch
from trimkv.models.qwen3 import TrimKVQwen3ForCausalLM
from trimkv.cache_utils import DynamicBudgetTrimKVCache
from transformers import AutoTokenizer

# model_path = "ngocjr7/trimkv_models/trimkv_qwen3_4b_openr1_math_512m:v0"
# download_from = "wandb" # there options: wandb, local, huggingface

model_path = "ngocbh/TrimKV-Qwen3-4B-Math"
download_from = "huggingface" # there options: wandb, local, huggingface

model = TrimKVQwen3ForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    load_trimkv_weights=True,
    download_from=download_from,
    use_cache=True,
    device_map="cuda",
)
model.config._attn_implementation = 'db_attn_flash'  
model.config.compress_memory = True
model.config.memory_size = 256
model.config.buffer_size = 32
# floor_budget_ratio=0. gives global top-k at 0, default is 0

tokenizer = AutoTokenizer.from_pretrained(
    model.config.base_model, use_fast=True, padding_side="left"
)

prompt = "A train travels 120 kilometers in 3 hours at a constant speed. How many kilometers will it travel in 5 hours at the same speed?"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
past_key_values = DynamicBudgetTrimKVCache(device='cuda')

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768,
    past_key_values=past_key_values,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

print("generated_text:", generated_text)
