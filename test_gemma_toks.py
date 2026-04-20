import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b-it')
eoi_str = "<end_of_turn>\n<start_of_turn>model\n"
eoi_toks = tokenizer.encode(eoi_str, add_special_tokens=False)
print("eoi_toks length:", len(eoi_toks))
print(eoi_toks)

decoded = tokenizer.batch_decode(eoi_toks)
print("decoded length:", len(decoded))
print(decoded)
