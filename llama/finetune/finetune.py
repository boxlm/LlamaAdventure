import torch

from transformers import LlamaForCausalLM, LlamaTokenizer
from llama.dataset import get_dataset

model_id = ''

## Get tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit = True, device_map = 'auto', torch_dtype = torch.float16)

## Load data 

get_dataset()


