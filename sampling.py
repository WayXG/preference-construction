from datasets import Dataset, load_dataset
import pandas as pd
import random
import json

import copy 

model_pool = [
    'google/gemma-7b-it', 
    'meta-llama/Llama-2-7b-chat-hf', 
    'meta-llama/Llama-2-13b-chat-hf',
    'mistralai/Mistral-7B-Instruct-v0.2',
    'HuggingFaceH4/mistral-7b-sft-beta',
    'snorkelai/Snorkel-Mistral-PairRM-DPO',
    'berkeley-nest/Starling-LM-7B-alpha',
    'mistralai/Mistral-7B-Instruct-v0.1',
    'Qwen/Qwen1.5-14B-Chat',
    'upstage/SOLAR-10.7B-Instruct-v1.0',
    'GPT-3.5-Turbo',
    'GPT-4',
    #'Qwen/Qwen1.5-7B-Chat',
    #'lmsys/vicuna-7b-v1.5',
    # '01-ai/Yi-6B-Chat',
    # 'deepseek-ai/deepseek-coder-6.7b-instruct',
    #'baichuan-inc/Baichuan2-7B-Chat',
]

def prompt_format(txt):
    return [{"role": "user", "content": txt}]

def sample_models():
    return random.sample(model_pool, 6)

for i in range(10):
    print(sample_models())


ds = load_dataset("DIBT/10k_prompts_ranked", split='train')
#print(list(set(ds['cluster_description'])))
ds = ds.map(lambda x: {"models": sample_models(), "completions": [
], 'message_prompt': prompt_format(x['prompt'])})
gathered_data = []

for sample in ds:
    gathered_data.append(sample)


output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = gathered_data
with open("/home/xw/ultra/data/raw_data.json", 'w', encoding='utf8') as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)
