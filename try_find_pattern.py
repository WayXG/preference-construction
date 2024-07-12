from datasets import load_dataset
import json

#dataset = load_dataset("json", data_files="/home/guhao/xw/ultra/data/iter12.json",
#                       split="train", field="instances")  # .select(range(100))

gathered_data = []

ds = load_dataset("cornfieldrm/10k_ver1", split="train")

for sample in ds:
    gathered_data.append(sample)


output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = gathered_data
with open("/data/raw_data.json", 'w', encoding='utf8') as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)


#dataset.push_to_hub("cornfieldrm/10k_ver1")
