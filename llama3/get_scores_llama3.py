from typing import List, Optional
from dataclasses import dataclass, field
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
tqdm.pandas()
import json
import requests
import re

from transformers import HfArgumentParser, AutoTokenizer
from datasets import load_dataset

from templates import *


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    url: Optional[str] = field(
        default="http://localhost",
        metadata={"help": "url of the model server"}, #m
    )
    ports: List[str] = field(
        default_factory=lambda: ["8000"], 
        metadata={"help": "ports of the model server"}, #m
    )
    tokenizer: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct", #m
        metadata={"help": "the tokenizer/model to use"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="HuggingFaceH4/ultrafeedback_binarized", #m
        metadata={"help": "the name or location of the dataset"}, #m
    )
    # used_model_name: Optional[str] = field(
    #     default="HuggingFaceH4/mistral-7b-sft-beta",
    #     metadata={"help": "the model used to generate task responses"}, #m
    # )
    output_dir_data: Optional[str] = field(
        default=".json",
        metadata={"help": "the location of the sampled dataset file"},
    )
    output_dir_pref: Optional[str] = field(
        default=".json",
        metadata={"help": "the location of the preference file"},
    )
    # K: Optional[int] = field(
    #     default=8,
    #     metadata={"help": "the number of generations per prompt"},
    # )
    max_tokens: Optional[int] = field( #m
        default=1024, #m
        metadata={"help": "the maximum length of the generated tokens"}, #m
    )
    seed: Optional[int] = field(
        default=37, #m
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0, #m
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use beam search"}, #m
    )
    max_workers: Optional[int] = field(
        default=1024,
        metadata={"help": "the number of workers"},
    )
    role_num: Optional[int] = field(
        default=38,
        metadata={"help": "the number of tasks for each annotator"}
    )
    task_num: Optional[int] = field(
        default=1000,
        metadata={"help": "the number of tasks for each annotator"}
    )
    reverse: Optional[int] = field(
        default=0,
        metadata={"help": "whether or not to reverse the order of responses"}
    )


def get_scored_sample(role:str, task_id:int, entry:str, port: int) -> dict:
    role_description = ROLE_TEMPLATE[role]
    task_info = {
        "instruction": ds[task_id]["instruction"],
        "response": ds[task_id][entry]
    }
    message = [
        {"role": "system", 
        "content": role_description},
        {"role": "user", 
        "content": SCORE_TASK_TEMPLATE.format(**task_info)}
    ]
    prompt = tokenizer.apply_chat_template(
        message, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # output = llm.generate([prompt], sampling_params, use_tqdm=False)
    json = {**sampling_args, "prompt": prompt}
    response = requests.post(
        url=script_args.url+":"+str(port)+"/generate",
        json=json
    ).json()
    # response = output[0].outputs[0].text
    response = response["text"][0][len(prompt):]
    
    # print(response)
    pattern = r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
    matches = re.findall(pattern, response)
    if matches == []:
        A, B, C, D, tot = None, None, None, None, None
    else:
        A, B, C, D = matches[-1]
        A, B, C, D = int(A), int(B), int(C), int(D)
        tot = A + B + C + D
    scored_sample = {
        "role": role,
        "task_id": task_id,
        "entry": entry,
        "result": {
            "analysis": response,
            # "honesty": int(A),
            "truthfulness": A,
            "instruction-following": B,
            "helpfulness": C,
            "personal preference": D,
            "overall": tot
        }
    }
    return scored_sample


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

random.seed(script_args.seed)

roles = random.sample(ROLE_TEMPLATE.keys(), k=script_args.role_num)

print("[Loading dataset...]")
if script_args.dataset_name_or_path[-4:] == "json":
    ds = json.load(open(script_args.dataset_name_or_path))
    ds = ds["instances"]
    if script_args.reverse:
        ds = [{"instruction": sample["instruction"],
               "response1": sample["response2"],
               "response2": sample["response1"]} for sample in ds]
else:
    ds = load_dataset(script_args.dataset_name_or_path, split="train_prefs")
    sampled_id = random.sample(range(len(ds)), k=script_args.task_num)
    ds_tmp = ds[sampled_id] # format: {'prompt': [...], 'chosen': [...], 'rejected': [...]}
    ds = []
    for i in range(len(ds_tmp["prompt"])):
        order = ["chosen","rejected"]
        random.shuffle(order)
        ds.append({"instruction": ds_tmp["prompt"][i],
                "response1": ds_tmp[order[0]][i][1]["content"],
                "response2": ds_tmp[order[1]][i][1]["content"]})
sampled_dataset = {}
sampled_dataset['type'] = 'text_only'
sampled_dataset['instances'] = ds
with open(script_args.output_dir_data, 'w', encoding='utf8') as f:
    json.dump(sampled_dataset, f, ensure_ascii=False)
print("Successfully loaded and saved!")

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer)
sampling_args = {
    "use_beam_search": script_args.use_beam_search,
    "n": 1,
    "temperature": script_args.temperature,
    "max_tokens": script_args.max_tokens,
    "seed": script_args.seed,
    "top_p": 0.6,
    "top_k": -1,
    "stop_token_ids": [tokenizer.eos_token_id,
                       tokenizer.convert_tokens_to_ids("<|eot_id|>")]
}

single_response_dicts = []
with ThreadPoolExecutor(max_workers=script_args.max_workers) as executor:
    single_response_dicts = [executor.submit(get_scored_sample, role, task_id, entry, 8000) 
                             for role in roles for task_id in range(len(ds)) for entry in ["response1", "response2"]]
    # use tqdm to show progress
    for _ in tqdm(as_completed(single_response_dicts), total=len(single_response_dicts), desc="Role-task pairs"):
        pass
    single_response_dicts = [d.result() for d in single_response_dicts]

assert(len(single_response_dicts) == 2 * script_args.role_num * script_args.task_num)
dataset_dicts = []
for i in range(0, len(single_response_dicts), 2):
    d1, d2 = single_response_dicts[i], single_response_dicts[i+1]
    assert(d1["role"] == d2["role"])
    assert(d1["task_id"] == d2["task_id"])
    pref = 2
    if d1["result"]["overall"] == None or d2["result"]["overall"] == None:
        pref = -1
    elif d1["result"]["overall"] > d2["result"]["overall"]:
        pref = 0
    elif d1["result"]["overall"] < d2["result"]["overall"]:
        pref = 1
    dataset_dicts.append({
        "role": d1["role"],
        "task_id": d1["task_id"],
        d1["entry"]: d1["result"],
        d2["entry"]: d2["result"],
        "preference": pref
    })
assert(len(dataset_dicts) == script_args.role_num * script_args.task_num)

personalized_dataset = {}
personalized_dataset['type'] = 'text_only'
personalized_dataset['instances'] = dataset_dicts
with open(script_args.output_dir_pref, 'w', encoding='utf8') as f:
    json.dump(personalized_dataset, f, ensure_ascii=False)