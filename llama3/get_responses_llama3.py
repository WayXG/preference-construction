from typing import List, Optional
from dataclasses import dataclass, field
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
tqdm.pandas()
import json
import requests

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
        default=44,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=1.0,
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


def get_annotated_sample(role:str, task_id:int, port: int) -> dict:
    role_description = ROLE_TEMPLATE[role]
    message = [
        {"role": "system", 
        "content": role_description},
        {"role": "user", 
        "content": TASK_HEAD + "\n" + TASK_TEMPLATE.format(**ds[task_id])}
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
    
    annotated_sample = {"role": role,
                        "task_id": task_id,
                        "analysis": response}
    if "[[1]]" in response and "[2]" not in response:
        annotated_sample["preference"] = 0
    elif "[[2]]" in response and "[1]" not in response:
        annotated_sample["preference"] = 1
    elif "[1]" in response and "[2]" in response:
        annotated_sample["preference"] = 2
    else:
        annotated_sample["preference"] = -1
    return annotated_sample


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

random.seed(script_args.seed)

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

# llm = LLM(model=script_args.model, 
#             trust_remote_code=True, 
#             seed=script_args.seed,
#             tensor_parallel_size=1) #m
tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer)
sampling_args = {
    "use_beam_search": script_args.use_beam_search,
    "n": 1,
    "temperature": script_args.temperature,
    "max_tokens": script_args.max_tokens,
    "seed": script_args.seed,
    "top_p": 1.0,
    "top_k": -1,
    "stop_token_ids": [tokenizer.eos_token_id,
                       tokenizer.convert_tokens_to_ids("<|eot_id|>")]
}
# sampling_params = SamplingParams(temperature=script_args.temperature, 
#                                  max_tokens=script_args.max_tokens, 
#                                  stop_token_ids=terminators, 
#                                  seed=script_args.seed)

roles = random.sample(ROLE_TEMPLATE.keys(), k=script_args.role_num)
dataset_dicts = []
# with ThreadPoolExecutor(max_workers=script_args.max_workers) as executor:
#     for role in roles:
#         role_dicts = [executor.submit(get_annotated_sample, role, 
#                                       task_id, script_args.ports[task_id % len(script_args.ports)])
#                       for task_id in range(len(ds))]
#         for _ in tqdm(as_completed(role_dicts), total=len(role_dicts), desc=role):
#             pass
#         role_dicts = [d.result() for d in role_dicts]
#         dataset_dicts += role_dicts
with ThreadPoolExecutor(max_workers=script_args.max_workers) as executor:
    dataset_dicts = [executor.submit(get_annotated_sample, role, task_id, 8000) 
                     for role in roles for task_id in range(len(ds))]
    # use tqdm to show progress
    for _ in tqdm(as_completed(dataset_dicts), total=len(dataset_dicts), desc="Role-task pairs"):
        pass
    dataset_dicts = [d.result() for d in dataset_dicts]

personalized_dataset = {}
personalized_dataset['type'] = 'text_only'
personalized_dataset['instances'] = dataset_dicts
with open(script_args.output_dir_pref, 'w', encoding='utf8') as f:
    json.dump(personalized_dataset, f, ensure_ascii=False)