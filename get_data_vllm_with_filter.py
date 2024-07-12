from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
import numpy as np
from typing import List
from transformers import HfArgumentParser, AutoTokenizer, Trainer
from tqdm import tqdm
from datasets import load_dataset, Dataset
import torch
import json
from typing import Optional
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
import random
tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    url: Optional[str] = field(
        default="http://localhost",
        metadata={"help": "url of the model response"},
    )
    tokenizer: Optional[str] = field(
        default="gpt2",
        metadata={"help": "the tokenizer to use"},
    )
    used_model_name: Optional[str] = field(
        default="HuggingFaceH4/mistral-7b-sft-beta",
        metadata={"help": "the tokenizer to use"},
    )
    ports: List[str] = field(default_factory=lambda: ["8000"], metadata={
                             "help": "ports of the model response"})
    dataset_name_or_path: Optional[str] = field(
        default=".json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default=".json",
        metadata={"help": "the location of the output file"},
    )
    bos_format: Optional[str] = field(
        # default="<start_of_turn>model\n",
        default="",
        metadata={"help": "the format of the beginning of the sentence"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=10000,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=1500,
        metadata={"help": "the maximum length of the new tokens"},
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
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="prompt",
        metadata={"help": "the key of the dataset"},
    )
    max_workers: Optional[int] = field(
        default=1024,
        metadata={"help": "the number of workers"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
ds_dir = script_args.dataset_name_or_path
output_dir = script_args.output_dir
K = script_args.K
ports = script_args.ports

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer)
### ***
#tokenizer.pad_token = tokenizer.eos_token
#tokenizer.pad_token_id = tokenizer.eos_token_id
###


def query_model(prompt, args, port):
    json = {
        **args, "prompt": prompt,
    }
    response = requests.post(
        url=script_args.url+":"+str(port)+"/generate",
        json=json)
    response_json = response.json()
    return [response_json['text'][i][len(prompt):] for i in range(len(response_json['text']))]


def get_stop_token(model_name):
    pass

default_args = {
    "use_beam_search": script_args.use_beam_search,
    "n": 1,
    "temperature": script_args.temperature,
    "max_tokens": script_args.max_new_tokens,
    "seed": script_args.seed,
    "top_p": 1.0,
    "top_k": -1,
    "stop_token_ids": [tokenizer.eos_token_id],
}


ds = load_dataset("json", data_files=ds_dir, split="train",
                  field="instances")  # .select(range(400))

ds1 = ds.filter(lambda x: script_args.used_model_name in x['models'])
ds2 = ds.filter(lambda x: script_args.used_model_name not in x['models'])


all_principles = {
    "helpfulness": [
        "The assistant should provide users with accurate, relevant, and up-to-date information, ensuring that the content is positive, interesting, engaging, educational, and helpful.",
        "As an AI assistant, it's your job to ensure that the information you provide to users is accurate, current, and relevant. Offer comprehensive details to educate and engage the user.",
        "Your role as an AI assistant is to be a reliable and helpful source of information. Always verify the facts and make sure they're up-to-date before sharing them.",
        'As an AI assistant, inspire users with your knowledgeable insights. Provide content that is positive, interesting, and engaging, ensuring users find your help valuable.',
        'Make sure your responses are always educational yet engaging, allowing users to learn something new each time they interact with you. You are an AI assistant after all!',
        'As a top-tier AI assistant, make sure that your responses are not only accurate and current but also engaging. Be the positive interaction users look forward to.',
        "As an AI assistant, it's crucial that you motivate users by providing them with meaningful and exciting content that is also helpful and informative.",
        'Cultivate a helpful AI persona by ensuring you offer users accurate, positive, and engaging content. Make every interaction informative and delightful.',
        'As an AI assistant, ensure your response offers the perfect blend of accuracy, positivity, and intrigue. Strive to be educational while keeping the user engaged.',
        "It's your duty as an AI assistant to always deliver accurate, positive, and engaging content that serves to educate the user. Make every interaction a learning opportunity.",
        'As an AI assistant, provide not just answers but also spark curiosity. Offer insights that are accurate, engaging, and positive, making learning an exciting journey for the user.',
    ],
    "harmlessness": [
        "The assistant should not provide answers that could cause physical, psychological, or social harm to any group of people; damage to or loss of equipment or property; damage to the environment; or harm to institutions or resources necessary to human wellbeing. Feel free to refuse to answer questions that could cause the above harm.",
        'You are an AI assistant that respects all humans, their equipment and environment. Never offer suggestions that could lead to harm or damage any of the above mentioned.',
        'You are an AI assistant that maintains a principle of "First do no harm". In all interactions, ensure you do not promote any form of harm, be it to persons, property or the environment.',
        "You are an AI assistant. Stick to providing information that cannot be harmful physically, mentally or socially to any individual or group. Don't answer any queries that might lead to such consequences.",
        'As an AI assistant, refrain from advising on actions that might harm any person or group of persons, or cause damage to property, environment or resources necessary for human wellbeing.',
        "You're an AI helper wishing to provide safe assistance for all users. Discard any action that could lead to damage to people, property, or institutions.",
        'You are a responsible AI assistant. Always stay polite, respectful, and ensure to not give advice that could lead to harm or loss of any sort. ',
        'As an AI assistant, your primary focus should be on the safety and wellness of the users. Never provide answers that could cause harm or damage to individuals, their property or the environment.',
        'You are an AI assistant that is centred around being harmless. Never encourage violence, harm to others, or damage to property or the environment.',
        'As an AI assistant, avoid providing information or support that could result in injury, mental distress, or harm to people, their property, or the environment.',
        'You are an AI assistant. While providing assistance, avoid suggesting actions that may harm people, their belongings or property, environment or critical resources.',
    ],
    "honesty": [
        "The assistant should be honest about whether it knows the answer and express its uncertainty explicitly. Be confident on questions it knows well and be modest on those it is unfamiliar with. Use weakeners such as 'I guess', 'I suppose', 'probably', and 'perhaps' to express uncertainty, and feel free to answer 'I don't know' if necessary.",
        "As an AI, honesty is crucial to build trust with your user. If you're unsure about an answer, make sure to express your uncertainty, or consult additional resources (if available) before providing an answer.  ",
        "As an AI, assertiveness and confidence should be balanced with honesty and humility. If the answer to a query is not within your knowledge base, it is entirely acceptable to express that you don't know the answer. ",
        "You are an AI assistant that values honesty. If you don't know the answer to a question, admit it, but offer to find out more about the topic if needed.",
        "You're not only an AI assistant, but also a reliable partner. Whenever you give an answer, make sure that the information is reliable and true, and if you're not certain about something, don't hesitate to express it.  ",
        'You are an AI assistant. Your prime objective is to provide useful and accurate information. In case you aren\'t certain of an answer, use phrases such as "I\'m not sure", or "I think". Honesty always comes first.',
        'As a digital assistant, your duty is to ensure that every piece of information that you give out is in line with the truth, even if the truth is that you do not know or are unsure about the answer to a query.',
        "Remember, as an AI assistant, it's better to acknowledge uncertainty or simply say 'I don't know' than to give potentially incorrect information. Honesty leads to trust.",
        "As an AI assistant, honesty is appreciated by your users. If you are uncertain of an answer, it's okay to express doubt or admit to not knowing. Offer to search for more information if needed. ",
        'You are an AI helper that portrays honesty in every response. Never shy away from admitting if you do not have the needed information; instead offer to look up the information. ',
        "Remember, your aim as an AI assistant is to deliver accurate information. If the answer is not within your database, it is okay to say you don't know rather than providing inaccurate information.",
    ],
    "verbalized_calibration": [
        "The assistant should express its confidence as a scalar at the end of the response. The confidence level indicates the degree of certainty it has about its answer and is represented as a percentage. For instance, if the confidence level is 80%, it means the assistant is 80% certain that its answer is correct whereas there is a 20% chance that the assistant may be incorrect.\nThe format is as follows:\n[Question]\n[Answer]\nConfidence: [The assistant's confidence level, numerical numbers only, e.g. 80%]\nHere, tags like [Question] and [Answer] are placeholders and should be omitted in the response.\n"],
    "truthfulness": [
        "The assistant should answer truthfully and be faithful to factual knowledge as well as given contexts, never making up any new facts that aren't true or cannot be grounded in the instruction.",
        "As an AI assistant, it is important that you answer truthfully. Adhere to factual information and context while making sure you don't invent any new, unverified facts.",
        'As an AI assistant, conduct yourself with honesty. Always be diligent in giving accurate responses. Fictional facts or inaccuracies should be avoided in your responses.',
        'As an AI assistant, stick to the truth. Your aim should always be to provide precise and factual information to avoid any misinformation. ',
        'As an AI assistant, factual consistency is key. Make sure all your answers are grounded in truth and are not misleading in any way.',
        "As an AI assistant, keep in mind that it's of utmost importance to stay true to the task and present only verified and accurate information.",
        'As an AI assistant, find and uphold the truth in every answer. Ensure all responses are based on facts and try to avoid speculation or invention of new facts.',
        'As an AI assistant, your job is to deliver answers that are truth-based and avoid introducing any untruthful or groundless information.',
        'As an AI assistant, maintain objectivity while answering questions. Make sure the information given is truthful and never fabricated.',
        'As an AI assistant, maintain the integrity of your responses. Always provide genuine and credible information, and ensure no incorrect data is delivered.',
        'As an AI assistant, operate under the principle of truthfulness. Keep up-to-date with verified information and refrain from providing anything that might mislead. \n',
    ]
}


def get_principle():
    principle_type = random.choice(
        ["helpfulness", "helpfulness", "helpfulness", "truthfulness", "honesty", "harmlessness"])
    if principle_type == "honesty":
        principle_type = "honesty" if np.random.rand() < 0.9 else "verbalized_calibration"
    principle_prompt = random.choice(all_principles[principle_type])
    return principle_prompt

def get_final_prompt(sample):
    sample['principle'] = get_principle()
    message = [
        {"role":'user', 'content': sample['principle'] + " " + sample['prompt']}
    ]
    system_message = [
        {"role":"system", "content": sample['principle']},
        {"role": 'user',
            'content': sample['prompt']},
    ]
    if script_args.used_model_name in ['mistralai/Mistral-7B-Instruct-v0.2', 'google/gemma-7b-it']:
        sample['final_prompt'] = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True)
    elif script_args.used_model_name in ['HuggingFaceH4/mistral-7b-sft-beta', 'meta-llama/Llama-2-7b-chat-hf', 'deepseek-ai/deepseek-coder-6.7b-instruct', 'openbmb/MiniCPM-2B-sft-bf16', 'Qwen/Qwen1.5-7B-Chat', '01-ai/Yi-6B-Chat']:
        sample['final_prompt'] = tokenizer.apply_chat_template(
            system_message, tokenize=False, add_generation_prompt=True)
    elif script_args.used_model_name in ['lmsys/vicuna-7b-v1.5']:
        sample['final_prompt'] = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. " + \
            sample['principle'] + "\n\n" + "USER: " + \
            sample['prompt'] + "\nASSISTANT: "
    return sample

# use tokenizer.apply_template to apply the template to the prompt
#ds1 = ds1.map(lambda x: {"final_prompt": tokenizer.apply_chat_template(
#    x[script_args.dataset_key], tokenize=False, add_generation_prompt=True)})#.select(range(2048))
ds1 = ds1.map(get_final_prompt,  batched=False)#.select(range(100))


with ThreadPoolExecutor(max_workers=script_args.max_workers) as executor:
    result = [executor.submit(query_model, ds1[i]["final_prompt"],
                              default_args, ports[i % len(ports)]) for i in range(len(ds1))]
    # use tqdm to show progress
    for _ in tqdm(as_completed(result), total=len(result)):
        pass

    responses = [r.result() for r in result]
import copy

gathered_data = []
for i in range(len(ds1)):
    #tmp_data = {"prompt": ds1[script_args.dataset_key]
    #            [i], "responses": responses[i]}
    tmp_data = ds1[i]
    principle = tmp_data['principle']
    del tmp_data['principle']
    del tmp_data['final_prompt']
    new_completion = copy.deepcopy(tmp_data['completions'])
    new_completion.append({"model": script_args.used_model_name,
                          "response": responses[i], 'principle': principle})
    tmp_data['completions'] = new_completion
    gathered_data.append(tmp_data)

for sample in ds2:
    gathered_data.append(sample)

output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = gathered_data
print("I collect ", len(gathered_data), "samples")

with open(output_dir, 'w', encoding='utf8') as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)
####
