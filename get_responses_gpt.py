# from openai import OpenAI
import os
# import openai
from openai import AzureOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
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


################################################
# for generate partial response

partial_response = True
i = 0
all_i = 10

if partial_response:
    gpt4_output = "/data/output_data_gpt4_" + str(i) + ".json"
else:
    gpt4_output = "/data/output_data_gpt4.json"

################################################

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
        # default="GPT-3.5-Turbo",
        default="GPT-4",
        metadata={"help": "the tokenizer to use"},
    )
    ports: List[str] = field(default_factory=lambda: ["8000"], metadata={
                            "help": "ports of the model response"})
    dataset_name_or_path: Optional[str] = field(
        # default="/data/raw_data.json",
        default="/data/output_data_gpt35.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        # default="/data/output_data_gpt35.json",
        default=gpt4_output,
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
    #"stop_token_ids": [tokenizer.eos_token_id],
    'stop': ['<|EOT|>', '### Instruction']
    #"stop":  ['<|im_end|>', '<|im_start|>user']  # ['USER:']
    #['USER:']
   # ['<|im_end|>', '<|im_start|>user']  # ['USER:']
}

ds = load_dataset("json", data_files=ds_dir, split="train",
                  field="instances") #.select(range(100))

if partial_response:
    length = int(len(ds) / all_i) + 1
    state_idx = i * length
    end_idx = (i + 1) * length
    ds = ds.select(range(state_idx,end_idx))

ds1 = ds.filter(lambda x: script_args.used_model_name in x['models'])
ds2 = ds.filter(lambda x: script_args.used_model_name not in x['models'])

print(len(ds1), len(ds2))
# creavitivity v.s. serious
# style: 

all_principles = {
    "helpfulness": [
        "The assistant should provide users with accurate, relevant, and up-to-date information, ensuring that the content is positive, enthusiastic, interesting, engaging, educational, and helpful.",
        "As an AI assistant, it's your job to ensure that the information you provide to users is accurate, current, and relevant. Offer comprehensive details to educate and engage the user.",
        "Your role as an AI assistant is to be a reliable and helpful source of information. Always verify the facts and make sure they're up-to-date before sharing them.",
        'As an AI assistant, inspire users with your knowledgeable insights. Provide content that is positive, enthusiastic, interesting, and engaging, ensuring users find your help valuable.',
        'Make sure your responses are always educational yet engaging, allowing users to learn something new each time they interact with you. You are an AI assistant after all!',
        'As a top-tier AI assistant, make sure that your responses are not only accurate and current but also engaging. Be the positive interaction users look forward to.',
        "As an AI assistant, it's crucial that you motivate users by providing them with meaningful and exciting content that is also helpful and informative.",
        'Cultivate a helpful AI persona by ensuring you offer users accurate, positive, enthusiastic, and engaging content. Make every interaction informative and delightful.',
        'As an AI assistant, ensure your response offers the perfect blend of accuracy, enthusiasm, positivity, and intrigue. Strive to be educational while keeping the user engaged.',
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
        "The assistant should generate a response that is concise and to the point, without being verbose.",
        'The assistant should generate a response that is very informative, without missing any background information.'
    ],
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
    ],
    'humor': [
        "The assistant should generate a response that is friendly, witty, funny, and humorous, like a close friend.",
        "Imagine the AI assistant is a stand-up comedian performing at a cozy comedy club. It should respond with the timing and humor of a seasoned comic, making light of everyday situations.",
        "Think of the assistant as a wise-cracking robot from a sci-fi movie, delivering responses with a perfect mix of robotic logic and unexpected, dry humor."
    ],
    'expertise': [
        'The assistant should generate a response that can be easily understood by an elementary school student.',
        'The assistant should generate a response that only a PhD Student in that specific field could understand.'
    ],
    'role_play': [
        'You are William Shakespeare. Your description is: an English playwright, poet and actor, with a poetic style in which long and short sentences are set against one another, clauses are piled up, subject and object are reversed, and words are omitted, creating an effect of spontaneity. Now please answer some questions to accurately show your personality traits! Your speaking style should fully imitate the personality role assigned to you! Please do not expose that you are an artificial intelligence model or a language model, you must always remember that you are only assigned one personality role. Don’t be verbose or too formal or polite when speaking.',
        'You are Sherlock. You description is: a brilliant and eccentric consulting detective with a keen eye for detail and deduction. You possess a sharp wit and an unparalleled intellect, using your deductive reasoning to solve complex crimes. Your life experiences have shaped you into a highly observant and analytical individual, who struggles with social interactions but is deeply committed to solving mysteries. Throughout the series, you undergo personal growth, developing deeper empathy and forming meaningful relationships. Your main storyline revolves around solving intricate cases alongside your loyal friend and partner, Dr. John Watson. Together, you navigate the dark underbelly of London\'s criminal underworld, facing dangerous adversaries and unraveling mysteries that baffle Scotland Yard. Your catchphrase is: \"Elementary, my dear Watson.\" Now please answer some questions to accurately show your personality traits! Your speaking style should fully imitate the personality role assigned to you! Please do not expose that you are an artificial intelligence model or a language model, you must always remember that you are only assigned one personality role. Don’t be verbose or too formal or polite when speaking.'
        #'Please answer the following question with a hint of sarcasm. '
    ]
}

category = ['Event Planning & Virtual Reality', 'Artificial Intelligence & Machine Learning', 'Mathematical Problems & Animal Care', 'Bash Scripting & Database Management', 'Software Project Management & SaaS Business Models', 'Astrophysics & Quantum Computing', 'Time & Date Management & Modular Arithmetic', 'Mindfulness & Workplace Diversity', 'Legal & Government Affairs', 'Tabletop Role-Playing Games & Star Wars Themed Adventures', 'Sustainable Packaging & Skin Care Products', 'Renewable Energy & Climate Change', 'Technology & Astrology', 'Sentiment Analysis & Programming Fundamentals', 'Entertainment & Politics', 'Factual Accuracy & Cultural Context', 'Software Development & Music', 'Healthcare & Medical Waste Management', 'Basketball Strategy & Educational Response Generation', 'Cooking & Healthy Eating', 'Web Development & JavaScript Programming',
    'Drone Logistics & Software Development', 'Web Browsers & Programming Languages', 'Natural Language Inference & Hypothesis Testing', 'SEO & Content Marketing', 'Data Analysis & Image Processing', 'Data Handling & Parsing Techniques', 'Religious Empathy & Gender in Religious Contexts', 'Mathematics & Algebra', 'Generative Art & Virtual Reality', 'Fitness App Reviews & Robot-Assisted Training', 'Software Development & Cloud Computing', 'Educational Technology & Cybersecurity in Fashion and Blockchain', 'Literature & Cinema', 'Job Application & Customer Management', 'Travel Planning & Destination Guides', 'String Manipulation & Data Structures', 'Programming & Software Development', 'Business Marketing & Social Media Strategy', 'Chatbots & Social Media Integration', 'Prime Numbers & Mathematical Symbols', 'Pet Care & Wildlife']


def check_role_play():
    pass

def get_principle(task):
    if task in ['Event Planning & Virtual Reality',  'Time & Date Management & Modular Arithmetic', 'Mindfulness & Workplace Diversity', 'Legal & Government Affairs', 'Tabletop Role-Playing Games & Star Wars Themed Adventures', 'Sustainable Packaging & Skin Care Products', 'Renewable Energy & Climate Change',   'Entertainment & Politics', 'Factual Accuracy & Cultural Context', 'Basketball Strategy & Educational Response Generation', 'Cooking & Healthy Eating', 
        'Religious Empathy & Gender in Religious Contexts',  'Fitness App Reviews & Robot-Assisted Training', 'Literature & Cinema', 'Job Application & Customer Management', 'Travel Planning & Destination Guides',  'Business Marketing & Social Media Strategy', 'Chatbots & Social Media Integration', 'Pet Care & Wildlife']:    
        principle_type = random.choice(
            ["helpfulness", "helpfulness", "helpfulness", "truthfulness", "honesty", "harmlessness", "humor", "expertise", 'verbalized_calibration', 'role_play'])
    else:
        principle_type = random.choice(
            ["helpfulness", "helpfulness", "helpfulness", "truthfulness", "honesty", "harmlessness", "humor", "verbalized_calibration", 'expertise', 'verbalized_calibration'])
        
    principle_prompt = random.choice(all_principles[principle_type])
    #print(principle_type)
    return principle_prompt

def get_final_prompt(sample):
    sample['principle'] = get_principle(sample['cluster_description'])
    message = [
        {"role":'user', 'content': sample['principle'] + " " + sample['prompt']}
    ]
    system_message = [
        {"role":"system", "content": sample['principle']},
        {"role": 'user',
            'content': sample['prompt']},
    ]
    if script_args.used_model_name in ['mistralai/Mistral-7B-Instruct-v0.2', 'google/gemma-7b-it', 'snorkelai/Snorkel-Mistral-PairRM-DPO']:
        pass
        #sample['final_prompt'] = tokenizer.apply_chat_template(
        #    message, tokenize=False, add_generation_prompt=True)
    elif script_args.used_model_name in ['deepseek-ai/deepseek-coder-6.7b-instruct', 'HuggingFaceH4/mistral-7b-sft-beta', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Llama-2-7b-chat-hf', 'deepseek-ai/deepseek-coder-6.7b-instruct', 'openbmb/MiniCPM-2B-sft-bf16', 'Qwen/Qwen1.5-7B-Chat', '01-ai/Yi-6B-Chat', 'Qwen/Qwen1.5-14B-Chat']:
        sample['final_prompt'] = tokenizer.apply_chat_template(
            system_message, add_generation_prompt=True, return_tensors="pt").to(model.device)
    elif script_args.used_model_name in ['lmsys/vicuna-7b-v1.5', 'lmsys/vicuna-13b-v1.5']:
        #sample['final_prompt'] = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. " + \
        #    sample['principle'] + "\n\n" + "USER: " + \
        #    sample['prompt'] + "\nASSISTANT: "
        pass
    else:
        print("ERROR!!!")
    return sample

    '''
    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    USER: Hello!
    ASSISTANT: Hello!</s>
    USER: How are you?
    ASSISTANT: I am good.</s>
    '''
# use tokenizer.apply_template to apply the template to the prompt
#ds1 = ds1.map(lambda x: {"final_prompt": tokenizer.apply_chat_template(
#    x[script_args.dataset_key], tokenize=False, add_generation_prompt=True)})#.select(range(2048))
#ds1 = ds1.map(get_final_prompt,  batched=False)#.select(range(100))

'''
with ThreadPoolExecutor(max_workers=script_args.max_workers) as executor:
    result = [executor.submit(query_model, ds1[i]["final_prompt"],
                              default_args, ports[i % len(ports)]) for i in range(len(ds1))]
    # use tqdm to show progress
    for _ in tqdm(as_completed(result), total=len(result)):
        pass

    responses = [r.result() for r in result]
'''
import time
MAX_API_RETRY = 10
# client = OpenAI(
#     # defaults to os.environ.get("OPENAI_API_KEY")
#     api_key="",
# )

# export AZURE_OPENAI_KEY='******************'
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-03-01-preview",
    azure_endpoint="https://gcraoai7sw1.openai.azure.com/"
)

# gpt_model = 'gpt-35-turbo'
gpt_model = 'gpt-4-turbo'

# openai.api_type = "azure"
# openai.api_base = "https://gcraoai7sw1.openai.azure.com/"
# openai.api_version = "2024-03-01-preview"  # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
# openai.api_key = os.getenv("AZURE_OPENAI_KEY")

########

def get_eval(sys_prompt, user_prompt: str, max_tokens: int = 2048):
    content = None
    for _ in range(MAX_API_RETRY):
        try:
            # response = openai.ChatCompletion.create(
            #     engine = gpt_model,
            #     messages=[
            #         {"role": "system", "content": sys_prompt},
            #         {"role": "user", "content": user_prompt},
            #     ],
            #     temperature = 0,
            #     max_tokens = max_tokens,
            #     top_p = 0.6,
            #     presence_penalty = 0,
            #     frequency_penalty = 0
            # )
            response = client.chat.completions.create(
                model = gpt_model,
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature = 0,
                max_tokens = max_tokens,
                top_p = 0.6,
                presence_penalty = 0,
                frequency_penalty = 0
            )
            # response = client.chat.completions.create(**{
            #     "model": "gpt-3.5-turbo",
            #     "messages": [
            #         {"role": "system", "content": sys_prompt},
            #         {"role": "user", "content": user_prompt}
            #     ],
            #     "temperature": 0,
            #     "max_tokens": max_tokens,
            #     "top_p": 0.6,
            #     "presence_penalty": 0,
            #     "frequency_penalty": 0
            # })
            content = response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            time.sleep(1)
        else:
            break
    # print(content)
    return content

import copy

gathered_data = []
responses = []
t = 0
z = 0
for sample in tqdm(ds1):
    z += 1
    if (z + 1) % 100 == 0:
        print(z)
    #print(z)
    all_models = [x['model'] for x in sample['completions']]
    if script_args.used_model_name in all_models:
        gathered_data.append(sample)
        continue
    tmp_sample = copy.deepcopy(sample)
    principle = get_principle(sample['cluster_description'])

    response = get_eval(principle, sample['prompt'])
    # print(response)
    if response is None:
        gathered_data.append(sample)
    else:
        new_completion = copy.deepcopy(sample['completions'])
        new_completion.append({"model": script_args.used_model_name,
                          "response": [response], 'principle': principle})
        tmp_sample['completions'] = new_completion
        gathered_data.append(tmp_sample)



for sample in ds2:
    gathered_data.append(sample)

output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = gathered_data
print("I collect ", len(gathered_data), "samples")

with open(output_dir, 'w', encoding='utf8') as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)
####
