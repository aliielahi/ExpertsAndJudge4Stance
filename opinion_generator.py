# python debator_llms.py ./vast_test.csv ./g29i.csv google/gemma-2-9b-it 0.7 0

print('Loading Libraries ...')

list_of_models = ['meta-llama/Meta-Llama-3.1-8B',
                  'meta-llama/Meta-Llama-3.1-8B-Instruct',
                  'meta-llama/Meta-Llama-3-8B',
                  'meta-llama/Meta-Llama-3-8B-Instruct',
                  'meta-llama/Meta-Llama-3.1-70B',
                  'meta-llama/Meta-Llama-3.1-70B-Instruct',
                  'meta-llama/Meta-Llama-3-70B',
                  'meta-llama/Meta-Llama-3-70B-Instruct',
                  'mistralai/Mistral-7B-v0.3',
                  'mistralai/Mistral-7B-Instruct-v0.3',
                  'mistralai/Mixtral-8x7B-v0.1',
                  'mistralai/Mixtral-8x7B-Instruct-v0.1',
                  'microsoft/Phi-3-small-8k-instruct',
                  'microsoft/Phi-3-medium-4k-instruct',
                  'google/gemma-2-2b',
                  'google/gemma-2-2b-it',
                  'google/gemma-2-9b',
                  'google/gemma-2-9b-it',
                  'google/gemma-2-27b-it'
                  ]

import gc
import sys
import csv
import json
import wandb
import pandas as pd
from tqdm import tqdm
import huggingface_hub as hf
from utils import VAST_loader, get_expert_opinion, get_expert_opinion_batch

import torch
import transformers
from datasets import Dataset
from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer, setup_chat_format
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, TrainingArguments, pipeline
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training

PARAMS = dict({
    "RunID":            sys.argv[5],
    "dataset_path":     sys.argv[1],
	"results_path":     sys.argv[2],    #./res.csv
    "model_name":       sys.argv[3],    #'meta-llama/Meta-Llama-3.1-8B',
    "model_temp":       float(sys.argv[4]),
    "max_new_toks":     50,
})

HUGGING_FACE_TOKEN = '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'
WB_TOKEN = '8eb7abb25a804ca9caaa71178f6ddfdc2f856866'

print('PARAMS:', PARAMS)

print('Logging into Hugging Face and WandB ...')
hf.login(token = HUGGING_FACE_TOKEN)
# wandb.login(key = WB_TOKEN)

print("Loading the dataset ...")
dataset_test = VAST_loader(PARAMS['dataset_path'])
d_dataset_test = Dataset.from_dict({"topic": [d["topic"] for d in dataset_test],
                   "post": [d["post"] for d in dataset_test]})
dataset = dataset_test

# Load model
print("Loading The Model and Tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(PARAMS['model_name'],
                                          padding_side="left")

model = AutoModelForCausalLM.from_pretrained(PARAMS['model_name'],
                                             device_map="auto",
                                             torch_dtype= torch.bfloat16,
                                             load_in_4bit=False)


updated_dataset = get_expert_opinion_batch({'model': model, 'tokenizer': tokenizer}, dataset)


print('sample results:', updated_dataset[:2])

print('Saving the results ...')
with open(PARAMS['results_path'], mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=updated_dataset[0].keys())
    writer.writeheader()  # Write the header
    writer.writerows(updated_dataset)

print('Claening Up ...')
del tokenizer
del pipeline
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()