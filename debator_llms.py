# python debator_llms.py ./res/opinions/g29i.csv ./g29i.csv google/gemma-2-9b-it 1 0.3 0

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
from utils import Vase_opinions_loader, save_res, predict_with_opinions_batch, evaluate

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
    "RunID":            sys.argv[6],
    "dataset_path":     sys.argv[1],
	"results_path":     sys.argv[2],    #./res.csv
    "model_name":       sys.argv[3],    #'meta-llama/Meta-Llama-3.1-8B',
    "add_sentence":     sys.argv[4],
    "model_temp":       float(sys.argv[5]),
    "max_new_toks":     50,
})

HUGGING_FACE_TOKEN = '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'
WB_TOKEN = '8eb7abb25a804ca9caaa71178f6ddfdc2f856866'

print('PARAMS:', PARAMS)

print('Logging into Hugging Face and WandB ...')
hf.login(token = HUGGING_FACE_TOKEN)
# wandb.login(key = WB_TOKEN)

print("Loading the dataset ...")
dataset_test = Vase_opinions_loader(PARAMS['dataset_path'])
dataset = dataset_test

# Load model
print("Loading The Model and Tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(PARAMS['model_name'],
                                          padding_side="left")

model = AutoModelForCausalLM.from_pretrained(PARAMS['model_name'],
                                             device_map="auto",
                                             torch_dtype= torch.bfloat16,
                                             load_in_4bit=False)


y_pred = predict_with_opinions_batch({'model': model, 'tokenizer': tokenizer}, dataset, add_sentence = PARAMS['add_sentence'])

y_test = [int(i['label']) for i in dataset]
classif_results = evaluate(y_test, y_pred)
print('metrics:', classif_results)

print('Saving the results ...')
save_res(PARAMS, y_test, y_pred, [], dataset, classif_results) 

print('Claening Up ...')
del tokenizer
del pipeline
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()