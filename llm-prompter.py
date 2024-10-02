# python llm-prompter.py ./vast_test.csv ./res.csv meta-llama/Meta-Llama-3.1-8B 0.7
# print("python llm-prompter.py data_path res_path model temprature")
print('Loading Libraries ...')

list_of_models = ['meta-llama/Meta-Llama-3.1-8B',
                  'meta-llama/Meta-Llama-3.1-8B-Instruct',
                  'meta-llama/Meta-Llama-3-8B',
                  'meta-llama/Meta-Llama-3-8B-Instruct',
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
                  ]

import gc
import sys
import json
import wandb
import pandas as pd
from tqdm import tqdm
import huggingface_hub as hf
from utils import evaluate, VAST_loader, save_res, predict_proba, predict

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
	"classes":          int(sys.argv[1]),    #'./vast_test.csv',
    "dataset_path":     sys.argv[2],
	"results_path":     sys.argv[3],    #./res.csv
    "model_name":       sys.argv[4],    #'meta-llama/Meta-Llama-3.1-8B',
    "model_temp":       float(sys.argv[5]),
    "labels":           ['against', 'support', 'neutral'][:int(sys.argv[1])], # , 'neutral' label: stance label, 0=con, 1=pro, 2=neutral
    "Fine-Tuned_Model": None,
    "max_new_toks":     20,
    "predict_Proba":    True,
    "contains_topic":   False,
	})

HUGGING_FACE_TOKEN = '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'
WB_TOKEN = '8eb7abb25a804ca9caaa71178f6ddfdc2f856866'

print('PARAMS:', PARAMS)

print('Logging into Hugging Face and WandB ...')
hf.login(token = HUGGING_FACE_TOKEN)
# wandb.login(key = WB_TOKEN)

print("Loading the dataset ...")
dataset_test = VAST_loader(PARAMS['dataset_path'],
                            classes = PARAMS['classes'],
                            contains_topic = PARAMS['contains_topic'])
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

if PARAMS['predict_Proba']:
    y_pred, y_pred_proba = predict_proba(dataset, model, tokenizer, PARAMS['labels'], PARAMS['max_new_toks'])
else:
    y_pred, y_pred_proba = predict(dataset, model, tokenizer, PARAMS['labels'], PARAMS['model_temp'],PARAMS['max_new_toks'])



y_test = [i['label'] for i in dataset]
classif_results = evaluate(y_test, y_pred)
print('metrics:', classif_results)

print('Saving the results ...')
save_res(PARAMS, y_test, y_pred, y_pred_proba, dataset, classif_results) 

print('Claening Up ...')
del tokenizer
del pipeline
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()