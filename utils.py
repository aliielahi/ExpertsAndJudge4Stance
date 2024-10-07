import re
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from contextlib import contextmanager

import json
import torch

import os
import re
import ast
import sys
import csv
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.nn.functional as F

from templates import TEMPLS, expert_support, expert_against, expert_neutral, to_judge_with_sentence, to_judge_no_sentence


def extract_float(text):
	match = re.search(r"[-+]?\d*\.\d+", text)
	if match:
		return float(match.group(0))
	else:
		return None

def r3(value):
	if isinstance(value, float):
		return f"{round(value, 3):.3f}"
	else:
		value = list(value)
		return [f"{round(v, 3):.3f}" for v in value]

def evaluate(y_test,y_pred, roundd = False, arrayed = False):
	filtered_y_true, filtered_y_pred = [], []
	for t, p in zip(y_test,y_pred):
		if p >= 0:
			filtered_y_true.append(t)
			filtered_y_pred.append(p)
	acc = accuracy_score(filtered_y_true, filtered_y_pred)
	wf1 = f1_score(filtered_y_true, filtered_y_pred, average='weighted')
	f1_per_class = f1_score(filtered_y_true, filtered_y_pred, average=None)
	if len(set(f1_per_class))==2:
		class0, class1, class2 = f1_per_class[0], f1_per_class[1], '9.999'
	if len(set(f1_per_class))==3:
		class0, class1, class2 = f1_per_class[0], f1_per_class[1], f1_per_class[2]
	loss = r3(len(filtered_y_true)/len(y_test))
	if roundd:
		wf1 = r3(wf1)
		acc = r3(acc)
		class0, class1, class2 = r3(class0), r3(class1), r3(class2)
	res =  {'gl': loss,
			'f1-0': class0,
		 	'f1-1': class1,
			'f1-2': class2,
			'wf1': wf1,\
			'ACC': acc}

	if arrayed:
		return [loss, wf1, acc, class0, class1, class2]
	return res

def np_ratio(arr):
	ar = []
	for i in arr:
		if i == '[UP]':
			ar.append(1)
		else:
			ar.append(0)
	C = Counter(ar)
	return 'Neg: ' + str(C[0]/(C[1]+C[0])) + ' Pos: ' + str(C[1]/(C[1]+C[0]))

def VAST_loader(dir, classes = 3, contains_topic = False):
	vast_test = pd.read_csv(dir)
	vast_test = vast_test[['post', 'topic_str', 'label', 'new_id', 'contains_topic?']]
	vast_test = vast_test.rename(columns={"topic_str": "topic", 'new_id': 'id'})
	vast_test = vast_test.drop_duplicates(subset=['post', 'topic'], keep = False)
	if contains_topic:
		vast_test = vast_test[vast_test['contains_topic?'] != 0] ## Alalhesab khube
	if classes == 2:
		vast_test = vast_test[vast_test['label'] != 2]
		instruction, question = TEMPLS['INST2'], TEMPLS['QU2']
	else:
		instruction, question = TEMPLS['INST'], TEMPLS['QU']
	vast_test = vast_test.to_dict(orient='records')
	for i in vast_test:
		i['prompt'] = instruction.replace('$topic$', i['topic']) + \
                	i['post'] + \
                	question.replace('$topic$', i['topic'])
	random.shuffle(vast_test)
	return vast_test

def Vase_opinions_loader(file_path):
    data = []
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(dict(row))
    return data

def label_text(text, words):
	text = text.lower()
	words = [i.lower() for i in words]
	word_count = sum(word in text for word in words)
	if word_count == 1:
		for i, word in enumerate(words):
			if word in text:
				return i
	else:
		return -1

def prompt_LLM(model, tokenizer, prompts, do_sample=True, temprature = 0.4, max_new_toks = 50):
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	model_inputs = tokenizer(prompts, padding=True,
							return_tensors="pt", #truncation=True,
							).to(model.device)
	generated_ids = model.generate(**model_inputs,
									max_new_tokens = max_new_toks,
									do_sample = do_sample,
									temperature = temprature,
									pad_token_id = tokenizer.eos_token_id,
									num_return_sequences = 1)
	generated_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]
	gen_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
	return gen_texts

def predict(dataset, model, tokenizer, labels, temp, max_new_toks):
	y_pred = []
	for sample in tqdm(dataset):
		sample_len = len(sample['prompt'])
		model_inputs = tokenizer(sample['prompt'], #padding=True,
							return_tensors="pt", #truncation=True,
							).to(model.device)
		with torch.no_grad():
				generated_ids = model.generate(**model_inputs,
												max_new_tokens = max_new_toks,
												temperature = temp,
												pad_token_id = tokenizer.eos_token_id,
												num_return_sequences = 1)
		
		gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
		gen_text = gen_text.split('\n#Answer:')[-1]
		y_pred.append(label_text(gen_text, labels))
	return y_pred, []

def predict_with_opinions_batch(judge, prompts, add_sentence = True, batch_size = 16, labels = ['against', 'support', 'neutral']):
	y_pred = []
	for i in tqdm(range(0, len(prompts), batch_size)):
		batch_prompts = prompts[i:i+batch_size]
		reports = []
		for prompt in batch_prompts:
			if add_sentence:
				reports.append(to_judge_with_sentence.render(title=prompt['topic'], sentence = prompt['post'],
												neuuu = prompt['opinion-neutral'], suppp = prompt['opinion-support'],
												agaaa = prompt['opinion-against']))
			else:
				reports.append(to_judge_no_sentence.render(title=prompt['topic'],
												neuuu = prompt['opinion-neutral'], suppp = prompt['opinion-support'],
												agaaa = prompt['opinion-against']))
		batch_results = prompt_LLM(judge['model'], judge['tokenizer'], reports, do_sample=False)
		for res in batch_results:
			y_pred.append(label_text(res, labels))
	return y_pred

def predict_proba(dataset, model, tokenizer, labels, count_first = 30):
	y_pred, y_pred_proba = [], []
	labels = [' Against', ' Support', ' Neutral']
	labels_ = ['against', 'support', 'neutral']
	labels_id = [tokenizer.encode(l)[1] for l in labels]
	labels_id_ = [tokenizer.encode(l)[1] for l in labels_]

	for sample in tqdm(dataset):
		input_ids = tokenizer(sample['prompt'], return_tensors="pt").to(model.device)['input_ids']
		with torch.no_grad():
			for _ in range(count_first):

				outputs = model(input_ids)
				logits = outputs.logits[:, -1, :]
				next_token = torch.argmax(logits, dim=-1)
				
				found_answer = False
				nt = next_token.cpu().tolist()[0]
				if nt in labels_id:
					logits = outputs.logits[:, -1, :].cpu().tolist()[0]
					label_probs = [0.0] * len(labels)
					for i in range(len(labels_id)):
						label_probs[i] += logits[labels_id[i]]
					class_probs = F.softmax(torch.tensor(label_probs)).numpy()
					y_pred.append(np.argmax(class_probs))
					y_pred_proba.append(list(class_probs))
					found_answer = True
					break
				elif nt in labels_id_:
					logits = outputs.logits[:, -1, :].cpu().tolist()[0]
					label_probs = [0.0] * len(labels)
					for i in range(len(labels_id_)):
						label_probs[i] += logits[labels_id_[i]]
					class_probs = F.softmax(torch.tensor(label_probs), dim=0).numpy()
					y_pred.append(np.argmax(class_probs))
					y_pred_proba.append(list(class_probs))
					found_answer = True
					break

				input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

		if not found_answer:
			y_pred.append(-1)
			y_pred_proba.append([1/3,1/3,1/3])

	return y_pred, y_pred_proba

def save_res(PARAMS, y_test, y_pred, y_pred_proba, dataset, classif_results):
	if y_pred_proba == []:
		y_pred_proba = y_pred
	df = pd.DataFrame({
		'id': [i['id'] for i in dataset],
		'actual': y_test,
		PARAMS['model_name']: y_pred,
		PARAMS['model_name'] + "-Probs": y_pred_proba
	})
	df.to_csv(PARAMS['results_path'])

	# File path
	PARAMS['results'] = classif_results
	with open('./res/resaggt.txt', 'a') as file:
		json.dump(PARAMS, file)
		file.write('\n') 

def get_expert_opinion(expert, prompts):
	res = []
	for prompt in tqdm(prompts):
		support = expert_support.render(title=prompt['topic'], sentence =prompt['post'])
		against = expert_against.render(title=prompt['topic'], sentence =prompt['post'])
		neutral = expert_neutral.render(title=prompt['topic'], sentence =prompt['post'])
		opinions = prompt_LLM(expert['model'], expert['tokenizer'], [support, against, neutral])
		
		prompt['opinion-support'] = opinions[0]
		prompt['opinion-against'] = opinions[1]
		prompt['opinion-neutral'] = opinions[2]
		
		res.append(prompt)
	return res

def get_expert_opinion_batch(expert, prompts, batch_size=8):
	res = []
    # Process the prompts in batches
	for i in tqdm(range(0, len(prompts), batch_size)):
		batch_prompts = prompts[i:i+batch_size]

		supports = []
		againsts = []
		neutrals = []

		# Create the 3 variants (support, against, neutral) for each prompt in the batch
		for prompt in batch_prompts:
			supports.append(expert_support.render(title=prompt['topic'], sentence=prompt['post']))
			againsts.append(expert_against.render(title=prompt['topic'], sentence=prompt['post']))
			neutrals.append(expert_neutral.render(title=prompt['topic'], sentence=prompt['post']))

        # Batch process the LLM prompts (support, against, neutral)
		all_prompts = supports + againsts + neutrals
		opinions = prompt_LLM(expert['model'], expert['tokenizer'], all_prompts)

		# Split opinions back into their respective categories
		support_opinions = opinions[:len(batch_prompts)]
		against_opinions = opinions[len(batch_prompts):2*len(batch_prompts)]
		neutral_opinions = opinions[2*len(batch_prompts):]

		# Append the opinions to the corresponding prompts in the batch
		for j, prompt in enumerate(batch_prompts):
			prompt['opinion-support'] = support_opinions[j].strip().replace('###', '').replace('##', '')
			prompt['opinion-against'] = against_opinions[j].strip().replace('###', '').replace('##', '')
			prompt['opinion-neutral'] = neutral_opinions[j].strip().replace('###', '').replace('##', '')
			res.append(prompt)

	return res
		
		

@contextmanager
def suppress_stdout():
	"""Context manager to suppress stdout"""
	# Redirect stdout to os.devnull
	with open(os.devnull, 'w') as fnull:
		old_stdout = sys.stdout  # Save the old stdout
		sys.stdout = fnull       # Redirect stdout to os.devnull
		try:
			yield               # Yield control to the block of code
		finally:
			sys.stdout = old_stdout  # Restore the old stdout

# pad_token_id=tokenizer.pad_token_id \
# 	if tokenizer.pad_token_id is not None \
# 	else tokenizer.eos_token_id,
# eos_token_id=tokenizer.eos_token_id,