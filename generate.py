'''
Generate and save hidden states dataset.
'''
import argparse
import os
import pandas as pd
import torch
from IPython.display import display
from time import time
from tqdm import trange
from pik.datasets.triviaqa_dataset import TriviaQADataset
from pik.models.model import Model
from pik.utils import prompt_eng, evaluate_answer

# Set params
parser = argparse.ArgumentParser()
parser.add_argument('--n_questions', type=int, default=10, help='number of q-a pairs to generate, index of selected questions is saved in `data_ids`; if <= 0, all questions are used')
parser.add_argument('--dataset_seed', type=int, default=420, help='seed for selecting questions from dataset')
parser.add_argument('--generation_seed', type=int, default=1337, help='seed for generation reproducibility')
parser.add_argument('--model_checkpoint', '-m', default='EleutherAI/gpt-j-6B', help='model checkpoint to use')
parser.add_argument('--precision', '-p', default='float16', help='model precision')
parser.add_argument('--n_answers_per_question', type=int, default=40, help='number of answers to generate per question')
parser.add_argument('--max_new_tokens', type=int, default=16, help='maximum number of tokens to generate per answer')
parser.add_argument('--temperature', type=float, default=1, help='temperature for generation')
parser.add_argument('--pad_token_id', type=int, default=50256, help='pad token id for generation')
parser.add_argument('--keep_all_hidden_layers', action='store_true', default=True, help='set to False to keep only hidden states of the last layer')
parser.add_argument('--save_frequency', type=int, default=99999, help='write results to disk after every `save_frequency` questions')
parser.add_argument('--data_folder', default='data', help='data folder')
parser.add_argument('--hidden_states_filename', default='hidden_states.pt', help='filename for saving hidden states')
parser.add_argument('--text_generations_filename', default='text_generations.csv', help='filename for saving text generations')
parser.add_argument('--qa_pairs_filename', default='qa_pairs.csv', help='filename for saving q-a pairs')
parser.add_argument('--estimate', action='store_true', default=False, help='set to True to estimate time to completion')
parser.add_argument('--n_test', type=int, default=3, help='number of questions to use for estimating time to completion')
args = parser.parse_args()
args.precision = torch.float16 if args.precision == 'float16' else torch.float32
torch.set_default_dtype(args.precision)
args.hidden_states_filename = os.path.join(args.data_folder, args.hidden_states_filename)
args.text_generations_filename = os.path.join(args.data_folder, args.text_generations_filename)
args.qa_pairs_filename = os.path.join(args.data_folder, args.qa_pairs_filename)

generation_options = {
	'max_new_tokens': args.max_new_tokens,
	'temperature': args.temperature,
	'do_sample': True,
	'pad_token_id': args.pad_token_id,
	# 'eos_token_id': 198,	# Stop generating more tokens when the model generates '\n'
	# 'eos_token_id': 13,	# Stop generating more tokens when the model generates '.'
}

# Load dataset and model
data = TriviaQADataset()
torch.manual_seed(args.dataset_seed)
data_ids = torch.randperm(len(data))
if args.n_questions > 0:
	data_ids = data_ids[:args.n_questions]
data_ids = data_ids.cpu().numpy().tolist()
model = Model(args.model_checkpoint, precision=args.precision)

### Start generating
results = pd.DataFrame()
start = time()
all_hidden_states = None
torch.manual_seed(args.generation_seed)
progress_bar = trange(args.n_questions) if not args.estimate else trange(args.n_test)
for i in progress_bar:
	# Prep inputs
	hid, qid = i, data_ids[i]
	question, answer = data[qid]
	progress_bar.set_description(question)
	text_input = prompt_eng(question)
	# Get hidden state
	hidden_state = model.get_hidden_states(text_input, keep_all=args.keep_all_hidden_layers)
	if hid == 0:
		all_hidden_states = hidden_state.unsqueeze(0)
	else:
		all_hidden_states = torch.cat((all_hidden_states, hidden_state.unsqueeze(0)), dim=0)
	# Generate multiple model answers
	for n in range(args.n_answers_per_question):
		model_answer = model.get_text_generation(text_input, generation_options=generation_options)
		eval = evaluate_answer(model_answer, answer)
		# Record results in memory
		df_idx = results.shape[0]
		results.loc[df_idx, 'hid'] = hid
		results.loc[df_idx, 'qid'] = qid
		results.loc[df_idx, 'n'] = n
		results.loc[df_idx, 'model_answer'] = model_answer
		results.loc[df_idx, 'evaluation'] = eval
	# Periodically write results to disk
	if not args.estimate and (hid + 1) % args.save_frequency == 0 and (hid + 1) != args.save_frequency:
		torch.save(all_hidden_states, args.hidden_states_filename)
		for col in results.columns:
			if col != 'model_answer':
				results[col] = results[col].astype(int)
		results.to_csv(args.text_generations_filename, index=False)
		print('-------------')
		print(results)
		print(args.text_generations_filename)

# Estimate time to completion and disk usage if `--estimate` is set, then exit
if args.estimate:
	time_taken = time() - start
	num_floats = args.n_questions
	for dim in all_hidden_states.shape[1:]:
		num_floats *= dim
	bytes_per_float = 2 if args.precision == torch.float16 else 4
	print(f'''n={args.n_test} questions
	{args.n_answers_per_question} generations per question
	{generation_options['max_new_tokens']} new tokens per generation
	-------------------------------------
	Average processing time per question:
	{time_taken / args.n_questions :.2f} seconds

	Estimated duration (give or take) to process all {args.n_questions} questions:
	{(time_taken / args.n_test) * args.n_questions / 3600 :.3f} hours
	
	Estimated disk usage for {args.hidden_states_filename}:
	{(num_floats * bytes_per_float) / 1e6:.3f} MB''')
	exit()

# Generate q-a pairs df
qa_pairs = pd.DataFrame()
for qid in data_ids:
	q, a = data[qid]
	df_idx = qa_pairs.shape[0]
	qa_pairs.loc[df_idx, 'qid'] = qid
	qa_pairs.loc[df_idx, 'question'] = q
	qa_pairs.loc[df_idx, 'answer'] = a
qa_pairs['qid'] = qa_pairs['qid'].astype(int)
display(qa_pairs.head())

# Generate results df
for col in results.columns:
	if col != 'model_answer':
		results[col] = results[col].astype(int)
display(results.head())
print('Mean evaluation score:', results.evaluation.mean())
print('Hidden states shape:', all_hidden_states.shape)

# Write final results to disk
torch.save(all_hidden_states, args.hidden_states_filename)
results.to_csv(args.text_generations_filename, index=False)
qa_pairs.to_csv(args.qa_pairs_filename, index=False)
