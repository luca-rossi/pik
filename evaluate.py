'''
Evaluate a model on the TriviaQA validation set (zero-shot).
'''
import argparse
import torch
from tqdm import trange
from pik.datasets.triviaqa_dataset import TriviaQADataset
from pik.models.model import Model
from pik.utils import evaluate_answer, prompt_eng

# Set params
parser = argparse.ArgumentParser()
parser.add_argument('--n_questions', type=int, default=1000, help='number of q-a pairs to evaluate, index of selected questions is saved in `data_ids`; if <= 0, all questions are used')
parser.add_argument('--dataset_seed', type=int, default=420, help='seed for selecting questions from dataset')
parser.add_argument('--model_checkpoint', '-m', default='EleutherAI/gpt-j-6B', help='model checkpoint to use')
parser.add_argument('--precision', '-p', default='float16', help='model precision')
parser.add_argument('--n_answers_per_question', type=int, default=10, help='number of answers to generate per question')
parser.add_argument('--max_new_tokens', type=int, default=16, help='maximum number of tokens to generate per answer')
parser.add_argument('--temperature', type=float, default=1, help='temperature for generation')
parser.add_argument('--pad_token_id', type=int, default=50256, help='pad token id for generation')
parser.add_argument('--verbose', '-v', action='store_true', default=False, help='print more info')
args = parser.parse_args()
args.precision = torch.float16 if args.precision == 'float16' else torch.float32
torch.set_default_dtype(args.precision)
generation_options = {
	'pad_token_id': args.pad_token_id,
	'max_new_tokens': args.max_new_tokens,
    'do_sample': True,
    'temperature': args.temperature,
}

# Load dataset and model
data = TriviaQADataset()
torch.manual_seed(args.dataset_seed)
data_ids = torch.randperm(len(data))
if args.n_questions > 0:
	data_ids = data_ids[:args.n_questions]
data_ids = data_ids.cpu().numpy().tolist()
model = Model(args.model_checkpoint, precision=args.precision)

# Loop through each question-answer pair in the validation set
tot = 0
correct = 0
progress_bar = trange(len(data_ids), position=0, leave=True)
for i in progress_bar:
	# Get the question and the correct answer
	hid, qid = i, data_ids[i]
	question, correct_answer = data[qid]
	desc = ''
	if args.verbose:
		desc += f'Question: {question}\n'
		desc += f'Correct answer: {correct_answer}\n'
	text_input = prompt_eng(question)
	# Generate multiple model answers
	for n in range(args.n_answers_per_question):
		model_answer = model.get_text_generation(text_input, generation_options=generation_options)
		eval = evaluate_answer(model_answer, correct_answer)
		if eval:
			if args.verbose:
				desc += f'Correct! Model answer: {model_answer}\n'
			correct += 1
			break
	tot += 1
	desc += f'Question {i+1}/{len(data_ids)} - Accuracy: {correct/tot}'
	if args.verbose:
		desc += '\n\n'
		desc += '=' * 80
		desc += '\n'
	progress_bar.set_description(desc)
