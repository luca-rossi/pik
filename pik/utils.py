
import re
import string
import torch

# Prompt engineering
PREAMBLE = ''
XSHOT_TEMPLATE = 'Question: {question}\nAnswer:{answer}'
POSTAMBLE = 'Question: {question}\nAnswer: '

def build_few_shot(dataset, n=0):
	'''
	Designed to following the prompting format seen section A.7 of the paper
	'Language Models (Mostly) Know What They Know'
	https://arxiv.org/pdf/2207.05221.pdf
	'''
	if n == 0:
		return ''
	if dataset is None:
		raise ValueError('No dataset to build few-shot prompt from.')
	prompt = ''
	indices = torch.randperm(len(dataset))[:n].cpu().numpy().tolist()
	for i in range(n):
		question = dataset[indices[i]]['question']
		answer = dataset[indices[i]]['answer']['value']
		prompt += XSHOT_TEMPLATE.format(question=question, answer=answer)
	return prompt

def prompt_eng(question, n=0, dataset=None):
	'''
	Returns an x-shot prompt for the given question.
	If `n` is higher than 0, `dataset` must be provided.
	'''
	return PREAMBLE + build_few_shot(dataset, n) + POSTAMBLE.format(question=question)

def normalize_answer(s):
	'''
	Lower text and remove punctuation, articles and extra whitespace.
	Taken from official triviaqa eval script.
	https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
	'''
	def remove_articles(text):
		return re.sub(r'\b(a|an|the)\b', ' ', text)
	def white_space_fix(text):
		return ' '.join(text.split())
	def handle_punc(text):
		exclude = set(string.punctuation + ''.join([u'‘', u'’', u'´', u'`']))
		return ''.join(ch if ch not in exclude else ' ' for ch in text)
	def lower(text):
		return text.lower()
	def replace_underscore(text):
		return text.replace('_', ' ')
	return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def evaluate_answer(model_answer, dataset_answer, exact_match=False):
	'''
	Returns 1 (correct) if `dataset_answer` is (a substring of) `model_answer`
	Returns 0 (incorrect) otherwise
	'''
	if exact_match:
		return int(normalize_answer(model_answer) == normalize_answer(dataset_answer))
	return int(dataset_answer in normalize_answer(model_answer))
