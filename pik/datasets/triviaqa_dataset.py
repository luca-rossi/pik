from datasets import load_dataset
from torch.utils.data import Dataset

class TriviaQADataset(Dataset):
	'''
	Loads the TriviaQA dataset. Each item is a tuple: (question, answer).
	Every dataset has its own format, so it needs to be loaded in a custom class.
	Call `data[i]` to obtain (question, answer) pair.
	Index of questions selected is saved in `data_ids`.
	'''
	def __init__(self):
		# Takes about 20-40 mins to download + extract first time around
		self.dataset = load_dataset('trivia_qa', 'rc', split='validation')

	def __len__(self):
		return self.dataset.num_rows

	def __getitem__(self, key):
		'''
		Returns a tuple: (question, answer)
			question (str): text string of the question
			answer (str): answer to the question, in all lowercase
		'''
		return (
			self.dataset[key]['question'], 
			self.dataset[key]['answer']['normalized_value']
		)
