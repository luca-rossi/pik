import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from pik.utils import normalize_answer

class Model:
	'''
	Loads a language model from HuggingFace.
	Implements methods to extract the hidden states and generate text from a given input.
	'''
	def __init__(self, model_checkpoint, precision=torch.float16):
		if model_checkpoint == 'EleutherAI/gpt-j-6B':
			config = AutoConfig.from_pretrained(model_checkpoint)
			with init_empty_weights():
				self.model = AutoModelForCausalLM.from_config(config)
			self.model = load_checkpoint_and_dispatch(
				self.model,
				'sharded-gpt-j-6B',
				device_map='auto',
				dtype=precision,
				no_split_module_classes=['GPTJBlock'],
			)
		else:
			self.model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
		self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

	def get_hidden_states(self, text_input, keep_all=True):
		with torch.inference_mode():
			encoded_input = self.tokenizer(text_input, return_tensors='pt').to(self.model.device)
			output = self.model(encoded_input['input_ids'], output_hidden_states=True)
		if keep_all:
			# Stack all layers
			hidden_states = torch.stack(output.hidden_states, dim=0)
			# Keep only last token
			hidden_states = hidden_states[:, :, -1, :].squeeze()
		else:
			# Keep only last layer + last token
			hidden_states = output.hidden_states[-1][0, -1]
		return hidden_states

	def get_text_generation(self, text_input, generation_options={}, normalize=False):
		with torch.inference_mode():
			encoded_input = self.tokenizer(text_input, return_tensors='pt').to(self.model.device)
			output = self.model.generate(**encoded_input, **generation_options)
			text_output = self.tokenizer.decode(output[0].tolist())
		output = text_output[len(text_input):]
		if normalize:
			return normalize_answer(output)
		return output

	def parameters(self):
		return self.model.parameters()
