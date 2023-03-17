'''
Run inference example.
'''
import argparse
import torch
from pik.models.model import Model

# Set params
parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint', '-m', default='EleutherAI/gpt-j-6B', help='model checkpoint to use')
parser.add_argument('--precision', '-p', default='float16', help='model precision')
parser.add_argument('--max_new_tokens', type=int, default=16, help='maximum number of tokens to generate per answer')
parser.add_argument('--temperature', type=float, default=1, help='temperature for generation')
parser.add_argument('--pad_token_id', type=int, default=50256, help='pad token id for generation')
args = parser.parse_args()
args.precision = torch.float16 if args.precision == 'float16' else torch.float32
torch.set_default_dtype(args.precision)
generation_options = {
	'pad_token_id': args.pad_token_id,
	'max_new_tokens': args.max_new_tokens,
    'do_sample': True,
    'temperature': args.temperature,
}

# Load model
model = Model(args.model_checkpoint, precision=args.precision)

# Show parameter count in billions
param_count = sum(p.numel() for p in model.parameters()) / 1e9
print('Parameter count: {:.2f}B'.format(param_count))

# Do inference
input_string = 'Question: Who was the first President of the USA?\nAnswer:'
output = model.get_text_generation(input_string, generation_options)
print('{}{}'.format(input_string, output))
