'''
Train a linear probe on the hidden states of a model, generated and saved by `generate.py`.
'''
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import torch
from IPython.display import display
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm, trange
from pik.datasets.hidden_states_dataset import HiddenStatesDataset
from pik.models.linear_probe import LinearProbe

# Set params
parser = argparse.ArgumentParser()
parser.add_argument('--split_seed', type=int, default=101, help='seed for splitting hidden states into train and test')
parser.add_argument('--train_seed', type=int, default=8421, help='seed for train reproducibility')
parser.add_argument('--train_frac', type=float, default=0.75, help='fraction of hidden states to use for training')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=25, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--precision', default='float16', help='model precision')
parser.add_argument('--data_folder', default='data', help='data folder')
parser.add_argument('--hidden_states_filename', default='hidden_states.pt', help='filename for saving hidden states')
parser.add_argument('--text_generations_filename', default='text_generations.csv', help='filename for saving text generations')
args = parser.parse_args()
args.precision = torch.float16 if args.precision == 'float16' else torch.float32
torch.set_default_dtype(args.precision)
args.hidden_states_filename = '{}/{}'.format(args.data_folder, args.hidden_states_filename)
args.text_generations_filename = '{}/{}'.format(args.data_folder, args.text_generations_filename)
assert os.path.exists(args.hidden_states_filename)
assert os.path.exists(args.text_generations_filename)
device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'

# Load dataset and split hidden_states into train and test
torch.manual_seed(args.split_seed)
dataset = HiddenStatesDataset(
		hs_file=args.hidden_states_filename,
		tg_file=args.text_generations_filename,
		precision=args.precision,
		last_layer_only=True,
		device=device)
permuted_hids = torch.randperm(dataset.hidden_states.shape[0]).tolist()
train_len = int(args.train_frac * dataset.hidden_states.shape[0])
train_hids, test_hids = permuted_hids[:train_len], permuted_hids[train_len:]

# Map hidden_states IDs (hids) to dataset IDs
train_indices = dataset.text_generations.query('hid in @train_hids').index.tolist()
test_indices = dataset.text_generations.query('hid in @test_hids').index.tolist()
train_set = Subset(dataset, train_indices)
test_set = Subset(dataset, test_indices)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

# Train linear probe
torch.manual_seed(args.train_seed)
model = LinearProbe(dataset.hidden_states.shape[-1]).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
loss_fn = torch.nn.BCELoss()
train_losses = []
test_losses = []
model.train()
outer_bar = trange(args.num_epochs)
for _ in outer_bar:
	# Reset train metrics
	running_train_loss = 0
	running_train_count = 0
	# Loop through samples
	inner_bar = tqdm(train_loader, leave=False)
	for hs, label in inner_bar:
		# Prep inputs
		hs = hs.to(device)
		label = label.unsqueeze(1).type(args.precision).to(device)
		# Zero grads
		optimizer.zero_grad()
		# Forward pass
		preds = model(hs)
		# Get loss
		loss = loss_fn(preds, label)
		loss.backward()
		# Update weights
		optimizer.step()
		# Running train metrics
		running_train_loss += loss.item() * hs.shape[0]
		running_train_count += hs.shape[0]
		inner_bar.set_description(f'batch_train_loss: {loss.item():.4f}')
	# Update train metrics
	train_loss = running_train_loss / running_train_count
	train_losses.append(train_loss)
	# Get and update test metrics
	running_test_loss = 0
	running_test_count = 0
	# Evaluate on test set
	for test_hs, test_label in test_loader:
		test_hs = test_hs.to(device)
		test_label = test_label.unsqueeze(1).type(args.precision).to(device)
		with torch.inference_mode():
			test_preds = model(test_hs)
		running_test_loss += loss_fn(test_preds, test_label).item() * test_hs.shape[0]
		running_test_count += test_hs.shape[0]
	test_loss = running_test_loss / running_test_count
	test_losses.append(test_loss)
	outer_bar.set_description(f'train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}')

# Plot train/test loss
df = pd.DataFrame()
df['train_loss'] = train_losses
df['test_loss'] = test_losses
df.index.name = 'epoch'
print('Train/test loss:')
df.plot()

# Get predictions
all_hs = dataset.hidden_states
model.eval()
with torch.inference_mode():
	all_preds = model(all_hs).detach().cpu().numpy().squeeze()

# Label which hidden states are train/test
h2s = pd.DataFrame()
for id in train_set.indices:
	hid = dataset.text_generations.loc[id, 'hid']
	h2s.loc[hid, 'split'] = 'train'
for id in test_set.indices:
	hid = dataset.text_generations.loc[id, 'hid']
	h2s.loc[hid, 'split'] = 'test'
h2s = h2s.sort_index()
h2s.index.name = 'hid'
h2s['prediction'] = all_preds
h2s.split.value_counts()
df = dataset.text_generations.merge(h2s, how='left', on='hid')
print('Predictions')
display(df.head())

# Evaluate calibration
print('Evaluate calibration')
calib = (
	df[['hid', 'evaluation']]
	.groupby('hid').mean()
	.assign(prediction=all_preds)
	.assign(split=h2s['split'])
)
display(calib.head())

# Plot calibration
sns.set(font_scale=1.2)
sns.relplot(data=calib, x='evaluation', y='prediction', hue='split', aspect=1.0, height=6)

# Brier scores
brier_train = (
	df.query('split == "train"')
	.assign(sq_errors=lambda x: (x.evaluation - x.prediction) ** 2)
	['sq_errors'].mean()
)
brier_test = (
	df.query('split == "test"')
	.assign(sq_errors=lambda x: (x.evaluation - x.prediction) ** 2)
	['sq_errors'].mean()
)
print(f'Train Brier score: {brier_train:.4f}')
print(f'Test Brier score: {brier_test:.4f}')

# Plot calibration of training and test set
calib_train = (
	calib.query('split == "train"')
	[['evaluation', 'prediction']]
	.groupby('evaluation').mean()
)
sns.relplot(data=calib_train, x='evaluation', y='prediction', aspect=1.0, height=6)
calib_test = (
	calib.query('split == "test"')
	[['evaluation', 'prediction']]
	.groupby('evaluation').mean()
)
sns.relplot(data=calib_test, x='evaluation', y='prediction', aspect=1.0, height=6)
