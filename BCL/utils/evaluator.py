import pandas as pd
from evaluation import calculate_logAUC


dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290']
split= 'split5'

def eval(dataset):
	# df = pd.read_csv(f'last_test_sample_scores.log', header=None)
	df = pd.read_csv(f'predicted_result/{dataset}_{split}_predict.csv', header=None)
	# print(df)

	# Without ids
	pred = df.iloc[:,0].tolist()
	true = df.iloc[:,1].tolist()

	# With ids
	pred = df.iloc[:,1].tolist()
	true = df.iloc[:,2].tolist()


	logAUC_0001_01=calculate_logAUC(true, pred)
	print(f'{dataset}-logAUC_0.001_0.1: {logAUC_0001_01}')

for dataset in dataset_list:
	eval(dataset)