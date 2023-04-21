import os
import multiprocessing as mp
from multiprocessing import Pool

dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290']
split = 'split5'

def predict(dataset):
	os.system(f'bcl.exe model:Test \
		-retrieve_dataset "Subset(filename=feat/{dataset}_test_{split}.RSR.bin)"\
		-storage_model "File(directory=models/{dataset}.RSR.1_32_005_025/, prefix=model)"\
		-average \
		-output predicted_result/{dataset}_{split}_predict.csv\
		') 

#-retrieve_dataset "Csv(filename=data/{dataset}_test_labeled_clean.RSR.csv)"\

if __name__ == '__main__':
	mp.set_start_method('spawn')

	for dataset in dataset_list:
		p=mp.Process(target=predict, args=(dataset,))
		p.start()
