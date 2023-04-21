import os
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm



dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290']
# dataset_list = ['435034']
split = 'split5'

def randomize(dataset):
	os.system(f'bcl.exe descriptor:GenerateDataset \
		-source "Randomize(Subset(filename=feat/{dataset}_train_{split}.RSR.bin))"\
		-output feat/{dataset}_train_{split}.RSR.rand.bin\
		')


if __name__ == '__main__':
	mp.set_start_method('spawn')

	for dataset in dataset_list:
		p=mp.Process(target=randomize, args=(dataset,))
		p.start()





