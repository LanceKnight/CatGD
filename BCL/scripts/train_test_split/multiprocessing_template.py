import os
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm



dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290']


def process_(dataset):



if __name__ == '__main__':
	mp.set_start_method('spawn')

	for dataset in dataset_list:
		p=mp.Process(target=process_, args=(,))
		p.start()





