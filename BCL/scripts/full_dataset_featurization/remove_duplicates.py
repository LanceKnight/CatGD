import os
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm



dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290']
# dataset_list = ['435034']
dataset_list = ['485290']
activity_list = ['actives', 'inactives']

def remove_duplicates(dataset):
	for set_type in ['train', 'test']:
		os.system(f'bcl.exe molecule:Unique\
			-compare Configurations\
			-input_filenames train_test_sdfs/{dataset}_act_and_inact_sdf.sdf\
			-output train_test_sdfs/{dataset}_act_and_inact_unique.sdf\
			-output_dupes train_test_sdfs/{dataset}_act_and_inact_dupes.sdf\
			')


if __name__ == '__main__':
	mp.set_start_method('spawn')

	for dataset in dataset_list:
		p=mp.Process(target=remove_duplicates, args=(dataset,))
		p.start()
