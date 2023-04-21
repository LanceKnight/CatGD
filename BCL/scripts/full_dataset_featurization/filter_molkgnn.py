import os
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm



dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290']
# dataset_list = ['435034']
dataset_list = ['999']
activity_list = ['actives', 'inactives']

def filter_data(dataset):
	for set_type in ['train', 'test']:
		os.system(f'bcl.exe molecule:Filter\
			-input_filenames sdfs/{dataset}_act_and_inact_sdf.sdf\
			-output_matched sdfs/cleaned_{dataset}.sdf\
			-output_unmatched sdfs/unmatched_{dataset}.sdf\
			-add_h\
			-neutralize\
			-defined_atom_types -simple\
			')


if __name__ == '__main__':
	mp.set_start_method('spawn')

	for dataset in dataset_list:
		p=mp.Process(target=filter_data, args=(dataset,))
		p.start()
