import os
from tqdm import tqdm
import torch
import multiprocessing as mp
from multiprocessing import Pool

dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290']
# dataset_list = ['9999']
split_id='split5'


def convert_dataset(dataset):
	input_id_file = f'shrink_{dataset}_seed2_{split_id}.pt'
	input_sdf_dir = f'/home/liuy69/projects/unified_framework/dataset/qsar/clean_sdf/raw/'

	id_dict = torch.load(f'data_split/{input_id_file}')
	train_id = id_dict['train']
	test_id = id_dict['test']
	print(f'num_train:{len(train_id)}')
	print(f'num_test:{len(test_id)}')


	# os.system(f'cat {input_sdf_dir}/{dataset}_actives_new.sdf {input_sdf_dir}/{dataset}_inactives_new.sdf > sdfs/{dataset}.sdf')

	# If sdf does not have id, create id sdf
	counter = 0
	is_first_line = True
	train_copy_content = False
	test_copy_content = False
	# for activity in ['actives', 'inactives']:
	# with open(f'{input_sdf_dir}'+f'{dataset}_{activity}_new.sdf') as in_file:
	with open(f'sdfs/{dataset}_act_and_inact_sdf.sdf') as in_file:
		with open(f'sdfs/id_{dataset}_from_convert_id_script.sdf', 'w+') as out_file:
			with open(f'sdfs/train_test/{dataset}_train_{split_id}.sdf', 'w+') as out_train_sdf_file:
				with open(f'sdfs/train_test/{dataset}_test_{split_id}.sdf', 'w+') as out_test_sdf_file:
					lines = in_file.readlines()
					for line in tqdm(lines):
						if is_first_line == True:
							new_line = f'-{counter}-'+line
							is_first_line == False
							if counter in train_id:
								train_copy_content=True
								# out_train_sdf_file.write(new_line)
							if counter in test_id:
								test_copy_content=True
								# out_test_sdf_file.write(new_line)
						if '$$$$' in line:
							counter +=1
							is_first_line = True
							if train_copy_content==True:
								out_train_sdf_file.write(line)
								train_copy_content=False
							if test_copy_content==True:
								out_test_sdf_file.write(line)
								test_copy_content=False
						else:
							if is_first_line == True:
								if train_copy_content==True:
									out_train_sdf_file.write(new_line)
								if test_copy_content==True:
									out_test_sdf_file.write(new_line)
							else:
								if train_copy_content==True:
									out_train_sdf_file.write(line)
								if test_copy_content==True:
									out_test_sdf_file.write(line)
							is_first_line = False
				
					out_file.write(new_line)

	print(f'dataset{dataset} train:')
	os.system(f"zgrep -c '$$$$' sdfs/train_test/{dataset}_train_{split_id}.sdf")
	print(f'dataset{dataset} test:')
	os.system(f"zgrep -c '$$$$' sdfs/train_test/{dataset}_test_{split_id}.sdf")


if __name__ == '__main__':
	mp.set_start_method('spawn')
	queue = mp.Queue()

	for dataset in dataset_list:
		p=mp.Process(target=convert_dataset, args=(dataset,))
		p.start()
