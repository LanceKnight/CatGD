import os
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm



dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290', '9999']
# dataset_list = ['1798']
dataset_list = ['999']
# activity_list = ['actives', 'inactives']

def process_dataset(dataset):
	print(f'current dataset:{dataset}')
	# input_filesname0 = f'/home/liuy69/projects/unified_framework/dataset/qsar/clean_sdf/raw/{dataset}_{activity_list[0]}_clean.sdf'
	# input_filesname1 = f'/home/liuy69/projects/unified_framework/dataset/qsar/clean_sdf/raw/{dataset}_{activity_list[1]}_clean.sdf'



	# os.system(f'bcl.exe descriptor:GenerateDataset \
	# 	-source "Combined(SdfFile(filename={input_filesname0}), SdfFile(filename={input_filesname1}))"\
	# 	-feature_labels RSR.object\
	# 	-result_labels "Combine(Greater(lhs=Subtract(lhs=Log(Constant(1e+06)),rhs=Log(MiscProperty(Activity,values per molecule=1))),rhs=Constant(3.5)))"\
	# 	-output data/{dataset}_labeled_clean.RSR.bin\
	# 	')


	input_filesname=f'sdfs/id_{dataset}.sdf'

	os.system(f'bcl.exe descriptor:GenerateDataset \
		-source "SdfFile(filename={input_filesname})"\
		-feature_labels RSR.object\
		-result_labels "Combine(\
								Greater(lhs=Subtract(lhs=Log(Constant(1e+06))\
													, rhs=Log(MiscProperty(Activity,values per molecule=1))\
										   			)\
										,rhs=Constant(3.5)\
										)\
								)"\
		-output feat/{dataset}.RSR.csv\
		-id_labels "FileID({input_filesname})"\
		')
		

if __name__ == '__main__':
	mp.set_start_method('spawn')
	queue = mp.Queue()

	for dataset in dataset_list:
		p=mp.Process(target=process_dataset, args=(dataset,))
		p.start()





