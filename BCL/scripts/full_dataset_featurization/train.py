import os
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm



dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290']
# dataset_list = ['1798']


# def train(dataset):
# 	os.system(f'/sb/apps/bcl/bcl/scripts/machine_learning/launch.py\
# 	 	-t cross_validation\
# 		--config-file config.ini\
# 		-d data/{dataset}_labeled_clean.RSR.rand.bin\
# 		--id {dataset}.RSR.1_32_005_025\
# 		--local --just-submit\
# 		')


def train(dataset):
	os.system(f'/sb/apps/bcl/bcl/scripts/machine_learning/launch.py\
		--config-file config.ini\
		-d feat/{dataset}.RSR.bin\
		-t cross_validation\
		--no-cross-validation\
		--local \
		--just-submit\
		--id {dataset}.RSR.1_32_005_025\
		')



if __name__ == '__main__':
	mp.set_start_method('spawn')

	for dataset in dataset_list:
		p=mp.Process(target=train, args=(dataset,))
		p.start()





