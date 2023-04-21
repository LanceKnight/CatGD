import os
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm



dataset_list = ['435008']#, '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290']

obj_function_name = '"FPPvsPPV(cutoff=0.5, cutoff_type=fpp_percent, parity=1)"'

def process_(dataset):
	# os.system(f'ls results/{dataset}.RSR.1_32_005_025/')#independent0-4_monitoring0-4_number0.gz.txt')
	os.system(f"bcl.exe model:ComputeStatistics \
		-input 20_models/results/{dataset}.RSR.1_32_005_025/independent0-4_monitoring0-4_number0.gz.txt\
		-plot_x FPR\
		-obj_function {obj_function_name}\
		-filename_obj_function 20_models/lance_logs/object_function_{obj_function_name}.txt\
		-image_format png\
		")
#'Bootstrap(repeats=10, function=AucRocCurve(cutoff=0.5, parity=1, x_axis_log=1, min fpr=0.001, max fpr=0.1), confidence interval=0.95)'\

if __name__ == '__main__':
	mp.set_start_method('spawn')

	for dataset in dataset_list:
		p=mp.Process(target=process_, args=(dataset,))
		p.start()





