import os
import glob
from pathlib import Path



def process_file_name(name):
	dir = os.path.dirname(name)
	dataset_dir = dir.split('/')[1]
	dataset = dataset_dir.split('.')[0]
	name = dataset
	return name

def process_data(data):
	
	return data

with open('results/all_results.txt', 'w+') as output_file:
	for file_name in glob.glob('results/*/final*'):
		with open(file_name ) as input_file:
			data = input_file.read()
			name = process_file_name(file_name)
			data = process_data(data)
			print(f'file_name:{name}')
			print(data)
			output_file.write(f'file_name:{name}\n')
			output_file.write(data)


