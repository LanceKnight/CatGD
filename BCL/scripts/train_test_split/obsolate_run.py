import os

dataset_path = '../../dataset/qsar/clean_sdf/raw/zip_files/'
dataset_name = '1798'
dataset_file = dataset_name+'_actives_new.sdf.gz'
dataset_full_path = dataset_path + dataset_file

# Dataset generation
os.system(f"bcl.exe descriptor:GenerateDataset -source 'SdfFile(filename={dataset_full_path})' -id_labels 'String(M1)' -result_labels 'Combine(IsActive)' -feature_labels 'Combine(Weight, LogP, HbondDonor, HbondAcceptor)' -output outputs/tests.bin")

# Train
os.system(f'launch -t cross-validation --local --dataset {dataset_full_path} --config-file config.example.ann.ini')
