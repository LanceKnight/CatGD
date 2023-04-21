import multiprocessing as mp
from multiprocessing import Pool
import os

def run(dataset):
    print('------')
    print(f'dataset{dataset}:')
    os.system(f"grep -c '$$$$' train_test_sdfs/{dataset}_act_and_inact_matched.sdf")




if __name__ == '__main__':
    # mp.set_start_method('spawn')

    dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290', 'Y1'] 
    dataset_list = ['485290'] 

    print(f'num of actives in each dataset')
    for dataset in dataset_list:
        run(dataset)

    # print('=====')
    # print(f'num of inactives in each dataset')
    # for dataset in dataset_list:
    #     run(dataset, 'inactives')


 