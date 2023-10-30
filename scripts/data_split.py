import torch
import random
import hashlib
import json

def checksum_and_save(file, base_file_name):
    file_name = base_file_name+'.pt'
    torch.save(file, file_name)
    data_md5 = hashlib.md5(json.dumps(file, sort_keys=True).encode('utf-8')).hexdigest()
    print(f'data_md5_checksum:{data_md5}')
    # print(f'file saved at {file_name}')
    with open(f'{file_name}.checksum', 'w+') as checksum_file:
        checksum_file.write(data_md5)

def get_random_split(num_active, num_inactive, seed, dataset_name, shrink, file_name):
    active_idx = list(range(num_active))
    inactive_idx = list(range(num_active, num_active + num_inactive))

    print(f'num_active:{num_active}, num_inactive:{num_inactive}')

    random.seed(seed)
    random.shuffle(active_idx)
    random.shuffle(inactive_idx)


    if shrink == False:
        num_active_train = round(num_active * 0.8)
        num_inactive_train = round(num_inactive * 0.8)
        num_active_test = num_active - num_active_train
        num_inactive_test = num_active - num_active_train
        
    else:
        shrink_num=10000 if dataset_name != '9999' else 50
        # num_inactive_train = shrink_num - num_active
        mix_idx = active_idx+inactive_idx
        mix_idx = mix_idx[:shrink_num]
        random.shuffle(mix_idx)

        trunk_size = round(shrink_num*0.2)

    split1_dict = {}
    split1_dict['train'] = mix_idx[:trunk_size*4]
    split1_dict['test'] = mix_idx[trunk_size*4:]
    # print(split1_dict)
    print(f'split1:train_len:{len(split1_dict["train"])} test_len:{len(split1_dict["test"])}')
    checksum_and_save(split1_dict, file_name+'_split1')

    split2_dict = {}
    split2_dict['train'] = mix_idx[:trunk_size*3]
    split2_dict['test'] = mix_idx[trunk_size*3:trunk_size*4]
    split2_dict['train'] = split2_dict['train'] + mix_idx[trunk_size*4:]
    # print(split2_dict)
    print(f'split2:train_len:{len(split2_dict["train"])} test_len:{len(split2_dict["test"])}')
    checksum_and_save(split2_dict, file_name+'_split2')

    split3_dict = {}
    split3_dict['train'] = mix_idx[:trunk_size*2]
    split3_dict['test'] = mix_idx[trunk_size*2:trunk_size*3]
    split3_dict['train'] = split3_dict['train'] + mix_idx[trunk_size*3:]
    # print(split3_dict)
    print(f'split3:train_len:{len(split3_dict["train"])} test_len:{len(split3_dict["test"])}')
    checksum_and_save(split3_dict, file_name+'_split3')

    split4_dict = {}
    split4_dict['train'] = mix_idx[:trunk_size*1]
    split4_dict['test'] = mix_idx[trunk_size*1:trunk_size*2]
    split4_dict['train'] = split4_dict['train'] + mix_idx[trunk_size*2:]
    # print(split4_dict)
    print(f'split4:train_len:{len(split4_dict["train"])} test_len:{len(split4_dict["test"])}')
    checksum_and_save(split4_dict, file_name+'_split4')

    split5_dict = {}
    split5_dict['train'] = mix_idx[trunk_size:]
    split5_dict['test'] = mix_idx[:trunk_size]
    # print(split5_dict)
    print(f'split5:train_len:{len(split5_dict["train"])} test_len:{len(split5_dict["test"])}')
    checksum_and_save(split5_dict, file_name+'_split5')
    return split1_dict, split2_dict, split3_dict, split4_dict, split5_dict, 

def count_active_inactive(split_id, split_dict, dataset_info, dataset_name):
    num_active = dataset_info[dataset_name]['num_active']
    active_counter = 0
    inactive_counter = 0
    for idx in split_dict['train']:
        if idx < num_active:
            active_counter+=1
        else:
            inactive_counter+=1
    print('-------')
    print(f'train split{split_id} num_active:{active_counter}, num_inactive:{inactive_counter}')
    active_counter = 0
    inactive_counter = 0
    for idx in split_dict['test']:
        if idx < num_active:
            active_counter+=1
        else:
            inactive_counter+=1
    print(f'test split{split_id} num_active:{active_counter}, num_inactive:{inactive_counter}')

   

if __name__ == '__main__':
    dataset_info = {
        '435008':{'num_active':233, 'num_inactive':217923-24},#{'num_active':233, 'num_inactive':217925},
        '1798':{'num_active':187, 'num_inactive':61645-6},#{'num_active':187, 'num_inactive':61645},
        '435034': {'num_active':362, 'num_inactive':61393-6},#{'num_active':362, 'num_inactive':61394},
        '1843': {'num_active':172, 'num_inactive':301318-30},#{'num_active':172, 'num_inactive':301321},
        '2258': {'num_active':213, 'num_inactive':302189-31},#{'num_active':213, 'num_inactive':302192},
        '463087': {'num_active':703, 'num_inactive':100171-17},#{'num_active':703, 'num_inactive':100172},
        '488997': {'num_active':252, 'num_inactive':302051-31},#{'num_active':252, 'num_inactive':302054},
        '2689': {'num_active':172, 'num_inactive':319617-29},#{'num_active':172, 'num_inactive':319620},
        '485290': {'num_active':278, 'num_inactive':341026-38},#{'num_active':281, 'num_inactive':341084},
        '9999':{'num_active':37, 'num_inactive':226-1},
    }
    seed_list = [2]
    dataset_name_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290', '9999']
    # dataset_name_list = ['435008']
    # dataset_name_list = ['9999']
    is_shrink = True
    for dataset_name in dataset_name_list:
        for seed in seed_list:
            if is_shrink:
                file_name = f'data_split/shrink_{dataset_name}_seed{seed}'
            else:
                file_name = f'data_split/{dataset_name}_seed{seed}'
            num_active = dataset_info[dataset_name]['num_active']
            num_inactive = dataset_info[dataset_name]['num_inactive']
            split_list = get_random_split(num_active, num_inactive, seed, dataset_name, shrink=is_shrink, file_name = file_name)
            print('===========')
            print(f'dataset:{dataset_name}, total_active:{dataset_info[dataset_name]["num_active"]} total_inactive:{dataset_info[dataset_name]["num_inactive"]}')
            for split_id, split in enumerate(split_list):
                count_active_inactive(split_id, split, dataset_info, dataset_name)
