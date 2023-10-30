import multiprocessing as mp
from multiprocessing import Pool, Value
import os
import os.path as osp
from tqdm import tqdm
import shutil, errno
import itertools
import time
from datetime import datetime
import torch

branch = 'schnet' # Change this
gnn_type = 'schnet'
task_comment = '\"with bcl-feat\"' # Change this

def gitclone(dir_name):
    cwd = os.getcwd()
    os.chdir(dir_name)
    os.system('git clone git@github.com:LanceKnight/JCIM.git')
    os.chdir('JCIM')
    os.system(f'git checkout {branch}') 
    os.chdir(cwd)

def gitupdate(dir_name):
    cwd = os.getcwd()
    os.chdir(dir_name+'/JCIM')
    os.system('git gc')
    os.system(f'git checkout {branch}') 
    os.system('git pull')
    os.chdir(cwd)

def run_command(exp_id, args): # Change this
    print(f'args:{args}')
    # Model=kgnn
    os.system(f'python -W ignore entry.py \
        --task_name id{exp_id}_lr{args[4]}_seed{args[1]}\
        --dataset_name {args[0]} \
        --seed {args[1]}\
        --num_workers 14 \
        --dataset_path ../../../dataset/ \
        --enable_oversampling_with_replacement \
        --warmup_iterations {args[2]} \
        --max_epochs {args[3]}\
        --peak_lr {args[4]} \
        --end_lr {args[5]} \
        --batch_size 32 \
        --default_root_dir actual_training_checkpoints \
        --gpus 1 \
        --task_comment {task_comment}\
        --split_num {args[6]}\
        ')\

def copyanything(src, dst):
    '''
    does not overwrite
    return True if created a new one
    return False if folder exist
    '''
    if os.path.exists(dst):
        # shutil.rmtree(dst)
        print(f'{dst} exists and remain untouched')
        return False
    else:
        try:
            shutil.copytree(src, dst)
        except OSError as exc: # python >2.5
            if exc.errno in (errno.ENOTDIR, errno.EINVAL):
                shutil.copy(src, dst)
            else: raise
        return True

# def copyanything(src, dst):
#     # If dst exits, remove it first
#     if os.path.exists(dst):
#         shutil.rmtree(dst)
#     try:
#         shutil.copytree(src, dst)
#     except OSError as exc: # python >2.5
#         if exc.errno in (errno.ENOTDIR, errno.EINVAL):
#             shutil.copy(src, dst)
#         else: raise
#     return True

def run(exp_id, *args):
    print(f'args1:{args}')
    exp_name = f'exp{exp_id}_schnet_dataset{args[0]}_{gnn_type}_seed{args[1]}_peak{args[4]}_{args[6]}' # Change this
    print(f'=====running {exp_name}')

    # Go to correct folder
    dir_name = f'../experiments/{exp_name}' 
    # if not os.path.exists(dir_name):
    #   os.mkdir(dir_name)
    #   gitclone(dir_name)
    # gitupdate(dir_name)

    global github_repo_dir
    newly_created =copyanything(github_repo_dir, dir_name)
    cwd = os.getcwd()
    os.chdir(dir_name+'/JCIM')


    # Task
    if not osp.exists('logs/test_sample_scores.log'):
        if not newly_created:
            os.chdir(cwd)
            overwrite_dir(github_repo_dir, dir_name)
            os.chdir(dir_name+'/JCIM') 
        os.makedirs('logs', exist_ok=True)
        with open('logs/params.log', 'w+') as out:
            out.write(f'dataset:{args[0]}')
            out.write(f'seed:{args[1]}')
            out.write(f'warmup:{args[2]}')
            out.write(f'epochs:{args[3]}')
            out.write(f'peak:{args[4]}')
            out.write(f'end:{args[5]}')
            out.write(f'batch_size:{32}')
            out.write(f'task_comment:{task_comment}')
            out.write(f'split_num:{args[6]}')\



    run_command(exp_id, args) # Change this
    # time.sleep(3)
    print(f'----{exp_name} finishes')
    os.chdir(cwd)
    

def attach_exp_id(input_tuple, tuple_id):
    # Add experiment id in front of the input hyperparam tuple
    record = [tuple_id]
    record.extend(list(input_tuple))
    return record


# Global variable
# Github repo template
github_repo_dir = f'../experiments/template_dataset_layers'

if __name__ == '__main__':
    mp.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    start_time = time.time()
    now = datetime.now()
    print(f'scheduler start time:{now}')


    dataset_list = [ '485290', '1843', '2258', '488997','2689', '435008', '1798', '435034', '463087'] # arg0
    # dataset_list = ['463087', '488997', '2258', '485290', '1843', '435008']
    # dataset_list = [ '1798', '435034', '2689']
    seed_list = [2] # arg1
    # seed_list = [1]
    warmup_list = [200] # arg2
    epochs_list = [40] # arg3
    peak_lr_list = [1.4e-4] # arg4
    end_lr_list = [1e-9] # arg5
    # split_id = ['split1', 'split2','split3', 'split4','split5'] # arg6
    split_id = ['split2'] # arg6
    

    data_pair = list(itertools.product(dataset_list, seed_list, warmup_list, epochs_list, peak_lr_list, end_lr_list, split_id)) # Change this
    print(f'num data_pair:{len(data_pair)}')
    data_pair_with_exp_id = list(map(attach_exp_id, data_pair, range(len(data_pair))))
    print(f'data_pair_with_exp_id:{data_pair_with_exp_id}')
    with open('logs/scheduler.log', "w+") as out_file:
        out_file.write(f'num data_pair:{len(data_pair)}\n\n')
        out_file.write(f'data_pair_with_exp_id:{data_pair_with_exp_id}\n')


    # Clone once from github
    
    if not os.path.exists(github_repo_dir):
        os.mkdir(github_repo_dir)
        gitclone(github_repo_dir)
    gitupdate(github_repo_dir)

    
    with Pool(processes = 5) as pool: # Change this
        pool.starmap(run, data_pair_with_exp_id)
    end_time=time.time()
    run_time = end_time-start_time
    print(f'scheduler running time: {run_time/3600:0.0f}h{(run_time)%3600/60:0.0f}m{run_time%60:0.0f}')
    now = datetime.now()
    print(f'scheduler finsh time:{now}')
   
    with open('logs/scheduler.log', "a") as out_file:
        out_file.write(f'run time:{run_time}\n')
    print(f'finish')
