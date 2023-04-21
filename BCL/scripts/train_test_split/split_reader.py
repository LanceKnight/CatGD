import torch

dataset = '1798'
input_id_file = f'shrink_{dataset}_seed2_split2.pt'

id_dict = torch.load(f'data_split/{input_id_file}')
train_id = id_dict['train']
test_id = id_dict['test']

test_id.sort()

print(test_id)