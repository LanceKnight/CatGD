from models.MolKGNN.MolKGNNNet import MolKGNNNet
from wrapper import QSARDataset, ToXAndPAndEdgeAttrForDeg

from argparse import ArgumentParser
import torch
from torch.nn import Linear, BCEWithLogitsLoss
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch import optim




def train(optimizer):
    for batch in train_loader:
        batch.to(device)
        graph_embedding = model(batch)
        pred = ffn(graph_embedding).squeeze(-1)
        loss = BCEWithLogitsLoss()(pred, batch.y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validate', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--record_valid_pred', action='store_true', default=False)
    parser.add_argument('--train_metric', action='store_true', default=False)
    parser.add_argument('--warmup_iterations', type=int, default=60000)
    parser.add_argument('--peak_lr', type=float, default=5e-2)
    parser.add_argument('--end_lr', type=float, default=1e-9)
    parser.add_argument('--weight_decay', type=float, default=0)

    # For linear layer
    parser.add_argument('--ffn_dropout_rate', type=float, default=0.25)
    parser.add_argument('--ffn_hidden_dim', type=int, default=64)
    parser.add_argument('--task_dim', type=int, default=1)

    # For data
    parser.add_argument('--dataset_name', type=str, default="435034")
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=17)
    parser.add_argument('--enable_oversampling_with_replacement', action='store_true', default=False)
    parser.add_argument('--dataset_path', type=str, default="../dataset/")
    parser.add_argument('--split_num', type=str, default="split1")

    gnn_type = 'kgnn'

    # KGNN
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_kernel1_1hop', type=int, default=10)
    parser.add_argument('--num_kernel2_1hop', type=int, default=20)
    parser.add_argument('--num_kernel3_1hop', type=int, default=30)
    parser.add_argument('--num_kernel4_1hop', type=int, default=50)
    parser.add_argument('--num_kernel1_Nhop', type=int, default=10)
    parser.add_argument('--num_kernel2_Nhop', type=int, default=20)
    parser.add_argument('--num_kernel3_Nhop', type=int, default=30)
    parser.add_argument('--num_kernel4_Nhop', type=int, default=50)
    parser.add_argument('--node_feature_dim', type=int, default=28)
    parser.add_argument('--edge_feature_dim', type=int, default=7)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--dropout_ratio', type=float, default=0)

    args = parser.parse_args()

    model = MolKGNNNet(num_layers=args.num_layers,
                                         num_kernel1_1hop = args.num_kernel1_1hop,
                                         num_kernel2_1hop = args.num_kernel2_1hop,
                                         num_kernel3_1hop = args.num_kernel3_1hop,
                                         num_kernel4_1hop = args.num_kernel4_1hop,
                                         num_kernel1_Nhop = args.num_kernel1_Nhop,
                                         num_kernel2_Nhop = args.num_kernel2_Nhop,
                                         num_kernel3_Nhop = args.num_kernel3_Nhop,
                                         num_kernel4_Nhop = args.num_kernel4_Nhop,
                                         x_dim = args.node_feature_dim,
                                         edge_attr_dim=args.edge_feature_dim,
                                         graph_embedding_dim = args.hidden_dim,
                                         drop_ratio=args.dropout_ratio
                       )

    ffn = Linear(32, 1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ffn.to(device)

    dataset = QSARDataset(
                root=args.dataset_path+'qsar/clean_sdf',
                dataset=args.dataset_name,
                gnn_type=gnn_type,
                pre_transform=ToXAndPAndEdgeAttrForDeg(),
                )

    train_idx = dataset.get_idx_split()['train']
    test_idx = dataset.get_idx_split()['test']
    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]

    # Make DataLoader
    num_train_active = len(torch.nonzero(torch.tensor([data.y for data in train_dataset])))
    num_train_inactive = len(train_dataset) - num_train_active

    train_sampler_weight = torch.tensor([(1. / num_train_inactive)
                                                         if data.y == 0
                                                         else (1. / num_train_active)
                                                         for data in
                                                         train_dataset])

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_sampler = WeightedRandomSampler(weights=train_sampler_weight,
                                          num_samples=len(train_sampler_weight),
                                          generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )


    optimizer = optim.Adam(model.parameters(), lr=1 * 10 ** -2)
    for i in tqdm(range(10)):
        train(optimizer)
    
