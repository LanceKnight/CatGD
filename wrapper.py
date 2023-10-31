import math
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.collate import collate
from torch_geometric.utils import degree
from tqdm import tqdm
import numpy as np
import rdkit
import rdkit.Chem.EState as EState
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.rdPartialCharges as rdPartialCharges


pattern_dict = {'[NH-]': '[N-]', '[OH2+]':'[O]'}

def smiles_cleaner(smiles):
    '''
    This function is to clean smiles for some known issues that makes
    rdkit:Chem.MolFromSmiles not working
    '''
    print('fixing smiles for rdkit...')
    new_smiles = smiles
    for pattern, replace_value in pattern_dict.items():
        if pattern in smiles:
            print('found pattern and fixed the smiles!')
            new_smiles = smiles.replace(pattern, replace_value)
    return new_smiles


def add_bcl_feature(data_list, dataset):
    with open(f'../bcl-feat/{dataset}.RSR.csv') as bcl_feat_file:
        lines = bcl_feat_file.readlines()
        num_filtered = {'9999':1,
                        '435034':6,
                        '1798':6,
                        '463087':17,
                        '435008':24,
                        '485290':38,
                        '1843':30,
                        '488997':31,
                        '2258':31,
                        '2689':29,
                        }

        if (len(lines)+num_filtered[dataset]) != len(data_list):
            raise Exception(f'numer of mol in bcl feature ({len(lines)}) does not equal to the number of mol in input ({len(data_list)})')
        new_data_list = []
        for _, line in enumerate(lines):
            values = line.split(',')
            id = int(values[0])
            feats = list(map(float,values[1:-1]))
            bcl_y = int(values[-1])
            y = data_list[id].y
            if  y!= bcl_y:
                raise Exception(f'id{id} does not have the same y label. BCL has a label of {bcl_y}, while data has a '
                                f'label of {y}')
            bcl_feat = torch.tensor(feats).unsqueeze(0)
            data_list[id].bcl_feat = bcl_feat
            new_data_list.append(data_list[id])
            # print(bcl_feat)
        print(len(new_data_list))
        return new_data_list

def one_hot_vector(val, lst):
	'''
	Converts a value to a one-hot vector based on options in lst
	'''
	if val not in lst:
		val = lst[-1]
	return map(lambda x: x == val, lst)


def get_atom_rep(atom):
    features = []
    # H, C, N, O, F, Si, P, S, Cl, Br, I, other
    features += one_hot_vector(atom.GetAtomicNum(), [1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 999])
    features += one_hot_vector(len(atom.GetNeighbors()), list(range(1, 5)))

    features.append(atom.GetFormalCharge())
    features.append(atom.IsInRing())
    features.append(atom.GetIsAromatic())
    features.append(atom.GetExplicitValence())
    features.append(atom.GetMass())

    # Add Gasteiger charge and set to 0 if it is NaN or Infinite
    gasteiger_charge = float(atom.GetProp('_GasteigerCharge'))
    if math.isnan(gasteiger_charge) or math.isinf(gasteiger_charge):
        gasteiger_charge = 0
    features.append(gasteiger_charge)

    # Add Gasteiger H charge and set to 0 if it is NaN or Infinite
    gasteiger_h_charge = float(atom.GetProp('_GasteigerHCharge'))
    if math.isnan(gasteiger_h_charge) or math.isinf(gasteiger_h_charge):
        gasteiger_h_charge = 0

    features.append(gasteiger_h_charge)
    return features

def get_extra_atom_feature(all_atom_features, mol):
    '''
    Get more atom features that cannot be calculated only with atom,
    but also with mol
    :param all_atom_features:
    :param mol:
    :return:
    '''
    # Crippen has two parts: first is logP, second is Molar Refactivity(MR)
    all_atom_crippen = rdMolDescriptors._CalcCrippenContribs(mol)
    all_atom_TPSA_contrib = rdMolDescriptors._CalcTPSAContribs(mol)
    all_atom_ASA_contrib = rdMolDescriptors._CalcLabuteASAContribs(mol)[0]
    all_atom_EState = EState.EStateIndices(mol)

    new_all_atom_features = []
    for atom_id, feature in enumerate(all_atom_features):
        crippen_logP = all_atom_crippen[atom_id][0]
        crippen_MR = all_atom_crippen[atom_id][1]
        atom_TPSA_contrib = all_atom_TPSA_contrib[atom_id]
        atom_ASA_contrib = all_atom_ASA_contrib[atom_id]
        atom_EState = all_atom_EState[atom_id]

        feature.append(crippen_logP)
        feature.append(crippen_MR)
        feature.append(atom_TPSA_contrib)
        feature.append(atom_ASA_contrib)
        feature.append(atom_EState)

        new_all_atom_features.append(feature)
    return new_all_atom_features


def mol2graph(mol, D=3):
    try:
        conf = mol.GetConformer()
    except Exception as e:
        smiles = AllChem.MolToSmiles(mol)
        print(f'smiles:{smiles} error message:{e}')

    atom_pos = []
    atomic_num_list = []
    all_atom_features = []

    # Get atom attributes and positions
    rdPartialCharges.ComputeGasteigerCharges(mol)

    for i, atom in enumerate(mol.GetAtoms()):
        atomic_num = atom.GetAtomicNum()
        atomic_num_list.append(atomic_num)
        atom_feature = get_atom_rep(atom)
        if D == 2:
            atom_pos.append(
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y])
        elif D == 3:
            atom_pos.append([conf.GetAtomPosition(
                i).x, conf.GetAtomPosition(i).y,
                             conf.GetAtomPosition(i).z])
        all_atom_features.append(atom_feature)
    # Add extra features that are needs to calculate using mol
    all_atom_features = get_extra_atom_feature(all_atom_features, mol)

    # Get bond attributes
    edge_list = []
    edge_attr_list = []
    for idx, bond in enumerate(mol.GetBonds()):
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_attr = []
        bond_attr += one_hot_vector(
            bond.GetBondTypeAsDouble(),
            [1.0, 1.5, 2.0, 3.0]
        )

        is_aromatic = bond.GetIsAromatic()
        is_conjugate = bond.GetIsConjugated()
        is_in_ring = bond.IsInRing()
        bond_attr.append(is_aromatic)
        bond_attr.append(is_conjugate)
        bond_attr.append(is_in_ring)

        edge_list.append((i, j))
        edge_attr_list.append(bond_attr)

        edge_list.append((j, i))
        edge_attr_list.append(bond_attr)

    x = torch.tensor(all_atom_features, dtype=torch.float32)
    p = torch.tensor(atom_pos, dtype=torch.float32)
    edge_index = torch.tensor(edge_list).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    atomic_num = torch.tensor(atomic_num_list, dtype=torch.int)


    data = Data(x=x, p=p, edge_index=edge_index,
                edge_attr=edge_attr, atomic_num=atomic_num)  # , adj=adj,
    return data

def smiles2graph(D, smiles):
    if D == None:
        raise Exception(
            'smiles2grpah() needs to input D to specifiy 2D or 3D graph '
            'generation.')
    smiles = smiles.replace(r'/=', '=')
    smiles = smiles.replace(r'\=', '=')
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception as e:
        print(f'Cannot generate mol, error:{e}, smiles:{smiles}')

    if mol is None:
        smiles = smiles_cleaner(smiles)
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
        except Exception as e:
            print(f'Generated mol is None, error:{e}, smiles:{smiles}')
            return None
        if mol is None:
            print(f'Generated mol is still None after cleaning, smiles'
                  f':{smiles}')
    try:
        mol = Chem.AddHs(mol)
    except Exception as e:
        print(f'error in adding Hs{e}, smiles:{smiles}')

    if D == 2:
        Chem.rdDepictor.Compute2DCoords(mol)
    if D == 3:
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception as e:
            print(f'smiles:{smiles} error message:{e}')

    data = mol2graph(mol)
    return data


def process_smiles(dataset, root, D):
    data_smiles_list = []
    data_list = []
    for file, label in [(f'{dataset}_actives.smi', 1),
                        (f'{dataset}_inactives.smi', 0)]:
        smiles_path = os.path.join(root, 'raw', file)
        smiles_list = pd.read_csv(
            smiles_path, sep='\t', header=None)[0]

        # Only get first N data, just for debugging
        smiles_list = smiles_list

        for i in tqdm(range(len(smiles_list)), desc=f'{file}'):
            smi = smiles_list[i]

            data = smiles2graph(D, smi)
            if data is None:
                continue

            data.idx = i
            data.y = torch.tensor([label], dtype=torch.int)
            data.smiles = smi

            data_list.append(data)
            data_smiles_list.append(smiles_list[i])
    return data_list, data_smiles_list


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
                     torch.arange(0, feature_num * offset, offset,
                                  dtype=torch.long)
    x = x + feature_offset
    return x


class QSARDataset(InMemoryDataset):
    """
    Dataset from Mariusz Butkiewics et al., 2013, Benchmarking ligand-based
    virtual High_Throughput Screening with the PubChem Database

    There are nine subsets in this dataset, identified by their summary assay
    IDs (SAIDs):
    435008, 1798, 435034, 1843, 2258, 463087, 488997,2689, 485290
    The statistics of each subset can be found in the original publication
    """

    def __init__(self,
                 root,
                 D=3,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='435008',
                 empty=False,
                 gnn_type=None,
                 split_num = 'split1'):

        self.dataset = dataset
        self.root = root
        self.D = D
        self.gnn_type = gnn_type
        self.split_num =split_num
        super(QSARDataset, self).__init__(root, transform, pre_transform,
                                          pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, \
                                                              pre_transform, \
                                                              pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return f'{self.gnn_type}-{self.dataset}-bcl.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        print(f'processing dataset {self.dataset}')
        if self.dataset not in ['435008', '1798', '435034', '1843', '2258',
                                '463087', '488997','2689', '485290','9999']:
            raise ValueError('Invalid dataset name')

        RDLogger.DisableLog('rdApp.*')

        data_smiles_list = []
        data_list = []
        counter = -1
        invalid_id_list = []
        for file_name, label in [(f'{self.dataset}_actives_new.sdf', 1),
                                 (f'{self.dataset}_inactives_new.sdf', 0)]:
            sdf_path = os.path.join(self.root, 'raw', file_name)
            sdf_supplier = Chem.SDMolSupplier(sdf_path)
            for i, mol in tqdm(enumerate(sdf_supplier)):
                counter+=1
                if self.gnn_type in ['schnet', 'spherenet']:
                    data = self.schnet_process(mol)
                else:
                    data = self.regular_process(mol)

                if data is None:
                    invalid_id_list.append([counter, label])
                    continue
                data.idx = counter
                data.y = torch.tensor([label], dtype=torch.int)

                if self.pre_filter is not None:
                    data = self.pre_filter(data)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                smiles = AllChem.MolToSmiles(mol)
                data.smiles = smiles

                data_list.append(data)
                data_smiles_list.append(smiles)

        # Write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir, f'{self.gnn_type}-{self.dataset}-smiles.csv'),
                                  index=False, header=False)
        invalid_id_series = pd.DataFrame(invalid_id_list)
        invalid_id_series.to_csv(os.path.join(self.processed_dir, f'{self.gnn_type}-{self.dataset}-invalid_id.csv'),
                                 index=False,
                                              header=False)
        data_list= add_bcl_feature(data_list, self.dataset)
        print(data_list[0])
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



    def schnet_process(self, mol):
        conformer = mol.GetConformer()
        adj = rdkit.Chem.GetAdjacencyMatrix(mol)
        adj = np.triu(np.array(adj, dtype=int))  # keeping just upper triangular entries from sym matrix
        array_adj = np.array(np.nonzero(adj), dtype=int)  # indices of non-zero values in adj matrix
        edge_index = np.zeros((2, 2 * array_adj.shape[1]), dtype=int)  # placeholder for undirected edge list
        edge_index[:, ::2] = array_adj
        edge_index[:, 1::2] = np.flipud(array_adj)

        atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
        node_features = np.array(
            [atom.GetAtomicNum() for atom in atoms])  # Z
        positions = np.array(
            [conformer.GetAtomPosition(atom.GetIdx()) for atom in atoms])  # xyz positions
        edge_index, Z, pos = edge_index, node_features, positions
        data = Data(
            x=torch.as_tensor(Z).unsqueeze(1),
            edge_index=torch.as_tensor(edge_index, dtype=torch.long))
        data.pos = torch.as_tensor(pos, dtype=torch.float)
        return data

    def regular_process(self, mol):
        data = mol2graph(mol)
        return data


    def get_idx_split(self):
        split_dict = torch.load(f'data_split/shrink_{self.dataset}_seed2_{self.split_num}.pt')
        print(f'using {self.split_num}')
        try:
            invalid_id_list = pd.read_csv(os.path.join(self.processed_dir, f'{self.gnn_type}-'
                                                                       f'{self.dataset}-invalid_id.csv')
                                      , header=None).values.tolist()
            for id, label in invalid_id_list:
                print(f'checking invalid id {id}')
                if label == 1:
                    print('====warning: a positive label is removed====')
                if id in split_dict['train']:
                    split_dict['train'].remove(id)
                    print(f'found in train and removed')
                if id in split_dict['valid']:
                    split_dict['valid'].remove(id)
                    print(f'found in valid and removed')
                if id in split_dict['test']:
                    split_dict['test'].remove(id)
                    print(f'found in test and removed')
        except:
            print(f'invalid_id_list is empty')

        return split_dict

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            # item.idx = idx
            return item
        else:
            return self.index_select(idx)

    @staticmethod
    def collate(data_list):
        r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
        to the internal storage format of
        :class:`~torch_geometric.data.InMemoryDataset`."""
        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
        )

        return data, slices



if __name__ == "__main__":
    from clearml import Task
    from argparse import ArgumentParser

    gnn_type = 'spherenet'
    # gnn_type = 'schnet'

    use_clearml = False


    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='1798')
    parser.add_argument('--gnn_type', type=str, default=gnn_type)
    parser.add_argument('--task_name', type=str, default='Unnamed')
    args = parser.parse_args()

    print(f'===={gnn_type}====')



    qsar_dataset = QSARDataset(root='../dataset/qsar/clean_sdf',
                               dataset=args.dataset,
                               pre_transform=transform,
                               gnn_type=args.gnn_type
                               )


    data = qsar_dataset[60584]
    print(f'data:{data}')
    print('\n')
    import sys
    print(f'mem size:{sys.getsizeof(data) } bytes')
    print(f'totl mem size = mem_size * 200k /1000 = '
          f'{sys.getsizeof(data) * 200000/1000} MB')
