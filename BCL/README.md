This folder is for generating BCL featureus.

For an introduction to BCL, please refer to:
https://www.frontiersin.org/articles/10.3389/fphar.2022.833099/full

The scripts for generating BCL features are in folder `scripts`. There are two folders in `scripts`. One is `full_dataset_featurization` that generates BCL features and store in .csv file that can be concatenated with GNNs (The `wrapper.py` will read in the .csv file and store it as `bcl_feat` attribute in `torch_geometric.data.Data` object). The other one is `train_test_split`, which is used to run training and testing in BCL to get its performance as a baseline benchmark.

To generate the .csv file, make sure you execute the command in this folder (i.e., `BCL` folder) and follow the steps below:

1. create a folder named `sdfs` and put your .sdf files in this folder
2. run `python scripts/full_dataset_featurization/1_add_id_to_sdf.py` to add ids to .sdf files (the ids are for manual lookup of molecules in the .sdf file if needed)
3. run `python scripts/full_dataset_featurizatio/2_data_preparation.py` to create the .csv file. The created file will be stored in the `bcl-feat` folder under the name `{dataset}.RSR.csv`
