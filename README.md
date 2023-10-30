# Integrating Expert Knowledge with Deep Learning Improves QSAR Models for CADD Modeling
This repository is the official implementation of paper <i>Integrating Expert Knowledge with Deep Learning Improves QSAR Models for CADD Modeling</i>.

This paper introduces a simple yet effective strategy to boost the predictive power of QSAR deep learning models. The strategy proposes to train GNNs together with traditional descriptors, combining the strengths of both methods. 
![image](https://user-images.githubusercontent.com/5760199/232235793-5f394dc9-5f51-47f5-844f-b2ab948d83c0.png)


# Acquire the Datasets

This repository does NOT include the datasets used in the experiment. Please download the datasets from [this link](https://figshare.com/articles/dataset/Well-curated_QSAR_datasets_for_diverse_protein_targets/20539893)

These are well-curated realistic datasets that removes false positves for a diverse important drug targets. The datasets also feature in its high imbalance nature (much more inactive molecules than active ones). Original papers of the datasets: see references [1,2]. 

**Introduction of the Datasets**

High-throughput screening (HTS) is the use of automated equipment to rapidly screen thousands to millions of molecules for the biological activity of interest in the early drug discovery process [3]. However, this brute-force approach has low hit rates, typically around 0.05\%-0.5\% [4]. Meanwhile, PubChem [5] is a database supported by the National Institute of Health (NIH) that contains biological activities for millions of drug-like molecules, often from HTS experiments. However, the raw primary screening data from the PubChem have a high false positive rate [6]. A series of secondary experimental screens on putative actives is used to remove these. While all relevant screens are linked, the datasets of molecules are often not curated to list all inactive molecules from the primary HTS and only confirmed actives after secondary screening. Thus, we identified nine high-quality HTS experiments in PubChem covering all important target protein classes for drug discovery. We carefully curated these datasets to have lists of inactive and confirmed active molecules. 

**Statistics of the Datasets, specified by PubChem Assay ID (AID)**

<p align="center">
  <img src="https://user-images.githubusercontent.com/5760199/186287898-30e5d105-6d80-4580-af9f-3044d9b2c8f8.png" />
</p>

# Process the Datasets

Uncompress the downloaded file and you will see several .sdf files. 

Use the scripts in BCL folder to process them into BCL features. See README.md in the BCL folder for the detailed instruction. The generated BCL features file (`{dataset}.RSR.object`) should be placed one folder above this working folder. (i.e., in `../bcl-feat/`)




# Run the Codes

Here is an exmaple for running the code:

`python entry.py --dataset_name 1798 --dataset_path ../dataset/ --num_workers 16 --enable_oversampling_with_replacement --warmup_iterations 200 --max_epochs 3 --peak_lr 5e-2 --end_lr 1e-9 --batch_size 32
--default_root_dir actual_training_checkpoints --seed 1 --gpus 1 --split_num split2`

# Q&A

Feel free to drop questions in the **<em>Issues</em>** tab, or contact me at yunchao.liu@vanderbilt.edu
  

# References
[[1] Butkiewicz, Mariusz, et al. "Benchmarking ligand-based virtual High-Throughput Screening with the PubChem database." Molecules 18.1 (2013): 735-756.](https://www.mdpi.com/1420-3049/18/1/735))

[[2] Butkiewicz, Mariusz, et al. "High-throughput screening assay datasets from the pubchem database." Chemical informatics (Wilmington, Del.) 3.1 (2017).](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5962024/)
