# PrimateAI-3D Torch

This model annotates human missense variants with their predicted effect on protein function (pathogenicity), as described in [H. Gao, T. Hamp, K. K. Farh et al. Science (2023)](https://www.science.org/doi/10.1126/science.abn8197) and [D. Parry, T. Bosc, T. Hamp, K. K. Farh et al. medRxiv (2024)](https://www.medrxiv.org/content/10.1101/2024.01.12.24301193v1). The annotations for all possible missense variants together with all data needed to train and evaluate PrimateAI-3D are included in the data package. To download the data package, please complete the [license agreement](https://illumina2.na1.adobesign.com/public/esignWidget?wid=CBFCIBAA3AAABLblqZhDaZSRjhLd-Jumb12j-ihAbO0vBakcvXgS2MpkFnF_VJXWW4J_DBF5yDTCzOQJ8zrU*); the download link will be shared via email shortly after submission. The data package is free for academic and not-for-profit use; other use requires a commercial license from Illumina, Inc.

## Data package
Please see above for downloading the data package. It includes the precomputed scores for PrimateAI-3D, model weights and all input and output data needed to train PrimateAI-3D models.

### Precomputed scores
`pai_scores/pai3d.ensPrim.orig.csv` has scores for PrimateAI-3D trained with primate variants. They are the same as those published in the [PrimateAI-3D paper](https://www.science.org/doi/10.1126/science.abn8197) with some longer structures added.

### Other files and folders in `pai_trainAndEvalData`
- `evaluation`: clinical and deep mutagenesis assay evaluation data; used in notebook `./notebooks/evalScores.ipynb` (this repo).
- `mapping`: helper data to map between protein, DNA coordinates and species.
- `pdbDict`: input features and output labels for PrimateAI-3D. Each file is a Python dictionary with gene names as keys and numpy arrays as values. The first dimension of each numpy array is the length of the gene in amino acids +1 (to enable 1-based indexing). The second dimension is the number of features (e.g. 20 for a 1-hot encoding of amino acids).
- `validation`: helper files for static validation during PrimateAI-3D training.

## Running PrimateAI-3D Torch
### Clone the repo
```
git clone URL
```

### Setup and activate environment
```
cd pai3d_torch
conda env create -f pai3d_torch.env.yml
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
conda activate pai3d_torch
```

### Download the data archive
Download the data package (see above for download link).

The data package directory is from now on referred to as `<dataFolderPath>`.

### Run PAI3D 
Execute the commands below to train one PrimateAI-3D model.

```
python ./modules/worker.py --json_conf=<configFilePath> --datafolder=<dataFolderPath> --runfolder=<runFolderPath>
```

`<dataFolderPath>` is the folder of the extracted data archive (download link above).

`<configFilePath>` is for example `pai_trainAndEvalData/human_singlemodels/conf/0.conf.json`.

`<runFolderPath>` is the folder where log and output files will be saved .

## Evaluation
Notebook `./notebooks/evalScores.ipynb` (this repo) performs an evaluation of the precomputed PrimateAI-3D scores.

## Contact
thamp@illumina.com
