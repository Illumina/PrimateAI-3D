# PrimateAI-3D Torch

This model annotates human missense variants with their predicted effect on protein function (pathogenicity), as described in [The landscape of tolerated genetic variation in humans and primates](https://www.science.org/doi/10.1126/science.abn8197) and [PrimateAI-3D outperforms AlphaMissense in real-world cohorts](https://www.medrxiv.org/content/10.1101/2024.01.12.24301193v1). The annotations for all possible missense variants are included in the [data package](https://ilmn-my.sharepoint.com/:u:/g/personal/thamp_illumina_com/EbBe877iGXxNrr4BpTCMdJgBkp1PvtZclmQX38Epaqp6xg?e=7Om5VJ) (file `./evaluation/pai3d.ensPrimAndHum.csv`). These and all other data TODO in the data package are free for academic and not-for-profit use; other use requires a commercial license from Illumina, Inc.

## License
PrimateAI-3D source code included in this repository is provided under the PolyForm Strict License 1.0.0. PrimateAI-3D includes several third party packages TODO provided under other open source licenses, please see NOTICE for additional details. All data in the data package is provided under the CC BY NC 4.0 license for academic and non-commercial use; other use requires a commercial license from Illumina, Inc.

## Data package
The data package is available here.
It includes the precomputed scores for PrimateAI-3D trained with human and/or primate variants, model weights and all input and output data needed to train PrimateAI-3D models.

### Precomputed scores
File `<dataFolderPath>/ens/ensemble_scores/pai3d.ensPrimAndHum.csv` has scores for PrimateAI-3D trained with both primate and human variants (evaluated in [PrimateAI-3D outperforms AlphaMissense in real-world cohorts](https://www.medrxiv.org/content/10.1101/2024.01.12.24301193v1)).

`<dataFolderPath>/ens/ensemble_scores/pai3d.ensPrim.orig.csv` has scores for PrimateAI-3D trained with primate variants. They are the same as those published in the [PrimateAI-3D paper](https://www.science.org/doi/10.1126/science.abn8197) with some longer structures added.

`<dataFolderPath>/ens/ensemble_scores/pai3d.ensHum.csv` has scores of PrimateAI-3D trained with human variants (introduced in [PrimateAI-3D outperforms AlphaMissense in real-world cohorts](https://www.medrxiv.org/content/10.1101/2024.01.12.24301193v1)). These scores are ensembles of 10 different PrimateAI-3D base models which are reproducible with files in `<dataFolderPath>/ens/human_singlemodels/`: `human_singlemodels/conf` has the config files for training (below); `human_singlemodels/weights` has the model weights created during training for each epoch and model; `human_singlemodels/scores` has the precomputed individual model output scores (saved after each epoch). Notebook `./notebooks/evalScores.ipynb` shows how to combine and evaluate the individual precomputed scores.

### Other files and folders
- `evaluation`: clinical and deep mutagenesis assay evaluation data; used in notebook `./notebooks/evalScores.ipynb`.
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

Extract with
```
tar -xvzf data_package.tar.gz
```

The extraction directory is from now on referred to as `<dataFolderPath>`.

### Run PAI3D 
Execute the commands below to train one PrimateAI-3D model.

```
python ./modules/worker.py --json_conf=<configFilePath> --datafolder=<dataFolderPath> --runfolder=<runFolderPath>
```

`<dataFolderPath>` is the folder of the extracted data archive (download link above).

`<configFilePath>` is for example `<dataFolderPath>/ens/human_singlemodels/conf/0.conf.json`.

`<runFolderPath>` is the folder where log and output files will be saved .

## Evaluation
Notebook `./notebooks/evalScores.ipynb` performs an evaluation of the precomputed PrimateAI-3D scores included in `<dataFolderPath>/ens/`.

## Contact
thamp@illumina.com
