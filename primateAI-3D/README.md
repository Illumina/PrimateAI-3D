
## PrimateAI-3D code

This folder contains all code to train the PrimateAI-3D model.

To install dependencies, create a new conda environment:
```sh
conda env create -f environment.yml
```

PrimateAI-3D can then be started as follows

```sh
python nn_worker.py \
  <config file path>
```

\<config file path\> is the path to the PrimateAI-3D configuration file.
