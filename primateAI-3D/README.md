
## PrimateAI-3D code

This folder contains code to train the PrimateAI-3D model.

To install dependencies, create a new conda environment:
```sh
conda env create -f dl_keras.yml
```

Extract data.tar.gz into the PrimateAI-3D directory.

PrimateAI-3D can then be started as follows

```sh
python worker.py config.json
```

Note that this is only a demo version of PrimateAI-3D, with many features and data removed.
