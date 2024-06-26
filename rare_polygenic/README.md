
## Rare variant polygenic models

This folder contains code to run rare variant poylgenic models. It can be 
installed with:

```sh
pip install -U -e .
```

That should give you a rvPRS command, CLI help is availble with `rvPRS --help`
A standard invocation is:

```sh
rvPRS \
  --rare-results RARE_RESULTS \
  --score-type SCORE_TYPE \
  --exome-db EXOME_DB \
  --pheno-db PHENO_DB \
  --train-samples TRAIN_SAMPLES \
  --test-samples TEST_SAMPLES \
  --ancestry-samples ANCESTRY_SAMPLES \
  --trait TRAIT \
  --gwas-db GWAS_DB \
  --gencode GENCODE \
  --max-p MAX_P \
  --output OUTPUT \
  --output-model OUTPUT_MODEL
```

### Arguments explained
 - RARE_RESULTS: path to rare variant results. This should be a tsv file with these 
    columns: 
    - symbol - HGNC symbol
    - consequence - 'del', syn' or 'ptv'
    - beta: average effect size for rare variants in gene
    - p_value: P-value from rare variant burden testing for the gene/consequence type
    - ac_threshold: maximum allele count threshold for including rare variants 
        (based on ukb450k release, this is the global AC)
    - pathogenicity_threshold: threshold for missense pathogenicity score (doesn't
        have to be PrimateAI-3D)
 - EXOME_DB: path to sqlite db of exome genotypes. This needs three tables: 
    "variants", "annotations" and "samples". See `rvPRS.rare.exome.py` for
    further details.
 - SCORE_TYPE: name of missense pathogenicity score. Needs to be a valid column 
    in the annotations table of the exome database.
 - PHENO_DB: path to sqlite db of phenotypes. See `rvPRS.rare.phenotype.py` for 
    further details.
 - TRAIN_SAMPLES: path to list of IDs for samples used during burden testing. This 
    is required as we have to remodel per-variant effects in the same subset.
 - TEST_SAMPLES: path to list of test samples to generate polygenic scores for. 
    This must be disjoint from the training samples.
 - ANCESTRY_SAMPLES: path to list of samples for checking variant allele
    frequencies. This can (and should!) be used multiple times to check in 
    multiple ancestries.
 - TRAIT: name of trait to check (for selecting from phenotype db)
 - GWAS_DB: path to sqlite db of GWAS results, for optionally regressing out 
    common variant effects. See `rvPRS.common.db.py` for further details.
 - GENCODE: path to GENCODE GTF annotations file (e.g. https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.annotation.gtf.gz)
 - MAX_P: maximum allowed P-value for inclusion
 - OUTPUT: where to save table of scores per individual.
 - OUTPUT_MODEL: optional path to save model data in json format.
