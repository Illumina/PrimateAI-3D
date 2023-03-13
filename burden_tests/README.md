### Covariate correction 
Covariate correction was performed with correct_ukb_phenotype_for_covariates.py

### GWAS finemapping

Candidate causal genes were derived from GWAS analysis of UK Biobank phenotypes performed within our work with finemap_UKB_GWAS_SNPs.py

In addition to that, candidate causal genes were derived for all GWAS index variants from the GWAS Catalog with finemap_GWAS_SNPs_v2.py

### Rare variant analysis

Burden tests were performed with test_ukb_phenotype_for_association.py

Individual variant effects were computed with test_each_variant_for_association.py after regressing out common GWAS variants with correct_phenotype_for_common_gwas_variants.py
