# Covariate correction 
Covariate correction was performed with correct_ukb_phenotype_for_covariates.py

# GWAS finemapping

Candidate causal genes were derived from GWAS analysis of UK Biobank phenotypes performed within our work with finemap_UKB_GWAS_SNPs.py

In addition to that, candidate causal genes were derived for all GWAS index variants from the GWAS Catalog with finemap_GWAS_SNPs_v2.py

# Rare variant analysis

Individual variant effects were computed with test_each_variant_for_association.py after regressing out common GWAS variants with correct_phenotype_for_common_gwas_variants.py

## Burden tests
Burden tests were performed with test_ukb_phenotype_for_association.py

### Column descriptions
For each phenotype, we perform burden tests for different groups of variant types with the following labels:

del - deleterious variants: the union of all protein-truncating variants (PTVs, defined below), pathogenic missense variants with pathogenicity score greater than a gene-specific threshold, and variants with SpliceAI score greater than 0.2

ptv - protein-truncating variants: SpliceAI score > 0.2 or consequence matching any of the tags: ['stop_gained', 'stop_lost', 'frameshift_variant', 'splice_donor_variant', 'splice_acceptor_variant', 'start_lost', 'transcript_ablation', 'transcript_truncation', 'exon_loss_variant', 'gene_fusion', 'bidirectional_gene_fusion']

missense_all - all missense variants regardless of their pathogenicity

missense_pAI_optimized - only pathogenic missense variants with pathogenicity score greater than a gene-specific threshold

missenses_and_ptvs_all - the union of all PTVs, spliceAI > 0.2 variants and all missense variants regardless of their pathogenicity scores

syn - all synonymous variants


For each variant type, the results table contains the following columns (vtype is one of ['del', 'missense_all', 'missense_pAI_optimized', 'missenses_and_ptvs_all', 'ptv', 'syn']):


[f'ALL/{vtype}/carrier/pvalue/fdr_corr', # P-value for the association of the gene with the phenotype

 f'ALL/{vtype}/carrier/pvalue/global_fdr', # FDR for the association of the gene with the phenotype corrected for testing 20,000 genes

 f'ALL/{vtype}/carrier/beta', # Effect size of being a carrier

 f'ALL/{vtype}/n_variants', # number of variants considered in the burden test

 f'ALL/{vtype}/n_carriers', # number of carriers considered in the burden test

 f'ALL/{vtype}/carrier/stat', # statistic of the burden test

 f'ALL/{vtype}/best_AC', # allele count threshold that maximizes separation between carriers and non-carriers determined by the grid search

 f'ALL/{vtype}/best_pAI_threshold', # pathogenicity score threshold that maximizes separation between carriers and non-carriers determined by the grid search

 f'ALL/{vtype}/pathogenicity_score/carrier_level/regression_beta', # slope of the regression line that predicts phenotype values from pathogenicity scores on carrier level (each individual's phenotype is correlated with the pathogenicity score of the variant they carry)

 f'ALL/{vtype}/pathogenicity_score/carrier_level/spearman_r', # Spearman correlation between pathogenicity scores and phenotype values on carrier level

 f'ALL/{vtype}/pathogenicity_score/carrier_level/spearman_pvalue', # P-value of the Spearman correlation on carrier level

 f'ALL/{vtype}/pathogenicity_score/variant_level/regression_beta', # slope of the regression line that predicts phenotype values from pathogenicity scores on variant level (for each variant, the average phenotyp value among all its carriers is correlated  with the pathogenicity score of the variant)

 f'ALL/{vtype}/pathogenicity_score/variant_level/spearman_pvalue', # Spearman correlation between pathogenicity scores and phenotype values on variant level

 f'ALL/{vtype}/pathogenicity_score/variant_level/spearman_r', # P-value of the Spearman correlation on variant level

 f'ALL/{vtype}/carriers', # sample ids of all carriers of variants considered in the burden test (i.e. variants that pass the thresholds for allele count and pathogenicity scores determined by the grid search)

 f'ALL/{vtype}/variants', # variant ids of all variants considered in the burden test (i.e. variants that pass the thresholds for allele count and pathogenicity scores determined by the grid search)

 f'ALL/{vtype}/all_carriers', # all carriers of all variants considered by the grid search including carriers of variants below the thresholds determined by the grid search

 f'ALL/{vtype}/n_total_carriers', # total number of carriers of all variants considered by the grid search including carriers of variants below the thresholds determined by the grid search

 f'ALL/{vtype}/n_total_variants', # total number of variants considered by the grid search including variants below the thresholds determined by the grid search

 f'ALL/{vtype}/discarded_variants', # variant ids of the variants that were discarded from the burden test because they were below the allele count and pathogenicity thresholds determined by the grid search

 f'ALL/{vtype}/n_discarded_variants', # total number of discarded variants

 f'ALL/{vtype}/carrier/n_tests', # total number of tests performed by the grid search for this gene

 f'ALL/{vtype}/pathogenicity_score_label', # the name of the pathogenicity score used to score missense variants

 f'ALL/{vtype}/test_type', # the type of the burden test

 f'ALL/{vtype}/carrier/pvalue/n_randomizations', # number of permutations generated for this gene to calibrate the raw p-values of the association

 f'ALL/{vtype}/carrier/pvalue', # raw p-value of the difference between carriers and non-carriers for the allele count and pathogenicity thresholds determined by the grid search. This p-value should be only used for debugging purposes, because they are not corrected for multiple testing within the same gene and therefore they are hugely inflated. The column f'ALL/{vtype}/carrier/pvalue/fdr_corr' has the corrected p-values and should be used for downstream analysis.

 f'ALL/{vtype}/carrier/pvalue/BH_fdr_corr', # FDR corrected p-values with standard Benjamini-Hochberg procedure for multiple testing within the same gene. These p-values are deflated, so we do not recommend using them for downstream analysis. The column f'ALL/{vtype}/carrier/pvalue/fdr_corr' has the corrected p-values and should be used for downstream analysis.

 ### The following columns are used for debugging and should be ignored:
 f'ALL/{vtype}/needs_randomization', 

 f'ALL/{vtype}/sort_by']