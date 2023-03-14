
import logging
import sys
import time

import numpy
from sklearn.linear_model import LinearRegression

from rvPRS.common.db import open_gwas_results, open_bgens
from rvPRS.common.polygenic_risk import get_peaks, sanatize_chrom
from rvPRS.common.ld import ld_prune_from_genotype

def get_genotype_indices(sample_ids_pheno, sample_ids_geno):
    ''' find the indices for each phenotyped sample in the genotype arrays
    
    We need to shift the genotype data to the same order as the phenotype data. 
    We do this by getting the indices of all samples in the genotype arrays, 
    then working through the pheno samples to find where we should fetch the
    genotype value from. This is cached at a later stage so we only compute this 
    once per chromosome bgen, since the sample IDs for genotypes are the same
    within a bgen file.
    
    Args:
        sample_ids_pheno: list of sample IDs matching the phenotype data
        sample_id_geno: list of sample IDs matching the genotype data
    
    Returns:
        list with same length as sample_ids_pheno, where each value holds the index
        position within the genotype data
    '''
    geno_samples = {k: i for i, k in enumerate(sample_ids_geno)}
    
    indices = []
    for sample_id in sample_ids_pheno:
        if sample_id not in geno_samples:
            indices.append(None)
        else:
            indices.append(geno_samples[sample_id])
    return indices

def align_dosage_to_phenotype(indices, dosage):
    ''' make sure the order of genotypes matches the order of phenotyped samples
    
    NOTE: this isn't efficient, but should only run on a few variants (< 1000?).
    NOTE: A better way would be to group by chromosome and realign en masse.
    
    Args:
        indices: list of indices to fetch genotype value matching a phenotype
        dosage: numpy array of genotype dosage values
    
    Returns:
        numpy array of dosages matching phenotype order
    '''
    # create array with nans for ebvery sample with a phenotype
    aligned = numpy.empty(len(indices), dtype=numpy.float32)
    aligned[:] = numpy.nan
   
    # fill in the dosages per sample
    for i, idx in enumerate(indices):
        if idx is not None:
            aligned[i] = dosage[idx]
    
    return aligned

def regress_out_common_variants(phenotypes, gwas_db_path):
    ''' regress out the effect of common variants on the phenotype
    
    NOTE: Samples with missing genotypes are dropped, since there are few of those. 
    NOTE: We could assign mean genotype for missing samples if this is a problem.
    
    Args:
        phenotypes: iterable of (sample_id, pheno_value) tuples
        gwas_db_path: path to db with results of GWAS run on same phenotype
    
    Returns:
        phenotypes formatted the same as input argument: iterable with tuples 
        of (sample ID, pheno_residual).
    '''
    logging.info(f'regressing out common variants from GWAS: {gwas_db_path}')
    bgens = open_bgens(gwas_db_path)
    variants = open_gwas_results(gwas_db_path, threshold=1e-8)
    
    sample_ids, phenotype = list(zip(*phenotypes))
    phenotype = numpy.array(phenotype)
    sample_ids = numpy.array(sample_ids)
    
    indices = {}
    dosages = []
    for var in ld_prune_from_genotype(get_peaks(variants), bgens, gwas_db_path=gwas_db_path):
        logging.info(f'getting doses for {var.chrom}:{var.pos}, GWAS p={var.p_value:.3g}')
        bfile = bgens[sanatize_chrom(var.chrom)]
        try:
            var = bfile.at_position(var.pos)
        except ValueError:
            continue
        if var.chrom not in indices:
            indices[var.chrom] = get_genotype_indices(sample_ids, bfile.samples)
        
        while True:
            chrom, pos = var.chrom, var.pos
            try:
                dose = align_dosage_to_phenotype(indices[chrom], var.minor_allele_dosage)
                break
            except ValueError:
                logging.debug(f'error obtaining dosage at {var.chrom}:{var.pos}')
                time.sleep(10)
                bgens = open_bgens(gwas_db_path)
                bfile = bgens[sanatize_chrom(chrom)]
                var = bfile.at_position(pos)
        dosages.append(dose)
    
    if len(dosages) == 0:
        return zip(sample_ids, phenotype)
    
    # concatenate dosages, and exclude samples with missing genotypes
    dosages = numpy.column_stack(dosages)
    present = numpy.isnan(dosages).sum(axis=1) == 0
    dosages = dosages[present, :]
    phenotype = phenotype[present]
    sample_ids = sample_ids[present]
    
    lost = len(present) - present.sum()
    lost_pct = (lost / len(present)) * 100
    logging.info(f'Fitting linear regression. {lost} of {len(present)} ' \
        f'({lost_pct:.3g}%) were lost with missing genotypes.')
    fit = LinearRegression(fit_intercept=True).fit(dosages, phenotype)
    residuals = phenotype - fit.predict(dosages)
    logging.info(f'resid vs original, r2: {numpy.corrcoef(phenotype, residuals)[0, 1] ** 2:.3g}')
    
    return zip(sample_ids, residuals)
