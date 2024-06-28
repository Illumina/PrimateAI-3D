
import logging
import sqlite3
from typing import Iterable, Set

from rvPRS.rare.exome import variants_for_gene
from rvPRS.rare.variant import Variant

def get_rare_variants(conn: sqlite3.Connection, 
                      symbol: str, 
                      score_type='primateai_v2',
                      max_af=0.001, 
                      exome_samples=None) -> Iterable[Variant]:
    ''' find the rare variants for a gene
    
    Args:
        conn: sqlite3 connection to exome database
        symbol: HGNC symbol for gene
        score_type: name of column for missense pathogenicity score
        max_af: max permitted allele frequency
        exome_samples: set of sample IDs, used for flipping high AF variants 
            and computing max AN
    
    Returns:
        variants which meet the criteria
    '''
    max_AN = len(exome_samples) * 2
    logging.info(f'{symbol}')
    variants = variants_for_gene(conn, symbol, score_col=score_type, max_af=max_af)
    variants = filter_by_af(variants, max_af)
    variants = filter_by_af(variants, max_af, field='gnomad_af')
    variants = filter_by_missingness(variants, max_AN)
    variants = flip_high_af_variants(variants, exome_samples)
    
    return variants

def filter_by_af(variants: Iterable[Variant], threshold=0.001, field='af') -> Iterable[Variant]:
    ''' remove common variants
    '''
    logging.debug(f'removing variants with AF > {threshold}')
    return (x for x in variants if (x[field] is None or x[field] <= threshold) or (x[field] >= (1 - threshold)))

def filter_by_missingness(variants: Iterable[Variant], size, threshold=0.05) -> Iterable[Variant]:
    ''' remove variants with many missing samples
    '''
    logging.debug(f'removing variants with missingness > {threshold}')
    min_an = (1 - threshold) * size
    return (x for x in variants if x.an >= min_an)

def filter_by_ac(variants: Iterable[Variant], threshold: int) -> Iterable[Variant]:
    ''' remove variants with high allele counts
    '''
    logging.debug(f'removing variants with AC > {threshold}')
    return (x for x in variants if x.ac <= threshold)

def flip_high_af_variants(variants: Iterable[Variant], sample_ids: Set[int]) -> Iterable[Variant]:
    ''' swaps alleles for variants where the alternate allele frequency is >0.5
    
    variants with af > 0.5 require a bit of time to adjust sample IDs, so should 
    only be run after excluding variants with AF > 0.001 and AF < 0.999
    '''
    for var in variants:
        if var.af > 0.5:
            var.flip_alleles(sample_ids)
        yield var
