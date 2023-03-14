
from itertools import chain
import logging
from pathlib import Path
import sqlite3
import time

import numpy
from regressor import linregress

from rvPRS.common.db import open_bgens
from rvPRS.common.polygenic_risk import sanatize_chrom

def open_ukb_ld(folder):
    ''' open the per-chromosome dataabses of LD results for UKBiobank
    
    Args:
        folder: folder containing UKB LD sqlite databases, e.g ukb_ld.*chr1.db
    
    Returns:
        dict of sqlte3 connections, indexed by chromosome
    '''
    ld = {}
    folder = Path(folder)
    for path in folder.glob('ukb_ld*.db'):
        chrom = path.stem.split('.')[-1]
        ld[chrom] = sqlite3.connect(path)
    return ld

def in_ld(var1, var2, ld, threshold=0.1):
    ''' check if two variants are in strong LD
    
    Args:
        var1: first Variant
        var2: second Variant
        ld: dict of sqlite databases, indexed by chromosome (e.g. chr1)
        threshold: LD r2 threshold for classifying high LD
    
    Returns:
        boolean, indicating if the two vaiants are in strong LD
    '''
    db = ld[f'chr{var1.chrom.lstrip("0")}']
    query = '''SELECT r2 FROM ld WHERE pos_1=? AND pos_2=?'''
    
    res = chain(db.execute(query, (var1.pos, var2.pos)), db.execute(query, (var2.pos, var1.pos)))
    for x in res:
        # TODO: we should also check varids to ensure we use the correct variants
        try:
            if x[0] > threshold:
                return True
        except TypeError:
            continue
    
    # pairs not present in the db do not have high LD
    return False

def ld_prune(loci, ld, window=10000000):
    ''' exclude loci in strong LD with more strongly associated loci
    
    The loci have previously been sorted by p-value, so if there is LD, 
    excluding the latter variant will keep the strongest signals.
    
    Args:
        loci: list of Variants
        ld: dict of sqlite databases, indexed by chromosome (e.g. chr1)
        window: max distance (in basepairs) to check LD out to. Default is 10 Mb,
            as common variants more than 10 Mb distant have very low LD. This
            avoids a lot of LD lookups. I checked one trait out to ~700 peak
            variants, and this did not change any variant output, but sped up
            processing 5X.
    
    Yields:
        Variants where pairs in strong LD have been trimmed
    '''
    passing = {}
    for locus in loci:
        if locus.chrom not in passing:
            passing[locus.chrom] = []
        
        # only check LD for variants on same chrom, and within window diastnce
        nearby = (x for x in passing[locus.chrom] if abs(locus.pos - x.pos) < window)
        if any(in_ld(locus, x, ld) for x in nearby):
            continue
        
        passing[locus.chrom].append(locus)
        
        yield locus

def get_variant(bgens, site):
    ''' extract a single variant from the BGENs
    '''
    chrom = sanatize_chrom(site.chrom)
    try:
        logging.debug(f'getting variant for LD at {chrom}:{site.pos}')
        return bgens[chrom].at_position(site.pos)
    except ValueError:
        logging.debug(f'fetching variants for LD at {chrom}:{site.pos}')
        for var in bgens[chrom].fetch(site.chrom, site.pos-1, site.pos):
            if var.pos == site.pos and set(var.alleles) == set([site.minor, site.major]):
                return var
    
    raise ValueError(f'cannot find {site.chrom}:{site.pos} [{site.major}, {site.minor}]')

def drop_nans(first, second):
    ''' drop samples if either variant has a missing genotype
    '''
    not_nan = ~(numpy.isnan(first) | numpy.isnan(second))
    return first[not_nan], second[not_nan]

def ld_prune_from_genotype(loci, bgens, window=10000000, gwas_db_path=None):
    ''' LD prune GWAS peak variants directly from genotype
    
    Args:
        loci: list of Variant objects, with chrom and pos attributes
        bgens: dict of chrom to BgenFile objects (for retrieving variant data)
        window: max distance (in basepairs) to check LD out to. Default is 10 Mb.
    '''
    
    passing = {}
    for locus in loci:
        chrom = locus.chrom
        if chrom not in passing:
            passing[chrom] = []
        
        while True:
            try:
                var = get_variant(bgens, locus)
                dose = var.minor_allele_dosage[:]
                break
            except ValueError:
                logging.debug(f'error obtaining dosage at {var.chrom}:{var.pos}')
                time.sleep(10)
                bgens = open_bgens(gwas_db_path)
        
        nearby = [x for x in passing[chrom] if abs(locus.pos - x['var'].pos) < window]
        r2 = [linregress(*drop_nans(dose, x['dose'])).rvalue ** 2 for x in nearby]
        
        if any(x > 0.1 for x in r2):
            continue
        
        passing[chrom].append({'var': locus, 'dose': dose.copy()})
        yield locus
