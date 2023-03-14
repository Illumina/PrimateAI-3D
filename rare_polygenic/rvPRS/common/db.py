
import logging
import sqlite3
from collections import namedtuple

from bgen.reader import BgenFile

Variant = namedtuple('Variant', ['varid', 'chrom', 'pos', 'major', 'minor', 'beta', 'p_value'])

def open_gwas_results(db_path, threshold=1e-4):
    ''' get GWAS results per variant, sorted by p-value
    
    Args:
        db_path: path to sqlite database of GWAS results, with a table 'variants'
                 that has columns for varid, chrom, pos, major allele (major),
                 minor allele (minor), effect size (beta) and p_value.
        threshold: p-value threshold to stop returning variants after
    
    Yields:
        variants as namedtuples, with chrom, pos, major, minor, beta and p-value
        attributes.
    '''
    logging.info(f'selecting variants from {db_path}, with P < {threshold}')
    db = sqlite3.connect(db_path)
    query = '''SELECT varid, chrom, pos, major, minor, beta, p_value 
                FROM variants
                WHERE p_value < ?
                ORDER BY p_value
            '''
    for res in db.execute(query, (threshold, )):
        yield Variant(*res)

def open_bgens(db_path):
    ''' open bgen files for UKBiobank imputed genotypes
    '''
    query = '''SELECT bgen_path, sample_path from filepaths'''
    chroms = {}
    with sqlite3.connect(db_path) as conn:
        for (bgen_path, sample_path) in conn.execute(query):
            bfile = BgenFile(bgen_path, sample_path)
            chroms[bfile[0].chrom] = bfile
    return chroms
