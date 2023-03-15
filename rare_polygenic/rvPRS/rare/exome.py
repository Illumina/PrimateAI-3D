
import logging

from rvPRS.rare.variant import Variant

def unique_genes(conn):
    ''' get unique gene symbols
    
    Args:
        conn: sqlite3 connection
    
    Yields:
        unique HGNC symbols
    '''
    logging.info('finding unique gene symbols')
    query = '''SELECT DISTINCT symbol FROM annotations'''
    for symbol in conn.execute(query):
        yield symbol[0]

def variants_for_gene(conn, symbol, score_col='primateai', max_af=0.001):
    ''' pull rows matching a single gene symbol
    
    Args:
        conn: sqlite3 connection
        symbol: hgnc symbol to extract variants for
    '''
    logging.debug(f'getting variants for {symbol}')
    query = f'''SELECT 
                    v.varid, v.rsid, v.chrom, v.pos, v.ref, v.alt, a.symbol, 
                    a.consequence, a.{score_col}, a.spliceai, v.af, v.ac, v.an, 
                    a.gnomad_af, a.topmed_af, v.missing, v.homs, v.hets
                FROM variants v
                INNER JOIN annotations a ON v.rowid = a.varid
                WHERE a.symbol=? AND (v.af<=? OR v.af>=?)'''
    params = (symbol, max_af, 1-max_af)
    for var in conn.execute(query, params):
        yield Variant(*var)

def get_exome_samples(conn):
    ''' find sample IDs for samples with exome data
    '''
    query = '''SELECT * from samples'''
    for x in conn.execute(query):
        yield x[0]

def count_exome_samples(conn):
    ''' count the number of exome samples
    '''
    query = '''SELECT count(rowid) from samples'''
    return conn.execute(query).fetchone()[0]
