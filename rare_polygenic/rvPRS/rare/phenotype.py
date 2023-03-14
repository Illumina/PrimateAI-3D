''' function to get phenotype data from sqlite db for rare variant analysis
'''

import logging
import sqlite3

def get_phenotypes(db_path, column, samples=None):
    ''' get values for one phenotype from phenotype database
    
    Args:
        db_path: path to sqlite3 database containing phenotype info
        column: name of phenotype to get e.g. 'Standing_height.all_ethnicities.both.original.RAW'
        samples: optional list of samples to restrict to. Uses all samples if not set
    
    Yields:
        tuples of (sample_id, value). 
    '''
    logging.info(f'getting {column} from phenotype database')
    if samples is not None:
        samples = set(samples)
    make_query = lambda col: f'''SELECT value from "{col}"'''
    with sqlite3.connect(db_path) as conn:
        all_samples = conn.execute(make_query('sample_id'))
        for k, v in zip(all_samples, conn.execute(make_query(column))):
            if samples is None or k[0] in samples:
                yield k[0], v[0]
    conn.close()

def get_covariates(db_path, covariates, samples=None):
    ''' get covariate columns from covariate database
    '''
    logging.info(f'getting {covariates} from phenotype database')
    if samples is not None:
        samples = set(samples)
    
    if isinstance(covariates, str):
        covariates = [covariates]
    covariates = [f'"{x}"' for x in covariates]
    
    make_query = lambda col: f'''SELECT sample_id,{",".join(covariates)} from covariates'''
    with sqlite3.connect(db_path) as conn:
        for x in conn.execute(make_query(covariates)):
            if samples is None or x[0] in samples:
                yield x
