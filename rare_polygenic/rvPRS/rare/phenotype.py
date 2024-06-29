''' function to load phenotype data for rare variant analysis
'''

import gzip
from io import TextIOWrapper
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

def open_file(path: Path, sep='\t') -> Tuple[Dict[str, int], TextIOWrapper]:
    ''' open a csv/tsv file and return header dict plus file handle
    '''
    opener = gzip.open if str(path).endswith('gz') else open
    handle = opener(path, 'rt')
    header = handle.readline().strip('\n').split(sep)
    indices = {k: i for i, k in enumerate(header)}
    return indices, handle

def get_phenotypes(path: Path, column: str, samples=None) -> Iterable[Tuple[str, float]]:
    ''' get values for one phenotype from phenotype database
    
    Args:
        path: path to file containing phenotype info
        column: name of phenotype column to use e.g. 'Standing_height.all_ethnicities.both.original.RAW'
        samples: optional list of samples to restrict to. Uses all samples if not set
    
    Yields:
        tuples of (sample_id, value). 
    '''
    logging.info(f'loading phenotypes from {path} via {column}')
    if samples is not None:
        samples = set(samples)
    
    indices, handle = open_file(path)
    sample_col = indices['sample_id']
    value_col = indices[column]
    
    for line in handle:
        line = line.strip('\n').split('\t')
        sample_id = line[sample_col]
        value = line[value_col]
        
        try:
            value = float(value)
        except ValueError:
            continue
        
        if math.isnan(value):
            continue
        
        if samples is None or sample_id in samples:
            yield sample_id, value
