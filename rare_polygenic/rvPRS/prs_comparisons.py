
import gzip
import logging
from pathlib import Path
from typing import Set

def open_sample_subset(path: Path, sep='\t') -> Set[str]:
    ''' open file containing sample IDs to subset by
    '''
    # define set of header values (by priority). We use these to check if file
    # has a header line (otherwise we include the value from the first line), 
    # and which column to use (if header exists).
    id_cols = ['sample_id', 'person_id', 'IID', 'FID']
    logging.info(f'opening sample list from {path}')
    opener = gzip.open if path.endswith('gz') else open
    sample_ids = set()
    with opener(path, 'rt') as handle:
        # allow for empty files
        header = ''
        for header in handle:
            # cope with empty lines at the start
            header = header.strip('\n#')
            if header != '':
                break
        
        if header != '':
            header = header.split(sep)
            idx = 0
            if len(set(header) & set(id_cols)) > 0:
                for id_col in id_cols:
                    if id_col in header:
                        idx = header.index(id_col)
            else:
                sample_ids.add(header(idx))
            
            for line in handle:
                line = line.strip('\n').split(sep)
                sample_ids.add(line[idx])
            
    return sample_ids
