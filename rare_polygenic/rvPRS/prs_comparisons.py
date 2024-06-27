
import gzip
import logging

def open_sample_subset(path, sep='\t'):
    ''' open file containing sample IDs to subset by
    '''
    # define set of header values (by priority). We use these to check if file
    # has a header line (otherwise we include the value from the first line), 
    # and which column to use (if header exists).
    id_cols = ['sample_id', 'person_id', 'IID', 'FID']
    logging.info(f'opening sample list from {path}')
    opener = gzip.open if path.endswith('gz') else open
    with opener(path, 'rt') as handle:
        header = handle.readline().strip('\n#').split(sep)
        
        sample_ids = set()
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
