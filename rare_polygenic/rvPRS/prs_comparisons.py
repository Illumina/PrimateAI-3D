
import gzip

def open_test_samples(path, sep='\t'):
    ''' open sample IDs for test samples (encoded as set of ints)
    '''
    opener = gzip.open if str(path).endswith('gz') else open
    with opener(path, 'rt') as handle:
        _ = handle.readline()
        lines = handle.read().strip().split('\n')
        if lines == ['']:
            return set()
        first = lines[0].strip('\n').split(sep)[0]
        first = set() if first == 'sample_id' else set([int(first)])
        return set(int(x.strip('\n').split(sep)[0]) for x in lines[1:]) | first
