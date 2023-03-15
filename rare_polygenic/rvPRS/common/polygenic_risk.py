
def sanatize_chrom(chrom):
    ''' make sure the chromosome matches the format found in the UKBiobank BGENs
    '''
    chrom = str(chrom).strip('chr')
    try:
        chrom = int(chrom)
        chrom = f'{chrom}'.zfill(2)
    except ValueError:
        pass
    
    if chrom == '23':
        chrom = 'X'
    
    return chrom

def get_peaks(variants, threshold=5e-5, window=500000, is_sorted=False):
    ''' find peak variants within clusters
    
    Args:
        variants: list of Variant objects with chrom, pos, major, minor, beta, and
            p_value attributes
    
    Yields:
        independent locuses
    '''
    
    variants = (x for x in variants if x.p_value < threshold)
    
    if not is_sorted:
        variants = sorted(variants, key=lambda x: x.p_value)
    
    clusters = {}
    for variant in variants:
        if variant.chrom not in clusters:
            clusters[variant.chrom] = []
        
        # if the variant is near an identified cluster, move to the next variant
        if any(abs(variant.pos - x) < window for x in clusters[variant.chrom]):
            continue
        
        clusters[variant.chrom].append(variant.pos)
        yield variant
