
def in_gencode(gencode, symbol):
    ''' check if a gene symbol is present in gencode
    
    This assumes gene symbol annotations came from gencode, and uses the same 
    version for the rare variant burden test as for the gencode loaded here.
    
    Args:
        gencode: gencodegenes object for all genes in gencode
        symbol: HGNC symbol for gene
    '''
    return symbol in gencode

def is_coding(gencode, symbol):
    ''' check if the canonical transcript of a gene is protein-coding
    
    Args:
        gencode: gencodegenes object for all genes in gencode
        symbol: HGNC symbol for gene
    '''
    return gencode[symbol].canonical.type == 'protein_coding'

def cds_overlap(tx1, tx2, window=50):
    ''' check if the CDS regions of two transcripts overlap
    
    We expand each CDS regions by 50 bp on each side by default, in order to
    catch genes which might be significant due to hitting the same intronic 
    splice variants. This seems unlikely, but adding the extra window didn't
    cull any true hits in the small checks I did.
    
    Args:
        tx1: gencodegenes.Transcript object, with get_cds() method
        tx2: gencodegenes.Transcript object, with get_cds() method
        window: distance to expand CDS regions by
    
    Returns:
        true/false for whether any of the CDS regions overlap between transcripts
    '''
    tx1_cds = [(x['start'] - window, x['end'] + window) for x in tx1.cds]
    tx2_cds = [(x['start'] - window, x['end'] + window) for x in tx2.cds]
    for start1, end1 in tx1_cds:
        for start2, end2 in tx2_cds:
            if end1 >= start2 and end2 >= start1:
                return True
    return False

def filter_overlapping(gencode, rare_genes):
    ''' drop genes which overlap more significant genes
    
    This deduplicates overlapping genes by picking the most significant only.
    '''
    # sort rare genes by p-value, so we can then drop the less signifcant 
    # overlapping hits
    rare_genes = sorted(rare_genes, key=lambda x: x['del'].p_value)
    included = set()
    for rare in rare_genes:
        gene = gencode[rare['del'].symbol]
        in_region = gencode.in_region(gene.chrom, gene.start, gene.end)
        intersected = set([x for x in in_region if x.symbol in included])
        if any(cds_overlap(gene.canonical, x.canonical) for x in intersected):
            continue
        else:
            included.add(rare['del'].symbol)
            yield rare

def filter_by_gencode(gencode, rare_genes):
    ''' drop rare genes if they are not protein coding, or overlap a more significant gene
    
    Args:
        gencode: gencodegenes object for all genes in gencode
        rare_genes: iterable of dicts, where each dict contains 'del', and the 
            'del' entry has a symbol attribute
    
    Returns:
        rare genes filtered by gencode for protein coding, and non-overlapping genes
    '''
    # drop genes not found in gencode or without protein-coding transcripts
    rare_genes = (x for x in rare_genes if in_gencode(gencode, x['del'].symbol))
    rare_genes = (x for x in rare_genes if is_coding(gencode, x['del'].symbol))
    return filter_overlapping(gencode, rare_genes)
