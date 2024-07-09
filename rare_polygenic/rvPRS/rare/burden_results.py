
from dataclasses import dataclass
import gzip
from itertools import groupby
import logging
from pathlib import Path
from typing import Dict, Iterable

from gencodegenes import Gencode

from rvPRS.rare.filter_genes import filter_by_gencode

@dataclass
class Result:
    symbol: str
    consequence: str
    beta: float
    p_value: float
    ac_threshold: float
    pathogenicity_threshold: float
    af_threshold: float = None
    chrom: str = None
    pos: int = None
    tss_pos: int = None
    
    def __post_init__(self):
        # make sure the relevant attributes are floats
        for field in ['beta', 'p_value', 'ac_threshold', 'pathogenicity_threshold']:
            setattr(self, field, float(getattr(self, field)))
    
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        # assign chrom, pos and tss_pos after result line is loaded from disk
        setattr(self, key, value)

def get_lines(path: Path) -> Iterable[Result]:
    ''' get rare variant result rows
    '''
    logging.info(f'opening rare variant results from {path}')
    opener = gzip.open if str(path).endswith('gz') else open
    with opener(path, 'rt') as handle:
        header = handle.readline().strip('\n').split('\t')
        header = {k :i for i, k in enumerate(header)}
        for line in handle:
            line = line.strip('\n').split('\t')
            symbol = line[header['symbol']]
            consequence = line[header['consequence']]
            beta = line[header['beta']]
            p_value = line[header['p_value']]
            ac_threshold = line[header['ac_threshold']]
            try:
                pathogenicity_threshold = line[header['pathogenicity_threshold']]
            except KeyError:
                # fix for older column name
                pathogenicity_threshold = line[header['primate_ai_threshold']]
            yield Result(symbol, consequence, beta, p_value, ac_threshold, pathogenicity_threshold)

def group_by_gene(results: Iterable[Result]) -> Iterable[Dict[str, Result]]:
    ''' group result lines by gene (come sorted in results file)
    '''
    for gene, group in groupby(results, key=lambda x: x.symbol):
        yield {x.consequence: x for x in group}

def filter_by_p_value(genes: Iterable[Dict[str, Result]], threshold=1, cq='del') -> Iterable[Dict[str, Result]]:
    ''' filter genes by p-value (includes everything by default)
    
    Args:
        cq: consequence to filter on by p-value
        threshold: max p-value permitted. Includes everything by default
    '''
    for gene in genes:
        if gene[cq].p_value <= threshold:
            yield gene

def get_gene_coords(gencode: Gencode, symbol: str):
    ''' get the chrom, pos (middle of gene) and transcript start site for gene
    '''
    tx = gencode[symbol].canonical
    chrom = tx.chrom
    mid_pos = (tx.start + tx.end) // 2
    tss_pos = tx.cds_start if tx.strand == '+' else tx.end
    return {'chrom': chrom, 'pos': mid_pos, 'tss_pos': tss_pos}

def annotate_coords(genes: Iterable[Dict[str, Result]], gencode: Gencode) -> Iterable[Dict[str, Result]]:
    ''' annotate each gene result with chrom, pos and tss_pos via gencode
    '''
    for result in genes:
        symbol = result['del'].symbol
        coords = get_gene_coords(gencode, symbol)
        for cq in result:
            for k, v in coords.items():
                result[cq][k] = v
        yield result

def open_rare_variant_results(path: Path, gencode: Gencode, threshold=1, cq='del', symbols=None):
    ''' open results from rare variant tests
    
    This restricts to protein-coding genes only, and deduplicates overlapping
    genes by picking the most significant only
    
    Args:
        path: path to results file
        gencode: gencodegenes object
        cq: consequence to filter on by p-value
        threshold: max p-value permitted. Includes everything by default
        symbols: if not None, restrict results with these HGNC symbols
    
    Yields:
        dict of results per gene, indexed by consequence type
    '''
    results = get_lines(path)
    genes = group_by_gene(results)
    genes = filter_by_p_value(genes, threshold, cq)
    if symbols:
        genes = (x for x in genes if x['del'].symbol in symbols)
    genes = filter_by_gencode(gencode, genes)
    return annotate_coords(genes, gencode)
