

import argparse
from dataclasses import dataclass
from itertools import groupby
import gzip
import json
import logging
import math
from pathlib import Path
import sqlite3
import sys
from typing import Dict, Iterable, Set

import numpy
from fisher import pvalue as fisher_exact
from gencodegenes import Gencode

from rvPRS.rare.variant import Variant
from rvPRS.rare.consequence import group_by_consequence
from rvPRS.prs_comparisons import open_sample_subset
from rvPRS.rare.exome import get_exome_samples
from rvPRS.rare.filter_variants import get_rare_variants
from rvPRS.rare.filter_genes import filter_by_gencode
from rvPRS.rare.phenotype import get_phenotypes
from rvPRS.rare.regress_out_common import regress_out_common_variants
from rvPRS.rare.per_variant_effects import (fit_gene, fit_genes,
    freq_model_predict, get_max_values_for_carriers, MockFit)

def get_options():
    parser = argparse.ArgumentParser(description='compute rare variant PRS')
    
    # define common options
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('--exome-db', required=True,
        help='path to sqlite db of exome genotypes')
    parent.add_argument('--test-samples',
        help='path to list of test samples to generate polygenic scores for. ' \
            'This must be disjoint from the training samples.')
    parent.add_argument('--ancestry-samples', nargs='*',
        help='path to list of samples for checking variant allele frequencies. ' \
            'This should be used multiple times to check in multiple ancestries.')
    parent.add_argument('--score-type')
    parent.add_argument('--include-metadata',
                        help='additional info that can be placed in output files')
    parent.add_argument('--output')
    
    # add options for using a pretrained model
    subparsers = parser.add_subparsers(required=True)
    group = subparsers.add_parser('pretrained', parents=[parent],
                                  help='compute rare variant PRS using a pretrained model with per-variant data',
                                  description='compute rare variant PRS using a pretrained model with per-variant data')
    group.add_argument('--rare-model', required=True,
                        help='path to json file with per-variant model data for the genes')
    
    # add options for training a model from scratch
    group = subparsers.add_parser('train', parents=[parent],
                                  help='train and compute rare variant PRS',
                                  description='train and compute rare variant PRS')
    group.add_argument('--train-samples', required=True,
        help='path to list of training samples used during burden testing. ' \
            'This is required as we have to remodel per-variant effects in the ' \
            'same subset.')
    group.add_argument('--gencode', required=True,
        help='path to GENCODE GTF annotations file')
    group.add_argument('--phenotype', required=True,
                        help='path/s to files containing phenotypes to test')
    group.add_argument('--phenotype-column', required=True, default='value_irnt',
                        help='name of column to use for phenotype values')
    group.add_argument('--gwas-db', 
        help='optional path to sqlite db of GWAS results, for regressing out ' \
             'common variant effects')
    group.add_argument('--max-p', default=0.01, type=float,
        help='maximum allowed P-value for inclusion')
    group.add_argument('--restrict-to', 
        help='Optional path for restricting the PRS to a subset of genes.')
    group.add_argument('--rare-results', required=True)
    group.add_argument('--output-model')
    
    return parser.parse_args()

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
            pathogenicity_threshold = line[header['pathogenicity_threshold']]
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

def open_gene_subset(path=None):
    ''' open list of gene symbols, for restricting the PRS to
    '''
    if path is None:
        return None
    
    with open(path, 'rt') as handle:
        return set(x.strip() for x in handle)

def save_model(gene_models, model_path):
    models = {}
    for gene, model in gene_models:
        if 'pathogenicity' in model.labels:
            pathogenicity = float(model.beta[model.labels.index('pathogenicity')])
        else:
            pathogenicity = 0
        models[gene['del'].symbol] = { 
            'ac_threshold': gene['del'].ac_threshold, 
            'af_threshold': gene['del'].af_threshold, 
            'pathogenicity_threshold': gene['del'].pathogenicity_threshold,
            'af_effect': float(model.beta[model.labels.index('af')]),
            'pathogenicity_effect': pathogenicity,
            'intercept': float(model.beta[model.labels.index('intercept')]),
            'gene_beta': gene['del'].beta
            }
    
    with open(model_path, 'wt') as handle:
        json.dump(models, handle, indent=4)

def select_variants(conn: sqlite3.Connection, gene: Dict, cq_type: str, 
                    score_type: str, max_af: float, exome_samples: Set[int]) -> Iterable[Variant]:
    ''' find variants for a gene for rare variant PRS
    
    Args:
        conn: sqlite3.Connection to exome database
        gene: dict with per consequence objects with variant filtering criteria
        cq_type: variant consequence type (del, ptv or syn)
        score_type: name of column for missense pathogenicity
        max_af: maximum allele frequence permitted
        exome_samples: set of samples IDs with exome data available (for total N)
    '''
    variants = get_rare_variants(conn, gene[cq_type].symbol, score_type, max_af, exome_samples)
    # variants = filter_by_ac(variants, gene[cq_type].ac_threshold)
    by_cq = group_by_consequence(variants)
    return (v for v in by_cq[cq_type] if v.missense_pathogenicity >= gene[cq_type].pathogenicity_threshold)

def get_ac_and_an(variant: Variant, samples: Set[int]):
    ''' get alternate allele count for variant within a set of samples
    '''
    ac = len(set(variant.homs) & samples) * 2 + len(set(variant.hets) & samples)
    an = (len(samples) - len(set(variant.missing) & samples)) * 2
    return ac, an

def get_af(variant: Variant, samples: Set[int]):
    ''' get allele frequency for variant among a set of samples
    '''
    ac, an = get_ac_and_an(variant, samples)
    return ac / an

def af_threshold(variants: Iterable[Variant], samples: Set[int]):
    ''' find highest AF within variants given samples for a population
    '''
    return max(get_af(x, samples) for x in variants)

def ac_threshold(variants: Iterable[Variant], samples: Set[int]):
    ''' find highest AC within variants given samples for a population
    '''
    return max(get_ac_and_an(x, samples)[0] for x in variants)

class prune_by_ancestry_af:
    ''' select variants with low AF within the ancestry specific samples
        
        This either uses a AC threshold (if using the original training set)
        to convert to an AF threshold, or uses the AF threshold converted from
        the original training set.
        
        The class records the converted AF threshold, so we can store it for
        filtering in different datasets.
        '''
    def __init__(self, 
                 variants: Iterable[Variant],
                 sample_ids: Set[int],
                 ancestries: Dict[str, Set[int]],
                 ac_thresh: int=None,
                 af_thresh: float=None):
        '''
        Args:
            variants: list of rare variants (filtered on max_af e.g. 0.001)
            sample_ids: set of sample IDs from either a training set, or from
                some independent group we want to generate a rare PRS for
            ancestries: sets of sample IDs for different ancestry groups, indexed
                by ancestry label
            ac_thresh: allele count threshold to filter for variants for rarity.
                We use this to determine the corresponding allele frequency
                threshold in the set of samples provided by "sample_ids".
            af_threshold: allele frequency threshold for filtering variants by
                rarity. Previously established by this function with the 
                "ac_thresh" argument.
        '''
        if ac_thresh is None and af_thresh is None:
            raise ValueError('must pass in either ac_thresh or af_thresh')
        
        if ac_thresh is not None and af_thresh is not None:
            raise ValueError('must use one of ac_thresh and af_thresh, not both')
        
        if ac_thresh is not None:
            # The burden test used variants and genotypes from a training set, but
            # only stored the maximum AC across variants from all samples(not just training)
            # training )to define rarity for selecting variants.
            
            # This converts the allele count threshold to an allele frequency 
            # threshold, and excludes variants if any ancestry subset is above.
            
            # get the AF threshold for the gene in the sample subset from the
            # variants which pass the AC threshold
            self.variants = [v for v in variants if v.ac <= ac_thresh]
            self.af_thresh = af_threshold(self.variants, sample_ids)
            self.ac_thresh = ac_threshold(self.variants, sample_ids)
        elif af_thresh is not None:
            # the alternative scenario, when we have pre-determined thr correct
            # AF threshold, so filter to those variants onle, then establish the 
            # AC threshold from those
            self.variants = [v for v in self.variants if get_af(v, sample_ids) <= af_thresh]
            self.ac_thresh = ac_threshold(self.variants, sample_ids)
        
        self.sample_ids = sample_ids
        self.ancestries = ancestries
    
    def __iter__(self):
        ''' yields variants which pass the ancestry AF filter
        '''
        for variant in self.variants:
            p_values = {'zzz': 1}
            if self.ancestries is not None:
                for x in self.ancestries:
                    # check if the ancestry AF significantly exceeds the training AF
                    ac, an = get_ac_and_an(variant, self.ancestries[x])
                    pval = fisher_exact(self.ac_thresh, (len(self.sample_ids) * 2) - self.ac_thresh,
                                        ac, an - ac).left_tail
                    p_values[x] = pval

            if min(p_values.values()) < 0.05:
                continue
            # if any ancestry has more than zero counts, yield the variant. We later
            # intersect with the samples we require. we can't simply check if it's
            # nonzero in the training samples, since that would bias toward variants
            # observed in the training group.
            if self.ancestries is not None:
                if any(get_ac_and_an(variant, self.ancestries[x])[0] > 0 for x in self.ancestries):
                    yield variant
            else:
                yield variant

def get_rare_prs_v1(include_samples, rare_genes, exome_db_path, ancestries, 
        cq_type='del', score_type='primateai_v2', max_af=0.001):
    ''' calculate rare polygenic risk scores with same effect for each variant
    
    Args:
        include_samples: set of samples used for the PRS
        rare_genes: list of dicts, each with 'symbol', 'p_value', 'beta', 
                    'ac_threshold' and 'pathogenicity_threshold' fields
        exome_db_path: path to exome database
        cq_type: variant type, one of ['del', 'syn' or 'ptv']
        score_type: name of column for missense pathogenicity
    '''
    with sqlite3.connect(exome_db_path) as conn:
        exome_samples = set(map(int, get_exome_samples(conn)))
    
    conn = sqlite3.connect(exome_db_path)
    risk_scores = {int(x): 0 for x in include_samples}
    for gene in sorted(rare_genes, key=lambda x: float(x[cq_type].p_value)):
        if math.isnan(gene[cq_type].p_value):
            continue
        variants = select_variants(conn, gene, cq_type, score_type, max_af * 10, exome_samples)
        variants = prune_by_ancestry_af(variants, exome_samples, ancestries, ac_threshold=gene[cq_type].ac_threshold)
        for sample in set(x for v in variants for x in v.all_samples):
            if sample in risk_scores:
                risk_scores[sample] += gene[cq_type].beta
    
    return risk_scores

def get_rare_prs_v2(include_samples, rare_genes, exome_db_path, ancestries, 
        pheno_path, pheno_col, train_samples, model_path=None, cq_type='del', 
        score_type='primateai_v2', max_af=0.001, gwas_db=None):
    ''' calculate rare polygenic risk scores with per-variant effects
    
    Args:
        include_samples: set of samples used for the PRS
        rare_genes: list of dicts, each with 'symbol', 'p_value', 'beta', 
                    'ac_threshold' and 'pathogenicity_threshold' fields
        exome_db_path: path to exome database
        pheno_path: path to phenotype file
        pheno_col: name of phenotype column to use from phenotype file
        train_samples: set of samples for training per-variant effects
        model_path: if included, will save per-variant and per-gene model 
            parameters to this path as json output.
        cq_type: variant type, one of ['del', 'syn' or 'ptv']
        score_type: name of column for missense pathogenicity
    '''
    with sqlite3.connect(exome_db_path) as conn:
        exome_samples = set(map(int, get_exome_samples(conn)))
    
    phenotype = get_phenotypes(pheno_path, pheno_col)
    phenotype = ((k, v) for k, v in phenotype if int(k) in exome_samples and int(k) in train_samples)
    phenotype = ((k, v) for k, v in phenotype if v is not None)
    if gwas_db:
        phenotype = regress_out_common_variants(phenotype, gwas_db)
    phenotype = {int(k): v for k, v in phenotype}
    
    models = []
    conn = sqlite3.connect(exome_db_path)
    risk_scores = {int(x): 0 for x in include_samples}
    for gene in sorted(rare_genes, key=lambda x: float(x[cq_type].p_value)):
        if math.isnan(gene[cq_type].p_value):
            continue
        variants = select_variants(conn, gene, cq_type, score_type, max_af * 10, exome_samples)
        variants = prune_by_ancestry_af(variants, train_samples, ancestries, gene[cq_type].ac_threshold)
        gene[cq_type].af_threshold = variants.af_thresh
        
        model = fit_gene(phenotype, gene[cq_type], conn, score_type, cq_type, exome_samples)
        models.append((gene, model))
        carriers = get_max_values_for_carriers(variants)
        for sample, var in carriers.items():
            if sample in risk_scores:
                risk_scores[sample] += freq_model_predict(model, var)
    
    risk_scores = {int(k): float(v) for k, v in risk_scores.items()}
    
    if model_path:
        save_model(models, model_path)
    
    return risk_scores

def get_rare_prs_v3(include_samples, rare_genes, exome_db_path, ancestries, 
        pheno_path, pheno_col, train_samples, model_path=None, cq_type='del', 
        score_type='primateai_v2', max_af=0.001, 
        gwas_db=None, model='linreg'):
    ''' calculate rare polygenic risk scores with jointly modelled per-variant effects
    
    Args:
        include_samples: set of samples used for the PRS
        rare_genes: list of dicts, each with 'symbol', 'p_value', 'beta', 
                    'ac_threshold' and 'pathogenicity_threshold' fields
        exome_db_path: path to exome database
        pheno_path: path to phenotype file
        pheno_col: name of phenotype column to use from phenotype file
        train_samples: set of samples for training per-variant effects
        model_path: if included, will save per-variant and per-gene model 
            parameters to this path as json output.
        cq_type: variant type, one of ['del', 'syn' or 'ptv']
        score_type: name of column for missense pathogenicity
    '''
    with sqlite3.connect(exome_db_path) as conn:
        exome_samples = set(map(int, get_exome_samples(conn)))
    
    phenotype = get_phenotypes(pheno_path, pheno_col)
    phenotype = ((k, v) for k, v in phenotype if int(k) in exome_samples and int(k) in train_samples)
    phenotype = ((k, v) for k, v in phenotype if v is not None)
    if gwas_db:
        phenotype = regress_out_common_variants(phenotype, gwas_db)
    phenotype = {int(k): v for k, v in phenotype}
    
    conn = sqlite3.connect(exome_db_path)
    rare_genes = [x for x in rare_genes if not math.isnan(x[cq_type].p_value)]
    rare_genes = sorted(rare_genes, key=lambda x: float(x[cq_type].p_value))
    genes = [x[cq_type] for x in rare_genes]
    model = fit_genes(phenotype, genes, conn, score_type, cq_type, exome_samples, model)
    
    include_samples = sorted(include_samples)
    sample_idx = {k: i for i, k in enumerate(sorted(include_samples))}
    arr = numpy.zeros((len(sample_idx), len(genes) * 2))
    
    col = 0
    for gene in rare_genes:
        variants = select_variants(conn, gene, cq_type, score_type, max_af * 10, exome_samples)
        variants = prune_by_ancestry_af(gene, cq_type, variants, train_samples, ancestries)
        carriers = get_max_values_for_carriers(variants)
        for sample, var in carriers.items():
            try:
                idx = sample_idx[sample]
            except KeyError:
                continue
            arr[idx, col] = 1 / (var.af ** 0.5)
            arr[idx, col+1] = var.pathogenicity
            
        col += 2
    
    scores = (model.beta[-1] + arr * model.beta[:-1]).sum(axis=1)
    risk_scores = dict(zip(include_samples, map(float, scores)))
    
    return risk_scores

def get_rare_prs(include_samples, rare_genes, exome_db_path, ancestries, 
        cq_type='del', score_type='primateai_v2', max_af=0.001, pheno_path=None, 
        pheno_col=None, train_samples=None, model_path=None, gwas_db=None, 
        version='v2', **kwargs):
    ''' calculate rare polygenic risk scores for samples with exomic genotypes
    
    Args:
        include_samples: set of samples used for the PRS
        rare_genes: list of dicts, each with 'symbol', 'p_value', 'beta', 
                    'ac_threshold' and 'pathogenicity_threshold' fields
        exome_db_path: path to exome database
        cq_type: variant type, one of ['del', 'syn' or 'ptv']
        score_type: name of column for missense pathogenicity
        pheno_path: path to phenotype file
        pheno_col: name of phenotype column to use from phenotype file
    '''
    # if the pheno_db, pheno_col, and train_samples are passed in, we want the
    # per-variant PRS model, which uses get_rare_prs_v2 instead of get_rare_prs_v1
    if pheno_path is None and pheno_col is None and train_samples is None:
        return get_rare_prs_v1(include_samples, rare_genes, exome_db_path, 
                               ancestries,  cq_type, score_type, max_af)
    elif version == 'v2':
        return get_rare_prs_v2(include_samples, rare_genes, exome_db_path, 
                               ancestries, pheno_path, pheno_col, 
                               train_samples, model_path, cq_type, score_type,
                               max_af, gwas_db)
    elif version == 'v3':
        return get_rare_prs_v3(include_samples, rare_genes, exome_db_path, 
                               ancestries, pheno_path, pheno_col, 
                               train_samples, model_path, cq_type, score_type, 
                               max_af, gwas_db, **kwargs)

def get_rare_prs_pretrained(model_path: Path, 
                            exome_db_path: Path, 
                            samples: Set[int], 
                            score_type: str,
                            ancestries: Dict[str, Set[int]]):
    ''' compute rare variant PRS using pretrained model
    '''
    logging.info(f'opening pretrained rvPRS model from {model_path}')
    model = json.load(open(model_path))
    conn = sqlite3.connect(exome_db_path)
    
    fit = MockFit(['intercept', 'af', 'pathogenicity'])
    fit.log10cols = ['af']
    
    cq_type = 'del'
    risk_scores = {k: 0 for k in samples}
    for symbol, data in model.items():
        gene = {cq_type: Result(symbol, 'del', data['gene_beta'], 1e-7, 
                                data['ac_threshold'], 
                                data['pathogenicity_threshold'], 
                                af_threshold=data['af_threshold'])}
        # find rare variants with pathogenicity above the gene threshold
        variants = select_variants(conn, gene, cq_type, score_type, 
                                   data['af_threshold'], samples)
        variants = prune_by_ancestry_af(variants, samples, ancestries, 
                                        af_thresh=data['af_threshold'])
        
        # swap the variant AF field to the value for the sample subset
        for variant in variants:
            variant.af = get_af(variant, samples)
        
        fit.beta = [data['intercept'], data['af_effect'], data['pathogenicity_effect']]
        
        carriers = get_max_values_for_carriers(variants)
        for sample, var in carriers.items():
            if sample in risk_scores:
                risk_scores[sample] += freq_model_predict(fit, var)
    
    return risk_scores, model

def write_output(risk: Dict[int, float], n_genes: int, metadata, path: Path):
    logging.info(f'writing risk scores to {path}')
    with gzip.open(path, 'wt') as output:
        if metadata is None:
            metadata = ''
        _ = output.write(f'#n_genes={n_genes},{metadata}\n')
        _ = output.write('sample\tscore\n')
        for sample_id in sorted(risk):
            output.write(f'{sample_id}\t{risk[sample_id]:.5g}\n')

def ensure_no_subset_overlap(train, test):
    ''' make sure the test and train samples sets do not overlap, exit otherwise
    '''
    if train is None:
       return
    
    if len(test & train) > 0:
        sys.exit(f'training and test subsets cannot overlap: {test & train}')

def main():
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', level=logging.INFO)
    args = get_options()
    
    test_samples = set()
    if args.test_samples is not None:
        test_samples = set(int(x) for x in open_sample_subset(args.test_samples))
    
    ancestries = {}
    if args.ancestry_samples is not None:
        for i, path in enumerate(args.ancestry_samples):
            samples = set(int(x) for x in open_sample_subset(path))
            ensure_no_subset_overlap(samples, test_samples)
            ancestries[i] = samples
    
    if hasattr(args, 'train_samples'):
        train_samples = set(int(x) for x in open_sample_subset(args.train_samples))
        ensure_no_subset_overlap(train_samples, test_samples)
        
        gencode = Gencode(args.gencode)
        genes = open_rare_variant_results(args.rare_results, gencode, args.max_p)
        genes = list(genes)

        if args.restrict_to:
            gene_subset = open_gene_subset(args.restrict_to)
            genes = [x for x in genes if x['del']['symbol'] in gene_subset]

        risk_scores = get_rare_prs(test_samples, genes, args.exome_db, ancestries, 
                                cq_type='del',
                                score_type=args.score_type, 
                                pheno_path=args.phenotype,
                                pheno_col=args.phenotype_column,
                                train_samples=train_samples,
                                gwas_db=args.gwas_db,
                                model_path=args.output_model,
                                version='v2')
    else:
        risk_scores, genes = get_rare_prs_pretrained(args.rare_model, args.exome_db, 
                                              test_samples, args.score_type, ancestries)
    
    if args.output is not None:
        write_output(risk_scores, len(genes), args.include_metadata, args.output)

if __name__ == "__main__":
    main()
