import matplotlib
matplotlib.use('Agg')

import argparse
import json
import pprint
import tempfile

from jutils import *
from ukb_analysis import *
from constants import *
import requests, sys
from finemap_utils import *

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from io import StringIO
import time
import urllib3

# DBSNP_PATH = ROOT_PATH + '/pfiziev/dbsnp/'
ANNOTATED_COMMON_VARIANTS = ROOT_PFIZIEV_PATH + '/rare_variants/data/finemapping/annotated_dbsnp.hg19.tsv.gz'
GENCODE_PATH = ROOT_PFIZIEV_PATH + '/rare_variants/data/gencode/gencode.v24lift37.canonical.with_CDS.tsv'

GENOMEWIDE_SIGN_THRESHOLD = 5e-8
MIN_LD_THRESHOLD = 0.5

WINDOW_NON_CODING = 50000

LAST_REQUEST_TIME = {}
ENSEMBL = 'ensembl'
PICS2 = 'pics2'
PVALUE = 'pvalue'


def finemap_lead_variants(gwas_variants,
                          pvalue_label='pvalue',
                          pvalue_threshold=GENOMEWIDE_SIGN_THRESHOLD,
                          ld_method='ukb',
                          fallback_local_LD_calculations_to_UKB=True,
                          genome='hg19',
                          out_dir=None,
                          batch_size=100,
                          min_spliceai_score=0.2):

    echo('[finemap_lead_variants] start!')

    gwas_variants = gwas_variants[gwas_variants[pvalue_label] <= pvalue_threshold].sort_values(pvalue_label).drop_duplicates(subset=[VCF_CHROM, VCF_POS]).copy()
    echo('Keeping variants below p-value:', pvalue_threshold, ', n=', len(gwas_variants))

    cols_to_keep = [VCF_CHROM, VCF_POS, VCF_RSID, 'lead_snp', PVALUE]
    if VCF_REF in list(gwas_variants):
        cols_to_keep += [VCF_REF]

    if VCF_ALT in list(gwas_variants):
        cols_to_keep += [VCF_ALT]

    echo('cols_to_keep:', cols_to_keep)

    gwas_variants = gwas_variants[cols_to_keep].copy()

    echo('Finemapping GWAS variants:', len(gwas_variants), 'above p-value=', pvalue_threshold, ', ld_method=', ld_method)

    if genome == 'hg19':
        spliceai_fname = ROOT_PFIZIEV_PATH + '/rare_variants/data/finemapping/spliceai_scores.hg19.min_score_0.1.pandas.pickle'
        eqtls_fname = ROOT_PFIZIEV_PATH + '/rare_variants/data/finemapping/gene_eqtls.hg19.csv'
        gencode_fname = GENCODE_PATH

    elif genome == 'hg38':
        spliceai_fname = ROOT_PFIZIEV_PATH + '/rare_variants/data/finemapping/spliceai_scores.hg38.min_score_0.1.pandas.pickle'
        eqtls_fname = ROOT_PFIZIEV_PATH + '/rare_variants/data/finemapping/gene_eqtls.hg38.csv'
        gencode_fname = GENCODE_PATH_HG38

    else:
        raise Exception('Unknown genome:', genome)

    echo('Reading:', gencode_fname)

    gencode = pd.read_csv(gencode_fname, sep='\t').rename(columns={'gene': GENE_NAME, 'chrom': VCF_CHROM})
    gencode[VCF_CHROM] = gencode[VCF_CHROM].str.replace('chr', '')

    echo('Reading:', spliceai_fname)
    spliceai_variants = pd.read_pickle(spliceai_fname)
    spliceai_variants = spliceai_variants[spliceai_variants['INFO'] >= min_spliceai_score].copy()
    SPLICEAI_CODING_TAG = f'spliceAI>={min_spliceai_score}'
    spliceai_variants[VCF_CONSEQUENCE] = SPLICEAI_CODING_TAG


    echo('Reading:', eqtls_fname)
    eqtl_variants = pd.read_csv(eqtls_fname, sep='\t', dtype={VCF_CHROM: str, VCF_POS: int, VCF_ALT: str, VCF_REF: str, 'consistent_effects': int})

    eqtl_variants = pd.merge(eqtl_variants,
                             gencode[[GENE_NAME]]).rename(columns={'ID': VCF_RSID})

    echo('initial eQTLs:', len(eqtl_variants))

    # eqtls = eqtls[eqtls['consistent_effects'] == 1]

    eqtl_variants['INFO'] = 'n_genes=' + eqtl_variants['n_genes'].map(str) + '|consistent_effect=' + eqtl_variants['consistent_effects'].map(
        str) + '|beta=' + eqtl_variants['max_effect'].map(str) + "|pval=" + eqtl_variants['best_pvalue'].map(str) + "|" + eqtl_variants[
                        'tissues']

    # eqtls = eqtls.drop_duplicates([VCF_CHROM, VCF_POS], keep=False)

    eqtl_variants[VCF_CONSEQUENCE] = 'eQTL'

    echo(len(eqtl_variants), list(eqtl_variants))

    res = None

    all_chroms = sorted(gwas_variants[VCF_CHROM].unique())
    echo('all_chroms:', all_chroms)

    # shuffle chromosome labels to avoid collisions in the file system due to multiple processes
    random.shuffle(all_chroms)
    n_chroms = len(all_chroms)

    for chrom_idx, chrom in enumerate(all_chroms):

        chrom_snps = gwas_variants[gwas_variants[VCF_CHROM] == chrom].sort_values(pvalue_label).copy()

        echo('Fine-mapping chromosome:', chrom, f', ({chrom_idx} / {n_chroms})', ', variants:', len(chrom_snps))

        if len(chrom_snps) > 0:
            for batch_idx, batch_chrom_snps in enumerate(batchify(chrom_snps, batch_size)):
                echo('Processing batch_idx:', batch_idx)
                chrom_res = annotate_chrom_lead_snps(batch_chrom_snps,
                                                     None,
                                                     None,
                                                     None,
                                                     eqtl_variants,
                                                     spliceai_variants,
                                                     gencode,
                                                     out_dir,
                                                     ld_method=ld_method,
                                                     fallback_local_LD_calculations_to_UKB=fallback_local_LD_calculations_to_UKB,
                                                     SPLICEAI_CODING_TAG=SPLICEAI_CODING_TAG)

                if chrom_res is None:
                    continue

                if res is None:
                    res = chrom_res
                else:
                    res = pd.concat([res, chrom_res], ignore_index=True)

    if res is not None and len(res) > 0:
        res[VCF_POS] = res[VCF_POS].astype(int)
        res[VARID_REF_ALT] = res[VCF_CHROM] + ':' + res[VCF_POS].astype(str) + ':' + res[VCF_REF] + ':' + res[VCF_ALT]
        res = res.sort_values(pvalue_label)

    return res

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', dest='in_fname', help='in_fname', required=True)

    parser.add_argument('-g', dest='genome', help='genome [%(default)s]', choices=['hg19', 'hg38'], default='hg19')
    parser.add_argument('--filter-chrom', dest='filter_chrom', type=str, help='filter chromosome [%(default)s]', default=None)

    parser.add_argument('--p-value-label', dest='pvalue_label', help='p-value label [%(default)s]', default='P-VALUE')
    parser.add_argument('--chrom-label', dest='chrom_label', help='chrom label [%(default)s]', default='CHR_ID')
    parser.add_argument('--pos-label', dest='pos_label', help='position label [%(default)s]', default='CHR_POS')
    parser.add_argument('--rsid-label', dest='rsid_label', help='position label [%(default)s]', default='SNPS')

    parser.add_argument('--ld-method',
                        dest='ld_method',
                        help='LD method',
                        default='ukb',
                        choices=['ukb', 'local', 'ensembl'])

    parser.add_argument('--fallback_local_LD_calculations_to_UKB',
                        dest='fallback_local_LD_calculations_to_UKB',
                        action='store_true',
                        help='phenotype column label')

    parser.add_argument('-o', dest='out_fname', help='output file name', required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    args = parser.parse_args()

    in_fname = args.in_fname
    filter_chrom = args.filter_chrom
    fallback_local_LD_calculations_to_UKB = args.fallback_local_LD_calculations_to_UKB
    out_fname = args.out_fname

    open_log(out_fname + '.log', 'wt')

    echo('CMD:', ' '.join(sys.argv))
    echo('Parameters:\n' + pprint.pformat(args.__dict__))
    out_dir = os.path.split(out_fname)[0]

    if out_dir == '':
        out_dir = './'

    echo('out_dir:', out_dir)

    gwas_data = pd.read_csv(in_fname, sep='\t')
    gwas_data = gwas_data.rename(columns={args.pvalue_label: PVALUE,
                                          args.chrom_label: VCF_CHROM,
                                          args.pos_label: VCF_POS,
                                          args.rsid_label: VCF_RSID})

    def cast_int(x):
        try:
            return int(x)
        except:
            return None

    def cast_float(x):
        try:
            return float(x)
        except:
            return None

    gwas_data[VCF_CHROM] = gwas_data[VCF_CHROM].astype(str)
    gwas_data[VCF_POS] = gwas_data[VCF_POS].apply(cast_int)
    gwas_data[PVALUE] = gwas_data[PVALUE].apply(cast_float)
    gwas_data['lead_snp'] = gwas_data[VCF_RSID]

    echo('Total variants:', len(gwas_data))

    gwas_data = gwas_data.dropna(subset=[VCF_CHROM, VCF_POS, PVALUE, VCF_RSID])

    # VCF_POS is still float because of the missing values, so convert one more time
    gwas_data[VCF_POS] = gwas_data[VCF_POS].astype(int)

    echo('Total variants after filtering out missing values:', len(gwas_data))

    AUTOSOMES = [str(i) for i in range(1, 23)]

    gwas_data = gwas_data[gwas_data[VCF_CHROM].isin(AUTOSOMES)].copy()
    if filter_chrom is not None:
        echo('Filtering chromosome:', filter_chrom)
        gwas_data = gwas_data[gwas_data[VCF_CHROM] == filter_chrom].copy()

    echo('Variants on autosomes:', len(gwas_data))
    echo('per chromosome:\n', gwas_data.groupby(VCF_CHROM).size())


    if len(gwas_data) == 0:
        echo('Nothing to finemap!')

    else:
        original_gwas_data = gwas_data.copy()
        gwas_cat_finemapped = finemap_lead_variants(gwas_data,
                                                    pvalue_label=PVALUE,
                                                    pvalue_threshold=GENOMEWIDE_SIGN_THRESHOLD,
                                                    ld_method=args.ld_method,
                                                    genome=args.genome,
                                                    fallback_local_LD_calculations_to_UKB=fallback_local_LD_calculations_to_UKB,
                                                    out_dir=out_dir)

        if gwas_cat_finemapped is not None:

            gwas_cat_finemapped = pd.merge(original_gwas_data,
                                           gwas_cat_finemapped[[c for c in list(gwas_cat_finemapped) if c not in [VCF_RSID, PVALUE]]],
                                           on=[VCF_CHROM, VCF_POS],
                                           suffixes=['/original', ''])

            echo('Saving table to:', out_fname)
            gwas_cat_finemapped.to_csv(out_fname + '.csv.gz', sep='\t', index=False)
            gwas_cat_finemapped.to_pickle(out_fname + '.pickle', protocol=4)
        else:
            echo('Nothing was finemapped!')
    close_log()



