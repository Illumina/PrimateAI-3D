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


ANNOTATED_COMMON_VARIANTS = ROOT_PATH + '/pfiziev/rare_variants/data/finemapping/annotated_dbsnp.hg19.tsv.gz'
GENCODE_PATH = ROOT_PATH + '/pfiziev/rare_variants/data/gencode/gencode.v24lift37.canonical.with_CDS.tsv'

GENOMEWIDE_SIGN_THRESHOLD = 5e-8
MIN_LD_THRESHOLD = 0.5

WINDOW_NON_CODING = 50000

LAST_REQUEST_TIME = {}
ENSEMBL = 'ensembl'
PICS2 = 'pics2'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', dest='in_fname', help='in_fname', required=True)
    parser.add_argument('-p', dest='ph_fname', help='phenotype data', required=True)
    parser.add_argument('-l', dest='ph_label', help='phenotype column label', required=True)
    parser.add_argument('--filter-samples', dest='filter_samples', help='filter_samples')

    parser.add_argument('--ld-method', dest='ld_method', help='LD method', default='ukb', choices=['ukb', 'local', 'ensembl'])

    parser.add_argument('--fallback_local_LD_calculations_to_UKB',
                        dest='fallback_local_LD_calculations_to_UKB',
                        action='store_true',
                        help='phenotype column label')

    parser.add_argument('-o', dest='out_fname', help='output file name', required=True)
    parser.add_argument('--exit-if-exists',
                        action='store_true',
                        dest='exit_if_exists',
                        help='Exit if results files exist')

    parser.add_argument('--is-gcta',
                        action='store_true',
                        dest='is_gcta',
                        help='Output is from GCTA')

    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    args = parser.parse_args()

    in_fname = args.in_fname
    ph_fname = args.ph_fname
    ph_label = args.ph_label
    is_gcta = args.is_gcta

    filter_samples = args.filter_samples

    fallback_local_LD_calculations_to_UKB = args.fallback_local_LD_calculations_to_UKB
    out_fname = args.out_fname

    exit_if_exists = args.exit_if_exists
    if exit_if_exists:
        if os.path.exists(out_fname.replace('.csv.gz', '.pickle')):
            echo('Results files exist:', out_fname.replace('.csv.gz', '.pickle'))
            echo('Exiting..')
            exit(0)

    open_log(out_fname + '.log', 'wt')

    echo('CMD:', ' '.join(sys.argv))
    echo('Parameters:\n' + pprint.pformat(args.__dict__))
    out_dir = os.path.split(out_fname)[0]

    if out_dir == '':
        out_dir = './'

    echo('Reading phenotype data:', ph_fname)

    ph_data = pd.read_pickle(ph_fname)
    ph_data = ph_data[[SAMPLE_ID, ph_label] + [c for c in list(ph_data) if 'gPC_' in c]].copy()

    if filter_samples:
        if filter_samples.endswith('.pickle'):
            samples_to_keep = pd.read_pickle(filter_samples)
        else:
            samples_to_keep = pd.read_csv(filter_samples, sep='\t', dtype={SAMPLE_ID: str})

        samples_to_keep = set(samples_to_keep[SAMPLE_ID])
        echo('samples_to_keep:', len(samples_to_keep))

        ph_data = ph_data[ph_data[SAMPLE_ID].isin(samples_to_keep)].copy()

    gc.collect()

    echo('ph_data:', ph_data.shape)

    ph_data = ph_data.dropna()
    echo('ph_data after dropping NaNs:', ph_data.shape)

    echo('out_dir:', out_dir)

    is_sqlite = in_fname.endswith('.db')
    echo('input is_sqlite:', is_sqlite)
    if not os.path.exists(in_fname):
        echo('[ERROR] File not found:', in_fname)
        raise Exception("File not found:" + in_fname)

    gwas_data = get_GWAS_summary_stats(in_fname, is_sqlite=is_sqlite, is_gcta=is_gcta)

    AUTOSOMES = [str(i) for i in range(1, 23)]

    echo('Total variants:', len(gwas_data))
    gwas_data = gwas_data[gwas_data[VCF_CHROM].isin(AUTOSOMES)]

    echo('Variants on autosomes:', len(gwas_data))
    echo('per chromosome:\n', gwas_data.groupby(VCF_CHROM).size())
    if len(gwas_data) == 0:
        echo('Nothing to finemap!')

    else:

        gwas_cat_finemapped = finemap(gwas_data,
                                      ph_data,
                                      ph_label,
                                      pvalue_label='pvalue',
                                      min_AF=0.01,
                                      odds_ratio_label='odds_ratio',
                                      pvalue_threshold=GENOMEWIDE_SIGN_THRESHOLD,
                                      ld_method=args.ld_method,
                                      genome='hg19',
                                      fallback_local_LD_calculations_to_UKB=fallback_local_LD_calculations_to_UKB,
                                      out_dir=out_dir)

        if gwas_cat_finemapped is not None:
            echo('Saving table to:', out_fname)
            gwas_cat_finemapped.to_csv(out_fname, sep='\t', index=False)
            gwas_cat_finemapped.to_pickle(out_fname.replace('.csv.gz', '.pickle'), protocol=4)
        else:
            echo('Nothing was finemapped!')
    close_log()


