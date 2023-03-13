import copy
import gc
import multiprocessing

import matplotlib
matplotlib.use('Agg')

import argparse
import pprint

from jutils import *

import pickle
import sys
import pandas as pd
import time
import statsmodels.api as sm

# from ukb_analysis import *
# from rare_variants_test_utils import *
from ukb_analysis import *

UKB_DATA_PATH = ''

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-i',
                        dest='input_fname',
                        help='pickle file the phenotype values for each subject [e.g ' +
                              UKB_DATA_PATH +
                              '/molecular_phenotypes.17_SEPT_2019/phenotypes_for_GWAS/LDL_direct.europeans/LDL_direct.europeans.both.med_corrected.phenotype_values.pickle]',
                        required=True)

    parser.add_argument('-p',
                        dest='phenotype_name',
                        help='phenotype name',
                        required=True)

    parser.add_argument('-g',
                        dest='finemapped_gwas',
                        help='finemapped gwas variants',
                        required=True)

    parser.add_argument('-o', dest='out_prefix', help='output file name', required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    args = parser.parse_args()

    log_name = args.out_prefix

    open_log(log_name + '.log', 'wt')

    echo('CMD:', ' '.join(sys.argv))

    echo('Parameters:\n' + pprint.pformat(args.__dict__))

    input_fname = args.input_fname
    finemapped_gwas = args.finemapped_gwas

    ph_label = args.phenotype_name

    out_prefix = args.out_prefix

    echo('Reading:', input_fname)
    ukb_phenotypes = pd.read_pickle(input_fname)

    echo('Extracting common variant genotypes')
    _, gwas_vars = read_finemapped_variants(finemapped_gwas)

    ukb_common_variants_gt = get_dosages_for_variants(gwas_vars.sort_values([VCF_CHROM, VCF_POS]),
                                                      sample_ids=None,
                                                      rsid_label='variant')

    echo('Preparing data for correction')
    reg_data = pd.merge(ukb_phenotypes[[SAMPLE_ID, ph_label]], ukb_common_variants_gt, on=SAMPLE_ID).dropna()
    reg_data[CONST_LABEL] = 1

    all_gwas_vars = [c for c in ukb_common_variants_gt if c != SAMPLE_ID]
    non_coding_gwas_vars = sorted(set(gwas_vars[~gwas_vars['assoc_type'].apply(
        lambda x: len(set(x.split(';')) & {'coding', 'coding/ambiguous', 'splicing', 'splicing/ambiguous'}) > 0)][
                                          VARID_REF_ALT]))

    coding_gwas_vars = sorted(set(gwas_vars[gwas_vars['assoc_type'].apply(
        lambda x: len(set(x.split(';')) & {'coding', 'coding/ambiguous', 'splicing', 'splicing/ambiguous'}) > 0)][
                                      VARID_REF_ALT]))

    resid = pd.DataFrame({SAMPLE_ID: reg_data[SAMPLE_ID],
                          ph_label: reg_data[ph_label]})

    for gwas_vt, gwas_label in [(all_gwas_vars, 'all_gwas_resid'),
                                (non_coding_gwas_vars, 'non_coding_gwas_resid'),
                                (coding_gwas_vars, 'coding_gwas_resid')]:
        lm = sm.OLS(reg_data[ph_label], reg_data[gwas_vt + [CONST_LABEL]])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            lm_res = lm.fit()

        echo(gwas_label, len(gwas_vt), '\n', str(lm_res.summary())[:2000] + '\n\n')

        resid = pd.merge(resid,
                         pd.DataFrame({SAMPLE_ID: reg_data[SAMPLE_ID],
                                       ph_label + '.' + gwas_label: lm_res.resid}),
                         on=SAMPLE_ID)

    echo('resid:', resid.shape)

    resid.to_pickle(out_prefix + '.pickle')
    resid.to_csv(out_prefix + '.csv.gz', sep='\t', index=False)

    echo('Done')
    close_log()


if __name__ == '__main__':
    main()
