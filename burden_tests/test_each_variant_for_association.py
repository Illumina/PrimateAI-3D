import copy
import gc
import multiprocessing

import matplotlib
matplotlib.use('Agg')

import argparse
import pprint

# from jutils import *

import pickle
import sys
import pandas as pd
import time

# from ukb_analysis import *
from rare_variants_test_utils import *
from ukb_analysis import *

UKB_DATA_PATH = ROOT_PATH + '/ukbiobank/data/'


def add_element(d, key, el):
    if key not in d:
        d[key] = []
    d[key].append(el)

#
# def test_each_variant_for_association(all_variants_in_phenotype_samples,
#                                       ph_data_for_testing,
#                                       phenotype_name=None,
#                                       n_shuffles=0,
#                                       cache=None
#                                       ):
#
#     echo('Testing variants for association:', len(all_variants_in_phenotype_samples), '!')
#
#     ph_data_for_testing = ph_data_for_testing.sort_values(SAMPLE_ID)
#     ph_std = np.std(ph_data_for_testing[phenotype_name])
#
#     echo('Generating shuffled phenotype values')
#     permuted_ph_values = np.tile(ph_data_for_testing[phenotype_name],
#                                  (n_shuffles + 1, 1))
#
#     for i in range(1, n_shuffles + 1):
#         np.random.shuffle(permuted_ph_values[i])
#
#     permuted_ph_values = permuted_ph_values.T
#
#     ph_values_and_randos = pd.DataFrame(permuted_ph_values)
#     ph_values = ph_values_and_randos[[0]]
#
#     all_samples = list(ph_data_for_testing[SAMPLE_ID])
#     all_samples_set = set(all_samples)
#     s2i = dict((s, i) for i, s in enumerate(all_samples))
#
#     gt_array = np.zeros((len(ph_data_for_testing), 1))
#
#     res = {}
#     n_cache_hits = 0
#
#     echo('Computing hom/het/miss stats')
#
#     all_variants_in_phenotype_samples[HOMOZYGOTES] = all_variants_in_phenotype_samples[HOMOZYGOTES].apply(lambda s: set(s.split(',')) & all_samples_set)
#     all_variants_in_phenotype_samples[HETEROZYGOTES] = all_variants_in_phenotype_samples[HETEROZYGOTES].apply(lambda s: set(s.split(',')) & all_samples_set)
#     all_variants_in_phenotype_samples[MISSING] = all_variants_in_phenotype_samples[MISSING].apply(lambda s: set(s.split(',')) & all_samples_set)
#
#     all_variants_in_phenotype_samples['n_homs'] = all_variants_in_phenotype_samples[HOMOZYGOTES].apply(len)
#     all_variants_in_phenotype_samples['n_hets'] = all_variants_in_phenotype_samples[HETEROZYGOTES].apply(len)
#     all_variants_in_phenotype_samples['n_miss'] = all_variants_in_phenotype_samples[MISSING].apply(len)
#
#     all_variants_in_phenotype_samples['key'] = zip(all_variants_in_phenotype_samples['n_hets'],
#                                                    all_variants_in_phenotype_samples['n_homs'])
#
#     all_variants_in_phenotype_samples[VCF_AC] = 2 * all_variants_in_phenotype_samples['n_homs'] + all_variants_in_phenotype_samples['n_hets']
#     all_variants_in_phenotype_samples[VCF_AN] = len(all_samples_set) - 2 * all_variants_in_phenotype_samples['n_miss']
#
#     all_variants_in_phenotype_samples = all_variants_in_phenotype_samples[all_variants_in_phenotype_samples[VCF_AC] > 0].copy()
#     all_variants_in_phenotype_samples[VCF_AF] = all_variants_in_phenotype_samples[VCF_AC] / all_variants_in_phenotype_samples[VCF_AN]
#
#     echo('Processing variants:', len(all_variants_in_phenotype_samples))
#
#     batch_size = 1000
#     for batch_idx, var_batch in enumerate(batchify(all_variants_in_phenotype_samples, batch_size=batch_size)):
#
#         cache_keys = set(cache.keys())
#         cached_vars = var_batch[var_batch['key'].isin(cache_keys)]
#         non_cached_vars = var_batch[~var_batch['key'].isin(cache_keys)]
#
#         echo('batch:', batch_idx, ', cached_vars:', cached_vars.shape, ', non_cached_vars:', non_cached_vars.shape)
#
#         for vs, is_cached in [(cached_vars, True), (non_cached_vars, False)]:
#
#             for v_idx, (_, v) in enumerate(vs.iterrows()):
#
#                 heterozygotes = v[HETEROZYGOTES]
#                 homozygotes = v[HOMOZYGOTES]
#
#                 n_hets = len(heterozygotes)
#                 n_homs = len(homozygotes)
#
#                 if n_hets == 0 and n_homs == 0:
#                     continue
#
#                 v_key = (n_hets, n_homs)
#
#                 for i in range(len(gt_array)):
#                     gt_array[i, v_idx] = 0
#
#                 for s in heterozygotes:
#                     gt_array[s2i[s], v_idx] = 1
#
#                 for s in homozygotes:
#                     gt_array[s2i[s], v_idx] = 2
#
#                 if v_key not in cache:
#                     r, prob = vcorrcoef(gt_array, ph_values_and_randos)
#
#                     r = r.flatten()
#                     prob = prob.flatten()[0]
#
#                     gt_std = np.std(gt_array.flatten())
#                     beta_on_std_ph = r / gt_std
#                     beta = beta_on_std_ph * ph_std
#
#                     beta_on_std_ph_std = np.std(beta_on_std_ph)
#                     abs_mean_on_std_ph = np.mean(np.abs(beta_on_std_ph))
#
#                     beta_std = np.std(beta)
#                     abs_mean = np.mean(np.abs(beta))
#
#                     cache[v_key] = (beta_on_std_ph_std, abs_mean_on_std_ph, beta_std, abs_mean, gt_std)
#
#                 else:
#                     r, prob = vcorrcoef(gt_array, ph_values)
#                     r = r.flatten()
#                     n_cache_hits += 1
#                     # echo('key found:', v_key, r, prob, cache[v_key])
#
#                 (beta_on_std_ph_std, abs_mean_on_std_ph, beta_std, abs_mean, gt_std) = cache[v_key]
#
#                 for k, v in [(VARID_REF_ALT, v[VARID_REF_ALT]),
#                              (phenotype_name + '/r', r[0]),
#                              (phenotype_name + '/beta', r[0] * ph_std / gt_std),
#                              (phenotype_name + '/beta_on_standardized_phenotype', r[0] / gt_std),
#                              (phenotype_name + '/pvalue', prob),
#                              (phenotype_name + '/std', ph_std),
#                              (phenotype_name + '/beta_on_standardized_phenotype/stddev', beta_on_std_ph_std),
#                              (phenotype_name + '/beta_on_standardized_phenotype/abs_mean', abs_mean_on_std_ph),
#                              (phenotype_name + '/beta/stddev', beta_std),
#                              (phenotype_name + '/beta/abs_mean', abs_mean),
#                              (VCF_AC, new_ac),
#                              (VCF_AN, new_an),
#                              (VCF_AF, new_af)]:
#
#                     if k not in res:
#                         res[k] = []
#
#                     res[k].append(v)
#
#     echo('Creating dataframe')
#     res = pd.DataFrame(res).drop_duplicates(VARID_REF_ALT)
#
#     echo('Merging back with variants')
#     res = pd.merge(all_variants_in_phenotype_samples, res, on=VARID_REF_ALT, suffixes=['/original', ''])
#     res = res.sort_values(phenotype_name + '/pvalue')
#
#     return res

def test_each_variant_for_association(all_variants_in_phenotype_samples,
                                      ph_data_for_testing,
                                      phenotype_name=None,
                                      n_shuffles=0,
                                      cache=None
                                      ):

    echo('Testing variants for association:', len(all_variants_in_phenotype_samples), '!')

    ph_data_for_testing = ph_data_for_testing.sort_values(SAMPLE_ID)
    ph_std = np.std(ph_data_for_testing[phenotype_name])

    echo('Generating shuffled phenotype values')
    permuted_ph_values = np.tile(ph_data_for_testing[phenotype_name],
                                 (n_shuffles + 1, 1))

    for i in range(1, n_shuffles + 1):
        np.random.shuffle(permuted_ph_values[i])

    permuted_ph_values = permuted_ph_values.T

    ph_values_and_randos = pd.DataFrame(permuted_ph_values)
    ph_values = ph_values_and_randos[[0]]

    all_samples = list(ph_data_for_testing[SAMPLE_ID])
    all_samples_set = set(all_samples)
    s2i = dict((s, i) for i, s in enumerate(all_samples))

    gt_array = np.zeros((len(ph_data_for_testing), 1))

    res = {}
    n_cache_hits = 0
    for v_idx, (_, v) in enumerate(all_variants_in_phenotype_samples.iterrows()):

        if v_idx % 1000 == 0:
            echo(v_idx, 'out of', len(all_variants_in_phenotype_samples), 'variants processed, cache hits:', n_cache_hits)

        heterozygotes = set(v[HETEROZYGOTES].split(',')) & all_samples_set
        homozygotes = set(v[HOMOZYGOTES].split(',')) & all_samples_set

        n_hets = len(heterozygotes)
        n_homs = len(homozygotes)

        if n_hets == 0 and n_homs == 0:
            continue

        new_ac = n_hets + 2 * n_homs
        new_an = 2 * len(all_samples_set - set(v[MISSING].split(',')))
        new_af = new_ac / new_an

        v_key = (n_hets, n_homs)

        for i in range(len(gt_array)):
            gt_array[i, 0] = 0

        for s in heterozygotes:
            gt_array[s2i[s], 0] = 1

        for s in homozygotes:
            gt_array[s2i[s], 0] = 2

        if v_key not in cache:
            r, prob = vcorrcoef(gt_array, ph_values_and_randos)

            r = r.flatten()
            prob = prob.flatten()[0]

            gt_std = np.std(gt_array.flatten())
            beta_on_std_ph = r / gt_std
            beta = beta_on_std_ph * ph_std

            beta_on_std_ph_std = np.std(beta_on_std_ph)
            abs_mean_on_std_ph = np.mean(np.abs(beta_on_std_ph))

            beta_std = np.std(beta)
            abs_mean = np.mean(np.abs(beta))

            cache[v_key] = (beta_on_std_ph_std, abs_mean_on_std_ph, beta_std, abs_mean, gt_std)

        else:
            r, prob = vcorrcoef(gt_array, ph_values)
            r = r.flatten()
            n_cache_hits += 1
            # echo('key found:', v_key, r, prob, cache[v_key])

        (beta_on_std_ph_std, abs_mean_on_std_ph, beta_std, abs_mean, gt_std) = cache[v_key]

        for k, v in [(VARID_REF_ALT, v[VARID_REF_ALT]),
                     (phenotype_name + '/r', r[0]),
                     (phenotype_name + '/beta', r[0] * ph_std / gt_std),
                     (phenotype_name + '/beta_on_standardized_phenotype', r[0] / gt_std),
                     (phenotype_name + '/pvalue', prob),
                     (phenotype_name + '/std', ph_std),
                     (phenotype_name + '/beta_on_standardized_phenotype/stddev', beta_on_std_ph_std),
                     (phenotype_name + '/beta_on_standardized_phenotype/abs_mean', abs_mean_on_std_ph),
                     (phenotype_name + '/beta/stddev', beta_std),
                     (phenotype_name + '/beta/abs_mean', abs_mean),
                     (VCF_AC, new_ac),
                     (VCF_AN, new_an),
                     (VCF_AF, new_af)]:

            if k not in res:
                res[k] = []

            res[k].append(v)

    echo('Creating dataframe')
    res = pd.DataFrame(res).drop_duplicates(VARID_REF_ALT)

    echo('Merging back with variants')
    res = pd.merge(all_variants_in_phenotype_samples, res, on=VARID_REF_ALT, suffixes=['/original', ''])
    res = res.sort_values(phenotype_name + '/pvalue')

    return res


def test_each_variant_for_association_old(all_variants_in_phenotype_samples,
                                          ph_data_for_testing,
                                          phenotype_name=None,
                                          batch_size=1000,
                                          output_dir=None,
                                          n_threads=1,
                                          n_shuffles=0
                                          ):

    echo('Testing variants for association:', len(all_variants_in_phenotype_samples))

    ph_data_for_testing = ph_data_for_testing.sort_values(SAMPLE_ID)
    ph_std = np.std(ph_data_for_testing[phenotype_name])

    if n_shuffles > 0:
        echo('Generating shuffled phenotype values')
        permuted_ph_values = np.tile(ph_data_for_testing[phenotype_name],
                                     (n_shuffles + 1, 1))

        for i in range(1, n_shuffles + 1):
            np.random.shuffle(permuted_ph_values[i])

        permuted_ph_values = permuted_ph_values.T
        ph_values = pd.DataFrame(permuted_ph_values)
    else:
        ph_values = pd.DataFrame({phenotype_name: ph_data_for_testing[phenotype_name]})

    # all_variants_in_phenotype_samples = all_variants_in_phenotype_samples.drop_duplicates(subset=[VARID_REF_ALT, GENE_NAME])

    all_samples = list(ph_data_for_testing[SAMPLE_ID])
    all_samples_set = set(all_samples)
    s2i = dict((s, i) for i, s in enumerate(all_samples))

    final_res = None

    for batch_idx, gene_vars_batch in enumerate(batchify(all_variants_in_phenotype_samples, batch_size=batch_size)):
        echo('batch_idx:', batch_idx)

        gene_vars_batch_nr = gene_vars_batch.drop_duplicates(VARID_REF_ALT)

        gt_matrix = np.zeros((len(ph_data_for_testing), len(gene_vars_batch_nr)))
        v_idx = 0
        # to_skip = []
        batch_varids = list(gene_vars_batch_nr[VARID_REF_ALT])
        for _, v in gene_vars_batch_nr.iterrows():

            heterozygotes = set(v[HETEROZYGOTES].split(',')) & all_samples_set
            homozygotes = set(v[HOMOZYGOTES].split(',')) & all_samples_set

            for s in heterozygotes:
                gt_matrix[s2i[s], v_idx] = 1

            for s in homozygotes:
                gt_matrix[s2i[s], v_idx] = 2

            v_idx += 1

        echo('Computing correlations')

        r, prob = vcorrcoef(gt_matrix, ph_values)
        gt_std = np.std(gt_matrix, axis=0)

        res = {}

        res[VARID_REF_ALT] = batch_varids
        res[phenotype_name + '/r'] = r[0]
        res[phenotype_name + '/beta'] = r[0] * ph_std / gt_std
        res[phenotype_name + '/beta_on_standardized_phenotype'] = r[0] / np.std(gt_matrix, axis=0)
        res[phenotype_name + '/pvalue'] = prob[0]
        res[phenotype_name + '/std'] = ph_std

        if n_shuffles > 0:
            beta_on_std_ph = r / gt_std

            res[phenotype_name + '/beta_on_standardized_phenotype/stddev'] = np.std(beta_on_std_ph, axis=0)
            res[phenotype_name + '/beta_on_standardized_phenotype/abs_mean'] = np.mean(np.abs(beta_on_std_ph), axis=0)

            beta = beta_on_std_ph * ph_std
            res[phenotype_name + '/beta/stddev'] = np.std(beta, axis=0)
            res[phenotype_name + '/beta/abs_mean'] = np.mean(np.abs(beta), axis=0)

        echo('Creating dataframe')
        res = pd.DataFrame(res)
        echo('Merging back with variants')
        res = pd.merge(gene_vars_batch, res, on=VARID_REF_ALT)

        if final_res is None:
            final_res = res
        else:
            final_res = pd.concat([final_res, res], ignore_index=True)

    return final_res.sort_values(phenotype_name + '/pvalue')

def read_rare_variants(gene_info_fname,
                       max_missingness,
                       gnomad_coverage,
                       genes_to_test,
                       original_ph_data,
                       phenotype_name,
                       is_binary,
                       is_age_at_diagnosis):

    echo('Reading variant info from disk:', gene_info_fname)

    with open(gene_info_fname, 'rb') as in_f:
        rare_variants_full = pickle.load(in_f)

    echo('Total variants:', rare_variants_full.shape)

    min_AN = (1 - max_missingness) * np.max(rare_variants_full[VCF_AN])
    echo('Filtering variants with missingness more than:', max_missingness, ', min_AN:', min_AN)
    rare_variants_full = rare_variants_full[rare_variants_full[VCF_AN] >= min_AN]

    echo('total variants after filtering for missingness:', rare_variants_full.shape)

    # return rare_variants_full.copy(deep=True), None

    if gnomad_coverage is not None:

        if len(gnomad_coverage) != 2:
            echo('[ERROR] gnomad coverage should be a tuple of two numbers:', gnomad_coverage)
            exit(1)

        gnomad_min_frac, gnomad_coverage_times = gnomad_coverage

        gnomad_min_frac = float(gnomad_min_frac)
        gnomad_coverage_times = int(gnomad_coverage_times)

        echo(f'Filtering {len(rare_variants_full)} variants with gnomad coverage', gnomad_coverage_times, 'in at least',
             gnomad_min_frac, 'samples')

        rare_variants_full = rare_variants_full[
            (rare_variants_full['COVERAGE_HISTOGRAM|over_%d' % gnomad_coverage_times] > gnomad_min_frac)]
        echo('After filtering:', len(rare_variants_full), 'variants')

    if genes_to_test is not None:
        echo('Filtering variants in genes:', genes_to_test)
        rare_variants_full = rare_variants_full[rare_variants_full[GENE_NAME].isin(genes_to_test)]
        echo(len(rare_variants_full), 'variants')

    if len(rare_variants_full) == 0:
        echo('Skipping:', gene_info_fname)
        return None, None

    echo('Filtering phenotype data for exome samples')

    echo('variants:', len(rare_variants_full))
    all_samples = sorted(set([s for ss in rare_variants_full[ALL_SAMPLES] for s in ss.split(',')]))

    echo('n_samples:', len(all_samples))

    ph_data = original_ph_data[original_ph_data[SAMPLE_ID].isin(set(all_samples))].copy()

    echo('Exome samples with phenotype:', len(ph_data))

    echo('Phenotype', phenotype_name, 'is binary:', is_binary)
    if is_binary:
        if is_age_at_diagnosis:
            all_vals = ph_data[phenotype_name + '/original']
        else:
            all_vals = ph_data[phenotype_name]

        echo('Cases:', np.sum(all_vals), ', Controls:', len(all_vals) - np.sum(all_vals))
        # echo('Treating as continuous phenotype!!')
        # is_binary = False

    echo('Rare variants:', len(rare_variants_full))

    # rare_variants_in_phenotype_samples = filter_variants_for_rv_test(rare_variants_full,
    #                                                                  all_samples=set(ph_data[SAMPLE_ID]),
    #                                                                  remove_missing_pAI=False)

    echo('Total rare variants:', len(rare_variants_full))

    if len(rare_variants_full) == 0:
        echo('Skipping')
        return None, None

    rare_variants_in_phenotype_samples = rare_variants_full[[VCF_CHROM,
                                                             VCF_POS,
                                                             VCF_REF,
                                                             VCF_ALT,
                                                             VARID_REF_ALT,
                                                             GENE_NAME,
                                                             VCF_AC,
                                                             VCF_AF,
                                                             VCF_AN,
                                                             VCF_CONSEQUENCE,
                                                             PRIMATEAI_SCORE,
                                                             SPLICEAI_MAX_SCORE,
                                                             HOMOZYGOTES,
                                                             HETEROZYGOTES,
                                                             MISSING]].copy()

    echo('Returning variants and ph_data')
    return rare_variants_in_phenotype_samples, ph_data


def _test_each_variant_for_association(batch_params):
    try:
        (all_variants_in_phenotype_samples_fname,
        ph_data_for_testing_fname,
        phenotype_name,
        n_shuffles,
        cache) = batch_params

        all_variants_in_phenotype_samples = load_from_tmp_file(all_variants_in_phenotype_samples_fname)
        ph_data_for_testing = load_from_tmp_file(ph_data_for_testing_fname)

        res = test_each_variant_for_association(all_variants_in_phenotype_samples,
                                                ph_data_for_testing,
                                                phenotype_name=phenotype_name,
                                                n_shuffles=n_shuffles,
                                                cache=cache)
        res_fname = dump_to_tmp_file(res)
        log_max_memory_usage()

        return res_fname

    except Exception as e:
        print ('Caught exception in output worker thread (pid: %d):' % os.getpid())

        echo(e)
        if hasattr(open_log, 'logfile'):
            traceback.print_exc(file=open_log.logfile)

        traceback.print_exc()

        # print

        raise e

        # print('%d: %s' % (os.getpid(), traceback.format_exc()))
        # exit(1)




def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-i',
                        dest='input_fname',
                        help='pickle file the phenotype values for each subject [e.g ' +
                              UKB_DATA_PATH +
                              '/molecular_phenotypes.17_SEPT_2019/phenotypes_for_GWAS/LDL_direct.all_ethnicities/LDL_direct.all_ethnicities.both.med_corrected.phenotype_values.pickle]',
                        required=True)

    parser.add_argument('-p',
                        nargs='+',
                        dest='phenotype_name',
                        help='phenotype name',
                        required=True)

    parser.add_argument('-f', '--filters', nargs='+', dest='filters', help='filter exomes based on binary covariates')
    parser.add_argument('-e', '--to-exclude', nargs='+', dest='to_exclude', help='exclude exomes based on binary covariates')

    parser.add_argument('--genes', nargs='+', dest='genes_to_test', help='genes to test')

    parser.add_argument('--filter-samples', dest='filter_samples',
                        help=f'filter_samples: e.g. {ROOT_PATH}/ukbiobank/data/exomes/26_OCT_2020/qc/ukb200k_unrelated_white_europeans.pickle'
                        )

    parser.add_argument('--is-age-at-diagnosis',
                        dest='is_age_at_diagnosis',
                        action='store_true',
                        help='The input phenotype is age at diagnosis'
                        )

    parser.add_argument('--regress-out-age-at-diagnosis',
                        dest='regress_out_age_at_diagnosis',
                        action='store_true',
                        help='Regress out age at diagnosis from binary phenotype'
                        )

    parser.add_argument('--regress-out-ukb-common-variants',
                        dest='regress_out_ukb_common_variants',
                        type=str,
                        help='Regress out common variants from the finemapping pipeline. e.g.: ' +
                             ROOT_PATH + '/pfiziev/rare_variants/data/finemapping/finemapped_gwas/ukbiobank.v3/Calcium.finemapped.csv.gz'
                        )

    parser.add_argument('--regress-out-all-ukb-common-variants',
                        dest='regress_out_all_ukb_common_variants',
                        action='store_true',
                        default=False,
                        help='Regress out all common variants from the finemapping pipeline and not just the  index variants'
                        )

    parser.add_argument('--regress-only-non-coding-variants',
                        dest='regress_only_non_coding_variants',
                        action='store_true',
                        default=False,
                        help='Regress out only non-coding variants'
                        )

    parser.add_argument('--gwas-pvalue-threshold',
                        type=float,
                        dest='GWAS_PVALUE_THRESHOLD',
                        default=5e-8,
                        help='GWAS_PVALUE_THRESHOLD [%(default)s]')

    parser.add_argument('--rare-var-results',
                        dest='rv_fname',
                        help='pickle file rare variants results to subset the genes to test'
                        )

    parser.add_argument('--rare-var-fdr-threshold',
                        dest='rv_fdr_threshold',
                        help='rare variants FDR threshold for deleterious or ptv variants [%(default)s]',
                        default=1e-5,
                        type=float
                        )

    parser.add_argument('--filter-controls',
                        nargs='+',
                        dest='filter_controls',
                        help='list of columns to use for exclusion criteria from the set of controls'
                        )

    parser.add_argument('--regress-out',
                        nargs='+',
                        dest='regress_out',
                        help='list of covariates to regress out from the phenotype values before association'
                        )


    parser.add_argument('--gnomad-coverage', type=str, nargs='+', dest='gnomad_coverage', default=None, help='gnomad coverage tuple "min_fraction coverage", ex. "0.8 20"  [%(default)s]')
    parser.add_argument('--gnomad-popmax-af', type=float, dest='gnomad_popmax_af', default=None, help='maximum gnomad popmax allele frequency [%(default)s]')

    parser.add_argument('--max-pvalue', type=float, dest='max_pvalue', default=1e-5, help='maximum pvalue to store in the pickled dataframe  [%(default)s]')
    parser.add_argument('--n-shuffles', type=int, dest='n_shuffles', default=0, help='number of shufflings to estimate variance [%(default)s]')
    parser.add_argument('--test_only_GWAS_genes', action='store_true', dest='test_only_GWAS_genes', help='test_only_GWAS_genes')

    parser.add_argument('--n-threads', type=int, dest='n_threads', default=1, help='number of processes to use [%(default)s]')
    # parser.add_argument('--batch-size', type=int, dest='batch_size', default=1000, help='batch_size [%(default)s]')
    parser.add_argument('--max-missingness', type=float, dest='max_missingness', default=0.05, help='max_missingness [%(default)s]')

    parser.add_argument('-g',
                        dest='gene_info_fnames',
                        nargs='+',
                        help=f'genotype info file name [e.g.: {ROOT_PATH}/ukbiobank/data/exomes/sparse.liftover_hg19/all_gene_var_info.UKB_MAF_0_01_GNOMAD_POPMAX_MAF_0_01.full_annotation_for_RV_analysis.pickle]',
                        required=True)

    parser.add_argument('-o', dest='out_prefix', help='output file name', required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    args = parser.parse_args()

    log_name = args.out_prefix
    to_exclude = args.to_exclude
    genes_to_test = args.genes_to_test
    max_pvalue = args.max_pvalue
    n_shuffles = args.n_shuffles
    GWAS_PVALUE_THRESHOLD = args.GWAS_PVALUE_THRESHOLD

    test_only_GWAS_genes = args.test_only_GWAS_genes

    rv_fname = args.rv_fname
    rv_fdr_threshold = args.rv_fdr_threshold

    open_log(log_name + '.log', 'wt')

    echo('CMD:', ' '.join(sys.argv))

    echo('Parameters:\n' + pprint.pformat(args.__dict__))

    phenotype_names = args.phenotype_name

    out_prefix = args.out_prefix

    n_threads = args.n_threads
    filters = args.filters
    gnomad_coverage = args.gnomad_coverage
    is_age_at_diagnosis = args.is_age_at_diagnosis

    regress_out_age_at_diagnosis = args.regress_out_age_at_diagnosis

    regress_out_ukb_common_variants = args.regress_out_ukb_common_variants
    regress_out_all_ukb_common_variants = args.regress_out_all_ukb_common_variants
    regress_only_non_coding_variants = args.regress_only_non_coding_variants

    filter_controls = args.filter_controls
    regress_out = args.regress_out

    if n_threads > 1:
        echo('Setting the number of threads for mkl to 1 to prevent multiprocessing deadlocks')
        import mkl
        mkl.set_num_threads(1)

    if is_age_at_diagnosis:
        phenotype_names = [p + '/age_at_diagnosis' for p in phenotype_names]

    echo('Testing phenotype=', phenotype_names, ', out_prefix=', out_prefix)

    final_result = None
    final_result_out_fname = None

    max_missingness = args.max_missingness

    filter_samples = None
    if args.filter_samples is not None:
        filter_samples_fname = args.filter_samples
        echo('Reading sample ids to use:', filter_samples_fname)
        filter_samples = set(pd.read_pickle(filter_samples_fname)[SAMPLE_ID])
        echo('filter_samples:', len(filter_samples))

    # reading in corrected biomarkers
    echo('Reading phenotypes from:', args.input_fname)
    with open(args.input_fname, 'rb') as in_f:
        original_ph_data = pickle.load(in_f)

    original_ph_data = original_ph_data.dropna(subset=phenotype_names)
    phenotype_name = ','.join(phenotype_names)

    echo('phenotype_name:', phenotype_name)

    if len(phenotype_names) > 1:
        if not is_age_at_diagnosis:
            echo('[ERROR] combining multiple phenotype names is supported only for age_at_diagnosis')
            exit(1)

        def earliest_age(ages):

            ages = ages[ages >= 0]

            if len(ages) > 0:
                return np.min(ages)
            else:
                return -1.0

        original_ph_data[phenotype_name] = np.apply_along_axis(earliest_age, 1, original_ph_data[phenotype_names].to_numpy())

    if is_age_at_diagnosis:
        echo('[TEMPORARY WORKAROUND] Converting age of diagnosis to binary phenotypes')
        original_ph_data[phenotype_name + '/original_age_at_diagnosis'] = list(original_ph_data[phenotype_name])
        original_ph_data[phenotype_name] = (original_ph_data[phenotype_name] >= 0).astype(int)
        echo('cases:', np.sum(original_ph_data[phenotype_name]), ', controls:', len(original_ph_data) - np.sum(original_ph_data[phenotype_name]))

    if filters:
        original_ph_data = original_ph_data[(original_ph_data[filters] == 1).all(axis=1)]
        echo('Keeping samples with:', filters, ', samples:', len(original_ph_data))

    if to_exclude:
        original_ph_data = original_ph_data[~(original_ph_data[to_exclude] == 1).all(axis=1)]
        echo('Excluding samples with:', to_exclude, ', samples:', len(original_ph_data))

    is_binary = (len(set(original_ph_data[phenotype_name].dropna())) == 2)
    cols_to_keep = [SAMPLE_ID] + [c for c in list(original_ph_data) if c.startswith(phenotype_name)]
    if regress_out is not None:
        cols_to_keep += regress_out

    echo('Keeping phenotype columns:', cols_to_keep)
    original_ph_data = original_ph_data[cols_to_keep].copy(deep=True)
    gc.collect()

    if filter_controls is not None:
        if not is_binary:
            echo('[ERROR] filter_controls works only for binary traits')
            exit(1)

        echo('Removing controls based on:', filter_controls)

        for cov_name in filter_controls:

            filter_value = 1

            if cov_name.startswith('^'):
                cov_name = cov_name[1:]
                filter_value = 0

            echo('Filtering controls with covariate:', cov_name, '=', filter_value, ', before:', len(original_ph_data))
            if cov_name.endswith('/age_at_diagnosis'):
                if filter_value == 0:
                    echo('Excluding controls that were diagnosed with:', cov_name)
                    original_ph_data = original_ph_data[(original_ph_data[phenotype_name] == 1) | (original_ph_data[cov_name] < 0)]
                else:
                    echo('Keeping only controls that were diagnosed with:', cov_name)
                    original_ph_data = original_ph_data[(original_ph_data[phenotype_name] == 1) | (original_ph_data[cov_name] > 0)]
            else:
                original_ph_data = original_ph_data[(original_ph_data[phenotype_name] == 1) | (original_ph_data[cov_name] == filter_value)]

            echo('Filtering controls with covariate:', cov_name, '=', filter_value, ', after:', len(original_ph_data))

    if regress_out_age_at_diagnosis:

        echo('Regressing out age at diagnosis with 0 for controls')
        original_ph_data[phenotype_name + '/augm'] = original_ph_data[phenotype_name + '/original_age_at_diagnosis'].fillna(0)
        if regress_out is None:
            regress_out = []

        regress_out += [phenotype_name + '/augm']

    if regress_out_ukb_common_variants:
        from ukb_analysis import get_dosages_for_variants
        if not os.path.exists(regress_out_ukb_common_variants):
            echo('[WARNING] regress_out_ukb_common_variants file not found, skipping:', regress_out_ukb_common_variants)
        else:
            _, ukb_common_variants = read_finemapped_variants(regress_out_ukb_common_variants, GWAS_PVALUE_THRESHOLD=GWAS_PVALUE_THRESHOLD)

            echo('ukb_common_variants:', ukb_common_variants.shape)
            if len(ukb_common_variants) > 0:

                if regress_only_non_coding_variants:
                    ukb_common_variants = ukb_common_variants[~ukb_common_variants['assoc_type'].apply(
                        lambda x: len(set(x.split(';')) & {'coding', 'coding/ambiguous', 'splicing', 'splicing/ambiguous'}) > 0)]

                if not regress_out_all_ukb_common_variants:
                    ukb_common_variants = ukb_common_variants[ukb_common_variants['index_variant'] == ukb_common_variants[VARID_REF_ALT]]

                if len(ukb_common_variants) > 0:
                    bgen_data_chrom_to_fname_mapping = None
                    if RUNNING_ON == DNANEXUS:
                        bgen_data_chrom_to_fname_mapping = UKB_COMMON_VARIANTS_ON_DNANEXUS_MAPPING

                    ukb_common_variants_gt = get_dosages_for_variants(ukb_common_variants.sort_values([VCF_CHROM, VCF_POS]),
                                                                      sample_ids=None,
                                                                      rsid_label='variant',
                                                                      bgen_data_chrom_to_fname_mapping=bgen_data_chrom_to_fname_mapping)

                    varids = [c for c in ukb_common_variants_gt if c != SAMPLE_ID]
                    if len(varids) > 0:
                        echo('Merging genotypes of common variants with original_ph_data:', original_ph_data.shape, ukb_common_variants_gt.shape)
                        original_ph_data = pd.merge(original_ph_data, ukb_common_variants_gt, on=SAMPLE_ID)

                        del ukb_common_variants_gt

                        echo('After merge:', original_ph_data.shape)

                        if regress_out is None:
                            regress_out = []

                        regress_out += varids

    if regress_out is not None:
        original_ph_data = original_ph_data.dropna(subset=regress_out)

        echo('Regressing out covariates:', regress_out, ', data points:', len(original_ph_data))
        resid = get_residuals(original_ph_data, phenotype_name, regress_out, verbose=True, make_sample_ids_index=False)
        original_ph_data = pd.merge(original_ph_data.rename(columns={phenotype_name: phenotype_name + '/original'}), resid)

        del resid

        echo('original_ph_data after regressing out covariates:', original_ph_data.shape)

    if filter_samples is not None:
        echo('Filtering samples with phenotypes:', len(original_ph_data))
        original_ph_data = original_ph_data[original_ph_data[SAMPLE_ID].isin(filter_samples)]
        echo('After filtering:', len(original_ph_data), 'samples')

    cols_to_keep = [SAMPLE_ID, phenotype_name]

    phenotype_name_original = None
    if phenotype_name + '/original' in list(original_ph_data):
        phenotype_name_original = phenotype_name + '/original'
        cols_to_keep += [phenotype_name_original]

    echo('cols_to_keep:', cols_to_keep)
    original_ph_data = original_ph_data[cols_to_keep].dropna().copy(deep=True)

    if test_only_GWAS_genes:
        gwas_genes_df, _ = read_finemapped_variants(regress_out_ukb_common_variants)
        gwas_genes = set(gwas_genes_df[GENE_NAME])
        echo('GWAS genes:', len(gwas_genes))
        genes_to_test = sorted(gwas_genes)

    if rv_fname is not None:
        echo('Subsetting genes to genes that are significant by the rare variants test at p-value:', rv_fdr_threshold)

        rv_res = pd.read_pickle(rv_fname)
        rv_sign_genes = set(rv_res[(rv_res[f'ALL/del/carrier/pvalue/fdr_corr'] <= rv_fdr_threshold) |
                                   (rv_res[f'ALL/ptv/carrier/pvalue/fdr_corr'] <= rv_fdr_threshold)][GENE_NAME])

        echo('Significant genes by RV test:', len(rv_sign_genes), ', genes:',  sorted(rv_sign_genes))

        gwas_genes_df, _ = read_finemapped_variants(regress_out_ukb_common_variants)
        gwas_genes = set(gwas_genes_df[GENE_NAME])

        echo('GWAS genes:', len(gwas_genes))
        genes_to_test = sorted(gwas_genes & rv_sign_genes)
        echo('GWAS & RV genes:', len(genes_to_test), ', genes:', genes_to_test)

    header_printed = False

    cache = {}

    echo('original_ph_data:', original_ph_data.shape)

    with open_file(out_prefix + '.all_variants.csv.gz', 'w') as out_f_all:
        for fno, gene_info_fname in enumerate(sorted(args.gene_info_fnames)):

            gc.collect()

            rare_variants_in_phenotype_samples, ph_data = read_rare_variants(gene_info_fname,
                                                                             max_missingness,
                                                                             gnomad_coverage,
                                                                             genes_to_test,
                                                                             original_ph_data,
                                                                             phenotype_name,
                                                                             is_binary,
                                                                             is_age_at_diagnosis)

            if rare_variants_in_phenotype_samples is None:
                continue

            if is_age_at_diagnosis:
                echo('[TEMPORARY WORKAROUND] Setting is_binary to False for age at diagnosis phenotypes')
                is_binary = False

            output_dir = os.path.split(out_prefix)[0]
            if output_dir == '':
                output_dir = os.getcwd()

            batch_size = int(1 + len(rare_variants_in_phenotype_samples) / n_threads)
            ph_data_fname = dump_to_tmp_file(ph_data, output_dir=output_dir)

            batch_params = [(dump_to_tmp_file(var_batch.copy(), output_dir=output_dir),
                             ph_data_fname,
                             phenotype_name,
                             n_shuffles,
                             cache) for var_batch in batchify(rare_variants_in_phenotype_samples, batch_size)]

            echo('Starting pool. Batch size:', batch_size, ', n_batches:', len(batch_params))

            del rare_variants_in_phenotype_samples
            del ph_data
            echo('Running garbage collection')
            gc.collect()

            if n_threads > 1:

                with multiprocessing.Pool(processes=n_threads) as pool:

                    echo('Computing batch results')

                    batch_results_fnames = pool.map(_test_each_variant_for_association, batch_params)
                    echo('Batch results:', batch_results_fnames)

                    batch_results = list(map(load_from_tmp_file, batch_results_fnames))

                    echo('Closing pool of workers')

                    pool.terminate()
                    pool.join()

            else:

                batch_results_fnames = [_test_each_variant_for_association(batch_params[0])]
                batch_results = [load_from_tmp_file(batch_results_fnames[0])]

            res = pd.concat(batch_results, ignore_index=True, sort=True)

            for fname in [ph_data_fname] + [b[0] for b in batch_params] + batch_results_fnames:
                echo('Removing temp fname:', fname)
                os.unlink(fname)

            echo('Done')

            if res is None:
                continue

            res['phenotype'] = phenotype_name

            out_f_all.write(res[[c for c in list(res) if c not in [ALL_SAMPLES, HOMOZYGOTES, HETEROZYGOTES, MISSING]]].to_csv(header=(not header_printed), index=False, sep='\t'))

            header_printed = True

            del res


    echo('Done')
    close_log()


if __name__ == '__main__':
    main()
