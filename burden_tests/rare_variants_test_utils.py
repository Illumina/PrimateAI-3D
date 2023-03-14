import multiprocessing
import random
import tempfile
import traceback

import os

import numpy as np
import pandas as pd
import scipy.stats

from jutils import *
from constants import *

import statsmodels
# from ukb_analysis import *
import statsmodels.api as sm

CONST_LABEL = 'intercept'

def get_ukb_exome_variants_for_genes_from_sqlite(gene_names=None,
                                                 varids=None,
                                                 chrom=None,
                                                 max_af=None,
                                                 max_ac=None,
                                                 exome_db=ROOT_PATH + '/ukbiobank/data/exomes/26_OCT_2020/ukb.exome.db',
                                                 batch_size=100,
                                                 samples_to_subset=None,
                                                 decode_sample_id_fields=True):

    final_res = None

    import sqlite3
    con = sqlite3.connect(exome_db)

    if varids is not None:

        positions = set([int(v.split(':')[1]) for v in varids])

        echo('Extracting:', len(positions), 'positions')

        final_res = pd.read_sql('select * from variants where pos IN (%s)' % ', '.join([str(p)  for p in positions]), con)
        final_res = final_res[final_res['varid'].isin(varids)].copy()

        final_res = reformat_sql_result(final_res, decode_sample_id_fields, max_ac, max_af, samples_to_subset)

    elif gene_names is not None:

        echo('Getting all variants for', len(gene_names), 'genes')
        for b_idx, batch_genes in enumerate(batchify(gene_names, batch_size)):
            echo(b_idx * batch_size, 'genes processed')

            gene_vars = pd.read_sql(
                'select * from variants where symbol IN (%s)' % ', '.join(['"' + g + '"' for g in batch_genes]), con)

            # Be sure to close the connection

            gene_vars = reformat_sql_result(gene_vars, decode_sample_id_fields, max_ac, max_af, samples_to_subset)

            if final_res is None:
                final_res = gene_vars
            else:
                final_res = pd.concat([final_res, gene_vars], ignore_index=True)

    elif chrom is not None:
        echo('Fetching all variants for chromosome:', chrom, 'from :', exome_db)
        final_res = pd.read_sql('select * from variants where chrom = %s' % chrom, con)
        final_res = reformat_sql_result(final_res, decode_sample_id_fields, max_ac, max_af, samples_to_subset)

    else:
        echo('Fetching all variants from:', exome_db)
        final_res = pd.read_sql('select * from variants', con)
        final_res = reformat_sql_result(final_res, decode_sample_id_fields, max_ac, max_af, samples_to_subset)

    con.close()

    return final_res


def reformat_sql_result(gene_vars, decode_sample_id_fields, max_ac, max_af, samples_to_subset):
    def from_bytes(byte_ids, bit_len=3):
        ''' convert byte sequence to list of ints from 24-bit encoded values

        Decode byte-sequences to int lists e.g. b'\xe8\x03\x00\xd0\x07\x00' -> [1000, 2000]
        \xe8\x03\x00 is 3-byte encoded 1000, and \xd0\x07\x00 is 3-byte encoded 2000
        '''
        if byte_ids is None:
            return []

        length = len(byte_ids)
        return [str(int.from_bytes(byte_ids[i:i + bit_len], byteorder='little', signed=True)) for i in
                range(0, length, bit_len)]

    if decode_sample_id_fields:
        gene_vars['all_samples'] = gene_vars['all_samples'].apply(from_bytes)
        gene_vars['homs'] = gene_vars['homs'].apply(from_bytes)
        gene_vars['hets'] = gene_vars['hets'].apply(from_bytes)
        gene_vars['missing'] = gene_vars['missing'].apply(from_bytes)
    if max_ac is not None:
        gene_vars = gene_vars[gene_vars['ac'] <= max_ac].copy()
    if max_af is not None:
        gene_vars = gene_vars[gene_vars['af'] <= max_af].copy()
    if samples_to_subset is not None:
        for k in ['all_samples', 'hets', 'homs', 'missing']:
            gene_vars[k] = gene_vars[k].apply(lambda s: set(s) & samples_to_subset)

        gene_vars = gene_vars[gene_vars['all_samples'].apply(len) > 0].copy()

        gene_vars['ac'] = gene_vars['hets'].apply(len) + 2 * gene_vars['homs'].apply(len)
        gene_vars['an'] = 2 * len(samples_to_subset) - 2 * gene_vars['missing'].apply(len)
        gene_vars['af'] = gene_vars['ac'] / gene_vars['an']

    if decode_sample_id_fields:
        for k in ['all_samples', 'hets', 'homs', 'missing']:
            gene_vars[k] = gene_vars[k].apply(lambda s: ','.join(s))

    gene_vars = gene_vars.rename(columns={'rsid': VCF_RSID,
                                          'chrom': VCF_CHROM,
                                          'pos': VCF_POS,
                                          'ref': VCF_REF,
                                          'alt': VCF_ALT,
                                          'symbol': GENE_NAME,
                                          'primateai': PRIMATEAI_SCORE,
                                          'spliceai': SPLICEAI_MAX_SCORE,
                                          'af': VCF_AF,
                                          'ac': VCF_AC,
                                          'an': VCF_AN,
                                          'gnomad_af': VCF_AF + '/gnomAD',
                                          'homs': HOMOZYGOTES,
                                          'hets': HETEROZYGOTES,
                                          'varid': VARID_REF_ALT})

    gene_vars['MAF'] = np.where(gene_vars[VCF_AF] < 0.5, gene_vars[VCF_AF], 1 - gene_vars[VCF_AF])
    # gene_vars[VARID_REF_ALT] = gene_vars[VCF_CHROM] + ':' + gene_vars[VCF_POS].astype(str) + ':' + gene_vars[VCF_REF] + ':' + gene_vars[VCF_ALT]

    return gene_vars


def filter_variants_for_rv_test(rv_full,
                                gene_names=None,
                                max_AC=None,
                                consequence=None,
                                all_samples=None,
                                remove_missing_pAI=True,
                                flip_alt_majors=False,
                                pathogenicity_score_label=None,
                                verbose=True,
                                DELETERIOUS_BY_SPLICEAI=0.2):

    if verbose:
        echo('filter_variants_for_rv_test:', len(rv_full), 'variants')

    res = rv_full.copy()
    n_samples = lambda x: 0 if x.strip() == '' else len(x.split(','))

    if all_samples is not None:
        all_samples = set(all_samples)
        if verbose:
            echo('All samples:', len(all_samples))

        to_keep = res.apply(lambda x: len(set(x[ALL_SAMPLES].split(',')) & all_samples) > 0, axis=1)

        res = res[to_keep].copy()

        res[ALL_SAMPLES] = res[ALL_SAMPLES].apply(lambda x: ','.join(sorted(set(x.split(',')) & all_samples)))
        res[HOMOZYGOTES] = res[HOMOZYGOTES].apply(lambda x: ','.join(sorted(set(x.split(',')) & all_samples)))
        res[HETEROZYGOTES] = res[HETEROZYGOTES].apply(lambda x: ','.join(sorted(set(x.split(',')) & all_samples)))

        res[VCF_AC] = res[HETEROZYGOTES].apply(n_samples) + 2 * res[HOMOZYGOTES].apply(n_samples)

        res[VCF_AN] = 2 * len(all_samples)

        if MISSING in list(rv_full):
            if verbose:
                echo('Updating missing samples')
            res[MISSING] = res[MISSING].apply(lambda x: ','.join(sorted(set(x.split(',')) & all_samples)))
            res[VCF_AN] = res[VCF_AN] - 2 * res[MISSING].apply(n_samples)

        res[VCF_AF] = res[VCF_AC] / res[VCF_AN]

    if flip_alt_majors:
        if verbose:
            echo('Flipping alt alleles above 50% AF!')
        to_flip = res[res[VCF_AF] > 0.5].copy()

        if verbose:
            echo('Variants to flip:', len(to_flip))

        if len(to_flip) > 0:
            to_flip[HOMOZYGOTES] = to_flip[ALL_SAMPLES].apply(lambda x: ','.join(sorted(all_samples - set(x.split(',')))))

            to_flip[ALL_SAMPLES] = to_flip.apply(lambda x: ','.join(sorted(set(x[HOMOZYGOTES].split(',')) |
                                                                           set(x[HETEROZYGOTES].split(',')))),
                                                 axis=1)

            if MISSING in list(to_flip):
                if verbose:
                    echo('Removing missing samples')

                for k in [ALL_SAMPLES, HOMOZYGOTES, HETEROZYGOTES]:
                    to_flip[k] = to_flip.apply(lambda x: ','.join(sorted(set(x[k].split(',')) - set(x[MISSING].split(',')))),
                                               axis=1)

            to_flip[VCF_AC] = to_flip[HETEROZYGOTES].apply(n_samples) + 2 * to_flip[HOMOZYGOTES].apply(n_samples)

            to_flip = to_flip[to_flip[VCF_AC] > 0].copy()

            to_flip[VCF_AF] = to_flip[VCF_AC] / to_flip[VCF_AN]

            # echo('Merging:', res[res[VCF_AF] <= 0.5].shape, to_flip.shape)
            res = pd.concat([res[res[VCF_AF] <= 0.5].copy(), to_flip[list(res)]], ignore_index=True).sort_values([VCF_CHROM, VCF_POS])

    if gene_names is not None:
        res = res[res[GENE_NAME].isin(gene_names)]

    if max_AC is not None:
        res = res[res[VCF_AC] <= max_AC]

    if consequence is not None:

        if consequence in [VCF_MISSENSE_VARIANT, DELETERIOUS_VARIANT, MISSENSE_AND_PTVS]:
            if remove_missing_pAI:
                if verbose:
                    echo(consequence, ': Excluding missense variants without primateAI scores. Total variants before excluding:', len(res))
                res = res[~((res[VCF_CONSEQUENCE] == VCF_MISSENSE_VARIANT) & res[pathogenicity_score_label].isnull())]
                if verbose:
                    echo('After:', len(res))
            else:
                res[pathogenicity_score_label] = np.where((res[VCF_CONSEQUENCE] == VCF_MISSENSE_VARIANT) & res[pathogenicity_score_label].isnull(),
                                                           0,
                                                           res[pathogenicity_score_label])

        if consequence in [DELETERIOUS_VARIANT, MISSENSE_AND_PTVS]:
            res = res[(res[VCF_CONSEQUENCE].isin(ALL_PTV + [VCF_MISSENSE_VARIANT])) |
                      (res[SPLICEAI_MAX_SCORE] >= DELETERIOUS_BY_SPLICEAI)].copy()

            res[pathogenicity_score_label] = np.where((res[SPLICEAI_MAX_SCORE] >= DELETERIOUS_BY_SPLICEAI),
                                                       1,
                                                       res[pathogenicity_score_label])
        else:
            if type(consequence) is not list:
                consequence = [consequence]

            res = res[res[VCF_CONSEQUENCE].isin(consequence)]

    return res.copy()


def get_samples(vcfdata, homozygotes_only=False, heterozygotes_only=False):
    tag = ALL_SAMPLES
    if homozygotes_only:
        tag = HOMOZYGOTES
    elif heterozygotes_only:
        tag = HETEROZYGOTES

    samples = sorted(set(sid for sids in vcfdata[tag] for sid in sids.split(',')))
    return samples


def get_carriers(phenotype_data, gene_variants, max_AC=None):

    gene_variants_above_threshold = gene_variants
    if max_AC is not None:
        gene_variants_above_threshold = gene_variants[gene_variants[VCF_AC] <= max_AC]

    fgr_sample_ids = get_samples(gene_variants_above_threshold)

    fgr_samples_phenotypes = pd.merge(phenotype_data, pd.DataFrame({SAMPLE_ID: fgr_sample_ids}))

    return set(fgr_samples_phenotypes[SAMPLE_ID])


def generate_random_variants(gene_variants, sample_ids, n=1):
    import random

    res = []

    for i in range(n):

        rv = gene_variants.copy()

        all_samples = []

        for ac in gene_variants[VCF_AC]:

            all_samples.append(','.join(random.sample(sample_ids, ac)))

        rv[ALL_SAMPLES] = all_samples

        res.append(rv)

    return res


class NullResult:
    pass


def rv_test_gene_variants(gene_variants, ph_data, ph_name, test_meds=False, test_PRS=None, is_binary=False):

    cur_results = {}

    echo_debug('Getting carriers')
    carriers = get_carriers(ph_data, gene_variants)

    echo_debug('Setting carrier column')
    ph_data['carrier'] = ph_data[SAMPLE_ID].isin(set(carriers)).astype(int)
    cols_to_skip = [SAMPLE_ID, ph_name]

    if is_binary:
        res_labels = ['carrier']
        n_carrier_cases = np.sum(ph_data[ph_data['carrier'] == 1][ph_name])
        n_carrier_ctrls = len(carriers) - n_carrier_cases
        n_all_cases = np.sum(ph_data[ph_name])
        n_all_ctrls = len(ph_data) - n_all_cases

    else:
        res_labels = ['carrier', CONST_LABEL]

    if test_meds:

        for med_label, med_name in zip(*get_med_columns(ph_data)):
            ph_data['carrier_X_' + med_label] = ph_data['carrier'] * ph_data[med_label]
            res_labels.extend([med_label, 'carrier_X_' + med_label])

            cur_results['n_carriers/' + med_name] = np.sum(ph_data['carrier_X_' + med_label])

    if test_PRS is not None:
        ph_data['carrier_X_PRS'] = ph_data['carrier'] * ph_data[test_PRS]
        res_labels.extend([test_PRS, 'carrier_X_PRS'])

    if is_binary:
        MIN_SAMPLES_FOR_REGRESSION = 2
    else:
        MIN_SAMPLES_FOR_REGRESSION = 10

    if len(carriers) >= MIN_SAMPLES_FOR_REGRESSION:
        echo_debug('Creating regression model, ', len(carriers))

        predictors = ph_data[[c for c in list(ph_data) if c not in cols_to_skip]]

        if test_meds:
            # remove binary columns with too few samples
            to_remove = []

            for pred_name in list(predictors):
                if pred_name.startswith('carrier_X_on_med.') or pred_name.startswith('on_med.'):
                    n_pred = np.sum(predictors[pred_name])

                    if n_pred < MIN_SAMPLES_FOR_REGRESSION:
                        to_remove.append(pred_name)

            predictors = predictors[[p for p in list(predictors) if p not in to_remove]]

        if is_binary:
            # regression_model = sm.Logit(ph_data[ph_name], predictors)
            regression_model = None
        else:
            regression_model = sm.OLS(ph_data[ph_name], predictors)

        echo_debug('Fitting model')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            echo_debug('regression_model.fit()')
            if is_binary:
                regression_result = NullResult()
                regression_result.prsquared = 0

                regression_result.pvalues = {'carrier':
                                                 chi2_or_fisher_test([[n_carrier_cases, n_all_cases - n_carrier_cases],
                                                                      [n_carrier_ctrls, n_all_ctrls - n_carrier_ctrls]])}
                regression_result.params = {'carrier': odds_ratio(n_carrier_cases, n_all_cases, n_carrier_ctrls, n_all_ctrls)}

            else:
                regression_result = regression_model.fit()

        echo_debug('Getting R2')

        if is_binary:
            rsquared_adj = regression_result.prsquared
        else:
            rsquared_adj = regression_result.rsquared_adj

        for k in res_labels:
            echo_debug('getting results for', k)
            cur_results[k + '/' + 'beta'] = regression_result.params.get(k, 0)
            cur_results[k + '/' + 'pvalue'] = regression_result.pvalues.get(k, 1)

            if np.isnan(cur_results[k + '/' + 'pvalue']):
                # echo(k, cur_results[k + '/' + 'beta'], cur_results[k + '/' + 'pvalue'])
                cur_results[k + '/' + 'pvalue'] = 1
                cur_results[k + '/' + 'beta'] = 0

        # cur_results[sort_by_pvalue_label + '/sort_by_pvalue'] = regression_result.pvalues[sort_by_pvalue_label]

    else:

        rsquared_adj = 0

        for k in res_labels:
            cur_results[k + '/' + 'beta'] = 0
            cur_results[k + '/' + 'pvalue'] = 1

        # cur_results[sort_by_pvalue_label + '/sort_by_pvalue'] = 1

    cur_results['n_carriers/total'] = np.sum(ph_data['carrier'])
    cur_results['rsquared_adj'] = rsquared_adj

    if is_binary:
        cur_results['n_carrier_cases'] = n_carrier_cases
        cur_results['n_carrier_ctrls'] = n_carrier_ctrls
        cur_results['n_all_cases'] = n_all_cases
        cur_results['n_all_ctrls'] = n_all_ctrls

    echo_debug('returning results')

    return cur_results


def get_med_columns(ph_data, meds_to_keep=None):

    med_labels_to_test = [c for c in list(ph_data) if c.startswith('on_med.') and '.1st_visit.' in c]
    med_names_to_test = [c.split('.')[1] for c in med_labels_to_test]

    if meds_to_keep is not None:

        _med_labels_to_test = []
        _med_names_to_test = []

        for med_name, med_label in zip(med_names_to_test, med_labels_to_test):
            if med_name in meds_to_keep:
                _med_names_to_test.append(med_name)
                _med_labels_to_test.append(med_label)

        med_labels_to_test = _med_labels_to_test
        med_names_to_test = _med_names_to_test

    return med_labels_to_test, med_names_to_test


def rv_test_gene(x,
                 ph_data,
                 ph_name,
                 key_prefix,
                 sort_by_pvalue_label,
                 test_meds,
                 test_PRS,
                 find_best_AC_threshold,
                 find_best_pAI_threshold,
                 pAI_thresholds_to_test,
                 multiple_testing_correction,
                 is_binary=False,
                 MAX_pAI_THRESHOLDS_TO_TEST=25,
                 meds_to_keep=None,
                 pathogenicity_score_label=None):


    # echo(rv_test_gene.cnt, 'genes tested.')

    gene_name = x.iloc[0][GENE_NAME]

    if rv_test_gene.cnt % 10 == 0:
        echo(rv_test_gene.cnt, 'genes tested. current gene:', gene_name, ', vars:',  len(x))

    rv_test_gene.cnt += 1
    echo_debug(gene_name, 'Getting AC thresholds')

    if find_best_AC_threshold:
        ac_thresholds = sorted(x[VCF_AC].unique())
    else:
        ac_thresholds = [np.max(x[VCF_AC].unique())]

    echo_debug(gene_name, 'Getting pAI thresholds')

    # echo('pAI_thresholds:', len(pAI_thresholds), pAI_thresholds)
    # echo(gene_name, len(ac_thresholds), len(pAI_thresholds), len(ac_thresholds) * len(pAI_thresholds))

    results_array = []
    echo_debug(gene_name, 'iterating over AC thresholds')

    for max_AC in ac_thresholds:
        echo_debug('Getting variants with AC threshold', max_AC)
        gene_variants = x[x[VCF_AC] <= max_AC].sort_values(pathogenicity_score_label)

        if find_best_pAI_threshold:

            pAI_thresholds = sorted(gene_variants[pathogenicity_score_label].unique())

            if len(pAI_thresholds) > MAX_pAI_THRESHOLDS_TO_TEST:
                pAI_thresholds = [np.quantile(pAI_thresholds,
                                              i / (MAX_pAI_THRESHOLDS_TO_TEST - 1))
                                  for i in range(MAX_pAI_THRESHOLDS_TO_TEST)]

        else:
            pAI_thresholds = [0]

        if pAI_thresholds_to_test is not None:
            pAI_thresholds = pAI_thresholds_to_test

        echo_debug(gene_name, 'interating over pAI thresholds')
        for best_pAI_threshold in pAI_thresholds:

            echo_debug(gene_name, 'getting variants over pAI:', best_pAI_threshold)

            gene_variants_above_pAI_t = gene_variants[gene_variants[pathogenicity_score_label] >= best_pAI_threshold]

            echo_debug(gene_name, 'Testing variants')
            cur_results = rv_test_gene_variants(gene_variants_above_pAI_t,
                                                ph_data,
                                                ph_name,
                                                test_meds=test_meds,
                                                test_PRS=test_PRS,
                                                is_binary=is_binary)

            cur_results['best_AC'] = max_AC
            cur_results['best_pAI_threshold'] = best_pAI_threshold

            results_array.append(cur_results)

    if multiple_testing_correction == 'fdr':

        pvalue_keys = sorted(set(k for r in results_array for k in r if k.endswith('pvalue')))
        # echo(pvalue_keys)
        for pvalue_key in pvalue_keys:

            pvalues = [r[pvalue_key] for r in results_array]
            echo_debug(gene_name, 'fdr correcting pvalues', pvalue_key)

            _, pvalues = statsmodels.stats.multitest.fdrcorrection(pvalues)

            for r, p in zip(results_array, pvalues):
                r[pvalue_key + '/fdr_corr'] = p

    # echo(results_array[0])

    # results = min(results_array, key=lambda r: r[sort_by_pvalue_label + '/sort_by_pvalue/fdr_corr'])
    def get_best_result(results_array, sort_by_pvalue_label, key_prefix, find_best_pAI_threshold):
        sorted_results_array = sorted(results_array,
                                      key=lambda r: (r[sort_by_pvalue_label + '/pvalue/fdr_corr'],
                                                     r[sort_by_pvalue_label + '/pvalue'],
                                                     r['best_AC'],
                                                     -r['best_pAI_threshold']))

        results = sorted_results_array[0]
        results['n_tests'] = len(results_array)

        results = dict((key_prefix + '/' + k, results[k]) for k in sorted(results))

        if not find_best_pAI_threshold:
            del results[key_prefix + '/best_pAI_threshold']

        results[key_prefix + '/sort_by'] = key_prefix + '/' + sort_by_pvalue_label + '/pvalue/fdr_corr'

        return results

    if test_meds:

        results = {}

        med_labels_to_test, med_names_to_test = get_med_columns(ph_data, meds_to_keep=meds_to_keep)

        for med_name, med_label in zip(med_names_to_test, med_labels_to_test):

            med_key_prefix = med_name + key_prefix
            sort_by_pvalue_label = 'carrier_X_' + med_label

            med_results = get_best_result(results_array, sort_by_pvalue_label, med_key_prefix, find_best_pAI_threshold)
            med_results[med_key_prefix + '/total_on_meds'] = np.sum(ph_data[med_label])

            for _mn, _ml in zip(med_names_to_test, med_labels_to_test):
                med_results = dict((k.replace(_ml, _mn), med_results[k]) for k in med_results)

            med_results[med_key_prefix + '/sort_by'] = med_results[med_key_prefix + '/sort_by'].replace(med_label, med_name)

            results[med_name] = med_results

    else:

        results = get_best_result(results_array, sort_by_pvalue_label, key_prefix, find_best_pAI_threshold)
        echo_debug(gene_name, 'returning results')
        results = pd.Series(results)

    return results

rv_test_gene.cnt = 0


def compute_results(batch_params):
    try:
        return _compute_results(batch_params)

    except Exception as e:
        print ('Caught exception in output worker thread (pid: %d):' % os.getpid())

        echo(e)
        if hasattr(open_log, 'logfile'):
            traceback.print_exc(file=open_log.logfile)

        traceback.print_exc()

        raise e

        # print('%d: %s' % (os.getpid(), traceback.format_exc()))
        # exit(1)



def _compute_results(batch_params):

    (batch_filtered_rv_fname,
     ph_data_fname,
     ph_name,
     key_prefix,
     sort_by_pvalue_label,
     test_meds,
     test_PRS,
     find_best_AC_threshold,
     find_best_pAI_threshold,
     multiple_testing_correction,
     output_dir,
     is_binary
     ) = batch_params

    batch_filtered_rv = load_from_tmp_file(batch_filtered_rv_fname)

    ph_data = load_from_tmp_file(ph_data_fname)

    gene_names = sorted(set(batch_filtered_rv[GENE_NAME]))

    echo('Processing batch with:', len(gene_names), 'genes:', gene_names[0], '-', gene_names[-1])

    echo(len(batch_filtered_rv), 'variants')

    rv_test_gene.cnt = 0

    results_array = [rv_test_gene(batch_filtered_rv[batch_filtered_rv[GENE_NAME] == gn],
                                  ph_data,
                                  ph_name,
                                  key_prefix,
                                  sort_by_pvalue_label,
                                  test_meds,
                                  test_PRS,
                                  find_best_AC_threshold,
                                  find_best_pAI_threshold,
                                  multiple_testing_correction,
                                  is_binary) for gn in gene_names]

    if test_meds:
        results = {}

        for gene_results, gene_name in zip(results_array, gene_names):
            add_new_med_result(gene_name, gene_results, results)

        convert_med_results_to_dataframes(results, key_prefix)

    else:

        echo('Adding gene name column')
        for r, gn in zip(results_array, gene_names):
            r[GENE_NAME] = gn

        echo('Converting to dataframe')
        full_sort_by_pvalue_label = key_prefix + '/' + sort_by_pvalue_label + '/pvalue/fdr_corr'
        results = pd.DataFrame(results_array).sort_values(full_sort_by_pvalue_label)
        results = results[[GENE_NAME, full_sort_by_pvalue_label] + [c for c in list(results)
                                                                    if c not in [GENE_NAME, full_sort_by_pvalue_label]]]

    echo('Sending back results')

    return dump_to_tmp_file(results, output_dir=output_dir)


def add_new_med_result(gene_name, gene_results, results):

    for med_name in gene_results:
        med_gene_results = gene_results[med_name]
        med_gene_results[GENE_NAME] = gene_name

        if med_name not in results:
            results[med_name] = dict((k, []) for k in med_gene_results)

        for k in med_gene_results:
            results[med_name][k].append(med_gene_results[k])


def rv_test(ph_data,
            ph_name,
            filtered_rv,
            var_label,
            test_meds=False,
            test_PRS=None,
            multiple_testing_correction='fdr',
            find_best_AC_threshold=True,
            find_best_pAI_threshold=True,
            n_threads=1,
            output_dir=None,
            is_binary=False,
            batch_size=10,
            MAX_pAI_THRESHOLDS_TO_TEST=25,
            meds_to_keep=None,
            phenotype_name_original=None,
            pathogenicity_score_label=None
            ):

    """ This performs a correlation based test for rare variants enrichment instead of fitting linear or logistic regression models """

    echo('output_dir:', output_dir)
    echo('phenotype:', ph_name)
    echo('var_label:', var_label)
    echo('variants:', len(filtered_rv))
    echo('samples:', len(ph_data))
    echo('test_meds:', test_meds)
    echo('test_PRS:', test_PRS)
    echo('multiple_testing_correction:', multiple_testing_correction)
    echo('find_best_AC_threshold:', find_best_AC_threshold)
    echo('find_best_pAI_threshold:', find_best_pAI_threshold)
    echo('n_threads:', n_threads)
    echo('is_binary:', is_binary)
    echo('batch_size:', batch_size)
    echo('MAX_pAI_THRESHOLDS_TO_TEST:', MAX_pAI_THRESHOLDS_TO_TEST)
    echo('pathogenicity_score_label:', pathogenicity_score_label)
    echo('meds_to_keep:', meds_to_keep)

    filtered_rv = filtered_rv[[GENE_NAME, VCF_AC, ALL_SAMPLES, pathogenicity_score_label, VARID_REF_ALT]].copy()
    filtered_rv[pathogenicity_score_label] = filtered_rv[pathogenicity_score_label].fillna(1)

    results = None

    if test_meds:
        # raise NotImplemented('testing medications is not implemented')

        key_prefix = '/' + var_label

        all_med_columns = [c for c in list(ph_data) if c.startswith('on_med.') and '.1st_visit.' in c]

        cols_to_copy = [SAMPLE_ID, ph_name] + all_med_columns

        sort_by_pvalue_label = None
        results = {}
        # echo('med_name:', med_name, ', med_label:', med_label)
    else:

        key_prefix = 'ALL/' + var_label
        cols_to_copy = [SAMPLE_ID, ph_name]
        if phenotype_name_original is not None:
            cols_to_copy += [phenotype_name_original]

        sort_by_pvalue_label = 'carrier'

    if test_PRS is not None:
        raise NotImplemented('Testing PRS interactions is not implemented')

        cols_to_copy.append(test_PRS)
        sort_by_pvalue_label = 'carrier_X_PRS'

    sort_by_label = key_prefix + '/%s/pvalue/fdr_corr' % sort_by_pvalue_label
    echo('sort_by_pvalue_label:', sort_by_pvalue_label)

    ph_data = ph_data[cols_to_copy].copy()

    # normalize phenotype to mean 0 and variance 1
    # ph_data[ph_name] = scipy.stats.zscore(ph_data[ph_name])

    echo('n_genes:', len(filtered_rv.drop_duplicates(GENE_NAME)))

    rv_test_gene.cnt = 0

    if test_meds:
        # raise NotImplemented('testing medications is not implemented')

        for gene_name, gene_variants in filtered_rv.groupby(GENE_NAME):

            gene_results = rv_test_gene(gene_variants,
                                        ph_data,
                                        ph_name,
                                        key_prefix,
                                        sort_by_pvalue_label,
                                        test_meds,
                                        test_PRS,
                                        find_best_AC_threshold,
                                        find_best_pAI_threshold,
                                        multiple_testing_correction,
                                        MAX_pAI_THRESHOLDS_TO_TEST=MAX_pAI_THRESHOLDS_TO_TEST,
                                        meds_to_keep=meds_to_keep,
                                        pathogenicity_score_label=pathogenicity_score_label
                                        )

            add_new_med_result(gene_name, gene_results, results)

        convert_med_results_to_dataframes(results, key_prefix)

    else:

        results = {GENE_NAME: [],
                   key_prefix + '/carrier/pvalue': [],
                   key_prefix + '/carrier/pvalue/fdr_corr': [],
                   key_prefix + '/sort_by': []}

        if is_binary:
            for k in ['carrier/odds_ratio', 'carrier/n_cases', 'carrier/n_controls', 'all/n_cases', 'all/n_controls']:
                results[key_prefix + '/' + k] = []

        if find_best_pAI_threshold:
            results[key_prefix + '/best_pAI_threshold'] = []

        all_genes = sorted(set(filtered_rv[GENE_NAME]))

        batch_genes = [all_genes[i : i + batch_size] for i in range(0, len(all_genes), batch_size)]

        echo('batches:', len(batch_genes))

        for batch_idx, genes in enumerate(batch_genes):

            echo('Batch:', batch_idx, ', genes:', genes)

            carrier_table, carrier_columns = create_carrier_table(filtered_rv[filtered_rv[GENE_NAME].isin(genes)],
                                                                  ph_data,
                                                                  ph_name,
                                                                  find_best_AC_threshold,
                                                                  find_best_pAI_threshold,
                                                                  MAX_pAI_THRESHOLDS_TO_TEST=MAX_pAI_THRESHOLDS_TO_TEST,
                                                                  phenotype_name_original=phenotype_name_original,
                                                                  pathogenicity_score_label=pathogenicity_score_label)

            # carrier_columns = [c for c in list(carrier_table) if c not in [SAMPLE_ID, ph_name, phenotype_name_original]]

            if is_binary:
                echo('Computing chi^2 tests for', len(carrier_columns), 'columns')

                n_all_cases = np.sum(carrier_table[ph_name])
                n_all_ctrls = len(carrier_table) - n_all_cases

                pval = {'index': [],
                        ph_name: [],
                        'carrier/odds_ratio': [],
                        'carrier/n_cases': [],
                        'carrier/n_controls': [],
                        'all/n_cases': [],
                        'all/n_controls': []}

                for c in carrier_columns:
                    pval['index'].append(c)
                    n_carriers = np.sum(carrier_table[c])

                    n_carrier_cases = np.sum(carrier_table[c] * carrier_table[ph_name])
                    n_carrier_ctrls = n_carriers - n_carrier_cases

                    pval['carrier/n_cases'].append(n_carrier_cases)
                    pval['carrier/n_controls'].append(n_carrier_ctrls)
                    pval['all/n_cases'].append(n_all_cases)
                    pval['all/n_controls'].append(n_all_ctrls)

                    if n_all_ctrls == 0 or n_all_cases == 0 or n_carriers == 0 or (n_carrier_cases == 0 and n_carrier_ctrls == 0):
                        pval[ph_name].append(1)
                        pval['carrier/odds_ratio'].append(0)

                    else:

                        try:
                            pval[ph_name].append(chi2_or_fisher_test([[n_carrier_cases, n_all_cases - n_carrier_cases],
                                                                      [n_carrier_ctrls, n_all_ctrls - n_carrier_ctrls]]))
                            pval['carrier/odds_ratio'].append(
                                odds_ratio(n_carrier_cases, n_all_cases, n_carrier_ctrls, n_all_ctrls))
                        except:
                            echo('PROBLEM with chi^2 test:',
                                 c,
                                 [[n_carrier_cases, n_all_cases - n_carrier_cases],
                                  [n_carrier_ctrls, n_all_ctrls - n_carrier_ctrls]],
                                 n_carriers,
                                 n_carrier_cases,
                                 n_carrier_ctrls,
                                 n_all_cases,
                                 n_all_ctrls)

                            pval[ph_name].append(1)
                            pval['carrier/odds_ratio'].append(0)

                pval = pd.DataFrame(pval).set_index('index')

            else:
                echo('Computing correlations for', len(carrier_columns), 'columns')

                corr, pval = vcorrcoef(ph_data[ph_name].values[:, None],
                                       carrier_table,
                                       axis='columns')

                pval = pd.DataFrame(pval, columns=[ph_name], index=carrier_columns)


                # corr, pval = vcorrcoef(carrier_table[[ph_name]],
                #                        carrier_table[carrier_columns],
                #                        axis='columns')

            cast_type = {'str': str,
                         'int': int,
                         'float': float}

            echo('Finding best enrichment for each gene')
            for gene in genes:

                gene_key = key_value_to_string(GENE_NAME, gene)

                gene_rows_labels = [c for c in carrier_columns if c.startswith(gene_key)]

                gene_rows = pval.loc[gene_rows_labels].sort_values(ph_name).reset_index().dropna()
                _, fdr_corrected_pvalues = statsmodels.stats.multitest.fdrcorrection(gene_rows[ph_name])

                gene_rows['fdr'] = fdr_corrected_pvalues

                best_pvalue_row = gene_rows.iloc[0]

                row_info = best_pvalue_row['index'].split(';')
                # echo(row_info)

                results[GENE_NAME].append(gene)

                for kv in row_info:

                    key, vt = kv.split('=')

                    if key == GENE_NAME:
                        continue

                    value_type, value = vt.split('|')

                    value = cast_type[value_type](value)

                    res_key = key_prefix + '/' + key
                    if res_key not in results:
                        results[res_key] = []

                    results[res_key].append(value)

                if is_binary:
                    for k in ['carrier/odds_ratio', 'carrier/n_cases', 'carrier/n_controls', 'all/n_cases', 'all/n_controls']:
                        results[key_prefix + '/' + k].append(best_pvalue_row[k])

                results[key_prefix + '/carrier/pvalue'].append(best_pvalue_row[ph_name])
                results[key_prefix + '/carrier/pvalue/fdr_corr'].append(best_pvalue_row['fdr'])
                results[key_prefix + '/sort_by'].append(key_prefix + '/%s/pvalue/fdr_corr' % sort_by_pvalue_label)

        results = pd.DataFrame(results).sort_values(sort_by_label)

    echo('Done')

    return results


def rv_test_parallel(ph_data,
                     ph_name,
                     filtered_rv,
                     var_label,
                     test_meds=False,
                     test_PRS=None,
                     multiple_testing_correction='fdr',
                     find_best_AC_threshold=True,
                     find_best_pAI_threshold=True,
                     pAI_thresholds_to_test=None,
                     n_threads=1,
                     output_dir=None,
                     is_binary=False,
                     batch_size=10,
                     MAX_pAI_THRESHOLDS_TO_TEST=25,
                     meds_to_keep=None,
                     phenotype_name_original=None,
                     pathogenicity_score_label=None,
                     n_randomizations=0,
                     random_seed=None,
                     adaptive_fdr=False,
                     is_age_at_diagnosis=False,
                     skip_randomizing_intermediate_significant=False,
                     approximate_null_distribution=False,
                     n_samples_for_approximation=10000,
                     parallelize_within_genes=False,
                     min_variants_per_gene=2,
                     pvalue_threshold_for_permutation_tests=1e-5
                     ):

    """ This performs a correlation based test for rare variants enrichment instead of fitting linear or logistic regression models """

    echo('output_dir:', output_dir)
    echo('phenotype:', ph_name)
    echo('var_label:', var_label)
    echo('variants:', len(filtered_rv))
    echo('samples:', len(ph_data))
    echo('test_meds:', test_meds)
    echo('test_PRS:', test_PRS)
    echo('multiple_testing_correction:', multiple_testing_correction)
    echo('find_best_AC_threshold:', find_best_AC_threshold)
    echo('find_best_pAI_threshold:', find_best_pAI_threshold)
    echo('pAI_thresholds_to_test:', pAI_thresholds_to_test)
    echo('n_threads:', n_threads)
    echo('is_binary:', is_binary)
    echo('batch_size:', batch_size)
    echo('MAX_pAI_THRESHOLDS_TO_TEST:', MAX_pAI_THRESHOLDS_TO_TEST)
    echo('pathogenicity_score_label:', pathogenicity_score_label)
    echo('meds_to_keep:', meds_to_keep)
    echo('n_randomizations:', n_randomizations)
    echo('random_seed:', random_seed)
    echo('adaptive_fdr:', adaptive_fdr)
    echo('skip_randomizing_intermediate_significant:', skip_randomizing_intermediate_significant)
    echo('approximate_null_distribution:', approximate_null_distribution)
    echo('n_samples_for_approximation:', n_samples_for_approximation)
    echo('parallelize_within_genes:', parallelize_within_genes)
    echo('min_variants_per_gene:', min_variants_per_gene)
    echo('pvalue_threshold_for_permutation_tests:', pvalue_threshold_for_permutation_tests)

    filtered_rv = filtered_rv.copy()
    filtered_rv[pathogenicity_score_label] = filtered_rv[pathogenicity_score_label].fillna(1)

    if test_meds:
        # raise NotImplemented('testing medications is not implemented')

        key_prefix = '/' + var_label

        all_med_columns = [c for c in list(ph_data) if c.startswith('on_med.') and '.1st_visit.' in c]

        cols_to_copy = [SAMPLE_ID, ph_name] + all_med_columns

        sort_by_pvalue_label = None
        # echo('med_name:', med_name, ', med_label:', med_label)
    else:

        key_prefix = 'ALL/' + var_label
        cols_to_copy = [SAMPLE_ID, ph_name]

        if CONST_LABEL in list(ph_data):
            cols_to_copy += [CONST_LABEL]

        if phenotype_name_original is not None:
            cols_to_copy += [phenotype_name_original]

        if is_age_at_diagnosis and AGE in list(ph_data):
            cols_to_copy += [AGE]

        sort_by_pvalue_label = 'carrier'

    if test_PRS is not None:
        raise NotImplemented('Testing PRS interactions is not implemented')

        # cols_to_copy.append(test_PRS)
        # sort_by_pvalue_label = 'carrier_X_PRS'

    ph_data = ph_data[cols_to_copy].copy()
    ph_data_fname = dump_to_tmp_file(ph_data, output_dir=output_dir)
    gc.collect()

    import multiprocessing

    all_genes = sorted(set(filtered_rv[GENE_NAME]))

    if parallelize_within_genes:
        # if parallelization is done within each gene, then put all genes here in a single batch
        thread_batch_size = len(all_genes)
        batch_genes = [all_genes]
    else:
        # otherwise:
        if len(all_genes) < n_threads:
            # place each gene in a separate batch
            thread_batch_size = 1
            batch_genes = [all_genes[i: i + 1] for i in range(len(all_genes))]
        else:
            # split the genes into equal size batches
            thread_batch_size = int(len(all_genes) / n_threads)

            batch_genes = [all_genes[i * thread_batch_size:
                                     ((i + 1) * thread_batch_size if i < n_threads - 1 else len(all_genes))]
                           for i in range(n_threads)]

    echo('Batch sizes:', list(map(len, batch_genes)), ', total=', sum(map(len, batch_genes)))

    batch_params = [(dump_to_tmp_file(filtered_rv[filtered_rv[GENE_NAME].isin(gn)].copy(), output_dir=output_dir),
                     MAX_pAI_THRESHOLDS_TO_TEST,
                     batch_size,
                     find_best_AC_threshold,
                     find_best_pAI_threshold,
                     pAI_thresholds_to_test,
                     is_binary,
                     meds_to_keep,
                     multiple_testing_correction,
                     pathogenicity_score_label,
                     ph_data_fname,
                     ph_name,
                     phenotype_name_original,
                     test_PRS,
                     test_meds,
                     var_label,
                     key_prefix,
                     sort_by_pvalue_label,
                     output_dir,
                     n_randomizations,
                     random_seed,
                     adaptive_fdr,
                     is_age_at_diagnosis,
                     skip_randomizing_intermediate_significant,
                     approximate_null_distribution,
                     n_samples_for_approximation,
                     n_threads if parallelize_within_genes else 1,
                     min_variants_per_gene,
                     pvalue_threshold_for_permutation_tests
                     ) for gn in batch_genes]

    if n_threads > 1 and not parallelize_within_genes:

        with multiprocessing.Pool(processes=n_threads) as pool:

            echo('Starting pool. Batch size:', thread_batch_size, ', n_batches:', len(batch_params), ', n_threads:', n_threads)

            batch_results_fnames = list(pool.map(_process_rv_test_batch_randomized, batch_params))
            echo('Batch results:', batch_results_fnames)

            batch_results = list(map(load_from_tmp_file, batch_results_fnames))

            echo('Closing pool of workers')
            pool.terminate()
            pool.join()
    else:
        echo('Processing batches within the same process. '
             'Batch size:', thread_batch_size,
             ', n_batches:', len(batch_params),
             ', n_threads:', n_threads)

        batch_results_fnames = list(map(process_rv_test_batch_randomized, batch_params))
        echo('Batch results:', batch_results_fnames)
        batch_results = list(map(load_from_tmp_file, batch_results_fnames))

    echo('Concatenating batch results')

    if test_meds:
        results = {}
        med_names = sorted(batch_results[0].keys())

        for med_name in med_names:
            results[med_name] = pd.concat([b[med_name] for b in batch_results], ignore_index=True)

            key_prefix = med_name + '/' + var_label
            sort_by_pvalue_label = 'carrier_X_' + med_name

            results[med_name] = results[med_name].sort_values(key_prefix + '/' + sort_by_pvalue_label + '/pvalue/fdr_corr')

    else:

        results = pd.concat(batch_results, ignore_index=True, sort=True)
        results = results.sort_values(key_prefix + '/' + sort_by_pvalue_label + '/pvalue/fdr_corr')

    for fname in [ph_data_fname] + [b[0] for b in batch_params] + batch_results_fnames:
        echo('Removing temp fname:', fname)
        os.unlink(fname)

    echo('Done')

    return results


def _process_rv_test_batch(batch_params):
    try:

        res = process_rv_test_batch(batch_params)

        log_max_memory_usage()

        return res

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


def _process_rv_test_batch_randomized(batch_params):
    try:

        res = process_rv_test_batch_randomized(batch_params)

        log_max_memory_usage()

        return res

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


def process_rv_test_batch(batch_params):

    (filtered_rv_fname,
     MAX_pAI_THRESHOLDS_TO_TEST,
     batch_size,
     find_best_AC_threshold,
     find_best_pAI_threshold,
     pAI_thresholds_to_test,
     is_binary,
     meds_to_keep,
     multiple_testing_correction,
     pathogenicity_score_label,
     ph_data_fname,
     ph_name,
     phenotype_name_original,
     test_PRS,
     test_meds,
     var_label,
     key_prefix,
     sort_by_pvalue_label,
     output_dir) = batch_params

    filtered_rv = load_from_tmp_file(filtered_rv_fname)

    ph_data = load_from_tmp_file(ph_data_fname)

    sort_by_label = key_prefix + '/%s/pvalue/fdr_corr' % sort_by_pvalue_label
    echo('sort_by_pvalue_label:', sort_by_pvalue_label)
    # normalize phenotype to mean 0 and variance 1
    # ph_data[ph_name] = scipy.stats.zscore(ph_data[ph_name])
    echo('n_genes:', len(filtered_rv.drop_duplicates(GENE_NAME)))
    rv_test_gene.cnt = 0
    if test_meds:
        results = {}

        for gene_name, gene_variants in filtered_rv.groupby(GENE_NAME):
            gene_results = rv_test_gene(gene_variants,
                                        ph_data,
                                        ph_name,
                                        key_prefix,
                                        sort_by_pvalue_label,
                                        test_meds,
                                        test_PRS,
                                        find_best_AC_threshold,
                                        find_best_pAI_threshold,
                                        pAI_thresholds_to_test,
                                        multiple_testing_correction,
                                        MAX_pAI_THRESHOLDS_TO_TEST=MAX_pAI_THRESHOLDS_TO_TEST,
                                        meds_to_keep=meds_to_keep,
                                        pathogenicity_score_label=pathogenicity_score_label
                                        )

            add_new_med_result(gene_name, gene_results, results)

        convert_med_results_to_dataframes(results, key_prefix)

    else:

        results = {GENE_NAME: [],
                   key_prefix + '/carrier/pvalue': [],
                   key_prefix + '/carrier/pvalue/fdr_corr': [],
                   key_prefix + '/sort_by': []}

        if is_binary:
            for k in ['carrier/odds_ratio', 'carrier/n_cases', 'carrier/n_controls', 'all/n_cases', 'all/n_controls']:
                results[key_prefix + '/' + k] = []

        if find_best_pAI_threshold:
            results[key_prefix + '/best_pAI_threshold'] = []

        all_genes = sorted(set(filtered_rv[GENE_NAME]))

        batch_genes = [all_genes[i: i + batch_size] for i in range(0, len(all_genes), batch_size)]

        echo('batches:', len(batch_genes))

        for batch_idx, genes in enumerate(batch_genes):

            echo('Batch:', batch_idx, ', genes:', genes)

            carrier_table, carrier_columns = create_carrier_table(filtered_rv[filtered_rv[GENE_NAME].isin(genes)],
                                                                  ph_data,
                                                                  ph_name,
                                                                  find_best_AC_threshold,
                                                                  find_best_pAI_threshold,
                                                                  pAI_thresholds_to_test,
                                                                  MAX_pAI_THRESHOLDS_TO_TEST=MAX_pAI_THRESHOLDS_TO_TEST,
                                                                  phenotype_name_original=phenotype_name_original,
                                                                  pathogenicity_score_label=pathogenicity_score_label)

            # carrier_columns = [c for c in list(carrier_table) if c not in [SAMPLE_ID, ph_name, phenotype_name_original]]

            if is_binary:
                echo('Computing chi^2 tests for', len(carrier_columns), 'columns')

                n_all_cases = np.sum(carrier_table[ph_name])
                n_all_ctrls = len(carrier_table) - n_all_cases

                pval = {'index': [],
                        ph_name: [],
                        'carrier/odds_ratio': [],
                        'carrier/n_cases': [],
                        'carrier/n_controls': [],
                        'all/n_cases': [],
                        'all/n_controls': []}

                for c in carrier_columns:
                    pval['index'].append(c)
                    n_carriers = np.sum(carrier_table[c])

                    n_carrier_cases = np.sum(carrier_table[c] * carrier_table[ph_name])
                    n_carrier_ctrls = n_carriers - n_carrier_cases

                    pval['carrier/n_cases'].append(n_carrier_cases)
                    pval['carrier/n_controls'].append(n_carrier_ctrls)
                    pval['all/n_cases'].append(n_all_cases)
                    pval['all/n_controls'].append(n_all_ctrls)

                    if n_all_ctrls == 0 or n_all_cases == 0 or n_carriers == 0 or (
                            n_carrier_cases == 0 and n_carrier_ctrls == 0):
                        pval[ph_name].append(1)
                        pval['carrier/odds_ratio'].append(0)

                    else:

                        try:
                            pval[ph_name].append(chi2_or_fisher_test([[n_carrier_cases, n_all_cases - n_carrier_cases],
                                                                      [n_carrier_ctrls,
                                                                       n_all_ctrls - n_carrier_ctrls]]))
                            pval['carrier/odds_ratio'].append(
                                odds_ratio(n_carrier_cases, n_all_cases, n_carrier_ctrls, n_all_ctrls))
                        except:
                            echo('PROBLEM with chi^2 test:',
                                 c,
                                 [[n_carrier_cases, n_all_cases - n_carrier_cases],
                                  [n_carrier_ctrls, n_all_ctrls - n_carrier_ctrls]],
                                 n_carriers,
                                 n_carrier_cases,
                                 n_carrier_ctrls,
                                 n_all_cases,
                                 n_all_ctrls)

                            pval[ph_name].append(1)
                            pval['carrier/odds_ratio'].append(0)

                pval = pd.DataFrame(pval).set_index('index')

            else:
                echo('Computing correlations for', len(carrier_columns), 'columns')

                corr, pval = vcorrcoef(ph_data[ph_name].values[:, None],
                                       carrier_table,
                                       axis='columns')

                pval = pd.DataFrame(pval, columns=[ph_name], index=carrier_columns)

                # corr, pval = vcorrcoef(carrier_table[[ph_name]],
                #                        carrier_table[carrier_columns],
                #                        axis='columns')

            cast_type = {'str': str,
                         'int': int,
                         'float': float}

            echo('Finding best enrichment for each gene')
            for gene in genes:

                gene_key = key_value_to_string(GENE_NAME, gene)

                gene_rows_labels = [c for c in carrier_columns if c.startswith(gene_key)]

                gene_rows = pval.loc[gene_rows_labels].sort_values(ph_name).reset_index().dropna()
                _, fdr_corrected_pvalues = statsmodels.stats.multitest.fdrcorrection(gene_rows[ph_name])

                gene_rows['fdr'] = fdr_corrected_pvalues

                best_pvalue_row = gene_rows.iloc[0]

                row_info = best_pvalue_row['index'].split(';')
                # echo(row_info)

                results[GENE_NAME].append(gene)

                for kv in row_info:

                    key, vt = kv.split('=')

                    if key == GENE_NAME:
                        continue

                    value_type, value = vt.split('|')

                    value = cast_type[value_type](value)

                    res_key = key_prefix + '/' + key
                    if res_key not in results:
                        results[res_key] = []

                    results[res_key].append(value)

                if is_binary:
                    for k in ['carrier/odds_ratio', 'carrier/n_cases', 'carrier/n_controls', 'all/n_cases',
                              'all/n_controls']:
                        results[key_prefix + '/' + k].append(best_pvalue_row[k])

                results[key_prefix + '/carrier/pvalue'].append(best_pvalue_row[ph_name])
                results[key_prefix + '/carrier/pvalue/fdr_corr'].append(best_pvalue_row['fdr'])
                results[key_prefix + '/sort_by'].append(key_prefix + '/%s/pvalue/fdr_corr' % sort_by_pvalue_label)

        results = pd.DataFrame(results).sort_values(sort_by_label)

    return dump_to_tmp_file(results, output_dir=output_dir)


def process_rv_test_batch_randomized(batch_params):

    (filtered_rv_fname,
     MAX_pAI_THRESHOLDS_TO_TEST,
     batch_size,
     find_best_AC_threshold,
     find_best_pAI_threshold,
     pAI_thresholds_to_test,
     is_binary,
     meds_to_keep,
     multiple_testing_correction,
     pathogenicity_score_label,
     ph_data_fname,
     ph_name,
     phenotype_name_original,
     test_PRS,
     test_meds,
     var_label,
     key_prefix,
     sort_by_pvalue_label,
     output_dir,
     n_randomizations,
     random_seed,
     adaptive_fdr,
     is_age_at_diagnosis,
     skip_randomizing_intermediate_significant,
     approximate_null_distribution,
     n_samples_for_approximation,
     n_threads,
     min_variants_per_gene,
     pvalue_threshold_for_permutation_tests
     ) = batch_params

    filtered_rv = load_from_tmp_file(filtered_rv_fname)

    ph_data = load_from_tmp_file(ph_data_fname)

    ph_std = np.std(ph_data[ph_name])
    ph_total = np.sum(ph_data[ph_name])
    ph_total_of_squares = np.sum(np.square(ph_data[ph_name]))

    if phenotype_name_original in list(ph_data):
        original_ph_std = np.std(ph_data[phenotype_name_original])
    else:
        original_ph_std = ph_std

    random.seed(random_seed)
    np.random.seed(random_seed)

    echo('random_seed:', random_seed)

    ph_values = ph_data[[ph_name]].to_numpy()

    echo('making sid_to_phenotypes')
    sid_to_phenotypes = {}
    for sid_idx, sid in enumerate(list(ph_data[SAMPLE_ID])):
        sid_to_phenotypes[sid] = ph_values[sid_idx, :]

    # echo('ph_data.head(10)')
    # echo(ph_data.head(10))
    #
    # echo('ph_data.tail(10)')
    # echo(ph_data.tail(10))

    sort_by_label = key_prefix + '/%s/pvalue/fdr_corr' % sort_by_pvalue_label
    echo('sort_by_pvalue_label:', sort_by_pvalue_label)

    # normalize phenotype to mean 0 and variance 1
    # ph_data[ph_name] = scipy.stats.zscore(ph_data[ph_name])
    echo('n_genes:', len(filtered_rv.drop_duplicates(GENE_NAME)))
    rv_test_gene.cnt = 0
    if test_meds:
        results = {}

        for gene_name, gene_variants in filtered_rv.groupby(GENE_NAME):
            gene_results = rv_test_gene(gene_variants,
                                        ph_data,
                                        ph_name,
                                        key_prefix,
                                        sort_by_pvalue_label,
                                        test_meds,
                                        test_PRS,
                                        find_best_AC_threshold,
                                        find_best_pAI_threshold,
                                        pAI_thresholds_to_test,
                                        multiple_testing_correction,
                                        MAX_pAI_THRESHOLDS_TO_TEST=MAX_pAI_THRESHOLDS_TO_TEST,
                                        meds_to_keep=meds_to_keep,
                                        pathogenicity_score_label=pathogenicity_score_label
                                        )

            add_new_med_result(gene_name, gene_results, results)

        convert_med_results_to_dataframes(results, key_prefix)

    else:

        results = {GENE_NAME: [],
                   key_prefix + '/carrier/pvalue': [],
                   key_prefix + '/carrier/pvalue/fdr_corr': [],
                   key_prefix + '/sort_by': []}

        # if is_binary:
        #     for k in ['carrier/odds_ratio', 'carrier/n_cases', 'carrier/n_controls', 'all/n_cases', 'all/n_controls']:
        #         results[key_prefix + '/' + k] = []

        if find_best_pAI_threshold:
            results[key_prefix + '/best_pAI_threshold'] = []

        all_genes = sorted(set(filtered_rv[GENE_NAME]))

        batch_genes = [all_genes[i: i + batch_size] for i in range(0, len(all_genes), batch_size)]

        echo('batches:', len(batch_genes))
        rnd_state = random.getstate()
        np_rnd_state = np.random.get_state()

        for batch_idx, genes in enumerate(batch_genes):

            random.setstate(rnd_state)
            np.random.set_state(np_rnd_state)

            echo('Batch:', batch_idx, ', genes:', genes)
            genes_variants = filtered_rv[filtered_rv[GENE_NAME].isin(genes)]

            if is_binary:

                rv_test_for_batch_binary_v4(genes_variants,
                                            MAX_pAI_THRESHOLDS_TO_TEST,
                                            find_best_AC_threshold,
                                            find_best_pAI_threshold,
                                            key_prefix,
                                            pAI_thresholds_to_test,
                                            pathogenicity_score_label,

                                            ph_data,
                                            ph_name,

                                            sort_by_pvalue_label,
                                            results,

                                            test_type=LOGIT_ADJUSTED_BY_CHI2,

                                            is_age_at_diagnosis=is_age_at_diagnosis,

                                            MAX_N_RANDOMIZATIONS=100000,
                                            adaptive_fdr=adaptive_fdr,
                                            skip_randomizing_intermediate_significant=skip_randomizing_intermediate_significant,
                                            approximate_null_distribution=approximate_null_distribution,
                                            n_samples_for_approximation=n_samples_for_approximation,
                                            randomization_batch_size=n_randomizations,

                                            return_random_stats=False,
                                            MIN_MORE_SIGNIFICANT=100,
                                            yates_correction=False,
                                            min_variants_per_gene=min_variants_per_gene,
                                            pvalue_threshold_for_permutation_tests=pvalue_threshold_for_permutation_tests)

            else:
                rv_test_for_batch_v4(genes_variants,
                                     MAX_pAI_THRESHOLDS_TO_TEST,
                                     find_best_AC_threshold,
                                     find_best_pAI_threshold,
                                     key_prefix,
                                     pAI_thresholds_to_test,
                                     pathogenicity_score_label,
                                     ph_data,
                                     ph_name,
                                     sort_by_pvalue_label,
                                     results,
                                     ph_total=ph_total,
                                     ph_total_of_squares=ph_total_of_squares,
                                     test_type=T_TEST,
                                     adaptive_fdr=adaptive_fdr,
                                     skip_randomizing_intermediate_significant=skip_randomizing_intermediate_significant,
                                     approximate_null_distribution=approximate_null_distribution,
                                     n_samples_for_approximation=n_samples_for_approximation,
                                     randomization_batch_size=n_randomizations,
                                     min_variants_per_gene=min_variants_per_gene,
                                     pvalue_threshold_for_permutation_tests=pvalue_threshold_for_permutation_tests
                                     )


        results = pd.DataFrame(results).sort_values(sort_by_label)

    return dump_to_tmp_file(results, output_dir=output_dir)


def append_value(d, key, value):
    if key not in d:
        d[key] = []
    d[key].append(value)


def rv_test_for_batch(genes_variants,
                      MAX_pAI_THRESHOLDS_TO_TEST,
                      find_best_AC_threshold,
                      find_best_pAI_threshold,
                      is_binary,
                      key_prefix,
                      pAI_thresholds_to_test,
                      pathogenicity_score_label,
                      ph_data,
                      ph_name,
                      ph_std,
                      original_ph_std,
                      phenotype_name_original,
                      sort_by_pvalue_label,
                      results,
                      n_randomizations):

    echo('[rv_test_for_batch]')
    genes = sorted(set(genes_variants[GENE_NAME]))
    ph_name_columns = [ph_name]
    for s_no in range(1, n_randomizations + 1):
        ph_name_columns += [ph_name + '/random_%d' % s_no]

    carrier_table, carrier_columns = create_carrier_table(genes_variants,
                                                          ph_data,
                                                          ph_name,
                                                          find_best_AC_threshold,
                                                          find_best_pAI_threshold,
                                                          pAI_thresholds_to_test,
                                                          MAX_pAI_THRESHOLDS_TO_TEST=MAX_pAI_THRESHOLDS_TO_TEST,
                                                          phenotype_name_original=phenotype_name_original,
                                                          pathogenicity_score_label=pathogenicity_score_label,
                                                          compute_betas=False)

    carrier_col_2_idx = dict((k, i) for i, k in enumerate(carrier_columns))

    # carrier_columns = [c for c in list(carrier_table) if c not in [SAMPLE_ID, ph_name, phenotype_name_original]]
    if is_binary:
        echo('Computing chi^2 tests for', len(carrier_columns), 'columns')

        n_all_cases = np.sum(carrier_table[ph_name])
        n_all_ctrls = len(carrier_table) - n_all_cases

        pval = {'index': [],
                ph_name: [],
                'carrier/odds_ratio': [],
                'carrier/n_cases': [],
                'carrier/n_controls': [],
                'all/n_cases': [],
                'all/n_controls': []}

        for c in carrier_columns:
            pval['index'].append(c)
            n_carriers = np.sum(carrier_table[c])

            n_carrier_cases = np.sum(carrier_table[c] * carrier_table[ph_name])
            n_carrier_ctrls = n_carriers - n_carrier_cases

            pval['carrier/n_cases'].append(n_carrier_cases)
            pval['carrier/n_controls'].append(n_carrier_ctrls)
            pval['all/n_cases'].append(n_all_cases)
            pval['all/n_controls'].append(n_all_ctrls)

            if n_all_ctrls == 0 or n_all_cases == 0 or n_carriers == 0 or (
                    n_carrier_cases == 0 and n_carrier_ctrls == 0):
                pval[ph_name].append(1)
                pval['carrier/odds_ratio'].append(0)

            else:

                try:
                    pval[ph_name].append(chi2_or_fisher_test([[n_carrier_cases, n_all_cases - n_carrier_cases],
                                                              [n_carrier_ctrls,
                                                               n_all_ctrls - n_carrier_ctrls]]))
                    pval['carrier/odds_ratio'].append(
                        odds_ratio(n_carrier_cases, n_all_cases, n_carrier_ctrls, n_all_ctrls))
                except:
                    echo('PROBLEM with chi^2 test:',
                         c,
                         [[n_carrier_cases, n_all_cases - n_carrier_cases],
                          [n_carrier_ctrls, n_all_ctrls - n_carrier_ctrls]],
                         n_carriers,
                         n_carrier_cases,
                         n_carrier_ctrls,
                         n_all_cases,
                         n_all_ctrls)

                    pval[ph_name].append(1)
                    pval['carrier/odds_ratio'].append(0)

        pval = pd.DataFrame(pval).set_index('index')

    else:
        echo('Computing correlations for', len(carrier_columns), 'columns')

        corr, pval = vcorrcoef(ph_data[ph_name_columns].values,
                               carrier_table,
                               axis='columns')

        pval = pd.DataFrame(pval, columns=ph_name_columns, index=carrier_columns)
        corr = pd.DataFrame(corr, columns=ph_name_columns, index=carrier_columns)

        # corr, pval = vcorrcoef(carrier_table[[ph_name]],
        #                        carrier_table[carrier_columns],
        #                        axis='columns')
    cast_type = {'str': str,
                 'int': int,
                 'float': float}

    echo('Finding best enrichment for each gene')
    for gene in genes:

        gene_key = key_value_to_string(GENE_NAME, gene)

        gene_rows_labels = [c for c in carrier_columns if c.startswith(gene_key)]

        all_gene_rows = pval.loc[gene_rows_labels].dropna().reset_index()

        for ph_col in ph_name_columns:

            results[GENE_NAME].append(gene)
            results[key_prefix + '/sort_by'].append(key_prefix + '/%s/pvalue/fdr_corr' % sort_by_pvalue_label)

            if ph_col == ph_name:
                rand_no = 0
                append_value(results, IS_REAL_DATA, 1)
            else:
                rand_no = int(ph_col.split('_')[-1])
                append_value(results, IS_REAL_DATA, 0)

            append_value(results, SHUFFLED_IDX, rand_no)

            gene_rows = all_gene_rows.sort_values(ph_col)

            _, fdr_corrected_pvalues = statsmodels.stats.multitest.fdrcorrection(gene_rows[ph_col])

            best_row_fdr = fdr_corrected_pvalues[0]

            best_pvalue_row = gene_rows.iloc[0]

            row_info = best_pvalue_row['index'].split(';')
            # echo(row_info)

            for kv in row_info:

                key, vt = kv.split('=')

                if key == GENE_NAME:
                    continue

                value_type, value = vt.split('|')

                value = cast_type[value_type](value)

                res_key = key_prefix + '/' + key
                append_value(results, res_key, value)

            if is_binary:
                for k in ['carrier/odds_ratio',
                          'carrier/n_cases',
                          'carrier/n_controls',
                          'all/n_cases',
                          'all/n_controls']:

                    res_key = key_prefix + '/' + k
                    append_value(results, res_key, best_pvalue_row[k])
            row_idx = best_pvalue_row['index']

            r = corr.loc[row_idx][ph_col]

            var_std = np.std(carrier_table[:, carrier_col_2_idx[row_idx]])

            beta = r * ph_std / var_std
            beta_in_original_units = r * original_ph_std / var_std
            beta_in_standardized_units = r / var_std

            append_value(results, key_prefix + '/carrier/beta', beta)
            append_value(results, key_prefix + '/carrier/beta_in_original_units', beta_in_original_units)
            append_value(results, key_prefix + '/carrier/beta_in_standardized_units', beta_in_standardized_units)

            append_value(results, key_prefix + '/carrier/pvalue', best_pvalue_row[ph_col])
            append_value(results, key_prefix + '/carrier/pvalue/fdr_corr', best_row_fdr)


def key_value_to_string(k, v):
    return k + '=' + {str: 'str',
                      int: 'int',
                      float: 'float'}[type(v)] + '|' + str(v)


def create_carrier_table(filtered_rv,
                         ph_data,
                         ph_name,
                         find_best_AC_threshold,
                         find_best_pAI_threshold,
                         pAI_thresholds_to_test=None,
                         MAX_pAI_THRESHOLDS_TO_TEST=25,
                         phenotype_name_original=None,
                         pathogenicity_score_label=None,
                         compute_betas=True,
                         min_carriers_for_binary_phenotype=1,
                         phenotype_is_binary=False,
                         all_cases=None
                         ):

    ''' create table of carrier status for all samples for several genes

    Args:
        filtered_rv: pandas DataFrame of filtered rare variants (for a subset of
            genes, not all genome-wide, or veen chromosome-wide)
        ph_data: pandas DataFrame of phenotype data per sample
        ph_name: name of column in 'ph_data'
        find_best_AC_threshold: bool indicating if its necessary to test every
             possible allele count
        find_best_pAI_threshold: bool indicating if its necessary to optimise
            over different primateAI thresholds.
        MAX_pAI_THRESHOLDS_TO_TEST:

    Returns:
        numpy array containing carrier status per sample (0=noncarrier, 1=carrier)
        for the genes in filtered_rv and a list of column names for the numpy array
    '''

    carrier_cols = []

    genes = sorted(set(filtered_rv[GENE_NAME]))

    echo('genes=', len(genes))
    all_samples = set(ph_data[SAMPLE_ID])
    sample_id_to_idx = {sid: idx for idx, sid in enumerate(ph_data[SAMPLE_ID].values)}
    ph_values = ph_data[ph_name].values

    if compute_betas:
        if phenotype_name_original is not None:
            ph_original_values = ph_data[phenotype_name_original].values
        else:
            ph_original_values = ph_values

    # count the number of allele count and primate AI thresholds to use per gene.
    # We do this before filtering and testing per gene so we can preallocate a
    # a numpy array of sufficient size. This process over-estimates the number of
    # required columns (some AC/pAI combos give identical variant lists), but we
    # fix that if necessary after checking all required gene combinations

    thresholds_per_gene = {}
    for gene in genes:
        gene_variants = filtered_rv[filtered_rv[GENE_NAME] == gene]
        if find_best_AC_threshold:
            ac_thresholds = sorted(gene_variants[VCF_AC].unique())
        else:
            ac_thresholds = [np.max(gene_variants[VCF_AC])]

        if find_best_pAI_threshold:
            pAI_thresholds = sorted(gene_variants[pathogenicity_score_label].unique())

            if len(pAI_thresholds) > MAX_pAI_THRESHOLDS_TO_TEST:
                pAI_thresholds = [np.quantile(pAI_thresholds,
                                              i / (MAX_pAI_THRESHOLDS_TO_TEST - 1))
                                  for i in range(MAX_pAI_THRESHOLDS_TO_TEST)]
        else:
            pAI_thresholds = [0]

        if pAI_thresholds_to_test is not None:
            pAI_thresholds = pAI_thresholds_to_test

        thresholds_per_gene[gene] = {'ac': ac_thresholds, 'pAI': pAI_thresholds}

    n_thresholds = sum(len(v['ac']) * len(v['pAI']) for v in thresholds_per_gene.values())
    carrier_table = np.zeros((n_thresholds, len(ph_values)))

    i = 0
    for gene in genes:

        gene_variants = filtered_rv[filtered_rv[GENE_NAME] == gene].copy()
        gene_variants = gene_variants.sort_values(pathogenicity_score_label)

        ac_thresholds = thresholds_per_gene[gene]['ac']
        pAI_thresholds = thresholds_per_gene[gene]['pAI']

        gene_name_key = key_value_to_string(GENE_NAME, gene)

        # n_tests = len(ac_thresholds) * len(pAI_thresholds)
        # n_tests_key = key_value_to_string('n_tests', n_tests)
        #
        # echo(gene, ', n_tests_key:', n_tests_key)

        splitted = gene_variants[ALL_SAMPLES].str.split(',')

        var_to_carriers = {k: set(v) & all_samples for k, v in zip(gene_variants[VARID_REF_ALT], splitted)}

        seen = set()
        gene_carrier_cols = []

        for ac_threshold in ac_thresholds:
            c_vars_by_ac = gene_variants[(gene_variants[VCF_AC] <= ac_threshold)]

            for pAI_threshold in pAI_thresholds:
                idx = c_vars_by_ac[pathogenicity_score_label].searchsorted(pAI_threshold, side='left')
                varids = c_vars_by_ac[VARID_REF_ALT][idx:]
                n_vars = len(varids)

                seen_key = tuple(sorted(varids))
                if seen_key in seen:
                    continue
                seen.add(seen_key)

                total_carrier_cases = 0
                total_carrier_controls = 0

                for v in varids:
                    for sid in var_to_carriers[v]:
                        carrier_table[i, sample_id_to_idx[sid]] = 1

                        if phenotype_is_binary:
                            if sid in all_cases:
                                total_carrier_cases += 1
                            else:
                                total_carrier_controls += 1

                if total_carrier_cases < min_carriers_for_binary_phenotype or total_carrier_controls < min_carriers_for_binary_phenotype:
                    continue

                n_carriers = int(np.sum(carrier_table[i]))
                n_non_carriers = len(ph_data) - n_carriers

                col_name_array = [gene_name_key,
                                  key_value_to_string('n_carriers', n_carriers),
                                  key_value_to_string('n_variants', n_vars),
                                  key_value_to_string('best_AC', int(ac_threshold)),
                                  key_value_to_string('best_pAI_threshold', float(pAI_threshold)),
                                  key_value_to_string('pathogenicity_score_label', pathogenicity_score_label)
                                  ]

                if compute_betas:
                    beta = np.sum(carrier_table[i] * ph_values) / n_carriers - np.sum(
                        (1 - carrier_table[i]) * ph_values) / n_non_carriers

                    beta_in_original_units = np.sum(carrier_table[i] * ph_original_values) / n_carriers - np.sum(
                        (1 - carrier_table[i]) * ph_original_values) / n_non_carriers

                    col_name_array += [key_value_to_string('beta', float(beta)),
                                       key_value_to_string('beta_in_original_units', float(beta_in_original_units))]

                # beta = 0
                col_name = ';'.join(col_name_array)
                gene_carrier_cols.append(col_name)
                i += 1

        n_tests = len(seen)
        echo(gene, ', n_tests:', n_tests)

        for gene_col_idx in range(len(gene_carrier_cols)):
            carrier_cols.append(gene_carrier_cols[gene_col_idx] + ';' + key_value_to_string('n_tests', n_tests))

    # echo(f'shapes: carrier_table - {carrier_table.shape}, carrier_cols - {len(carrier_cols)}, index - {i}')
    # sometimes varids are the same between different threshold combinations,
    # these duplicates are skipped, so the table size is a bit smaller
    if i < carrier_table.shape[0]:
        carrier_table = carrier_table[:i, ]

    return carrier_table.T, carrier_cols


def convert_med_results_to_dataframes(results, key_prefix):

    for med_name in results:
        med_results = results[med_name]

        med_results = pd.DataFrame(med_results)

        med_key_prefix = med_name + key_prefix
        sort_by_pvalue_label = 'carrier_X_' + med_name

        med_results = med_results.sort_values(med_key_prefix + '/' + sort_by_pvalue_label + '/pvalue/fdr_corr')

        first_cols = [GENE_NAME] + [c for c in list(med_results) if ('carrier_X_' + med_name) in c or
                                    ('/' + med_name + '/') in c]

        rest_cols = [c for c in list(med_results) if c not in first_cols]

        med_results = med_results[first_cols + rest_cols]

        results[med_name] = med_results


def test_ukb_quantitive_phenotype_for_rare_variants_associations(variants_full,
                                                                 phenotype_data,
                                                                 phenotype_name=None,
                                                                 phenotype_name_original=None,
                                                                 find_best_AC_threshold=True,
                                                                 find_best_pAI_threshold=True,
                                                                 variant_types=None,
                                                                 test_PRS=None,
                                                                 test_meds=False,
                                                                 is_binary=False,
                                                                 n_threads=1,
                                                                 output_dir=None,
                                                                 MAX_pAI_THRESHOLDS_TO_TEST=25,
                                                                 pAI_thresholds_to_test=None,
                                                                 meds_to_keep=None,
                                                                 pathogenicity_score_label=None,
                                                                 n_randomizations=0,
                                                                 random_seed=None,
                                                                 adaptive_fdr=False,
                                                                 is_age_at_diagnosis=False,
                                                                 approximate_null_distribution=False,
                                                                 n_samples_for_approximation=10000,
                                                                 min_variants_per_gene=2,
                                                                 pvalue_threshold_for_permutation_tests=1e-5
                                                                 ):

    MISSENSE_LABEL = 'missense'

    echo('Testing for rare variants associations:', phenotype_name,
         ', find_best_AC_threshold:', find_best_AC_threshold,
         ', find_best_pAI_threshold:', find_best_pAI_threshold,
         ', pAI_thresholds_to_test:', pAI_thresholds_to_test,
         ', test_PRS:', test_PRS,
         ', test_meds:', test_meds,
         ', is_binary:', is_binary,
         ', pathogenicity_score_label:', pathogenicity_score_label,
         ', MAX_pAI_THRESHOLDS_TO_TEST:', MAX_pAI_THRESHOLDS_TO_TEST,
         ', adaptive_fdr:', adaptive_fdr,
         ', approximate_null_distribution:', approximate_null_distribution,
         ', n_samples_for_approximation:', n_samples_for_approximation,
         ', n_randomizations:', n_randomizations,
         ', min_variants_per_gene:', min_variants_per_gene,
         ', random_seed:', random_seed)

    if test_meds:
        result = dict()
    else:
        result = None

    def merge_results(prev_result, new_result):
        if prev_result is None:
            return new_result
        else:
            merge_cols = [GENE_NAME]
            if SHUFFLED_IDX in list(new_result):
                merge_cols += [SHUFFLED_IDX]

            if IS_REAL_DATA in list(new_result):
                merge_cols += [IS_REAL_DATA]

            res = pd.merge(prev_result, new_result, on=merge_cols, how='outer')

            for c in [SHUFFLED_IDX, IS_REAL_DATA]:
                if c in list(res):
                    res[c] = res[c].astype(int)

            return res

    variant_types_to_test = [(DELETERIOUS_VARIANT, 'del'),
                             (ALL_PTV, 'ptv'),
                             (VCF_MISSENSE_VARIANT, MISSENSE_LABEL + '_pAI_optimized'),
                             (VCF_MISSENSE_VARIANT, MISSENSE_LABEL + '_all'),
                             (MISSENSE_AND_PTVS, 'missenses_and_ptvs_all'),
                             (VCF_SYNONYMOUS_VARIANT, 'syn')
                             ]

    if variant_types is not None:
        variant_types_to_test = [(vt, vl) for vt, vl in variant_types_to_test if vl in variant_types or vt in variant_types]

    echo('variant_types_to_test:', variant_types_to_test)

    for vtype, vlabel in variant_types_to_test:

        echo('Testing:', vlabel)

        # remove_missing_pAI = (vlabel in ['del', MISSENSE_LABEL + '_pAI_optimized'])

        filtered_rv = filter_variants_for_rv_test(variants_full,
                                                  consequence=vtype,
                                                  remove_missing_pAI=False,
                                                  pathogenicity_score_label=pathogenicity_score_label)

        filtered_rv = filtered_rv.drop_duplicates(subset=[VARID_REF_ALT, GENE_NAME])

        cur_find_best_pAI_threshold = False

        if vlabel in ['del', MISSENSE_LABEL + '_pAI_optimized'] and find_best_pAI_threshold:
            cur_find_best_pAI_threshold = True

        if vtype == MISSENSE_AND_PTVS and not find_best_pAI_threshold:
            echo('Skipping because missenses and PTVs will be tested with the "del" label:', vlabel)
            continue

        if vlabel == MISSENSE_LABEL + '_all' or vtype == MISSENSE_AND_PTVS:
            if find_best_pAI_threshold:
                cur_find_best_pAI_threshold = False

        if vlabel == MISSENSE_LABEL + '_pAI_optimized' and not find_best_pAI_threshold:
            echo('Skipping because missense variants will be tested with the _all label:', vlabel)
            continue

        n_variants = len(filtered_rv)
        if n_variants == 0:
            echo('No variants found:', vlabel)
            continue

        all_sample_ids = get_samples(filtered_rv)

        echo('n_variants=', n_variants,
             'from n_genes=', len(filtered_rv.drop_duplicates(GENE_NAME)),
             'from n_samples=', len(all_sample_ids),
             'cur_find_best_pAI_threshold=', cur_find_best_pAI_threshold)

        if len(all_sample_ids) == 0:
            echo('Skipping')
            continue

        new_result = rv_test_parallel(phenotype_data,
                                      phenotype_name,
                                      filtered_rv,
                                      vlabel,
                                      test_meds=test_meds,
                                      test_PRS=test_PRS,
                                      multiple_testing_correction='fdr',
                                      find_best_AC_threshold=find_best_AC_threshold,
                                      find_best_pAI_threshold=cur_find_best_pAI_threshold,
                                      pAI_thresholds_to_test=pAI_thresholds_to_test,
                                      n_threads=n_threads,
                                      output_dir=output_dir,
                                      is_binary=is_binary,
                                      MAX_pAI_THRESHOLDS_TO_TEST=MAX_pAI_THRESHOLDS_TO_TEST,
                                      meds_to_keep=meds_to_keep,
                                      phenotype_name_original=phenotype_name_original,
                                      pathogenicity_score_label=pathogenicity_score_label,
                                      n_randomizations=n_randomizations,
                                      random_seed=random_seed,
                                      adaptive_fdr=adaptive_fdr,
                                      is_age_at_diagnosis=is_age_at_diagnosis,
                                      skip_randomizing_intermediate_significant=True,
                                      approximate_null_distribution=approximate_null_distribution,
                                      n_samples_for_approximation=n_samples_for_approximation,
                                      parallelize_within_genes=False,
                                      min_variants_per_gene=min_variants_per_gene,
                                      pvalue_threshold_for_permutation_tests=pvalue_threshold_for_permutation_tests

                                      )

        retest_label = 'ALL/' + vlabel + '/needs_randomization'

        if retest_label in new_result:
            to_retest = set(new_result[new_result[retest_label] == 1][GENE_NAME])

            if len(to_retest) > 0:
                echo('Retesting', len(to_retest), 'genes in parallel:', sorted(to_retest))

                new_result = new_result[~new_result[GENE_NAME].isin(to_retest)].copy()

                retest_result = rv_test_parallel(phenotype_data,
                                                 phenotype_name,
                                                 filtered_rv[filtered_rv[GENE_NAME].isin(to_retest)],
                                                 vlabel,
                                                 test_meds=test_meds,
                                                 test_PRS=test_PRS,
                                                 multiple_testing_correction='fdr',
                                                 find_best_AC_threshold=find_best_AC_threshold,
                                                 find_best_pAI_threshold=cur_find_best_pAI_threshold,
                                                 pAI_thresholds_to_test=pAI_thresholds_to_test,
                                                 n_threads=n_threads,
                                                 output_dir=output_dir,
                                                 is_binary=is_binary,
                                                 MAX_pAI_THRESHOLDS_TO_TEST=MAX_pAI_THRESHOLDS_TO_TEST,
                                                 meds_to_keep=meds_to_keep,
                                                 phenotype_name_original=phenotype_name_original,
                                                 pathogenicity_score_label=pathogenicity_score_label,
                                                 n_randomizations=n_randomizations,
                                                 random_seed=random_seed,
                                                 adaptive_fdr=adaptive_fdr,
                                                 is_age_at_diagnosis=is_age_at_diagnosis,
                                                 skip_randomizing_intermediate_significant=False,
                                                 approximate_null_distribution=approximate_null_distribution,
                                                 n_samples_for_approximation=n_samples_for_approximation,
                                                 parallelize_within_genes=False,
                                                 min_variants_per_gene=min_variants_per_gene,
                                                 pvalue_threshold_for_permutation_tests=pvalue_threshold_for_permutation_tests
                                                 )

                new_result = pd.concat([new_result, retest_result], ignore_index=True)

        if test_meds:
            for med_name in new_result:
                key = new_result[med_name].iloc[0][f'{med_name}/{vlabel}/sort_by']

                highlight_significant_results(new_result[med_name], key)

                result[med_name] = merge_results(result.get(med_name, None),
                                                 new_result[med_name])
        else:
            if new_result is not None and len(new_result) > 0:
                key = new_result.iloc[0][f'ALL/{vlabel}/sort_by']

                highlight_significant_results(new_result, key)

                result = merge_results(result, new_result)

    return result


def highlight_significant_results(result, key):
    SIGNIFICANCE_THRESHOLD = 1e-5
    sig_results = result[result[key] <= SIGNIFICANCE_THRESHOLD]

    def to_dict(r):
        return dict((k, r[k]) for k in r.keys() if not any(k.endswith(s) for s in ['/carriers',
                                                                                   '/all_carriers',
                                                                                   '/cases',
                                                                                   '/controls',
                                                                                   '/variants',
                                                                                   '/all_variants',
                                                                                   '/discarded_variants']))

    for r_no, (_, r) in enumerate(sig_results.iterrows()):
        echo('SIGNIFICANT', r_no + 1, key, r[GENE_NAME], r[key], to_dict(r))

    TOP_K = 20
    for r_no, (_, r) in enumerate(result.sort_values(key).head(TOP_K).iterrows()):
        echo('TOP', r_no + 1, key, r[GENE_NAME], r[key], to_dict(r))


def get_unrelated_samples_with_phenotypes(samples_with_phenotypes=None, KINSHIP_THRESHOLD=0.0884, king_fname=None):
    # KINSHIP_THRESHOLD=0.0884 corresponds to 2nd degree relatives,
    # use KINSHIP_THRESHOLD=0.0442 for 3rd degree relatives

    if king_fname is None:
        king_fname = ROOT_PATH + '/ukbiobank/data/exomes/26_OCT_2020/qc/ukb_merged.king.kin0'

    echo('Loading KING statistics:', king_fname)

    king_relatives = pd.read_csv(
        king_fname,
        sep='\t',
        dtype={'FID1': str, 'ID1': str, 'FID2': str, 'ID2': str, 'N_SNP': int, 'HetHet': float, 'IBS0': float,
               'Kinship': float})

    samples_with_phenotypes = set(samples_with_phenotypes)

    close_relatives = king_relatives[king_relatives['Kinship'] >= KINSHIP_THRESHOLD]
    all_relatives = sorted(set(close_relatives['ID1']) | set(close_relatives['ID2']))

    relatives_to_exclude = set()

    echo('all_relatives:', len(all_relatives))

    for _, r in close_relatives.iterrows():
        pair = (set([r['ID1'], r['ID2']]) & samples_with_phenotypes) - relatives_to_exclude

        if len(pair) == 0 or len(pair) == 1:
            continue
        else:
            to_exclude = random.choice(list(pair))
            relatives_to_exclude.add(to_exclude)

    echo('relatives_to_exclude:', len(relatives_to_exclude))

    unrelated_samples = sorted(samples_with_phenotypes - relatives_to_exclude)

    return unrelated_samples


def precompute_variant_stats_for_rv_test(gene_name,
                                         gene_variants,
                                         sid_to_phenotypes,
                                         n_phenotypes,
                                         all_samples,
                                         MAX_pAI_THRESHOLDS_TO_TEST,
                                         find_best_AC_threshold,
                                         find_best_pAI_threshold,

                                         pAI_thresholds_to_test,
                                         pathogenicity_score_label,
                                         test_type=RANKSUM_TEST
                                         ):

    variants_info = []
    gene_variants = gene_variants.sort_values(VCF_AC)

    if find_best_pAI_threshold:
        if pAI_thresholds_to_test is None:
            pAI_thresholds = sorted(gene_variants[pathogenicity_score_label].unique())

            if len(pAI_thresholds) > MAX_pAI_THRESHOLDS_TO_TEST:
                pAI_thresholds = [np.quantile(pAI_thresholds,
                                              i / (MAX_pAI_THRESHOLDS_TO_TEST - 1))
                                  for i in range(MAX_pAI_THRESHOLDS_TO_TEST)]
        else:
            pAI_thresholds = pAI_thresholds_to_test

    else:
        pAI_thresholds = [0]


    splitted = gene_variants[ALL_SAMPLES].str.split(',')

    var_to_carriers = {k: set(v) & all_samples for k, v in zip(gene_variants[VARID_REF_ALT], splitted)}
    var_to_pAI = {k: v for k, v in zip(gene_variants[VARID_REF_ALT], gene_variants[pathogenicity_score_label])}

    carrier_to_pAI = {}
    for v in var_to_carriers:
        for sid in var_to_carriers[v]:
            if sid not in carrier_to_pAI:
                carrier_to_pAI[sid] = var_to_pAI[v]
            else:
                carrier_to_pAI[sid] = max(carrier_to_pAI[sid], var_to_pAI[v])

    all_carriers = sorted(carrier_to_pAI)

    v1 = [sid_to_phenotypes[sid][0] for sid in all_carriers]
    v2 = [carrier_to_pAI[sid] for sid in all_carriers]

    if len(v1) < 2:
        pAI_corr_carrier_level = [0, 1]
        pAI_carrier_level_beta = 0
    else:
        pAI_corr_carrier_level = scipy.stats.spearmanr(v1, v2)
        corr = scipy.stats.pearsonr(v1, v2)[0]
        pAI_carrier_level_beta = corr * np.std(v2) / np.std(v1)

    all_variants = sorted(var_to_carriers)

    v1 = [np.median([sid_to_phenotypes[sid][0] for sid in var_to_carriers[v]]) for v in all_variants]
    v2 = [var_to_pAI[v] for v in all_variants]

    if len(v1) < 2:
        pAI_corr_variant_level = [0, 1]
        pAI_variant_level_beta = 0
    else:
        pAI_corr_variant_level = scipy.stats.spearmanr(v1, v2)
        corr = scipy.stats.pearsonr(v1, v2)[0]
        pAI_variant_level_beta = corr * np.std(v2) / np.std(v1)

    max_ac = np.max(gene_variants[VCF_AC])

    def get_new_combo_info(combo_totals, totals_of_squares, carriers, variants, ac_threshold, pAI_threshold,
                           pathogenicity_score_label):
        return {'totals': combo_totals,
                'totals_of_squares': totals_of_squares,
                'n_carriers': len(carriers),
                'carriers': carriers,
                'variants': variants,
                'info': {'n_variants': len(variants),
                         'best_AC': int(ac_threshold),
                         'best_AF': int(ac_threshold) / (2 * len(all_samples)),
                         'best_pAI_threshold': float(pAI_threshold),
                         'pathogenicity_score_label': pathogenicity_score_label}}

    for pAI_threshold_idx, pAI_threshold in enumerate(sorted(pAI_thresholds, reverse=True)):
        c_vars_by_pAI = gene_variants[gene_variants[pathogenicity_score_label] >= pAI_threshold]

        if len(c_vars_by_pAI) == 0:
            continue

        if find_best_AC_threshold:
            ac_thresholds = sorted(c_vars_by_pAI[VCF_AC].unique())
        else:
            ac_thresholds = [max_ac]

        c_idx = 0

        ac_t_idx = 0
        c_ac = ac_thresholds[ac_t_idx]

        combo_info = get_new_combo_info(np.zeros(n_phenotypes),
                                        np.zeros(n_phenotypes),
                                        set(),
                                        set(),
                                        c_ac,
                                        pAI_threshold,
                                        pathogenicity_score_label)
        variants_info.append(combo_info)

        while c_idx < len(c_vars_by_pAI):
            var_id = c_vars_by_pAI.iloc[c_idx][VARID_REF_ALT]
            new_ac = c_vars_by_pAI.iloc[c_idx][VCF_AC]

            if new_ac <= c_ac:
                samples_to_add = var_to_carriers[var_id] - combo_info['carriers']
                combo_info['variants'].add(var_id)

                for sid in samples_to_add:
                    combo_info['totals'] = combo_info['totals'] + sid_to_phenotypes[sid]
                    if test_type == T_TEST:
                        combo_info['totals_of_squares'] = combo_info['totals_of_squares'] + np.square(
                            sid_to_phenotypes[sid])

                combo_info['carriers'] = combo_info['carriers'] | samples_to_add
                combo_info['n_carriers'] += len(samples_to_add)
                combo_info['info']['n_variants'] += 1
                c_idx += 1
            else:

                ac_t_idx += 1
                c_ac = ac_thresholds[ac_t_idx]

                combo_info = get_new_combo_info(np.array(combo_info['totals']),
                                                np.array(combo_info['totals_of_squares']),
                                                set(combo_info['carriers']),
                                                set(combo_info['variants']),
                                                c_ac,
                                                pAI_threshold,
                                                pathogenicity_score_label)

                variants_info.append(combo_info)

    # make variant_info unique based on the set of variants
    # and keep only combos with more than one carrier

    d = dict((tuple(sorted(v['variants'])), v) for v in variants_info if v['n_carriers'] > 1)
    # echo('variants_info:', len(variants_info), len(d))
    variants_info = [d[k] for k in d]

    n_tests = len(variants_info)

    echo(gene_name, ', n_tests:', n_tests)
    for cur_combo_info in variants_info:
        cur_combo_info['n_tests'] = n_tests

        cur_combo_info['info']['pathogenicity_score/carrier_level/spearman_r'] = pAI_corr_carrier_level[0]
        cur_combo_info['info']['pathogenicity_score/carrier_level/spearman_pvalue'] = pAI_corr_carrier_level[1]
        cur_combo_info['info']['pathogenicity_score/carrier_level/regression_beta'] = pAI_carrier_level_beta

        cur_combo_info['info']['pathogenicity_score/variant_level/spearman_r'] = pAI_corr_variant_level[0]
        cur_combo_info['info']['pathogenicity_score/variant_level/spearman_pvalue'] = pAI_corr_variant_level[1]
        cur_combo_info['info']['pathogenicity_score/variant_level/regression_beta'] = pAI_variant_level_beta

        cur_combo_info['info']['all_carriers'] = all_carriers

        discarded_variants = sorted(set(all_variants) - set(cur_combo_info['variants']))
        cur_combo_info['info']['discarded_variants'] = ','.join(discarded_variants)
        cur_combo_info['info']['n_discarded_variants'] = len(discarded_variants)

        cur_combo_info['info']['n_total_carriers'] = len(all_carriers)
        cur_combo_info['info']['n_total_variants'] = len(all_variants)

    return variants_info


cast_type = {'str': str,
             'int': int,
             'float': float}


def rv_test_for_batch_binary(genes_variants,
                             MAX_pAI_THRESHOLDS_TO_TEST,
                             find_best_AC_threshold,
                             find_best_pAI_threshold,
                             key_prefix,
                             pAI_thresholds_to_test,
                             pathogenicity_score_label,
                             ph_data,
                             ph_name,
                             sort_by_pvalue_label,
                             results,
                             test_type=LOGIT,
                             const=None,
                             covariates=None,
                             debug=False,
                             is_age_at_diagnosis=False,
                             n_threads=1
                             ):

    echo('rv_test_for_batch_binary, is_age_at_diagnosis=', is_age_at_diagnosis)

    if const is None:
        const = np.ones(len(ph_data))

    if covariates is None:
        covariates = const[..., None]
    else:
        covariates = covariates.to_numpy()

    gene_names = sorted(genes_variants[GENE_NAME].unique())

    all_cases = set(ph_data[ph_data[ph_name] == 1][SAMPLE_ID])
    all_controls = set(ph_data[ph_data[ph_name] == 0][SAMPLE_ID])

    sid_to_aod = None
    sid_to_age = None

    if is_age_at_diagnosis:
        aod_label = ph_name + '/original_age_at_diagnosis'
        aod_samples = ph_data[ph_data[aod_label] >= 0]

        sid_to_aod = dict((sid, aod) for sid, aod in zip(aod_samples[SAMPLE_ID],
                                                         aod_samples[aod_label]))

        sid_to_age = dict((sid, age) for sid, age in zip(ph_data[SAMPLE_ID],
                                                         ph_data[AGE]))


    for gene_name in gene_names:

        cur_gene_variants = genes_variants[genes_variants[GENE_NAME] == gene_name]
        carrier_table, carrier_columns = create_carrier_table(cur_gene_variants,
                                                              ph_data,
                                                              ph_name,
                                                              find_best_AC_threshold,
                                                              find_best_pAI_threshold,
                                                              pAI_thresholds_to_test=pAI_thresholds_to_test,
                                                              MAX_pAI_THRESHOLDS_TO_TEST=MAX_pAI_THRESHOLDS_TO_TEST,
                                                              phenotype_name_original=None,
                                                              pathogenicity_score_label=pathogenicity_score_label,
                                                              compute_betas=False,
                                                              min_carriers_for_binary_phenotype=1,
                                                              phenotype_is_binary=True,
                                                              all_cases=all_cases
                                                              )

        gene_res = {'col_name': [],
                    'pvalue': [],
                    'beta': [],
                    'tvalue': []}

        echo('carrier_columns:', len(carrier_columns))
        if len(carrier_columns) == 0:
            echo('No eligible variant combinations for:', gene_name)
            continue

        best_model = None
        best_pvalue = 1
        for col_idx, col_name in enumerate(carrier_columns):
            lm = sm.Logit(ph_data[ph_name], np.concatenate([carrier_table[:, col_idx][..., None], covariates], axis=1))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                try:
                    lm_res = lm.fit(disp=0)
                    pvalue = lm_res.pvalues[0]
                    tvalue = lm_res.tvalues[0]
                    beta = lm_res.params[0]

                    if debug and best_pvalue >= pvalue:
                        best_model = lm_res

                except np.linalg.LinAlgError:
                    pvalue = 1
                    tvalue = 0
                    beta = 0

                # lm_res = lm.fit(disp=0, skip_hessian=True, maxiter=10)

            gene_res['col_name'].append(col_name)
            gene_res['pvalue'].append(pvalue)
            gene_res['tvalue'].append(tvalue)
            gene_res['beta'].append(beta)

        if debug:
            echo('Best model')
            display(best_model.summary())

        gene_res = pd.DataFrame(gene_res).sort_values('pvalue')

        best_combo = gene_res.iloc[0]

        ph_pvalues = gene_res['pvalue']

        _, fdr_corrected_pvalues = statsmodels.stats.multitest.fdrcorrection(ph_pvalues)

        best_combo_fdr = fdr_corrected_pvalues[0]
        best_combo_pvalue = best_combo['pvalue']

        beta = best_combo['beta']
        stat = best_combo['tvalue']

        row_info = best_combo['col_name'].split(';')

        best_AC = None
        best_pAI_score = None

        for kv in row_info:

            key, vt = kv.split('=')

            if key == GENE_NAME:
                continue

            value_type, value = vt.split('|')

            value = cast_type[value_type](value)
            if key == 'best_pAI_threshold':
                best_pAI_score = value
            elif key == 'best_AC':
                best_AC = value

            res_key = key_prefix + '/' + key
            append_value(results, res_key, value)

        append_value(results, IS_REAL_DATA, 1)
        append_value(results, SHUFFLED_IDX, 0)

        append_value(results, GENE_NAME, gene_name)
        append_value(results, key_prefix + '/sort_by', key_prefix + '/%s/pvalue/fdr_corr' % sort_by_pvalue_label)
        # append_value(results, key_prefix + '/n_carriers', best_combo['n_carriers'])

        best_combo_variants = cur_gene_variants[(cur_gene_variants[VCF_AC] <= best_AC) & (cur_gene_variants[pathogenicity_score_label] >= best_pAI_score)]

        all_carriers = set([s for r in best_combo_variants[ALL_SAMPLES] for s in r.split(',')])
        append_value(results, key_prefix + '/carriers', ','.join(sorted(all_carriers)))
        append_value(results, key_prefix + '/variants', ','.join(sorted(best_combo_variants[VARID_REF_ALT])))

        carrier_cases = all_carriers & all_cases
        carrier_controls = all_carriers & all_controls

        append_value(results, key_prefix + '/carriers/cases', ','.join(sorted(carrier_cases)))
        append_value(results, key_prefix + '/carriers/controls', ','.join(sorted(carrier_controls)))
        append_value(results, key_prefix + '/carriers/n_cases', len(carrier_cases))
        append_value(results, key_prefix + '/carriers/n_controls', len(carrier_controls))

        append_value(results, key_prefix + '/carrier/beta', beta)
        append_value(results, key_prefix + '/odds_ratio', math.exp(beta))
        append_value(results, key_prefix + '/carrier/stat', stat)
        append_value(results, key_prefix + '/test_type', test_type)

        append_value(results, key_prefix + '/carrier/pvalue', best_combo_pvalue)
        append_value(results, key_prefix + '/carrier/pvalue/fdr_corr', best_combo_fdr)

        all_gene_variants = set(cur_gene_variants[VARID_REF_ALT])
        discarded_variants = sorted(all_gene_variants - set(best_combo_variants[VARID_REF_ALT]))
        append_value(results, key_prefix + '/discarded_variants', ','.join(discarded_variants))
        append_value(results, key_prefix + '/n_discarded_variants', len(discarded_variants))
        append_value(results, key_prefix + '/n_all_variants', len(all_gene_variants))

        all_gene_carriers = sorted(set([sid for r in cur_gene_variants[ALL_SAMPLES] for sid in r.split(',')]))

        append_value(results, key_prefix + '/all_carriers', ','.join(sorted(all_gene_carriers)))
        append_value(results, key_prefix + '/n_all_carriers', len(all_gene_carriers))

        carrier_to_pAI = {}
        frac_cases_per_variant = {}
        avg_aod_per_variant = {}
        avg_control_age_per_variant = {}

        var_to_pAI = {}
        for _, v in cur_gene_variants.iterrows():
            v_carriers = v[ALL_SAMPLES].split(',')

            frac_cases_per_variant[v[VARID_REF_ALT]] = len(set(v_carriers) & all_cases) / len(v_carriers)
            var_to_pAI[v[VARID_REF_ALT]] = v[pathogenicity_score_label]

            if is_age_at_diagnosis:
                aod_for_cases = [sid_to_aod[sid] for sid in v_carriers if sid in sid_to_aod]
                if len(aod_for_cases) > 0:
                    avg_aod_per_variant[v[VARID_REF_ALT]] = np.median(aod_for_cases)

                age_for_controls = [sid_to_age[sid] for sid in v_carriers if sid in carrier_controls]
                if len(age_for_controls) > 0:
                    avg_control_age_per_variant[v[VARID_REF_ALT]] = np.median(age_for_controls)

            for sid in v_carriers:
                if sid not in carrier_to_pAI:
                    carrier_to_pAI[sid] = v[pathogenicity_score_label]
                else:
                    carrier_to_pAI[sid] = max(carrier_to_pAI[sid], v[pathogenicity_score_label])

        all_cases_among_all_carriers = sorted(all_cases & set(carrier_to_pAI.keys()))
        all_controls_among_all_carriers = sorted(all_controls & set(carrier_to_pAI.keys()))

        append_value(results, key_prefix + '/all_carriers/cases', ','.join(all_cases_among_all_carriers))
        append_value(results, key_prefix + '/all_carriers/n_cases', len(all_cases_among_all_carriers))

        append_value(results, key_prefix + '/all_carriers/controls', ','.join(all_controls_among_all_carriers))
        append_value(results, key_prefix + '/all_carriers/n_controls', len(all_controls_among_all_carriers))

        cases_pai = [carrier_to_pAI[sid] for sid in all_cases_among_all_carriers]
        ctrls_pai = [carrier_to_pAI[sid] for sid in all_controls_among_all_carriers]

        rs_stat, rs_pval = scipy.stats.ranksums(cases_pai, ctrls_pai)
        cases_pai_mean = np.mean(cases_pai)
        ctrls_pai_mean = np.mean(ctrls_pai)

        append_value(results, key_prefix + '/carrier/pathogenicity_score/ranksums_stat', rs_stat)
        append_value(results, key_prefix + '/carrier/pathogenicity_score/ranksums_pvalue', rs_pval)
        append_value(results, key_prefix + '/carrier/pathogenicity_score/mean_diff', cases_pai_mean - ctrls_pai_mean)
        append_value(results, key_prefix + '/carrier/pathogenicity_score/cases_mean', cases_pai_mean)
        append_value(results, key_prefix + '/carrier/pathogenicity_score/controls_mean', ctrls_pai_mean)

        sc_stat, sc_pval = scipy.stats.spearmanr([frac_cases_per_variant[vid] for vid in sorted(frac_cases_per_variant)],
                                                 [var_to_pAI[vid] for vid in sorted(frac_cases_per_variant)])

        append_value(results, key_prefix + '/carrier/pathogenicity_score/fraction_cases_vs_score/spearmanr_stat', sc_stat)
        append_value(results, key_prefix + '/carrier/pathogenicity_score/fraction_cases_vs_score/spearmanr_pvalue', sc_pval)

        if is_age_at_diagnosis:

            # compute correlation between pathogenicity scores and age at diagnosis for cases (this should be negative for true positive genes)

            aod_per_carrier = [sid_to_aod[sid] for sid in all_cases_among_all_carriers]
            aod_carrier_level_corr, aod_carrier_level_pvalue = scipy.stats.spearmanr(aod_per_carrier, cases_pai)

            if len(aod_per_carrier) > 1:
                corr = scipy.stats.pearsonr(aod_per_carrier, cases_pai)[0]
            else:
                corr = 0

            pAI_aod_carrier_level_beta = corr * np.std(cases_pai) / np.std(aod_per_carrier)

            append_value(results, key_prefix + '/pathogenicity_score/age_at_diagnosis/carrier_level/spearman_r', aod_carrier_level_corr)
            append_value(results, key_prefix + '/pathogenicity_score/age_at_diagnosis/carrier_level/spearman_pvalue', aod_carrier_level_pvalue)
            append_value(results, key_prefix + '/pathogenicity_score/age_at_diagnosis/carrier_level/regression_beta', pAI_aod_carrier_level_beta)
            append_value(results, key_prefix + '/pathogenicity_score/age_at_diagnosis/carrier_level/n_carriers', len(aod_per_carrier))

            v1 = [avg_aod_per_variant[vid] for vid in sorted(avg_aod_per_variant)]
            v2 = [var_to_pAI[vid] for vid in sorted(avg_aod_per_variant)]
            aod_variant_level_corr, aod_variant_level_pvalue = scipy.stats.spearmanr(v1, v2)

            if len(v1) > 1:
                corr = scipy.stats.pearsonr(v1, v2)[0]
            else:
                corr = 0

            pAI_aod_variant_level_beta = corr * np.std(v2) / np.std(v1)

            append_value(results, key_prefix + '/pathogenicity_score/age_at_diagnosis/variant_level/spearman_r', aod_variant_level_corr)
            append_value(results, key_prefix + '/pathogenicity_score/age_at_diagnosis/variant_level/spearman_pvalue', aod_variant_level_pvalue)
            append_value(results, key_prefix + '/pathogenicity_score/age_at_diagnosis/variant_level/regression_beta', pAI_aod_variant_level_beta)
            append_value(results, key_prefix + '/pathogenicity_score/age_at_diagnosis/variant_level/n_variants', len(avg_aod_per_variant))

            # compute correlation between pathogenicity scores and age for controls (this should be positive for true positive genes)

            age_per_carrier_control = [sid_to_age[sid] for sid in all_controls_among_all_carriers]
            control_age_carrier_level_corr, control_age_carrier_level_pvalue = scipy.stats.spearmanr(age_per_carrier_control, ctrls_pai)

            append_value(results, key_prefix + '/pathogenicity_score/controls_age/carrier_level/spearman_r', control_age_carrier_level_corr)
            append_value(results, key_prefix + '/pathogenicity_score/controls_age/carrier_level/spearman_pvalue', control_age_carrier_level_pvalue)
            append_value(results, key_prefix + '/pathogenicity_score/controls_age/carrier_level/n_carriers', len(age_per_carrier_control))

            control_age_variant_level_corr, control_age_variant_level_pvalue = scipy.stats.spearmanr(
                [avg_control_age_per_variant[vid] for vid in sorted(avg_control_age_per_variant)],
                [var_to_pAI[vid] for vid in sorted(avg_control_age_per_variant)])

            append_value(results, key_prefix + '/pathogenicity_score/controls_age/variant_level/spearman_r', control_age_variant_level_corr)
            append_value(results, key_prefix + '/pathogenicity_score/controls_age/variant_level/spearman_pvalue', control_age_variant_level_pvalue)
            append_value(results, key_prefix + '/pathogenicity_score/controls_age/variant_level/n_variants', len(avg_control_age_per_variant))


import ctypes
cutils = ctypes.CDLL(os.path.join(os.path.split(__file__)[0],
                                  'utils.so'))

def rv_test_for_batch_v4(genes_variants,
                         MAX_pAI_THRESHOLDS_TO_TEST,
                         find_best_AC_threshold,
                         find_best_pAI_threshold,
                         key_prefix,
                         pAI_thresholds_to_test,
                         pathogenicity_score_label,

                         ph_data,
                         ph_name,

                         sort_by_pvalue_label,
                         results,
                         ph_total=None,
                         ph_total_of_squares=None,
                         test_type=T_TEST,
                         MAX_N_RANDOMIZATIONS=100000,
                         randomization_batch_size=1000,
                         adaptive_fdr=True,
                         skip_randomizing_intermediate_significant=False,
                         approximate_null_distribution=False,
                         n_samples_for_approximation=10000,
                         return_random_stats=False,
                         min_variants_per_gene=2,
                         pvalue_threshold_for_permutation_tests=1e-5):

    echo('[rv_test_for_batch v4], adaptive_fdr:', adaptive_fdr,
         ', MAX_N_RANDOMIZATIONS', MAX_N_RANDOMIZATIONS,
         ', test_type:', test_type,
         ', approximate_null_distribution:', approximate_null_distribution,
         ', n_samples_for_approximation:', n_samples_for_approximation,
         ', return_random_stats:', return_random_stats,
         ', min_variants_per_gene:', min_variants_per_gene,
         ', pvalue_threshold_for_permutation_tests:', pvalue_threshold_for_permutation_tests
         )

    rnd_state = random.getstate()
    np_rnd_state = np.random.get_state()

    gene_names = sorted(genes_variants[GENE_NAME].unique())
    all_samples_set = set(ph_data[SAMPLE_ID])
    n_all_samples = len(all_samples_set)

    ph_values_array = list(ph_data[ph_name])
    sid_to_idx = dict((sid, idx) for idx, sid in enumerate(ph_data[SAMPLE_ID]))

    for gene_name in gene_names:

        random.setstate(rnd_state)
        np.random.set_state(np_rnd_state)

        cur_gene_variants = genes_variants[genes_variants[GENE_NAME] == gene_name]
        all_carriers = sorted(set(sid for _, r in cur_gene_variants.iterrows() for sid in r[ALL_SAMPLES].split(','))
                              & all_samples_set)

        n_all_carriers = len(all_carriers)
        keep_testing = True

        c_more_significant = 0
        n_randomizations = 0
        gene_iteration_idx = 0
        random_stats = []

        echo(gene_name, ', n_all_carriers:', n_all_carriers, ', n_all_variants:', len(cur_gene_variants))

        while keep_testing:

            C_INT = ctypes.c_int

            result = (C_INT * (n_all_carriers * (randomization_batch_size)))()

            cutils.shuffle_array(n_all_samples, n_all_carriers, randomization_batch_size, result)

            result = np.array(result).reshape((randomization_batch_size, n_all_carriers))
            rand_idx = dict((sid_no,
                             np.concatenate([[sid_to_idx[sid]],
                                             result[:, sid_no]
                                             ])) for sid_no, sid in enumerate(all_carriers))

            sid_to_phenotypes = {}
            for sid_no, sid in enumerate(all_carriers):
                sid_to_phenotypes[sid] = np.array(
                    [ph_values_array[c_rand_idx] for c_rand_idx in rand_idx[sid_no]])

            # echo('[precompute_variant_stats_for_rv_test]')
            echo('Randomizing done')
            gene_iteration_idx += 1
            variants_info_for_rv = precompute_variant_stats_for_rv_test(gene_name,
                                                                        cur_gene_variants,
                                                                        sid_to_phenotypes,
                                                                        randomization_batch_size + 1,
                                                                        all_samples_set,
                                                                        MAX_pAI_THRESHOLDS_TO_TEST,
                                                                        find_best_AC_threshold,
                                                                        find_best_pAI_threshold,

                                                                        pAI_thresholds_to_test,
                                                                        pathogenicity_score_label,
                                                                        test_type=test_type
                                                                        )

            variants_info_for_rv = [cur_var_info for cur_var_info in variants_info_for_rv
                                    if cur_var_info['info']['n_variants'] >= min_variants_per_gene]

            if len(variants_info_for_rv) == 0:
                break

            gene_results = []

            # echo('performing tests')
            for cur_var_info in variants_info_for_rv:
                cur_var_info['n_tests'] = len(variants_info_for_rv)

                carrier_totals = cur_var_info['totals']
                n_carriers = cur_var_info['n_carriers']

                n_non_carriers = len(all_samples_set) - n_carriers

                if test_type == RANKSUM_TEST:
                    expected = n_carriers * (n_carriers + n_non_carriers + 1) / 2.0

                    std_u = math.sqrt((n_carriers * n_non_carriers * (n_carriers + n_non_carriers + 1) / 12.0))
                    z_scores = (carrier_totals - expected) / std_u

                    p_values = 2 * scipy.stats.norm.sf(np.abs(z_scores))

                    betas = np.zeros(randomization_batch_size + 1)
                    stats = z_scores

                elif test_type == T_TEST:

                    carrier_totals_of_squares = cur_var_info['totals_of_squares']
                    carrier_means = carrier_totals / n_carriers

                    # carrier_var = (n_carriers * carrier_totals_of_squares - np.square(carrier_totals)) / (n_carriers ** 2)
                    carrier_var = carrier_totals_of_squares / (n_carriers - 1) - np.square(carrier_totals) / (n_carriers * (n_carriers - 1))

                    non_carrier_totals = ph_total - carrier_totals

                    non_carrier_means = non_carrier_totals / n_non_carriers
                    non_carrier_total_of_squares = ph_total_of_squares - carrier_totals_of_squares

                    # non_carrier_var = (n_non_carriers * non_carrier_total_of_squares - np.square(non_carrier_totals)) / (n_non_carriers ** 2)
                    non_carrier_var = non_carrier_total_of_squares / (n_non_carriers - 1) - np.square(non_carrier_totals) / (n_non_carriers * (n_non_carriers - 1))

                    df = len(all_samples_set) - 2

                    denom = math.sqrt(1 / n_carriers + 1 / n_non_carriers)
                    denom *= np.sqrt(((n_carriers - 1) * carrier_var +
                                      (n_non_carriers - 1) * non_carrier_var) / df)

                    t_stat = (carrier_means - non_carrier_means) / denom

                    p_values = 2 * scipy.stats.t.sf(np.abs(t_stat), df)

                    # if np.any(np.isnan(p_values)):
                    #     echo(f'[WARNING] {gene_name}: NaN p-values from the T-test, replacing NaNs with 1:', np.sum(np.isnan(p_values)), np.argwhere(np.isnan(p_values)))
                    #     p_values = np.nan_to_num(p_values, nan=1)

                    betas = carrier_means - non_carrier_means
                    stats = t_stat

                else:
                    echo('[ERROR] Unknown test type:', test_type)
                    return None

                gene_results.append(p_values)
                cur_var_info['betas'] = betas
                cur_var_info['stats'] = stats

            gene_results = np.array(gene_results)
            best_combo_indexes = np.argmin(gene_results, axis=0)

            COLUMNS_FOR_RANDOMIZED_DATA = ['carrier/beta', 'carrier/stat', 'carrier/pvalue', 'carrier/pvalue/fdr_corr', 'best_pAI_threshold', 'best_AC', 'n_variants', 'n_carriers']
            random_cols = dict((c, []) for c in COLUMNS_FOR_RANDOMIZED_DATA)
            best_real_pvalue = None
            best_real_bh_fdr = None
            best_real_stat = None

            # echo('Finding best combo for each permutation')
            for ph_idx in range(randomization_batch_size + 1):
                best_combo_idx = best_combo_indexes[ph_idx]
                best_combo = variants_info_for_rv[best_combo_idx]

                ph_pvalues = gene_results[:, ph_idx]

                _, fdr_corrected_pvalues = statsmodels.stats.multitest.fdrcorrection(ph_pvalues)

                best_combo_fdr = fdr_corrected_pvalues[best_combo_idx]
                best_combo_pvalue = ph_pvalues[best_combo_idx]

                beta = best_combo['betas'][ph_idx]
                stat = best_combo['stats'][ph_idx]

                row_info = best_combo['info']
                if ph_idx == 0:
                    best_real_stat = stat
                    best_real_pvalue = best_combo_pvalue
                    best_real_bh_fdr = best_combo_fdr

                if ph_idx > 0:
                    for c in COLUMNS_FOR_RANDOMIZED_DATA:
                        if c == 'carrier/beta':
                            v = beta
                        elif c == 'carrier/stat':
                            v = stat
                        elif c == 'carrier/pvalue':
                            v = best_combo_pvalue
                        elif c == 'carrier/pvalue/fdr_corr':
                            v = best_combo_fdr
                        elif c == 'n_carriers':
                            v = best_combo['n_carriers']
                        else:
                            v = row_info[c]

                        random_cols[c].append(v)

                    continue

                if gene_iteration_idx == 1:
                    append_value(results, IS_REAL_DATA, 1)

                    append_value(results, SHUFFLED_IDX, ph_idx)

                    append_value(results, GENE_NAME, gene_name)
                    append_value(results, key_prefix + '/sort_by', key_prefix + '/%s/pvalue/fdr_corr' % sort_by_pvalue_label)
                    append_value(results, key_prefix + '/n_carriers', best_combo['n_carriers'])

                    append_value(results, key_prefix + '/carriers', ','.join(sorted(best_combo['carriers']) if ph_idx == 0 else []))
                    append_value(results, key_prefix + '/variants', ','.join(sorted(best_combo['variants']) if ph_idx == 0 else []))

                    for key in row_info:
                        value = row_info[key]

                        res_key = key_prefix + '/' + key
                        append_value(results, res_key, value)

                    append_value(results, key_prefix + '/carrier/n_tests', best_combo['n_tests'])
                    append_value(results, key_prefix + '/carrier/beta', beta)
                    append_value(results, key_prefix + '/carrier/stat', stat)
                    append_value(results, key_prefix + '/test_type', test_type)

                    append_value(results, key_prefix + '/carrier/pvalue', best_combo_pvalue)

                    if return_random_stats:
                        for c in random_cols:
                            rand_col_name = key_prefix + '/' + c + '/random'
                            append_value(results, rand_col_name, [])

                    if adaptive_fdr:
                        append_value(results, key_prefix + '/carrier/pvalue/BH_fdr_corr', best_combo_fdr)

                    if skip_randomizing_intermediate_significant:
                        if pvalue_threshold_for_permutation_tests < best_combo_fdr <= 1e-2 and adaptive_fdr:
                            append_value(results, key_prefix + '/carrier/pvalue/fdr_corr', best_combo_fdr)
                            append_value(results, key_prefix + '/carrier/pvalue/n_randomizations', 0)
                            append_value(results, key_prefix + '/needs_randomization', 1)
                            keep_testing = False
                            break

                    if best_combo_fdr < pvalue_threshold_for_permutation_tests or not adaptive_fdr or np.isnan(best_combo_pvalue):
                        append_value(results, key_prefix + '/carrier/pvalue/fdr_corr', best_combo_fdr)
                        append_value(results, key_prefix + '/carrier/pvalue/n_randomizations', 0)
                        append_value(results, key_prefix + '/needs_randomization', 0)
                        keep_testing = False
                        break

            if return_random_stats:
                for c in random_cols:
                    rand_col_name = key_prefix + '/' + c + '/random'
                    results[rand_col_name][-1].extend(random_cols[c])

            if approximate_null_distribution:
                random_stats.extend(random_cols['carrier/stat'])

            if keep_testing:

                c_more_significant += np.sum(np.array(random_cols['carrier/pvalue']) <= best_real_pvalue)
                n_randomizations += randomization_batch_size
                c_fdr = c_more_significant / n_randomizations

                if c_fdr > 0:
                    rands_needed = 100 / c_fdr
                else:
                    rands_needed = MAX_N_RANDOMIZATIONS

                keep_testing = n_randomizations < min(MAX_N_RANDOMIZATIONS, rands_needed)

                echo('Current adaptive FDR for gene:',
                     gene_name,
                     key_prefix,
                     ', iteration:', gene_iteration_idx,
                     ', best_real_pvalue:', best_real_pvalue,
                     ', best_real_bh_fdr:', best_real_bh_fdr,
                     ', c_fdr:', c_fdr,
                     ', c_more_significant:', c_more_significant,
                     ', n_randomizations:', n_randomizations,
                     ', rands_needed:', rands_needed,
                     ', random_stats:', len(random_stats),
                     ', keep_testing:', keep_testing,
                     )

                if keep_testing and approximate_null_distribution and len(random_stats) >= n_samples_for_approximation:

                    shape, loc, scale = scipy.stats.genextreme.fit(np.abs(random_stats))
                    approx_c_fdr = scipy.stats.genextreme.sf(np.abs(best_real_stat), shape, loc, scale)

                    echo(gene_name, key_prefix, 'Approximating null distribution with scipy.stats.genextreme(',
                         shape,
                         loc,
                         scale,
                         '), stat=',
                         best_real_stat, ', n=', len(random_stats), ', approx_c_fdr:', approx_c_fdr)

                    if approx_c_fdr < best_real_bh_fdr:
                        c_fdr = approx_c_fdr
                        keep_testing = False
                    else:
                        echo(gene_name, key_prefix, 'Approximation failed! Discarding random stats!')
                        random_stats = []

                if not keep_testing:
                    # echo('Done')
                    if c_fdr == 0:
                        echo(gene_name, key_prefix, 'c_fdr was estimated to be 0, reverting to BH FDR:', best_real_bh_fdr)
                        c_fdr = best_real_bh_fdr

                    append_value(results, key_prefix + '/carrier/pvalue/fdr_corr', c_fdr)
                    append_value(results, key_prefix + '/carrier/pvalue/n_randomizations', n_randomizations)
                    append_value(results, key_prefix + '/needs_randomization', 0)

#####


def rv_test_for_batch_binary_v4(genes_variants,
                                MAX_pAI_THRESHOLDS_TO_TEST,
                                find_best_AC_threshold,
                                find_best_pAI_threshold,
                                key_prefix,
                                pAI_thresholds_to_test,
                                pathogenicity_score_label,

                                ph_data,
                                ph_name,

                                sort_by_pvalue_label,
                                results,

                                test_type=LOGIT_ADJUSTED_BY_CHI2,

                                is_age_at_diagnosis=False,

                                MAX_N_RANDOMIZATIONS=100000,
                                randomization_batch_size=1000,
                                adaptive_fdr=True,
                                skip_randomizing_intermediate_significant=False,
                                approximate_null_distribution=False,
                                n_samples_for_approximation=10000,
                                return_random_stats=False,
                                MIN_MORE_SIGNIFICANT=100,
                                debug=False,
                                yates_correction=False,
                                min_variants_per_gene=2,
                                min_cases_in_carriers=5,
                                pvalue_threshold_for_permutation_tests=1e-5
                                ):

    echo('[rv_test_for_batch_binary_v4]!, adaptive_fdr:', adaptive_fdr,
         ', MAX_N_RANDOMIZATIONS:', MAX_N_RANDOMIZATIONS,
         ', MIN_MORE_SIGNIFICANT:', MIN_MORE_SIGNIFICANT,
         ', test_type:', test_type,
         ', approximate_null_distribution:', approximate_null_distribution,
         ', n_samples_for_approximation:', n_samples_for_approximation,
         ', return_random_stats:', return_random_stats,
         ', randomization_batch_size:', randomization_batch_size,
         ', skip_randomizing_intermediate_significant:', skip_randomizing_intermediate_significant,
         ', min_variants_per_gene:', min_variants_per_gene,
         ', min_cases_in_carriers:', min_cases_in_carriers,
         ', yates_correction:', yates_correction,
         ', pvalue_threshold_for_permutation_tests:', pvalue_threshold_for_permutation_tests
         )

    gene_names = sorted(genes_variants[GENE_NAME].unique())

    all_samples_set = set(ph_data[SAMPLE_ID])
    n_all_samples = len(all_samples_set)

    ph_values_array = list(ph_data[ph_name])
    sid_to_idx = dict((sid, idx) for idx, sid in enumerate(ph_data[SAMPLE_ID]))

    n_all_cases = np.sum(ph_values_array)

    p_case = n_all_cases / n_all_samples
    p_control = 1 - p_case

    all_cases = set(ph_data[ph_data[ph_name] == 1][SAMPLE_ID])
    all_controls = set(ph_data[ph_data[ph_name] == 0][SAMPLE_ID])

    if is_age_at_diagnosis:
        aod_label = ph_name + '/original_age_at_diagnosis'
        aod_samples = ph_data[ph_data[aod_label] >= 0]

        sid_to_aod = dict((sid, aod) for sid, aod in zip(aod_samples[SAMPLE_ID],
                                                         aod_samples[aod_label]))

        sid_to_age = dict((sid, age) for sid, age in zip(ph_data[SAMPLE_ID],
                                                         ph_data[AGE]))

    for gene_name in gene_names:
        cur_gene_variants = genes_variants[genes_variants[GENE_NAME] == gene_name]

        all_carriers = sorted(set(sid for _, r in cur_gene_variants.iterrows() for sid in r[ALL_SAMPLES].split(',')) & all_samples_set)
        all_carriers_set = set(all_carriers)

        frac_cases_per_variant = {}
        avg_aod_per_variant = {}
        avg_control_age_per_variant = {}
        var_to_pAI = {}

        carrier_to_pAI = {}
        for _, v in cur_gene_variants.iterrows():
            v_carriers = v[ALL_SAMPLES].split(',')

            v_cases = set(v_carriers) & all_cases
            v_controls = set(v_carriers) & all_controls

            frac_cases_per_variant[v[VARID_REF_ALT]] = len(v_cases) / len(v_carriers)
            var_to_pAI[v[VARID_REF_ALT]] = v[pathogenicity_score_label]

            if is_age_at_diagnosis:
                aod_for_cases = [sid_to_aod[sid] for sid in v_cases if sid in sid_to_aod]
                if len(aod_for_cases) > 0:
                    avg_aod_per_variant[v[VARID_REF_ALT]] = np.median(aod_for_cases)

                age_for_controls = [sid_to_age[sid] for sid in v_controls if sid in sid_to_age]
                if len(age_for_controls) > 0:
                    avg_control_age_per_variant[v[VARID_REF_ALT]] = np.median(age_for_controls)

            for sid in v_carriers:
                if sid not in carrier_to_pAI:
                    carrier_to_pAI[sid] = v[pathogenicity_score_label]
                else:
                    carrier_to_pAI[sid] = max(carrier_to_pAI[sid], v[pathogenicity_score_label])

        n_all_carriers = len(all_carriers)
        keep_testing = True

        c_more_significant = 0
        n_randomizations = 0
        gene_iteration_idx = 0
        random_stats = []

        best_real_pvalue = None
        best_real_bh_fdr = None
        best_real_stat = None
        best_chi2_pvalue = None

        while keep_testing:
            gene_iteration_idx += 1

            # randomizing data
            C_INT = ctypes.c_int

            result = (C_INT * (n_all_carriers * (randomization_batch_size)))()

            cutils.shuffle_array(n_all_samples, n_all_carriers, randomization_batch_size, result)

            result = np.array(result).reshape((randomization_batch_size, n_all_carriers))
            rand_idx = dict((sid_no,
                             np.concatenate([[sid_to_idx[sid]],
                                              result[:, sid_no]
                                             ])) for sid_no, sid in enumerate(all_carriers))

            sid_to_phenotypes = {}
            for sid_no, sid in enumerate(all_carriers):
                sid_to_phenotypes[sid] = np.array(
                    [ph_values_array[c_rand_idx] for c_rand_idx in rand_idx[sid_no]])

            # pre-computing chi2 statistics across the real data and all randomized samples
            variants_info_for_rv = precompute_variant_stats_for_chi2_test(gene_name,
                                                                          cur_gene_variants,
                                                                          sid_to_phenotypes,
                                                                          randomization_batch_size + 1,
                                                                          all_samples_set,

                                                                          MAX_pAI_THRESHOLDS_TO_TEST,
                                                                          find_best_AC_threshold,
                                                                          find_best_pAI_threshold,

                                                                          pAI_thresholds_to_test,
                                                                          pathogenicity_score_label,
                                                                          )

            # filter out combos that have no cases among the carriers
            variants_info_for_rv = [cur_var_info for cur_var_info in variants_info_for_rv
                                    if cur_var_info['n_cases'][0] >= min_cases_in_carriers and
                                       cur_var_info['info']['n_variants'] >= min_variants_per_gene]
            gene_results = []

            # echo('performing tests')
            for cur_var_info in variants_info_for_rv:

                cur_var_info['n_tests'] = len(variants_info_for_rv)

                combo_n_cases = cur_var_info['n_cases']
                combo_n_controls = cur_var_info['n_carriers'] - combo_n_cases

                n_carriers = cur_var_info['n_carriers']

                if test_type in [LOGIT_ADJUSTED_BY_CHI2,
                                 CHI2_TEST
                                 ]:

                    if yates_correction:
                        expected_cases = n_carriers * p_case
                        expected_controls = n_carriers * p_control

                        chi2_stats = (np.square(np.abs(combo_n_cases - expected_cases) - 0.5) / expected_cases +
                                      np.square(np.abs(combo_n_controls - expected_controls) - 0.5) / expected_controls)

                    else:
                        chi2_stats = np.square(combo_n_cases) / (n_carriers * p_case) + np.square(combo_n_controls) / (n_carriers * p_control) - n_carriers

                    p_values = scipy.stats.chi2.sf(chi2_stats, 1)
                    stats = chi2_stats
                    betas = (combo_n_cases / combo_n_controls) / (p_case / p_control)

                else:
                    echo('[ERROR] Unknown test type:', test_type)
                    return None

                gene_results.append(p_values)
                cur_var_info['betas'] = betas
                cur_var_info['stats'] = stats

            if len(gene_results) == 0:
                echo(gene_name, 'No viable combos found!')
                break

            gene_results = np.array(gene_results)
            best_combo_indexes = np.argmin(gene_results, axis=0)

            COLUMNS_FOR_RANDOMIZED_DATA = ['carrier/beta', 'carrier/stat', 'carrier/pvalue', 'carrier/pvalue/fdr_corr', 'best_pAI_threshold', 'best_AC', 'n_variants', 'n_carriers']
            random_cols = dict((c, []) for c in COLUMNS_FOR_RANDOMIZED_DATA)

            # echo('Finding best combo for each permutation')
            for ph_idx in range(randomization_batch_size + 1):
                best_combo_idx = best_combo_indexes[ph_idx]
                best_combo = variants_info_for_rv[best_combo_idx]

                ph_pvalues = gene_results[:, ph_idx]

                _, fdr_corrected_pvalues = statsmodels.stats.multitest.fdrcorrection(ph_pvalues)

                best_combo_BH_fdr = fdr_corrected_pvalues[best_combo_idx]

                best_combo_pvalue = ph_pvalues[best_combo_idx]

                beta = best_combo['betas'][ph_idx]
                odds_ratio = beta

                stat = best_combo['stats'][ph_idx]

                row_info = best_combo['info']

                if ph_idx > 0:
                    # for random data, only collect the summary stats
                    for c in COLUMNS_FOR_RANDOMIZED_DATA:
                        if c == 'carrier/beta':
                            v = beta
                        elif c == 'carrier/stat':
                            v = stat
                        elif c == 'carrier/pvalue':
                            v = best_combo_pvalue
                        elif c == 'carrier/pvalue/fdr_corr':
                            v = best_combo_BH_fdr
                        elif c == 'n_carriers':
                            v = best_combo['n_carriers']
                        else:
                            v = row_info[c]

                        random_cols[c].append(v)

                    continue

                if gene_iteration_idx == 1:

                    best_real_stat = stat
                    best_real_pvalue = best_combo_pvalue
                    best_real_bh_fdr = best_combo_BH_fdr

                    if test_type == LOGIT_ADJUSTED_BY_CHI2:
                        best_chi2_pvalue = best_combo_pvalue
                        ph_data['carrier'] = ph_data[SAMPLE_ID].isin(best_combo['carriers']).astype(int)

                        lm = sm.Logit(ph_data[ph_name],
                                      ph_data[['carrier', CONST_LABEL]])

                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")

                            try:

                                lm_res = lm.fit(disp=0)
                                logit_pvalue = lm_res.pvalues[0]
                                logit_tvalue = lm_res.tvalues[0]
                                logit_beta = lm_res.params[0]
                                if debug:
                                    display(lm_res.summary())

                            except np.linalg.LinAlgError:
                                logit_pvalue = 1
                                logit_tvalue = 0
                                logit_beta = 0

                    append_value(results, IS_REAL_DATA, 1)

                    append_value(results, SHUFFLED_IDX, ph_idx)

                    append_value(results, GENE_NAME, gene_name)
                    append_value(results, key_prefix + '/sort_by', key_prefix + '/%s/pvalue/fdr_corr' % sort_by_pvalue_label)
                    append_value(results, key_prefix + '/n_carriers', best_combo['n_carriers'])

                    append_value(results, key_prefix + '/carriers', ','.join(sorted(best_combo['carriers'])))
                    append_value(results, key_prefix + '/n_tests', best_combo['n_tests'])

                    append_value(results, key_prefix + '/variants', ','.join(sorted(best_combo['variants'])))

                    for key in row_info:
                        value = row_info[key]

                        res_key = key_prefix + '/' + key
                        append_value(results, res_key, value)

                    if test_type == LOGIT_ADJUSTED_BY_CHI2:
                        append_value(results, key_prefix + '/carrier/chi2/pvalue', best_combo_pvalue)
                        append_value(results, key_prefix + '/carrier/chi2/pvalue/BH_fdr_corr', best_combo_BH_fdr)
                        append_value(results, key_prefix + '/carrier/chi2/stat', stat)
                        append_value(results, key_prefix + '/carrier/chi2/odds_ratio', odds_ratio)

                        best_combo_pvalue = logit_pvalue
                        stat = logit_tvalue
                        beta = logit_beta
                        odds_ratio = np.exp(logit_beta)

                    append_value(results, key_prefix + '/carrier/beta', beta)
                    append_value(results, key_prefix + '/carrier/odds_ratio', odds_ratio)
                    append_value(results, key_prefix + '/carrier/stat', stat)

                    append_value(results, key_prefix + '/test_type', test_type)

                    append_value(results, key_prefix + '/carrier/pvalue', best_combo_pvalue)

                    carrier_cases = set(best_combo['carriers']) & all_cases
                    carrier_controls = set(best_combo['carriers']) & all_controls

                    append_value(results, key_prefix + '/carriers/cases', ','.join(sorted(carrier_cases)))
                    append_value(results, key_prefix + '/carriers/controls', ','.join(sorted(carrier_controls)))
                    append_value(results, key_prefix + '/carriers/n_cases', len(carrier_cases))
                    append_value(results, key_prefix + '/carriers/n_controls', len(carrier_controls))

                    all_cases_among_all_carriers = sorted(all_cases & all_carriers_set)
                    all_controls_among_all_carriers = sorted(all_controls & all_carriers_set)

                    append_value(results, key_prefix + '/all_carriers/cases', ','.join(all_cases_among_all_carriers))
                    append_value(results, key_prefix + '/all_carriers/n_cases', len(all_cases_among_all_carriers))

                    append_value(results, key_prefix + '/all_carriers/controls', ','.join(all_controls_among_all_carriers))
                    append_value(results, key_prefix + '/all_carriers/n_controls', len(all_controls_among_all_carriers))

                    cases_pai = [carrier_to_pAI[sid] for sid in all_cases_among_all_carriers]
                    ctrls_pai = [carrier_to_pAI[sid] for sid in all_controls_among_all_carriers]

                    rs_stat, rs_pval = scipy.stats.ranksums(cases_pai, ctrls_pai)
                    cases_pai_mean = np.mean(cases_pai)
                    ctrls_pai_mean = np.mean(ctrls_pai)

                    append_value(results, key_prefix + '/carrier/pathogenicity_score/ranksums_stat', rs_stat)
                    append_value(results, key_prefix + '/carrier/pathogenicity_score/ranksums_pvalue', rs_pval)
                    append_value(results, key_prefix + '/carrier/pathogenicity_score/mean_diff',
                                 cases_pai_mean - ctrls_pai_mean)
                    append_value(results, key_prefix + '/carrier/pathogenicity_score/cases_mean', cases_pai_mean)
                    append_value(results, key_prefix + '/carrier/pathogenicity_score/controls_mean', ctrls_pai_mean)

                    sc_stat, sc_pval = scipy.stats.spearmanr(
                        [frac_cases_per_variant[vid] for vid in sorted(frac_cases_per_variant)],
                        [var_to_pAI[vid] for vid in sorted(frac_cases_per_variant)])

                    append_value(results,
                                 key_prefix + '/carrier/pathogenicity_score/fraction_cases_vs_score/spearmanr_stat',
                                 sc_stat)
                    append_value(results,
                                 key_prefix + '/carrier/pathogenicity_score/fraction_cases_vs_score/spearmanr_pvalue',
                                 sc_pval)

                    if is_age_at_diagnosis:

                        # compute correlation between pathogenicity scores and age at diagnosis for cases (this should be negative for true positive genes)

                        aod_per_carrier = [sid_to_aod[sid] for sid in all_cases_among_all_carriers]
                        aod_carrier_level_corr, aod_carrier_level_pvalue = scipy.stats.spearmanr(aod_per_carrier,
                                                                                                 cases_pai)

                        if len(aod_per_carrier) > 1:
                            corr = scipy.stats.pearsonr(aod_per_carrier, cases_pai)[0]
                        else:
                            corr = 0

                        pAI_aod_carrier_level_beta = corr * np.std(cases_pai) / np.std(aod_per_carrier)

                        append_value(results,
                                     key_prefix + '/pathogenicity_score/age_at_diagnosis/carrier_level/spearman_r',
                                     aod_carrier_level_corr)
                        append_value(results,
                                     key_prefix + '/pathogenicity_score/age_at_diagnosis/carrier_level/spearman_pvalue',
                                     aod_carrier_level_pvalue)
                        append_value(results,
                                     key_prefix + '/pathogenicity_score/age_at_diagnosis/carrier_level/regression_beta',
                                     pAI_aod_carrier_level_beta)
                        append_value(results,
                                     key_prefix + '/pathogenicity_score/age_at_diagnosis/carrier_level/n_carriers',
                                     len(aod_per_carrier))

                        v1 = [avg_aod_per_variant[vid] for vid in sorted(avg_aod_per_variant)]
                        v2 = [var_to_pAI[vid] for vid in sorted(avg_aod_per_variant)]
                        aod_variant_level_corr, aod_variant_level_pvalue = scipy.stats.spearmanr(v1, v2)

                        if len(v1) > 1:
                            corr = scipy.stats.pearsonr(v1, v2)[0]
                        else:
                            corr = 0

                        pAI_aod_variant_level_beta = corr * np.std(v2) / np.std(v1)

                        append_value(results,
                                     key_prefix + '/pathogenicity_score/age_at_diagnosis/variant_level/spearman_r',
                                     aod_variant_level_corr)
                        append_value(results,
                                     key_prefix + '/pathogenicity_score/age_at_diagnosis/variant_level/spearman_pvalue',
                                     aod_variant_level_pvalue)
                        append_value(results,
                                     key_prefix + '/pathogenicity_score/age_at_diagnosis/variant_level/regression_beta',
                                     pAI_aod_variant_level_beta)
                        append_value(results,
                                     key_prefix + '/pathogenicity_score/age_at_diagnosis/variant_level/n_variants',
                                     len(avg_aod_per_variant))

                        # compute correlation between pathogenicity scores and age for controls (this should be positive for true positive genes)

                        age_per_carrier_control = [sid_to_age[sid] for sid in all_controls_among_all_carriers]
                        control_age_carrier_level_corr, control_age_carrier_level_pvalue = scipy.stats.spearmanr(
                            age_per_carrier_control, ctrls_pai)

                        append_value(results, key_prefix + '/pathogenicity_score/controls_age/carrier_level/spearman_r',
                                     control_age_carrier_level_corr)
                        append_value(results,
                                     key_prefix + '/pathogenicity_score/controls_age/carrier_level/spearman_pvalue',
                                     control_age_carrier_level_pvalue)
                        append_value(results, key_prefix + '/pathogenicity_score/controls_age/carrier_level/n_carriers',
                                     len(age_per_carrier_control))

                        control_age_variant_level_corr, control_age_variant_level_pvalue = scipy.stats.spearmanr(
                            [avg_control_age_per_variant[vid] for vid in sorted(avg_control_age_per_variant)],
                            [var_to_pAI[vid] for vid in sorted(avg_control_age_per_variant)])

                        append_value(results, key_prefix + '/pathogenicity_score/controls_age/variant_level/spearman_r',
                                     control_age_variant_level_corr)
                        append_value(results,
                                     key_prefix + '/pathogenicity_score/controls_age/variant_level/spearman_pvalue',
                                     control_age_variant_level_pvalue)
                        append_value(results, key_prefix + '/pathogenicity_score/controls_age/variant_level/n_variants',
                                     len(avg_control_age_per_variant))

                    if return_random_stats:
                        for c in random_cols:
                            rand_col_name = key_prefix + '/' + c + '/random'
                            append_value(results, rand_col_name, [])

                    if adaptive_fdr:
                        append_value(results, key_prefix + '/carrier/pvalue/BH_fdr_corr', best_real_bh_fdr)

                    if skip_randomizing_intermediate_significant:
                        if pvalue_threshold_for_permutation_tests < best_real_bh_fdr <= 1e-2 and adaptive_fdr:
                            append_value(results, key_prefix + '/carrier/pvalue/fdr_corr', best_real_bh_fdr)
                            append_value(results, key_prefix + '/carrier/pvalue/n_randomizations', 0)
                            append_value(results, key_prefix + '/needs_randomization', 1)

                            if test_type == LOGIT_ADJUSTED_BY_CHI2:
                                append_value(results, key_prefix + '/carrier/pvalue/chi2_correction', 0)
                                append_value(results, key_prefix + '/carrier/chi2/pvalue/fdr_corr', best_combo_BH_fdr)

                            keep_testing = False
                            break

                    if ((test_type == CHI2_TEST and
                         best_real_bh_fdr < pvalue_threshold_for_permutation_tests) or

                       (test_type == LOGIT_ADJUSTED_BY_CHI2 and
                         best_real_bh_fdr < pvalue_threshold_for_permutation_tests and
                         logit_pvalue < pvalue_threshold_for_permutation_tests) or

                        not adaptive_fdr):

                        if test_type == LOGIT_ADJUSTED_BY_CHI2:
                            best_real_logit_bh_fdr = get_logit_BH_fdr(ph_data, ph_name, variants_info_for_rv, best_combo_idx, debug=False)

                            echo(gene_name, ', too significant for permutation tests'
                                            ', best_real_bh_fdr:', best_real_bh_fdr,
                                            ', best_real_logit_bh_fdr:', best_real_logit_bh_fdr,
                                            ', logit_pvalue:', logit_pvalue)

                            best_real_bh_fdr = best_real_logit_bh_fdr

                        append_value(results, key_prefix + '/carrier/pvalue/fdr_corr', best_real_bh_fdr)
                        append_value(results, key_prefix + '/carrier/pvalue/n_randomizations', 0)
                        append_value(results, key_prefix + '/needs_randomization', 0)

                        if test_type == LOGIT_ADJUSTED_BY_CHI2:
                            append_value(results, key_prefix + '/carrier/pvalue/chi2_correction', 0)
                            append_value(results, key_prefix + '/carrier/chi2/pvalue/fdr_corr', best_combo_BH_fdr)

                        keep_testing = False
                        break

            if return_random_stats:
                for c in random_cols:
                    rand_col_name = key_prefix + '/' + c + '/random'
                    results[rand_col_name][-1].extend(random_cols[c])

            if approximate_null_distribution:
                random_stats.extend(random_cols['carrier/stat'])

            if keep_testing:

                c_more_significant += np.sum(np.array(random_cols['carrier/pvalue']) <= best_real_pvalue)
                n_randomizations += randomization_batch_size
                c_fdr = c_more_significant / n_randomizations

                if c_more_significant >= MIN_MORE_SIGNIFICANT:
                    keep_testing = False
                else:
                    keep_testing = n_randomizations < MAX_N_RANDOMIZATIONS

                echo('Current adaptive FDR for gene:',
                     gene_name,
                     key_prefix,
                     ', iteration:', gene_iteration_idx,
                     ', best_real_pvalue:', best_real_pvalue,
                     ', best_real_bh_fdr:', best_real_bh_fdr,
                     ', c_fdr:', c_fdr,
                     ', c_more_significant:', c_more_significant,
                     ', n_randomizations:', n_randomizations,
                     ', rands_needed:', MIN_MORE_SIGNIFICANT / c_fdr if c_fdr > 0 else MAX_N_RANDOMIZATIONS,
                     ', random_stats:', len(random_stats),
                     ', keep_testing:', keep_testing,
                     )

                if keep_testing and approximate_null_distribution and len(random_stats) >= n_samples_for_approximation:

                    shape, loc, scale = scipy.stats.genextreme.fit(np.abs(random_stats))
                    approx_c_fdr = scipy.stats.genextreme.sf(np.abs(best_real_stat), shape, loc, scale)

                    echo(gene_name, key_prefix, 'Approximating null distribution with scipy.stats.genextreme(',
                         shape,
                         loc,
                         scale,
                         '), stat=',
                         best_real_stat, ', n=', len(random_stats), ', approx_c_fdr:', approx_c_fdr)

                    if approx_c_fdr < best_real_bh_fdr:
                        c_fdr = approx_c_fdr
                        keep_testing = False
                    else:
                        echo(gene_name, key_prefix, 'Approximation failed! Discarding random stats!')
                        random_stats = []

                if not keep_testing:

                    if c_fdr == 0:
                        echo(gene_name, key_prefix, 'c_fdr was estimated to be 0, reverting to BH FDR:', best_real_bh_fdr)
                        c_fdr = best_real_bh_fdr

                    if test_type == LOGIT_ADJUSTED_BY_CHI2:
                        chi2_correction = 1

                        if best_chi2_pvalue < 1 and c_fdr < 1 and logit_pvalue < 1:
                            chi2_correction = np.log(1 - c_fdr) / np.log(1 - best_chi2_pvalue)

                            if np.isnan(chi2_correction) or np.isinf(chi2_correction):
                                chi2_correction = 10 ** (np.log10(c_fdr) - np.log10(best_chi2_pvalue))
                                echo(gene_name, 'Underflow problem, setting chi2_correction=', chi2_correction)

                            if chi2_correction < 1:
                                echo('[WARNING]', gene_name, 'chi2_correction is less than 1, setting it to 1:',
                                     chi2_correction)
                                chi2_correction = 1

                            chi2_c_fdr = c_fdr
                            c_fdr = 1 - (1 - logit_pvalue) ** chi2_correction
                        else:
                            chi2_c_fdr = c_fdr
                            c_fdr = 1

                        append_value(results, key_prefix + '/carrier/chi2/pvalue/fdr_corr', chi2_c_fdr)

                        echo(gene_name, ', chi2_correction:', chi2_correction, ', corrected c_fdr:', c_fdr,
                             ', chi2_c_fdr:', chi2_c_fdr,
                             ', logit_pvalue:', logit_pvalue,
                             ', best_chi2_pvalue:', best_chi2_pvalue)

                        append_value(results, key_prefix + '/carrier/pvalue/chi2_correction', chi2_correction)

                    append_value(results, key_prefix + '/carrier/pvalue/fdr_corr', c_fdr)
                    append_value(results, key_prefix + '/carrier/pvalue/n_randomizations', n_randomizations)
                    append_value(results, key_prefix + '/needs_randomization', 0)


def get_logit_BH_fdr(ph_data, ph_name, combos, best_combo_idx, debug=False):

    pvalues = []
    for cur_combo in combos:
        ph_data['carrier'] = ph_data[SAMPLE_ID].isin(cur_combo['carriers']).astype(int)

        lm = sm.Logit(ph_data[ph_name],
                      ph_data[['carrier', CONST_LABEL]])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            try:

                lm_res = lm.fit(disp=0)
                logit_pvalue = lm_res.pvalues[0]
                if debug:
                    display(lm_res.summary())

            except np.linalg.LinAlgError:
                logit_pvalue = 1

        pvalues.append(logit_pvalue)

    _, fdr_corrected_pvalues = statsmodels.stats.multitest.fdrcorrection(pvalues)

    return fdr_corrected_pvalues[best_combo_idx]




def precompute_variant_stats_for_chi2_test(gene_name,
                                           gene_variants,
                                           sid_to_phenotypes,
                                           n_phenotypes,
                                           all_samples,

                                           MAX_pAI_THRESHOLDS_TO_TEST,
                                           find_best_AC_threshold,
                                           find_best_pAI_threshold,

                                           pAI_thresholds_to_test,
                                           pathogenicity_score_label,
                                           ):

    variants_info = []
    gene_variants = gene_variants.sort_values(VCF_AC)

    if find_best_pAI_threshold:
        pAI_thresholds = sorted(gene_variants[pathogenicity_score_label].unique())

        if len(pAI_thresholds) > MAX_pAI_THRESHOLDS_TO_TEST:
            pAI_thresholds = [np.quantile(pAI_thresholds,
                                          i / (MAX_pAI_THRESHOLDS_TO_TEST - 1))
                              for i in range(MAX_pAI_THRESHOLDS_TO_TEST)]
    else:
        pAI_thresholds = [0]

    if pAI_thresholds_to_test is not None:
        pAI_thresholds = pAI_thresholds_to_test

    splitted = gene_variants[ALL_SAMPLES].str.split(',')

    var_to_carriers = {k: set(v) & all_samples for k, v in zip(gene_variants[VARID_REF_ALT], splitted)}
    var_to_pAI = {k: v for k, v in zip(gene_variants[VARID_REF_ALT], gene_variants[pathogenicity_score_label])}

    carrier_to_pAI = {}
    for v in var_to_carriers:
        for sid in var_to_carriers[v]:
            if sid not in carrier_to_pAI:
                carrier_to_pAI[sid] = var_to_pAI[v]
            else:
                carrier_to_pAI[sid] = max(carrier_to_pAI[sid], var_to_pAI[v])

    all_carriers = sorted(carrier_to_pAI)

    v1 = [sid_to_phenotypes[sid][0] for sid in all_carriers]
    v2 = [carrier_to_pAI[sid] for sid in all_carriers]

    if len(v1) < 2:
        pAI_corr_carrier_level = [0, 1]
        pAI_carrier_level_beta = 0
    else:
        pAI_corr_carrier_level = scipy.stats.spearmanr(v1, v2)
        corr = scipy.stats.pearsonr(v1, v2)[0]
        pAI_carrier_level_beta = corr * np.std(v2) / np.std(v1)

    all_variants = sorted(var_to_carriers)

    v1 = [np.median([sid_to_phenotypes[sid][0] for sid in var_to_carriers[v]]) for v in all_variants]
    v2 = [var_to_pAI[v] for v in all_variants]

    if len(v1) < 2:
        pAI_corr_variant_level = [0, 1]
        pAI_variant_level_beta = 0
    else:
        pAI_corr_variant_level = scipy.stats.spearmanr(v1, v2)
        corr = scipy.stats.pearsonr(v1, v2)[0]
        pAI_variant_level_beta = corr * np.std(v2) / np.std(v1)

    max_ac = np.max(gene_variants[VCF_AC])

    def get_new_combo_info(n_combo_cases,
                           carriers,
                           variants,
                           ac_threshold,
                           pAI_threshold,
                           pathogenicity_score_label):

        return {'n_cases': n_combo_cases,
                'n_carriers': len(carriers),
                'carriers': carriers,
                'variants': variants,
                'info': {'n_variants': len(variants),
                         'best_AC': int(ac_threshold),
                         'best_AF': int(ac_threshold) / (2 * len(all_samples)),
                         'best_pAI_threshold': float(pAI_threshold),
                         'pathogenicity_score_label': pathogenicity_score_label}}

    for pAI_threshold_idx, pAI_threshold in enumerate(sorted(pAI_thresholds, reverse=True)):
        c_vars_by_pAI = gene_variants[gene_variants[pathogenicity_score_label] >= pAI_threshold]

        if len(c_vars_by_pAI) == 0:
            continue

        if find_best_AC_threshold:
            ac_thresholds = sorted(c_vars_by_pAI[VCF_AC].unique())
        else:
            ac_thresholds = [max_ac]

        c_idx = 0

        ac_t_idx = 0
        c_ac = ac_thresholds[ac_t_idx]

        combo_info = get_new_combo_info(np.zeros(n_phenotypes),
                                        set(),
                                        set(),
                                        c_ac,
                                        pAI_threshold,
                                        pathogenicity_score_label)

        variants_info.append(combo_info)

        while c_idx < len(c_vars_by_pAI):
            var_id = c_vars_by_pAI.iloc[c_idx][VARID_REF_ALT]
            new_ac = c_vars_by_pAI.iloc[c_idx][VCF_AC]

            if new_ac <= c_ac:
                samples_to_add = var_to_carriers[var_id] - combo_info['carriers']
                combo_info['variants'].add(var_id)

                for sid in samples_to_add:
                    combo_info['n_cases'] = combo_info['n_cases'] + sid_to_phenotypes[sid]

                combo_info['carriers'] = combo_info['carriers'] | samples_to_add
                combo_info['n_carriers'] += len(samples_to_add)
                combo_info['info']['n_variants'] += 1
                c_idx += 1
            else:

                ac_t_idx += 1
                c_ac = ac_thresholds[ac_t_idx]

                combo_info = get_new_combo_info(np.array(combo_info['n_cases']),
                                                set(combo_info['carriers']),
                                                set(combo_info['variants']),
                                                c_ac,
                                                pAI_threshold,
                                                pathogenicity_score_label)

                variants_info.append(combo_info)

    # make variant_info unique based on the set of variants
    d = dict((tuple(sorted(v['variants'])), v) for v in variants_info)
    # echo('variants_info:', len(variants_info), len(d))
    variants_info = [d[k] for k in d]

    n_tests = len(variants_info)

    echo(gene_name, ', n_tests:', n_tests)
    for cur_combo_info in variants_info:
        cur_combo_info['n_tests'] = n_tests

        cur_combo_info['info']['pathogenicity_score/carrier_level/spearman_r'] = pAI_corr_carrier_level[0]
        cur_combo_info['info']['pathogenicity_score/carrier_level/spearman_pvalue'] = pAI_corr_carrier_level[1]
        cur_combo_info['info']['pathogenicity_score/carrier_level/regression_beta'] = pAI_carrier_level_beta

        cur_combo_info['info']['pathogenicity_score/variant_level/spearman_r'] = pAI_corr_variant_level[0]
        cur_combo_info['info']['pathogenicity_score/variant_level/spearman_pvalue'] = pAI_corr_variant_level[1]
        cur_combo_info['info']['pathogenicity_score/variant_level/regression_beta'] = pAI_variant_level_beta

        cur_combo_info['info']['all_carriers'] = all_carriers

        discarded_variants = sorted(set(all_variants) - set(cur_combo_info['variants']))
        cur_combo_info['info']['discarded_variants'] = ','.join(discarded_variants)
        cur_combo_info['info']['n_discarded_variants'] = len(discarded_variants)

        cur_combo_info['info']['n_total_carriers'] = len(all_carriers)
        cur_combo_info['info']['n_total_variants'] = len(all_variants)

    return variants_info
