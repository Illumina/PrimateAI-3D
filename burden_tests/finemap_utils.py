from jutils import *
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from ukb_analysis import *
import math
import pickle

import matplotlib
matplotlib.use('Agg')

import argparse
import json
import pprint
import tempfile

from constants import *
import requests, sys
import sqlite3

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from io import StringIO
import time
import urllib3

ANNOTATED_COMMON_VARIANTS = ROOT_PFIZIEV_PATH +'/rare_variants/data/finemapping/annotated_dbsnp.hg19.tsv.gz'
GENCODE_PATH = ROOT_PFIZIEV_PATH + '/rare_variants/data/gencode/gencode.v24lift37.canonical.with_CDS.tsv'
GENCODE_PATH_HG38 = ROOT_PFIZIEV_PATH + '/rare_variants/data/gencode/gencode.v24.annotation.gene_info.csv'

GENOMEWIDE_SIGN_THRESHOLD = 5e-8
MIN_LD_THRESHOLD = 0.5

WINDOW_NON_CODING = 50000

LAST_REQUEST_TIME = {}
ENSEMBL = 'ensembl'
PICS2 = 'pics2'


def finemap_region_conditional(snp_genotypes, ph_data, ph_name_label, data=None, interactions=False, plot_figures=True, pvalue_threshold=5e-8, verbose=True, batch_size=100):

    if data is None:
        echo('Merging genotypes and ph_data')
        d = pd.merge(ph_data[[SAMPLE_ID, ph_name_label]].set_index(SAMPLE_ID),
                     snp_genotypes,
                     left_index=True,
                     right_index=True)
    else:
        d = data.copy()

    d[ph_name_label + '/corrected'] = list(d[ph_name_label])

    var_labels = [c for c in list(d) if c not in [ph_name_label, ph_name_label + '/corrected']]
    echo('variants:', len(var_labels))

    lead_snp_no = 0
    lead_snps = []

    while True:
        lead_snp_no += 1

        if verbose:
            echo('Computing GWAS correlations for lead SNP:', lead_snp_no, ', batch_size:', batch_size)

        local_gwas_corr = None

        for batch_idx, batch_varids in enumerate(batchify(var_labels, batch_size)):

            batch_local_gwas_corr = vcorrcoef(d[[ph_name_label + '/corrected']],
                                              d[batch_varids],
                                              axis='columns',
                                              return_single_df=True,
                                              index_label='varid')

            if local_gwas_corr is None:
                local_gwas_corr = batch_local_gwas_corr
            else:
                local_gwas_corr = pd.concat([local_gwas_corr, batch_local_gwas_corr], ignore_index=True)

        local_gwas_corr['r2'] = local_gwas_corr[ph_name_label + '/corrected/r'] ** 2
        local_gwas_corr = local_gwas_corr.sort_values('r2', ascending=False)

        best_local_snp_id = local_gwas_corr.iloc[0]['varid']

        echo('local_gwas_corr:', len(local_gwas_corr))

        positive_pvalues = [p for p in list(local_gwas_corr[ph_name_label + '/corrected/pvalue']) if p > 0]
        if len(positive_pvalues) > 0:
            min_p = min(positive_pvalues)
        else:
            echo('All', len(local_gwas_corr), 'p-values are zero. Replacing with 1e-300:', list(local_gwas_corr[ph_name_label + '/corrected/pvalue']))
            min_p = 1e-300

        echo('min_p:', min_p)
        local_gwas_corr[ph_name_label + '/corrected/pvalue'] = np.where(local_gwas_corr[ph_name_label + '/corrected/pvalue'] == 0,
                                                                        min_p,
                                                                        local_gwas_corr[ph_name_label + '/corrected/pvalue'])

        best_p = local_gwas_corr.iloc[0][ph_name_label + '/corrected/pvalue']
        if plot_figures:
            plt.figure(figsize=(10, 5))

            plt.title(best_local_snp_id)

            plt.plot([int(k.split(':')[1]) for k in list(local_gwas_corr['varid'])], -np.log10(local_gwas_corr[ph_name_label + '/corrected/pvalue']), '.')
            plt.plot([int(best_local_snp_id.split(':')[1])], [-np.log10(best_p)], 'o', color='red')
            plt.show()

        if verbose:
            echo(best_local_snp_id, ', r=', local_gwas_corr.iloc[0][ph_name_label + '/corrected/r'], ', p=', best_p)

        if best_p > pvalue_threshold:
            break

        if verbose:
            echo('Regressing out lead SNPs:', lead_snps)

        lead_snps.append(best_local_snp_id)

        best_local_snp_ph_resid = get_residuals(d[[ph_name_label] + lead_snps],
                                                response_labels=[ph_name_label],
                                                covariate_labels=lead_snps,
                                                interactions=interactions,
                                                verbose=verbose)

        d[ph_name_label + '/corrected'] = best_local_snp_ph_resid[ph_name_label]

    echo('finemap_region_conditional: Lead SNPs=', lead_snps)
    return lead_snps


def plot_gwas_region(gene_name=None,
                     lead_snp_chrom=None,
                     lead_snp_pos=None,
                     finemapped_snps=None,
                     gwas_variants=None,
                     gencode=None,
                     c_rare_var_results=None,
                     rv_label=None,
                     pli_info=None,
                     window=500000,
                     pvalue_label='p_value',
                     n_rv_genes_to_show=None):

    if gene_name is not None:
        gene_info = gencode[gencode[GENE_NAME] == gene_name].iloc[0]
        lead_snp_chrom = gene_info[VCF_CHROM]
        lead_snp_pos = gene_info['tss_pos']

    echo('gene=', str(gene_name) + f', {lead_snp_chrom}:{lead_snp_pos}')

    plt.figure(figsize=(20, 10))

    plt.title((gene_name + ', ' if gene_name is not None else '') + f'{lead_snp_chrom}:{lead_snp_pos}')

    left_pos = lead_snp_pos - window
    right_pos = lead_snp_pos + window

    c_snps = gwas_variants[(gwas_variants[VCF_CHROM] == lead_snp_chrom) &
                           (gwas_variants[VCF_POS] > left_pos) &
                           (gwas_variants[VCF_POS] < right_pos)]

    echo('snps in region:', len(c_snps), ', p < 5e-8:', np.sum(c_snps[pvalue_label] < 5e-8))

    plt.plot(c_snps[VCF_POS], -np.log10(c_snps[pvalue_label]), '.', color='blue')
    if finemapped_snps is not None:
        index_snps_in_region = finemapped_snps[(finemapped_snps[VCF_CHROM] == lead_snp_chrom) &
                                               (finemapped_snps[VCF_POS] > left_pos) &
                                               (finemapped_snps[VCF_POS] < right_pos)].sort_values(VCF_POS)

        echo('index_snps_in_region')
        display(index_snps_in_region)
        for _, r in index_snps_in_region.iterrows():

            index_pval = r['pvalue']
            if index_pval == 0:
                index_pval = np.min(c_snps[pvalue_label])
            plt.plot([r[VCF_POS]], -np.log10(index_pval), 'o', color='red')

    if gencode is not None:
        locus_genes = gencode[
            (gencode['tss_pos'] > left_pos) & (gencode['tss_pos'] < right_pos) & (gencode['CHROM'] == lead_snp_chrom)]

        var_type = rv_label.split('/')[1]
        locus_genes = pd.merge(locus_genes, c_rare_var_results, on=GENE_NAME, how='left').fillna({rv_label: 10,
                                                                                                  f'ALL/{var_type}/n_carriers/total': -1})
        locus_genes = pd.merge(locus_genes, pli_info, on=GENE_NAME, how='left').fillna(-1)

        echo('Locus genes RV')
        display(c_rare_var_results[c_rare_var_results[GENE_NAME].isin(set(locus_genes[GENE_NAME]))].sort_values(rv_label)[[GENE_NAME] +
                                                                                                                          [c for c in list(c_rare_var_results) if var_type in c] +
                                                                                                                          [c for c in list(c_rare_var_results) if var_type not in c and c != GENE_NAME]].head(n_rv_genes_to_show))

        for _, gene_row in locus_genes.iterrows():
            plt.plot([max(gene_row['tx_start'], left_pos),
                      min(gene_row['tx_end'], right_pos)],
                     [-5, -5],
                     lw=5,
                     label=gene_row[GENE_NAME] + (', rv_p=%d (n=%d)' % (
                         -math.log(gene_row[rv_label], 10),
                         gene_row[f'ALL/{var_type}/n_carriers/total']) +
                                                  ', pLI=%.2lf' % (gene_row['pLI'])))
    plt.legend()

    plt.show()

    return c_snps


def plot_ld_conditional_gwas_region(lead_snp_chrom=None,
                                    lead_snp_pos=None,
                                    c_snps_genotypes=None,
                                    finemapped_snps=None,
                                    gwas_variants=None,
                                    gencode=None,
                                    rsid_label=VCF_RSID,
                                    varid_label=VARID_REF_ALT,
                                    c_rare_var_results=None,
                                    rv_label=None,
                                    pli_info=None):

    left_pos = lead_snp_pos - 500000
    right_pos = lead_snp_pos + 500000

    c_snps_varids = [c for c in list(c_snps_genotypes) if c != SAMPLE_ID]
    c_snps = gwas_variants[gwas_variants[varid_label].isin(c_snps_varids)]
    echo('c_snps:', len(c_snps))

    lead_snps_in_region = finemapped_snps[(finemapped_snps[VCF_CHROM] == lead_snp_chrom) &
                                          (finemapped_snps[VCF_POS] > left_pos) &
                                          (finemapped_snps[VCF_POS] < right_pos)]
    display(lead_snps_in_region)

    lead_snps_in_region = list(lead_snps_in_region['index_variant'])
    echo('lead_snps_in_region:', lead_snps_in_region)

    for c_lead_snp in lead_snps_in_region:
        echo(c_lead_snp)

        lead_snp_row = finemapped_snps[finemapped_snps['index_variant'] == c_lead_snp].iloc[0]

        lead_snp_info = c_snps[c_snps[rsid_label] == c_lead_snp].iloc[0]
        lead_snp_varid = lead_snp_info[VCF_CHROM] + ':' + str(lead_snp_info[VCF_POS]) + ':' + lead_snp_info[
            VCF_REF] + ':' + lead_snp_info[VCF_ALT]

        echo('Computing LD for:', lead_snp_varid)
        corr, pval = vcorrcoef(c_snps_genotypes, c_snps_genotypes[[lead_snp_varid]],
                               axis='columns')

        for _cls in lead_snps_in_region:
            _lead_snp_info = c_snps[c_snps[rsid_label] == _cls].iloc[0]
            _lead_snp_varid = _lead_snp_info[VCF_CHROM] + ':' + str(_lead_snp_info[VCF_POS]) + ':' + _lead_snp_info[
                VCF_REF] + ':' + _lead_snp_info[VCF_ALT]
            echo(c_lead_snp, _cls, ', r=', corr.loc[lead_snp_varid][_lead_snp_varid], ', r2=',
                 corr.loc[lead_snp_varid][_lead_snp_varid] ** 2, ', p=', pval.loc[lead_snp_varid][_lead_snp_varid])

        var_pos = [int(c.split(':')[1]) for c in list(corr)]
        var_r2 = dict((p, r ** 2) for p, r in zip(var_pos, corr.iloc[0]))

        plt.figure(figsize=(20, 10))
        plt.title(c_lead_snp + ', pos= ' + str(lead_snp_row[VCF_POS]))

        window = 500000

        left_pos = lead_snp_pos - window
        right_pos = lead_snp_pos + window

        d = pd.DataFrame({VCF_POS: c_snps[VCF_POS],
                          '-log10_gwas_p': -np.log10(c_snps['p_value']),
                          'r2': [var_r2[p] for p in c_snps[VCF_POS]]}).dropna()
        echo(np.max(d['r2']), np.min(d['r2']))

        sns.scatterplot(data=d,
                        x=VCF_POS,
                        y='-log10_gwas_p',
                        hue='r2',
                        size='r2',
                        sizes=(20, 200),
                        palette='YlOrRd')

        plt.plot([lead_snp_row[VCF_POS]], -np.log10(lead_snp_row['pvalue']), 'o', color='red')

        locus_genes = gencode[
            (gencode['tss_pos'] > left_pos) & (gencode['tss_pos'] < right_pos) & (gencode['CHROM'] == lead_snp_chrom)]

        for _, gene_row in locus_genes.iterrows():
            plt.plot([max(gene_row['tx_start'], left_pos),
                      min(gene_row['tx_end'], right_pos)],
                     [-5, -5],
                     lw=5,
                     #              color='red' if gene_row[GENE_NAME] == row[GENE_NAME] else None,
                     label=gene_row[GENE_NAME])
        plt.legend()

        plt.show()


def finemap_stepwise(gwas_variants,
                     min_AF,
                     ph_data,
                     ph_name_label,
                     window=500000,
                     zscore_label='zscore',
                     pvalue_label='pvalue',
                     max_p_value=5e-8,
                     bgen_data_chrom_to_fname_mapping=None):

    echo('Step-wise finemapping!!')

    result = {}
    seen = {}

    if max_p_value is not None:
        potential_lead_snps = gwas_variants[gwas_variants[pvalue_label] <= max_p_value]
    else:
        potential_lead_snps = gwas_variants

    potential_lead_snps = potential_lead_snps[
        (potential_lead_snps[VCF_AF] >= min_AF) & (potential_lead_snps[VCF_AF] <= 1 - min_AF)]

    echo('gwas_variants with p-value < ', max_p_value, ' and MAF >=', min_AF, ':', len(potential_lead_snps))
    sorted_snps = potential_lead_snps.copy()

    sorted_snps['__sort_by/' + zscore_label] = np.abs(sorted_snps[zscore_label])
    sorted_snps = sorted_snps.sort_values('__sort_by/' + zscore_label, ascending=False)
    bgen_data = {}
    main_lead_snp_idx = 0
    for row_idx, row in sorted_snps.iterrows():

        chrom = row[VCF_CHROM]
        pos = row[VCF_POS]

        varid = ':'.join([chrom, str(pos), row[VCF_REF], row[VCF_ALT]])

        if chrom not in seen:
            seen[chrom] = set()

        skip = False
        if varid in seen[chrom]:
            skip = True

        if not skip:
            main_lead_snp_idx += 1
            echo('Main lead SNP:', main_lead_snp_idx, ', '.join(k + '=' + str(row[k]) for k in row.keys()))

            lead_snps, snps_in_region, bgen_data = get_stepwise_independent_snps(chrom,
                                                                                 pos,
                                                                                 gwas_variants,
                                                                                 ph_data,
                                                                                 ph_name_label,
                                                                                 max_p_value=1e-3,
                                                                                 pvalue_label=pvalue_label,
                                                                                 window=window,
                                                                                 bgen_data=bgen_data,
                                                                                 bgen_data_chrom_to_fname_mapping=bgen_data_chrom_to_fname_mapping)
            echo('snps_in_region:', len(snps_in_region))

            for _varid in snps_in_region:
                seen[chrom].add(_varid)

            for _, lead_snp_row in lead_snps.iterrows():
                for k in lead_snp_row.keys():
                    if k not in result:
                        result[k] = []

                    result[k].append(lead_snp_row[k])

    return pd.DataFrame(result)


def get_stepwise_independent_snps(lead_snp_chrom,
                                  lead_snp_pos,
                                  gwas_variants,
                                  ph_data,
                                  ph_name_label,
                                  max_p_value=1.,
                                  pvalue_label='pvalue',
                                  window=500000,
                                  bgen_data=None,
                                  ALPHA_CUTOFF=0.05,
                                  bgen_data_chrom_to_fname_mapping=None):

    left_pos = lead_snp_pos - window
    right_pos = lead_snp_pos + window

    c_snps = gwas_variants[(gwas_variants[VCF_CHROM] == lead_snp_chrom) &
                           (gwas_variants[VCF_POS] > left_pos) &
                           (gwas_variants[VCF_POS] < right_pos) &
                           (gwas_variants[pvalue_label] <= max_p_value)].sort_values(VCF_POS)

    echo('c_snps:', c_snps.shape)

    c_snps_genotypes, bgen_data = get_dosages_for_variants(c_snps,
                                                           sample_ids=None,
                                                           rsid_label=VCF_RSID,
                                                           bgen_data=bgen_data,
                                                           return_bgen_data=True,
                                                           bgen_data_chrom_to_fname_mapping=bgen_data_chrom_to_fname_mapping)
    echo('c_snps_genotypes:', c_snps_genotypes.shape)

    echo('Correct genotypes for genetic principal components in europeans')
    gPC_data = ph_data[[SAMPLE_ID] + [c for c in list(ph_data) if 'gPC_' in c]]
    echo('gPC_data:', gPC_data.shape)

    c_snps_genotypes_eur_gPC_corrected = correct_genotypes_for_genetic_PCs(c_snps_genotypes, gPC_data, plot_figures=False, batch_size=100)

    echo('c_snps_genotypes_eur_gPC_corrected:', c_snps_genotypes_eur_gPC_corrected.shape)

    del c_snps_genotypes
    gc.collect()

    effective_number_of_independent_snps = get_effective_number_of_dimensions(c_snps_genotypes_eur_gPC_corrected)
    if np.isnan(effective_number_of_independent_snps):
        effective_number_of_independent_snps = len(list(c_snps_genotypes_eur_gPC_corrected))
        echo('effective_number_of_independent_snps is nan, setting to:', effective_number_of_independent_snps)

    pvalue_threshold = 1 - (1 - ALPHA_CUTOFF) ** (1 / effective_number_of_independent_snps)

    echo('pvalue_threshold at alpha=', ALPHA_CUTOFF, ':', pvalue_threshold)

    lead_snps = finemap_region_conditional(c_snps_genotypes_eur_gPC_corrected,
                                           ph_data,
                                           ph_name_label,
                                           interactions=False,
                                           plot_figures=False,
                                           pvalue_threshold=pvalue_threshold)

    snps_in_region = list(c_snps_genotypes_eur_gPC_corrected)

    return gwas_variants[gwas_variants[VARID_REF_ALT].isin(lead_snps)], snps_in_region, bgen_data


def get_connected_components_by_LD(varids, snp_genotypes, gwas_variants, ORDERS_OF_MAGNITUDE_THRESHOLD=2, P_VALUE_THRESHOLD=1e-6):

    echo('Getting connected components by LD')

    result = None

    corr, pval = vcorrcoef(snp_genotypes[varids], axis='columns')

    d = pval.reset_index().melt(id_vars='index')

    d = d[d['value'] < 1e-5]

    d['chr1'] = d['index'].apply(lambda s: s.split(':')[0])
    d['pos1'] = d['index'].apply(lambda s: s.split(':')[1]).astype(int)

    d['chr2'] = d['variable'].apply(lambda s: s.split(':')[0])
    d['pos2'] = d['variable'].apply(lambda s: s.split(':')[1]).astype(int)

    d = d[d['chr1'] != d['chr2']].sort_values('value')

    if P_VALUE_THRESHOLD is None:
        if len(d) > 0:
            echo('corr from different chromosomes:', len(d))

            P_VALUE_THRESHOLD = np.min(d['value'])
        else:
            P_VALUE_THRESHOLD = 1e-6

    echo('P_VALUE_THRESHOLD=', P_VALUE_THRESHOLD)

    while len(varids) > 0:

        queue = [varids[0]]

        c_idx = 0
        while True:

            c_varid_1 = queue[c_idx]

            added = False

            for c_varid_2 in varids:
                if pval.loc[c_varid_1][c_varid_2] <= P_VALUE_THRESHOLD:
                    if c_varid_2 not in queue:
                        added = True
                        queue.append(c_varid_2)
            c_idx += 1

            if not added:
                break

        varids = sorted(set(varids) - set(queue))

        queue_gwas_variants = gwas_variants[gwas_variants[VARID_REF_ALT].isin(queue)].sort_values('pvalue')

        lead_snp_row = queue_gwas_variants.iloc[0]

        queue_gwas_variants['lead_snp'] = lead_snp_row[VARID_REF_ALT]
        queue_gwas_variants['corr_with_lead_snp'] = list(corr.loc[lead_snp_row[VARID_REF_ALT]][queue_gwas_variants[VARID_REF_ALT]])

        if result is None:
            result = queue_gwas_variants
        else:
            result = pd.concat([result, queue_gwas_variants])

    return result.sort_values('pvalue')

UKB_VARIANTS_FOR_GWAS_PATH = UKB_DATA_PATH + '/array_genotypes/subset_for_gwas.europeans'

UKB_VARIANTS_FOR_GWAS_FNAME_MAPPING = dict((chrom, (UKB_VARIANTS_FOR_GWAS_PATH + '/' + f'ukb_imp_chr{chrom}_v3.ukb_variants_for_gwas.non_related_european.bgen',
                                                    UKB_VARIANTS_FOR_GWAS_PATH + '/' + f'ukb_imp_chr{chrom}_v3.ukb_variants_for_gwas.non_related_european.sample',
                                                    UKB_VARIANTS_FOR_GWAS_PATH + '/' + f'ukb_mfi_chr{chrom}_v3.ukb_variants_for_gwas.pickle'
                                                   )) for chrom in list(map(str, range(1, 23))) + ['X'])

UKB_VARIANTS_FOR_GWAS_FNAME_MAPPING[SPLIT_SAMPLE_IDS_BY_UNDERSCORE] = True

def finemap_chromosome(chrom,
                       all_gwas_variants,
                       min_AF,
                       ph_data,
                       ph_name_label,
                       window=500000,
                       zscore_label='zscore',
                       pvalue_label='pvalue',
                       max_p_value=5e-8,
                       spliceai_variants=None,
                       eqtl_variants=None,
                       gencode=None,
                       odds_ratio_label='odds_ratio',
                       ld_method='local',
                       fallback_local_LD_calculations_to_UKB=True,
                       out_dir=None,
                       ALPHA_CUTOFF=0.05,
                       SPLICEAI_CODING_TAG=None):

    echo('fine-mapping chrom:', chrom, ', ld_method:', ld_method)
    chrom_gwas_variants = all_gwas_variants[all_gwas_variants[VCF_CHROM] == chrom]

    fm = finemap_stepwise(chrom_gwas_variants,
                          min_AF,
                          ph_data,
                          ph_name_label,
                          window=window,
                          zscore_label=zscore_label,
                          pvalue_label=pvalue_label,
                          max_p_value=max_p_value,
                          bgen_data_chrom_to_fname_mapping=UKB_VARIANTS_FOR_GWAS_FNAME_MAPPING)

    echo('len(fm):', len(fm))
    if len(fm) == 0:
        echo('Nothing was fine-mapped on:', chrom)
        return None

    c_snps_genotypes = get_dosages_for_variants(fm.sort_values([VCF_CHROM, VCF_POS]),
                                                sample_ids=None,
                                                rsid_label=VCF_RSID,
                                                bgen_data_chrom_to_fname_mapping=UKB_VARIANTS_FOR_GWAS_FNAME_MAPPING
                                                )

    echo('c_snps_genotypes:', c_snps_genotypes.shape)

    echo('Correct genotypes for genetic principal components in europeans')
    gPC_data = ph_data[[SAMPLE_ID] + [c for c in list(ph_data) if 'gPC_' in c]]
    echo('gPC_data:', gPC_data.shape)

    c_snps_genotypes_eur_gPC_corrected = correct_genotypes_for_genetic_PCs(c_snps_genotypes,
                                                                           gPC_data,
                                                                           plot_figures=False)

    echo('c_snps_genotypes_eur_gPC_corrected:', c_snps_genotypes_eur_gPC_corrected.shape)

    effective_number_of_independent_snps = get_effective_number_of_dimensions(c_snps_genotypes_eur_gPC_corrected)
    if np.isnan(effective_number_of_independent_snps):
        effective_number_of_independent_snps = len(list(c_snps_genotypes_eur_gPC_corrected))
        echo('effective_number_of_independent_snps is nan, setting to:', effective_number_of_independent_snps)

    pvalue_threshold = 1 - (1 - ALPHA_CUTOFF) ** (1 / effective_number_of_independent_snps)

    echo('pvalue_threshold at alpha=', ALPHA_CUTOFF, ':', pvalue_threshold)

    fm2 = finemap_region_conditional(c_snps_genotypes_eur_gPC_corrected,
                                     ph_data,
                                     ph_name_label,
                                     interactions=False,
                                     plot_figures=False,
                                     pvalue_threshold=pvalue_threshold)

    echo('fm2:', fm2)
    res = get_connected_components_by_LD(fm2,
                                         c_snps_genotypes_eur_gPC_corrected,
                                         chrom_gwas_variants[chrom_gwas_variants[VARID_REF_ALT].isin(fm2)])

    echo('connected components:', list(zip(list(res[VARID_REF_ALT]), list(res['lead_snp']))))
    res = annotate_chrom_lead_snps(res,
                                   all_gwas_variants,
                                   ph_data,
                                   ph_name_label,
                                   eqtl_variants,
                                   spliceai_variants,
                                   gencode,
                                   out_dir,
                                   ld_method=ld_method,
                                   fallback_local_LD_calculations_to_UKB=fallback_local_LD_calculations_to_UKB,
                                   SPLICEAI_CODING_TAG=SPLICEAI_CODING_TAG)

    return res


def annotate_chrom_lead_snps(chrom_snps,
                             all_gwas_variants,
                             ph_data,
                             ph_name_label,
                             eqtl_variants,
                             spliceai_variants,
                             gencode,
                             out_dir,
                             ld_method='local',
                             fallback_local_LD_calculations_to_UKB=True,
                             ukb_ld_window=500000,
                             SPLICEAI_CODING_TAG=None
                             ):

    echo('[annotate_chrom_lead_snps]', len(chrom_snps), 'variants, ld_method:', ld_method, ', fallback_local_LD_calculations_to_UKB:', fallback_local_LD_calculations_to_UKB, '!!')

    if ld_method == 'local':
        find_all_snps_in_ld_local.ld_chrom = None
        find_all_snps_in_ld_local.ld_data = None

    chrom = chrom_snps.iloc[0][VCF_CHROM]

    echo('annotate_chrom_lead_snps:', chrom)

    CODING_SNPS = set(ALL_PTV_EXCEPT_SPLICE_VARIANTS + [VCF_MISSENSE_VARIANT])
    SPLICE_SNPS = SPLICE_VARIANTS + [SPLICEAI_CODING_TAG]

    res = {GENE_NAME: [],
           'assoc_type': [],
           'variant': [],
           'index_variant': [],
           'finemapped_via': [],
           'variant_type': [],
           'pvalue': [],
           'odds_ratio': [],
           'beta': [],
           VCF_POS: [],
           VCF_REF: [],
           VCF_ALT: [],
           VCF_AF: []
           }

    LD_LABEL = None
    chrom_snps_in_ld = None

    pvalue_label = 'pvalue'
    odds_ratio_label = 'odds_ratio'
    beta_label = 'beta'

    if ld_method == 'local':
        LD_LABEL = 'DELTASQ'
        chrom_snps_in_ld = find_all_snps_in_ld_local(chrom_snps,
                                                     min_ld=MIN_LD_THRESHOLD,
                                                     LD_LABEL=LD_LABEL,
                                                     fallback_local_LD_calculations_to_UKB=fallback_local_LD_calculations_to_UKB)

    elif ld_method == 'ensembl':
        LD_LABEL = 'r2'
        chrom_snps_in_ld = find_all_snps_in_ld_ensembl(chrom_snps, min_ld=MIN_LD_THRESHOLD, LD_LABEL=LD_LABEL)

    elif ld_method == 'ukb':
        LD_LABEL = 'r2'
        chrom_snps_in_ld = find_all_snps_in_ld_ukbiobank(chrom_snps, min_ld=MIN_LD_THRESHOLD, window=ukb_ld_window)
        chrom_snps_in_ld = chrom_snps_in_ld.rename(
            columns={'source_variant/' + VCF_RSID: 'source_variant',
                     'source_variant/' + VCF_CHROM: VCF_CHROM,
                     'source_variant/' + VCF_POS: 'pos_M1',
                     'source_variant/' + VCF_REF: 'REF_M1',
                     'source_variant/' + VCF_ALT: 'ALT_M1',

                     'ld_partner/' + VCF_RSID: VCF_RSID,
                     'ld_partner/' + VARID_REF_ALT: VARID_REF_ALT,
                     'ld_partner/' + VCF_POS: VCF_POS,
                     'ld_partner/' + VCF_REF: VCF_REF,
                     'ld_partner/' + VCF_ALT: VCF_ALT,
                     QC_AF: QC_AF + '/ld_partner'})

    echo('SNPs in LD:', len(chrom_snps_in_ld))

    var_to_ref_alt = dict((r['pos_M1'], (r['REF_M1'], r['ALT_M1'])) for _, r in chrom_snps_in_ld.iterrows())

    for _, r in chrom_snps_in_ld.iterrows():
        var_to_ref_alt[r[VCF_POS]] = (r[VCF_REF], r[VCF_ALT])

    chrom_snps_in_ld = annotate_variants(chrom_snps_in_ld, eqtl_variants, spliceai_variants, gencode, out_dir=out_dir)

    chrom_snps_cols = list(chrom_snps)

    chrom_snps = pd.merge(chrom_snps,
                          gencode[[VCF_CHROM, 'tss_pos', GENE_NAME]].rename(columns={GENE_NAME: 'nearest_gene_name',
                                                                                     'tss_pos': 'nearest_gene_tss_pos'}),
                          on=VCF_CHROM)

    chrom_snps['nearest_gene_distance'] = np.abs(chrom_snps[VCF_POS] - chrom_snps['nearest_gene_tss_pos'])

    chrom_snps = chrom_snps.sort_values('nearest_gene_distance').drop_duplicates(subset=chrom_snps_cols).sort_values(pvalue_label)

    echo('[annotate_chrom_lead_snps] chrom_snps_in_ld prior to computing GWAS p-values:', len(chrom_snps_in_ld))
    if ph_data is not None:
        chrom_snps_in_ld = get_gwas_pvalues(chrom, chrom_snps_in_ld, ph_data, ph_name_label)
        chrom_snps_in_ld = chrom_snps_in_ld.rename(columns={'gwas_pvalue': 'gwas_pvalue/ld_partner',
                                                            'gwas_beta': 'gwas_beta/ld_partner'})

    else:
        # use fake values if not phenotype values are provided
        chrom_snps_in_ld['gwas_pvalue/ld_partner'] = 1e-300
        chrom_snps_in_ld['gwas_beta/ld_partner'] = 1


    echo('after merge')

    src_var_info = chrom_snps_in_ld[[VCF_CHROM,
                                     VCF_POS,
                                     VCF_REF,
                                     VCF_ALT,
                                     'gwas_pvalue/ld_partner',
                                     QC_AF + '/ld_partner']].rename(
                                     columns={'gwas_pvalue/ld_partner': 'gwas_pvalue/source_variant',
                                             QC_AF + '/ld_partner': QC_AF + '/source_variant',
                                             VCF_REF: 'REF_M1',
                                             VCF_ALT: 'ALT_M1',
                                             VCF_POS: 'pos_M1'}).drop_duplicates(subset=[VCF_CHROM, 'pos_M1', 'REF_M1', 'ALT_M1'])

    chrom_snps_in_ld = pd.merge(chrom_snps_in_ld,
                                src_var_info,
                                on=[VCF_CHROM, 'pos_M1', 'REF_M1', 'ALT_M1'])

    echo('[annotate_chrom_lead_snps] chrom_snps_in_ld after joining src_var_info:', len(chrom_snps_in_ld))

    MAX_LOG_PVALUE_FRAC = 0.8

    chrom_snps_in_ld = chrom_snps_in_ld[-np.log10(chrom_snps_in_ld['gwas_pvalue/source_variant']) +
                                         np.log10(chrom_snps_in_ld['gwas_pvalue/ld_partner'])
                                         <= -MAX_LOG_PVALUE_FRAC * np.log10(chrom_snps_in_ld['gwas_pvalue/source_variant'])]

    echo('[annotate_chrom_lead_snps] chrom_snps_in_ld after GWAS p-value magnitude filtering:', len(chrom_snps_in_ld))

    seen = set()

    snp_id = lambda row: row[VCF_RSID]

    for row_idx, row in chrom_snps.sort_values(pvalue_label).iterrows():

        if row[VCF_RSID] in seen:
            echo('Seen:', row[VCF_RSID], ', skipping..')
            continue

        echo('Processing:', ', '.join(k + '=' + str(row[k]) for k in sorted(row.keys())))

        seen.add(row[VCF_RSID])

        snps_in_ld = chrom_snps_in_ld[chrom_snps_in_ld['source_variant'] == row[VCF_RSID]]

        res['variant'].append(snp_id(row))

        res['index_variant'].append(row.get('lead_snp', None))

        res['pvalue'].append(row[pvalue_label])
        res['odds_ratio'].append(row.get(odds_ratio_label, None))
        res['beta'].append(row.get(beta_label, None))

        res[VCF_POS].append(row[VCF_POS])

        cur_ref = row.get(VCF_REF, None)
        cur_alt = row.get(VCF_ALT, None)

        if cur_ref is None:
            cur_ref, cur_alt = var_to_ref_alt.get(row[VCF_POS], (None, None))

        res[VCF_REF].append(cur_ref)
        res[VCF_ALT].append(cur_alt)

        source_AF = snps_in_ld.iloc[0][QC_AF + '/source_variant'] if len(snps_in_ld) > 0 else None
        res[VCF_AF].append(source_AF)

        coding_snps = snps_in_ld[snps_in_ld[VCF_CONSEQUENCE].isin(CODING_SNPS)]
        splice_snps = snps_in_ld[snps_in_ld[VCF_CONSEQUENCE].isin(SPLICE_SNPS)]

        def finemapped_via_info(relevant_snps):
            return ','.join([snp_id(ld_snp) + '|%s|%s|%s|AF=%s|pvalue=%s|beta=%s|r2=%s|r=%s' % (
                                                                   ld_snp[VARID_REF_ALT],
                                                                   str(ld_snp[GENE_NAME]),
                                                                   str(ld_snp[VCF_CONSEQUENCE]),
                                                                   str(ld_snp[QC_AF + '/ld_partner']),
                                                                   str(ld_snp['gwas_pvalue/ld_partner']),
                                                                   str(ld_snp['gwas_beta/ld_partner']),
                                                                   str(ld_snp[LD_LABEL]),
                                                                   str(ld_snp['r'])) for _, ld_snp in relevant_snps.iterrows()])

        if len(coding_snps) >= 1 or len(splice_snps) >= 1:

            coding_genes = []
            for _, r in coding_snps.iterrows():
                if r[GENE_NAME] not in coding_genes:
                    coding_genes.append(r[GENE_NAME])

            splice_genes = []
            for _, r in splice_snps.iterrows():
                if r[GENE_NAME] not in splice_genes:
                    splice_genes.append(r[GENE_NAME])

            all_genes = sorted(set(coding_genes + splice_genes))

            res[GENE_NAME].append(','.join(all_genes))

            if len(coding_snps) >= 1 and len(splice_snps) == 0:
                res['assoc_type'].append('coding' if len(coding_genes) == 1 else 'coding/ambiguous')
                res['finemapped_via'].append(finemapped_via_info(coding_snps))
                res['variant_type'].append(','.join(coding_snps[VCF_CONSEQUENCE]))

            elif len(coding_snps) == 0 and len(splice_snps) >= 1:
                res['assoc_type'].append('splicing' if len(splice_genes) == 1 else 'splicing/ambiguous')
                res['finemapped_via'].append(finemapped_via_info(splice_snps))
                res['variant_type'].append(','.join(splice_snps[VCF_CONSEQUENCE]))
            else:
                res['assoc_type'].append(('coding' if len(all_genes) == 1 else 'coding/ambiguous') + ';' +
                                         ('splicing' if len(all_genes) == 1 else 'splicing/ambiguous'))

                res['finemapped_via'].append(finemapped_via_info(coding_snps) + ';' + finemapped_via_info(splice_snps))
                res['variant_type'].append(','.join(coding_snps[VCF_CONSEQUENCE]) + ';' + ','.join(splice_snps[VCF_CONSEQUENCE]))

        else:

            left_most = np.min(snps_in_ld[VCF_POS]) - WINDOW_NON_CODING
            right_most = np.max(snps_in_ld[VCF_POS]) + WINDOW_NON_CODING

            non_coding_genes = sorted(set(gencode[(gencode[VCF_CHROM] == chrom) &
                                                  (gencode['tss_pos'] >= left_most) &
                                                  (gencode['tss_pos'] <= right_most)][GENE_NAME]))

            # look for eQTLs
            eQTLs = snps_in_ld[snps_in_ld[VCF_CONSEQUENCE] == EQTL]
            eqtl_genes = sorted(set(eQTLs[GENE_NAME]))

            all_non_coding_genes = sorted(set(eqtl_genes) | set(non_coding_genes))

            if len(all_non_coding_genes) == 0:
                # default to nearest gene
                res[GENE_NAME].append(row['nearest_gene_name'])
                res['assoc_type'].append('non_coding')
                res['finemapped_via'].append(finemapped_via_info(snps_in_ld))
                res['variant_type'].append('nearest_gene|distance=' + str(row['nearest_gene_distance']))

            else:

                if len(all_non_coding_genes) == 1:

                    res[GENE_NAME].append(','.join(all_non_coding_genes))
                    res['assoc_type'].append('non_coding')

                    if len(eqtl_genes) == 1:
                        res['finemapped_via'].append(finemapped_via_info(eQTLs))
                        res['variant_type'].append(EQTL)
                    else:
                        res['finemapped_via'].append(finemapped_via_info(snps_in_ld))
                        res['variant_type'].append('non_coding')

                else:
                    res['assoc_type'].append('non_coding/ambiguous')
                    if len(eqtl_genes) > 0 and len(non_coding_genes) > 0:
                        res[GENE_NAME].append(','.join(non_coding_genes) + ';' + ','.join(eqtl_genes))
                        res['finemapped_via'].append(finemapped_via_info(snps_in_ld))
                        res['variant_type'].append('non_coding/eQTLs')

                    elif len(eqtl_genes) > 0:
                        res[GENE_NAME].append(','.join(eqtl_genes))
                        res['finemapped_via'].append(finemapped_via_info(eQTLs))
                        res['variant_type'].append(EQTL)
                    else:
                        res[GENE_NAME].append(','.join(non_coding_genes))
                        res['finemapped_via'].append(finemapped_via_info(snps_in_ld))
                        res['variant_type'].append('non_coding')

        for _, ld_snp in snps_in_ld.iterrows():
            seen.add(ld_snp[VCF_RSID])

        echo_last_row(res)

    res = pd.DataFrame(res)

    res[VCF_CHROM] = chrom

    return res


def get_gwas_pvalues(chrom, chrom_snps_in_ld, ph_data, ph_name_label, batch_size=100):
    # extract genotypes, compute correlations with phenotype and return p-values

    col_names = list(chrom_snps_in_ld)

    fname = UKB_DATA_PATH + f'/array_genotypes/qc/chr{chrom}.qc.eur.pickle'

    echo('[get_gwas_pvalues] chrom_snps_in_ld:', len(chrom_snps_in_ld))

    echo('Loading QC metrics:', fname)
    qc = pd.read_pickle(fname)

    chrom_snps_in_ld[VARID_REF_ALT] = (chrom_snps_in_ld[VCF_CHROM] + ':' +
                                       chrom_snps_in_ld[VCF_POS].astype(str) + ':' +
                                       chrom_snps_in_ld[VCF_REF] + ':' +
                                       chrom_snps_in_ld[VCF_ALT])

    chrom_snps_in_ld = pd.merge(chrom_snps_in_ld, qc, on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT], suffixes=('', '/qc'))

    chrom_snps_in_ld_with_gwas_stats = None
    echo('[get_gwas_pvalues] chrom_snps_in_ld:', len(chrom_snps_in_ld))

    bgen_data = None

    for batch_idx, batch in enumerate(batchify(list(chrom_snps_in_ld.index), batch_size)):

        batch_chrom_snps_in_ld = chrom_snps_in_ld.loc[batch]
        echo('[get_gwas_pvalues] batch_idx:', batch_idx, ', batch_varids:', len(batch_chrom_snps_in_ld))

        batch_chrom_snps_in_ld_non_red = batch_chrom_snps_in_ld.drop_duplicates(VARID_REF_ALT)

        c_snps_genotypes, bgen_data = get_dosages_for_variants(batch_chrom_snps_in_ld_non_red.sort_values([VCF_CHROM, VCF_POS]),
                                                               bgen_data=bgen_data,
                                                               sample_ids=None,
                                                               rsid_label=VCF_RSID,
                                                               return_bgen_data=True,
                                                               bgen_data_chrom_to_fname_mapping=None)

        echo('[get_gwas_pvalues] Correct genotypes for genetic principal components in europeans')
        gPC_data = ph_data[[SAMPLE_ID] + [c for c in list(ph_data) if 'gPC_' in c]]
        echo('gPC_data:', gPC_data.shape)

        c_snps_genotypes_eur_gPC_corrected = correct_genotypes_for_genetic_PCs(c_snps_genotypes,
                                                                               gPC_data,
                                                                               plot_figures=False,
                                                                               set_sample_ids_as_index=False)

        echo('[get_gwas_pvalues] c_snps_genotypes_eur_gPC_corrected:', c_snps_genotypes_eur_gPC_corrected.shape)

        varids = [c for c in list(c_snps_genotypes_eur_gPC_corrected) if c != SAMPLE_ID]
        ph_gt = pd.merge(ph_data, c_snps_genotypes_eur_gPC_corrected, on=SAMPLE_ID)
        r, _ = vcorrcoef(ph_gt[varids], ph_gt[ph_name_label])

        std_y = np.std(ph_gt[ph_name_label])
        std_vars = np.std(ph_gt[varids], axis=0)

        info_scores = dict((r[VARID_REF_ALT], r['info_score']) for _, r in batch_chrom_snps_in_ld_non_red.iterrows())

        n = len(ph_gt) * np.array([info_scores[v] for v in varids])

        ab = n / 2 - 1

        prob = 2 * scipy.special.btdtr(ab, ab, 0.5 * (1 - abs(np.float64(r))))

        gwas_res = pd.DataFrame({VARID_REF_ALT: varids,
                                 'gwas_pvalue': prob[0],
                                 'gwas_r': r.iloc[0],
                                 'gwas_beta': r.iloc[0] * std_y / std_vars,
                                 'effective_size': n})

        MIN_P = 1e-323
        gwas_res['gwas_pvalue'] = np.where(gwas_res['gwas_pvalue'] == 0, MIN_P, gwas_res['gwas_pvalue'])

        batch_chrom_snps_in_ld = pd.merge(batch_chrom_snps_in_ld,
                                          gwas_res,
                                          on=VARID_REF_ALT)[col_names + ['gwas_pvalue', 'gwas_r', 'gwas_beta', 'effective_size']]

        if chrom_snps_in_ld_with_gwas_stats is None:
            chrom_snps_in_ld_with_gwas_stats = batch_chrom_snps_in_ld
        else:
            chrom_snps_in_ld_with_gwas_stats = pd.concat([chrom_snps_in_ld_with_gwas_stats, batch_chrom_snps_in_ld], ignore_index=True)

    echo('[get_gwas_pvalues] chrom_snps_in_ld_with_gwas_stats:', len(chrom_snps_in_ld_with_gwas_stats) if chrom_snps_in_ld_with_gwas_stats is not None else 0)

    return chrom_snps_in_ld_with_gwas_stats


def find_all_snps_in_ld_ensembl(row, min_ld=MIN_LD_THRESHOLD, LD_LABEL='r2'):

    all_snp_ids = [row[VCF_RSID]]
    if 'all_RSIDs' in row:
        all_snp_ids = sorted(set(all_snp_ids + row['all_RSIDs'].split(',')))

    main_snp_id = row[VCF_RSID]

    echo('main_snp_id:', main_snp_id, ', all_snp_ids:', all_snp_ids)

    final_result = None

    for snp_id in all_snp_ids:

        POP_1000G = 'CEU'
        ENSEMBL_REST_URL = f"https://grch37.rest.ensembl.org/ld/human/{snp_id}/1000GENOMES:phase_3:{POP_1000G}?r2={min_ld};attribs=1"

        echo('Fetching LD for:', snp_id, row[VCF_CHROM], row[VCF_POS], row[VCF_REF], row[VCF_ALT])

        if ENSEMBL in LAST_REQUEST_TIME:
            seconds_since_last_request = (
                        datetime.datetime.now() - LAST_REQUEST_TIME[ENSEMBL]).total_seconds()
            echo('Seconds since last request to ensembl:', seconds_since_last_request)

            if seconds_since_last_request < 5:
                echo('Sleeping for 5 second to prevent overloading server')
                time.sleep(5)

        r = requests.get(ENSEMBL_REST_URL, headers={"Content-Type": "application/json"})

        LAST_REQUEST_TIME[ENSEMBL] = datetime.datetime.now()

        if not r.ok:
            echo('[WARNING]: No LD info found for:', snp_id)
            response_json = []

        else:
            response_json = r.json()

            for s in response_json:
                s['r2'] = float(s['r2'])
                s['d_prime'] = float(s['d_prime'])
                s['start'] = int(s['start'])
                s['end'] = int(s['end'])

        echo('SNPs in LD:', len(response_json))

        if not any(s['variation'] == snp_id for s in response_json):
            self_ld = {'chr': row[VCF_CHROM],
                       'clinical_significance': [],
                       'd_prime': 1.,
                       'end': row[VCF_POS],
                       'population_name': '1000GENOMES:phase_3:' + POP_1000G,
                       'consequence_type': '',
                       'r2': 1.,
                       'start': row[VCF_POS],
                       'strand': 1,
                       'variation': snp_id
                       }

            response_json.append(self_ld)

        result = pd.DataFrame(response_json).rename(columns={'variation': VCF_RSID, 'start': VCF_POS, 'chr': VCF_CHROM})

        result['source_variant'] = main_snp_id

        snp_info = get_variant_info_from_ensembl(result, genome='hg19')

        result = pd.merge(result, snp_info, on=VCF_RSID, suffixes=['', '_snp_info'])
        result = result[[c for c in list(result) if c != 'clinical_significance']]

        if final_result is None:
            final_result = result
        else:
            final_result = pd.concat([final_result, result], ignore_index=True).drop_duplicates()

    return final_result


def find_all_snps_in_ld_ukbiobank(chrom_snps,
                                  min_ld=MIN_LD_THRESHOLD,
                                  window=500000,
                                  rsid_label=VCF_RSID,
                                  varid_label='varid',
                                  info_score_threshold=0.4,
                                  min_af=0,
                                  purity_threshold=0.5,
                                  purity_delta=0.1,
                                  min_purity_k=50,
                                  HWE_pvalue_threshold=10e-12,
                                  keep_self_ld_despite_qc=True
                                  ):

    echo('Fetching LD from UKB for:', len(chrom_snps), 'variants !!')

    if type(chrom_snps) is pd.Series:
        chrom_snps = pd.DataFrame([chrom_snps])

    chrom = list(set(chrom_snps[VCF_CHROM]))

    if len(chrom) > 1:
        echo('[ERROR]: LD in UKB should be done one chromsome at a time:', chrom)
        return None

    chrom = chrom[0]

    if find_all_snps_in_ld_ukbiobank.chrom != chrom:

        echo('Loading LD cache for chromosome:', chrom)
        find_all_snps_in_ld_ukbiobank.chrom = chrom

        fname = UKB_DATA_PATH + f'/array_genotypes/ld/ukb_ld.small.chr{chrom}.db'
        echo('Connecting to LD db:', fname)
        find_all_snps_in_ld_ukbiobank.db = sqlite3.connect(fname)

        fname = UKB_DATA_PATH + f'/array_genotypes/qc/chr{chrom}.qc.eur.pickle'
        echo('Loading QC metrics:', fname)
        find_all_snps_in_ld_ukbiobank.qc = pd.read_pickle(fname).fillna(0)

    db = find_all_snps_in_ld_ukbiobank.db
    qc = find_all_snps_in_ld_ukbiobank.qc

    def ld_for_varid(db, chrom_snps, min_r2=None, qc=None):

        positions = sorted(set(chrom_snps[VCF_POS]))
        pos_to_var_info = {}
        for _, r in chrom_snps.iterrows():
            pos_to_var_info[r[VCF_POS]] = {}
            pos_to_var_info[r[VCF_POS]][VCF_RSID] = r.get(rsid_label, chrom + ':' + str(r[VCF_POS]))
            pos_to_var_info[r[VCF_POS]][VCF_REF] = r.get(VCF_REF, None)
            pos_to_var_info[r[VCF_POS]][VCF_ALT] = r.get(VCF_ALT, None)
            pos_to_var_info[r[VCF_POS]][VARID] = r.get(VARID,
                                                       chrom + ':' + str(r[VCF_POS]) + ':' + r.get(VCF_REF, 'nan') + ':' + r.get(VCF_ALT, 'nan'))

        positions_string = ', '.join(map(str, positions))

        echo('Executing query 1')
        query = f'SELECT * from ld WHERE pos_1 IN ({positions_string})'

        if min_r2 is not None:
            query += ' AND r2 >= ?'
            params = (min_r2,)
        else:
            params = None

        res_1 = pd.read_sql_query(query, db, params=params)

        echo('Executing query 2')
        query = f'SELECT * from ld WHERE pos_2 IN ({positions_string})'

        if min_r2 is not None:
            query += ' AND r2 >= ?'
            params = (min_r2,)
        else:
            params = None

        # swap pos and varid for the results from the second query
        # so that varid_1 and pos_1 point to the source variant in both results
        res_2 = pd.read_sql_query(query, db, params=params)
        _temp_varid = list(res_2['varid_1'])
        _temp_pos = list(res_2['pos_1'])

        res_2['varid_1'] = res_2['varid_2']
        res_2['pos_1'] = res_2['pos_2']

        res_2['varid_2'] = _temp_varid
        res_2['pos_2'] = _temp_pos

        if len(res_1) == 0:
            res = res_2
        elif len(res_2) == 0:
            res = res_1
        else:
            res = pd.concat([res_1, res_2], ignore_index=True).drop_duplicates(subset=['varid_1', 'varid_2'])

        res['keep_despite_qc'] = False

        # create entries for each variant being in perfect LD with itself
        self_ld = res[['varid_1', 'pos_1']].drop_duplicates()
        self_ld['varid_2'] = self_ld['varid_1']
        self_ld['pos_2'] = self_ld['pos_1']
        self_ld['r'] = 1
        self_ld['r2'] = 1
        self_ld['p'] = 0
        self_ld['r2_max'] = 1

        # add variants that were not found in the LD database
        missing_positions = sorted(set(positions) - set(self_ld['pos_1']))

        if len(missing_positions) > 0:
            self_ld = pd.concat([self_ld,
                                 pd.DataFrame({'pos_1': missing_positions,
                                               'pos_2': missing_positions,
                                               'varid_1': [pos_to_var_info[pos][VARID] for pos in missing_positions],
                                               'varid_2': [pos_to_var_info[pos][VARID] for pos in missing_positions],
                                               'r': 1,
                                               'r2': 1,
                                               'p': 0,
                                               'r2_max': 1})],
                                ignore_index=True,
                                sort=True)

        self_ld['keep_despite_qc'] = keep_self_ld_despite_qc

        if len(res) > 0:
            res = pd.concat([self_ld, res],
                            ignore_index=True,
                            sort=True).sort_values('r2',
                                                   ascending=False)
        else:
            res = self_ld

        # look up QC statistics for all LD partners
        res = pd.merge(res, qc, left_on=['varid_2', 'pos_2'], right_on=['varid', VCF_POS], how='left')

        # fix values for variants that were not found in the QC table
        res[VCF_REF] = res[VCF_REF].astype(str)
        res[VCF_ALT] = res[VCF_ALT].astype(str)

        def replace_missing_rsid(r):
            if not r[VCF_RSID] or (type(r[VCF_RSID]) is float and np.isnan(r[VCF_RSID])):
                if r['pos_2'] in pos_to_var_info:
                    return pos_to_var_info[r['pos_2']][VCF_RSID]
                else:
                    return None
            else:
                return r[VCF_RSID]

        res[VCF_RSID] = res.apply(replace_missing_rsid, axis=1)

        # get variant info for all source variants from the QC table
        src_varids = pd.merge(self_ld,
                              res,
                              left_on=['varid_1', 'pos_1'],
                              right_on=['varid_2', 'pos_2'],
                              suffixes=['', '/qc'])[[VCF_RSID, 'pos_1', VCF_REF, VCF_ALT, 'varid_1']]

        # use pos_1 as position, because some variants may be missing in the QC table
        src_varids = src_varids.rename(columns={'pos_1': VCF_POS})
        src_varids[VCF_CHROM] = chrom
        src_varids[VARID_REF_ALT] = src_varids[VCF_CHROM] + ':' + src_varids[VCF_POS].astype(str) + ':' + src_varids[VCF_REF] + ':' + src_varids[VCF_ALT]

        # replace missing RSIDs with the varid entry, which comes from self_ld for variants that are not found in the QC table
        src_varids[VCF_RSID] = np.where(src_varids[VCF_RSID].isnull(), src_varids['varid_1'], src_varids[VCF_RSID])

        src_varids = src_varids.rename(columns=dict((k, 'source_variant/' + k) for k in [VCF_RSID, VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT, VARID_REF_ALT]))

        res = pd.merge(res, src_varids, on='varid_1')

        varid_ref_alt = sorted(set(res['source_variant/' + VARID_REF_ALT]))
        echo(f'Filtering by info_score above {info_score_threshold}:', len(res), ', varid_ref_alt:', varid_ref_alt, ', min info_score:', np.min(res['info_score']))
        res = res[(res['info_score'] >= info_score_threshold) | res['keep_despite_qc']].copy()
        echo('After filtering by info_score:', len(res), ', varid_ref_alt:', varid_ref_alt)

        echo(f'Filtering by HWE_pvalue above {HWE_pvalue_threshold}:', len(res), ', varid_ref_alt:', varid_ref_alt)
        res = res[(res['HWE_pvalue'] >= HWE_pvalue_threshold) | res['keep_despite_qc']].copy()
        echo('After filtering by HWE_pvalue:', len(res), ', varid_ref_alt:', varid_ref_alt)

        echo(f'Filtering by AF above {min_af}:', len(res), ', varid_ref_alt:', varid_ref_alt)
        res = res[(res[QC_AF] >= min_af) & (res[QC_AF] <= 1 - min_af) | res['keep_despite_qc']].copy()
        echo('After filtering by AF:', len(res), ', varid_ref_alt:', varid_ref_alt)

        echo(f'Filtering by purity delta: {purity_delta} above threshold {purity_threshold}, min k={min_purity_k}:', len(res), ', varid_ref_alt:', varid_ref_alt)
        res = res[((res[f'purity|delta_{purity_delta}|ref_hom|frac'] >= purity_threshold) &
                  (res[f'purity|delta_{purity_delta}|alt_hom|frac'] >= purity_threshold) &
                  (res[f'purity|delta_{purity_delta}|het|frac'] >= purity_threshold)) | res['keep_despite_qc']].copy()

        res = res[((res[f'purity|delta_{purity_delta}|ref_hom|k'] >= min_purity_k) &
                  (res[f'purity|delta_{purity_delta}|alt_hom|k'] >= min_purity_k) &
                  (res[f'purity|delta_{purity_delta}|het|k'] >= min_purity_k)) | res['keep_despite_qc']].copy()

        echo('After filtering by purity:', len(res), ', varid_ref_alt:', varid_ref_alt)

        # use pos_2 instead of VCF_POS for the ld_partner because some variants may not be found in the LD or QC tables
        res['ld_partner/' + rsid_label] = res[VCF_RSID]
        res['ld_partner/' + VARID_REF_ALT] = chrom + ':' + res['pos_2'].astype(str) + ':' + res[VCF_REF] + ':' + res[VCF_ALT]
        res['ld_partner/' + VCF_CHROM] = chrom
        res['ld_partner/' + VCF_POS] = res['pos_2']
        res['ld_partner/' + VCF_REF] = res[VCF_REF]
        res['ld_partner/' + VCF_ALT] = res[VCF_ALT]

        return res

    results = { 'source_variant/' + rsid_label: [],
                'source_variant/' + VARID_REF_ALT: [],
                'source_variant/' + VCF_CHROM: [],
                'source_variant/' + VCF_POS: [],
                'source_variant/' + VCF_REF: [],
                'source_variant/' + VCF_ALT: [],

                'ld_partner/' + rsid_label: [],
                'ld_partner/' + VARID_REF_ALT: [],
                'ld_partner/' + VCF_CHROM: [],
                'ld_partner/' + VCF_POS: [],
                'ld_partner/' + VCF_REF: [],
                'ld_partner/' + VCF_ALT: [],
                'r2': [],
                'r': []}

    for k in list(qc):
        if k not in [VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT, VCF_RSID, 'varid']:
                results[k] = []

    snps_with_corr_info = ld_for_varid(db, chrom_snps, min_r2=min_ld, qc=qc)

    for _, ld_row in snps_with_corr_info.iterrows():
        for k in results:
            results[k].append(ld_row[k])

    results = pd.DataFrame(results)
    echo('results before dedup:', len(results))
    results = results.drop_duplicates(subset=['source_variant/' + VCF_POS, 'ld_partner/' + VCF_POS])
    echo('results after dedup:', len(results))

    return results

find_all_snps_in_ld_ukbiobank.chrom = None
find_all_snps_in_ld_ukbiobank.db = None
find_all_snps_in_ld_ukbiobank.qc = None


def find_all_snps_in_ld_ukbiobank_old(chrom_snps,
                                      min_ld=MIN_LD_THRESHOLD,
                                      window=500000,
                                      rsid_label=VCF_RSID,
                                      varid_label='varid',
                                      info_score_threshold=0.4,
                                      min_af=0,
                                      purity_threshold=0.5,
                                      purity_delta=0.1,
                                      min_purity_k=50,
                                      HWE_pvalue_threshold=10e-12,
                                      keep_self_ld_despite_qc=True
                                      ):

    echo('Fetching LD from UKB for:', len(chrom_snps), 'variants')

    if type(chrom_snps) is pd.Series:
        chrom_snps = pd.DataFrame([chrom_snps])

    chrom = list(set(chrom_snps[VCF_CHROM]))

    if len(chrom) > 1:
        echo('[ERROR]: LD in UKB should be done one chromsome at a time:', chrom)
        return None

    chrom = chrom[0]

    if find_all_snps_in_ld_ukbiobank.chrom != chrom:

        echo('Loading LD cache for chromosome:', chrom)
        find_all_snps_in_ld_ukbiobank.chrom = chrom

        fname = UKB_DATA_PATH + f'/array_genotypes/ld/ukb_ld.small.chr{chrom}.db'
        echo('Connecting to LD db:', fname)
        find_all_snps_in_ld_ukbiobank.db = sqlite3.connect(fname)

        fname = UKB_DATA_PATH + f'/array_genotypes/qc/chr{chrom}.qc.eur.pickle'
        echo('Loading QC metrics:', fname)
        find_all_snps_in_ld_ukbiobank.qc = pd.read_pickle(fname).fillna(0)

    db = find_all_snps_in_ld_ukbiobank.db
    qc = find_all_snps_in_ld_ukbiobank.qc

    def ld_for_varid(db, pos, ref, alt, varid, varid_ref_alt, rsid, min_r2=None, qc=None):

        query = '''SELECT * from ld WHERE (pos_1 = ? OR pos_2 = ?)'''

        if min_r2 is not None:
            query += ' AND r2 >= ?'
            params = (pos, pos, min_r2)
        else:
            params = (pos, pos)

        res = pd.read_sql_query(query, db, params=params)
        if len(res) == 0:
            echo('No SNPs in LD found for:', chrom, pos, varid)

        update_ref_alt = False
        if varid is None:

            _d1 = res[res['pos_1'] == pos].groupby('varid_1').size()
            _d2 = res[res['pos_2'] == pos].groupby('varid_2').size()

            _d = pd.Series(dict((k, _d1.get(k, 0) + _d2.get(k, 0)) for k in set(_d1.keys()) | set(_d2.keys()))).sort_values(
                    ascending=False)

            if len(_d) > 0:
                varid = _d.index[0]
                update_ref_alt = True


        res_1 = res[res['varid_1'] == varid].copy()
        res_2 = res[res['varid_2'] == varid].copy()

        res_2['varid_2'] = res_2['varid_1']
        res_2['varid_1'] = varid

        res_2['pos_2'] = res_2['pos_1']
        res_2['pos_1'] = pos

        self_ld = pd.DataFrame({'varid_1': [varid],
                                'varid_2': [varid],
                                'pos_1': [pos],
                                'pos_2': [pos],
                                'r': [1],
                                'r2': [1],
                                'p': [0],
                                'r2_max': [1]
                                })

        res = pd.concat([self_ld, res_1, res_2],
                        ignore_index=True,
                        sort=True).sort_values('r2',
                                               ascending=False).drop_duplicates(subset=['varid_1', 'varid_2'])
        if qc is not None:
            # echo('before merge with qc:', res.shape)
            res = pd.merge(res, qc, left_on=['varid_2', 'pos_2'], right_on=['varid', VCF_POS])
            if update_ref_alt:
                _d = res[res['varid'] == varid]
                if len(_d) > 0:
                    _d = _d.iloc[0]
                    ref = _d[VCF_REF]
                    alt = _d[VCF_ALT]
                    rsid = _d[VCF_RSID]
                    varid_ref_alt = chrom + ':' + str(pos) + ':' + ref + ':' + alt


        if keep_self_ld_despite_qc:
            res['keep_despite_qc'] = (res['varid_1'] == res['varid_2'])
        else:
            res['keep_despite_qc'] = False

        echo(f'Filtering by info_score above {info_score_threshold}:', len(res), ', varid_ref_alt:', varid_ref_alt, ', min info_score:', np.min(res['info_score']))
        res = res[(res['info_score'] >= info_score_threshold) | res['keep_despite_qc']].copy()
        echo('After filtering by info_score:', len(res), ', varid_ref_alt:', varid_ref_alt)

        echo(f'Filtering by HWE_pvalue above {HWE_pvalue_threshold}:', len(res), ', varid_ref_alt:', varid_ref_alt)
        res = res[(res['HWE_pvalue'] >= HWE_pvalue_threshold) | res['keep_despite_qc']].copy()
        echo('After filtering by HWE_pvalue:', len(res), ', varid_ref_alt:', varid_ref_alt)

        echo(f'Filtering by AF above {min_af}:', len(res), ', varid_ref_alt:', varid_ref_alt)
        res = res[(res[QC_AF] >= min_af) & (res[QC_AF] <= 1 - min_af) | res['keep_despite_qc']].copy()
        echo('After filtering by AF:', len(res), ', varid_ref_alt:', varid_ref_alt)

        echo(f'Filtering by purity delta: {purity_delta} above threshold {purity_threshold}, min k={min_purity_k}:', len(res), ', varid_ref_alt:', varid_ref_alt)
        res = res[((res[f'purity|delta_{purity_delta}|ref_hom|frac'] >= purity_threshold) &
                  (res[f'purity|delta_{purity_delta}|alt_hom|frac'] >= purity_threshold) &
                  (res[f'purity|delta_{purity_delta}|het|frac'] >= purity_threshold)) | res['keep_despite_qc']].copy()

        res = res[((res[f'purity|delta_{purity_delta}|ref_hom|k'] >= min_purity_k) &
                  (res[f'purity|delta_{purity_delta}|alt_hom|k'] >= min_purity_k) &
                  (res[f'purity|delta_{purity_delta}|het|k'] >= min_purity_k)) | res['keep_despite_qc']].copy()

        echo('After filtering by purity:', len(res), ', varid_ref_alt:', varid_ref_alt)

        res['source_variant/' + rsid_label] = rsid
        res['source_variant/varid_ref_alt'] = varid_ref_alt
        res['source_variant/' + VCF_CHROM] = chrom
        res['source_variant/' + VCF_POS] = pos
        res['source_variant/' + VCF_REF] = ref
        res['source_variant/' + VCF_ALT] = alt

        res['ld_partner/' + rsid_label] = res[VCF_RSID]
        res['ld_partner/varid_ref_alt'] = chrom + ':' + res[VCF_POS].astype(str) + ':' + res[VCF_REF] + ':' + res[VCF_ALT]
        res['ld_partner/' + VCF_CHROM] = chrom
        res['ld_partner/' + VCF_POS] = res[VCF_POS]
        res['ld_partner/' + VCF_REF] = res[VCF_REF]
        res['ld_partner/' + VCF_ALT] = res[VCF_ALT]

        return res

    results = { 'source_variant/' + rsid_label: [],
                'source_variant/varid_ref_alt': [],
                'source_variant/' + VCF_CHROM: [],
                'source_variant/' + VCF_POS: [],
                'source_variant/' + VCF_REF: [],
                'source_variant/' + VCF_ALT: [],

                'ld_partner/' + rsid_label: [],
                'ld_partner/varid_ref_alt': [],
                'ld_partner/' + VCF_CHROM: [],
                'ld_partner/' + VCF_POS: [],
                'ld_partner/' + VCF_REF: [],
                'ld_partner/' + VCF_ALT: [],
                'r2':[]}

    for k in list(qc):
        if k not in [VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT, VCF_RSID, 'varid']:
                results[k] = []

    for var_idx, (_, var_info) in enumerate(chrom_snps.iterrows()):
        if var_idx % 100 == 0:
            echo(var_idx, 'variants processed')

        pos = var_info[VCF_POS]

        varid = var_info.get(varid_label, None)
        rsid = var_info.get(rsid_label, None)

        ref = var_info.get(VCF_REF, None)
        alt = var_info.get(VCF_ALT, None)

        varid_ref_alt = ':'.join(map(str, [chrom, pos, ref, alt]))

        snps_with_corr_info = ld_for_varid(db, pos, ref, alt, varid, varid_ref_alt, rsid, min_r2=min_ld, qc=qc)

        for _, ld_row in snps_with_corr_info.iterrows():
            for k in results:
                results[k].append(ld_row[k])

    results = pd.DataFrame(results).drop_duplicates()

    return results

find_all_snps_in_ld_ukbiobank_old.chrom = None
find_all_snps_in_ld_ukbiobank_old.db = None
find_all_snps_in_ld_ukbiobank_old.qc = None

def find_all_snps_in_ld_local(chrom_snps, min_ld=MIN_LD_THRESHOLD, LD_LABEL='DELTASQ', fallback_local_LD_calculations_to_UKB=True, ukb_ld_window=500000):

    LD_1000_GENOMES_PATH = ROOT_PFIZIEV_PATH + '/dbsnp/1000genomes.ld.EUR'

    snp_chrom = chrom_snps.iloc[0][VCF_CHROM]

    if find_all_snps_in_ld_local.ld_chrom != snp_chrom:

        find_all_snps_in_ld_local.ld_chrom = snp_chrom
        ld_fname = LD_1000_GENOMES_PATH + f'/v2.20101123.EUR.chr{snp_chrom}.xt.dbsnp.r2_greater_0_5.mini.multiallelic_fixed.pickle'

        echo('Loading LD for chromosome:', snp_chrom, ', fname:', ld_fname)

        if not os.path.exists(ld_fname):
            echo('[WARNING] NO LD data found for chromosome:', snp_chrom)
            find_all_snps_in_ld_local.ld_data = pd.DataFrame({'RSID_M1': [],
                                                              'CHROM': [],
                                                              'pos_M1': [],
                                                              'REF_M1': [],
                                                              'ALT_M1': [],
                                                              'RSID_M2': [],
                                                              'pos_M2': [],
                                                              'REF_M2': [],
                                                              'ALT_M2': [],
                                                              'DELTASQ': []
                                                              })
        else:
            with open(ld_fname, 'rb') as in_f:
                ld_data = pickle.load(in_f)

            echo('LD pairs:', len(ld_data))

            find_all_snps_in_ld_local.ld_data = ld_data

    ld_data = find_all_snps_in_ld_local.ld_data

    alt_rsids = dict((row[VCF_RSID], row[VCF_RSID]) for _, row in chrom_snps.iterrows())

    if 'all_RSIDs' in list(chrom_snps):
        for _, row in chrom_snps.iterrows():
            for alt_id in row['all_RSIDs'].split(','):
                alt_rsids[alt_id] = row[VCF_RSID]

    all_snp_ids = set(alt_rsids.keys())

    result = ld_data[(ld_data['RSID_M1'].isin(all_snp_ids)) &
                     (ld_data[LD_LABEL] >= min_ld)].rename(columns={'RSID_M1': 'source_variant',
                                                                    'RSID_M2': VCF_RSID,
                                                                    'pos_M2': VCF_POS,
                                                                    'CHROM_M2': VCF_CHROM,
                                                                    'REF_M2': VCF_REF,
                                                                    'ALT_M2': VCF_ALT})

    result['source_variant'] = result['source_variant'].apply(lambda x: alt_rsids[x])

    for _, row in chrom_snps.iterrows():
        snp_id = row[VCF_RSID]

        if len(result[result['source_variant'] == snp_id]) == 0:

            echo('[WARNING] NO other variants found in LD for:', snp_id)

            echo('Trying to fetch LD partners from ENSEMBL:', snp_id)

            snps_in_ld_from_other_sources = find_all_snps_in_ld_ensembl(row, min_ld=min_ld, LD_LABEL='r2')

            if fallback_local_LD_calculations_to_UKB and len(snps_in_ld_from_other_sources) == 1:
                # if no LD partners were found in Ensembl
                echo('NO LD partners found in ENSEMBL. Trying to fetch LD partners from UKB data:', snp_id)
                snps_in_ld_from_other_sources = find_all_snps_in_ld_ukbiobank(row, min_ld=min_ld, window=ukb_ld_window)
                echo('LD partners found in UKB data for', snp_id, ':', len(snps_in_ld_from_other_sources))

                snps_in_ld_from_other_sources = snps_in_ld_from_other_sources.rename(columns={'source_variant/' + VCF_RSID: 'source_variant',
                                                                                              'source_variant/' + VCF_CHROM: VCF_CHROM,
                                                                                              'source_variant/' + VCF_POS: 'pos_M1',
                                                                                              'source_variant/' + VCF_REF: 'REF_M1',
                                                                                              'source_variant/' + VCF_ALT: 'ALT_M1',

                                                                                              'ld_partner/' + VCF_RSID: VCF_RSID,
                                                                                              'ld_partner/' + VCF_POS: VCF_POS,
                                                                                              'ld_partner/' + VCF_REF: VCF_REF,
                                                                                              'ld_partner/' + VCF_ALT: VCF_ALT,
                                                                                              'r2': 'DELTASQ'})
            else:
                snps_in_ld_from_other_sources = snps_in_ld_from_other_sources.rename(columns={'r2': 'DELTASQ'})
                snps_in_ld_from_other_sources['pos_M1'] = row[VCF_POS]
                snps_in_ld_from_other_sources['REF_M1'] = row[VCF_REF]
                snps_in_ld_from_other_sources['ALT_M1'] = row[VCF_ALT]

            snps_in_ld_from_other_sources = snps_in_ld_from_other_sources[list(result)]

            result = pd.concat([result, snps_in_ld_from_other_sources])

    result = result.sort_values('source_variant')

    result = result.sort_values(LD_LABEL, ascending=False).drop_duplicates(subset=['source_variant', VCF_RSID])

    return result

find_all_snps_in_ld_local.ld_chrom = None
find_all_snps_in_ld_local.ld_data = None


def annotate_snps_with_nirvana(gwas_snps, genome, gene_names=None, out_dir=None):

    echo('Annotating variants with Nirvana:', len(gwas_snps))

    if out_dir is None:
        out_prefix = tempfile._get_default_tempdir() + '/' + next(tempfile._get_candidate_names())
    else:
        out_prefix = out_dir + '/' + next(tempfile._get_candidate_names()) + '.' + str(random.randint(0, 10**10))

    vcf_fname = out_prefix + '.tmp.vcf'
    nirvana_out_fname = out_prefix + '.tmp.nirvana'

    echo('Creating temp VCF for Nirvana:', vcf_fname)
    with open(vcf_fname, 'w') as out_f:
        out_f.write('''##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n''')

        to_output = gwas_snps[[VCF_CHROM, VCF_POS, VCF_RSID, VCF_REF, VCF_ALT]].drop_duplicates(subset=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT]).copy()
        echo('to_output:', len(to_output))

        for col_name in ['QUAL', 'FILTER', 'INFO']:
            to_output[col_name] = '.'

        to_output.sort_values([VCF_CHROM, VCF_POS]).to_csv(out_f, sep='\t', index=False, header=False)

    if genome == 'hg19':
        genome_assembly = 'GRCh37'
    else:
        genome_assembly = 'GRCh38'

    nirvana_cmd = f'/home/pfiziev/software/dotnet_2.1.SDK/dotnet /home/pfiziev/software/Nirvana/bin/Release/netcoreapp2.1/Nirvana.dll ' \
        f'-c {ROOT_PFIZIEV_PATH}/resources/Nirvana/Cache/{genome_assembly}/Both ' \
        f'-r {ROOT_PFIZIEV_PATH}/resources/Nirvana/References/Homo_sapiens.{genome_assembly}.Nirvana.dat ' \
        f'--sd {ROOT_PFIZIEV_PATH}/resources/Nirvana/SupplementaryAnnotation/{genome_assembly} ' \
        f'-i {vcf_fname} ' \
        f'-o {nirvana_out_fname}'

    echo('Running: ' + nirvana_cmd)
    p = subprocess.Popen([nirvana_cmd], shell=True)

    p.wait()

    echo('Reading nirvana output: ' + nirvana_out_fname + '.json.gz')

    with gzip.open(nirvana_out_fname + '.json.gz') as nirvana_outf:
        nirvana_output = json.load(nirvana_outf)

    annotation = {VCF_CHROM: [],
                  VCF_POS: [],
                  VCF_REF: [],
                  VCF_ALT: [],
                  VCF_CONSEQUENCE: [],
                  BIOTYPE: [],
                  GENE_NAME: []}

    for nirvana_record in nirvana_output['positions']:

        ref_allele = nirvana_record['refAllele']
        alt_alleles = nirvana_record['altAlleles']

        if len(nirvana_record['variants']) != len(alt_alleles):
            echo('ERROR:', nirvana_record)
            raise Exception("alt alleles don't match variants in json:" + str(alt_alleles) + ', ' + str(nirvana_record['variants']))

        for (variant, alt_allele) in zip(nirvana_record['variants'], alt_alleles):

            chrom = variant['chromosome']

            chrom = chrom.replace('chr', '')

            # store nirvana annotations
            seen = set()
            for cur_anno in variant.get('transcripts', []):

                if 'isCanonical' not in cur_anno or not cur_anno['isCanonical']:
                    continue

                gene_name = cur_anno['hgnc']
                bioType = cur_anno['bioType']

                for consequence in cur_anno['consequence']:
                    # remove duplicate annotations from Ensemble and Refseq
                    key = (gene_name, consequence)

                    if key in seen:
                        continue

                    seen.add(key)

                    annotation[VCF_CHROM].append(chrom)

                    annotation[VCF_POS].append(nirvana_record['position'])

                    annotation[VCF_REF].append(ref_allele)
                    annotation[VCF_ALT].append(alt_allele)

                    annotation[GENE_NAME].append(gene_name)
                    annotation[VCF_CONSEQUENCE].append(consequence)
                    annotation[BIOTYPE].append(bioType)

    an_df = pd.DataFrame(annotation)

    if gene_names is not None:
        an_df = an_df[an_df[GENE_NAME].isin(set(gene_names))]

    echo('an_df:', len(an_df), ', gwas_snps:', len(gwas_snps))
    snp_annotation = pd.merge(gwas_snps, an_df, on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT])

    echo('snp_annotation:', len(snp_annotation))
    # echo('Keeping nirvana output')
    os.unlink(nirvana_out_fname + '.json.gz')

    if os.path.exists(nirvana_out_fname + '.json.gz.jsi'):
        os.unlink(nirvana_out_fname + '.json.gz.jsi')

    os.unlink(vcf_fname)

    return snp_annotation


def echo_last_row(res):
    echo('row_idx=' + str(len(res[list(res.keys())[0]])), '\t', ', '.join(k + '=' + str(v[-1]) for k, v in res.items()))


def annotate_variants(snps_in_ld, eqtl_variants, spliceai_variants, gencode, out_dir=None):

    COLS_TO_KEEP = [VCF_RSID, VCF_CONSEQUENCE, GENE_NAME, 'INFO']

    annotated_snps_in_ld = annotate_snps_with_nirvana(snps_in_ld, 'hg19', gene_names=gencode[GENE_NAME], out_dir=out_dir)
    annotated_snps_in_ld['INFO'] = None

    spliceai_snps_in_ld = pd.merge(snps_in_ld,
                                   spliceai_variants,
                                   on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT],
                                   suffixes=['', '_spliceai_info'])

    eqtls_snps_in_ld = pd.merge(snps_in_ld,
                                eqtl_variants,
                                on=[VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT],
                                suffixes=['', '_eqtl_info'])

    all_annotated_snps_in_ld = pd.concat([annotated_snps_in_ld[COLS_TO_KEEP],
                                          spliceai_snps_in_ld[COLS_TO_KEEP],
                                          eqtls_snps_in_ld[COLS_TO_KEEP]],
                                         ignore_index=True)

    snps_in_ld = pd.merge(snps_in_ld, all_annotated_snps_in_ld, on=VCF_RSID, how='left')
    snps_in_ld[GENE_NAME] = np.where(snps_in_ld[GENE_NAME].isnull(), '', snps_in_ld[GENE_NAME])

    return snps_in_ld


def finemap(gwas_variants,
            ph_data,
            ph_name_label,
            pvalue_label='pvalue',
            min_AF=0.01,
            odds_ratio_label='odds_ratio',
            pvalue_threshold=GENOMEWIDE_SIGN_THRESHOLD,
            ld_method='local',
            fallback_local_LD_calculations_to_UKB=True,
            genome='hg19',
            min_spliceai_score=0.2,
            out_dir=None):

    all_gwas_variants = gwas_variants

    gwas_variants = gwas_variants[gwas_variants[pvalue_label] <= pvalue_threshold]
    echo('Keeping variants below p-value:', pvalue_threshold, ', n=', len(gwas_variants))

    gwas_variants = gwas_variants[(gwas_variants[VCF_AF] >= min_AF) & (gwas_variants[VCF_AF] <= 1 - min_AF)]
    echo('Keeping variants with MAF >=', min_AF, ', n=', len(gwas_variants))

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

    echo('Min. spliceAI score:', min_spliceai_score)
    spliceai_variants = spliceai_variants[spliceai_variants['INFO'] >= min_spliceai_score].copy()
    SPLICEAI_CODING_TAG = f'spliceAI>={min_spliceai_score}'
    spliceai_variants[VCF_CONSEQUENCE] = SPLICEAI_CODING_TAG

    echo('Reading:', eqtls_fname)
    eqtl_variants = pd.read_csv(eqtls_fname, sep='\t', dtype={VCF_CHROM: str, VCF_POS: int, VCF_ALT: str, VCF_REF: str, 'consistent_effects': int})

    eqtl_variants = eqtl_variants.drop_duplicates(subset=[GENE_NAME, VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT])

    eqtl_variants = pd.merge(eqtl_variants,
                             gencode[[GENE_NAME]]).rename(columns={'ID': VCF_RSID})

    echo('initial eQTLs:', len(eqtl_variants))

    eqtl_variants = eqtl_variants[eqtl_variants['consistent_effects'] == 1].copy()

    eqtl_variants['INFO'] = 'n_genes=' + eqtl_variants['n_genes'].map(str) + '|consistent_effect=' + eqtl_variants['consistent_effects'].map(
        str) + '|beta=' + eqtl_variants['max_effect'].map(str) + "|pval=" + eqtl_variants['best_pvalue'].map(str) + "|" + eqtl_variants[
                        'tissues']

    eqtl_variants[VCF_CONSEQUENCE] = EQTL

    echo('consistent effect eQTLs', len(eqtl_variants), list(eqtl_variants))

    res = None
    all_chroms = sorted(gwas_variants[VCF_CHROM].unique())

    # shuffle chromosome labels to avoid collisions in the file system due to multiple processes
    random.shuffle(all_chroms)
    n_chroms = len(all_chroms)

    for chrom_idx, chrom in enumerate(all_chroms):

        chrom_snps = gwas_variants[gwas_variants[VCF_CHROM] == chrom].sort_values(pvalue_label)

        echo('Fine-mapping chromosome:', chrom, f', ({chrom_idx} / {n_chroms})', ', variants:', len(chrom_snps))

        if len(chrom_snps) > 0:
            chrom_res = finemap_chromosome(chrom,
                                           all_gwas_variants,
                                           min_AF,
                                           ph_data,
                                           ph_name_label,
                                           window=500000,
                                           zscore_label='zscore',
                                           pvalue_label=pvalue_label,
                                           max_p_value=5e-8,
                                           spliceai_variants=spliceai_variants,
                                           eqtl_variants=eqtl_variants,
                                           gencode=gencode,
                                           odds_ratio_label=odds_ratio_label,
                                           ld_method=ld_method,
                                           fallback_local_LD_calculations_to_UKB=fallback_local_LD_calculations_to_UKB,
                                           out_dir=out_dir,
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


def get_snps_from_Jeremy_gwas_catalog(fname):

    echo('Reading GWAS index variants:', fname)

    d = pd.read_csv(fname,
                    sep='\t',
                    dtype={'chrom': str}).rename(columns={'chrom': VCF_CHROM,
                                                          'pos': VCF_POS,
                                                          'ref': VCF_REF,
                                                          'alt': VCF_ALT,
                                                          'rsid': VCF_RSID}).sort_values('p_value').drop_duplicates(
        subset=
        [VCF_CHROM,
         VCF_POS,
         VCF_REF,
         VCF_ALT]).sort_values([VCF_CHROM, VCF_POS])

    snpinfo = get_variant_info_from_ensembl(d, genome='hg19')

    res = pd.merge(d, snpinfo, on=VCF_RSID, suffixes=['_hg38', '']).sort_values('p_value')
    echo('total variants:', len(d), ', with hg19 coordinates:', len(res))

    #     display(res.head(20))
    return res


def get_GWAS_summary_stats(fname,
                           replace_zero_pvalues=True,
                           is_sqlite=False,
                           ukb_variants=None,
                           remove_variants_without_rsid=True,
                           is_gcta=False):

    echo('[get_GWAS_summary_stats]')

    P_VALUE_LABEL = 'pvalue'

    echo('Reading GWAS variants from:', fname)
    if is_sqlite:
        echo('Fetching data from sqlite')
        con = sqlite3.connect(fname)
        gwas_variants = pd.read_sql_query("SELECT * from variants", con).rename(columns={'rsid': VCF_RSID,
                                                                                         'chrom': VCF_CHROM,
                                                                                         'pos': VCF_POS,
                                                                                         'p_value': P_VALUE_LABEL,
                                                                                         'minor': 'minor_allele',
                                                                                         'major': 'major_allele',
                                                                                         'se': 'stderr',
                                                                                         'r2': 'r2_gwas'}).sort_values('varid')
        con.close()

        import struct
        echo('Converting bytes to floats with struct')
        gwas_variants['stderr'] = gwas_variants['stderr'].apply(lambda x: struct.unpack('f', x)[0] if type(x) is bytes else x).astype(float)
        gwas_variants['beta'] = gwas_variants['beta'].apply(lambda x: struct.unpack('f', x)[0] if type(x) is bytes else x).astype(float)

    elif is_gcta:
        echo('Reading GCTA output')
        gwas_variants = pd.read_csv(fname,
                                    sep='\t',
                                    dtype={'CHR': str,
                                           'SNP': str,
                                           'POS': int,
                                           'A1': str,
                                           'A2': str,
                                           'N': int,
                                           'AF1': float,
                                           'BETA': float,
                                           'SE': float,
                                           'P': float}).rename(columns={'CHR': VCF_CHROM,
                                                                        'SNP': VCF_RSID,
                                                                        'POS': VCF_POS,
                                                                        'A1': 'minor_allele',
                                                                        'A2': 'major_allele',
                                                                        'BETA': 'beta',
                                                                        'SE': 'stderr',
                                                                        'P': P_VALUE_LABEL})

        gwas_variants['varid1'] = gwas_variants[VCF_CHROM] + ':' + gwas_variants[VCF_POS].astype(str) + '_' + \
                                  gwas_variants['major_allele'] + '_' + gwas_variants['minor_allele']

        gwas_variants['varid2'] = gwas_variants[VCF_CHROM] + ':' + gwas_variants[VCF_POS].astype(str) + '_' + \
                                  gwas_variants['minor_allele'] + '_' + gwas_variants['major_allele']

    else:

        gwas_variants = pd.read_csv(fname,
                                header=None,
                                names=['varid', VCF_RSID, VCF_CHROM, VCF_POS, 'minor_allele', 'major_allele', 'beta',
                                       'stderr', 'r2_gwas', P_VALUE_LABEL],
                                sep='\t',
                                dtype={'varid': str, VCF_RSID: str, VCF_CHROM: str, VCF_POS: int, 'minor_allele': str,
                                       'major_allele': str, 'beta': float, 'stderr': float, 'r2_gwas': float,
                                       P_VALUE_LABEL: float}).sort_values('varid')

    echo('variants:', len(gwas_variants))

    gwas_variants[VCF_CHROM] = gwas_variants[VCF_CHROM].astype(str).str.replace('^0', '', regex=True)

    gwas_variants['variant'] = gwas_variants.apply(
        lambda x: ':'.join(map(str, [x['CHROM'], x['pos'], x['major_allele'], x['minor_allele']])), axis=1)

    if ukb_variants is None:
        echo('Reading ukb_variants info')
        ukb_variants = pd.read_pickle(UKB_DATA_PATH + '/variants.pickle')

    if 'varid' not in gwas_variants:
        ukb_variants['major_allele'] = np.where(ukb_variants[VCF_ALT] == ukb_variants['minor_allele'],
                                                ukb_variants[VCF_REF],
                                                ukb_variants[VCF_ALT])
        gwas_variants = pd.merge(ukb_variants,
                                 gwas_variants,
                                 on=[VCF_CHROM, VCF_POS, 'major_allele', 'minor_allele'],
                                 suffixes=('', '_ukb'))

    else:
        echo('Merging with ukb_variants')
        gwas_variants = pd.merge(gwas_variants, ukb_variants, on='varid', suffixes=('', '_ukb'))

    gwas_variants['beta'] = np.where(gwas_variants[VCF_ALT] == gwas_variants['minor_allele'],
                                     gwas_variants['beta'],
                                     -gwas_variants['beta'])

    gwas_variants['odds_ratio'] = np.exp(gwas_variants['beta'])

    gwas_variants = gwas_variants[
        [c for c in list(gwas_variants) if not c.endswith('_ukb') and c not in ['minor_allele', 'major_allele']]].copy()

    # gwas_variants = gwas_variants.rename(columns={'minor_allele': VCF_ALT, 'major_allele':VCF_REF})
    if replace_zero_pvalues:
        min_p = np.min(gwas_variants[gwas_variants[P_VALUE_LABEL] > 0][P_VALUE_LABEL])
        echo('min p-value=', min_p)
        gwas_variants[P_VALUE_LABEL] = np.where(gwas_variants[P_VALUE_LABEL] == 0, min_p / 10, gwas_variants[P_VALUE_LABEL])

    gwas_variants['zscore'] = gwas_variants['beta'] / gwas_variants['stderr']
    gwas_variants['wald_stat'] = np.square(gwas_variants['zscore'])

    gwas_variants[VARID_REF_ALT] = gwas_variants[VCF_CHROM] + ':' + gwas_variants[VCF_POS].astype(str) + ':' + \
                                   gwas_variants[VCF_REF] + ':' + gwas_variants[VCF_ALT]

    if remove_variants_without_rsid:
        echo('Removing variants without rsids')
        gwas_variants = gwas_variants[gwas_variants['all_RSIDs'].str.startswith('rs')]

    echo('n variants=', len(gwas_variants))

    return gwas_variants

get_snps_from_Jeremy_UKB_GWAS = get_GWAS_summary_stats

def get_variant_info_from_ensembl(variant_table, genome='hg19', verbose=True):

    rsids = sorted(set(variant_table[VCF_RSID]))

    rsid_chrom = dict((k, v) for k, v in zip(variant_table[VCF_RSID], variant_table[VCF_CHROM]))

    MAX_N_RSIDS = 150

    variant_info = None

    for batch_idx in range(0, len(rsids), MAX_N_RSIDS):
        echo('batch no rsids=', batch_idx)
        batch = rsids[batch_idx: batch_idx + MAX_N_RSIDS]

        if genome == 'hg19':
            server = "https://grch37.rest.ensembl.org"
        elif genome == 'hg38':
            server = "https://rest.ensembl.org"
        else:
            raise Exception('Unknown genome=' + str(genome))

        ext = "/variation/homo_sapiens"

        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        if ENSEMBL in LAST_REQUEST_TIME:
            seconds_since_last_request = (
                    datetime.datetime.now() - LAST_REQUEST_TIME[ENSEMBL]).total_seconds()
            if verbose:
                echo('Seconds since last request to ensembl:', seconds_since_last_request)

            if seconds_since_last_request < 5:
                if verbose:
                    echo('Sleeping for 5 second to prevent overloading server')
                time.sleep(5)

        r = requests.post(server + ext, headers=headers,
                          data='{ "ids" : [' + ', '.join('"%s"' % r for r in batch) + '] }')

        LAST_REQUEST_TIME[ENSEMBL] = datetime.datetime.now()

        to_replace = []

        if not r.ok:
            request_success = False
            if verbose:
                echo('ERROR:', r.text)
            for i in range(10):
                if verbose:
                    echo('Retrying request after sleeping for 2 minutes. Attempt:', i)
                time.sleep(120)

                r = requests.post(server + ext,
                                  headers=headers,
                                  data='{ "ids" : [' + ', '.join('"%s"' % r for r in batch) + '] }')
                if r.ok:
                    request_success = True
                    break

            if not request_success:
                continue
        else:
            request_success = True

        if request_success:
            response = r.json()
            for rsid in response:
                c_chrom = None

                rsids_to_try = [rsid] + response[rsid]['synonyms']
                syn_rsid = rsid
                for _r in rsids_to_try:
                    c_chrom = rsid_chrom.get(_r, None)
                    if c_chrom is not None:
                        syn_rsid = _r
                        break

                if syn_rsid != rsid:
                    to_replace.append((rsid, syn_rsid))

                response[rsid]['all_RSIDs'] = ','.join(sorted(set([r for r in rsids_to_try if r.startswith('rs')])))
                response[rsid][VCF_CHROM] = None
                response[rsid][VCF_POS] = None
                response[rsid][VCF_REF] = None
                response[rsid][VCF_ALT] = None
                response[rsid][ANCESTRAL_ALLELE] = None

                for m in response[rsid]['mappings']:
                    if m['seq_region_name'] == c_chrom:

                        response[rsid][VCF_CHROM] = c_chrom
                        response[rsid][VCF_POS] = m['start']
                        ancestral_allele = m.get('ancestral_allele', None)

                        alleles = m['allele_string'].split('/')

                        if m['strand'] != 1:
                            if verbose:
                                echo('minus strand:', rsid, syn_rsid)
                            alleles = [REVCOMP[a.upper()] for a in alleles]

                            if ancestral_allele is not None:
                                ancestral_allele = REVCOMP[ancestral_allele.upper()]

                        if len(alleles) < 2:
                            if verbose:
                                echo('[ERROR] allele array length <= 2:', rsid, alleles, response[rsid])
                            continue

                        if len(alleles) > 2:
                            multi_allelic_variants_fname = ROOT_PFIZIEV_PATH + f'/dbsnp/{genome}/multi_allelic_variants.pickle'
                            if get_variant_info_from_ensembl.multi_allelic_variants is None:
                                if verbose:
                                    echo('Loading:', multi_allelic_variants_fname)
                                get_variant_info_from_ensembl.multi_allelic_variants = pd.read_pickle(multi_allelic_variants_fname)

                            mav_df = get_variant_info_from_ensembl.multi_allelic_variants
                            var_info = None
                            for var_rsid in response[rsid]['all_RSIDs'].split(','):
                                var_info = mav_df[mav_df[VCF_RSID] == var_rsid]
                                if len(var_info) >= 1:
                                    break

                            if var_info is None or len(var_info) == 0:
                                if verbose:
                                    echo(f'[ERROR] multi-allelic variant not found in {multi_allelic_variants_fname} :', rsid, response[rsid])
                                continue

                            ref = var_info.iloc[0][VCF_REF]
                            alt = var_info.iloc[0][VCF_ALT]

                        else:
                            ref = alleles[0]
                            alt = alleles[1]

                        response[rsid][VCF_REF] = ref
                        response[rsid][VCF_ALT] = alt
                        response[rsid][ANCESTRAL_ALLELE] = ancestral_allele

        for from_rsid, to_rsid in to_replace:
            response[to_rsid] = response[from_rsid]
            del response[from_rsid]

        if variant_info is None:
            variant_info = pd.DataFrame.from_dict(response, orient='index')

        else:

            variant_info = pd.concat([variant_info, pd.DataFrame.from_dict(response, orient='index')], sort=True)

    if variant_info is not None and len(variant_info) > 0:
        variant_info = variant_info.reset_index().rename(columns={'index': VCF_RSID})[
            [VCF_RSID, VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT, ANCESTRAL_ALLELE, 'all_RSIDs']]

        variant_info = variant_info[~variant_info[VCF_REF].isnull()].copy()

    else:
        variant_info = pd.DataFrame({VCF_RSID: [],
                                     VCF_CHROM: [],
                                     VCF_POS: [],
                                     VCF_REF: [],
                                     VCF_ALT: [],
                                     ANCESTRAL_ALLELE: [],
                                     'all_RSIDs': []})
    return variant_info

get_variant_info_from_ensembl.last_request_time = None
get_variant_info_from_ensembl.multi_allelic_variants = None

