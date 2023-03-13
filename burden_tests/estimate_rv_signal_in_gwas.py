import argparse

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

from jutils import *
from ukb_analysis import *


def get_gwas_genes_and_rv_results(nr_sign_phenotypes,
                                  gencode,
                                  all_genes_in_locus=False,
                                  MAX_DISTANCE_FROM_INDEX_VARIANT=1000000,
                                  MAX_N_GENES_IN_LOCUS=1000,
                                  N_SHUFF=1000,
                                  fm_fnames=None,
                                  rv_fnames=None,
                                  FINEMAPPED_DIR = '',
                                  RV_DIR = '',
                                  rv_suffix = f'.all_ethnicities.both.dynamic_med_corrected.IRNT.ukb200k_unrelated_all_ethnicities.maf_0.001.common_vars_regressed.all_vars_regressed.score_PAI3D_percentile.main_analysis.pickle',
                                  exclude_eqtls=True):

    echo('[get_gwas_genes_and_rv_results]')

    GWAS_THRESHOLD = 5e-8
    MIN_GWAS_AF = 0.01

    missing = []
    # rv_suffix = f'.all_ethnicities.both.dynamic_med_corrected.IRNT.ukb200k_unrelated_all_ethnicities.maf_{RV_MAF}.common_vars_regressed.primateAI_scores.main_analysis.pickle'

    all_genes_in_gwas_loci = None

    # for ph_name in nr_quant_phenotypes:
    for ph_name in nr_sign_phenotypes:

        echo(ph_name)
        if fm_fnames is not None:
            fm_fname = fm_fnames[ph_name]
        else:
            fm_fname = FINEMAPPED_DIR + f'/{ph_name}.finemapped.csv.gz'

        if not os.path.exists(fm_fname):
            missing.append(ph_name)
            echo('Skipping:', ph_name)
            continue

        _, finemapped_snps = read_finemapped_variants(fm_fname, GWAS_PVALUE_THRESHOLD=GWAS_THRESHOLD, min_AF=MIN_GWAS_AF, exclude_eqtls=exclude_eqtls)

        finemapped_snps['phenotype'] = ph_name
        finemapped_snps = finemapped_snps[finemapped_snps['index_variant'] == finemapped_snps['varid_ref_alt']].copy()

        echo(finemapped_snps.shape)

        if all_genes_in_locus:
            echo('Using all genes in each GWAS locus')

            res = pd.merge(finemapped_snps, gencode[[VCF_CHROM, GENE_NAME, 'tss_pos', 'tx_start', 'tx_end']],
                           on=VCF_CHROM, suffixes=['/fm', ''])
            res['distance_to_index_variant'] = np.abs(res['tss_pos'] - res[VCF_POS])
            res = res[res['distance_to_index_variant'] <= MAX_DISTANCE_FROM_INDEX_VARIANT].copy()

            def keep_n_genes(vars, MAX_N_GENES_IN_LOCUS):
                vars = vars.sort_values('distance_to_index_variant').head(MAX_N_GENES_IN_LOCUS)
                del vars[VARID_REF_ALT]
                return vars

            echo('Keeping up to:', MAX_N_GENES_IN_LOCUS, 'genes in locus')
            res = res.groupby(VARID_REF_ALT).apply(keep_n_genes, MAX_N_GENES_IN_LOCUS).reset_index()
            # display(res.head(1))
        else:
            res = gwas_genes_from_finemapped_df(finemapped_snps)
            res['phenotype'] = ph_name

        echo('Reading rare variants results for all ethnicities')

        if rv_fnames is not None:
            rv_fname = rv_fnames[ph_name]
        else:
            rv_fname = RV_DIR + f'/{ph_name}/{ph_name}' + rv_suffix

        if not os.path.exists(rv_fname):
            echo('RV missing:', ph_name)
            missing.append(ph_name)
            continue

        echo('Reading rv res:', rv_fname)

        rare_var_results = pd.read_pickle(rv_fname)

        metric_label = 'ALL/del/carrier/pvalue/fdr_corr'

        rare_var_results = rare_var_results.sort_values(metric_label)
        rare_var_results['del/rank'] = list(range(1, len(rare_var_results) + 1))
        rare_var_results['del/total'] = np.sum(~rare_var_results['ALL/del/carrier/pvalue/fdr_corr'].isnull())

        for vt in ['del', 'ptv']:
            rare_var_results[f'ALL/{vt}/carrier/pvalue/fdr_corr'] = rare_var_results[
                f'ALL/{vt}/carrier/pvalue/fdr_corr'].fillna(1)
            _, pvalues = statsmodels.stats.multitest.fdrcorrection(
                rare_var_results[f'ALL/{vt}/carrier/pvalue/fdr_corr'])
            rare_var_results[f'ALL/{vt}/carrier/pvalue/fdr_corr/global_fdr'] = pvalues

        rand_df = {GENE_NAME: list(res[GENE_NAME].unique())}
        n_gwas_genes = len(rand_df[GENE_NAME])

        values = list(rare_var_results[metric_label])
        values_global_fdr = list(rare_var_results[metric_label + '/global_fdr'])

        for r_idx in range(N_SHUFF):
            rand_df[metric_label + '/global_fdr' + '/random/' + str(r_idx + 1)] = random.sample(values_global_fdr,
                                                                                                n_gwas_genes)
            rand_df[metric_label + '/random/' + str(r_idx + 1)] = random.sample(values, n_gwas_genes)

        rand_df = pd.DataFrame(rand_df)

        rare_var_results = rare_var_results[[GENE_NAME, 'del/rank'] + [c for c in list(rare_var_results) if
                                                                       c not in [GENE_NAME, VCF_CHROM, 'del/rank']]]

        res = pd.merge(res.rename(columns=dict((k, k + '/gwas')
                                               for k in list(res) if k not in [GENE_NAME, 'phenotype'] and not k.endswith('/gwas'))),
                       rare_var_results,
                       on=GENE_NAME,
                       suffixes=['/gwas', '/rv'],
                       how='left' if all_genes_in_locus else 'inner')

        res = pd.merge(res, rand_df, on=GENE_NAME)

        echo('res:',
             len(res), len(res[GENE_NAME].unique()), n_gwas_genes,
             len(res.drop_duplicates(subset=['index_variant/gwas', GENE_NAME]))
             )

        if all_genes_in_gwas_loci is None:
            all_genes_in_gwas_loci = res
        else:
            all_genes_in_gwas_loci = pd.concat([all_genes_in_gwas_loci, res], ignore_index=True)

    all_genes_in_gwas_loci = all_genes_in_gwas_loci.sort_values(['phenotype', 'ALL/del/carrier/pvalue/fdr_corr'])

    def get_max_r2(fmv):
        r2s = []
        for v in re.split(r'[,;]', fmv):
            for t in v.split('|'):
                if t.startswith('r2='):
                    r2s.append(float(t[3:]))
        return max(r2s)

    all_genes_in_gwas_loci['max_r2_with_index_variant/gwas'] = all_genes_in_gwas_loci['finemapped_via/gwas'].apply(get_max_r2)

    echo(all_genes_in_gwas_loci.shape)
    echo('missing:', missing)

    return all_genes_in_gwas_loci


def compute_rv_vs_gwas_stats_per_assoc_type(gwas_genes_to_use_for_all_stats,
                                            assoc_type=None,
                                            genes_to_keep=None,
                                            gene_carrier_stats=None,
                                            debug=False,
                                            max_genes_per_locus=5,
                                            max_distance_from_index_variant=1000000,
                                            min_distance_of_next_gene=0,
                                            metric_label='ALL/del/carrier/pvalue/fdr_corr'):

    echo('[compute_rv_vs_gwas_stats_per_assoc_type] assoc_type:', assoc_type,
         ', genes_to_keep:', len(genes_to_keep) if genes_to_keep is not None else genes_to_keep,
         ', max_genes_per_locus:', max_genes_per_locus,
         ', debug:', debug)

    stats = {'pvalue/gwas/left': [],
             'pvalue/gwas/right': [],
             'pvalue/rv': [],
             'max_pvalue/rv': [],

             'phenotype': [],
             'rv_sign/genes': [],
             'rv_sign/loci': [],

             'exp_sign/genes': [],
             'exp_sign/loci': [],

             'total/genes': [],
             'total/loci': [],

             'frac/genes': [],
             'frac/loci': [],
             'exp/frac/genes': [],
             'exp/frac/loci': [],
             'real-exp/frac/genes': [],
             'real-exp/frac/loci': [],

             'enr/genes': [],
             'enr/loci': []
             }

    n_gwas_intervals = 3
    gwas_pvalue_edges = [1e-323, 1e-100, 1e-50, 1e-20, 5e-8]

    gwas_genes_to_use = gwas_genes_to_use_for_all_stats.copy()


    # gwas_genes_to_use = gwas_genes_to_use[gwas_genes_to_use['ALL/del/n_variants'] >= 10].copy()
    # gwas_genes_to_use = gwas_genes_to_use[gwas_genes_to_use['max_r2_with_index_variant/gwas'] > 0.9].copy()
    gwas_genes_to_use = gwas_genes_to_use[
        gwas_genes_to_use['index_variant/gwas'] == gwas_genes_to_use['varid_ref_alt/gwas']].copy()

    if assoc_type == 'coding':
        gwas_genes_to_use = gwas_genes_to_use[
            gwas_genes_to_use['assoc_type/gwas'].isin(['coding', 'coding;splicing'])].copy()

    if assoc_type == 'non_coding':
        gwas_genes_to_use = gwas_genes_to_use[gwas_genes_to_use['assoc_type/gwas'].str.contains('non_coding') |
                                              (gwas_genes_to_use['assoc_type/gwas'].isin(
                                                  ['splicing', 'splicing/ambiguous']))].copy()

        if 'distance_to_index_variant/gwas' in list(gwas_genes_to_use):


            # index_variants_to_remove = gwas_genes_to_use[gwas_genes_to_use['distance_to_index_variant/gwas'] < min_distance_of_next_gene].groupby('index_variant/gwas').size().reset_index()

            loci_to_remove = gwas_genes_to_use[(gwas_genes_to_use['distance_to_index_variant/gwas'] <= min_distance_of_next_gene)].copy()

            gwas_genes_to_use = gwas_genes_to_use[gwas_genes_to_use['distance_to_index_variant/gwas'] <= max_distance_from_index_variant].copy()

            echo('keeping', max_genes_per_locus, 'genes per locus, before:', len(gwas_genes_to_use))

            gwas_genes_to_use = gwas_genes_to_use.sort_values(['pvalue/gwas',
                                                               'index_variant/gwas',
                                                               'phenotype',
                                                               'distance_to_index_variant/gwas']).groupby(
                ['index_variant/gwas', 'phenotype']).head(max_genes_per_locus)
            echo('after:', len(gwas_genes_to_use))

            max_gene_distance_per_locus = gwas_genes_to_use.sort_values('distance_to_index_variant/gwas',
                                                                        ascending=False).drop_duplicates('index_variant/gwas')[['index_variant/gwas',
                                                                                                                                'distance_to_index_variant/gwas']]

            loci_to_remove = pd.merge(loci_to_remove, max_gene_distance_per_locus, on='index_variant/gwas', suffixes=['', '/farthest_gene_distance'])
            loci_to_remove = loci_to_remove[loci_to_remove['distance_to_index_variant/gwas'] > loci_to_remove['distance_to_index_variant/gwas/farthest_gene_distance']]

            echo('loci_to_remove:', len(set(loci_to_remove['index_variant/gwas'])))
            echo('Filtering based on maximum distance to index_variant:', min_distance_of_next_gene, ', before:', len(gwas_genes_to_use))

            gwas_genes_to_use = gwas_genes_to_use[~gwas_genes_to_use['index_variant/gwas'].isin(loci_to_remove['index_variant/gwas'])].copy()

            echo('after:', len(gwas_genes_to_use))



    else:
        gwas_genes_to_use = gwas_genes_to_use[~gwas_genes_to_use['assoc_type/gwas'].str.contains('ambiguous')].copy()

    if 'pLI' in list(gwas_genes_to_use):
        d = gwas_genes_to_use.sort_values(metric_label).drop_duplicates(GENE_NAME).dropna(subset=['pLI', metric_label])
        echo(metric_label, 'vs pLI :', scipy.stats.spearmanr(d[metric_label], d['pLI']), len(d))

    gwas_genes_to_use = gwas_genes_to_use.sort_values('pvalue/gwas').drop_duplicates(subset=[GENE_NAME])

    if 'pLI' in list(gwas_genes_to_use):
        d = gwas_genes_to_use.sort_values(metric_label).drop_duplicates(GENE_NAME).dropna(subset=['pLI', metric_label])
        echo(metric_label, 'vs pLI :', scipy.stats.spearmanr(d[metric_label], d['pLI']), len(d))

    if genes_to_keep is None:
        genes_to_keep = set(gwas_genes_to_use_for_all_stats[GENE_NAME])

    gwas_genes_to_use = gwas_genes_to_use[gwas_genes_to_use[GENE_NAME].isin(genes_to_keep)].copy()
    echo('after filtering by genes_to_keep:', len(gwas_genes_to_use))

    if gene_carrier_stats is not None:
        gwas_genes_to_use = remove_genes_with_too_few_variants(gene_carrier_stats, gwas_genes_to_use)


    most_filtered_max_gwas_pvalue = 1e-100
    most_filtered_gwas_interval = f'[1e-323, {most_filtered_max_gwas_pvalue}]'
    most_filtered = gwas_genes_to_use_for_all_stats[gwas_genes_to_use_for_all_stats[GENE_NAME].isin(genes_to_keep) &
                                                    (gwas_genes_to_use_for_all_stats['index_variant/gwas'] ==
                                                     gwas_genes_to_use_for_all_stats['varid_ref_alt/gwas']) &
                                                    # (gwas_genes_to_use_for_all_stats['ALL/del/n_variants'] >= 50) &
                                                    (gwas_genes_to_use_for_all_stats[
                                                         'max_r2_with_index_variant/gwas'] >= 0.9) &
                                                    (gwas_genes_to_use_for_all_stats[
                                                         'pvalue/gwas'] <= most_filtered_max_gwas_pvalue)
                                                    & (~gwas_genes_to_use_for_all_stats['assoc_type/gwas'].str.contains('ambiguous'))
                                                    ].copy()
    if gene_carrier_stats is not None:
        most_filtered = remove_genes_with_too_few_variants(gene_carrier_stats, most_filtered)

    most_filtered_coding = most_filtered[(most_filtered['assoc_type/gwas'] == 'coding')].copy()
    most_filtered_coding = most_filtered_coding.sort_values('pvalue/gwas').drop_duplicates(subset=[GENE_NAME])

    most_filtered_non_coding = most_filtered[(most_filtered['assoc_type/gwas'] == 'non_coding') |
                                             (most_filtered['assoc_type/gwas'] == 'splicing')].copy()

    most_filtered_non_coding = most_filtered_non_coding.sort_values('pvalue/gwas').drop_duplicates(subset=[GENE_NAME])

    most_filtered_stats_coding = dict((k, []) for k in stats)
    most_filtered_stats_non_coding = dict((k, []) for k in stats)

    debug_info = []

    for gwas_bin_idx in range(len(gwas_pvalue_edges) - 1):

        left_gwas_edge = gwas_pvalue_edges[gwas_bin_idx]
        right_gwas_edge = gwas_pvalue_edges[gwas_bin_idx + 1]

        bin_gwas_genes_to_use = gwas_genes_to_use[(gwas_genes_to_use['pvalue/gwas'] > left_gwas_edge) &
                                                  (gwas_genes_to_use['pvalue/gwas'] <= right_gwas_edge)]

        for ph_name in sorted(bin_gwas_genes_to_use['phenotype'].unique()):
            #         metric_label = 'ALL/del/carrier/pvalue/fdr_corr/global_fdr'

            for rv_pval in [0.05]:  # , 0.01, 0.001, 0.0001, 0.00001, 0.000001]:

                def get_gwas_rv_phen_stats(left_gwas_edge, right_gwas_edge, bin_gwas_genes_to_use, rv_pval, ph_name,
                                           _stats, debug_label=None):


                    ph_all_gwas_genes = bin_gwas_genes_to_use[bin_gwas_genes_to_use['phenotype'] == ph_name].dropna(
                        subset=[metric_label]).copy()

                    if len(ph_all_gwas_genes) == 0:
                        return

                    n_ph_all_loci = len(ph_all_gwas_genes['index_variant/gwas'].unique())

                    ph_sign_genes = ph_all_gwas_genes[ph_all_gwas_genes[metric_label] <= rv_pval]

                    max_rv_pval = np.max(ph_sign_genes[metric_label])

                    ph_sign_loci = len(ph_sign_genes['index_variant/gwas'].unique())

                    d1 = ph_all_gwas_genes.copy()
                    d1['debug_label'] = debug_label
                    d1['pvalue/rv'] = rv_pval
                    d1['gwas_pvalue_bin_label'] = '[' + str(left_gwas_edge) + ', ' + str(right_gwas_edge) + ']'
                    d1['rv/significant'] = (d1[metric_label] <= rv_pval).astype(int)
                    d1['phenotype'] = ph_name

                    debug_info.append(d1)

                    ph_sign_genes_rand = []
                    ph_sign_loci_rand = []

                    # echo(N_SHUFF)
                    for r_idx in range(1, 1000):
                        _r = ph_all_gwas_genes[ph_all_gwas_genes[metric_label + '/random/' + str(r_idx)] <= rv_pval]
                        ph_sign_genes_rand.append(len(_r))
                        ph_sign_loci_rand.append(len(_r['index_variant/gwas'].unique()))

                    n_ph_all_gwas_genes = len(ph_all_gwas_genes)

                    frac_genes = len(ph_sign_genes) / n_ph_all_gwas_genes
                    frac_loci = ph_sign_loci / n_ph_all_loci

                    exp_frac_genes = np.mean(ph_sign_genes_rand) / n_ph_all_gwas_genes
                    exp_frac_loci = np.mean(ph_sign_loci_rand) / n_ph_all_loci

                    for (k, v) in [('pvalue/gwas/left', left_gwas_edge),
                                   ('pvalue/gwas/right', right_gwas_edge),
                                   ('pvalue/rv', rv_pval),

                                   ('max_pvalue/rv', max_rv_pval),
                                   ('phenotype', ph_name),

                                   ('rv_sign/genes', len(ph_sign_genes)),
                                   ('rv_sign/loci', ph_sign_loci),

                                   ('exp_sign/genes', np.mean(ph_sign_genes_rand)),
                                   ('exp_sign/loci', np.mean(ph_sign_loci_rand)),

                                   ('total/genes', n_ph_all_gwas_genes),
                                   ('total/loci', n_ph_all_loci),

                                   ('frac/genes', frac_genes),
                                   ('frac/loci', frac_loci),

                                   ('exp/frac/genes', exp_frac_genes),
                                   ('exp/frac/loci', exp_frac_loci),

                                   ('real-exp/frac/genes', frac_genes - exp_frac_genes),
                                   ('real-exp/frac/loci', frac_loci - exp_frac_loci),

                                   ('enr/genes', frac_genes / exp_frac_genes),
                                   ('enr/loci', frac_loci / exp_frac_loci)]:

                        _stats[k].append(v)

                get_gwas_rv_phen_stats(left_gwas_edge, right_gwas_edge, bin_gwas_genes_to_use, rv_pval, ph_name, stats,
                                       debug_label='all')

                if right_gwas_edge == most_filtered_max_gwas_pvalue:
                    get_gwas_rv_phen_stats(left_gwas_edge, right_gwas_edge, most_filtered_coding, rv_pval, ph_name,
                                           most_filtered_stats_coding, debug_label='most_filtered_coding')
                    get_gwas_rv_phen_stats(left_gwas_edge, right_gwas_edge, most_filtered_non_coding, rv_pval, ph_name,
                                           most_filtered_stats_non_coding, debug_label='most_filtered_non_coding')

    stats = pd.DataFrame(stats)
    stats['pvalue/gwas/interval'] = '[' + stats['pvalue/gwas/left'].astype(str) + ', ' + stats[
        'pvalue/gwas/right'].astype(str) + ']'
    most_filtered_stats_coding = pd.DataFrame(most_filtered_stats_coding)
    most_filtered_stats_coding['pvalue/gwas/interval'] = most_filtered_gwas_interval

    most_filtered_stats_non_coding = pd.DataFrame(most_filtered_stats_non_coding)
    most_filtered_stats_non_coding['pvalue/gwas/interval'] = most_filtered_gwas_interval

    debug_info = pd.concat(debug_info, ignore_index=True)

    if debug:
        return stats, most_filtered_stats_coding, most_filtered_stats_non_coding, n_gwas_intervals, debug_info
    else:
        return stats, most_filtered_stats_coding, most_filtered_stats_non_coding, n_gwas_intervals


def remove_genes_with_too_few_variants(gene_carrier_stats, gwas_genes_to_use):

    res = None
    echo('filtering by gene_carrier_stats, before:', len(gwas_genes_to_use))
    for ph_name in gene_carrier_stats['phenotype'].unique():

        genes_to_exclude = set(gene_carrier_stats[gene_carrier_stats['phenotype'] == ph_name][GENE_NAME])

        gv = gwas_genes_to_use[(gwas_genes_to_use['phenotype'] == ph_name) &
                               (~gwas_genes_to_use[GENE_NAME].isin(genes_to_exclude))].copy()

        if res is None:
            res = gv
        else:
            res = pd.concat([res, gv], ignore_index=True)

    echo('filtering by gene_carrier_stats, after:', len(res))

    return res


def plot_rv_vs_gwas_stats(stats,
                          most_filtered_stats_coding=None,
                          most_filtered_stats_non_coding=None,
                          most_filtered_gwas_interval=3,
                          rv_pvalue=0.05,
                          figsize=(10, 6),
                          n_gwas_intervals=None,
                          out_prefix=None,
                          store_data=False):

    echo('[plot_rv_vs_gwas_stats]')

    if type(stats) is str:
        echo('Reading stats from:', stats)
        stats = pd.read_csv(stats, sep='\t')

    if type(most_filtered_stats_coding) is str:
        echo('Reading stats from:', most_filtered_stats_coding)
        most_filtered_stats_coding = pd.read_csv(most_filtered_stats_coding, sep='\t')

    if type(most_filtered_stats_non_coding) is str:
        echo('Reading stats from:', most_filtered_stats_non_coding)
        most_filtered_stats_non_coding = pd.read_csv(most_filtered_stats_non_coding, sep='\t')

    to_plot = {'pvalue/gwas/interval': [],
               'frac': [],
               'frac_corr': [],
               'n': [],
               'pvalue/rv': [],
               'max_pvalue/rv': [],
               'x': []}

    plt.figure(figsize=figsize)

    for xx, gwas_int in enumerate(stats["pvalue/gwas/interval"].unique()):
        to_plot['pvalue/gwas/interval'].append(gwas_int)

        _d = stats[(stats["pvalue/gwas/interval"] == gwas_int) & (stats["pvalue/rv"] == rv_pvalue)]

        frac_corr = (np.sum(_d['rv_sign/loci']) - np.sum(_d['exp/frac/loci'])) / np.sum(_d['total/loci'])
        frac = np.sum(_d['rv_sign/loci']) / np.sum(_d['total/loci'])

        to_plot['n'].append(np.sum(_d['total/loci']))
        to_plot['pvalue/rv'].append(rv_pvalue)
        to_plot['max_pvalue/rv'].append(np.max(_d['max_pvalue/rv']) if len(_d) > 0 else 1)
        to_plot['frac'].append(frac)
        to_plot['frac_corr'].append(frac_corr)
        to_plot['x'].append(xx)

    echo(rv_pvalue, to_plot['pvalue/gwas/interval'][-n_gwas_intervals:], to_plot['n'][-3:],
         to_plot['frac_corr'][-3:])

        # sns.scatterplot(x, to_plot['frac'], marker='o', label='frac', size=to_plot['n'])

    if most_filtered_stats_coding is not None:
        best_stats = most_filtered_stats_coding[
            (most_filtered_stats_coding['pvalue/gwas/interval'] == most_filtered_gwas_interval) & (
                        most_filtered_stats_coding['pvalue/rv'] == rv_pvalue)]
        best_stats_n_total = np.sum(best_stats['total/loci'])

        best_frac_corr = (np.sum(best_stats['rv_sign/loci']) - np.sum(best_stats['exp/frac/loci'])) / best_stats_n_total
        best_frac = np.sum(best_stats['rv_sign/loci']) / best_stats_n_total
        echo('best_frac_corr:', best_frac_corr, best_frac)
        plt.scatter(x=[0], y=[best_frac], s=10 * np.sqrt(best_stats_n_total), marker='*',
                    label='High confidence, coding, n= ' + str(best_stats_n_total))

    if most_filtered_stats_non_coding is not None:
        best_stats = most_filtered_stats_non_coding[
            (most_filtered_stats_non_coding['pvalue/gwas/interval'] == most_filtered_gwas_interval) & (
                        most_filtered_stats_non_coding['pvalue/rv'] == rv_pvalue)]
        best_stats_n_total = np.sum(best_stats['total/loci'])

        best_frac_corr = (np.sum(best_stats['rv_sign/loci']) - np.sum(best_stats['exp/frac/loci'])) / best_stats_n_total
        best_frac = np.sum(best_stats['rv_sign/loci']) / best_stats_n_total
        echo('best_frac_corr:', best_frac_corr, best_frac)
        plt.scatter(x=[0], y=[best_frac], s=10 * np.sqrt(best_stats_n_total), marker='o',
                    label='High confidence, non-coding, n= ' + str(best_stats_n_total))

    for x, f, s in zip(to_plot['x'], to_plot['frac'], to_plot['n']):
        plt.scatter(x=x, y=f, s=10 * np.sqrt(s), marker=',', label='n= ' + str(s))

    to_plot = pd.DataFrame(to_plot)
    to_plot['sign'] = to_plot['n'] * to_plot['frac']

    display(to_plot)

    n_gwas_intervals = len(stats["pvalue/gwas/interval"].unique())
    plt.plot([-.5, n_gwas_intervals + .5 - 1], [rv_pvalue, rv_pvalue], color='black', ls='--')
    plt.xlim([-.5, n_gwas_intervals + .5 - 1])

    plt.xticks(to_plot['x'], to_plot['pvalue/gwas/interval'][:n_gwas_intervals], rotation='horizontal')
    plt.xlabel('GWAS P-value Interval')
    plt.ylabel('Fraction of GWAS Loci')
    plt.ylim(0)
    plt.legend(markerscale=0.5)

    if out_prefix is not None:
        echo('Saving figures to:', out_prefix)
        plt.savefig(out_prefix + '.svg')
        plt.savefig(out_prefix + '.png', dpi=300)

        if store_data:
            echo('Saving data to:', out_prefix)
            stats.to_csv(out_prefix + '.stats.csv.gz', sep='\t', index=False)
            most_filtered_stats_coding.to_csv(out_prefix + '.most_filtered_stats_coding.csv.gz', sep='\t', index=False)
            most_filtered_stats_non_coding.to_csv(out_prefix + '.most_filtered_stats_non_coding.csv.gz', sep='\t', index=False)

    plt.show()


def plot_multiple_rv_vs_gwas_stats(all_stats,
                                   all_most_filtered_stats_coding=None,
                                   all_most_filtered_stats_non_coding=None,
                                   stat_labels=None,
                                   stat_colors=None,
                                   all_most_filtered_gwas_interval=None,
                                   extra_most_filtered_stats_coding=None,
                                   rv_pvalue=0.05,
                                   figsize=(10, 6),
                                   all_n_gwas_intervals=None,
                                   out_prefix=None,
                                   store_data=False,
                                   upper_ylim=None):

    echo('[plot_multiple_rv_vs_gwas_stats]!')

    plt.figure(figsize=figsize)

    most_filtered_gwas_interval = None
    for stats, most_filtered_stats_coding, most_filtered_stats_non_coding, n_gwas_intervals, most_filtered_gwas_interval, stat_label, stat_color in zip(
        all_stats,
        all_most_filtered_stats_coding,
        all_most_filtered_stats_non_coding,
        all_n_gwas_intervals,
        all_most_filtered_gwas_interval,
        stat_labels,
        stat_colors):

        if type(stats) is str:
            echo('Reading stats from:', stats)
            stats = pd.read_csv(stats, sep='\t')

        if type(most_filtered_stats_coding) is str:
            echo('Reading stats from:', most_filtered_stats_coding)
            most_filtered_stats_coding = pd.read_csv(most_filtered_stats_coding, sep='\t')

        if type(most_filtered_stats_non_coding) is str:
            echo('Reading stats from:', most_filtered_stats_non_coding)
            most_filtered_stats_non_coding = pd.read_csv(most_filtered_stats_non_coding, sep='\t')

        to_plot = {'pvalue/gwas/interval': [],
                   'frac': [],
                   'frac_corr': [],
                   'n': [],
                   'pvalue/rv': [],
                   'max_pvalue/rv': [],
                   'x': []}

        for xx, gwas_int in enumerate(stats["pvalue/gwas/interval"].unique()):
            to_plot['pvalue/gwas/interval'].append(gwas_int)

            _d = stats[(stats["pvalue/gwas/interval"] == gwas_int) & (stats["pvalue/rv"] == rv_pvalue)]

            frac_corr = (np.sum(_d['rv_sign/loci']) - np.sum(_d['exp/frac/loci'])) / np.sum(_d['total/loci'])
            frac = 100 * np.sum(_d['rv_sign/loci']) / np.sum(_d['total/loci'])

            to_plot['n'].append(np.sum(_d['total/loci']))
            to_plot['pvalue/rv'].append(rv_pvalue)
            to_plot['max_pvalue/rv'].append(np.max(_d['max_pvalue/rv']) if len(_d) > 0 else 1)
            to_plot['frac'].append(frac)
            to_plot['frac_corr'].append(frac_corr)
            to_plot['x'].append(xx)

        echo(rv_pvalue, to_plot['pvalue/gwas/interval'][-n_gwas_intervals:], to_plot['n'][-3:],
             to_plot['frac_corr'][-3:])

            # sns.scatterplot(x, to_plot['frac'], marker='o', label='frac', size=to_plot['n'])

        # if most_filtered_stats_coding is not None:
        #     best_stats = most_filtered_stats_coding[
        #         (most_filtered_stats_coding['pvalue/gwas/interval'] == most_filtered_gwas_interval) & (
        #                     most_filtered_stats_coding['pvalue/rv'] == rv_pvalue)]
        #     best_stats_n_total = np.sum(best_stats['total/loci'])
        #
        #     best_frac_corr = (np.sum(best_stats['rv_sign/loci']) - np.sum(best_stats['exp/frac/loci'])) / best_stats_n_total
        #     best_frac = np.sum(best_stats['rv_sign/loci']) / best_stats_n_total
        #     echo('best_frac_corr:', best_frac_corr, best_frac)
        #     plt.scatter(x=[0], y=[best_frac], s=10 * np.sqrt(best_stats_n_total), marker='*',
        #                 label='High confidence, coding, n= ' + str(best_stats_n_total))
        #
        # if most_filtered_stats_non_coding is not None:
        #     best_stats = most_filtered_stats_non_coding[
        #         (most_filtered_stats_non_coding['pvalue/gwas/interval'] == most_filtered_gwas_interval) & (
        #                     most_filtered_stats_non_coding['pvalue/rv'] == rv_pvalue)]
        #     best_stats_n_total = np.sum(best_stats['total/loci'])
        #
        #     best_frac_corr = (np.sum(best_stats['rv_sign/loci']) - np.sum(best_stats['exp/frac/loci'])) / best_stats_n_total
        #     best_frac = np.sum(best_stats['rv_sign/loci']) / best_stats_n_total
        #     echo('best_frac_corr:', best_frac_corr, best_frac)
        #     plt.scatter(x=[0], y=[best_frac], s=10 * np.sqrt(best_stats_n_total), marker='o',
        #                 label='High confidence, non-coding, n= ' + str(best_stats_n_total))

        plt.plot(to_plot['x'], to_plot['frac'], '-', label=stat_label, color=stat_color)

        for x, f, s in zip(to_plot['x'], to_plot['frac'], to_plot['n']):
            plt.scatter(x=x, y=f, s=50 * np.log10(s), marker=',', color=stat_color)#, label='n= ' + str(s))

        to_plot = pd.DataFrame(to_plot)
        to_plot['sign'] = to_plot['n'] * to_plot['frac']

        n_gwas_intervals = len(stats["pvalue/gwas/interval"].unique())

        plt.xticks(to_plot['x'], to_plot['pvalue/gwas/interval'][:n_gwas_intervals], rotation='horizontal')
        plt.plot([-.5, n_gwas_intervals + .5 - 1], [100 * rv_pvalue, 100 * rv_pvalue], color='black', ls='--')
        plt.xlim([-.5, n_gwas_intervals + .5 - 1])

    if extra_most_filtered_stats_coding is not None:
        best_stats = extra_most_filtered_stats_coding[
            (extra_most_filtered_stats_coding['pvalue/gwas/interval'] == most_filtered_gwas_interval) & (
                        extra_most_filtered_stats_coding['pvalue/rv'] == rv_pvalue)]
        best_stats_n_total = np.sum(best_stats['total/loci'])

        best_frac_corr = (np.sum(best_stats['rv_sign/loci']) - np.sum(best_stats['exp/frac/loci'])) / best_stats_n_total
        best_frac = 100 * np.sum(best_stats['rv_sign/loci']) / best_stats_n_total
        echo('best_frac_corr:', best_frac_corr, best_frac)
        plt.scatter(x=[0], y=[best_frac], s=50 * np.log10(best_stats_n_total), marker='s',
                    label='High confidence coding hits, n= ' + '{:,}'.format(best_stats_n_total),
                    color='black')

    plt.xlabel('GWAS P-value Interval')
    plt.ylabel('% of GWAS Hits with Rare Variant Signal')
    plt.ylim(0)

    if upper_ylim is not None:
        plt.ylim((0, upper_ylim))

    plt.legend(markerscale=0.5)

    if out_prefix is not None:
        echo('Saving figures to:', out_prefix)
        plt.savefig(out_prefix + '.svg')
        plt.savefig(out_prefix + '.png', dpi=300)

        # if store_data:
        #     echo('Saving data to:', out_prefix)
        #     stats.to_csv(out_prefix + '.stats.csv.gz', sep='\t', index=False)
        #     most_filtered_stats_coding.to_csv(out_prefix + '.most_filtered_stats_coding.csv.gz', sep='\t', index=False)
        #     most_filtered_stats_non_coding.to_csv(out_prefix + '.most_filtered_stats_non_coding.csv.gz', sep='\t', index=False)

    plt.show()



def estimate_rv_signal_in_gwas(ph_name='Cholesterol',
                               N_SHUFF=1000,
                               N_MAX_GENES=4,
                               MAX_GWAS_LOCI=None,
                               best_k_pvalue_cutoff=0.05,
                               out_fname=None,
                               GWAS_THRESHOLD=5e-8,
                               randomize_rv=False):

    echo('[estimate_rv_signal_in_gwas] ph_name=', ph_name, ', randomize_rv:', randomize_rv)

    def fm_split(r):
        return re.split(r'[|,;]', r)

    RV_DIR = ROOT_PATH + '/ukbiobank/data/molecular_phenotypes.17_SEPT_2019/phenotypes_for_GWAS.ukb200k/quantitative_phenotypes.results/'
    FINEMAPPED_DIR = ROOT_PATH + '/pfiziev/rare_variants/data/finemapping/finemapped_gwas/ukbiobank.v4/'

    finemapped_fname_template = '%s.finemapped.csv.gz'
    strip_colon_from_gene_names = False
    min_AF = 0.01

    rv_fname_suffix = '.all_ethnicities.both.dynamic_med_corrected.IRNT.ukb200k_unrelated_all_ethnicities.maf_0.001.common_vars_regressed.primateDL_score_3D_newAvg.main_analysis.pickle'

    finemapped_fname = FINEMAPPED_DIR + '/' + finemapped_fname_template % ph_name
    finemapped_snps = pd.read_csv(finemapped_fname, sep='\t', dtype={VCF_CHROM: str})
    if strip_colon_from_gene_names:
        finemapped_snps[GENE_NAME] = finemapped_snps[GENE_NAME].apply(
            lambda x: ','.join([gn.split(':')[0] for gn in fm_split(x)]))

    finemapped_snps = finemapped_snps[finemapped_snps['pvalue'] <= GWAS_THRESHOLD].copy()

    if VCF_AF in list(finemapped_snps):
        after = finemapped_snps[(finemapped_snps[VCF_AF] >= min_AF) &
                                (finemapped_snps[VCF_AF] <= 1 - min_AF)].copy()
        fmv_skipped = 0
        rows = []

        for _, r in after.iterrows():
            fmv = r['finemapped_via'].split(',')
            to_skip = False
            for _fmv in fmv:
                af = [k for k in _fmv.split('|') if k.startswith('AF=')]
                if len(af) > 0:
                    af = float(af[0][3:])
                    if af < min_AF or af > 1 - min_AF:
                        to_skip = True
                        fmv_skipped += 1
                        break

            if not to_skip:
                rows.append(r)

        after = pd.DataFrame(rows)
        echo(ph_name, ', Filtering variants below', min_AF, ', before:', len(finemapped_snps), ', after:', len(after),
             ', fmv_skipped:', fmv_skipped)
        finemapped_snps = after

    finemapped_snps['n_genes'] = finemapped_snps[GENE_NAME].apply(lambda x: len(set(fm_split(x))))

    gwas_genes = gwas_genes_from_finemapped_df(finemapped_snps)

    echo('gwas_genes:', gwas_genes.shape)

    rv_res = pd.read_pickle(RV_DIR + f'/{ph_name}/{ph_name}{rv_fname_suffix}')
    rv_res = rv_res.rename(columns={f'ALL/del/carrier/beta': f'ALL/del/beta',
                                    f'ALL/ptv/carrier/beta': f'ALL/ptv/beta'})
    if randomize_rv:
        values = list(rv_res['ALL/del/carrier/pvalue/fdr_corr'])
        np.random.shuffle(values)
        rv_res['ALL/del/carrier/pvalue/fdr_corr'] = values

    gwas_genes_rv = pd.merge(gwas_genes, rv_res, on=[GENE_NAME, VCF_CHROM])

    echo('gwas_genes_rv:', gwas_genes_rv.shape)

    k_genes_to_use = set(rv_res[GENE_NAME])

    # gwas_genes_to_use = gwas_genes[(gwas_genes['n_genes_near_index_variant/gwas'] <= N_MAX_GENES) &
    #                                (gwas_genes[GENE_NAME].isin(set(rv_res[GENE_NAME])))].copy()

    gwas_genes_to_use = gwas_genes[(gwas_genes['n_genes/gwas'] <= N_MAX_GENES) &
                                   (gwas_genes[GENE_NAME].isin(set(rv_res[GENE_NAME])))].copy()

    removed_genes = set(gwas_genes[gwas_genes['n_genes/gwas'] > N_MAX_GENES][GENE_NAME]) - set(gwas_genes_to_use[GENE_NAME])

    echo('removed_genes:', len(removed_genes))
    n_gwas_loci = len(gwas_genes_rv['index_variant/gwas'].unique())
    n_gwas_loci = n_gwas_loci if MAX_GWAS_LOCI is None else min(MAX_GWAS_LOCI, n_gwas_loci)
    res = {}
    original_best_idx = None

    echo('Testing ranking of', n_gwas_loci, 'gwas loci')

    best_k = 0
    update_best_k = True
    original_loci_ranks = None
    k = 0
    while True:

        k_genes_to_use = k_genes_to_use - removed_genes

        k_rv_res = rv_res[rv_res[GENE_NAME].isin(k_genes_to_use)].sort_values('ALL/del/carrier/pvalue/fdr_corr').copy()
        k_rv_res['del/rank'] = list(range(1, len(k_rv_res) + 1))

        k_gwas_genes_rv = pd.merge(gwas_genes_to_use[gwas_genes_to_use[GENE_NAME].isin(k_genes_to_use)],
                                   k_rv_res,
                                   on=[GENE_NAME, VCF_CHROM]).sort_values('del/rank')

        if len(k_gwas_genes_rv) == 0:
            break

        k_real_ranks = k_gwas_genes_rv.drop_duplicates('index_variant/gwas')['del/rank'].values
        if original_loci_ranks is None:
            original_loci_ranks = list(k_real_ranks)
            echo('original_loci_ranks:', original_loci_ranks)

        if original_best_idx is None:
            original_best_idx = {}
            for _, r in k_gwas_genes_rv.iterrows():
                original_best_idx[r[GENE_NAME]] = r['del/rank']

        k_best_idx = {}
        for _, r in k_gwas_genes_rv.iterrows():
            k_best_idx[r[GENE_NAME]] = r['del/rank']

        top_gwas_locus = k_gwas_genes_rv.iloc[0]['index_variant/gwas']

        k_genes = set(gwas_genes_to_use[gwas_genes_to_use['index_variant/gwas'] == top_gwas_locus][GENE_NAME])

        removed_genes |= k_genes
        removed_genes |= set(k_rv_res[k_rv_res['del/rank'] <= np.min(k_gwas_genes_rv['del/rank'])][GENE_NAME])

        k_n_total = len(k_rv_res)
        k_all_ranks = list(range(1, k_n_total + 1))

        p_values = np.array([0] * len(k_real_ranks))

        for r_no in range(N_SHUFF):

            k_gwas_genes_rv['random/rank'] = random.sample(k_all_ranks, len(k_gwas_genes_rv))

            k_shuffled = k_gwas_genes_rv.sort_values('random/rank').drop_duplicates('index_variant/gwas')[
                'random/rank'].values

            p_values += (k_real_ranks >= k_shuffled).astype(int)

        p_values = p_values / N_SHUFF
        uncorrected_pvalues = list(p_values)
        p_values = statsmodels.stats.multitest.fdrcorrection(p_values)[1]

        best_n = 0
        best_p = 1

        rv_rank = original_loci_ranks[k]
        best_uncorr_p = 1

        for i in range(len(p_values)):
            if p_values[i] <= best_p:
                best_p = p_values[i]
                best_n = i
                best_uncorr_p = uncorrected_pvalues[i]

        echo(k, len(p_values), len(k_rv_res), best_n, best_p, top_gwas_locus, [(g, original_best_idx[g], k_best_idx[g])
                                                                               for g in
                                                                               sorted(k_genes & set(k_best_idx),
                                                                                      key=lambda _g: k_best_idx[_g])])

        for (key, value) in [('k', k),
                             ('best_n', best_n),
                             ('best_p', best_p),
                             ('rv_rank', rv_rank),
                             ('best_uncorr_p', best_uncorr_p)]:

            if key not in res:
                res[key] = []

            res[key].append(value)

        if best_p > best_k_pvalue_cutoff:
            update_best_k = False

        if update_best_k:
            best_k = k + 1

        k += 1

    res = pd.DataFrame(res)

    res['phenotype'] = ph_name
    res['signal'] = best_k
    res['signal_frac'] = best_k / k

    if out_fname is not None:
        echo('Saving results to:', out_fname)
        res.to_csv(out_fname, index=False, sep='\t')

    return res


def estimate_rv_signal_in_gwas_as_function_of_gwas_pvalue(ph_name='Cholesterol', keep_genes=None):

    echo('[estimate_rv_signal_in_gwas] ph_name=', ph_name, ', keep_genes:', len(keep_genes))

    def fm_split(r):
        return re.split(r'[|,;]', r)

    RV_DIR = ROOT_PATH + '/ukbiobank/data/molecular_phenotypes.17_SEPT_2019/phenotypes_for_GWAS.ukb200k/quantitative_phenotypes.results/'
    FINEMAPPED_DIR = ROOT_PATH + '/pfiziev/rare_variants/data/finemapping/finemapped_gwas/ukbiobank.v4/'

    finemapped_fname_template = '%s.finemapped.csv.gz'
    strip_colon_from_gene_names = False
    min_AF = 0.01

    rv_fname_suffix = '.all_ethnicities.both.dynamic_med_corrected.IRNT.ukb200k_unrelated_all_ethnicities.maf_0.001.common_vars_regressed.primateDL_score_3D_newAvg.main_analysis.pickle'

    finemapped_fname = FINEMAPPED_DIR + '/' + finemapped_fname_template % ph_name
    finemapped_snps = pd.read_csv(finemapped_fname, sep='\t', dtype={VCF_CHROM: str})
    if strip_colon_from_gene_names:
        finemapped_snps[GENE_NAME] = finemapped_snps[GENE_NAME].apply(
            lambda x: ','.join([gn.split(':')[0] for gn in fm_split(x)]))

    finemapped_snps = finemapped_snps[finemapped_snps['pvalue'] <= 5e-8].copy()

    if VCF_AF in list(finemapped_snps):
        after = finemapped_snps[(finemapped_snps[VCF_AF] >= min_AF) &
                                (finemapped_snps[VCF_AF] <= 1 - min_AF)].copy()
        fmv_skipped = 0
        rows = []

        for _, r in after.iterrows():
            fmv = r['finemapped_via'].split(',')
            to_skip = False
            for _fmv in fmv:
                af = [k for k in _fmv.split('|') if k.startswith('AF=')]
                if len(af) > 0:
                    af = float(af[0][3:])
                    if af < min_AF or af > 1 - min_AF:
                        to_skip = True
                        fmv_skipped += 1
                        break

            if not to_skip:
                rows.append(r)

        after = pd.DataFrame(rows)
        echo(ph_name, ', Filtering variants below', min_AF, ', before:', len(finemapped_snps), ', after:', len(after),
             ', fmv_skipped:', fmv_skipped)
        finemapped_snps = after

    finemapped_snps['n_genes'] = finemapped_snps[GENE_NAME].apply(lambda x: len(set(fm_split(x))))

    gwas_genes = gwas_genes_from_finemapped_df(finemapped_snps)

    echo('gwas_genes:', gwas_genes.shape)

    rv_res = pd.read_pickle(RV_DIR + f'/{ph_name}/{ph_name}{rv_fname_suffix}')
    rv_res = rv_res.rename(columns={f'ALL/del/carrier/beta': f'ALL/del/beta',
                                    f'ALL/ptv/carrier/beta': f'ALL/ptv/beta'})

    if keep_genes is not None:
        rv_res = rv_res[rv_res[GENE_NAME].isin(keep_genes)].copy()

    rv_res = rv_res.sort_values('ALL/del/carrier/pvalue/fdr_corr').copy()
    rv_res['del/rank'] = range(1, len(rv_res) + 1)

    # genes_to_test = set(gwas_genes.sort_values('sum_log_p/gwas').drop_duplicates('index_variant/gwas')[GENE_NAME])
    to_plot = {'pvalue/gwas': [],
               'log_p/gwas': [],
               'frac': [],
               'signal': [], 'rv_rank': [], 'n': [], 'pvalue/ranks': []}

    for p in sorted(gwas_genes['pvalue/gwas'].unique()):
        genes_to_test = set(gwas_genes[gwas_genes['pvalue/gwas'] <= p].sort_values('sum_log_p/gwas').drop_duplicates(
            'index_variant/gwas')[GENE_NAME])

        if keep_genes is not None:
            genes_to_test = genes_to_test & keep_genes

        #     genes_to_test = set(gwas_genes[gwas_genes['pvalue/gwas'] <= p].sort_values('sum_log_p/gwas')[GENE_NAME])

        genes_to_test_ranks = rv_res[rv_res[GENE_NAME].isin(genes_to_test)]

        if len(genes_to_test_ranks) == 0:
            continue


        real_ranks = sorted(genes_to_test_ranks['del/rank'])

        best_n, best_p = test_ranks(real_ranks, len(rv_res))

        #     echo(p, best_p, best_n, best_n / len(genes_to_test_ranks), len(genes_to_test_ranks))
        to_plot['log_p/gwas'].append(np.log10(p))
        to_plot['pvalue/gwas'].append(p)
        to_plot['frac'].append(best_n / len(genes_to_test_ranks))
        to_plot['signal'].append(best_n)
        to_plot['rv_rank'].append(real_ranks[best_n - 1])
        to_plot['n'].append(len(genes_to_test_ranks))

        to_plot['pvalue/ranks'].append(best_p)

    echo(ph_name, scipy.stats.ranksums(to_plot['log_p/gwas'], to_plot['frac']))

    to_plot = pd.DataFrame(to_plot)
    to_plot['phenotype'] = ph_name

    return to_plot.sort_values('log_p/gwas')


def estimate_rv_signal_in_gwas_as_function_of_gwas_pvalue_and_rv_pvalue(ph_name='Cholesterol',
                                                                        keep_genes=None,
                                                                        rv_res=None,
                                                                        pvalue_label=None,
                                                                        finemapped_ambiguous=True,
                                                                        index_variants_only=False,
                                                                        rv_pvalues=None,
                                                                        n_randomizations=0,
                                                                        gwas_edge_step=5,
                                                                        gwas_pvalue_edges=None):


    echo('[estimate_rv_signal_in_gwas] ph_name=', ph_name,
         ', keep_genes:', len(keep_genes) if keep_genes is not None else 0,
         ', rv_pvalues:', rv_pvalues,
         ', n_randomizations:', n_randomizations)

    if pvalue_label is None:
        pvalue_label = 'ALL/del/carrier/pvalue/fdr_corr'


    def fm_split(r):
        return re.split(r'[|,;]', r)

    RV_DIR = ROOT_PATH + '/ukbiobank/data/molecular_phenotypes.17_SEPT_2019/phenotypes_for_GWAS.ukb200k/quantitative_phenotypes.results/'
    FINEMAPPED_DIR = ROOT_PATH + '/pfiziev/rare_variants/data/finemapping/finemapped_gwas/ukbiobank.v4/'

    finemapped_fname_template = '%s.finemapped.csv.gz'
    strip_colon_from_gene_names = False
    min_AF = 0.01

    rv_fname_suffix = '.all_ethnicities.both.dynamic_med_corrected.IRNT.ukb200k_unrelated_all_ethnicities.maf_0.001.common_vars_regressed.primateDL_score_3D_newAvg.main_analysis.pickle'

    finemapped_fname = FINEMAPPED_DIR + '/' + finemapped_fname_template % ph_name
    finemapped_snps = pd.read_csv(finemapped_fname, sep='\t', dtype={VCF_CHROM: str})
    if strip_colon_from_gene_names:
        finemapped_snps[GENE_NAME] = finemapped_snps[GENE_NAME].apply(
            lambda x: ','.join([gn.split(':')[0] for gn in fm_split(x)]))

    finemapped_snps = finemapped_snps[finemapped_snps['pvalue'] <= 5e-8].copy()

    if VCF_AF in list(finemapped_snps):
        after = finemapped_snps[(finemapped_snps[VCF_AF] >= min_AF) &
                                (finemapped_snps[VCF_AF] <= 1 - min_AF)].copy()
        fmv_skipped = 0
        rows = []

        for _, r in after.iterrows():
            fmv = r['finemapped_via'].split(',')
            to_skip = False
            for _fmv in fmv:
                af = [k for k in _fmv.split('|') if k.startswith('AF=')]
                if len(af) > 0:
                    af = float(af[0][3:])
                    if af < min_AF or af > 1 - min_AF:
                        to_skip = True
                        fmv_skipped += 1
                        break

            if not to_skip:
                rows.append(r)

        after = pd.DataFrame(rows)
        echo(ph_name, ', Filtering variants below', min_AF, ', before:', len(finemapped_snps), ', after:', len(after),
             ', fmv_skipped:', fmv_skipped)
        finemapped_snps = after

    finemapped_snps['n_genes'] = finemapped_snps[GENE_NAME].apply(lambda x: len(set(fm_split(x))))

    if index_variants_only:
        finemapped_snps = finemapped_snps[finemapped_snps['index_variant'] == finemapped_snps[VARID_REF_ALT]].copy()
    if not finemapped_ambiguous:
        finemapped_snps = finemapped_snps[~finemapped_snps['assoc_type'].str.contains('/ambiguous')].copy()

    gwas_genes = gwas_genes_from_finemapped_df(finemapped_snps)

    if keep_genes is not None:
        gwas_genes = gwas_genes[gwas_genes[GENE_NAME].isin(keep_genes)].copy()

    if rv_res is None:
        rv_res = pd.read_pickle(RV_DIR + f'/{ph_name}/{ph_name}{rv_fname_suffix}')
        rv_res = rv_res.rename(columns={f'ALL/del/carrier/beta': f'ALL/del/beta',
                                        f'ALL/ptv/carrier/beta': f'ALL/ptv/beta'})
    else:
        rv_res = rv_res.copy()

    if keep_genes is not None:
        rv_res = rv_res[rv_res[GENE_NAME].isin(keep_genes)].copy()

    gwas_genes = gwas_genes[gwas_genes[GENE_NAME].isin(set(rv_res[GENE_NAME]))].copy()
    echo('gwas_genes:', gwas_genes.shape)

    shuffled_idx = 0
    while True:

        rv_res = rv_res.sort_values(pvalue_label).copy()

        rv_res['del/rank'] = range(1, len(rv_res) + 1)

        genes_to_test = set(gwas_genes[gwas_genes['pvalue/gwas'] <= 5e-8].sort_values('sum_log_p/gwas').drop_duplicates(
            'index_variant/gwas')[GENE_NAME])

        if rv_pvalues is None:
            rv_pvalues = sorted(set(rv_res[rv_res[GENE_NAME].isin(genes_to_test)][pvalue_label]))

        gwas_pvalues = sorted(gwas_genes['pvalue/gwas'].unique())

        if gwas_pvalue_edges is None:
            gwas_pvalue_edges = [-i for i in list(range(5, 350, gwas_edge_step))]

        if shuffled_idx == 0:
            echo('rv_pvalues:', len(rv_pvalues), ', gwas_pvalues:', len(gwas_pvalues))

        to_plot = {'pvalue/gwas': [],
                   'pvalue/rv': [],
                   'frac': [],
                   'n': [],
                   'total': [],
                   'shuffled_idx': []}

        for gwas_bin_idx in range(len(gwas_pvalue_edges) - 1):

            left_p = 10 ** gwas_pvalue_edges[gwas_bin_idx]
            right_p = 10 ** gwas_pvalue_edges[gwas_bin_idx + 1]

            gwas_bin_center = (gwas_pvalue_edges[gwas_bin_idx] + gwas_pvalue_edges[gwas_bin_idx + 1]) / 2

            genes_to_test = set(gwas_genes[(gwas_genes['pvalue/gwas'] <= left_p) &
                                           (gwas_genes['pvalue/gwas'] >= right_p)].sort_values('sum_log_p/gwas').drop_duplicates('index_variant/gwas')[GENE_NAME])

            if keep_genes is not None:
                genes_to_test = genes_to_test & keep_genes

            if len(genes_to_test) == 0:
                continue

            rv_genes_to_test = rv_res[rv_res[GENE_NAME].isin(genes_to_test)]

            for rv_p in rv_pvalues:
                rv_genes = rv_genes_to_test[rv_genes_to_test[pvalue_label] <= rv_p]

                frac = len(rv_genes) / len(genes_to_test)

                for (k, v) in [('pvalue/gwas', gwas_bin_center),
                               ('pvalue/rv', rv_p),
                               ('frac', frac),
                               ('n', len(rv_genes)),
                               ('total', len(genes_to_test)),
                               ('shuffled_idx', shuffled_idx)]:
                    to_plot[k].append(v)

        shuffled_idx += 1

        if shuffled_idx > n_randomizations:
            break

        values = list(rv_res[pvalue_label])
        random.shuffle(values)
        rv_res[pvalue_label] = values

    to_plot = pd.DataFrame(to_plot)
    to_plot['phenotype'] = ph_name

    return to_plot


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p',
                        dest='ph_name',
                        help='phenotype name',
                        required=True)

    parser.add_argument('-o',
                        dest='out_fname',
                        help='phenotype name',
                        required=True)

    parser.add_argument('-g',
                        dest='gwas_threshold',
                        help='gwas_threshold [%(default)s]',
                        default=5e-8)

    parser.add_argument('-r',
                        dest='randomize_rv',
                        help='randomize',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    args = parser.parse_args()

    log_name = args.out_fname + '.log'
    open_log(log_name)

    ph_name = args.ph_name
    randomize_rv = args.randomize_rv

    out_fname = args.out_fname
    GWAS_THRESHOLD = args.gwas_threshold

    echo(ph_name, out_fname, GWAS_THRESHOLD)

    estimate_rv_signal_in_gwas(ph_name, N_SHUFF=10000, out_fname=out_fname, GWAS_THRESHOLD=GWAS_THRESHOLD, randomize_rv=randomize_rv)

    echo('Done')

    close_log()
