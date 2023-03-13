import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import statsmodels.api as sm
import seaborn as sns


# def fit_gmm(fgr_af, fgr_effect, bgr_af, bgr_effect):
#     C = ['bgr', 'gof', 'lof']
#
#     a = [0, 1, -1]
#     b = [0, 0, 0]
#
#     priors = [1 / 3, 1 / 3, 1 / 3]
#     stddevs = estimate_stddevs(bgr_af, bgr_effect)
#
#     def posteriors(af, effect, priors, a, b, stddevs):
#         post = []
#         for pr, aa, bb in zip(priors, a, b):
#             post.append(scipy.stats.norm.pdf(effect, loc=(aa * np.log10(af) + bb), scale=stddevs[y]) * pr)
#
#         total = sum(post)
#         return [p / total for p in post]
#
#     def compute_log_likelihood(allele_freqs, effects, a, b, priors, stddevs):
#         ll = 0
#         for af, eff in zip(allele_freqs, effects):
#             post = posteriors(af, eff, priors, a, b, stddevs)
#             for c in range(3):
#
#
#
#
#     prev_ll = None
#     ll = None
#     LL_DELTA = 1e-5
#     MAX_ITER = 200
#
#     n_iter = 0
#     while True:
#         ll = compute_log_likelihood(fgr_af, fgr_effect, a, b, priors, stddevs)
#
#         if (prev_ll is not None and ll - prev_ll < LL_DELTA) or n_iter > MAX_ITER:
#             break
#
#         a[1], b[1] = find_optimal(partial_ll, 1, fgr_af, fgr_effect)
#         a[2], b[2] = find_optimal(partial_ll, 2, fgr_af, fgr_effect)
#
#         prev_ll = ll
#         n_iter += 1
#
#     return priors, means

import pandas as pd
import numpy as np
import scipy.stats
from jutils import *


def fit_gmm(fgr_af, fgr_effect, bgr_af, bgr_effect, slope=None, bias=None, priors=None, method='BFGS'):

    echo('Fitting GMM!!')

    if slope is None:
        slope = [0, .1, -.1]

    if bias is None:
        bias = [0, 0, 0]

    if priors is None:
        priors = [1 / 3, 1 / 3, 1 / 3]

    def estimate_stddevs(bgr_af, bgr_effect):
        stddevs = {}

        for af, eff in zip(bgr_af, bgr_effect):
            int_log10_af = int(np.log10(af))
            if int_log10_af not in stddevs:
                stddevs[int_log10_af] = []

            stddevs[int_log10_af].append(eff)

        for int_log10_af in stddevs:
            stddevs[int_log10_af] = np.std(stddevs[int_log10_af])

        return stddevs

    stddevs = estimate_stddevs(bgr_af, bgr_effect)

    def get_stddev_af_x(af, stddevs):
        int_log10_af = int(np.log10(af))
        return stddevs[int_log10_af]

    def calculate_posteriors(af, effect, priors, slope, bias, stddevs):
        post = []
        for pr, a, b in zip(priors, slope, bias):
            post.append(pr * scipy.stats.norm.pdf(effect, loc=(a * np.log10(af) + b), scale=get_stddev_af_x(af, stddevs)))

        total = sum(post)
        return [p / total for p in post]

    def ll_c(a, b, allele_freqs, effects, posteriors, stddevs):
        res = 0
        for af, eff, post in zip(allele_freqs, effects, posteriors):
            std_af = get_stddev_af_x(af, stddevs)

            c_ll = -np.log(std_af * math.sqrt(2 * np.pi))
            c_ll += -(1/2) * ((eff - (a * np.log10(af) + b)) / std_af) ** 2
            c_ll *= post

            res += c_ll

        return res

    def neg_ll_c(x, allele_freqs, effects, posteriors, stddevs):
        return -ll_c(x[0], x[1], allele_freqs, effects, posteriors, stddevs)

    prev_ll = None

    LL_DELTA = 1e-5
    MAX_ITER = 200

    n_iter = 0

    echo('INIT, slope:', slope, ', bias:', bias, ', priors:', priors)

    while True:
        c_posteriors = [calculate_posteriors(af, eff, priors, slope, bias, stddevs) for af, eff in zip(fgr_af, fgr_effect)]
        c_posteriors = [[p[0] for p in c_posteriors],
                        [p[1] for p in c_posteriors],
                        [p[2] for p in c_posteriors]]

        opt_res = scipy.optimize.minimize(neg_ll_c, (slope[1], bias[1]), args=(fgr_af, fgr_effect, c_posteriors[1], stddevs), method=method)
        a_lof, b_lof = opt_res.x
        if a_lof > 0:
            slope[1] = a_lof
            bias[1] = b_lof
        else:
            echo('Not updating:', a_lof, b_lof)

        opt_res = scipy.optimize.minimize(neg_ll_c, (slope[2], bias[2]), args=(fgr_af, fgr_effect, c_posteriors[2], stddevs), method=method)
        a_gof, b_gof = opt_res.x

        if a_gof < 0:
            slope[2] = a_gof
            bias[2] = b_gof
        else:
            echo('Not updating:', a_gof, b_gof)

        # total_p = np.sum([p for pp in c_posteriors for p in pp])
        # for c in range(3):
        #     priors[c] = np.sum(c_posteriors[c]) / total_p

        ll = 0
        for a, b, pr, post in zip(slope, bias, priors, c_posteriors):
            class_ll = ll_c(a, b, fgr_af, fgr_effect, post, stddevs)
            class_ll += np.log(pr) * np.sum(post)

            ll += class_ll

        echo('n_iter=', n_iter, ', LL=', ll, ', delta=', ('None' if prev_ll is None else ll - prev_ll), ', slope=', slope, ', bias=', bias, ', priors=', priors)

        if (prev_ll is not None and ll - prev_ll < LL_DELTA) or n_iter > MAX_ITER:
            break

        prev_ll = ll
        n_iter += 1

    return priors, slope, bias


def fit_gmm2(fgr_af, fgr_effect, bgr_af, bgr_effect, means=None, priors=None, method=None):

    echo('Fitting GMM /// 2!')

    if means is None:
        means = [0, .1, -.1]

    if priors is None:
        priors = [1 / 3, 1 / 3, 1 / 3]

    def estimate_stddevs(bgr_af, bgr_effect):
        stddevs = {}

        for af, eff in zip(bgr_af, bgr_effect):
            int_log10_af = int(np.log10(af))
            if int_log10_af not in stddevs:
                stddevs[int_log10_af] = []

            stddevs[int_log10_af].append(eff)

        for int_log10_af in stddevs:
            stddevs[int_log10_af] = np.std(stddevs[int_log10_af])

        return stddevs

    stddevs = estimate_stddevs(bgr_af, bgr_effect)

    def get_stddev_af_x(af, stddevs):
        int_log10_af = int(np.log10(af))
        return stddevs[int_log10_af]

    def calculate_posteriors(af, effect, priors, means, stddevs):
        post = []
        for pr, m in zip(priors, means):
            post.append(pr * scipy.stats.norm.pdf(effect, loc=m * np.log10(af), scale=get_stddev_af_x(af, stddevs)))

        total = sum(post)
        return [p / total for p in post]

    def ll_c(m, allele_freqs, effects, posteriors, stddevs):
        res = 0
        for af, eff, post in zip(allele_freqs, effects, posteriors):
            std_af = get_stddev_af_x(af, stddevs)

            c_ll = -np.log(std_af * math.sqrt(2 * np.pi))
            c_ll += -(1/2) * ((eff - m * np.log10(af)) / std_af) ** 2
            c_ll *= post

            res += c_ll

        return res

    def neg_ll_c(x, allele_freqs, effects, posteriors, stddevs):
        return -ll_c(x[0], allele_freqs, effects, posteriors, stddevs)

    prev_ll = None

    LL_DELTA = 1e-5
    MAX_ITER = 200

    n_iter = 0

    echo('INIT, means:', means, ', priors:', priors)

    while True:
        c_posteriors = [calculate_posteriors(af, eff, priors, means, stddevs) for af, eff in zip(fgr_af, fgr_effect)]
        c_posteriors = [[p[0] for p in c_posteriors],
                        [p[1] for p in c_posteriors],
                        [p[2] for p in c_posteriors]]

        # opt_res = scipy.optimize.minimize(neg_ll_c, [means[1]], args=(fgr_af, fgr_effect, c_posteriors[1], stddevs), method=method)
        # means[1] = opt_res.x[0]

        opt_res = scipy.optimize.minimize(neg_ll_c, [means[2]], args=(fgr_af, fgr_effect, c_posteriors[2], stddevs), method=method)
        means[2] = opt_res.x[0]

        # total_p = np.sum([p for pp in c_posteriors for p in pp])
        # for c in range(3):
        #     priors[c] = np.sum(c_posteriors[c]) / total_p

        ll = 0
        for m, pr, post in zip(means, priors, c_posteriors):
            class_ll = ll_c(m, fgr_af, fgr_effect, post, stddevs)
            class_ll += np.log(pr) * np.sum(post)

            ll += class_ll

        echo('n_iter:', n_iter, ', LL:', ll, ', delta:', (ll - prev_ll) if prev_ll is not None else 'None' , ', means:', means, ', priors:', priors)

        # if (prev_ll is not None and ll - prev_ll < LL_DELTA) or n_iter > MAX_ITER:
        if n_iter > MAX_ITER:
            break

        prev_ll = ll
        n_iter += 1

    return priors, means


def match_quantiles(d_fgr, d_bgr, n_quants=20):

    quants = [0] + [i / n_quants for i in range(1, n_quants)] + [1]

    quant_edges = np.quantile(d_fgr['MAF'], quants)
    # quant_edges = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5]

    bgr_to_plot = []
    fgr_to_plot = []

    for q_idx in range(len(quant_edges) - 1):
        q_lo = quant_edges[q_idx]
        q_hi = quant_edges[q_idx + 1]

        if q_idx == n_quants - 1:
            fgr_vars = d_fgr[(d_fgr['MAF'] >= q_lo) & (d_fgr['MAF'] <= q_hi)]
            bgr_vars = d_bgr[(d_bgr['MAF'] >= q_lo) & (d_bgr['MAF'] <= q_hi)]

        else:

            fgr_vars = d_fgr[(d_fgr['MAF'] >= q_lo) & (d_fgr['MAF'] < q_hi)]
            bgr_vars = d_bgr[(d_bgr['MAF'] >= q_lo) & (d_bgr['MAF'] < q_hi)]

        if len(bgr_vars) == 0 or len(fgr_vars) == 0:
            continue

        bgr_std = np.std(bgr_vars['beta'])

        echo(q_lo, q_hi, len(fgr_vars), len(bgr_vars), ', std(bgr)=', bgr_std)

        q_n = min(len(fgr_vars), len(bgr_vars))

        bgr_vars = bgr_vars.sample(n=q_n)

        for _, r in bgr_vars.iterrows():
            r['norm_beta'] = r['beta'] / bgr_std

            bgr_to_plot.append(r)

        fgr_vars = fgr_vars.sample(n=q_n)
        for _, r in fgr_vars.iterrows():
            r['norm_beta'] = r['beta'] / bgr_std
            fgr_to_plot.append(r)

    fgr_to_plot = pd.DataFrame(fgr_to_plot)
    bgr_to_plot = pd.DataFrame(bgr_to_plot)

    return fgr_to_plot, bgr_to_plot


def compute_interval_medians(d_fgr, n_quants=20, edges=None, return_stds=True, maf_key=None, compute_means=False, groupby_gene=False):
    echo('[compute_interval_medians]')
    values = {}
    if maf_key is None:
        maf_key = 'MAF'

    if edges is None:
        quants = [0] + [i / n_quants for i in range(1, n_quants)] + [1]
        edges = np.quantile(d_fgr[maf_key], quants)
    else:
        edges = sorted(edges)

    bin_centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]

    medians = []
    stds = []
    for q_idx in range(len(edges) - 1):
        q_lo = edges[q_idx]
        q_hi = edges[q_idx + 1]

        if q_idx == n_quants - 1:
            fgr_vars = d_fgr[(d_fgr[maf_key] >= q_lo) & (d_fgr[maf_key] <= q_hi)]
        else:
            fgr_vars = d_fgr[(d_fgr[maf_key] >= q_lo) & (d_fgr[maf_key] < q_hi)]

        if groupby_gene:
            if compute_means:
                fgr_vars = fgr_vars.groupby(GENE_NAME).mean()
            else:
                fgr_vars = fgr_vars.groupby(GENE_NAME).median()

        if len(fgr_vars) > 0:
            m = np.mean(fgr_vars['beta']) if compute_means else np.median(fgr_vars['beta'])
            stds.append(np.std(fgr_vars['beta']))
            values[bin_centers[q_idx]] = list(fgr_vars['beta'])
        else:
            m = 0
            stds.append(0)
            values[bin_centers[q_idx]] = []

        medians.append(m)

    medians = np.array(medians)
    stds = np.array(stds)

    if return_stds:
        return medians, stds, bin_centers, values
    else:
        return medians, bin_centers, values


NON_DELETERIOUS = 'non_del'
DELETERIOUS = 'deleterious'
PTV = 'ptv'

def determine_variant_type(x, pAI_t, spAI_t, non_del_pAI_t, non_del_spAI_t, pathogenicity_score_label=PRIMATEAI_SCORE):

    cons = set(x[VCF_CONSEQUENCE].split(','))

    if len(cons & set(ALL_PTV)) > 0:
        return PTV

    is_syn = len(cons & {VCF_SYNONYMOUS_VARIANT}) > 0

    pai = x[pathogenicity_score_label] if not np.isnan(x[pathogenicity_score_label]) else 0
    sai = x[SPLICEAI_MAX_SCORE] if not np.isnan(x[SPLICEAI_MAX_SCORE]) else 0

    if pai >= pAI_t or sai >= spAI_t:
        return DELETERIOUS

    if not np.isnan(x[pathogenicity_score_label]) and pai <= non_del_pAI_t and sai <= non_del_spAI_t and len(cons & {VCF_MISSENSE_VARIANT}) > 0:
        return NON_DELETERIOUS

    if is_syn:
        return VCF_SYNONYMOUS_VARIANT

    return 'other'

LoF_label = 'Loss of Function'
DELETERIOUS_MISSENSE_label = 'Deleterious Missenses'
SPLICING_label = 'Cryptic Splicing'
SYNONYMOUS_label = 'Synonymous'
BENIGN_MISSENSE_label = 'Benign Missenses'


def plot_effect_size_vs_AF(all_vars_stats,
                           significant_gene_phenotypes,
                           S_FACTOR=8,
                           figsize=(8, 8),
                           compute_means=False,
                           to_keep=None,
                           to_exclude=None,
                           pathogenicity_score_label=PRIMATEAI_SCORE,
                           spAI_t=0.2,
                           pAI_t=0.8,
                           benign_pAI_t=0.1,
                           out_prefix=None,
                           store_data=False,
                           plot_all_GWAS_hits_together=False,
                           skip_GWAS_hits=False,
                           fit_alkes_curve=False,
                           fit_curves_on_means=False,
                           curve_parameters_in_squared_space=None,
                           ball_alpha=1):

    echo('[plot_effect_size_vs_AF]')
    echo('del. missenses: pAI >', pAI_t)
    echo('benign missenses: pAI <', benign_pAI_t)
    echo('splice variants: spAI >', spAI_t)
    echo('fit_alkes_curve:', fit_alkes_curve)
    echo('fit_curves_on_means:', fit_curves_on_means)

    if type(all_vars_stats) is str:
        echo('Reading all_vars_stats:', all_vars_stats)
        all_vars_stats = pd.read_csv(all_vars_stats, sep='\t')

    if type(significant_gene_phenotypes) is str:
        echo('Reading significant_gene_phenotypes:', significant_gene_phenotypes)
        significant_gene_phenotypes = pd.read_csv(significant_gene_phenotypes, sep='\t')

    all_vars_stats = all_vars_stats[(all_vars_stats[VCF_CONSEQUENCE].isin(ALL_PTV + [VCF_SYNONYMOUS_VARIANT])) |
                                    (~all_vars_stats[SPLICEAI_MAX_SCORE].isnull()) |
                                    (~all_vars_stats[pathogenicity_score_label].isnull())].copy()

    if out_prefix is not None and store_data:
        cols_to_keep = [GENE_NAME,
                        VARID_REF_ALT,
                        VCF_CONSEQUENCE,
                        pathogenicity_score_label,
                        SPLICEAI_MAX_SCORE,
                        'MAF',
                        'ALL/del/carrier/pvalue/fdr_corr',
                        'seen_in_gnomad',
                        'seen_in_topmed',
                        'pvalue/gwas',
                        'beta',
                        'beta/gwas',
                        'std'
                         ]

        echo('Saving all_var_stats:', out_prefix + '.variant_data.csv.gz')
        all_vars_stats[cols_to_keep].to_csv(out_prefix + '.variant_data.csv.gz', sep='\t', index=False)

        cols_to_keep = [GENE_NAME,
                        'index_variant/gwas',
                        'AF/gwas',
                        'beta/gwas',
                        'pvalue/gwas',
                        'ALL/del/carrier/beta',
                        'ALL/ptv/carrier/beta'
                        ]

        echo('Saving significant_gene_phenotypes:', out_prefix + '.gwas_data.csv.gz')
        significant_gene_phenotypes[cols_to_keep].to_csv(out_prefix + '.gwas_data.csv.gz', sep='\t', index=False)

    echo('Setting var_type')

    all_vars_stats['var_type'] = all_vars_stats.apply(determine_variant_type,
                                                      args=(pAI_t,
                                                            spAI_t,
                                                            benign_pAI_t,
                                                            0,
                                                            pathogenicity_score_label),
                                                      axis=1)

    d_fgr = all_vars_stats[all_vars_stats['var_type'].isin([DELETERIOUS, 'ptv'])].copy()
    echo('d_fgr:', d_fgr.shape)

    d_bgr_non_del = all_vars_stats[all_vars_stats['var_type'] == NON_DELETERIOUS].copy()
    echo('d_bgr_non_del:', d_bgr_non_del.shape)

    d_bgr_syn = all_vars_stats[all_vars_stats['var_type'].isin([VCF_SYNONYMOUS_VARIANT])].copy()
    echo('d_bgr_syn:', d_bgr_syn.shape)

    gwas_vars = significant_gene_phenotypes.sort_values('pvalue/gwas').drop_duplicates('index_variant/gwas').copy()

    gwas_vars['beta/gwas/flipped'] = np.where(gwas_vars['AF/gwas'] < 0.5,
                                              gwas_vars['beta/gwas'],
                                              -gwas_vars['beta/gwas'])

    gwas_vars['beta/gwas/flipped'] = -np.where(gwas_vars['ALL/del/carrier/beta'] < 0,
                                               gwas_vars['beta/gwas/flipped'],
                                              -gwas_vars['beta/gwas/flipped'])

    gwas_vars['MAF'] = np.where(gwas_vars['AF/gwas'] < 0.5,
                                gwas_vars['AF/gwas'],
                                1 - gwas_vars['AF/gwas'])

    gwas_vars['type'] = 'GWAS'
    echo('gwas_vars:', gwas_vars.shape)

    fig = plt.figure(figsize=figsize)
    if to_exclude is None:
        to_exclude = set()

    to_exclude = set(significant_gene_phenotypes[
                     significant_gene_phenotypes['ALL/del/carrier/beta'] *
                     significant_gene_phenotypes['ALL/ptv/carrier/beta'] < 0][GENE_NAME]) | to_exclude

    if to_keep is None:
        to_keep = set(significant_gene_phenotypes[GENE_NAME])

    to_keep = to_keep - to_exclude

    echo('to_exclude:', len(to_exclude), ', to_keep:', len(to_keep))

    gwas_vars = gwas_vars[gwas_vars[GENE_NAME].isin(to_keep)].copy()
    echo('Total genes used:', len(set(all_vars_stats[
                                          (~all_vars_stats[GENE_NAME].isin(to_exclude)) &
                                          all_vars_stats[GENE_NAME].isin(to_keep)][GENE_NAME])))

    def transform_beta(d):
        return -d['beta']

    def transform_maf(maf):
        return np.log10(maf)

    x_values = transform_maf(d_fgr['MAF'])

    min_log_af = np.min(transform_maf(all_vars_stats['MAF']))
    frac = abs(min_log_af - int(min_log_af))
    min_log_af = int(min_log_af) - (0.25 if frac < 0.25 else 0.75 if frac < 0.5 else 1.25)

    n_bins = abs(int((min_log_af - 0.25) / 0.5))
    edges = np.linspace(min_log_af, -0.25, n_bins)

    echo('n_bins:', n_bins, 'edges:', edges, ', centers:', [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)])

    beta_label = 'log_beta'

    all_values = {}
    all_medians = {}

    variants_by_type = {}

    # all_data_for_plotting = None
    def ball_func(v):
        return math.pow(v, 1/3)
    for d_to_use, label, color in [(d_fgr[d_fgr[VCF_CONSEQUENCE].isin(ALL_PTV)], LoF_label, 'red'),
                                   (d_fgr[(d_fgr[VCF_CONSEQUENCE] == VCF_MISSENSE_VARIANT) & (d_fgr[pathogenicity_score_label] >= pAI_t)], DELETERIOUS_MISSENSE_label, 'orange'),
                                   (d_fgr[(d_fgr[SPLICEAI_MAX_SCORE] >= spAI_t) &
                                          (~d_fgr[VCF_CONSEQUENCE].isin(SPLICE_VARIANTS)) &
                                          ((d_fgr[pathogenicity_score_label].fillna(0) <= benign_pAI_t))
                                    ], SPLICING_label, 'brown'),
                                   (d_bgr_syn,SYNONYMOUS_label , 'blue'),
                                   (d_bgr_non_del, BENIGN_MISSENSE_label, 'green')

                                   ]:

        d_to_use = d_to_use.copy()
        d_to_use = d_to_use[(~d_to_use[GENE_NAME].isin(to_exclude)) & d_to_use[GENE_NAME].isin(to_keep)].copy()
        d_to_use = d_to_use.sort_values('pvalue/gwas').drop_duplicates(subset=[VARID_REF_ALT])
        echo(label, len(d_to_use))

        #     res_d = None
        #     for i in range(len(edges) - 1):
        #         e1, e2 = edges[i], edges[i + 1]
        #         _d = d_to_use[(d_to_use[VCF_AF] >= 10 ** e1) & (d_to_use[VCF_AF] < 10 ** e2)]
        #         _d = _d[[VCF_AF, 'beta', GENE_NAME, 'MAF', 'std']].groupby(GENE_NAME).median().reset_index()
        #         if res_d is None:
        #             res_d = _d
        #         else:
        #             res_d = pd.concat([res_d, _d], ignore_index=True)
        #     d_to_use = res_d

        d_to_use[beta_label] = transform_beta(d_to_use)
        d_to_use['rv_gwas_ratio'] = -d_to_use['beta'] / np.abs(d_to_use['beta/gwas'])

        variants_by_type[label] = d_to_use

        # ac = 1
        # v = d_to_use[d_to_use[VCF_AC] <= ac]['rv_gwas_ratio']
        # echo('singleton rv_gwas_ratio:', np.min(v), np.median(v), np.max(v), ', n=', len(v))

        #     wls_weights = [1 / (s ** 2) for s in d_to_use['std']]
        #     wls_model = sm.WLS(d_to_use[beta_label], pd.DataFrame({'log_MAF': transform_maf(d_to_use['MAF']), 'bias': 1}),
        #                        weights=wls_weights)

        #     results = wls_model.fit()

        #     display(pd.DataFrame({beta_label: results.params, 'p': results.pvalues}))

        #     plt.plot(x, results.params['log_MAF'] * x + results.params['bias'], color=color)
        n_variants = len(d_to_use)
        cur_data_for_plotting = pd.DataFrame({'log_MAF': transform_maf(d_to_use['MAF']),
                                              'beta': d_to_use[beta_label],
                                              'std': d_to_use['std'],
                                              GENE_NAME: d_to_use[GENE_NAME]})

        medians, stds, edges_to_plot, values = compute_interval_medians(cur_data_for_plotting,
                                                                        edges=edges,
                                                                        return_stds=True,
                                                                        maf_key='log_MAF',
                                                                        compute_means=compute_means,
                                                                        groupby_gene=False)

        # if all_data_for_plotting is None:
        #     all_data_for_plotting = cur_data_for_plotting
        # else:
        #     all_data_for_plotting = pd.concat([all_data_for_plotting, cur_data_for_plotting], ignore_index=True)
        if fit_curves_on_means:
            betas_to_fit = medians
            mafs_to_fit = edges_to_plot
        else:
            betas_to_fit = d_to_use[beta_label]
            mafs_to_fit = transform_maf(d_to_use['MAF'])

        if curve_parameters_in_squared_space is not None:

            echo('Plotting curve_parameters_in_squared_space')
            curve_parameters = curve_parameters_in_squared_space[label]

            c_y = curve_parameters['log_MAF'] * np.log10(2 * (10.0 ** x_values) * (1 - (10.0 ** x_values))) / 2 + curve_parameters['bias']
            y_values = 10.0 ** c_y
            # y_values = curve_parameters['bias'] * np.power(curve_parameters['log_MAF'] / 2,
            #                                                2 * (10.0 ** x_values) * (1 - (10.0 ** x_values)))

            d_to_plot = pd.DataFrame({'x': x_values,
                                      'y': y_values}).sort_values('x')

            plt.plot(d_to_plot['x'], d_to_plot['y'], color=color)

        else:
            if fit_alkes_curve:
                glm_x = []
                glm_y = []

                for v_beta, v_log10_maf in zip(betas_to_fit, mafs_to_fit):

                    if not np.isnan(v_beta):
                        glm_y.append(v_beta)
                        c_maf = 10**v_log10_maf
                        c_x = np.log10(np.sqrt(2 * c_maf * (1 - c_maf)))
                        glm_x.append(c_x)

                echo('Plotting Alkes curve')
                glm_model = sm.GLM(np.array(glm_y),
                                   pd.DataFrame({'log_MAF': glm_x,
                                                 'bias': 2}),
                                   family=sm.families.Gaussian(sm.families.links.log())
                                   #                        family=sm.families.Gaussian(sm.families.links.identity())
                                   )

                results = glm_model.fit(method="lbfgs")

                display(results.summary())

                display(pd.DataFrame({beta_label: results.params, 'p': results.pvalues}))

                c_x = np.log10(np.sqrt(2 * (10.0**x_values) * (1 - (10.0 ** x_values))))
                y_values = np.power(math.e, results.params['log_MAF'] * c_x + 2 * results.params['bias'])

                # y_values = []
                # for e in x_values:
                #     c_maf = 10**e
                #     c_x = np.log10(np.sqrt(2 * c_maf * (1 - c_maf)))
                #     y_val = np.power(math.e, results.params['log_MAF'] * c_x + results.params['bias'])
                #     y_values.append(y_val)

                d_to_plot = pd.DataFrame({'x': x_values,
                                          'y': y_values}).sort_values('x')

                plt.plot(d_to_plot['x'], d_to_plot['y'], color=color)
            else:

                glm_x = []
                glm_y = []

                for v_beta, v_log10_maf in zip(betas_to_fit, mafs_to_fit):
                    if not np.isnan(v_beta):
                        glm_y.append(v_beta)
                        glm_x.append(v_log10_maf)

                glm_model = sm.GLM(np.array(glm_y), pd.DataFrame({'log_MAF': glm_x,
                                                                  'bias': 1}),
                                   family=sm.families.Gaussian(sm.families.links.log())
                                   #                        family=sm.families.Gaussian(sm.families.links.identity())
                                   )

                results = glm_model.fit(method="lbfgs")

                display(results.summary())

                display(pd.DataFrame({beta_label: results.params, 'p': results.pvalues}))

                d_to_plot = pd.DataFrame({'x': x_values,
                                          'y': np.power(math.e, results.params['log_MAF'] * x_values +
                                                        results.params['bias'])}).sort_values('x')

                plt.plot(d_to_plot['x'], d_to_plot['y'], color=color)

        all_values[label] = values
        all_medians[label] = medians

        plt.scatter(edges_to_plot,
                    medians,
                    s=[S_FACTOR * ball_func(len(values[k])) for k in edges_to_plot],
                    color=color,
                    label=label + ', n= ' + '{:,}'.format(n_variants), # + ', slope=%.3lf' % results.params['log_MAF']
                    alpha=ball_alpha)

    if not skip_GWAS_hits:
        if compute_means:
            echo('mean GWAS effect:', np.mean(np.abs(gwas_vars['beta/gwas/flipped'])))
        else:
            echo('median GWAS effect:', np.median(np.abs(gwas_vars['beta/gwas/flipped'])))

        gwas_data_for_plotting = pd.DataFrame({'log_MAF': np.log10(gwas_vars['MAF']),
                                               'beta': gwas_vars['beta/gwas/flipped'],
                                               'std': [1] * len(gwas_vars),
                                               GENE_NAME: gwas_vars[GENE_NAME]})
        if plot_all_GWAS_hits_together:

            mean_gwas_af = np.mean(gwas_data_for_plotting['log_MAF'])
            mean_gwas_effect = np.mean(gwas_data_for_plotting['beta'])
            echo('mean_gwas_af:', mean_gwas_af, ', mean_gwas_effect:', mean_gwas_effect)

            plt.scatter([mean_gwas_af],
                        [mean_gwas_effect],
                        s=[S_FACTOR * ball_func(len(gwas_data_for_plotting))],
                        color='black',
                        label='GWAS variants, n= ' + '{:,}'.format(len(gwas_data_for_plotting)),
                        marker='D',
                        alpha=ball_alpha)


        else:
            medians, edges_to_plot, values = compute_interval_medians(gwas_data_for_plotting,
                                                                      edges=edges,
                                                                      return_stds=False,
                                                                      maf_key='log_MAF',
                                                                      compute_means=compute_means)

            # all_data_for_plotting = pd.concat([all_data_for_plotting, gwas_data_for_plotting], ignore_index=True)

            echo('GWAS effects:', list(zip(edges_to_plot, medians)))
            # display(d_to_use.sort_values('MAF').head())

            variants_by_type['GWAS'] = gwas_vars

            # to_plot = pd.DataFrame({'e': edges_to_plot, 'm': medians})
            # to_plot = to_plot[to_plot['m'] != 0]
            plt.scatter(edges_to_plot,
                        medians,
                        s=[S_FACTOR * ball_func(len(values[k])) for k in edges_to_plot],
                        color='black',
                        label='GWAS variants, n= ' + '{:,}'.format(len(gwas_data_for_plotting)),
                        marker='D',
                        alpha=ball_alpha)

    # plt.plot(to_plot['e'], to_plot['m'], 'o', color='yellow', markersize=6, label='GWAS', markeredgecolor='black')

    for label in [LoF_label, DELETERIOUS_MISSENSE_label]:
        for x, y in zip(edges_to_plot, all_medians[label]):
            p_syn = scipy.stats.ttest_ind(all_values[label][x], all_values[SYNONYMOUS_label][x])[1]
            p_non_del = scipy.stats.ttest_ind(all_values[label][x], all_values[BENIGN_MISSENSE_label][x])[1]
            echo(x, y, len(all_values[label][x]), len(all_values[SYNONYMOUS_label][x]),
                 len(all_values[BENIGN_MISSENSE_label][x]), p_syn, p_non_del)

            # if p_syn < 0.05:
            #     plt.text(x+.05, y, '*', color='green', fontsize=16, fontweight='extra bold' if p_syn < 1e-10 else 'ultralight')
            #
            # if p_non_del < 0.05:
            #     plt.text(x+.15, y, '*', color='blue', fontsize=16, fontweight='extra bold' if p_syn < 1e-10 else 'ultralight')

    ylim = plt.ylim()
    echo('ylim:', ylim)
    # sns.set_style("whitegrid", {'axes.grid': False})

    plt.xlabel('Allele Frequency')
    plt.ylabel('Per Allele Effect (Z-score)')

    lgnd = plt.legend(labelspacing=1)
    # change the marker size manually for both lines
    for lgnd_idx in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[lgnd_idx]._sizes = [70]

    sns.despine()

    # ax = fig.axes[0]
    # plt.draw()
    #
    # labels = [int(item.get_text()) for item in ax.get_xticklabels()]
    # echo(labels)
    # for i in range(len(labels)):
    #     l = labels[i]
    #     if l <= -3:
    #         labels[i] = '$10^%d$' % l
    #     elif l == -2:
    #         labels[i] = '0.01'
    #     elif l == -1:
    #         labels[i] = '0.1'
    # echo(labels)
    # ax.set_xticklabels(labels)

    if out_prefix is not None:
        echo('Saving figures to:', out_prefix)
        plt.savefig(out_prefix + '.svg')
        plt.savefig(out_prefix + '.png', dpi=300)

    plt.show()

    return variants_by_type


def plot_squared_effect_size_vs_AF(all_vars_stats,
                                   significant_gene_phenotypes,
                                   S_FACTOR=8,
                                   figsize=(8, 8),
                                   compute_means=False,
                                   to_keep=None,
                                   to_exclude=None,
                                   pathogenicity_score_label=PRIMATEAI_SCORE,
                                   spAI_t=0.2,
                                   pAI_t=0.8,
                                   benign_pAI_t=0.1,
                                   out_prefix=None,
                                   store_data=False,
                                   plot_all_GWAS_hits_together=False,
                                   skip_GWAS_hits=False,
                                   fit_alkes_curve=False,
                                   alkes_over=True,
                                   min_AF_for_alkes_curve=1e-100):

    echo('[plot_squared_effect_size_vs_AF]')
    echo('del. missenses: pAI >', pAI_t)
    echo('benign missenses: pAI <', benign_pAI_t)
    echo('splice variants: spAI >', spAI_t)

    if type(all_vars_stats) is str:
        echo('Reading all_vars_stats:', all_vars_stats)
        all_vars_stats = pd.read_csv(all_vars_stats, sep='\t')

    if type(significant_gene_phenotypes) is str:
        echo('Reading significant_gene_phenotypes:', significant_gene_phenotypes)
        significant_gene_phenotypes = pd.read_csv(significant_gene_phenotypes, sep='\t')

    all_vars_stats = all_vars_stats[(all_vars_stats[VCF_CONSEQUENCE].isin(ALL_PTV + [VCF_SYNONYMOUS_VARIANT])) |
                                    (~all_vars_stats[SPLICEAI_MAX_SCORE].isnull()) |
                                    (~all_vars_stats[pathogenicity_score_label].isnull())].copy()

    if out_prefix is not None and store_data:
        cols_to_keep = [GENE_NAME,
                        VARID_REF_ALT,
                        VCF_CONSEQUENCE,
                        pathogenicity_score_label,
                        SPLICEAI_MAX_SCORE,
                        'MAF',
                        'ALL/del/carrier/pvalue/fdr_corr',
                        'seen_in_gnomad',
                        'seen_in_topmed',
                        'pvalue/gwas',
                        'beta',
                        'beta/gwas',
                        'std'
                         ]
        echo('Saving all_var_stats:', out_prefix + '.variant_data.csv.gz')
        all_vars_stats[cols_to_keep].to_csv(out_prefix + '.variant_data.csv.gz', sep='\t', index=False)

        cols_to_keep = [GENE_NAME,
                        'index_variant/gwas',
                        'AF/gwas',
                        'beta/gwas',
                        'pvalue/gwas',
                        'ALL/del/carrier/beta',
                        'ALL/ptv/carrier/beta'
                        ]

        echo('Saving significant_gene_phenotypes:', out_prefix + '.gwas_data.csv.gz')
        significant_gene_phenotypes[cols_to_keep].to_csv(out_prefix + '.gwas_data.csv.gz', sep='\t', index=False)

    echo('Setting var_type')

    all_vars_stats['var_type'] = all_vars_stats.apply(determine_variant_type,
                                                      args=(pAI_t,
                                                            spAI_t,
                                                            benign_pAI_t,
                                                            0,
                                                            pathogenicity_score_label),
                                                      axis=1)

    d_fgr = all_vars_stats[all_vars_stats['var_type'].isin([DELETERIOUS, 'ptv'])].copy()
    echo('d_fgr:', d_fgr.shape)

    d_bgr_non_del = all_vars_stats[all_vars_stats['var_type'] == NON_DELETERIOUS].copy()
    echo('d_bgr_non_del:', d_bgr_non_del.shape)

    d_bgr_syn = all_vars_stats[all_vars_stats['var_type'].isin([VCF_SYNONYMOUS_VARIANT])].copy()
    echo('d_bgr_syn:', d_bgr_syn.shape)

    gwas_vars = significant_gene_phenotypes.sort_values('pvalue/gwas').drop_duplicates('index_variant/gwas').copy()

    gwas_vars['beta/gwas/flipped'] = np.where(gwas_vars['AF/gwas'] < 0.5,
                                              gwas_vars['beta/gwas'],
                                              -gwas_vars['beta/gwas'])

    gwas_vars['beta/gwas/flipped'] = -np.where(gwas_vars['ALL/del/carrier/beta'] < 0,
                                               gwas_vars['beta/gwas/flipped'],
                                              -gwas_vars['beta/gwas/flipped'])

    gwas_vars['MAF'] = np.where(gwas_vars['AF/gwas'] < 0.5,
                                gwas_vars['AF/gwas'],
                                1 - gwas_vars['AF/gwas'])

    gwas_vars['type'] = 'GWAS'
    echo('gwas_vars:', gwas_vars.shape)

    fig = plt.figure(figsize=figsize)
    if to_exclude is None:
        to_exclude = set()

    to_exclude = set(significant_gene_phenotypes[
                         significant_gene_phenotypes['ALL/del/carrier/beta'] *
                         significant_gene_phenotypes['ALL/ptv/carrier/beta'] < 0][GENE_NAME]) | to_exclude

    if to_keep is None:
        to_keep = set(significant_gene_phenotypes[GENE_NAME])

    to_keep = to_keep - to_exclude

    echo('to_exclude:', len(to_exclude), ', to_keep:', len(to_keep))

    gwas_vars = gwas_vars[gwas_vars[GENE_NAME].isin(to_keep)].copy()

    def transform_beta(d):
        return d['beta'] ** 2

    def transform_maf(maf):
        return np.log10(maf)

    x_values = transform_maf(d_fgr['MAF'])

    min_log_af = np.min(transform_maf(all_vars_stats['MAF']))
    frac = abs(min_log_af - int(min_log_af))
    min_log_af = int(min_log_af) - (0.25 if frac < 0.25 else 0.75)

    n_bins = abs(int((min_log_af - 0.25) / 0.5))
    edges = np.linspace(min_log_af, -0.25, n_bins)

    echo('n_bins:', n_bins, 'edges:', edges, ', centers:', [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)])

    beta_label = 'log_beta'

    all_values = {}
    all_medians = {}

    variants_by_type = {}

    # all_data_for_plotting = None
    curve_parameters = {}

    for d_to_use, label, color in [(d_fgr[d_fgr[VCF_CONSEQUENCE].isin(ALL_PTV)], LoF_label, 'red'),
                                   (d_fgr[(d_fgr[VCF_CONSEQUENCE] == VCF_MISSENSE_VARIANT) & (d_fgr[pathogenicity_score_label] >= pAI_t)], DELETERIOUS_MISSENSE_label, 'orange'),
                                   (d_fgr[(d_fgr[SPLICEAI_MAX_SCORE] >= spAI_t) &
                                          (~d_fgr[VCF_CONSEQUENCE].isin(SPLICE_VARIANTS)) &
                                          ((d_fgr[pathogenicity_score_label].fillna(0) <= benign_pAI_t))
                                    ], SPLICING_label, 'brown'),
                                   (d_bgr_syn,SYNONYMOUS_label , 'blue'),
                                   (d_bgr_non_del, BENIGN_MISSENSE_label, 'green')

                                   ]:

        d_to_use = d_to_use.copy()
        d_to_use = d_to_use[(~d_to_use[GENE_NAME].isin(to_exclude)) & d_to_use[GENE_NAME].isin(to_keep)].copy()
        d_to_use = d_to_use.sort_values('pvalue/gwas').drop_duplicates(subset=[VARID_REF_ALT])
        echo(label, len(d_to_use))

        #     res_d = None
        #     for i in range(len(edges) - 1):
        #         e1, e2 = edges[i], edges[i + 1]
        #         _d = d_to_use[(d_to_use[VCF_AF] >= 10 ** e1) & (d_to_use[VCF_AF] < 10 ** e2)]
        #         _d = _d[[VCF_AF, 'beta', GENE_NAME, 'MAF', 'std']].groupby(GENE_NAME).median().reset_index()
        #         if res_d is None:
        #             res_d = _d
        #         else:
        #             res_d = pd.concat([res_d, _d], ignore_index=True)
        #     d_to_use = res_d

        d_to_use[beta_label] = transform_beta(d_to_use)
        d_to_use['rv_gwas_ratio'] = -d_to_use['beta'] / np.abs(d_to_use['beta/gwas'])

        variants_by_type[label] = d_to_use

        # ac = 1
        # v = d_to_use[d_to_use[VCF_AC] <= ac]['rv_gwas_ratio']
        # echo('singleton rv_gwas_ratio:', np.min(v), np.median(v), np.max(v), ', n=', len(v))

        #     wls_weights = [1 / (s ** 2) for s in d_to_use['std']]
        #     wls_model = sm.WLS(d_to_use[beta_label], pd.DataFrame({'log_MAF': transform_maf(d_to_use['MAF']), 'bias': 1}),
        #                        weights=wls_weights)

        #     results = wls_model.fit()

        #     display(pd.DataFrame({beta_label: results.params, 'p': results.pvalues}))

        #     plt.plot(x, results.params['log_MAF'] * x + results.params['bias'], color=color)
        n_variants = len(d_to_use)
        cur_data_for_plotting = pd.DataFrame({'log_MAF': transform_maf(d_to_use['MAF']),
                                              'beta': d_to_use[beta_label],
                                              'std': d_to_use['std'],
                                              GENE_NAME: d_to_use[GENE_NAME]})

        medians, stds, edges_to_plot, values = compute_interval_medians(cur_data_for_plotting,
                                                                        edges=edges,
                                                                        return_stds=True,
                                                                        maf_key='log_MAF',
                                                                        compute_means=compute_means,
                                                                        groupby_gene=False)

        # if all_data_for_plotting is None:
        #     all_data_for_plotting = cur_data_for_plotting
        # else:
        #     all_data_for_plotting = pd.concat([all_data_for_plotting, cur_data_for_plotting], ignore_index=True)

        if fit_alkes_curve:

            echo('Plotting Alkes curve with all')

            glm_x = []
            glm_y = []

            # for beta_squared, log10_maf in zip(medians, edges_to_plot):
            for beta_squared, log10_maf in zip(d_to_use[beta_label], transform_maf(d_to_use['MAF'])):
                if not np.isnan(beta_squared):
                    c_maf = 10.0**log10_maf
                    if alkes_over and c_maf < min_AF_for_alkes_curve:
                        continue

                    if not alkes_over and c_maf > min_AF_for_alkes_curve:
                        continue

                    c_x = np.log10(2 * c_maf * (1 - c_maf))

                    glm_x.append(c_x)
                    glm_y.append(beta_squared)

            if alkes_over:
                echo('Using variants with MAF >', min_AF_for_alkes_curve, ':', len(glm_x))
            else:
                echo('Using variants with MAF <', min_AF_for_alkes_curve, ':', len(glm_x))

            # echo('Log-linear model')
            # wls_weights = [1 / (s**3) for s in d_to_use['std']]
            # # wls_weights = [1 for s in d_to_use['std']]
            # lm_model = sm.WLS(np.log10(glm_y),
            #                   pd.DataFrame({'log_MAF': glm_x,
            #                                 'bias': 2})
            #                   ,weights=wls_weights)
            #
            # results = lm_model.fit()
            #
            # display(results.summary())
            # c_x = np.log10(2 * (10.0**x_values) * (1 - (10.0 ** x_values)))
            # y_values = np.power(10, results.params['log_MAF'] * c_x + 2 * results.params['bias'])
            #
            # # for e in list(x_values):
            # #     c_maf = 10**e
            # #     c_x = np.log10(2 * c_maf * (1 - c_maf))
            # #     y_val = np.power(math.e, results.params['log_MAF'] * c_x + results.params['bias'])
            # #     y_values.append(y_val)
            #
            # d_to_plot = pd.DataFrame({'x': x_values,
            #                           'y': y_values}).sort_values('x')
            #
            # plt.plot(d_to_plot['x'], d_to_plot['y'], color=color, ls='--')

            glm_model = sm.GLM(np.array(glm_y),
                               pd.DataFrame({'log_MAF': glm_x,
                                             'bias': 2}),
                               family=sm.families.Gaussian(sm.families.links.log())
                               #                        family=sm.families.Gaussian(sm.families.links.identity())
                               )

            results = glm_model.fit(method="lbfgs")

            display(results.summary())

            display(pd.DataFrame({beta_label: results.params, 'p': results.pvalues}))

            # y_values = []

            c_x = np.log10(2 * (10.0**x_values) * (1 - (10.0 ** x_values)))
            y_values = np.power(math.e, results.params['log_MAF'] * c_x + 2 * results.params['bias'])

            # for e in list(x_values):
            #     c_maf = 10**e
            #     c_x = np.log10(2 * c_maf * (1 - c_maf))
            #     y_val = np.power(math.e, results.params['log_MAF'] * c_x + results.params['bias'])
            #     y_values.append(y_val)

            d_to_plot = pd.DataFrame({'x': x_values,
                                      'y': y_values}).sort_values('x')

            plt.plot(d_to_plot['x'], d_to_plot['y'], color=color)
        else:

            glm_x = []
            glm_y = []

            for m, e in zip(medians, edges_to_plot):
                if not np.isnan(m):
                    glm_y.append(m)
                    glm_x.append(e)

            glm_model = sm.GLM(np.array(glm_y), pd.DataFrame({'log_MAF': glm_x,
                                                              'bias': 2}),
                               family=sm.families.Gaussian(sm.families.links.log())
                               #                        family=sm.families.Gaussian(sm.families.links.identity())
                               )

            results = glm_model.fit(method="lbfgs")

            display(results.summary())

            display(pd.DataFrame({beta_label: results.params, 'p': results.pvalues}))

            d_to_plot = pd.DataFrame({'x': x_values,
                                      'y': np.power(math.e, results.params['log_MAF'] * x_values +
                                                    2 * results.params['bias'])}).sort_values('x')

            plt.plot(d_to_plot['x'], d_to_plot['y'], color=color)

        curve_parameters[label] = results.params.copy()

        all_values[label] = values
        all_medians[label] = medians

        plt.scatter(edges_to_plot,
                    medians,
                    s=[S_FACTOR * math.sqrt(len(values[k])) for k in edges_to_plot],
                    color=color,
                    label=label + ', n= ' + '{:,}'.format(n_variants) # + ', slope=%.3lf' % results.params['log_MAF']
                    )

    if not skip_GWAS_hits:
        if compute_means:
            echo('mean GWAS effect:', np.mean(np.abs(gwas_vars['beta/gwas/flipped'])))
        else:
            echo('median GWAS effect:', np.median(np.abs(gwas_vars['beta/gwas/flipped'])))

        gwas_data_for_plotting = pd.DataFrame({'log_MAF': np.log10(gwas_vars['MAF']),
                                               'beta': gwas_vars['beta/gwas/flipped'],
                                               'std': [1] * len(gwas_vars),
                                               GENE_NAME: gwas_vars[GENE_NAME]})
        if plot_all_GWAS_hits_together:

            mean_gwas_af = np.mean(gwas_data_for_plotting['log_MAF'])
            mean_gwas_effect = np.mean(gwas_data_for_plotting['beta'])
            echo('mean_gwas_af:', mean_gwas_af, ', mean_gwas_effect:', mean_gwas_effect)

            plt.scatter([mean_gwas_af],
                        [mean_gwas_effect],
                        s=[S_FACTOR * math.sqrt(len(gwas_data_for_plotting))],
                        color='black',
                        label='GWAS variants, n= ' + '{:,}'.format(len(gwas_data_for_plotting)),
                        marker='D')


        else:
            medians, edges_to_plot, values = compute_interval_medians(gwas_data_for_plotting,
                                                                      edges=edges,
                                                                      return_stds=False,
                                                                      maf_key='log_MAF',
                                                                      compute_means=compute_means)

            # all_data_for_plotting = pd.concat([all_data_for_plotting, gwas_data_for_plotting], ignore_index=True)

            echo('GWAS effects:', list(zip(edges_to_plot, medians)))
            # display(d_to_use.sort_values('MAF').head())

            variants_by_type['GWAS'] = gwas_vars

            # to_plot = pd.DataFrame({'e': edges_to_plot, 'm': medians})
            # to_plot = to_plot[to_plot['m'] != 0]
            plt.scatter(edges_to_plot,
                        medians,
                        s=[S_FACTOR * math.sqrt(len(values[k])) for k in edges_to_plot],
                        color='black',
                        label='GWAS variants, n= ' + '{:,}'.format(len(gwas_data_for_plotting)),
                        marker='D')

    # plt.plot(to_plot['e'], to_plot['m'], 'o', color='yellow', markersize=6, label='GWAS', markeredgecolor='black')

    for label in [LoF_label, DELETERIOUS_MISSENSE_label]:
        for x, y in zip(edges_to_plot, all_medians[label]):
            p_syn = scipy.stats.ttest_ind(all_values[label][x], all_values[SYNONYMOUS_label][x])[1]
            p_non_del = scipy.stats.ttest_ind(all_values[label][x], all_values[BENIGN_MISSENSE_label][x])[1]
            echo(x, y, len(all_values[label][x]), len(all_values[SYNONYMOUS_label][x]),
                 len(all_values[BENIGN_MISSENSE_label][x]), p_syn, p_non_del)

            # if p_syn < 0.05:
            #     plt.text(x+.05, y, '*', color='green', fontsize=16, fontweight='extra bold' if p_syn < 1e-10 else 'ultralight')
            #
            # if p_non_del < 0.05:
            #     plt.text(x+.15, y, '*', color='blue', fontsize=16, fontweight='extra bold' if p_syn < 1e-10 else 'ultralight')

    ylim = plt.ylim()
    echo('ylim:', ylim)
    # sns.set_style("whitegrid", {'axes.grid': False})

    plt.xlabel('Allele Frequency')
    plt.ylabel('Average Effect Size (std. units)')

    lgnd = plt.legend(labelspacing=1)
    # change the marker size manually for both lines
    for lgnd_idx in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[lgnd_idx]._sizes = [70]

    sns.despine()

    # ax = fig.axes[0]
    # plt.draw()
    #
    # labels = [int(item.get_text()) for item in ax.get_xticklabels()]
    # echo(labels)
    # for i in range(len(labels)):
    #     l = labels[i]
    #     if l <= -3:
    #         labels[i] = '$10^%d$' % l
    #     elif l == -2:
    #         labels[i] = '0.01'
    #     elif l == -1:
    #         labels[i] = '0.1'
    # echo(labels)
    # ax.set_xticklabels(labels)

    if out_prefix is not None:
        echo('Saving figures to:', out_prefix)
        plt.savefig(out_prefix + '.svg')
        plt.savefig(out_prefix + '.png', dpi=300)

    plt.show()

    return curve_parameters
