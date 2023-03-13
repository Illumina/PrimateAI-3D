import pandas as pd

from jutils import *
import scipy

REPLICATES = 'replicates'
def eval_scores_against_replicates(gene_phenotype_pairs,
                                   preselected_gene_variants,
                                   N_RANDS=1000,
                                   rv_dir=ROOT_PATH + f'/ukbiobank/data/molecular_phenotypes.17_SEPT_2019/phenotypes_for_GWAS.ukb200k/quantitative_phenotypes.results',
                                   rv_suffix=f'.all_ethnicities.both.dynamic_med_corrected.IRNT.ukb200k_unrelated_all_ethnicities.maf_0.001.common_vars_regressed.all_vars_regressed.score_PAI3D_percentile.adaptive_FDR.ph_data.pickle',
                                   corr_func=scipy.stats.spearmanr,
                                   score_labels=None,
                                   ignore_missing_genes=True,
                                   min_n_carriers=2,
                                   min_n_variants_per_gene=2,
                                   var_types=None,
                                   include_random_scores=False,
                                   include_synonymous_variants=False,
                                   invert_sign_based_on_lof=False,
                                   min_AF=0.001
                                   ):

    echo('[eval_scores_against_replicates]!')

    echo('gene_phenotype_pairs:', len(gene_phenotype_pairs))

    echo('invert_sign_based_on_lof:', invert_sign_based_on_lof)
    gene_variants = []

    gene_variants_syn = None
    res_syn = None

    if include_synonymous_variants:
        echo('Testing same-variant carriers for synonymous variants')
        res_syn, gene_variants_syn = eval_scores_against_replicates(gene_phenotype_pairs,
                                                                    preselected_gene_variants,
                                                                    N_RANDS=N_RANDS,
                                                                    rv_dir=rv_dir,
                                                                    rv_suffix=rv_suffix,
                                                                    corr_func=corr_func,
                                                                    score_labels=[],
                                                                    ignore_missing_genes=ignore_missing_genes,
                                                                    min_n_carriers=min_n_carriers,
                                                                    min_n_variants_per_gene=min_n_variants_per_gene,
                                                                    var_types=[VCF_SYNONYMOUS_VARIANT],
                                                                    include_random_scores=False,
                                                                    include_synonymous_variants=False,
                                                                    invert_sign_based_on_lof=invert_sign_based_on_lof
                                                                    )

        res_syn['score'] = 'Same synonymous variant carriers'

    RANDOM_SCORE = 'random_score'
    if include_random_scores:
        score_labels += [RANDOM_SCORE]

    echo('score_labels:', score_labels)
    agr_stats = {'score': [], 'r2': []}

    if var_types is None:
        var_types = [VCF_MISSENSE_VARIANT]
    echo('var_types:', var_types)

    res = {'phenotype': [],
           GENE_NAME: [],
           'score': [],
           # 'enr': [],
           'score|pvalue': [],
           'score|r': [],
           'score|r2': [],

           'repl|pvalue': [],
           'repl|r': [],
           'repl|r2': [],

           'n_variants': []
           }

    for ph_name in sorted(gene_phenotype_pairs['phenotype'].unique()):

        ph_genes = gene_phenotype_pairs[gene_phenotype_pairs['phenotype'] == ph_name]

        echo(ph_name, len(ph_genes))

        ph_label = ph_name + '.all_ethnicities.both.dynamic_med_corrected.IRNT'

        d = pd.read_pickle(rv_dir + f'/{ph_name}/{ph_name}' + rv_suffix)
        sid_to_ph = dict((sid, ph) for sid, ph in zip(d[SAMPLE_ID], d[ph_label]))

        for _, gpp in ph_genes.iterrows():
            gene_name = gpp[GENE_NAME]

            corr_sign = 1
            if invert_sign_based_on_lof:
                corr_sign = 1 if gpp['ALL/ptv/carrier/beta'] > 0 else -1

            g_vars = preselected_gene_variants[(preselected_gene_variants[GENE_NAME] == gene_name) &
                                               (preselected_gene_variants[VCF_CONSEQUENCE].isin(var_types)) &
                                               (preselected_gene_variants[VCF_AF] <= min_AF)
                                               ].sort_values(VCF_AC, ascending=False).copy()

            n_vars = len(g_vars[g_vars[VCF_AC] >= min_n_carriers])

            if include_random_scores:
                g_vars[RANDOM_SCORE] = np.random.rand(len(g_vars))

            cur_g_vars = g_vars[~g_vars[score_labels].isnull().any(axis=1)]

            scores_to_skip = []

            if ignore_missing_genes:
                if len(cur_g_vars) < min_n_variants_per_gene:
                    for sl in score_labels:
                        if np.sum(~g_vars[sl].isnull()) < min_n_variants_per_gene:
                            echo('Skipping score,', sl, 'for', ph_name, gene_name)
                            scores_to_skip.append(sl)

            g_vars = g_vars[~g_vars[[sl for sl in score_labels if sl not in scores_to_skip]].isnull().any(axis=1)]

            non_singletons = g_vars[g_vars[VCF_AC] >= min_n_carriers].copy()
            all_samples = []
            all_vars = []

            for _, v in non_singletons.iterrows():
                all_s = [s for s in v[ALL_SAMPLES].split(',') if s in sid_to_ph]
                if len(all_s) >= min_n_carriers:
                    all_samples.append(all_s)
                    all_vars.append(v)

                    vv = v.copy()

                    ph_vals = [sid_to_ph[s] for s in all_s]
                    vv['Mean phenotype value'] = np.mean(ph_vals)
                    vv['Phenotype values'] = ph_vals
                    vv['Phenotype'] = ph_name

                    vv[VCF_AC] = len(ph_vals)
                    vv[ALL_SAMPLES] = ','.join(all_s)

                    gene_variants.append(vv)

            #             echo(ph_name, gene_name, len(all_vars), n_vars)

            if len(all_vars) < min_n_variants_per_gene:
                echo('Too few variants:', ph_name, gene_name, len(all_vars), 'out of', len(g_vars), n_vars)
                continue

            for k in range(N_RANDS):

                test_set = [all_s[random.randint(0, len(all_s) - 1)] for all_s in all_samples]
                test_set_values = [sid_to_ph[s] for s in test_set]

                training_set = [[s for s in all_s if s != ts] for all_s, ts in zip(all_samples, test_set)]
                training_set_values = [np.mean([sid_to_ph[s] for s in sids]) for sids in training_set]

                r_repl, p_repl = corr_func(training_set_values, test_set_values)

                # if invert_sign_based_on_lof:
                #     r_repl *= corr_sign

                # if r_repl == 0:
                #     echo('correlation of 0 between replicates, skipping permutation!', gene_name, ph_name)
                #     continue

                agr_stats['score'].append(REPLICATES)
                agr_stats['r2'].append(r_repl ** 2)

                for score_label in score_labels + [REPLICATES]:
                    if score_label in scores_to_skip:
                        continue

                    #             echo(score_label, 'scores:', len(scores))

                    if score_label == REPLICATES:
                        scores = training_set_values
                    else:
                        scores = [v[score_label] for v in all_vars]

                    res['phenotype'].append(ph_name)
                    res[GENE_NAME].append(gene_name)
                    res['score'].append(score_label)
                    res['repl|pvalue'].append(p_repl)
                    res['repl|r'].append(r_repl)
                    res['repl|r2'].append(r_repl ** 2)

                    r, p = corr_func(scores, test_set_values)

                    if invert_sign_based_on_lof and score_label != REPLICATES:
                        r *= corr_sign

                    if np.isnan(r):
                        #                         echo('[WARNING] corr_func returned NaNs for', gene_name, score_label)
                        r = 0
                        p = 1

                    res['score|pvalue'].append(p)
                    res['score|r'].append(r)
                    res['score|r2'].append(r ** 2)

                    # res['enr'].append(np.square(r / r_repl))
                    res['n_variants'].append(len(scores))

                    for k in gpp.keys():

                        if k + '/repl_info' not in res:
                            res[k + '/repl_info'] = []

                        res[k + '/repl_info'].append(gpp[k])

                    agr_stats['score'].append(score_label)
                    agr_stats['r2'].append(r ** 2)

    res = pd.DataFrame(res)
    if include_synonymous_variants:
        res = pd.concat([res, res_syn], ignore_index=True)

    # agr_stats = pd.DataFrame(agr_stats)
    # echo(gene_variants)
    gene_variants = pd.DataFrame(gene_variants)
    if gene_variants_syn is not None:
        gene_variants = pd.concat([gene_variants, gene_variants_syn], ignore_index=True, sort=True)

    return res, gene_variants

def eval_ethnicities_against_replicates(gene_phenotype_pairs,
                                        preselected_gene_variants,
                                        N_RANDS=1000,
                                        rv_dir=ROOT_PATH + f'/ukbiobank/data/molecular_phenotypes.17_SEPT_2019/phenotypes_for_GWAS.ukb200k/quantitative_phenotypes.results',
                                        rv_suffix=f'.all_ethnicities.both.dynamic_med_corrected.IRNT.ukb200k_unrelated_all_ethnicities.maf_0.001.common_vars_regressed.all_vars_regressed.score_PAI3D_percentile.adaptive_FDR.ph_data.pickle',
                                        corr_func=scipy.stats.spearmanr,
                                        score_labels=None,
                                        ignore_missing_genes=True,
                                        min_n_carriers=2,
                                        min_n_variants_per_gene=2,
                                        var_types=None,
                                        include_random_scores=False,
                                        include_synonymous_variants=False,
                                        invert_sign_based_on_lof=False,
                                        min_AF=0.001,
                                        ethn_samples_labels=None
                                        ):

    echo('[eval_scores_against_replicates]!')

    echo('gene_phenotype_pairs:', len(gene_phenotype_pairs))

    echo('invert_sign_based_on_lof:', invert_sign_based_on_lof)
    gene_variants = []

    gene_variants_syn = None
    res_syn = None

    if include_synonymous_variants:
        echo('Testing same-variant carriers for synonymous variants')
        res_syn, gene_variants_syn = eval_ethnicities_against_replicates(gene_phenotype_pairs,
                                                                         preselected_gene_variants,
                                                                         N_RANDS=N_RANDS,
                                                                         rv_dir=rv_dir,
                                                                         rv_suffix=rv_suffix,
                                                                         corr_func=corr_func,
                                                                         score_labels=[],
                                                                         ignore_missing_genes=ignore_missing_genes,
                                                                         min_n_carriers=min_n_carriers,
                                                                         min_n_variants_per_gene=min_n_variants_per_gene,
                                                                         var_types=[VCF_SYNONYMOUS_VARIANT],
                                                                         include_random_scores=False,
                                                                         include_synonymous_variants=False,
                                                                         invert_sign_based_on_lof=invert_sign_based_on_lof,
                                                                         ethn_samples_labels=ethn_samples_labels
                                                                         )

        res_syn['score'] = 'Same synonymous variant carriers'

    RANDOM_SCORE = 'random_score'
    if include_random_scores:
        score_labels += [RANDOM_SCORE]

    echo('score_labels:', score_labels)
    agr_stats = {'score': [], 'r2': []}

    if var_types is None:
        var_types = [VCF_MISSENSE_VARIANT]
    echo('var_types:', var_types)

    res = {'phenotype': [],
           GENE_NAME: [],
           'score': [],
           # 'enr': [],
           'score|pvalue': [],
           'score|r': [],
           'score|r2': [],

           'repl|pvalue': [],
           'repl|r': [],
           'repl|r2': [],

           'n_variants': [],
           # 'ethnicity': []
           }

    for ph_name in sorted(gene_phenotype_pairs['phenotype'].unique()):

        ph_genes = gene_phenotype_pairs[gene_phenotype_pairs['phenotype'] == ph_name]

        echo(ph_name, len(ph_genes))

        ph_label = ph_name + '.all_ethnicities.both.dynamic_med_corrected.IRNT'

        d = pd.read_pickle(rv_dir + f'/{ph_name}/{ph_name}' + rv_suffix)
        sid_to_ph = dict((sid, ph) for sid, ph in zip(d[SAMPLE_ID], d[ph_label]))

        for _, gpp in ph_genes.iterrows():
            gene_name = gpp[GENE_NAME]

            corr_sign = 1
            if invert_sign_based_on_lof:
                corr_sign = 1 if gpp['ALL/ptv/carrier/beta'] > 0 else -1

            for ethn_ac, ethn_samples, ethn in ethn_samples_labels:
                g_vars = preselected_gene_variants[(preselected_gene_variants[GENE_NAME] == gene_name) &
                                                   (preselected_gene_variants[VCF_CONSEQUENCE].isin(var_types)) &
                                                   (preselected_gene_variants[VCF_AF] <= min_AF)
                                                   ].sort_values(ethn_ac, ascending=False).copy()

                n_vars = len(g_vars[g_vars[ethn_ac] >= min_n_carriers])

                if include_random_scores:
                    g_vars[RANDOM_SCORE] = np.random.rand(len(g_vars))

                cur_g_vars = g_vars[~g_vars[score_labels].isnull().any(axis=1)]

                scores_to_skip = []

                if ignore_missing_genes:
                    if len(cur_g_vars) < min_n_variants_per_gene:
                        for sl in score_labels:
                            if np.sum(~g_vars[sl].isnull()) < min_n_variants_per_gene:
                                echo('Skipping score,', sl, 'for', ph_name, gene_name)
                                scores_to_skip.append(sl)

                g_vars = g_vars[~g_vars[[sl for sl in score_labels if sl not in scores_to_skip]].isnull().any(axis=1)]

                non_singletons = g_vars[g_vars[ethn_ac] >= min_n_carriers].copy()
                all_samples = []
                all_vars = []

                for _, v in non_singletons.iterrows():
                    all_s = [s for s in v[ethn_samples].split(',') if s in sid_to_ph]
                    if len(all_s) >= min_n_carriers:
                        all_samples.append(all_s)
                        all_vars.append(v)

                        vv = v.copy()

                        ph_vals = [sid_to_ph[s] for s in all_s]
                        vv['Mean phenotype value'] = np.mean(ph_vals)
                        vv['Phenotype values'] = ph_vals
                        vv['Phenotype'] = ph_name

                        vv[ethn_ac] = len(ph_vals)

                        gene_variants.append(vv)

                #             echo(ph_name, gene_name, len(all_vars), n_vars)

                if len(all_vars) < min_n_variants_per_gene:
                    echo('Too few variants:', ph_name, gene_name, len(all_vars), 'out of', len(g_vars), n_vars)
                    continue

                for k in range(N_RANDS):

                    test_set = [all_s[random.randint(0, len(all_s) - 1)] for all_s in all_samples]
                    test_set_values = [sid_to_ph[s] for s in test_set]

                    training_set = [[s for s in all_s if s != ts] for all_s, ts in zip(all_samples, test_set)]
                    training_set_values = [np.mean([sid_to_ph[s] for s in sids]) for sids in training_set]

                    r_repl, p_repl = corr_func(training_set_values, test_set_values)

                    # if invert_sign_based_on_lof:
                    #     r_repl *= corr_sign

                    # if r_repl == 0:
                    #     echo('correlation of 0 between replicates, skipping permutation!', gene_name, ph_name)
                    #     continue

                    agr_stats['score'].append(REPLICATES)
                    agr_stats['r2'].append(r_repl ** 2)

                    for score_label in score_labels + [REPLICATES]:
                        if score_label in scores_to_skip:
                            continue

                        #             echo(score_label, 'scores:', len(scores))

                        if score_label == REPLICATES:
                            scores = training_set_values
                        else:
                            scores = [v[score_label] for v in all_vars]

                        res['phenotype'].append(ph_name)
                        res[GENE_NAME].append(gene_name)
                        res['score'].append(score_label + '/' + ethn)
                        res['repl|pvalue'].append(p_repl)
                        res['repl|r'].append(r_repl)
                        res['repl|r2'].append(r_repl ** 2)

                        r, p = corr_func(scores, test_set_values)

                        if invert_sign_based_on_lof and score_label != REPLICATES:
                            r *= corr_sign

                        if np.isnan(r):
                            #                         echo('[WARNING] corr_func returned NaNs for', gene_name, score_label)
                            r = 0
                            p = 1

                        res['score|pvalue'].append(p)
                        res['score|r'].append(r)
                        res['score|r2'].append(r ** 2)

                        # res['enr'].append(np.square(r / r_repl))
                        res['n_variants'].append(len(scores))

                        for k in gpp.keys():

                            if k + '/repl_info' not in res:
                                res[k + '/repl_info'] = []

                            res[k + '/repl_info'].append(gpp[k])

                        agr_stats['score'].append(score_label)
                        agr_stats['r2'].append(r ** 2)

    res = pd.DataFrame(res)
    if include_synonymous_variants:
        res = pd.concat([res, res_syn], ignore_index=True)

    # agr_stats = pd.DataFrame(agr_stats)
    # echo(gene_variants)
    gene_variants = pd.DataFrame(gene_variants)
    if gene_variants_syn is not None:
        gene_variants = pd.concat([gene_variants, gene_variants_syn], ignore_index=True, sort=True)

    return res, gene_variants


def eval_scores_on_all_variants(gene_phenotype_pairs,
                                preselected_gene_variants,
                                rv_dir=ROOT_PATH + f'/ukbiobank/data/molecular_phenotypes.17_SEPT_2019/phenotypes_for_GWAS.ukb200k/quantitative_phenotypes.results/',
                                rv_suffix=f'.all_ethnicities.both.dynamic_med_corrected.IRNT.ukb200k_unrelated_all_ethnicities.maf_0.001.common_vars_regressed.all_vars_regressed.score_PAI3D_percentile.adaptive_FDR.ph_data.pickle',
                                corr_func=scipy.stats.spearmanr,
                                max_AC=None,
                                score_labels=None,
                                ignore_missing_genes=True,
                                min_vars_per_gene=2
                                ):
    echo('gene_phenotype_pairs:', len(gene_phenotype_pairs))
    gene_variants = []

    res = {'phenotype': [],
           GENE_NAME: [],

           'score': [],

           'variant_level/pvalue': [],
           'variant_level/r': [],
           'variant_level/r2': [],

           'carrier_level/pvalue': [],
           'carrier_level/r': [],
           'carrier_level/r2': [],

           'n_variants': [],
           'n_carriers': [],

           }

    for ph_name in sorted(gene_phenotype_pairs['phenotype'].unique()):

        ph_genes = gene_phenotype_pairs[gene_phenotype_pairs['phenotype'] == ph_name]

        echo(ph_name, len(ph_genes))

        ph_label = ph_name + '.all_ethnicities.both.dynamic_med_corrected.IRNT'

        #     d = ukb_phenotypes[[SAMPLE_ID, ph_label]].dropna()
        #     sid_to_ph = dict((sid, ph) for sid, ph in zip(d[SAMPLE_ID], d[ph_label]))

        d = pd.read_pickle(rv_dir + f'/{ph_name}/{ph_name}' + rv_suffix)
        sid_to_ph = dict((sid, ph) for sid, ph in zip(d[SAMPLE_ID], d[ph_label]))

        for _, gpp in ph_genes.iterrows():
            gene_name = gpp[GENE_NAME]

            # score_label = PRIMATEAI_SCORE

            var_type = [VCF_MISSENSE_VARIANT]

            g_vars = preselected_gene_variants[(preselected_gene_variants[GENE_NAME] == gene_name) &
                                               (preselected_gene_variants[VCF_CONSEQUENCE].isin(var_type)) &
                                               (preselected_gene_variants[VCF_AF] <= 0.001)
                                               ].sort_values(VCF_AC, ascending=False)

            if max_AC is not None:
                g_vars = g_vars[g_vars[VCF_AC] <= max_AC].copy()

            cur_g_vars = g_vars[~g_vars[score_labels].isnull().any(axis=1)]

            scores_to_skip = []

            if ignore_missing_genes:
                if len(cur_g_vars) < 2:
                    for sl in score_labels:
                        if np.sum(~g_vars[sl].isnull()) < 2:
                            echo('Skipping score,', sl, 'for', ph_name, gene_name)
                            scores_to_skip.append(sl)

            g_vars = g_vars[~g_vars[[sl for sl in score_labels if sl not in scores_to_skip]].isnull().any(axis=1)]

            if len(g_vars) < min_vars_per_gene:
                echo('Too few variants:', ph_name, gene_name, len(g_vars), 'out of', len(all_vars))
                continue

            all_carriers = []
            all_variant_carriers = []
            all_vars = []
            c2v = {}
            for _, v in g_vars.iterrows():

                all_s = [s for s in v[ALL_SAMPLES].split(',') if s in sid_to_ph]
                if len(all_s) == 0:
                    continue

                all_vars.append(v)

                for s in all_s:
                    all_carriers.append(s)
                    c2v[s] = v

                all_variant_carriers.append(all_s)

                vv = v.copy()

                ph_vals = [sid_to_ph[s] for s in all_s]
                vv['Mean phenotype value'] = np.mean(ph_vals)
                vv['Phenotype values'] = ph_vals
                vv['Phenotype'] = ph_name

                vv[VCF_AC] = len(ph_vals)
                vv[ALL_SAMPLES] = ','.join(all_s)

                gene_variants.append(vv)

            if len(all_variant_carriers) < 2:
                continue

            carrier_level_values = [sid_to_ph[sid] for sid in all_carriers]
            variant_level_values = [np.mean([sid_to_ph[sid] for sid in all_s]) for all_s in all_variant_carriers]

            #             echo(len(variant_level_values), variant_level_values)
            for score_label in score_labels:
                if score_label in scores_to_skip:
                    continue

                res['phenotype'].append(ph_name)
                res[GENE_NAME].append(gene_name)
                res['score'].append(score_label)

                variant_level_scores = [v[score_label] for v in all_vars]
                #                 echo(len(variant_level_scores), variant_level_scores)
                carrier_level_scores = [c2v[sid][score_label] for sid in all_carriers]

                r, p = corr_func(variant_level_scores, variant_level_values)
                if np.isnan(r):
                    echo('[WARNING] corr_func returned NaNs for', gene_name, score_label)
                    r = 0
                    p = 1

                res['variant_level/pvalue'].append(p)
                res['variant_level/r'].append(r)
                res['variant_level/r2'].append(r ** 2)

                res['n_variants'].append(len(variant_level_scores))

                r, p = corr_func(carrier_level_scores, carrier_level_values)
                if np.isnan(r):
                    echo('[WARNING] corr_func returned NaNs for', gene_name, score_label)
                    r = 0
                    p = 1

                res['carrier_level/pvalue'].append(p)
                res['carrier_level/r'].append(r)
                res['carrier_level/r2'].append(r ** 2)

                res['n_carriers'].append(len(carrier_level_scores))

                for k in gpp.keys():

                    if k + '/repl_info' not in res:
                        res[k + '/repl_info'] = []

                    res[k + '/repl_info'].append(gpp[k])

    gene_variants = pd.DataFrame(gene_variants)

    res = pd.DataFrame(res)

    return res, gene_variants


def eval_ethnicities_on_all_variants(gene_phenotype_pairs,
                                     preselected_gene_variants,
                                     rv_dir=ROOT_PATH + f'/ukbiobank/data/molecular_phenotypes.17_SEPT_2019/phenotypes_for_GWAS.ukb200k/quantitative_phenotypes.results/',
                                     rv_suffix=f'.all_ethnicities.both.dynamic_med_corrected.IRNT.ukb200k_unrelated_all_ethnicities.maf_0.001.common_vars_regressed.all_vars_regressed.score_PAI3D_percentile.adaptive_FDR.ph_data.pickle',
                                     corr_func=scipy.stats.spearmanr,
                                     max_AC=None,
                                     score_labels=None,
                                     ignore_missing_genes=True,
                                     ethn_samples_labels=None,
                                     min_vars_per_gene=2,
                                     max_AF=0.001
                                     ):

    echo('[eval_ethnicities_on_all_variants]@')
    echo('gene_phenotype_pairs:', len(gene_phenotype_pairs), ', min_vars_per_gene:', min_vars_per_gene)
    gene_variants = []

    res = {'phenotype': [],
           GENE_NAME: [],

           'score': [],

           'variant_level/pvalue': [],
           'variant_level/r': [],
           'variant_level/r2': [],

           'carrier_level/pvalue': [],
           'carrier_level/r': [],
           'carrier_level/r2': [],

           'n_variants': [],
           'n_carriers': [],

           }

    for ph_name in sorted(gene_phenotype_pairs['phenotype'].unique()):

        ph_genes = gene_phenotype_pairs[gene_phenotype_pairs['phenotype'] == ph_name]

        echo(ph_name, len(ph_genes))

        ph_label = ph_name + '.all_ethnicities.both.dynamic_med_corrected.IRNT'

        #     d = ukb_phenotypes[[SAMPLE_ID, ph_label]].dropna()
        #     sid_to_ph = dict((sid, ph) for sid, ph in zip(d[SAMPLE_ID], d[ph_label]))

        d = pd.read_pickle(rv_dir + f'/{ph_name}/{ph_name}' + rv_suffix)
        sid_to_ph = dict((sid, ph) for sid, ph in zip(d[SAMPLE_ID], d[ph_label]))

        for _, gpp in ph_genes.iterrows():
            gene_name = gpp[GENE_NAME]

            var_type = [VCF_MISSENSE_VARIANT]

            for ethn_ac, ethn_samples, ethn in ethn_samples_labels:

                g_vars = preselected_gene_variants[(preselected_gene_variants[GENE_NAME] == gene_name) &
                                                   (preselected_gene_variants[VCF_CONSEQUENCE].isin(var_type)) &
                                                   (preselected_gene_variants[VCF_AF] <= max_AF) &
                                                   (preselected_gene_variants[ethn_ac] > 0)
                                                   ].sort_values(ethn_ac, ascending=False)

                if max_AC is not None:
                    g_vars = g_vars[g_vars[ethn_ac] <= max_AC].copy()

                cur_g_vars = g_vars[~g_vars[score_labels].isnull().any(axis=1)]

                scores_to_skip = []

                if ignore_missing_genes:
                    if len(cur_g_vars) < min_vars_per_gene:
                        for sl in score_labels:
                            if np.sum(~g_vars[sl].isnull()) < min_vars_per_gene:
                                echo('Skipping score,', sl, 'for', ph_name, gene_name)
                                scores_to_skip.append(sl)

                g_vars = g_vars[~g_vars[[sl for sl in score_labels if sl not in scores_to_skip]].isnull().any(axis=1)]

                if len(g_vars) < min_vars_per_gene:
                    echo('Too few variants:', ph_name, gene_name, len(g_vars))
                    continue

                all_carriers = []
                all_variant_carriers = []
                all_vars = []
                c2v = {}
                n_survived_vars = 0

                for _, v in g_vars.iterrows():

                    all_s = [s for s in v[ethn_samples].split(',') if s in sid_to_ph]
                    if len(all_s) == 0:
                        continue

                    all_vars.append(v)

                    for s in all_s:
                        all_carriers.append(s)
                        c2v[s] = v

                    all_variant_carriers.append(all_s)
                    n_survived_vars += 1

                    vv = v.copy()

                    ph_vals = [sid_to_ph[s] for s in all_s]
                    vv['Mean phenotype value'] = np.mean(ph_vals)
                    vv['Phenotype values'] = ph_vals
                    vv['Phenotype'] = ph_name

                    vv[VCF_AC + '/' + ethn] = len(ph_vals)
                    vv[ALL_SAMPLES] = ','.join(all_s)
                    # if vv[VARID_REF_ALT] in [_v[VARID_REF_ALT] for _v in gene_variants]:
                    #     echo('Variant seen:', vv)
                    gene_variants.append(vv)

                if len(all_variant_carriers) < min_vars_per_gene:
                    echo('Too few carriers:', ph_name, gene_name, len(all_variant_carriers))
                    continue

                if n_survived_vars < min_vars_per_gene:
                    echo('Too few variants:', ph_name, gene_name, n_survived_vars, 'out of', len(all_vars))
                    continue

                carrier_level_values = [sid_to_ph[sid] for sid in all_carriers]
                variant_level_values = [np.mean([sid_to_ph[sid] for sid in all_s]) for all_s in all_variant_carriers]

                #             echo(len(variant_level_values), variant_level_values)
                for score_label in score_labels:
                    if score_label in scores_to_skip:
                        continue

                    res['phenotype'].append(ph_name)
                    res[GENE_NAME].append(gene_name)
                    res['score'].append(score_label + '/' + ethn)

                    variant_level_scores = [v[score_label] for v in all_vars]
                    #                 echo(len(variant_level_scores), variant_level_scores)
                    carrier_level_scores = [c2v[sid][score_label] for sid in all_carriers]

                    r, p = corr_func(variant_level_scores, variant_level_values)

                    if np.isnan(r):
                        echo('[WARNING] corr_func returned NaNs for', gene_name, score_label)
                        r = 0
                        p = 1

                    res['variant_level/pvalue'].append(p)
                    res['variant_level/r'].append(r)
                    res['variant_level/r2'].append(r ** 2)

                    res['n_variants'].append(len(variant_level_scores))

                    r, p = corr_func(carrier_level_scores, carrier_level_values)

                    if np.isnan(r):
                        echo('[WARNING] corr_func returned NaNs for', gene_name, score_label)
                        r = 0
                        p = 1

                    res['carrier_level/pvalue'].append(p)
                    res['carrier_level/r'].append(r)
                    res['carrier_level/r2'].append(r ** 2)

                    res['n_carriers'].append(len(carrier_level_scores))

                    for k in gpp.keys():

                        if k + '/repl_info' not in res:
                            res[k + '/repl_info'] = []

                        res[k + '/repl_info'].append(gpp[k])

    res = pd.DataFrame(res)

    gene_variants = pd.DataFrame(gene_variants)

    return res, gene_variants


def compare_every_pair_of_scores(gene_list_for_primateAI_evaluation, metric_label='variant_level/r2',
                                 compare_func=scipy.stats.ranksums):
    x = gene_list_for_primateAI_evaluation.sort_values(GENE_NAME)

    #     echo(x.shape, len(set(x[GENE_NAME])))

    rs_s = {'score': []}
    rs_p = {'score': []}

    aggr_stats = {'score': [], 'stat': [], 'pvalue': []}

    for s1 in sorted(x['score'].unique()):
        rs_s[s1] = []
        rs_p[s1] = []

        s1_vals = x[x['score'] == s1][metric_label]

        rs_s['score'].append(s1)
        rs_p['score'].append(s1)

        aggr_stats['score'].append(s1)
        if compare_func is scipy.stats.wilcoxon:

            s1_s, s1_p = scipy.stats.ranksums(s1_vals, x[x['score'] != s1][metric_label])
        else:
            s1_s, s1_p = compare_func(s1_vals, x[x['score'] != s1][metric_label])

        aggr_stats['stat'].append(s1_s)
        aggr_stats['pvalue'].append(s1_p)

        for s2 in sorted(x['score'].unique()):
            s2_vals = x[x['score'] == s2][metric_label]

            c_rs_s, c_rs_p = compare_func(s1_vals, s2_vals)
            rs_s[s1].append(c_rs_s)
            rs_p[s1].append(c_rs_p)

    rs_p = pd.DataFrame(rs_p)
    rs_s = pd.DataFrame(rs_s)
    aggr_stats = pd.DataFrame(aggr_stats).sort_values('pvalue')

    return rs_s, rs_p, aggr_stats