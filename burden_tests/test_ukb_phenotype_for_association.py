import matplotlib
matplotlib.use('Agg')

import copy

import argparse
import pprint

# from jutils import *

import pickle
import sys
import pandas as pd
import time

# from ukb_analysis import *
from rare_variants_test_utils import *

pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", 50)

UKB_DATA_PATH = ROOT_PATH + '/ukbiobank/data/'


def read_rare_variants(gene_info_fname,
                       pathogenicity_score_file,
                       pathogenicity_score_label,
                       max_maf,
                       max_AC,
                       max_missingness,
                       gnomad_coverage,
                       gnomad_popmax_af,
                       genes_to_test,
                       samples_with_phenotypes,
                       decode_sample_id_fields=False,
                       white_listed_exons_fname=None,
                       gencode_fname=None,
                       exclude_variants=None):

    echo('Reading variant info from disk:', gene_info_fname)
    echo('samples_with_phenotypes:', len(samples_with_phenotypes))

    rare_variants_full = None
    if gene_info_fname.endswith('.pickle'):
        rare_variants_full = pd.read_pickle(gene_info_fname)

        if ALL_SAMPLES not in list(rare_variants_full):
            rare_variants_full[ALL_SAMPLES] = (rare_variants_full[HOMOZYGOTES] + ',' + rare_variants_full[HETEROZYGOTES]).str.strip(',')

    elif gene_info_fname.endswith('.db'):
        rare_variants_full = get_ukb_exome_variants_for_genes_from_sqlite(exome_db=gene_info_fname,
                                                                          decode_sample_id_fields=decode_sample_id_fields)
    else:
        echo('Unknown format for variants info:', gene_info_fname)
        exit(1)

    echo('Total variants:', rare_variants_full.shape)
    if gencode_fname is not None:
        echo('Filtering by gencode file:', gencode_fname)
        gencode_genes = set(pd.read_csv(gencode_fname, sep='\t')[GENE_NAME])
        rare_variants_full = rare_variants_full[rare_variants_full[GENE_NAME].isin(gencode_genes)].copy()
        echo('Filtered by gencode:', rare_variants_full.shape, ', from', len(rare_variants_full[GENE_NAME].unique()), ' genes')

    if exclude_variants is not None:
        echo('Excluding variants from:', exclude_variants)
        exclude_variants_df = pd.read_pickle(exclude_variants)

        echo('Variants to exclude:', len(exclude_variants_df))
        echo('Before excluding, rare_variants_full:', rare_variants_full.shape)
        rare_variants_full = rare_variants_full[~rare_variants_full[VARID_REF_ALT].isin(set(exclude_variants_df[VARID_REF_ALT]))].copy()
        echo('After excluding, rare_variants_full:', rare_variants_full.shape)

    missing_varids = np.sum(rare_variants_full[VARID_REF_ALT].isnull())
    if missing_varids > 0:
        echo(f'[WARNING] {missing_varids} missing varids detected. Regenerating varids!')
        rare_variants_full[VARID_REF_ALT] = rare_variants_full[VCF_CHROM] + ':' + rare_variants_full[VCF_POS].astype(str) + ':' + rare_variants_full[VCF_REF] + ':' + rare_variants_full[VCF_ALT]

    if white_listed_exons_fname is not None:
        echo('Filtering variants in white listed exons in:', white_listed_exons_fname)
        white_listed_exons = pd.read_pickle(white_listed_exons_fname)

        echo('before filtering:', len(rare_variants_full))

        rare_variants_full = pd.merge(rare_variants_full,
                                      white_listed_exons[[VCF_CHROM, START, END, GENE_NAME]],
                                      on=[VCF_CHROM, GENE_NAME])

        rare_variants_full = rare_variants_full[(rare_variants_full[VCF_POS] >= rare_variants_full[START] - 25) &
                                                (rare_variants_full[VCF_POS] <= rare_variants_full[END] + 25)]

        rare_variants_full = rare_variants_full.drop_duplicates(subset=[VARID_REF_ALT, GENE_NAME, VCF_CONSEQUENCE])
        echo('after filtering:', len(rare_variants_full))

    if pathogenicity_score_file:
        echo('Reading pathogenicity scores from:', pathogenicity_score_file)
        if pathogenicity_score_file.endswith('.pickle'):
            pathogenicity_scores = pd.read_pickle(pathogenicity_score_file)
        else:
            pathogenicity_scores = pd.read_csv(pathogenicity_score_file,
                                               sep='\t',
                                               dtype={VCF_CHROM: str,
                                                      VCF_POS: int,
                                                      VCF_REF: str,
                                                      VCF_ALT: str,
                                                      pathogenicity_score_label: float})

        echo('Merging rare_variants_full with pathogenicity_scores')
        on_cols = [VCF_CHROM, VCF_POS, VCF_REF, VCF_ALT]
        if GENE_NAME in list(pathogenicity_scores):
            on_cols += [GENE_NAME]

        rare_variants_full = pd.merge(rare_variants_full,
                                      pathogenicity_scores,
                                      on=on_cols,
                                      how='left',
                                      suffixes=['', '/pathogenicity_scores'])

        echo('pathogenicity_scores:', pathogenicity_scores.shape,
             ', rare_variants_full:', rare_variants_full.shape,
             ', variants with score:', np.sum(~rare_variants_full[pathogenicity_score_label].isnull()),
             ', out of', np.sum(rare_variants_full[VCF_CONSEQUENCE] == VCF_MISSENSE_VARIANT), ' missense variants')

        del pathogenicity_scores

    echo('Filtering below MAF:', max_maf)
    rare_variants_full = rare_variants_full[(rare_variants_full[VCF_AF] <= max_maf) |
                                            (rare_variants_full[VCF_AF] >= 1 - max_maf)]

    echo('total variants after filtering for MAF:', rare_variants_full.shape)

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

    if gnomad_popmax_af is not None:
        echo('Filtering variants with maximum gnomad popmax AF:', gnomad_popmax_af)
        gnomad_AF_tag = 'AF/gnomAD/popmax'

        if gnomad_AF_tag not in list(rare_variants_full):
            gnomad_AF_tag = 'AF/gnomAD'
            echo(gnomad_AF_tag, 'not found in list of fields, using instead:', gnomad_AF_tag)

        echo('Replacing NaN values in', gnomad_AF_tag, 'with 0:', np.sum(rare_variants_full[gnomad_AF_tag].isnull()))

        rare_variants_full[gnomad_AF_tag] = np.where(rare_variants_full[gnomad_AF_tag].isnull(), 0, rare_variants_full[gnomad_AF_tag])

        rare_variants_full = rare_variants_full[rare_variants_full[gnomad_AF_tag] <= gnomad_popmax_af]
        echo('After filtering by gnomad popmax AF:', len(rare_variants_full), 'variants')

    if genes_to_test is not None:
        echo('Filtering variants in genes:', genes_to_test)
        rare_variants_full = rare_variants_full[rare_variants_full[GENE_NAME].isin(genes_to_test)]
        echo(len(rare_variants_full), 'variants')

    if len(rare_variants_full) == 0:
        echo('Skipping:', gene_info_fname)
        return None

    echo('Filtering phenotype data for exome samples')

    echo('variants:', len(rare_variants_full))
    all_samples = sorted(set([s for ss in rare_variants_full[ALL_SAMPLES] for s in ss.split(',')]))

    echo('n_samples:', len(all_samples))

    # ph_data = original_ph_data # [original_ph_data[SAMPLE_ID].isin(set(all_samples))].copy()

    # echo('Exome samples with phenotype:', len(ph_data))

    # echo('Phenotype', phenotype_name, 'is binary:', is_binary)
    # if is_binary:
    #     if is_age_at_diagnosis:
    #         all_vals = ph_data[phenotype_name + '/original']
    #     else:
    #         all_vals = ph_data[phenotype_name]
    #
    #     echo('Cases:', np.sum(all_vals), ', Controls:', len(all_vals) - np.sum(all_vals))
    #     # echo('Treating as continuous phenotype!!')
    #     # is_binary = False

    echo('Rare variants:', len(rare_variants_full))

    rare_variants_in_phenotype_samples = filter_variants_for_rv_test(rare_variants_full,
                                                                     all_samples=samples_with_phenotypes,
                                                                     max_AC=max_AC,
                                                                     remove_missing_pAI=False,
                                                                     flip_alt_majors=True,
                                                                     pathogenicity_score_label=pathogenicity_score_label)

    echo('Rare variants in exomes with phenotypes:', len(rare_variants_in_phenotype_samples))

    rare_variants_in_phenotype_samples = rare_variants_in_phenotype_samples[
        ((rare_variants_in_phenotype_samples[VCF_AF] <= max_maf) &
         (rare_variants_in_phenotype_samples[VCF_AF] > 0)) |
        ((rare_variants_in_phenotype_samples[VCF_AF] >= 1 - max_maf) &
         (rare_variants_in_phenotype_samples[VCF_AF] < 1))]

    echo('Rare variants in exomes with phenotypes with MAF <=', max_maf, ':', len(rare_variants_in_phenotype_samples))
    if len(rare_variants_in_phenotype_samples) == 0:
        echo('Skipping')
        return None

    rare_variants_in_phenotype_samples = rare_variants_in_phenotype_samples[[VCF_CHROM,
                                                                             VARID_REF_ALT,
                                                                             GENE_NAME,
                                                                             VCF_AC,
                                                                             VCF_CONSEQUENCE,
                                                                             SPLICEAI_MAX_SCORE,
                                                                             pathogenicity_score_label,
                                                                             ALL_SAMPLES]].copy()

    echo('Returning variants and ph_data')
    return rare_variants_in_phenotype_samples


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-i',
                        dest='input_fname',
                        help='pickle file the phenotype values for each subject [e.g ' +
                              UKB_DATA_PATH +
                              '/molecular_phenotypes.17_SEPT_2019/phenotypes_for_GWAS/LDL_direct.europeans/LDL_direct.europeans.both.med_corrected.phenotype_values.pickle]',
                        required=True)

    parser.add_argument('-p',
                        nargs='+',
                        dest='phenotype_name',
                        help='phenotype name',
                        required=True)

    parser.add_argument('-f', '--filters', nargs='+', dest='filters', help='filter exomes based on binary covariates')
    parser.add_argument('-e', '--to-exclude', nargs='+', dest='to_exclude', help='exclude exomes based on binary covariates')

    parser.add_argument('--exclude-variants', dest='exclude_variants', help='variants to exclude from the burden tests')

    parser.add_argument('--exit-if-exists',
                        action='store_true',
                        dest='exit_if_exists',
                        help='Exit if results files exist')

    parser.add_argument('--n-randomizations',
                        dest='n_randomizations',
                        type=int,
                        default=0,
                        help='n randomizations'
                        )

    parser.add_argument('--adaptive-fdr',
                        dest='adaptive_fdr',
                        action='store_true',
                        help='Compute FDR based on adaptive number of randomizations'
                        )

    parser.add_argument('--apply-genomic-correction',
                        dest='apply_genomic_correction',
                        action='store_true',
                        help='Apply genomic correction on top of chi2 correction for binary phenotypes'
                        )

    parser.add_argument('--approximate-null-distribution',
                        dest='approximate_null_distribution',
                        action='store_true',
                        help='Approximate null distribution when computing adaptive FDR'
                        )

    parser.add_argument('--n-samples-for-approximation',
                        dest='n_samples_for_approximation',
                        type=int,
                        default=10000,
                        help='number of samples for approximation of the null distribution'
                        )

    parser.add_argument('--genes', nargs='+', dest='genes_to_test', help='genes to test')

    parser.add_argument('--gene-list', dest='gene_list', help='file with gene list to test')

    parser.add_argument('--variant-types', nargs='+', dest='variant_types', help='variant types to test')

    parser.add_argument('--gencode', dest='gencode_fname', help='gencode file with genes to test')

    parser.add_argument('--filter-samples', dest='filter_samples',
                        help=f'filter_samples: e.g. {ROOT_PATH}/ukbiobank/data/exomes/26_OCT_2020/qc/ukb200k_unrelated_white_europeans.pickle'
                        )

    parser.add_argument('--is-age-at-diagnosis',
                        dest='is_age_at_diagnosis',
                        action='store_true',
                        help='The input phenotype is age at diagnosis'
                        )

    parser.add_argument('--is-binary',
                        dest='is_binary',
                        action='store_true',
                        help='Treat the phenotype as binary and run logistic regression'
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

    parser.add_argument('--gwas-pvalue-threshold',
                        type=float,
                        dest='GWAS_PVALUE_THRESHOLD',
                        default=5e-8,
                        help='GWAS_PVALUE_THRESHOLD [%(default)s]')

    parser.add_argument('--covariates',
                        dest='cov_fname',
                        nargs='+',
                        help='dataframe with covariates to join'
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

    # parser.add_argument('--is-binary', dest='is_binary', action='store_true', help='is_binary')
    parser.add_argument('--skip-pvalues-for-covariates', dest='skip_pvalues_for_covariates', action='store_true', help='Do not output p-values and effect sizes for the covariates during covariate correction. '
                                                                                                                       'This uses a faster regression method with less memory and does not affect the final results.')
    parser.add_argument('--test-meds', dest='test_meds', action='store_true', help='test medications')
    parser.add_argument('--test-PRS', dest='test_PRS', help='PRS name to test')
    parser.add_argument('--no_pAI_threshold_optimization', dest='no_pAI_threshold_optimization', action='store_true', help='no_pAI_threshold_optimization')
    parser.add_argument('--no_AC_threshold_optimization', dest='no_AC_threshold_optimization', action='store_true', help='no_AC_threshold_optimization')

    parser.add_argument('--pathogenicity-score-label', dest='pathogenicity_score_label', type=str, default=PRIMATEAI_SCORE, help='pathogenicity_score_label: [%(default)s]')
    parser.add_argument('--pathogenicity-score-file', dest='pathogenicity_score_file', help='File with additional scores for variants to be merged with the input variants file by chrom, pos, ref alt.'
                                                                                            'The label of the score column should be given with --pathogenicity-score-label')

    parser.add_argument('--MAX_pAI_THRESHOLDS_TO_TEST', dest='MAX_pAI_THRESHOLDS_TO_TEST', type=int, default=5, help='MAX_pAI_THRESHOLDS_TO_TEST [%(default)s]')
    parser.add_argument('--pAI-thresholds-to-test', dest='pAI_thresholds_to_test', type=float, nargs='+', help='array of primateAI (or other pathogenicity score) thresholds to test')

    parser.add_argument('--gnomad-coverage', type=str, nargs='+', dest='gnomad_coverage', default=None, help='gnomad coverage tuple "min_fraction coverage", ex. "0.8 20"  [%(default)s]')
    parser.add_argument('--gnomad-popmax-af', type=float, dest='gnomad_popmax_af', default=None, help='maximum gnomad popmax allele frequency [%(default)s]')
    parser.add_argument('--white-listed-exons', type=str, dest='white_listed_exons_fname', help='use only variants from these white listed exons. format: CHROM <tab> start <tab> end <tab> etc')

    parser.add_argument('--n-threads', type=int, dest='n_threads', default=1, help='number of processes to use [%(default)s]')
    parser.add_argument('--maf', type=float, dest='maf', default=0.001, help='maximum MAF [%(default)s]')
    parser.add_argument('--max-AC', type=int, dest='max_AC', help='maximum AC')

    parser.add_argument('--min-variants-per-gene', type=int, dest='min_variants_per_gene', help='minimum number of variants per gene [%(default)s]', default=1)

    parser.add_argument('--pvalue-threshold-for-permutation-tests', type=float, dest='pvalue_threshold_for_permutation_tests',
                        help='The maximum FDR-corrected p-value for a gene for which permutation tests will be run to correct for testing multiple allele count and pathogenicity score thresholds. '
                             'P-values less than this will be corrected with standard Benjamini-Hochberg FDR correction [%(default)s]', default=1e-5)

    parser.add_argument('--max-missingness', type=float, dest='max_missingness', default=0.05, help='max_missingness [%(default)s]')
    parser.add_argument('--min-pvalue-to-store-metadata', type=float, dest='min_pvalue_to_store_metadata', default=0.1, help='min_pvalue_to_store_metadata [%(default)s]')

    parser.add_argument('-g',
                        dest='gene_info_fnames',
                        nargs='+',
                        help=f'genotype info file name [e.g.: {ROOT_PATH}/data/exomes/sparse.liftover_hg19/all_gene_var_info.UKB_MAF_0_01_GNOMAD_POPMAX_MAF_0_01.full_annotation_for_RV_analysis.pickle]',
                        required=True)

    parser.add_argument('-o', dest='out_prefix', help='output file name', required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    args = parser.parse_args()

    log_name = args.out_prefix
    to_exclude = args.to_exclude
    genes_to_test = args.genes_to_test
    gene_list = args.gene_list

    n_randomizations = args.n_randomizations
    is_binary = args.is_binary
    cov_fname = args.cov_fname
    min_pvalue_to_store_metadata = args.min_pvalue_to_store_metadata
    GWAS_PVALUE_THRESHOLD = args.GWAS_PVALUE_THRESHOLD
    exclude_variants = args.exclude_variants
    exit_if_exists = args.exit_if_exists

    out_prefix = args.out_prefix
    final_result_out_fname = out_prefix + '.main_analysis'

    if exit_if_exists:
        if os.path.exists(final_result_out_fname + '.pickle'):
            echo('Results files exist:', final_result_out_fname + '.pickle')
            echo('Exiting..')
            exit(0)

    if args.test_PRS is not None:
        log_name += '.PRS_interactions_' + args.test_PRS
    if args.test_meds:
        log_name += '.MED_interactions'

    open_log(log_name + '.log', 'wt')

    echo('CMD:', ' '.join(sys.argv))

    echo('Parameters:\n' + pprint.pformat(args.__dict__))

    phenotype_names = args.phenotype_name

    test_meds = args.test_meds
    test_PRS = args.test_PRS
    n_threads = args.n_threads
    filters = args.filters
    find_best_pAI_threshold = not args.no_pAI_threshold_optimization
    find_best_AC_threshold = not args.no_AC_threshold_optimization
    gnomad_coverage = args.gnomad_coverage
    is_age_at_diagnosis = args.is_age_at_diagnosis
    regress_out_age_at_diagnosis = args.regress_out_age_at_diagnosis
    regress_out_ukb_common_variants = args.regress_out_ukb_common_variants
    regress_out_all_ukb_common_variants = args.regress_out_all_ukb_common_variants
    pathogenicity_score_label = args.pathogenicity_score_label
    pathogenicity_score_file = args.pathogenicity_score_file
    gnomad_popmax_af = args.gnomad_popmax_af
    skip_pvalues_for_covariates = args.skip_pvalues_for_covariates
    max_AC = args.max_AC
    white_listed_exons_fname = args.white_listed_exons_fname
    adaptive_fdr = args.adaptive_fdr
    approximate_null_distribution = args.approximate_null_distribution
    n_samples_for_approximation = args.n_samples_for_approximation
    apply_genomic_correction = args.apply_genomic_correction
    min_variants_per_gene = args.min_variants_per_gene
    pvalue_threshold_for_permutation_tests = args.pvalue_threshold_for_permutation_tests
    variant_types = args.variant_types

    if gene_list is not None:
        echo('Reading gene list for burden tests from:', gene_list)
        with open(gene_list, 'rt') as in_f:
            genes_to_test = [l.strip() for l in in_f]
        echo('Genes to test:', len(genes_to_test))

    if type(variant_types) is list and len(variant_types) == 0:
        variant_types = None

    if adaptive_fdr and n_randomizations == 0:
        echo('[WARNING] adaptive_fdr is True, but n_randomizations = 0. Setting n_randomizations = 100!')
        n_randomizations = 100

    filter_controls = args.filter_controls
    regress_out = args.regress_out

    MAX_pAI_THRESHOLDS_TO_TEST = args.MAX_pAI_THRESHOLDS_TO_TEST
    pAI_thresholds_to_test = args.pAI_thresholds_to_test

    max_maf = args.maf

    if n_threads > 1:
        echo('Setting the number of threads for mkl to 1 to prevent multiprocessing deadlocks')
        import mkl
        mkl.set_num_threads(1)

    if is_age_at_diagnosis:
        phenotype_names = [p + '/age_at_diagnosis' for p in phenotype_names]

    echo('Testing phenotype=', phenotype_names, ', out_prefix=', out_prefix)

    final_result = None

    max_missingness = args.max_missingness

    filter_samples = None
    if args.filter_samples is not None:
        filter_samples_fname = args.filter_samples
        echo('Reading sample ids to use:', filter_samples_fname)

        if filter_samples_fname.endswith('.pickle'):
            filter_samples = set(pd.read_pickle(filter_samples_fname)[SAMPLE_ID])
        elif filter_samples_fname.endswith('.csv') or filter_samples_fname.endswith('.csv.gz'):
            filter_samples = set(map(str, pd.read_csv(filter_samples_fname)[SAMPLE_ID]))
        else:
            with open_file(filter_samples_fname) as in_f:
                filter_samples = set([l.strip() for l in in_f])

        echo('filter_samples:', len(filter_samples))

    # reading in corrected biomarkers
    input_fname = args.input_fname
    echo('Reading phenotypes from:', input_fname)
    if input_fname.endswith('.pickle'):
        original_ph_data = pd.read_pickle(input_fname)
    else:
        original_ph_data = pd.read_csv(input_fname, sep='\t', dtype={SAMPLE_ID: str})

    echo('Excluding', len(UKB_SAMPLES_TO_EXCLUDE), 'sample ids that withdrew consent')
    original_ph_data = original_ph_data[~original_ph_data[SAMPLE_ID].isin(UKB_SAMPLES_TO_EXCLUDE)].copy()

    original_ph_data = original_ph_data.dropna(subset=phenotype_names)

    if cov_fname is not None:
        covariates = None
        for cur_cov_fname in cov_fname:
            echo('Reading covariates from:', cur_cov_fname)
            if cur_cov_fname.endswith('.pickle'):
                _covs = pd.read_pickle(cur_cov_fname)
            else:
                _covs = pd.read_csv(cur_cov_fname, sep='\t')
                _covs[SAMPLE_ID] = _covs[SAMPLE_ID].astype(str)

            if covariates is None:
                covariates = _covs
            else:
                covariates = pd.merge(covariates, _covs, on=SAMPLE_ID)

        original_ph_data = pd.merge(original_ph_data, covariates, on=SAMPLE_ID, suffixes=['', '/covariate'])

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
        is_binary = True
        original_ph_data[phenotype_name + '/original_age_at_diagnosis'] = list(original_ph_data[phenotype_name])
        original_ph_data[phenotype_name] = (original_ph_data[phenotype_name] >= 0).astype(int)
        echo('cases:', np.sum(original_ph_data[phenotype_name]), ', controls:', len(original_ph_data) - np.sum(original_ph_data[phenotype_name]))

    if filters:
        original_ph_data = original_ph_data[(original_ph_data[filters] == 1).all(axis=1)]
        echo('Keeping samples with:', filters, ', samples:', len(original_ph_data))

    if to_exclude:
        original_ph_data = original_ph_data[~(original_ph_data[to_exclude] == 1).all(axis=1)]
        echo('Excluding samples with:', to_exclude, ', samples:', len(original_ph_data))

    # is_binary = (len(set(original_ph_data[phenotype_name].dropna())) == 2)
    cols_to_keep = [SAMPLE_ID] + [c for c in list(original_ph_data) if c.startswith(phenotype_name)]
    if regress_out is not None:
        cols_to_keep += regress_out

    if AGE in list(original_ph_data) and AGE not in cols_to_keep and is_age_at_diagnosis:
        echo('Adding age to list of covariates')
        cols_to_keep += [AGE]

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
                    original_ph_data = original_ph_data[(original_ph_data[phenotype_name] == 1) | (original_ph_data[cov_name] < 0)].copy()
                else:
                    echo('Keeping only controls that were diagnosed with:', cov_name)
                    original_ph_data = original_ph_data[(original_ph_data[phenotype_name] == 1) | (original_ph_data[cov_name] > 0)].copy()
            else:
                original_ph_data = original_ph_data[(original_ph_data[phenotype_name] == 1) | (original_ph_data[cov_name] == filter_value)].copy()

            echo('Filtering controls with covariate:', cov_name, '=', filter_value, ', after:', len(original_ph_data))

    if regress_out_age_at_diagnosis:

        echo('Regressing out age at diagnosis with 0 for controls')
        original_ph_data[phenotype_name + '/augm'] = original_ph_data[phenotype_name + '/original_age_at_diagnosis'].fillna(0)
        if regress_out is None:
            regress_out = []

        regress_out += [phenotype_name + '/augm']

    if regress_out_ukb_common_variants:
        from ukb_analysis import get_dosages_for_variants, UKB_COMMON_VARIANTS_ON_DNANEXUS_MAPPING

        if not os.path.exists(regress_out_ukb_common_variants):
            echo('[WARNING] regress_out_ukb_common_variants file not found, skipping:', regress_out_ukb_common_variants)
        else:
            echo('Reading common variants from:', regress_out_ukb_common_variants)
            ukb_common_variants = pd.read_csv(regress_out_ukb_common_variants, sep='\t', dtype={VCF_CHROM: str, VCF_POS: int})
            echo('ukb_common_variants:', ukb_common_variants.shape)

            ukb_common_variants = ukb_common_variants[ukb_common_variants['pvalue'] <= GWAS_PVALUE_THRESHOLD].copy()

            if VCF_AF in list(ukb_common_variants):
                GWAS_MIN_MAF = 0.01
                ukb_common_variants = ukb_common_variants[(ukb_common_variants[VCF_AF] >= GWAS_MIN_MAF) &
                                                          (ukb_common_variants[VCF_AF] <= 1 - GWAS_MIN_MAF)].copy()

            echo('ukb_common_variants after filtering by MAF and p-value:', ukb_common_variants.shape)

            if len(ukb_common_variants) > 0:

                if not regress_out_all_ukb_common_variants:
                    ukb_common_variants = ukb_common_variants[ukb_common_variants['index_variant'] == ukb_common_variants[VARID_REF_ALT]]

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

    original_ph_data[CONST_LABEL] = 1

    if filter_samples is not None:
        echo('Filtering samples with phenotypes:', len(original_ph_data))
        original_ph_data = original_ph_data[original_ph_data[SAMPLE_ID].isin(filter_samples)]
        echo('After filtering:', len(original_ph_data), 'samples')

    if regress_out is not None:
        original_ph_data = original_ph_data.dropna(subset=regress_out)

        # convert all covariates to z-scores to avoid problems with fitting the regression models
        for cov_label in regress_out:
            original_ph_data[cov_label] = scipy.stats.zscore(original_ph_data[cov_label])

        echo('Regressing out covariates after converting them to z-scores:', regress_out, ', data points:', len(original_ph_data))
        import statsmodels.api as sm

        if is_binary:

            lm = sm.Logit(original_ph_data[phenotype_name], original_ph_data[regress_out + [CONST_LABEL]])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                lm_res = lm.fit()

            echo(lm_res.summary())
            original_ph_data[CONST_LABEL] = (lm_res.params * original_ph_data[regress_out + [CONST_LABEL]]).sum(axis=1)

        else:
            if skip_pvalues_for_covariates:
                resid = get_residuals(original_ph_data, phenotype_name, regress_out, verbose=True, make_sample_ids_index=False)
            else:
                lm = sm.OLS(original_ph_data[phenotype_name], original_ph_data[regress_out + [CONST_LABEL]])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    lm_res = lm.fit()

                echo(lm_res.summary())
                resid = pd.DataFrame({SAMPLE_ID: original_ph_data[SAMPLE_ID],
                                      phenotype_name: lm_res.resid})

            original_ph_data = pd.merge(original_ph_data.rename(columns={phenotype_name: phenotype_name + '/original'}),
                                        resid,
                                        on=SAMPLE_ID)

            del resid

            echo('original_ph_data after regressing out covariates:', original_ph_data.shape)

    cols_to_keep = [SAMPLE_ID, phenotype_name, CONST_LABEL]

    phenotype_name_original = None
    if phenotype_name + '/original' in list(original_ph_data):
        phenotype_name_original = phenotype_name + '/original'
        cols_to_keep += [phenotype_name_original]

    if phenotype_name + '/original_age_at_diagnosis' in list(original_ph_data):
        phenotype_name_original = phenotype_name + '/original_age_at_diagnosis'
        cols_to_keep += [phenotype_name_original]

    if is_age_at_diagnosis and AGE in list(original_ph_data):
        cols_to_keep += [AGE]

    echo('cols_to_keep:', cols_to_keep)
    ph_data = original_ph_data[cols_to_keep].dropna().copy(deep=True)

    if n_randomizations > 0:
        random_seed = random.randint(1, 10000000)
    else:
        random_seed = None

    samples_with_phenotypes = set(ph_data[SAMPLE_ID])

    echo('Storing phenotype values:', out_prefix + '.ph_data.pickle')
    ph_data.to_pickle(out_prefix + '.ph_data.pickle')
    ph_data.to_csv(out_prefix + '.ph_data.csv.gz', sep='\t', index=False)

    echo('ph_data:', ph_data.shape)
    if is_binary:
        n_cases = np.sum(ph_data[phenotype_name])
        echo('cases:', n_cases, ', controls:', len(ph_data) - n_cases)

    header_printed = False

    with open_file(final_result_out_fname + '.csv.gz', 'w') as out_f_all:

        for gene_info_fname in sorted(args.gene_info_fnames):

            gc.collect()

            gene_info_fname_label = os.path.split(gene_info_fname)[1]

            rare_variants_in_phenotype_samples = read_rare_variants(gene_info_fname,
                                                                    pathogenicity_score_file,
                                                                    pathogenicity_score_label,
                                                                    max_maf,
                                                                    max_AC,
                                                                    max_missingness,
                                                                    gnomad_coverage,
                                                                    gnomad_popmax_af,
                                                                    genes_to_test,
                                                                    samples_with_phenotypes,
                                                                    white_listed_exons_fname=white_listed_exons_fname,
                                                                    gencode_fname=args.gencode_fname,
                                                                    exclude_variants=exclude_variants)

            if rare_variants_in_phenotype_samples is None:
                continue

            echo('Creating gene to chromosome dataframe')
            gene_chrom_df = rare_variants_in_phenotype_samples[[VCF_CHROM, GENE_NAME]].drop_duplicates().copy()

            if not test_meds and test_PRS is None:

                # echo('Ranking data')
                # ph_data_for_testing[phenotype_name] = scipy.stats.rankdata(ph_data_for_testing[phenotype_name])
                output_dir = os.path.split(out_prefix)[0]
                if output_dir == '':
                    output_dir = os.getcwd()

                res = test_ukb_quantitive_phenotype_for_rare_variants_associations(rare_variants_in_phenotype_samples,
                                                                                   ph_data,
                                                                                   phenotype_name=phenotype_name,
                                                                                   phenotype_name_original=phenotype_name_original,
                                                                                   find_best_pAI_threshold=find_best_pAI_threshold,
                                                                                   find_best_AC_threshold=find_best_AC_threshold,
                                                                                   variant_types=variant_types,
                                                                                   test_PRS=None,
                                                                                   test_meds=False,
                                                                                   is_binary=is_binary,
                                                                                   n_threads=n_threads,
                                                                                   output_dir=output_dir,
                                                                                   MAX_pAI_THRESHOLDS_TO_TEST=MAX_pAI_THRESHOLDS_TO_TEST,
                                                                                   pAI_thresholds_to_test=pAI_thresholds_to_test,
                                                                                   pathogenicity_score_label=pathogenicity_score_label,
                                                                                   n_randomizations=n_randomizations,
                                                                                   random_seed=random_seed,
                                                                                   adaptive_fdr=adaptive_fdr,
                                                                                   is_age_at_diagnosis=is_age_at_diagnosis,
                                                                                   approximate_null_distribution=approximate_null_distribution,
                                                                                   n_samples_for_approximation=n_samples_for_approximation,
                                                                                   min_variants_per_gene=min_variants_per_gene,
                                                                                   pvalue_threshold_for_permutation_tests=pvalue_threshold_for_permutation_tests
                                                                                   )

                if res is None:
                    continue

                res = pd.merge(gene_chrom_df, res, on=GENE_NAME).sort_values('ALL/del/carrier/pvalue/fdr_corr')
                cols_at_the_front = [GENE_NAME, VCF_CHROM]
                for c in [SHUFFLED_IDX, IS_REAL_DATA]:
                    if c in list(res):
                        cols_at_the_front += [c]

                res = res[cols_at_the_front + [c for c in list(res) if c not in cols_at_the_front]]

                for col in list(res):
                    if col.endswith('/all_carriers') or col.endswith('/carriers'):
                        col_vt = col.split('/')[1]
                        pvalue_col = f'ALL/{col_vt}/carrier/pvalue/fdr_corr'
                        res[col] = np.where(res[pvalue_col] <= min_pvalue_to_store_metadata, res[col], '')

                echo('res:', res.shape, len(set(res[GENE_NAME])), ', columns:', list(res))

                echo('Storing results to csv:', final_result_out_fname + '.csv.gz')

                out_f_all.write(res.to_csv(header=(not header_printed), index=False, sep='\t'))
                header_printed = True

            if test_PRS is not None:

                echo('Testing rare variants interactions with PRS for:', test_PRS)

                ph_data_for_testing = ph_data[[SAMPLE_ID, phenotype_name, test_PRS]].copy().dropna()

                if len(ph_data_for_testing) == 0:
                    echo('No PRS scores found!')
                    close_log()
                    exit()

                res = test_ukb_quantitive_phenotype_for_rare_variants_associations(rare_variants_in_phenotype_samples,
                                                                                   ph_data_for_testing,
                                                                                   phenotype_name=phenotype_name,
                                                                                   find_best_pAI_threshold=find_best_pAI_threshold,
                                                                                   find_best_AC_threshold=find_best_AC_threshold,
                                                                                   test_PRS=test_PRS,
                                                                                   test_meds=False,
                                                                                   n_threads=n_threads,
                                                                                   MAX_pAI_THRESHOLDS_TO_TEST=MAX_pAI_THRESHOLDS_TO_TEST
                                                                                   )

                out_fname = out_prefix + '.' + gene_info_fname_label + f'.PRS_interactions_{test_PRS}.csv.gz'
                final_result_out_fname = out_prefix + f'.PRS_interactions_{test_PRS}'

                echo('Storing results in:', out_fname)
                res.to_csv(out_fname, sep='\t', index=False)
                res.to_pickle(out_fname.replace('.csv.gz', '.pickle'))

            if test_meds:

                # med_labels_to_test, med_names_to_test = get_med_columns(ph_data)

                assoc_type = 'IRNT' if phenotype_name.endswith('.IRNT') else 'RAW'

                med_labels_to_test = [m for m in list(ph_data) if m.startswith('on_med.') and '.1st_visit.' in m and ph_data.iloc[0][
                    'is_associated_med.' + m.split('.')[1] + '.' + assoc_type]]

                med_names_to_test = [m.split('.')[1] for m in med_labels_to_test]

                echo('Testing for interactions with medications:', med_names_to_test)

                ph_data_for_testing = ph_data[[SAMPLE_ID, phenotype_name] + med_labels_to_test].copy()

                med_res = test_ukb_quantitive_phenotype_for_rare_variants_associations(rare_variants_in_phenotype_samples,
                                                                                       ph_data_for_testing,
                                                                                       phenotype_name=phenotype_name,
                                                                                       find_best_pAI_threshold=find_best_pAI_threshold,
                                                                                       find_best_AC_threshold=find_best_AC_threshold,
                                                                                       test_PRS=None,
                                                                                       test_meds=True,
                                                                                       n_threads=n_threads
                                                                                       )

                if final_result is None:
                    final_result = {}

                final_result_out_fname = out_prefix + f'.MED_interactions_'
                for med_name in med_res:
                    clean_med_name = remove_special_chars(med_name)[:100]

                    out_fname = out_prefix + '.' + gene_info_fname_label+ f'.MED_interactions_{clean_med_name}.csv.gz'

                    final_result_out_fname = out_prefix + f'.MED_interactions_{clean_med_name}'

                    if final_result_out_fname not in final_result:
                        final_result[final_result_out_fname] = med_res[med_name]
                    else:
                        final_result[final_result_out_fname] = pd.concat([med_res[med_name],
                                                                          final_result[final_result_out_fname]],
                                                                         ignore_index=True)

                    echo('Storing results in:', out_fname)
                    med_res[med_name].to_csv(out_fname, sep='\t', index=False)
                    med_res[med_name].to_pickle(out_fname.replace('.csv.gz', '.pickle'))

    if type(final_result) is dict:
        for fname in final_result:
            echo('Storing final results in:', fname)
            final_result[fname].to_pickle(fname + '.pickle', protocol=4)
            final_result[fname].to_csv(fname + '.csv.gz', sep='\t', index=False)
    else:
        echo('Reading all results:', final_result_out_fname + '.csv.gz')
        final_result = pd.read_csv(final_result_out_fname + '.csv.gz', sep='\t', dtype={VCF_CHROM: str})

        sort_by = []
        if SHUFFLED_IDX in list(final_result):
            sort_by += [SHUFFLED_IDX]

        sort_by += ['ALL/del/carrier/pvalue/fdr_corr']

        var_types_tested = sorted(set([k.split('/')[1] for k in list(final_result) if k.endswith('/carrier/pvalue/fdr_corr')]))
        for vtype in var_types_tested:
            cols_to_keep_for_fdr = [GENE_NAME, f'ALL/{vtype}/carrier/pvalue/fdr_corr']

            if apply_genomic_correction:
                cols_to_keep_for_fdr += [f'ALL/{vtype}/carrier/pvalue', f'ALL/{vtype}/carrier/pvalue/chi2_correction']

            _r = final_result[cols_to_keep_for_fdr].dropna()

            if apply_genomic_correction:
                echo('Applying genomic correction for:', vtype)

                _r = _r.sort_values(f'ALL/{vtype}/carrier/pvalue/fdr_corr')
                m_idx = int(len(_r) / 2)
                median_el = _r.iloc[m_idx]

                # raw_factor = 0.5 / median_el[f'ALL/{vtype}/carrier/pvalue/fdr_corr']
                m_star = -np.log10(2) / np.log10(1 - median_el[f'ALL/{vtype}/carrier/pvalue'])
                m_factor = m_star / median_el[f'ALL/{vtype}/carrier/pvalue/chi2_correction']

                echo('Genomic correction factor:', m_factor)

                gc_corr = 1 - np.power((1 - _r[f'ALL/{vtype}/carrier/pvalue']),
                                       _r[f'ALL/{vtype}/carrier/pvalue/chi2_correction'] * m_factor)

                # gc_corr_raw = _r[f'ALL/{vtype}/carrier/pvalue/fdr_corr'] * raw_factor
                # correct only genes, for which chi2_correction was calculated to be non-zero
                _r[f'ALL/{vtype}/carrier/pvalue/fdr_corr'] = np.where(_r[f'ALL/{vtype}/carrier/pvalue/chi2_correction'] > 0,
                                                                      gc_corr,
                                                                      _r[f'ALL/{vtype}/carrier/pvalue/fdr_corr'])

                # _r[f'ALL/{vtype}/carrier/pvalue/genomic_factor/is_raw'] = (~_r[f'ALL/{vtype}/carrier/pvalue/chi2_correction'] > 0).astype(int)

                _r[f'ALL/{vtype}/carrier/pvalue/genomic_factor'] = np.where(_r[f'ALL/{vtype}/carrier/pvalue/chi2_correction'] > 0,
                                                                            m_factor,
                                                                            np.nan)


            echo('Adding FDR info for:', vtype)
            _r[f'ALL/{vtype}/carrier/pvalue/global_fdr'] = statsmodels.stats.multitest.multipletests(_r[f'ALL/{vtype}/carrier/pvalue/fdr_corr'], method='fdr_bh')[1]

            # remove columns to prevent merging conflicts when genomic correction is applied
            for c in [f'ALL/{vtype}/carrier/pvalue', f'ALL/{vtype}/carrier/pvalue/chi2_correction']:
                if c in _r:
                    del _r[c]

            # it is important to remove this column from final_result so that the /fdr_corr values get updated
            # when genomic correction is applied
            del final_result[f'ALL/{vtype}/carrier/pvalue/fdr_corr']

            final_result = pd.merge(_r, final_result, on=GENE_NAME, how='right')

        final_result = final_result.sort_values(sort_by)

        echo('Storing final results in:', final_result_out_fname)
        final_result.to_pickle(final_result_out_fname + '.pickle')
        final_result.to_csv(final_result_out_fname + '.csv.gz', index=False, sep='\t')


    echo('Done')
    close_log()


if __name__ == '__main__':
    main()
